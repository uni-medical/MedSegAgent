"""Bounded A2A conversations in the service's existing SQLite transaction domain.

Only public user/assistant messages enter model history. Execution manifests,
paths and tool internals remain private. Resuming a waiting task re-enters the
original Agent with its image and conversation; it is not process checkpointing.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid

MAX_MESSAGES = 40
MAX_CHARACTERS = 16000


def bounded(messages):
    kept, size = [], 0
    for message in reversed(messages[-MAX_MESSAGES:]):
        length = sum(len(part.get("text", "")) for part in message.get("parts", []))
        if size + length > MAX_CHARACTERS:
            break
        kept.append(message)
        size += length
    return list(reversed(kept))


def provider_history(messages):
    return [
        {
            "role": "user" if row["role"] == "ROLE_USER" else "assistant",
            "content": "\n".join(part["text"] for part in row["parts"] if "text" in part),
        }
        for row in bounded(messages)
    ]


class Conversations:
    def __init__(self, service):
        self.service = service
        self.db = service.db
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS a2a_contexts
              (id TEXT PRIMARY KEY, principal TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS a2a_requests
              (principal TEXT NOT NULL, message_id TEXT NOT NULL,
               fingerprint TEXT NOT NULL, task_id TEXT NOT NULL,
               PRIMARY KEY(principal,message_id));
        """)

    def _error(self, code, message, status=400):
        from .service import ServiceError

        raise ServiceError(code, message, status)

    def _context(self, principal, context_id):
        row = self.db.execute(
            "SELECT principal,data FROM a2a_contexts WHERE id=?", (context_id,)
        ).fetchone()
        if row:
            if row["principal"] != principal:
                self._error("INVALID_PARAMS", "contextId is not available to this identity.")
            context = json.loads(row["data"])
            if time.time() >= context["expires_at"]:
                self._error("CONTEXT_EXPIRED", "Conversation expired. Start a new context.")
            return context
        # Existing pre-upgrade contexts remain usable without migrating records.
        row = self.db.execute(
            "SELECT principal,data FROM tasks WHERE context_id=? ORDER BY created DESC LIMIT 1",
            (context_id,),
        ).fetchone()
        if not row or row["principal"] != principal:
            self._error("INVALID_PARAMS", "contextId is not available to this identity.")
        old = json.loads(row["data"])
        return {
            "id": context_id,
            "principal": principal,
            "history": [],
            "upload_id": old["upload_id"],
            "modality": old["modality"],
            "last_task_id": old["id"],
            "expires_at": time.time() + self.service.retention_seconds,
        }

    def _save(self, context):
        self.db.execute(
            "INSERT INTO a2a_contexts VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
            (context["id"], context["principal"], json.dumps(context)),
        )

    def submit(
        self, principal, upload_id, text, modality, message_id, context_id=None, task_id=None
    ):
        from . import agent
        from .service import INPUT_FIELDS, TERMINAL

        agent.provider_payload(text, modality)
        for name, value in (
            ("messageId", message_id),
            ("contextId", context_id),
            ("taskId", task_id),
        ):
            if (name == "messageId" or value is not None) and (
                not isinstance(value, str) or not 1 <= len(value) <= 128 or value != value.strip()
            ):
                self._error("INVALID_PARAMS", f"{name} must have 1–128 characters.")
        # Fingerprint the submitted references BEFORE resolving defaults. Transport
        # retries cannot change meaning because a later turn updated the context.
        fingerprint = hashlib.sha256(
            json.dumps([upload_id, text, modality, context_id, task_id]).encode()
        ).hexdigest()
        replay = self.db.execute(
            "SELECT fingerprint,task_id FROM a2a_requests WHERE principal=? AND message_id=?",
            (principal, message_id),
        ).fetchone()
        if replay:
            if replay["fingerprint"] != fingerprint:
                self._error(
                    "IDEMPOTENCY_CONFLICT", "messageId already belongs to different input.", 409
                )
            return self.service.get(principal, replay["task_id"])
        legacy = self.db.execute(
            "SELECT id,fingerprint FROM tasks WHERE principal=? AND message_id=?",
            (principal, message_id),
        ).fetchone()
        if legacy:
            old_fingerprint = hashlib.sha256(
                json.dumps([upload_id, text, modality, context_id]).encode()
            ).hexdigest()
            if task_id is None and legacy["fingerprint"] == old_fingerprint:
                return self.service.get(principal, legacy["id"])
            self._error(
                "IDEMPOTENCY_CONFLICT", "messageId already belongs to different input.", 409
            )

        previous = None
        if task_id:
            self.service.get(principal, task_id)  # Enforce ownership before reading private state.
            previous = self.service._task(task_id)
            if not previous.get("a2a") or previous["status"] != "input_required":
                self._error(
                    "TASK_NOT_RESUMABLE", "Only an input-required A2A Task can continue.", 409
                )
            if context_id and context_id != previous["context_id"]:
                self._error("INVALID_PARAMS", "taskId and contextId do not identify the same task.")
            context_id = previous["context_id"]
        now = time.time()
        context = (
            self._context(principal, context_id)
            if context_id
            else {
                "id": str(uuid.uuid4()),
                "principal": principal,
                "history": [],
                "upload_id": None,
                "modality": None,
                "last_task_id": None,
                "expires_at": now + self.service.retention_seconds,
            }
        )
        if context["last_task_id"]:
            last = self.service._task(context["last_task_id"])
            if last["status"] not in TERMINAL and last["status"] != "input_required":
                self._error(
                    "CONTEXT_BUSY",
                    "Wait for the current context task before sending a new turn.",
                    409,
                )
            if previous is None and last.get("a2a") and last["status"] == "input_required":
                previous = last  # A context-only reply also continues its waiting task.
        if previous and now >= previous.get("input_expires_at", now):
            self._error("TASK_NOT_RESUMABLE", "The task's input-waiting period has expired.", 409)
        if previous and upload_id and previous["upload_id"] not in (None, upload_id):
            self._error(
                "INVALID_PARAMS",
                "A resumed task keeps its image. Start a new context to replace it.",
            )
        if previous and previous["id"] in self.service.canceling:
            self._error("TASK_NOT_RESUMABLE", "Task cancellation is in progress.", 409)

        inherited = context["upload_id"]
        image_changed = upload_id is not None and inherited not in (None, upload_id)
        resolved_upload = upload_id or inherited
        resolved_modality = (
            modality if modality is not None else (None if image_changed else context["modality"])
        )
        upload = self.service.get_upload(principal, resolved_upload) if resolved_upload else {}
        if image_changed:
            context["history"] = []  # Do not attach the previous image's findings to a new image.
        history = provider_history(context["history"])
        pending = self.db.execute(
            "SELECT principal FROM tasks WHERE status NOT IN ('completed','failed','canceled','input_required')"
        ).fetchall()
        if resolved_upload and (len(pending) >= 8 or sum(r[0] == principal for r in pending) >= 4):
            self._error(
                "CAPACITY_EXCEEDED", "Task capacity reached; retry after a task finishes.", 429
            )
        if not previous and not resolved_upload:
            waiting = self.db.execute(
                "SELECT count(*) FROM tasks WHERE principal=? AND status='input_required'",
                (principal,),
            ).fetchone()[0]
            if waiting >= 16:
                self._error(
                    "CAPACITY_EXCEEDED", "Cancel old waiting tasks before starting another.", 429
                )

        task_id = previous["id"] if previous else str(uuid.uuid4())
        task = (
            dict(previous)
            if previous
            else {
                "id": task_id,
                "principal": principal,
                "context_id": context["id"],
                "created_at": now,
                "result": None,
                "error": None,
            }
        )
        if previous and (previous.get("result") or {}).get("outputs"):
            task["a2a_prior_result"] = previous["result"]
        state = "queued" if resolved_upload else "input_required"
        question = "Upload a 3D .nii or .nii.gz image, then send its upload_id with this taskId and contextId."
        task.update(
            a2a=True,
            upload_id=resolved_upload,
            input={key: value for key, value in upload.items() if key in INPUT_FIELDS},
            text=text,
            modality=resolved_modality,
            modality_source="parameter" if resolved_modality is not None else "unknown",
            status=state,
            progress="Waiting for inference slot" if resolved_upload else question,
            error=None if resolved_upload else {"code": "INPUT_REQUIRED", "message": question},
            updated_at=now,
            attempt_started_at=now,
            agent_history=history,
            a2a_message_id=message_id,
        )
        task.pop("finished_at", None)
        task.pop("expires_at", None)
        task.pop("files_expired", None)
        task.pop("input_expires_at", None)
        task.pop("input_requested_at", None)
        if state == "input_required":
            task["input_expires_at"] = now + self.service.retention_seconds
            task["input_requested_at"] = now
        user_message = {
            "messageId": message_id,
            "role": "ROLE_USER",
            "taskId": task_id,
            "contextId": context["id"],
            "parts": [{"text": text, "mediaType": "text/plain"}],
        }
        task["a2a_history"] = bounded([*task.get("a2a_history", []), user_message])
        context.update(
            history=bounded([*context["history"], user_message]),
            upload_id=resolved_upload,
            modality=resolved_modality,
            last_task_id=task_id,
            expires_at=now + self.service.retention_seconds,
        )
        # No await between context admission, request deduplication and persistence.
        # The service owns one event loop and one database writer.
        with self.db:
            if previous:
                self.db.execute(
                    "UPDATE tasks SET status=?,updated=?,data=? WHERE id=?",
                    (state, now, json.dumps(task), task_id),
                )
            else:
                self.db.execute(
                    "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        task_id,
                        principal,
                        message_id,
                        fingerprint,
                        context["id"],
                        state,
                        now,
                        now,
                        json.dumps(task),
                    ),
                )
            self.db.execute(
                "INSERT INTO a2a_requests VALUES(?,?,?,?)",
                (principal, message_id, fingerprint, task_id),
            )
            self.db.execute(
                "INSERT INTO events(task_id,time,status,code) VALUES(?,?,?,?)",
                (task_id, now, state, "INPUT_REQUIRED" if not resolved_upload else None),
            )
            self._save(context)
        if resolved_upload:
            self.service.launch(task_id)
        else:
            self.record_response(task_id, question)
        return self.service.get(principal, task_id)

    def record_response(self, task_id, text):
        task = self.service._task(task_id)
        if not task.get("a2a") or task.get("a2a_last_response_to") == task["a2a_message_id"]:
            return
        message = {
            "messageId": f"reply-{task_id}-{task['a2a_message_id']}",
            "role": "ROLE_AGENT",
            "taskId": task_id,
            "contextId": task["context_id"],
            "parts": [{"text": str(text)[:4000], "mediaType": "text/plain"}],
        }
        row = self.db.execute(
            "SELECT data FROM a2a_contexts WHERE id=?", (task["context_id"],)
        ).fetchone()
        if not row:
            return
        context = json.loads(row[0])
        context["history"] = bounded([*context["history"], message])
        context["expires_at"] = time.time() + self.service.retention_seconds
        with self.db:
            self._save(context)
            self.service.update(
                task_id,
                a2a_history=bounded([*task.get("a2a_history", []), message]),
                a2a_last_response_to=task["a2a_message_id"],
            )

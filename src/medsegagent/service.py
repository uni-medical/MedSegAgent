"""Durable, identity-scoped jobs around the single local inference core.

Run one ASGI worker per data directory. SQLite owns status and idempotency; files are
private and served only through explicit authenticated endpoints. A crash never silently
restarts a job that had already begun inference.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from medsegagent import agent, core

TERMINAL = {"completed", "failed", "canceled"}
MAX_UPLOAD_BYTES = 90 * 1024 * 1024


class ServiceError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code


def uid() -> str:
    return str(uuid.uuid4())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Service:
    def __init__(self, root: Path, public_url: str):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        for name in ("uploads", "tasks"):
            (self.root / name).mkdir(exist_ok=True, mode=0o700)
        self.public_url = public_url.rstrip("/")
        self.max_upload_bytes = MAX_UPLOAD_BYTES
        self.retention_seconds = int(os.environ.get("MEDSEGAGENT_RETENTION_HOURS", "24")) * 3600
        if self.retention_seconds < 3600:
            raise ValueError("Retention must be at least one hour.")
        # Construct before the ASGI lifespan if needed; all operations run synchronously
        # on the service event loop, never in the image-validation worker thread.
        self.db = sqlite3.connect(self.root / "state.sqlite3", timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS uploads
              (id TEXT PRIMARY KEY, principal TEXT NOT NULL, path TEXT NOT NULL,
               size INTEGER NOT NULL, created REAL NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks
              (id TEXT PRIMARY KEY, principal TEXT NOT NULL, message_id TEXT NOT NULL,
               fingerprint TEXT NOT NULL, context_id TEXT NOT NULL, status TEXT NOT NULL,
               created REAL NOT NULL, updated REAL NOT NULL, data TEXT NOT NULL,
               UNIQUE(principal,message_id));
            CREATE TABLE IF NOT EXISTS events
              (seq INTEGER PRIMARY KEY, task_id TEXT, time REAL, status TEXT, code TEXT);
            CREATE TABLE IF NOT EXISTS sessions
              (hash TEXT PRIMARY KEY, principal TEXT NOT NULL, expires REAL NOT NULL);
        """)
        self.active: dict[str, asyncio.Task] = {}
        self.worker_slot = asyncio.Semaphore(1)
        self.uploading: set[str] = set()
        self.lock = None
        self.cleanup_task = None
        self.closed = False

    async def start(self):
        self.lock = (self.root / "service.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Another MedSegAgent service owns this data directory.") from None
        # Holding the service lock establishes that no earlier upload is still being
        # written by another service. Remove crash leftovers that never entered SQLite.
        known_uploads = {row[0] for row in self.db.execute("SELECT id FROM uploads")}
        for directory in (self.root / "uploads").iterdir():
            if (
                directory.is_dir()
                and not directory.is_symlink()
                and directory.name not in known_uploads
            ):
                shutil.rmtree(directory)
        for row in self.db.execute(
            "SELECT data FROM tasks WHERE status NOT IN ('completed','failed','canceled')"
        ).fetchall():
            task = json.loads(row["data"])
            if task["status"] == "queued":
                self.launch(task["id"])
            else:
                self.update(
                    task["id"],
                    status="failed",
                    progress="Interrupted by service restart",
                    error={
                        "code": "SERVER_RESTART",
                        "message": "Inference was interrupted. Submit a new message to retry.",
                    },
                )
        self.cleanup()
        self.cleanup_task = asyncio.create_task(self.cleanup_loop())

    async def close(self):
        self.closed = True
        if self.cleanup_task:
            self.cleanup_task.cancel()
            await asyncio.gather(self.cleanup_task, return_exceptions=True)
        for task in list(self.active.values()):
            task.cancel()
        await asyncio.gather(*list(self.active.values()), return_exceptions=True)
        if self.lock:
            self.lock.close()
        self.db.close()

    def _task(self, task_id):
        row = self.db.execute("SELECT data FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise ServiceError("TASK_NOT_FOUND", "Task not found.", 404)
        return json.loads(row["data"])

    def get(self, principal, task_id):
        task = self._task(task_id)
        if task["principal"] != principal:
            raise ServiceError("TASK_NOT_FOUND", "Task not found.", 404)
        return {k: v for k, v in task.items() if k not in {"principal", "selection"}}

    def list(self, principal):
        rows = self.db.execute(
            "SELECT id FROM tasks WHERE principal=? ORDER BY created DESC LIMIT 100", (principal,)
        ).fetchall()
        return [self.get(principal, row["id"]) for row in rows]

    def update(self, task_id, **changes):
        task = self._task(task_id)
        task.update(changes, updated_at=time.time())
        with self.db:
            self.db.execute(
                "UPDATE tasks SET status=?,updated=?,data=? WHERE id=?",
                (task["status"], task["updated_at"], json.dumps(task), task_id),
            )
            self.db.execute(
                "INSERT INTO events(task_id,time,status,code) VALUES(?,?,?,?)",
                (
                    task_id,
                    task["updated_at"],
                    task["status"],
                    (task.get("error") or {}).get("code"),
                ),
            )
        return task

    def get_upload(self, principal, upload_id):
        row = self.db.execute(
            "SELECT * FROM uploads WHERE id=? AND principal=?", (upload_id, principal)
        ).fetchone()
        if not row or not Path(row["path"]).is_file():
            raise ServiceError("FILE_NOT_FOUND", "Upload not found or expired.", 404)
        return json.loads(row["data"])

    def upload_path(self, principal, upload_id):
        self.get_upload(principal, upload_id)
        return Path(
            self.db.execute("SELECT path FROM uploads WHERE id=?", (upload_id,)).fetchone()[0]
        )

    def reserve_upload(self, principal, filename):
        if principal in self.uploading or len(self.uploading) >= 4:
            raise ServiceError(
                "CAPACITY_EXCEEDED", "An upload is already in progress; retry later.", 429
            )
        if (
            not isinstance(filename, str)
            or len(filename) > 160
            or not re.fullmatch(r"[\w .()-]+\.nii(?:\.gz)?", filename, flags=re.ASCII)
        ):
            raise ServiceError(
                "INVALID_FILE", "Use a simple .nii or .nii.gz filename without paths."
            )
        size, count = self.db.execute(
            "SELECT COALESCE(SUM(size),0),COUNT(*) FROM uploads WHERE principal=?", (principal,)
        ).fetchone()
        if size + MAX_UPLOAD_BYTES > 512 * 1024**2 or count >= 16:
            raise ServiceError(
                "QUOTA_EXHAUSTED",
                "Upload quota reached. Delete old uploads or wait for expiry.",
                429,
            )
        self.uploading.add(principal)
        upload_id = uid()
        directory = self.root / "uploads" / upload_id
        directory.mkdir(mode=0o700)
        return upload_id, directory / ("image.nii.gz" if filename.endswith(".gz") else "image.nii")

    async def finish_upload(self, principal, upload_id, path, filename, size):
        await core.validate_input_async(str(path))
        import nibabel as nib

        image = nib.load(path)
        data = {
            "id": upload_id,
            "name": filename,
            "size": size,
            "shape": list(image.shape),
            "spacing": list(image.header.get_zooms()),
            "sha256": await asyncio.to_thread(digest, path),
            "created_at": time.time(),
            "expires_at": time.time() + self.retention_seconds,
        }
        data["spacing"] = [float(x) for x in data["spacing"]]
        with self.db:
            self.db.execute(
                "INSERT INTO uploads VALUES(?,?,?,?,?,?)",
                (upload_id, principal, str(path), size, time.time(), json.dumps(data)),
            )
        return data

    def delete_upload(self, principal, upload_id):
        self.get_upload(principal, upload_id)
        rows = self.db.execute(
            "SELECT data FROM tasks WHERE principal=? AND status NOT IN ('completed','failed','canceled')",
            (principal,),
        ).fetchall()
        for row in rows:
            task = json.loads(row[0])
            if task["upload_id"] == upload_id and task["status"] not in TERMINAL:
                raise ServiceError(
                    "TASK_ACTIVE", "Cancel the active task before deleting its input.", 409
                )
        shutil.rmtree(self.root / "uploads" / upload_id, ignore_errors=True)
        with self.db:
            self.db.execute("DELETE FROM uploads WHERE id=?", (upload_id,))

    async def submit(self, principal, upload_id, text, modality, message_id, context_id=None):
        agent.provider_payload(text, modality)  # Local validation before accepting work.
        if not isinstance(message_id, str) or not 1 <= len(message_id) <= 128:
            raise ServiceError("INVALID_PARAMS", "messageId must have 1–128 characters.")
        if context_id is not None and (
            not isinstance(context_id, str) or not 1 <= len(context_id) <= 128
        ):
            raise ServiceError("INVALID_PARAMS", "Invalid contextId.")
        fingerprint = hashlib.sha256(
            json.dumps([upload_id, text, modality, context_id]).encode()
        ).hexdigest()
        existing = self.db.execute(
            "SELECT id,fingerprint FROM tasks WHERE principal=? AND message_id=?",
            (principal, message_id),
        ).fetchone()
        if existing:
            if existing["fingerprint"] != fingerprint:
                raise ServiceError(
                    "IDEMPOTENCY_CONFLICT", "messageId already belongs to different input.", 409
                )
            return self.get(principal, existing["id"])
        self.get_upload(principal, upload_id)
        if context_id:
            owner = self.db.execute(
                "SELECT principal FROM tasks WHERE context_id=? LIMIT 1", (context_id,)
            ).fetchone()
            if not owner or owner[0] != principal:
                raise ServiceError("INVALID_PARAMS", "contextId is not available to this identity.")
        pending = self.db.execute(
            "SELECT principal FROM tasks WHERE status NOT IN ('completed','failed','canceled')"
        ).fetchall()
        if len(pending) >= 8 or sum(row[0] == principal for row in pending) >= 2:
            raise ServiceError(
                "CAPACITY_EXCEEDED", "Task capacity reached; retry after a task finishes.", 429
            )
        task_id, now = uid(), time.time()
        task = {
            "id": task_id,
            "context_id": context_id or uid(),
            "principal": principal,
            "upload_id": upload_id,
            "text": text,
            "modality": modality,
            "status": "queued",
            "progress": "Waiting for inference slot",
            "error": None,
            "result": None,
            "created_at": now,
            "updated_at": now,
        }
        with self.db:
            self.db.execute(
                "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    principal,
                    message_id,
                    fingerprint,
                    task["context_id"],
                    "queued",
                    now,
                    now,
                    json.dumps(task),
                ),
            )
        self.launch(task_id)
        return self.get(principal, task_id)

    def launch(self, task_id):
        worker = asyncio.create_task(self.run(task_id))
        self.active[task_id] = worker
        worker.add_done_callback(lambda _: self.active.pop(task_id, None))

    async def run(self, task_id):
        try:
            created = self._task(task_id)["created_at"]
            deadline = created + int(os.environ.get("MEDSEGAGENT_TIMEOUT_SECONDS", "7200"))
            async with asyncio.timeout(max(0, deadline - time.time())), self.worker_slot:
                task = self._task(task_id)
                self.update(task_id, status="routing", progress="Selecting local tool")
                selected = await agent.select_tool(task["text"], task["modality"])
                self.update(
                    task_id,
                    status="running",
                    progress="Running local segmentation",
                    selection=asdict(selected),
                )
                parent = self.root / "tasks" / task_id
                result = await core.segment(
                    task=selected.task,
                    input_path=str(self.upload_path(task["principal"], task["upload_id"])),
                    output_dir=str(parent),
                    targets=selected.targets,
                )
                # Copy only validated artifacts to stable allowlisted names, never serve a directory.
                source = Path(result["segmentation_path"])
                destination = parent / "segmentation.nii.gz"
                await asyncio.to_thread(shutil.copyfile, source, destination)
                public = {
                    k: result[k]
                    for k in (
                        "task",
                        "device",
                        "targets",
                        "labels",
                        "segmentation_shape",
                        "segmentation_voxel_spacing",
                        "runtime_seconds",
                    )
                    if k in result
                }
                public.update(
                    model=agent.MODEL,
                    tool=selected.tool,
                    warning="Research use only. Outputs require review; no clinical validation.",
                )
                files = [
                    {
                        "name": "segmentation.nii.gz",
                        "url": f"{self.public_url}/api/tasks/{task_id}/files/segmentation.nii.gz",
                        "media_type": "application/gzip",
                        "sha256": await asyncio.to_thread(digest, destination),
                        "size_bytes": destination.stat().st_size,
                    }
                ]
                public["files"] = files
                (parent / "result.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
                self.update(
                    task_id,
                    status="completed",
                    progress="Segmentation complete",
                    result=public,
                    files=files,
                    expires_at=time.time() + self.retention_seconds,
                )
        except asyncio.CancelledError:
            self.update(
                task_id,
                status="canceled" if not self.closed else "failed",
                progress="Canceled" if not self.closed else "Interrupted by service shutdown",
                error={
                    "code": "CANCELED" if not self.closed else "SERVER_RESTART",
                    "message": "Task stopped. Submit a new message to retry.",
                },
            )
        except TimeoutError:
            self.update(
                task_id,
                status="failed",
                progress="Task time limit exceeded",
                error={
                    "code": "TASK_TIMEOUT",
                    "message": "The task exceeded its total queue/routing/inference time limit.",
                },
            )
        except agent.RoutingError as exc:
            self.update(
                task_id,
                status="failed",
                progress="Tool selection failed",
                error={"code": "ROUTING_FAILED", "message": str(exc)},
            )
        except Exception:  # noqa: BLE001 - project a generic public error and persist failure.
            # Provider bodies, subprocess output, paths and credentials are never HTTP errors.
            self.update(
                task_id,
                status="failed",
                progress="Local inference failed",
                error={
                    "code": "INFERENCE_FAILED",
                    "message": "Local inference failed. The operator can inspect the private run log.",
                },
            )

    async def cancel(self, principal, task_id):
        task = self.get(principal, task_id)
        if task["status"] in TERMINAL:
            if task["status"] == "canceled":
                return task
            raise ServiceError("TASK_NOT_CANCELABLE", "Task is already terminal.", 409)
        worker = self.active.get(task_id)
        if worker:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        # A queued coroutine may have been canceled before its first instruction.
        if self.get(principal, task_id)["status"] not in TERMINAL:
            self.update(task_id, status="canceled", progress="Canceled")
        return self.get(principal, task_id)

    def file_path(self, principal, task_id, name):
        task = self.get(principal, task_id)
        if name not in {"segmentation.nii.gz", "result.json"} or task["status"] != "completed":
            raise ServiceError("FILE_NOT_FOUND", "Result not found.", 404)
        path = self.root / "tasks" / task_id / name
        if task.get("files_expired") or not path.is_file() or path.is_symlink():
            raise ServiceError("FILE_NOT_FOUND", "Result expired or unavailable.", 404)
        return path

    def cleanup(self):
        cutoff = time.time() - self.retention_seconds
        for row in self.db.execute(
            "SELECT id,data FROM tasks WHERE updated<? AND status IN ('completed','failed','canceled')",
            (cutoff,),
        ).fetchall():
            data = json.loads(row["data"])
            if not data.get("files_expired"):
                shutil.rmtree(self.root / "tasks" / row["id"], ignore_errors=True)
                self.update(row["id"], files_expired=True, files=[], result=None)
        for row in self.db.execute(
            "SELECT id,principal FROM uploads WHERE created<?", (cutoff,)
        ).fetchall():
            try:
                self.delete_upload(row["principal"], row["id"])
            except ServiceError:
                pass
        with self.db:
            self.db.execute("DELETE FROM sessions WHERE expires<?", (time.time(),))

    async def cleanup_loop(self):
        while True:
            await asyncio.sleep(900)
            self.cleanup()

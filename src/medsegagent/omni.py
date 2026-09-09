"""Opt-in, isolated image workspaces and real specialist-worker execution.

The Web adapter decides whether to construct this service. No legacy table is changed;
images, revisions and jobs are owned by the caller and stored under ``base.root/omni``.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import time
import uuid
from dataclasses import replace
from pathlib import Path

from medsegagent import core
from medsegagent.gpu_scheduler import GPUScheduler, SchedulerConfig
from medsegagent.service import ServiceError

OPERATIONS = frozenset({"refine", "agent"})
TERMINAL = frozenset({"completed", "failed", "canceled", "input_required"})
MAX_PROMPTS = 128
MAX_REVISIONS = 64
_SAFE_ERRORS = {
    "INVALID_PARAMS": "The workspace request is invalid.",
    "NOT_FOUND": "Workspace, revision or job not found or expired.",
    "REVISION_CONFLICT": "The workspace changed. Reload it before submitting this revision.",
    "WORKSPACE_BUSY": "This workspace already has an active operation.",
    "CAPACITY_EXCEEDED": "Specialist workers are busy. Retry after an operation finishes.",
    "INPUT_CHANGED": "The source image no longer matches its verified copy.",
    "PROMPT_INVALID": "Marks must lie inside the image; a box must lie on one native image slice.",
    "BACKEND_NOT_CONFIGURED": "The local specialist environment or model is not configured.",
    "DEVICE_UNSUPPORTED": "This specialist does not support the configured device.",
    "OUTPUT_INVALID": "The specialist output did not pass validation.",
    "INFERENCE_FAILED": "The specialist operation failed. Its private log is retained.",
    "TIMEOUT": "The specialist operation exceeded its deadline.",
    "SERVICE_RESTARTED": "The service stopped before this operation completed.",
    "SERVICE_CLOSED": "The specialist service is closing.",
    "AGENT_UNAVAILABLE": "The specialist Agent is not configured.",
    "AGENT_FAILED": "The Agent could not complete all requested operations.",
    "EXPIRED": "The workspace expired before this operation completed.",
}


def _error(code, status=400):
    return ServiceError(code, _SAFE_ERRORS.get(code, _SAFE_ERRORS["INFERENCE_FAILED"]), status)


def _id():
    return uuid.uuid4().hex


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _regular(path):
    return path.is_file() and not path.is_symlink()


def _freeze_image(source, destination, expected_hash, *, _stop_event=None):
    """Copy and validate exactly the uploaded bytes, without trusting a path twice."""
    import nibabel as nib

    limit = core._positive_int("MEDSEGAGENT_MAX_INPUT_BYTES", core.DEFAULT_MAX_INPUT_BYTES)
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest = hashlib.sha256()
    with os.fdopen(fd, "rb") as reader, destination.open("xb") as writer:
        os.chmod(destination, 0o600)
        before = os.fstat(reader.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise _error("INPUT_CHANGED")
        size = 0
        while block := reader.read(1024 * 1024):
            core._check_stop(_stop_event)
            size += len(block)
            if size > limit:
                raise _error("INPUT_CHANGED")
            digest.update(block)
            writer.write(block)
        after = os.fstat(reader.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if size != before.st_size or any(getattr(before, k) != getattr(after, k) for k in fields):
            raise _error("INPUT_CHANGED")
        writer.flush()
        os.fsync(writer.fileno())
    actual = digest.hexdigest()
    if expected_hash and actual != expected_hash:
        raise _error("INPUT_CHANGED")
    geometry = core._inspect_nifti(destination, _stop_event=_stop_event)
    image = nib.load(destination)
    return actual, {
        "shape": geometry["shape"],
        "affine": geometry["affine"],
        "spacing": geometry["voxel_spacing"],
        "unit": image.header.get_xyzt_units()[0],
    }


def world_prompts(prompts, geometry):
    """RAS affine-space values → native XYZ. Never guess an oblique rectangle."""
    import numpy as np

    if not isinstance(prompts, list) or not 1 <= len(prompts) <= MAX_PROMPTS:
        raise _error("PROMPT_INVALID")
    inverse = np.linalg.inv(np.asarray(geometry["affine"], dtype=float))
    shape = np.asarray(geometry["shape"], dtype=int)

    def coordinate(value):
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(type(x) not in {int, float} or not math.isfinite(x) for x in value)
        ):
            raise _error("PROMPT_INVALID")
        voxel = (inverse @ np.array([*value, 1.0]))[:3]
        rounded = np.floor(voxel + 0.5).astype(int)
        if np.any(voxel < -0.5) or np.any(voxel > shape - 0.5):
            raise _error("PROMPT_INVALID")
        if np.any(rounded < 0) or np.any(rounded >= shape):
            raise _error("PROMPT_INVALID")
        return voxel, rounded

    result = []
    for prompt in prompts:
        if not isinstance(prompt, dict) or type(prompt.get("positive")) is not bool:
            raise _error("PROMPT_INVALID")
        if prompt.get("kind") == "point" and set(prompt) == {"kind", "world", "positive"}:
            _, point = coordinate(prompt["world"])
            result.append(
                {"kind": "point", "voxel": point.tolist(), "positive": prompt["positive"]}
            )
        elif prompt.get("kind") == "box" and set(prompt) == {
            "kind",
            "world_start",
            "world_end",
            "positive",
        }:
            a, start = coordinate(prompt["world_start"])
            b, end = coordinate(prompt["world_end"])
            fixed = np.flatnonzero(np.isclose(a, b, rtol=0, atol=1e-4))
            if (
                len(fixed) != 1
                or abs(a[fixed[0]] - round(float(a[fixed[0]]))) > 1e-3
                or np.count_nonzero(start == end) != 1
            ):
                raise _error("PROMPT_INVALID")
            bounds = [[int(min(x, y)), int(max(x, y)) + 1] for x, y in zip(start, end, strict=True)]
            result.append({"kind": "box", "bounds": bounds, "positive": prompt["positive"]})
        else:
            raise _error("PROMPT_INVALID")
    return result


def _verify_file(path, expected_hash, *, _stop_event=None):
    if not _regular(path) or core._file_digest(path, _stop_event) != expected_hash:
        raise _error("INPUT_CHANGED")


def _validate_mask(path, geometry, *, _stop_event=None):
    import nibabel as nib
    import numpy as np

    if not _regular(path):
        raise _error("OUTPUT_INVALID")
    inspected = core._inspect_nifti(path, label_map={1: "region"}, _stop_event=_stop_event)
    image = nib.load(path)
    if (
        inspected["shape"] != geometry["shape"]
        or not np.allclose(inspected["affine"], geometry["affine"], rtol=1e-5, atol=1e-4)
        or not np.allclose(inspected["voxel_spacing"], geometry["spacing"], rtol=1e-5, atol=1e-6)
        or image.header.get_xyzt_units()[0] != geometry["unit"]
        or image.dataobj.slope != 1
        or image.dataobj.inter != 0
    ):
        raise _error("OUTPUT_INVALID")
    count = inspected["nonzero_voxels"]
    factor = {"meter": 1000.0, "mm": 1.0, "micron": 0.001}.get(geometry["unit"])
    voxel_volume = (
        abs(float(np.linalg.det(np.asarray(geometry["affine"])[:3, :3]))) * factor**3
        if factor is not None
        else None
    )
    return {
        "voxel_count": count,
        "volume_mm3": count * voxel_volume if voxel_volume is not None else None,
        "volume_ml": count * voxel_volume / 1000 if voxel_volume is not None else None,
        "volume_measurement": {
            "method": "voxel_count_times_absolute_affine_determinant",
            "source_spatial_unit": geometry["unit"],
            "voxel_volume_mm3": voxel_volume,
            "unit_assumption": None,
            "status": "measured" if voxel_volume is not None else "unknown_spatial_unit",
        },
        "sha256": core._file_digest(path, _stop_event),
        "size_bytes": path.stat().st_size,
    }


class OmniService:
    def __init__(self, base_service):
        self.base = base_service
        self.root = base_service.root / "omni"
        self.root.mkdir(mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)
        (self.root / "workspaces").mkdir(mode=0o700, exist_ok=True)
        self.db = sqlite3.connect(self.root / "state.sqlite3", timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS workspaces
              (id TEXT PRIMARY KEY, principal TEXT NOT NULL, upload_id TEXT NOT NULL,
               expires REAL NOT NULL, latest TEXT, data TEXT NOT NULL,
               UNIQUE(principal,upload_id));
            CREATE TABLE IF NOT EXISTS revisions
              (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id)
               ON DELETE CASCADE, created REAL NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs
              (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id)
               ON DELETE CASCADE, principal TEXT NOT NULL, message_id TEXT NOT NULL,
               fingerprint TEXT NOT NULL, operation TEXT NOT NULL, status TEXT NOT NULL,
               created REAL NOT NULL, updated REAL NOT NULL, data TEXT NOT NULL,
               UNIQUE(principal,message_id));
        """)
        os.chmod(self.root / "state.sqlite3", 0o600)
        with self.db:
            for row in self.db.execute(
                "SELECT id,data FROM jobs WHERE status NOT IN ('completed','failed','canceled','input_required')"
            ).fetchall():
                data = json.loads(row["data"])
                data["error"] = {
                    "code": "SERVICE_RESTARTED",
                    "message": _SAFE_ERRORS["SERVICE_RESTARTED"],
                }
                self.db.execute(
                    "UPDATE jobs SET status='failed',updated=?,data=? WHERE id=?",
                    (time.time(), _json(data), row["id"]),
                )
        self.active = {}
        self._open_lock = asyncio.Lock()
        self._opening = set()
        self._workspace_locks = {}
        self._slots = asyncio.Semaphore(2)
        self.closed = False
        self.agent_runner = None
        self.cleanup()

    def capabilities(self):
        return {
            name: {
                "configured": bool(
                    os.environ.get(f"MEDSEGAGENT_{prefix}_PYTHON")
                    and os.environ.get(f"MEDSEGAGENT_{prefix}_MODEL_PATH")
                ),
                "inference_verified": False,
            }
            for name, prefix in [("nninteractive", "NNINTERACTIVE")]
        }

    def _workspace(self, principal, workspace_id):
        if self.closed:
            raise _error("SERVICE_CLOSED", 503)
        self.cleanup()
        row = self.db.execute(
            "SELECT * FROM workspaces WHERE id=? AND principal=?", (workspace_id, principal)
        ).fetchone()
        if row is None or row["expires"] <= time.time():
            raise _error("NOT_FOUND", 404)
        return {
            **json.loads(row["data"]),
            "id": row["id"],
            "upload_id": row["upload_id"],
            "expires_at": row["expires"],
            "latest_revision": row["latest"],
            "_principal": principal,
        }

    def _directory(self, workspace_id):
        if not isinstance(workspace_id, str) or not re.fullmatch(r"[a-f0-9]{32}", workspace_id):
            raise _error("NOT_FOUND", 404)
        return self.root / "workspaces" / workspace_id

    def _revision(self, workspace_id, revision_id):
        if not isinstance(revision_id, str):
            raise _error("NOT_FOUND", 404)
        row = self.db.execute(
            "SELECT data FROM revisions WHERE id=? AND workspace_id=?", (revision_id, workspace_id)
        ).fetchone()
        if row is None:
            raise _error("NOT_FOUND", 404)
        return json.loads(row["data"])

    def get_workspace(self, principal, workspace_id):
        workspace = self._workspace(principal, workspace_id)
        workspace.pop("_principal")
        workspace.pop("source_file")
        workspace["revisions"] = [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT data FROM revisions WHERE workspace_id=? ORDER BY created,id",
                (workspace_id,),
            )
        ]
        jobs = self.db.execute(
            "SELECT * FROM jobs WHERE workspace_id=? ORDER BY created DESC LIMIT 20",
            (workspace_id,),
        ).fetchall()
        workspace["jobs"] = [self._public_job(row) for row in jobs]
        workspace["active_job"] = next(
            (row["id"] for row in jobs if row["status"] not in TERMINAL), None
        )
        workspace["capabilities"] = self.capabilities()
        return workspace

    def list_workspaces(self, principal):
        if self.closed:
            raise _error("SERVICE_CLOSED", 503)
        self.cleanup()
        rows = self.db.execute(
            "SELECT id,upload_id,expires,latest,data FROM workspaces "
            "WHERE principal=? ORDER BY expires DESC LIMIT 64", (principal,)
        ).fetchall()
        result = []
        for row in rows:
            revision = self._revision(row["id"], row["latest"]) if row["latest"] else None
            result.append({
                "id": row["id"], "upload_id": row["upload_id"],
                "created_at": json.loads(row["data"])["created_at"],
                "expires_at": row["expires"], "latest_revision": row["latest"],
                "latest_name": revision["name"] if revision else "未命名区域",
                "revision_count": self.db.execute(
                    "SELECT COUNT(*) FROM revisions WHERE workspace_id=?", (row["id"],)
                ).fetchone()[0],
            })
        return result

    async def prepare_source(self, principal, workspace_id):
        workspace = self._workspace(principal, workspace_id)
        path = self._directory(workspace_id) / workspace["source_file"]
        await core._validation(_verify_file, path, workspace["source_sha256"])
        return path

    async def open_workspace(self, principal, upload_id):
        if not isinstance(upload_id, str) or not 1 <= len(upload_id) <= 128:
            raise _error("INVALID_PARAMS")
        task = asyncio.current_task()
        self._opening.add(task)
        try:
            return await self._open_workspace(principal, upload_id)
        finally:
            self._opening.discard(task)

    async def _open_workspace(self, principal, upload_id):
        if self.closed:
            raise _error("SERVICE_CLOSED", 503)
        async with self._open_lock:
            self.cleanup()
            existing = self.db.execute(
                "SELECT id FROM workspaces WHERE principal=? AND upload_id=?",
                (principal, upload_id),
            ).fetchone()
            if existing:
                return self.get_workspace(principal, existing[0])
            metadata = self.base.get_upload(principal, upload_id)
            source = self.base.upload_path(principal, upload_id)
            workspace_id = _id()
            directory = self._directory(workspace_id)
            directory.mkdir(mode=0o700)
            filename = "image.nii.gz" if source.name.endswith(".gz") else "image.nii"
            expiry = min(
                metadata.get("expires_at") or math.inf, time.time() + self.base.retention_seconds
            )
            try:
                digest, geometry = await core._validation(
                    _freeze_image, source, directory / filename, metadata.get("sha256")
                )
                if expiry <= time.time():
                    raise _error("EXPIRED", 410)
                data = {
                    "source_file": filename,
                    "source_sha256": digest,
                    "geometry": geometry,
                    "created_at": time.time(),
                }
                with self.db:
                    self.db.execute(
                        "INSERT INTO workspaces VALUES(?,?,?,?,?,?)",
                        (workspace_id, principal, upload_id, expiry, None, _json(data)),
                    )
            except BaseException as exc:
                shutil.rmtree(directory, ignore_errors=True)
                if isinstance(exc, (asyncio.CancelledError, ServiceError)):
                    raise
                raise _error("INPUT_CHANGED") from None
            return self.get_workspace(principal, workspace_id)

    def _prepare_body(self, workspace, operation, body):
        if (
            not isinstance(operation, str)
            or operation not in OPERATIONS
            or not isinstance(body, dict)
        ):
            raise _error("INVALID_PARAMS")
        common = {"message_id", "base_revision", "expected_revision"}
        extra = {
            "refine": {"prompts", "name"},
            "agent": {"instruction", "prompts", "name"},
        }[operation]
        if set(body) - common - extra:
            raise _error("INVALID_PARAMS")
        value = copy.deepcopy(body)
        base = value.get("base_revision")
        if base is not None:
            self._revision(workspace["id"], base)
        expected = value.get("expected_revision", base)
        if operation in {"refine", "agent"} and expected != workspace["latest_revision"]:
            raise _error("REVISION_CONFLICT", 409)
        if "name" in value and (
            not isinstance(value["name"], str) or not 1 <= len(value["name"].strip()) <= 80
        ):
            raise _error("INVALID_PARAMS")
        if "instruction" in value and (
            not isinstance(value["instruction"], str)
            or not 1 <= len(value["instruction"].strip()) <= 4000
        ):
            raise _error("INVALID_PARAMS")
        if operation == "refine" or value.get("prompts"):
            world_prompts(value.get("prompts"), workspace["geometry"])
            history = self._revision(workspace["id"], base)["prompts"] if base else []
            if len(history) + len(value["prompts"]) > MAX_PROMPTS:
                raise _error("PROMPT_INVALID")
        value["base_revision"] = base
        if operation in {"refine", "agent"}:
            value["expected_revision"] = expected
        return value

    def refresh_workspace(self, workspace):
        """Host-only refresh for a running Agent; public clients use get_workspace."""
        return self._workspace(workspace["_principal"], workspace["id"])

    async def inspect_revision(self, workspace, revision_id=None):
        workspace = self.refresh_workspace(workspace)
        revision_id = revision_id or workspace["latest_revision"]
        if revision_id is None:
            return {"revision": None, "geometry": workspace["geometry"]}
        revision = self._revision(workspace["id"], revision_id)
        path = self._directory(workspace["id"]) / "revisions" / revision_id / "mask.nii.gz"
        await core._validation(_verify_file, path, revision["sha256"])
        return {"revision": revision, "geometry": workspace["geometry"]}

    def _finished_job(self, job_id, task):
        self.active.pop(job_id, None)
        # Cancellation before the coroutine starts skips its finally block.
        if task.cancelled():
            code = "SERVICE_RESTARTED" if self.closed else None
            self._update_job(
                job_id,
                "failed" if code else "canceled",
                error={"code": code, "message": _SAFE_ERRORS[code]} if code else None,
            )

    async def submit(self, principal, workspace_id, operation, body):
        workspace = self._workspace(principal, workspace_id)
        if not isinstance(body, dict):
            raise _error("INVALID_PARAMS")
        message_id = body.get("message_id")
        if not isinstance(message_id, str) or not 1 <= len(message_id) <= 128:
            raise _error("INVALID_PARAMS")
        fingerprint = hashlib.sha256(_json([workspace_id, operation, body]).encode()).hexdigest()
        old = self.db.execute(
            "SELECT * FROM jobs WHERE principal=? AND message_id=?", (principal, message_id)
        ).fetchone()
        if old is not None:
            if old["fingerprint"] != fingerprint:
                raise _error("REVISION_CONFLICT", 409)
            return self._public_job(old)
        prepared = self._prepare_body(workspace, operation, body)
        pending = self.db.execute(
            "SELECT principal,workspace_id FROM jobs WHERE status NOT IN ('completed','failed','canceled','input_required')"
        ).fetchall()
        if any(row["workspace_id"] == workspace_id for row in pending):
            raise _error("WORKSPACE_BUSY", 409)
        if len(pending) >= 2 or any(row["principal"] == principal for row in pending):
            raise _error("CAPACITY_EXCEEDED", 429)
        job_id, now = _id(), time.time()
        data = {"request": prepared, "result": None, "error": None}
        with self.db:
            self.db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    workspace_id,
                    principal,
                    message_id,
                    fingerprint,
                    operation,
                    "queued",
                    now,
                    now,
                    _json(data),
                ),
            )
        self.active[job_id] = asyncio.create_task(self._run_job(job_id))
        self.active[job_id].add_done_callback(lambda task: self._finished_job(job_id, task))
        return self.get_job(principal, job_id)

    @staticmethod
    def _public_job(row):
        data = json.loads(row["data"])
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "operation": row["operation"],
            "status": row["status"],
            "created_at": row["created"],
            "updated_at": row["updated"],
            "result": data.get("result"),
            "error": data.get("error"),
        }

    def get_job(self, principal, job_id):
        if self.closed:
            raise _error("SERVICE_CLOSED", 503)
        row = self.db.execute(
            "SELECT * FROM jobs WHERE id=? AND principal=?", (job_id, principal)
        ).fetchone()
        if row is None:
            raise _error("NOT_FOUND", 404)
        self._workspace(principal, row["workspace_id"])
        return self._public_job(row)

    def _update_job(self, job_id, status, **changes):
        row = self.db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row:
            data = json.loads(row[0])
            data.update(changes)
            with self.db:
                self.db.execute(
                    "UPDATE jobs SET status=?,updated=?,data=? WHERE id=?",
                    (status, time.time(), _json(data), job_id),
                )

    async def _run_job(self, job_id):
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        try:
            expiry = self.db.execute(
                "SELECT expires FROM workspaces WHERE id=?", (row["workspace_id"],)
            ).fetchone()[0]
            timeout = min(
                core._positive_int("MEDSEGAGENT_TIMEOUT_SECONDS", 7200),
                7200,
                max(0.001, expiry - time.time()),
            )
            async with asyncio.timeout(timeout), self._slots:
                lock = self._workspace_locks.setdefault(row["workspace_id"], asyncio.Lock())
                async with lock:
                    workspace = self._workspace(row["principal"], row["workspace_id"])
                    body = json.loads(row["data"])["request"]
                    self._update_job(job_id, "running")
                    jobdir = self._directory(workspace["id"]) / "jobs" / job_id
                    jobdir.mkdir(mode=0o700, parents=True)
                    if row["operation"] == "agent":
                        if self.agent_runner is None:
                            raise _error("AGENT_UNAVAILABLE", 503)
                        result = await self.agent_runner(self, workspace, body, jobdir)
                    else:
                        result = await self.run_operation(workspace, row["operation"], body, jobdir)
                    if workspace["expires_at"] <= time.time():
                        raise _error("EXPIRED", 410)
                    status, error = "completed", None
                    if row["operation"] == "agent":
                        if not isinstance(result, dict) or result.get("status") not in {
                            "completed",
                            "failed",
                            "needs_input",
                        }:
                            raise _error("OUTPUT_INVALID")
                        if result["status"] == "failed":
                            status = "failed"
                            error = {
                                "code": "AGENT_FAILED",
                                "message": _SAFE_ERRORS["AGENT_FAILED"],
                            }
                        elif result["status"] == "needs_input":
                            status = "input_required"
                        else:
                            revision = result.get("revision")
                            if isinstance(revision, dict) and isinstance(revision.get("id"), str):
                                verified = await self.inspect_revision(workspace, revision["id"])
                                if body.get("prompts"):
                                    prior = (
                                        self._revision(workspace["id"], body["base_revision"])
                                        if body.get("base_revision")
                                        else None
                                    )
                                    expected_prompts = (prior["prompts"] if prior else []) + body[
                                        "prompts"
                                    ]
                                    if (
                                        verified["revision"]["prompts"] != expected_prompts
                                        or verified["revision"]["created_at"] < row["created"]
                                        or verified["revision"]["parent_revision"]
                                        != body.get("base_revision")
                                    ):
                                        raise _error("OUTPUT_INVALID")
                                result["revision"] = verified["revision"]
                            else:
                                # Geometry-only questions may complete with a real, bounded
                                # host inspection receipt, but never with unapplied marks.
                                name = result.get("inspect_artifact")
                                if (
                                    body.get("prompts")
                                    or not isinstance(name, str)
                                    or not re.fullmatch(r"agent-inspection-[0-9]+\.json", name)
                                ):
                                    raise _error("OUTPUT_INVALID")
                                receipt = jobdir / name
                                if not _regular(receipt) or receipt.stat().st_size > 65536:
                                    raise _error("OUTPUT_INVALID")
                                try:
                                    inspected = json.loads(receipt.read_text())
                                except (ValueError, OSError, UnicodeError):
                                    raise _error("OUTPUT_INVALID") from None
                                if inspected != {
                                    "ok": True,
                                    "shape": workspace["geometry"]["shape"],
                                    "revision": None,
                                }:
                                    raise _error("OUTPUT_INVALID")
                                await core._validation(
                                    _verify_file,
                                    self._directory(workspace["id"]) / workspace["source_file"],
                                    workspace["source_sha256"],
                                )
                    self._update_job(job_id, status, result=result, error=error)
        except asyncio.CancelledError:
            code = "SERVICE_RESTARTED" if self.closed else None
            self._update_job(
                job_id,
                "failed" if code else "canceled",
                error={"code": code, "message": _SAFE_ERRORS[code]} if code else None,
            )
        except TimeoutError:
            self._update_job(
                job_id, "failed", error={"code": "TIMEOUT", "message": _SAFE_ERRORS["TIMEOUT"]}
            )
        except Exception as exc:  # noqa: BLE001 - never expose backend paths or logs.
            code = (
                exc.code
                if isinstance(exc, ServiceError) and exc.code in _SAFE_ERRORS
                else "INFERENCE_FAILED"
            )
            self._update_job(job_id, "failed", error={"code": code, "message": _SAFE_ERRORS[code]})
        finally:
            self.active.pop(job_id, None)

    async def cancel_job(self, principal, job_id):
        job = self.get_job(principal, job_id)
        pending = self.active.get(job_id)
        if pending is not None and not pending.done():
            pending.cancel()
            drained = asyncio.gather(pending, return_exceptions=True)
            while not drained.done():
                try:
                    await asyncio.shield(drained)
                except asyncio.CancelledError:
                    continue
        if (
            self.db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            not in TERMINAL
        ):
            self._update_job(job_id, "canceled")
        return self.get_job(principal, job_id) if job["status"] not in TERMINAL else job

    async def run_operation(self, workspace, operation, body, jobdir):
        """Trusted Agent callback entry; caller already owns the workspace/job admission."""
        if operation == "agent":
            raise _error("INVALID_PARAMS")
        workspace = self._workspace(workspace["_principal"], workspace["id"])
        body = self._prepare_body(workspace, operation, body)
        directory = self._directory(workspace["id"])
        source = directory / workspace["source_file"]
        await core._validation(_verify_file, source, workspace["source_sha256"])
        base = (
            self._revision(workspace["id"], body["base_revision"])
            if body.get("base_revision")
            else None
        )
        mask = directory / "revisions" / base["id"] / "mask.nii.gz" if base else None
        if base:
            await core._validation(_verify_file, mask, base["sha256"])
        run = Path(jobdir) / ("operation-" + _id())
        run.mkdir(mode=0o700)
        if operation == "refine":
            count = self.db.execute(
                "SELECT COUNT(*) FROM revisions WHERE workspace_id=?", (workspace["id"],)
            ).fetchone()[0]
            if count >= MAX_REVISIONS:
                raise _error("CAPACITY_EXCEEDED", 429)
            history = (base["prompts"] if base else []) + body["prompts"]
            output = run / "mask.nii.gz"
            request = {
                "operation": "refine",
                "image_path": str(source),
                "output_path": str(output),
                "prompts": world_prompts(history, workspace["geometry"]),
            }
            response = await self._worker("nninteractive", request, run)
            try:
                measurement = await core._validation(_validate_mask, output, workspace["geometry"])
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - image parser diagnostics remain private.
                raise _error("OUTPUT_INVALID") from None
            current = self._workspace(workspace["_principal"], workspace["id"])
            if current["latest_revision"] != body["expected_revision"]:
                raise _error("REVISION_CONFLICT", 409)
            revision_id = _id()
            published = directory / "revisions" / revision_id
            published.mkdir(mode=0o700, parents=True)
            revision = {
                "id": revision_id,
                "name": body.get("name", base["name"] if base else "Region").strip(),
                "parent_revision": base["id"] if base else None,
                "created_at": time.time(),
                "prompts": history,
                "backend": "nninteractive",
                "backend_version": response.get("version"),
                "source_sha256": workspace["source_sha256"],
                **measurement,
            }
            try:
                os.chmod(output, 0o600)
                os.replace(output, published / "mask.nii.gz")
                with self.db:
                    self.db.execute(
                        "INSERT INTO revisions VALUES(?,?,?,?)",
                        (revision_id, workspace["id"], revision["created_at"], _json(revision)),
                    )
                    updated = self.db.execute(
                        "UPDATE workspaces SET latest=? WHERE id=? AND latest IS ?",
                        (revision_id, workspace["id"], body["expected_revision"]),
                    )
                    if updated.rowcount != 1:
                        raise _error("REVISION_CONFLICT", 409)
            except BaseException:
                shutil.rmtree(published, ignore_errors=True)
                raise
            return {"revision": revision}
        raise _error("INVALID_PARAMS")

    async def _worker(self, backend, request, run):
        if backend != "nninteractive":
            raise _error("INVALID_PARAMS")
        prefix = "NNINTERACTIVE"
        executable = os.environ.get(f"MEDSEGAGENT_{prefix}_PYTHON", "")
        model = os.environ.get(f"MEDSEGAGENT_{prefix}_MODEL_PATH", "")
        if (
            not executable
            or not model
            or not Path(executable).is_file()
            or not Path(model).is_dir()
        ):
            raise _error("BACKEND_NOT_CONFIGURED", 503)
        device = core.device()
        if backend == "nninteractive" and device == "mps":
            raise _error("DEVICE_UNSUPPORTED", 503)
        config = SchedulerConfig.from_environment()
        minimum = 10240
        scheduler = GPUScheduler(
            replace(config, min_free_memory_mib=max(config.min_free_memory_mib, minimum)),
            device=device,
        )
        deadline = time.monotonic() + min(
            core._positive_int("MEDSEGAGENT_TIMEOUT_SECONDS", 7200), 7200
        )
        request = {
            **request,
            "model_path": str(Path(model).resolve()),
            "device": "cuda:0" if device == "gpu" else device,
        }
        request_path, response_path = run / "request.json", run / "response.json"
        core._write_json(request_path, request)
        script = Path(__file__).with_name(backend + "_worker.py")
        async with scheduler.acquire(deadline=deadline) as lease:
            await core._run_command(
                [
                    str(Path(executable).expanduser().absolute()),
                    str(script),
                    "--request",
                    str(request_path),
                    "--response",
                    str(response_path),
                ],
                timeout_seconds=max(0.01, deadline - time.monotonic()),
                output_dir=run,
                lock_fd=lease,
                on_start=lambda _pid: None,
            )
        if not _regular(response_path) or response_path.stat().st_size > 256 * 1024:
            raise _error("OUTPUT_INVALID")
        try:
            response = json.loads(response_path.read_text())
        except (OSError, UnicodeError, ValueError):
            raise _error("OUTPUT_INVALID") from None
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise _error("INFERENCE_FAILED")
        return response

    def download_path(self, principal, workspace_id, revision_id):
        self._workspace(principal, workspace_id)
        revision = self._revision(workspace_id, revision_id)
        path = self._directory(workspace_id) / "revisions" / revision_id / "mask.nii.gz"
        _verify_file(path, revision["sha256"])
        return path

    async def prepare_download(self, principal, workspace_id, revision_id):
        self._workspace(principal, workspace_id)
        revision = self._revision(workspace_id, revision_id)
        path = self._directory(workspace_id) / "revisions" / revision_id / "mask.nii.gz"
        await core._validation(_verify_file, path, revision["sha256"])
        self._workspace(principal, workspace_id)
        return path

    def cleanup(self):
        if self.closed:
            return
        for row in self.db.execute(
            "SELECT id FROM workspaces WHERE expires<=?", (time.time(),)
        ).fetchall():
            workspace_id = row[0]
            active = self.db.execute(
                "SELECT id FROM jobs WHERE workspace_id=? AND status NOT IN ('completed','failed','canceled','input_required')",
                (workspace_id,),
            ).fetchall()
            if active:
                for job in active:
                    pending = self.active.get(job[0])
                    if pending:
                        pending.cancel()
                continue
            with self.db:
                self.db.execute("DELETE FROM workspaces WHERE id=?", (workspace_id,))
            shutil.rmtree(self._directory(workspace_id), ignore_errors=True)
            self._workspace_locks.pop(workspace_id, None)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        pending = list(set(self.active.values()) | self._opening)
        for task in pending:
            task.cancel()
        drained = asyncio.gather(*pending, return_exceptions=True)
        while not drained.done():
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                continue
        for row in self.db.execute(
            "SELECT id FROM jobs WHERE status NOT IN ('completed','failed','canceled','input_required')"
        ).fetchall():
            self._update_job(
                row[0],
                "failed",
                error={"code": "SERVICE_RESTARTED", "message": _SAFE_ERRORS["SERVICE_RESTARTED"]},
            )
        self.db.close()

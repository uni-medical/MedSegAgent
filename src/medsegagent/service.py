"""Durable, identity-scoped jobs around the single local inference core.

Run one ASGI worker per data directory. SQLite owns status and idempotency; files are
private and served only through explicit authenticated endpoints. A crash never silently
restarts a job that had already begun inference.
"""

from __future__ import annotations

import asyncio
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import struct
import tempfile
import time
import uuid
from pathlib import Path
from time import perf_counter

from medsegagent import agent, core
from medsegagent.execution import TaskExecution
from medsegagent.gpu_scheduler import GPUScheduler
from medsegagent.result_metadata import model_provenance, result_metadata

PUBLIC_A2A_PRINCIPAL = "__public_a2a__"
TERMINAL = {"completed", "failed", "canceled"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
SINGLE_UPLOAD_BYTES = 90 * 1024 * 1024
UPLOAD_QUOTA_BYTES = 2 * 1024**3
INPUT_FIELDS = {"id", "name", "size", "shape", "spacing", "created_at", "expires_at", "example_id"}


class ServiceError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code


def uid() -> str:
    return str(uuid.uuid4())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _publish_requested_artifact(
    parent, artifact, requested, *, single_output=False, _stop_event=None
):
    """Freeze verified bytes, publish only requested regions, and leave originals private.

    This entire worker is owned by core._validation: cancellation waits for cleanup.
    Multi-output names preserve artifact identity even when mask bytes are identical.
    A single output retains the legacy filename.
    """
    import nibabel as nib
    import numpy as np

    parent = Path(parent).resolve()
    source = Path(artifact["path"])
    expected_hash = artifact.get("sha256")
    if (
        source.is_symlink()
        or not source.resolve().is_relative_to(parent)
        or not isinstance(expected_hash, str)
        or re.fullmatch(r"[a-f0-9]{64}", expected_hash) is None
    ):
        raise ValueError("Artifact ownership or verification receipt is invalid.")
    token = uuid.uuid4().hex
    snapshot = parent / f".publish-{token}.source.nii.gz"
    temporary = parent / f".publish-{token}.nii.gz"
    label_map = {row["id"]: row["name"] for row in artifact["labels"]}
    source_rows = {row["id"]: row for row in artifact["labels"]}
    selected_values, rows, regions = set(), [], []
    for index, region in enumerate(requested, 1):
        values = region.get("values")
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value not in label_map for value in values)
            or len(set(values)) != len(values)
            or selected_values.intersection(values)
        ):
            raise ValueError("Requested regions must have valid, disjoint artifact label values.")
        selected_values.update(values)
        row = {"id": index, "name": region["target"], "color": core._label_color(index)}
        if len(values) == 1 and "source_id" in source_rows[values[0]]:
            row["source_id"] = source_rows[values[0]]["source_id"]
        rows.append(row)
        regions.append({**region, "values": [index]})
    if not rows or any(type(value) is not int or not 0 < value <= 65535 for value in label_map):
        raise ValueError("The artifact label table is invalid.")
    try:
        # Hash the same private snapshot that is later read. A path-only pre-check
        # would allow changed bytes to acquire a new, misleading publication hash.
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as incoming, snapshot.open("xb") as outgoing:
            os.chmod(snapshot, 0o600)
            before = os.fstat(incoming.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("The artifact must be a regular file.")
            if before.st_size != artifact.get("size_bytes"):
                raise ValueError("The artifact changed after verification.")
            checksum = hashlib.sha256()
            while block := incoming.read(1024 * 1024):
                core._check_stop(_stop_event)
                checksum.update(block)
                outgoing.write(block)
            after = os.fstat(incoming.fileno())
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ) or checksum.hexdigest() != expected_hash:
                raise ValueError("The artifact changed after verification.")
        geometry = core._inspect_nifti(snapshot, label_map=label_map, _stop_event=_stop_event)
        expected_geometry = artifact["geometry"]
        if (
            geometry["shape"] != expected_geometry["shape"]
            or not np.allclose(
                geometry["affine"], expected_geometry["affine"], rtol=1e-5, atol=1e-4
            )
            or not np.allclose(
                geometry["voxel_spacing"], expected_geometry["spacing"], rtol=1e-5, atol=1e-6
            )
        ):
            raise ValueError("The artifact geometry does not match its verification receipt.")
        dtype = np.uint8 if len(rows) <= 255 else np.uint16
        lookup = np.zeros(max(label_map) + 1, dtype=dtype)
        for row, region in zip(rows, requested, strict=True):
            lookup[region["values"]] = row["id"]
        counts = np.zeros(len(rows) + 1, dtype=np.int64)
        limit = core._positive_int(
            "MEDSEGAGENT_MAX_UNCOMPRESSED_BYTES", core.DEFAULT_MAX_UNCOMPRESSED_BYTES
        )
        with (
            core._uncompressed_nifti(snapshot, limit, _stop_event) as expanded,
            tempfile.TemporaryFile(mode="w+b") as raw,
        ):
            prefix = expanded.read(4)
            sizes = {struct.unpack("<i", prefix)[0], struct.unpack(">i", prefix)[0]}
            image_type = nib.Nifti2Image if 540 in sizes else nib.Nifti1Image
            expanded.seek(0)
            image = image_type.from_file_map(
                {"image": nib.FileHolder(fileobj=expanded)}, mmap=False
            )
            header = image.header.copy()
            header.set_data_dtype(dtype)
            header.set_slope_inter(1, 0)
            header.set_intent("label", name="MedSegAgent")
            header["descrip"] = b"MedSegAgent requested regions"
            header["aux_file"] = b""
            header.extensions.clear()
            header.extensions.append(core._label_extension(rows))
            header.set_data_offset(0)
            header.write_to(raw)
            raw.seek(int(header["vox_offset"]))
            for z in range(image.shape[2]):
                core._check_stop(_stop_event)
                plane = np.asanyarray(image.dataobj[:, :, z])
                remapped = lookup[np.asarray(plane, dtype=np.intp)]
                counts += np.bincount(remapped.ravel(), minlength=len(rows) + 1)
                raw.write(remapped.tobytes(order="F"))
            raw.seek(0)
            with temporary.open("xb") as outgoing:
                os.chmod(temporary, 0o600)
                with gzip.GzipFile(filename="", mode="wb", fileobj=outgoing, mtime=0) as compressed:
                    while block := raw.read(1024 * 1024):
                        core._check_stop(_stop_event)
                        compressed.write(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        verified = core._inspect_nifti(
            temporary, label_map={row["id"]: row["name"] for row in rows}, _stop_event=_stop_event
        )
        if (
            verified["shape"] != geometry["shape"]
            or not np.allclose(verified["affine"], geometry["affine"], rtol=1e-5, atol=1e-4)
            or not np.allclose(
                verified["voxel_spacing"], geometry["voxel_spacing"], rtol=1e-5, atol=1e-6
            )
        ):
            raise ValueError("Published geometry does not match its source.")
        observed = {row["id"]: row["voxels"] for row in verified["labels"]}
        voxel_volume = artifact["volume_measurement"]["voxel_volume_mm3"]
        for row, region in zip(rows, regions, strict=True):
            count = int(counts[row["id"]])
            if count != observed.get(row["id"], 0) or count != region["voxels"]:
                raise ValueError(
                    "Published region measurements do not match verified source values."
                )
            row.update(
                voxels=count, volume_mm3=count * voxel_volume, volume_ml=count * voxel_volume / 1000
            )
            region.update(voxels=count, volume_mm3=row["volume_mm3"], volume_ml=row["volume_ml"])
        output_hash = core._file_digest(temporary, _stop_event)
        identity = hashlib.sha256(
            json.dumps(
                [
                    artifact["id"],
                    artifact.get("task"),
                    artifact.get("quality"),
                    [region["id"] for region in requested],
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:32]
        filename = (
            "segmentation.nii.gz" if single_output else f"output-{identity}-{output_hash}.nii.gz"
        )
        destination = parent / filename
        core._check_stop(_stop_event)
        if destination.is_symlink():
            raise ValueError("The public destination must not be a symlink.")
        os.replace(temporary, destination)
        nonzero = sum(row["voxels"] for row in rows)
        return {
            "schema_version": core.RESULT_SCHEMA_VERSION,
            "id": artifact["id"],
            "name": ", ".join(row["name"] for row in rows),
            "targets": [row["name"] for row in rows],
            "labels": rows,
            "regions": regions,
            "geometry": {
                "shape": verified["shape"],
                "spacing": verified["voxel_spacing"],
                "affine": verified["affine"],
            },
            "segmentation_shape": verified["shape"],
            "segmentation_voxel_spacing": verified["voxel_spacing"],
            "volume_measurement": artifact["volume_measurement"],
            "nonzero_voxels": nonzero,
            "detection_status": "target_detected" if nonzero else "no_target_detected",
            "no_target_detected": not bool(nonzero),
            "file": {
                "name": filename,
                "media_type": "application/gzip",
                "sha256": output_hash,
                "size_bytes": destination.stat().st_size,
            },
        }
    finally:
        snapshot.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


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
            CREATE TABLE IF NOT EXISTS upload_sessions
              (id TEXT PRIMARY KEY, principal TEXT NOT NULL, message_id TEXT NOT NULL,
               name TEXT NOT NULL, size INTEGER NOT NULL, received INTEGER NOT NULL,
               created REAL NOT NULL, updated REAL NOT NULL, chunks TEXT NOT NULL,
               UNIQUE(principal,message_id));
            CREATE TABLE IF NOT EXISTS service_metadata
              (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        try:
            self._ensure_public_namespace()
        except Exception:
            self.db.close()
            raise
        from .conversations import Conversations

        self.conversations = Conversations(self)
        self.active: dict[str, asyncio.Task] = {}
        self.canceling: set[str] = set()
        self.worker_slot = asyncio.Semaphore(GPUScheduler(device=core.device()).capacity)
        self.export_slots = asyncio.Semaphore(2)
        self.exporting: dict[str, int] = {}
        self.uploading: set[str] = set()
        self.lock = None
        self.cleanup_task = None
        self.closed = False

    def _ensure_public_namespace(self):
        """Never reinterpret an existing account as the anonymous public namespace."""
        key = "public_a2a_namespace_v1"
        marker = self.db.execute(
            "SELECT value FROM service_metadata WHERE key=?", (key,)
        ).fetchone()
        if marker:
            if marker[0] != PUBLIC_A2A_PRINCIPAL:
                raise ValueError("The persisted public A2A namespace is incompatible.")
            return
        for table in ("tasks", "uploads", "upload_sessions", "sessions"):
            if self.db.execute(
                f"SELECT 1 FROM {table} WHERE principal=? LIMIT 1", (PUBLIC_A2A_PRINCIPAL,)
            ).fetchone():
                raise ValueError("The public A2A namespace collides with existing private records.")
        if (
            self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='auth_principals'"
            ).fetchone()
            and self.db.execute(
                "SELECT 1 FROM auth_principals WHERE id=?", (PUBLIC_A2A_PRINCIPAL,)
            ).fetchone()
        ):
            raise ValueError("The public A2A namespace collides with an existing private account.")
        with self.db:
            self.db.execute(
                "INSERT INTO service_metadata(key,value) VALUES(?,?)", (key, PUBLIC_A2A_PRINCIPAL)
            )

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
        known_uploads.update(row[0] for row in self.db.execute("SELECT id FROM upload_sessions"))
        for row in self.db.execute("SELECT * FROM upload_sessions").fetchall():
            directory = self.root / "uploads" / row["id"]
            if directory.is_dir() and not directory.is_symlink():
                for temporary in directory.glob(".*.chunk"):
                    temporary.unlink(missing_ok=True)
                name = "image.nii.gz" if row["name"].endswith(".gz") else "image.nii"
                path = directory / name
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and path.stat().st_size > row["received"]
                ):
                    with os.fdopen(os.open(path, os.O_WRONLY | os.O_NOFOLLOW), "wb") as file:
                        file.truncate(row["received"])
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
            if task["status"] == "input_required" and task.get("a2a"):
                continue  # Waiting conversations persist across process restarts.
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

    def expire_a2a_input(self, task_id):
        task = self._task(task_id)
        if (
            task.get("a2a")
            and task["status"] == "input_required"
            and time.time()
            >= task.get("input_expires_at", task["updated_at"] + self.retention_seconds)
        ):
            self.update(
                task_id,
                status="failed",
                progress="Waiting for input timed out",
                error={
                    "code": "INPUT_EXPIRED",
                    "message": "The input-waiting period expired. Start a new task.",
                },
            )

    def get(self, principal, task_id):
        task = self._task(task_id)
        if task["principal"] != principal:
            raise ServiceError("TASK_NOT_FOUND", "Task not found.", 404)
        self.expire_a2a_input(task_id)
        task = self._task(task_id)
        now = time.time()
        if task["status"] in TERMINAL:
            finished_at = task.get("finished_at")
            if "finished_at" not in task:
                # Recover old records from their first terminal event without
                # rewriting them. Later publication/cleanup changes updated_at.
                event = self.db.execute(
                    "SELECT time FROM events WHERE task_id=? "
                    "AND status IN ('completed','failed','canceled') ORDER BY seq LIMIT 1",
                    (task_id,),
                ).fetchone()
                finished_at = event["time"] if event else None
            task["finished_at"] = finished_at
            task["elapsed_seconds"] = (
                max(0, finished_at - task["created_at"]) if finished_at is not None else None
            )
        elif task["status"] == "input_required":
            task["elapsed_seconds"] = max(
                0, task.get("input_requested_at", task["updated_at"]) - task["created_at"]
            )
        else:
            # Queueing, Agent reasoning, inference and publication share one clock.
            task["elapsed_seconds"] = max(0, now - task["created_at"])
        task["input"] = self.input_metadata(task)
        task["upload_name"] = task["input"].get("name", "")
        task["input_available"] = task["input"]["available"]
        expires_at = self.task_expiry(task)
        if expires_at is not None:
            task["expires_at"] = expires_at
            if now >= expires_at or task.get("files_expired"):
                task.update(files_expired=True, files=[], result=None)
        result = task.get("result")
        file_base = "/a2a/tasks" if principal == PUBLIC_A2A_PRINCIPAL else "/api/tasks"
        available = False
        if isinstance(result, dict):
            public = result_metadata(result)
            outputs = public.get("outputs") or [public]
            files = []
            for output in outputs:
                output_id = output.get("id", "segmentation")
                output_files = []
                for file in output.get("files", task.get("files", [])):
                    name = file.get("name", "")
                    # Files come from the stored publication manifest; never resolve a
                    # caller-supplied path or advertise private inference artifacts.
                    if (
                        file.get("kind") == "label"
                        or not isinstance(name, str)
                        or name != Path(name).name
                        or not (
                            name == "segmentation.nii.gz"
                            or re.fullmatch(r"output-[a-f0-9]{32}-[a-f0-9]{64}\.nii\.gz", name)
                        )
                    ):
                        continue
                    path = self.root / "tasks" / task_id / name
                    if (
                        (task["status"] not in TERMINAL and not task.get("a2a"))
                        or not path.is_file()
                        or path.is_symlink()
                    ):
                        continue
                    available = True
                    output_files.append(
                        {
                            **file,
                            "kind": "overlay",
                            "output_id": output_id,
                            "url": f"{self.public_url}{file_base}/{task_id}/files/{name}",
                        }
                    )
                if output_files:
                    token = hashlib.sha256(str(output_id).encode()).hexdigest()[:32]
                    for label in sorted(
                        (
                            x
                            for x in output.get("labels", [])
                            if isinstance(x, dict) and type(x.get("id")) is int and x["id"] > 0
                        ),
                        key=lambda x: x["id"],
                    ):
                        try:
                            name = core.label_mask_filename(label)
                        except ValueError:
                            continue
                        if task.get("a2a") or len(outputs) > 1:
                            name = name.removesuffix(".nii.gz") + f"-{token}.nii.gz"
                        output_files.append(
                            {
                                "kind": "label",
                                "output_id": output_id,
                                "label_id": label["id"],
                                "label_name": label["name"],
                                "mask_value": 1,
                                "name": name,
                                "media_type": "application/gzip",
                                "url": f"{self.public_url}{file_base}/{task_id}/files/{name}",
                            }
                        )
                output["files"] = output_files
                files.extend(output_files)
            task["files"] = public["files"] = files
            task["result"] = public
        task["result_available"] = available
        return {
            k: v
            for k, v in task.items()
            if k
            not in {
                "principal",
                "selection",
                "agent_history",
                "a2a_prior_result",
                "a2a_last_response_to",
                "a2a_message_id",
            }
        }

    def task_expiry(self, task):
        if task.get("a2a") and task["status"] == "input_required":
            return task.get("input_expires_at", task["updated_at"] + self.retention_seconds)
        if task["status"] not in TERMINAL:
            return None
        return task.get("expires_at", task["updated_at"] + self.retention_seconds)

    def upload_metadata(self, row):
        data = json.loads(row["data"])
        expires_at = data.get("expires_at", row["created"] + self.retention_seconds)
        active = False
        for record in self.db.execute(
            "SELECT data FROM tasks WHERE principal=? AND json_extract(data, '$.upload_id')=?",
            (row["principal"], row["id"]),
        ).fetchall():
            task = json.loads(record["data"])
            if task["status"] not in TERMINAL:
                active = True
            elif not task.get("files_expired") or "expires_at" in task:
                # Older expired records used updated_at for cleanup itself. That timestamp
                # must not accidentally renew input retention after an upgrade.
                expires_at = max(expires_at, self.task_expiry(task))
        path = Path(row["path"])
        data.update(
            expires_at=None if active else expires_at,
            available=(active or time.time() < expires_at)
            and path.is_file()
            and not path.is_symlink(),
        )
        return data

    def input_metadata(self, task):
        row = self.db.execute(
            "SELECT * FROM uploads WHERE id=? AND principal=?",
            (task["upload_id"], task["principal"]),
        ).fetchone()
        if row:
            data = self.upload_metadata(row)
            return {
                key: value for key, value in data.items() if key in INPUT_FIELDS | {"available"}
            }
        # Preserve the identity of an unavailable input without preserving the image itself.
        data = {key: value for key, value in task.get("input", {}).items() if key in INPUT_FIELDS}
        expires_at = self.task_expiry(task)
        if expires_at is not None:
            data["expires_at"] = max(data.get("expires_at") or 0, expires_at)
        data.update(id=task["upload_id"], available=False)
        return data

    def list(self, principal):
        if principal == PUBLIC_A2A_PRINCIPAL:
            raise ServiceError("LIST_UNSUPPORTED", "Public task listing is not supported.", 405)
        rows = self.db.execute(
            "SELECT id FROM tasks WHERE principal=? ORDER BY created DESC LIMIT 100", (principal,)
        ).fetchall()
        return [self.get(principal, row["id"]) for row in rows]

    def update(self, task_id, **changes):
        task = self._task(task_id)
        now = time.time()
        # Completion time belongs to the lifecycle, not later metadata updates.
        changes.pop("finished_at", None)
        if changes.get("status") == "input_required" and task.get("a2a"):
            changes.setdefault("input_expires_at", now + self.retention_seconds)
            changes.setdefault("input_requested_at", now)
        if changes.get("status") in TERMINAL and task["status"] not in TERMINAL:
            changes.setdefault("expires_at", now + self.retention_seconds)
            if task.get("finished_at") is None:
                changes["finished_at"] = now
        elif task["status"] in TERMINAL and "finished_at" not in task:
            event = self.db.execute(
                "SELECT time FROM events WHERE task_id=? "
                "AND status IN ('completed','failed','canceled') ORDER BY seq LIMIT 1",
                (task_id,),
            ).fetchone()
            # This metadata update must not become false completion evidence
            # for an old record whose original terminal event is missing.
            changes["finished_at"] = event["time"] if event else None
        task.update(changes, updated_at=now)
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
        data = self.upload_metadata(row) if row else None
        if not data or not data["available"]:
            raise ServiceError("FILE_NOT_FOUND", "Upload not found or expired.", 404)
        return data

    def upload_path(self, principal, upload_id):
        self.get_upload(principal, upload_id)
        return Path(
            self.db.execute("SELECT path FROM uploads WHERE id=?", (upload_id,)).fetchone()[0]
        )

    def reserve_upload(self, principal, filename, size_bytes=None):
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
        reserved, pending_count = self.db.execute(
            "SELECT COALESCE(SUM(size),0),COUNT(*) FROM upload_sessions "
            "WHERE principal=? AND id NOT IN (SELECT id FROM uploads)",
            (principal,),
        ).fetchone()
        expected = self.max_upload_bytes if size_bytes is None else size_bytes
        if size + reserved + expected > UPLOAD_QUOTA_BYTES or count + pending_count >= 16:
            raise ServiceError(
                "QUOTA_EXHAUSTED",
                "Upload quota reached. Delete old uploads or wait for expiry.",
                429,
            )
        if shutil.disk_usage(self.root).free < expected + 4 * 1024**3:
            raise ServiceError("CAPACITY_EXCEEDED", "Storage is busy; retry later.", 503)
        upload_id = uid()
        directory = self.root / "uploads" / upload_id
        directory.mkdir(mode=0o700)
        self.uploading.add(principal)
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
            self.db.execute("DELETE FROM upload_sessions WHERE id=?", (upload_id,))

    async def submit(self, principal, upload_id, text, modality, message_id, context_id=None):
        agent.provider_payload(text, modality)  # Local validation before accepting work.
        if not isinstance(message_id, str) or not 1 <= len(message_id) <= 128:
            raise ServiceError("INVALID_PARAMS", "messageId must have 1–128 characters.")
        if context_id is not None and (
            not isinstance(context_id, str) or not 1 <= len(context_id) <= 128
        ):
            raise ServiceError("INVALID_PARAMS", "Invalid contextId.")
        if self.db.execute(
            "SELECT 1 FROM a2a_requests WHERE principal=? AND message_id=?", (principal, message_id)
        ).fetchone():
            raise ServiceError("IDEMPOTENCY_CONFLICT", "messageId belongs to an A2A message.", 409)
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
        upload = self.get_upload(principal, upload_id)
        if context_id:
            owner = self.db.execute(
                "SELECT principal FROM tasks WHERE context_id=? LIMIT 1", (context_id,)
            ).fetchone()
            if not owner or owner[0] != principal:
                raise ServiceError("INVALID_PARAMS", "contextId is not available to this identity.")
        pending = self.db.execute(
            "SELECT principal FROM tasks WHERE status NOT IN ('completed','failed','canceled','input_required')"
        ).fetchall()
        if len(pending) >= 8 or sum(row[0] == principal for row in pending) >= 4:
            raise ServiceError(
                "CAPACITY_EXCEEDED", "Task capacity reached; retry after a task finishes.", 429
            )
        task_id, now = uid(), time.time()
        task = {
            "id": task_id,
            "context_id": context_id or uid(),
            "principal": principal,
            "upload_id": upload_id,
            "input": {key: value for key, value in upload.items() if key in INPUT_FIELDS},
            "text": text,
            "modality": modality,
            "modality_source": "parameter" if modality is not None else "unknown",
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

    async def submit_a2a(
        self, principal, upload_id, text, modality, message_id, context_id=None, task_id=None
    ):
        if task_id:
            # Check identity before allowing an expiry transition.
            self.get(principal, task_id)
            self.expire_a2a_input(task_id)
        return self.conversations.submit(
            principal, upload_id, text, modality, message_id, context_id, task_id
        )

    def record_a2a_response(self, task_id, text):
        self.conversations.record_response(task_id, text)

    def launch(self, task_id):
        worker = asyncio.create_task(self.run(task_id))
        self.active[task_id] = worker

        def finished(completed):
            if self.active.get(task_id) is completed:
                self.active.pop(task_id, None)

        worker.add_done_callback(finished)

    async def publish_execution(self, task_id, execution, outcome):
        """Publish requested objects, with source-only labels removed from public masks."""
        exported = execution.export_result()
        a2a = bool(getattr(execution, "_service_a2a_publication", False))
        cache = getattr(execution, "_service_published_outputs", {}) if a2a else {}
        if a2a:
            execution._service_published_outputs = cache
        parent = self.root / "tasks" / task_id
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        artifacts = {row["id"]: row for row in exported.get("artifacts", [])}
        regions = {row["id"]: row for row in exported.get("regions", [])}
        requested = exported.get("outputs", [])
        if a2a and not requested:
            previous = self._task(task_id).get("result")
            if isinstance(previous, dict) and previous.get("outputs"):
                # A resumed conversation has a fresh execution. A clarification
                # without new outputs must not erase that Task's verified files.
                public = result_metadata(previous)
                public.update(
                    summary=outcome.get("summary", ""),
                    completion={
                        "status": outcome.get("status", "failed"),
                        "unresolved": outcome.get("unresolved", []),
                    },
                    agent={
                        key: outcome[key]
                        for key in ("model", "model_requests", "tool_calls")
                        if key in outcome
                    },
                )
                self.update(task_id, result=public, files=public.get("files", []))
                return public
        grouped = {}
        seen_outputs = set()
        for item in requested:
            region = regions.get(item.get("region_id"))
            identity = (item.get("task"), item.get("target"), item.get("quality"))
            artifact = artifacts.get(item.get("artifact_id"))
            if (
                region is None
                or region.get("artifact_id") != item.get("artifact_id")
                or artifact is None
                or region.get("target") != item.get("target")
                or region.get("task") != item.get("task")
                or region.get("quality") != item.get("quality")
                or artifact.get("task") != item.get("task")
                or artifact.get("quality") != item.get("quality")
                or identity in seen_outputs
            ):
                raise ValueError("The requested-output manifest is inconsistent.")
            seen_outputs.add(identity)
            metadata = model_provenance(region, regions)
            grouped.setdefault(item["artifact_id"], []).append({**region, **metadata})
        outputs, files, public_regions = [], [], []
        for artifact_id, selected_regions in grouped.items():
            signature = hashlib.sha256(
                json.dumps([artifacts[artifact_id], selected_regions], sort_keys=True).encode()
            ).hexdigest()
            cached = cache.get(signature)
            destination = parent / cached["output"]["file"]["name"] if cached else None
            if (
                cached
                and destination.is_file()
                and not destination.is_symlink()
                and destination.stat().st_mtime_ns == cached["mtime_ns"]
                and destination.stat().st_size == cached["size"]
            ):
                output = json.loads(json.dumps(cached["output"]))
            else:
                output = await core._validation(
                    _publish_requested_artifact,
                    parent,
                    artifacts[artifact_id],
                    selected_regions,
                    # An A2A result can grow after its first publication. Never
                    # overwrite a previously advertised source with a larger mask.
                    single_output=not a2a and len(grouped) == 1,
                )
                if a2a:
                    destination = parent / output["file"]["name"]
                    cache[signature] = {
                        "output": json.loads(json.dumps(output)),
                        "mtime_ns": destination.stat().st_mtime_ns,
                        "size": destination.stat().st_size,
                    }
            file = output.pop("file")
            file["url"] = f"/api/tasks/{task_id}/files/{file['name']}"
            files.append(file)
            output.update(
                **{
                    key: selected_regions[0][key]
                    for key in ("task", "quality", "usage_license", "requirements", "model_sources")
                    if key in selected_regions[0]
                },
                modality=execution.modality,
                warning=(
                    "Research use only. Outputs require review; no clinical validation."
                    + (
                        " No requested target was detected; this does not rule out disease."
                        if output["no_target_detected"]
                        else ""
                    )
                ),
                files=[file],
            )
            projected = result_metadata(output) | {"id": output["id"], "name": output["name"]}
            outputs.append(projected)
            public_regions.extend(projected["regions"])
        prior = result_metadata(self._task(task_id).get("a2a_prior_result") or {}) if a2a else {}
        if prior.get("outputs"):
            # Each resumed turn starts from one fixed snapshot. Merge that
            # snapshot, never the preceding progress publication, so repeated
            # observed events retain earlier URLs without duplicating outputs.
            output_ids = {output["id"] for output in outputs}
            outputs = [
                output for output in prior["outputs"] if output["id"] not in output_ids
            ] + outputs
            region_ids = {region["id"] for region in public_regions}
            public_regions = [
                region for region in prior.get("regions", []) if region["id"] not in region_ids
            ] + public_regions
            filenames = {file["name"] for file in files}
            files = [
                file for file in prior.get("files", []) if file["name"] not in filenames
            ] + files
        target_counts = {}
        for output in outputs:
            for target in output["targets"]:
                target_counts[target] = target_counts.get(target, 0) + 1
        for output in outputs:
            if any(target_counts[target] > 1 for target in output["targets"]):
                qualifier = " / ".join(
                    value for value in (output["task"], output.get("quality")) if value
                )
                suffix = f" ({qualifier})"
                if not output["name"].endswith(suffix):
                    output["name"] += suffix
        model_sources = {
            (item["task"], item["quality"]): item for item in prior.get("model_sources", [])
        }
        for output in outputs:
            for item in output.get("model_sources", [output]):
                model_sources[(item["task"], item["quality"])] = {
                    key: item[key] for key in ("task", "quality", "usage_license", "requirements")
                }
        public = {
            "schema_version": 5,
            "modality": execution.modality,
            "modality_detection": exported.get("modality_detection"),
            "outputs": outputs,
            "regions": public_regions,
            "files": files,
            "model_sources": list(model_sources.values()),
            "summary": outcome.get("summary", ""),
            "completion": {
                "status": outcome.get("status", "failed"),
                "unresolved": outcome.get("unresolved", []),
            },
            "agent": {
                key: outcome[key]
                for key in ("model", "model_requests", "tool_calls")
                if key in outcome
            },
            "warning": "Research use only. Outputs require review; no clinical validation.",
        }
        # Old clients retain the complete single-mask view. Never select only the
        # first artifact when several outputs are required.
        if len(outputs) == 1:
            public = {**outputs[0], **public}
            public["warning"] = outputs[0]["warning"]
        public = result_metadata(public)
        self.update(task_id, result=public, files=files)
        return public

    async def run(self, task_id):
        execution = None
        final_changes = None
        final_response = ""
        published_final = False
        intermediate_signature = None
        started = perf_counter()
        timings = {"events": [], "model_seconds": 0.0, "tool_seconds": 0.0}
        try:
            task = self._task(task_id)
            created = task.get("attempt_started_at", task["created_at"])
            deadline = created + int(os.environ.get("MEDSEGAGENT_TIMEOUT_SECONDS", "7200"))
            async with asyncio.timeout(max(0, deadline - time.time())), self.worker_slot:
                timings["service_queue_seconds"] = perf_counter() - started
                task = self._task(task_id)
                self.update(
                    task_id, status="routing", progress="Understanding segmentation request"
                )
                # Only the explicit API parameter binds execution. Natural-language
                # declarations, negations and history remain for the Agent to interpret.
                modality = task["modality"]
                modality_source = "user_declaration"
                example_modality_hint = None
                if modality is None:
                    upload = self.get_upload(task["principal"], task["upload_id"])
                    if upload.get("example_id") and upload.get("source_modality") in {"CT", "MR"}:
                        example_modality_hint = upload["source_modality"]
                parent = self.root / "tasks" / task_id
                execution = TaskExecution(
                    input_path=str(self.upload_path(task["principal"], task["upload_id"])),
                    modality=modality,
                    output_dir=str(parent / "execution"),
                    modality_source=modality_source,
                    example_modality_hint=example_modality_hint,
                    on_inference_start=lambda: self.update(
                        task_id, status="running", progress="Running local segmentation"
                    ),
                )
                execution._service_a2a_publication = bool(task.get("a2a"))

                async def progress(event):
                    nonlocal intermediate_signature
                    timing = event.get("timing")
                    if isinstance(timing, dict):
                        stage, duration = timing.get("stage"), timing.get("duration_seconds")
                        if (
                            stage in {"model_request", "tool"}
                            and type(duration) in {int, float}
                            and math.isfinite(duration)
                            and duration >= 0
                        ):
                            row = {
                                "stage": stage,
                                "duration_seconds": duration,
                                "ended_at_seconds": perf_counter() - started,
                                "status": timing.get("status"),
                            }
                            if timing.get("tool") in agent.WORK_TOOLS:
                                row["tool"] = timing["tool"]
                            if type(timing.get("model_request")) is int:
                                row["model_request"] = timing["model_request"]
                            timings["events"].append(row)
                            key = "model_seconds" if stage == "model_request" else "tool_seconds"
                            timings[key] += duration
                    safe = {
                        k: event[k]
                        for k in ("phase", "tool", "step", "model_requests")
                        if k in event
                    }
                    fields = {"modality": execution.modality}
                    detection = execution.snapshot().get("modality_detection")
                    if detection:
                        fields["modality_detection"] = detection
                    if execution.modality is not None and modality is None:
                        fields["modality_source"] = "agent"
                    self.update(
                        task_id,
                        status="running",
                        progress="Working on segmentation request",
                        agent_progress=safe,
                        timings=timings,
                        **fields,
                    )
                    if (
                        task.get("a2a")
                        and event.get("phase") == "observed"
                        and execution.has_outputs
                    ):
                        exported = execution.export_result()
                        signature = hashlib.sha256(
                            json.dumps(
                                {
                                    key: exported.get(key)
                                    for key in ("artifacts", "regions", "outputs")
                                },
                                sort_keys=True,
                            ).encode()
                        ).hexdigest()
                        if signature != intermediate_signature:
                            publication_started = perf_counter()
                            try:
                                await self.publish_execution(
                                    task_id,
                                    execution,
                                    {
                                        "status": "working",
                                        "summary": "Verified intermediate segmentation results are available.",
                                        "unresolved": [],
                                    },
                                )
                                intermediate_signature = signature
                            finally:
                                timings["intermediate_publication_seconds"] = (
                                    timings.get("intermediate_publication_seconds", 0.0)
                                    + perf_counter()
                                    - publication_started
                                )

                self.update(
                    task_id,
                    modality=modality,
                    modality_source="parameter" if modality is not None else "unknown",
                )
                outcome = await agent.run_agent(
                    task["text"],
                    modality,
                    execution,
                    on_progress=progress,
                    **({"history": task.get("agent_history", [])} if task.get("a2a") else {}),
                )
                self.update(
                    task_id,
                    progress="Preparing segmentation results",
                    agent_progress={
                        **(self._task(task_id).get("agent_progress") or {}),
                        **{
                            key: outcome[key]
                            for key in ("model_requests", "tool_calls")
                            if key in outcome
                        },
                        "phase": "publishing",
                    },
                    modality=execution.modality,
                    modality_source=(
                        "parameter"
                        if modality is not None
                        else "agent"
                        if execution.modality is not None
                        else "unknown"
                    ),
                )
                complete = (
                    outcome["status"] == "completed"
                    and execution.has_outputs
                    and not execution.unresolved_failures
                    and not outcome.get("unresolved")
                )
                if outcome["status"] == "completed" and not complete:
                    outcome = {
                        **outcome,
                        "status": "failed",
                        "summary": "任务尚未完整完成；已生成的结果不能代表全部要求已满足。",
                        "unresolved": outcome.get("unresolved")
                        or ["执行结果缺失或仍有未解决的操作失败。"],
                    }
                publication_started = perf_counter()
                try:
                    await self.publish_execution(task_id, execution, outcome)
                    published_final = True
                finally:
                    timings["publication_seconds"] = perf_counter() - publication_started
                final_response = outcome.get("summary") or (
                    "Segmentation completed."
                    if complete
                    else "Some requested outputs could not be completed."
                )
                waiting = bool(task.get("a2a") and outcome["status"] == "needs_input")
                final_changes = {
                    "status": "completed"
                    if complete
                    else "input_required"
                    if waiting
                    else "failed",
                    "progress": "Segmentation complete"
                    if complete
                    else "Request has unresolved requirements",
                    "error": None
                    if complete
                    else {
                        "code": "INPUT_REQUIRED"
                        if outcome["status"] == "needs_input"
                        else "INCOMPLETE_TASK",
                        "message": outcome.get("summary")
                        or "Some requested outputs could not be completed.",
                    },
                    "expires_at": time.time() + self.retention_seconds,
                }
                if waiting:
                    final_changes["input_expires_at"] = time.time() + self.retention_seconds
        except asyncio.CancelledError:
            final_changes = {
                "status": "canceled" if not self.closed else "failed",
                "progress": "Canceled" if not self.closed else "Interrupted by service shutdown",
                "error": {
                    "code": "CANCELED" if not self.closed else "SERVER_RESTART",
                    "message": "Task stopped. Submit a new message to retry.",
                },
            }
        except TimeoutError:
            final_changes = {
                "status": "failed",
                "progress": "Task time limit exceeded",
                "error": {
                    "code": "TASK_TIMEOUT",
                    "message": "The task exceeded its total queue/routing/inference time limit.",
                },
            }
        except agent.RoutingError as exc:
            waiting = bool(
                self._task(task_id).get("a2a")
                and exc.code in {"INPUT_REQUIRED", "MODALITY_REQUIRED"}
            )
            final_changes = {
                "status": "input_required" if waiting else "failed",
                "progress": "请补充影像模态"
                if exc.code == "MODALITY_REQUIRED"
                else "Segmentation request could not be fulfilled",
                "error": {"code": exc.code, "message": str(exc)},
            }
            if waiting:
                final_changes["input_expires_at"] = time.time() + self.retention_seconds
        except Exception:  # noqa: BLE001 - project a generic public error and persist failure.
            # Provider bodies, subprocess output, paths and credentials are never HTTP errors.
            final_changes = {
                "status": "failed",
                "progress": "Local inference failed",
                "error": {
                    "code": "INFERENCE_FAILED",
                    "message": "Local inference failed. The operator can inspect the private run log.",
                },
            }

        finally:

            async def finalize():
                # Tool wall time includes its nested inference work. Per-producer
                # durations may overlap on different GPUs; do not add them to wall time.
                if execution is not None and hasattr(execution, "export_result"):
                    try:
                        manifest = execution.export_result()
                    except Exception:  # noqa: BLE001 — a damaged manifest cannot prevent task closure.
                        manifest = {}
                        final_changes["publication_error"] = "ARTIFACT_PUBLICATION_FAILED"
                    timings["inferences"] = manifest.get("inference_timings") or [
                        {
                            "task": result.get("task"),
                            "quality": result.get("speed"),
                            "engine": result.get("inference_engine"),
                            "seconds": result.get("timings_seconds", {}),
                        }
                        for result in manifest.get("backend_results", [])
                    ]
                if execution is not None and execution.has_outputs and not published_final:
                    partial_started = perf_counter()
                    try:
                        await self.publish_execution(
                            task_id,
                            execution,
                            {
                                "status": "needs_input"
                                if final_changes["status"] == "input_required"
                                else "failed",
                                "summary": "Verified partial results are available; the task did not finish.",
                                "unresolved": [
                                    (final_changes.get("error") or {}).get(
                                        "code", "INCOMPLETE_TASK"
                                    )
                                ],
                            },
                        )
                    except Exception:  # noqa: BLE001 — close the lifecycle without leaking backend errors.
                        final_changes["publication_error"] = "ARTIFACT_PUBLICATION_FAILED"
                        # Earlier intermediate outputs remain useful when a later
                        # publication fails, but their completion metadata must
                        # describe the final failed or paused turn.
                        previous = self._task(task_id).get("result")
                        if isinstance(previous, dict):
                            self.update(
                                task_id,
                                result={
                                    **previous,
                                    "summary": "Previously verified partial results remain available.",
                                    "completion": {
                                        "status": "needs_input"
                                        if final_changes["status"] == "input_required"
                                        else "failed",
                                        "unresolved": ["ARTIFACT_PUBLICATION_FAILED"],
                                    },
                                },
                            )
                    finally:
                        timings["partial_publication_seconds"] = perf_counter() - partial_started
                current = self._task(task_id)
                if current.get("a2a") and not published_final:
                    previous = current.get("result")
                    if isinstance(previous, dict) and previous.get("outputs"):
                        # An exception can happen before a resumed execution makes
                        # any new output. Keep the files, but describe this turn's
                        # failure instead of the preceding input-required state.
                        error = final_changes.get("error") or {}
                        self.update(
                            task_id,
                            result={
                                **previous,
                                "summary": error.get("message")
                                or "Verified partial results remain available.",
                                "completion": {
                                    "status": "needs_input"
                                    if final_changes["status"] == "input_required"
                                    else "failed",
                                    "unresolved": [
                                        final_changes.get("publication_error")
                                        or error.get("code", "INCOMPLETE_TASK")
                                    ],
                                },
                            },
                        )
                if current.get("a2a"):
                    self.record_a2a_response(
                        task_id,
                        final_response
                        or (final_changes.get("error") or {}).get("message")
                        or final_changes["progress"],
                    )
                # A terminal snapshot is authoritative: publish all available files,
                # record the reply and persist timings before announcing completion.
                self.update(task_id, **final_changes, timings=timings)

            # Disconnects or repeated cancel requests must not interrupt verified
            # artifact publication and leave a terminal Task with changing results.
            closing = asyncio.create_task(finalize())
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    continue
            closing.result()

    async def cancel(self, principal, task_id):
        task = self.get(principal, task_id)
        if task["status"] in TERMINAL:
            if task["status"] == "canceled":
                return task
            raise ServiceError("TASK_NOT_CANCELABLE", "Task is already terminal.", 409)
        # Reserve cancellation before yielding to cleanup. A waiting task's old
        # worker may still be exiting; a continuation must not replace it here.
        self.canceling.add(task_id)
        try:
            worker = self.active.get(task_id)
            if worker:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            # A queued coroutine may have been canceled before its first instruction.
            if self.get(principal, task_id)["status"] not in TERMINAL:
                self.record_a2a_response(task_id, "Task canceled.")
                self.update(task_id, status="canceled", progress="Canceled")
            return self.get(principal, task_id)
        finally:
            self.canceling.discard(task_id)

    def file_path(self, principal, task_id, name):
        task = self.get(principal, task_id)
        if not isinstance(name, str) or not any(
            file.get("kind") == "overlay" and file.get("name") == name
            for file in task.get("files", [])
        ):
            raise ServiceError("FILE_NOT_FOUND", "Result not found.", 404)
        path = self.root / "tasks" / task_id / name
        if task.get("files_expired") or not path.is_file() or path.is_symlink():
            raise ServiceError("FILE_NOT_FOUND", "Result expired or unavailable.", 404)
        return path

    async def download_path(self, principal, task_id, name):
        task = self.get(principal, task_id)
        entry = next((file for file in task.get("files", []) if file.get("name") == name), None)
        if entry is None or not task["result_available"]:
            raise ServiceError("FILE_NOT_FOUND", "Result not found.", 404)
        if entry.get("kind") == "overlay":
            return self.file_path(principal, task_id, name)
        outputs = task["result"].get("outputs") or [task["result"]]
        output = next(
            (item for item in outputs if item.get("id", "segmentation") == entry["output_id"]), None
        )
        source_file = (
            next((file for file in output.get("files", []) if file.get("kind") == "overlay"), None)
            if output
            else None
        )
        if source_file is None:
            raise ServiceError("FILE_NOT_FOUND", "Result not found.", 404)
        async with self.export_slots:
            source = self.file_path(principal, task_id, source_file["name"])
            self.exporting[task_id] = self.exporting.get(task_id, 0) + 1
            try:
                options = {}
                if task.get("a2a") or len(outputs) > 1:
                    token = hashlib.sha256(str(entry["output_id"]).encode()).hexdigest()[:32]
                    cache = source.parent / "class_masks"
                    if cache.is_symlink():
                        raise core.SegmentationError("Class mask cache must not contain symlinks.")
                    options["output_dir"] = cache / token
                exported = await core.export_label_mask(
                    source, labels=output["labels"], label_id=entry["label_id"], **options
                )
                # Expiry/ownership still applies if it changes while producing the cache.
                self.file_path(principal, task_id, source_file["name"])
                return Path(exported["path"])
            except core.SegmentationError:
                raise ServiceError("EXPORT_FAILED", "分割文件导出失败，请重试。", 503) from None
            finally:
                self.exporting[task_id] -= 1
                if not self.exporting[task_id]:
                    del self.exporting[task_id]

    def cleanup(self):
        for row in self.db.execute("SELECT id FROM tasks WHERE status='input_required'").fetchall():
            self.expire_a2a_input(row["id"])
        for row in self.db.execute(
            "SELECT id,principal FROM upload_sessions WHERE updated<?", (time.time() - 3600,)
        ).fetchall():
            if row["principal"] in self.uploading:
                continue
            if not self.db.execute("SELECT 1 FROM uploads WHERE id=?", (row["id"],)).fetchone():
                shutil.rmtree(self.root / "uploads" / row["id"], ignore_errors=True)
            with self.db:
                self.db.execute("DELETE FROM upload_sessions WHERE id=?", (row["id"],))
        for row in self.db.execute(
            "SELECT id,data FROM tasks WHERE status IN ('completed','failed','canceled')",
        ).fetchall():
            data = json.loads(row["data"])
            if row["id"] in self.exporting:
                continue
            expires_at = self.task_expiry(data)
            if not data.get("files_expired") and time.time() >= expires_at:
                shutil.rmtree(self.root / "tasks" / row["id"], ignore_errors=True)
                self.update(
                    row["id"], files_expired=True, files=[], result=None, expires_at=expires_at
                )
        for row in self.db.execute("SELECT * FROM uploads").fetchall():
            expires_at = self.upload_metadata(row)["expires_at"]
            if expires_at is not None and time.time() >= expires_at:
                shutil.rmtree(self.root / "uploads" / row["id"], ignore_errors=True)
                with self.db:
                    self.db.execute("DELETE FROM uploads WHERE id=?", (row["id"],))
        with self.db:
            self.db.execute("DELETE FROM sessions WHERE expires<?", (time.time(),))

    async def cleanup_loop(self):
        while True:
            await asyncio.sleep(900)
            self.cleanup()

"""Bounded, resumable upload chunks over the existing authenticated origin."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager

from medsegagent import core
from medsegagent.service import ServiceError, uid

CHUNK_BYTES = 8 * 1024**2
CHUNK_TIMEOUT = 180
VALIDATION_TIMEOUT = 180


class UploadSessions:
    def __init__(self, service):
        self.service = service

    def row(self, principal, upload_id):
        row = self.service.db.execute(
            "SELECT * FROM upload_sessions WHERE id=? AND principal=?", (upload_id, principal)
        ).fetchone()
        if not row or time.time() - row["updated"] >= 3600:
            raise ServiceError("UPLOAD_NOT_FOUND", "上传已过期，请重新选择影像。", 404)
        return row

    def path(self, row):
        filename = "image.nii.gz" if row["name"].endswith(".gz") else "image.nii"
        return self.service.root / "uploads" / row["id"] / filename

    def get(self, principal, upload_id):
        row = self.row(principal, upload_id)
        result = {
            "id": row["id"],
            "offset": row["received"],
            "total_bytes": row["size"],
            "chunk_bytes": CHUNK_BYTES,
        }
        if self.service.db.execute("SELECT 1 FROM uploads WHERE id=?", (upload_id,)).fetchone():
            result["upload"] = self.service.get_upload(principal, upload_id)
        return result

    def start(self, principal, body):
        filename, size, message_id = body.get("name"), body.get("size"), body.get("message_id")
        if not isinstance(message_id, str) or not 1 <= len(message_id) <= 128:
            raise ServiceError("INVALID_PARAMS", "Upload message_id must have 1–128 characters.")
        if type(size) is not int or not 0 < size <= self.service.max_upload_bytes:
            raise ServiceError("REQUEST_TOO_LARGE", "影像不得超过 500 MiB。", 413)
        existing = self.service.db.execute(
            "SELECT * FROM upload_sessions WHERE principal=? AND message_id=?",
            (principal, message_id),
        ).fetchone()
        if existing:
            if existing["name"] != filename or existing["size"] != size:
                raise ServiceError(
                    "IDEMPOTENCY_CONFLICT", "Upload message_id has different input.", 409
                )
            return self.get(principal, existing["id"])
        upload_id, destination = self.service.reserve_upload(principal, filename, size)
        try:
            now = time.time()
            with self.service.db:
                self.service.db.execute(
                    "INSERT INTO upload_sessions VALUES(?,?,?,?,?,?,?,?,?)",
                    (upload_id, principal, message_id, filename, size, 0, now, now, "{}"),
                )
            return self.get(principal, upload_id)
        except BaseException:
            shutil.rmtree(destination.parent, ignore_errors=True)
            raise
        finally:
            self.service.uploading.discard(principal)

    @contextmanager
    def busy(self, principal):
        if principal in self.service.uploading or len(self.service.uploading) >= 4:
            raise ServiceError("CAPACITY_EXCEEDED", "上传正在处理中，请稍后重试。", 429)
        self.service.uploading.add(principal)
        try:
            yield
        finally:
            self.service.uploading.discard(principal)

    async def put(self, principal, upload_id, request):
        row = self.row(principal, upload_id)
        offset = request.headers.get("upload-offset", "")
        if len(offset) > 20 or not (offset.isascii() and offset.isdigit()):
            raise ServiceError("INVALID_PARAMS", "Upload-Offset must be a nonnegative integer.")
        offset = int(offset)
        if offset > row["received"] or offset >= row["size"]:
            raise ServiceError("UPLOAD_OFFSET_CONFLICT", "上传位置不一致，请恢复上传进度。", 409)
        chunks = json.loads(row["chunks"])
        if offset < row["received"] and str(offset) not in chunks:
            raise ServiceError(
                "UPLOAD_OFFSET_CONFLICT", "Upload offset is not a chunk boundary.", 409
            )
        limit = min(CHUNK_BYTES, row["size"] - offset)
        length = request.headers.get("content-length")
        if length == "0":
            raise ServiceError("INVALID_PARAMS", "Upload chunk is empty.")
        if length and (
            len(length) > 20
            or not (length.isascii() and length.isdigit())
            or not 0 < int(length) <= limit
        ):
            raise ServiceError("REQUEST_TOO_LARGE", "Each upload chunk is limited to 8 MiB.", 413)
        path = self.path(row)
        temporary = path.parent / ("." + uid() + ".chunk")
        with self.busy(principal):
            try:
                async with asyncio.timeout(CHUNK_TIMEOUT):
                    size, checksum = 0, hashlib.sha256()
                    with temporary.open("xb") as output:
                        os.chmod(temporary, 0o600)
                        async for block in request.stream():
                            size += len(block)
                            if size > limit:
                                raise ServiceError(
                                    "REQUEST_TOO_LARGE", "Upload chunk is too large.", 413
                                )
                            checksum.update(block)
                            output.write(block)
                    if not size:
                        raise ServiceError("INVALID_PARAMS", "Upload chunk is empty.")
                    if size != limit:
                        raise ServiceError("INVALID_PARAMS", "Upload the full next chunk.")
                    chunk = {"size": size, "sha256": checksum.hexdigest()}
                    if offset < row["received"]:
                        if chunks[str(offset)] != chunk:
                            raise ServiceError(
                                "IDEMPOTENCY_CONFLICT", "Repeated chunk differs.", 409
                            )
                        return self.get(principal, upload_id)

                    def append(*, _stop_event=None):
                        if path.is_symlink():
                            raise ServiceError("INVALID_FILE", "Upload file is unavailable.")
                        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
                        with (
                            os.fdopen(os.open(path, flags, 0o600), "r+b") as target,
                            temporary.open("rb") as source,
                        ):
                            if target.seek(0, os.SEEK_END) < offset:
                                raise ServiceError(
                                    "INVALID_FILE", "Upload is incomplete; start again."
                                )
                            # A crash may leave uncommitted bytes past the durable offset.
                            target.truncate(offset)
                            target.seek(offset)
                            while block := source.read(1024 * 1024):
                                core._check_stop(_stop_event)
                                target.write(block)
                            target.flush()
                            os.fsync(target.fileno())

                    await core._validation(append)
                    chunks[str(offset)] = chunk
                    with self.service.db:
                        self.service.db.execute(
                            "UPDATE upload_sessions SET received=?,updated=?,chunks=? WHERE id=?",
                            (offset + size, time.time(), json.dumps(chunks), upload_id),
                        )
                    return self.get(principal, upload_id)
            except TimeoutError:
                raise ServiceError("UPLOAD_TIMEOUT", "上传超时，可重试当前分块。", 408) from None
            finally:
                temporary.unlink(missing_ok=True)

    async def complete(self, principal, upload_id):
        data = self.get(principal, upload_id)
        if data.get("upload"):
            return data["upload"]
        row = self.row(principal, upload_id)
        if row["received"] != row["size"]:
            raise ServiceError("UPLOAD_INCOMPLETE", "影像尚未完整上传。", 409)
        path = self.path(row)
        with self.busy(principal):
            try:
                async with asyncio.timeout(VALIDATION_TIMEOUT):
                    if path.is_symlink() or path.stat().st_size != row["size"]:
                        raise ValueError("Invalid completed upload")
                    result = await self.service.finish_upload(
                        principal, upload_id, path, row["name"], row["size"]
                    )
                    return result
            except TimeoutError:
                raise ServiceError("UPLOAD_TIMEOUT", "影像校验超时，请重试。", 408) from None
            except ValueError:
                self._discard(upload_id)
                raise ServiceError("INVALID_FILE", "影像不是完整、有效的 3D NIfTI。") from None

    def _discard(self, upload_id):
        with self.service.db:
            self.service.db.execute("DELETE FROM upload_sessions WHERE id=?", (upload_id,))
        shutil.rmtree(self.service.root / "uploads" / upload_id, ignore_errors=True)

    def delete(self, principal, upload_id):
        self.row(principal, upload_id)
        with self.busy(principal):
            if self.service.db.execute("SELECT 1 FROM uploads WHERE id=?", (upload_id,)).fetchone():
                raise ServiceError("UPLOAD_COMPLETE", "影像已完成上传，请使用影像删除接口。", 409)
            self._discard(upload_id)
        return {"deleted": True}

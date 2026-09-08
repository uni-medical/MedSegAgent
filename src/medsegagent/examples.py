"""Operator-installed, licensed example volumes; each user gets an owned upload."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from medsegagent import core
from medsegagent.service import ServiceError

OPEN_TIMEOUT_SECONDS = 180


class Examples:
    def __init__(self, service):
        self.service = service
        self.root = Path(os.environ.get("MEDSEGAGENT_EXAMPLE_DIR", service.root / "examples"))
        self.cases = {}
        manifest = self.root / "manifest.json"
        if manifest.is_file():
            for case in json.loads(manifest.read_text())["cases"]:
                if not re.fullmatch(r"[a-z0-9-]{1,48}", case["id"]):
                    raise ValueError("Invalid installed example ID.")
                if case["id"] in self.cases:
                    raise ValueError("Duplicate installed example ID.")
                self.cases[case["id"]] = case

    def case(self, example_id):
        if example_id not in self.cases:
            raise ServiceError("EXAMPLE_NOT_FOUND", "示例不存在。", 404)
        return self.cases[example_id]

    def path(self, example_id, field):
        filename = self.case(example_id)[field]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ServiceError("EXAMPLE_UNAVAILABLE", "示例暂不可用。", 503)
        path = self.root / filename
        if path.is_symlink() or not path.is_file():
            raise ServiceError("EXAMPLE_UNAVAILABLE", "示例暂不可用。", 503)
        return path

    def catalog(self):
        return [
            {
                **{
                    key: case[key]
                    for key in (
                        "id",
                        "title",
                        "modality",
                        "description",
                        "size_bytes",
                        "prompts",
                        "attribution",
                    )
                },
                "preview_url": f"/api/examples/{case['id']}/preview",
            }
            for case in self.cases.values()
        ]

    async def open(self, principal, example_id):
        case = self.case(example_id)
        source = self.path(example_id, "filename")
        if not 0 < case["size_bytes"] <= self.service.max_upload_bytes:
            raise ServiceError("EXAMPLE_UNAVAILABLE", "示例大小超出当前限制。", 503)
        # Repeated browsing reuses this identity's still-available copy, never another user's.
        rows = self.service.db.execute(
            "SELECT * FROM uploads WHERE principal=? ORDER BY created DESC", (principal,)
        ).fetchall()
        for row in rows:
            data = self.service.upload_metadata(row)
            if (
                data.get("example_id") == example_id
                and data["available"]
                and (
                    case.get("modality") not in {"CT", "MR"}
                    or data.get("source_modality") == case["modality"]
                )
            ):
                return data
        # Older copies remain usable by their tasks; refresh missing or changed declarations
        # through the normal checksum-verified copy instead of relabeling an existing file.
        filename = example_id + ".nii.gz"
        upload_id, destination = self.service.reserve_upload(
            principal, filename, case["size_bytes"]
        )
        success = False
        try:

            def copy(*, _stop_event=None):
                fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
                size = 0
                checksum = hashlib.sha256()
                limit = min(case["size_bytes"], self.service.max_upload_bytes)
                with os.fdopen(fd, "rb") as incoming, destination.open("xb") as outgoing:
                    os.chmod(destination, 0o600)
                    while block := incoming.read(1024 * 1024):
                        core._check_stop(_stop_event)
                        size += len(block)
                        if size > limit:
                            raise ServiceError("EXAMPLE_UNAVAILABLE", "示例大小超出当前限制。", 503)
                        checksum.update(block)
                        outgoing.write(block)
                if size != case["size_bytes"] or checksum.hexdigest() != case["sha256"]:
                    raise ServiceError("EXAMPLE_UNAVAILABLE", "示例校验失败，请稍后重试。", 503)

            # The shared I/O helper drains canceled readers before files are reclaimed.
            async with asyncio.timeout(OPEN_TIMEOUT_SECONDS):
                await core._validation(copy)
                data = await self.service.finish_upload(
                    principal, upload_id, destination, filename, destination.stat().st_size
                )
            data["example_id"] = example_id
            # Preserve the installed source declaration after checksum verification.
            # A later task need not infer what is already known about this example.
            if case.get("modality") in {"CT", "MR"}:
                data["source_modality"] = case["modality"]
            with self.service.db:
                self.service.db.execute(
                    "UPDATE uploads SET data=? WHERE id=?", (json.dumps(data), upload_id)
                )
            success = True
            return data
        except TimeoutError:
            raise ServiceError("EXAMPLE_TIMEOUT", "示例加载超时，请重试。", 408) from None
        finally:
            self.service.uploading.discard(principal)
            if not success:
                shutil.rmtree(destination.parent, ignore_errors=True)

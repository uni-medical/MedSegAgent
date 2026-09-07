"""Independent regressions for durable input retention and active-task safety."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time

import pytest

from medsegagent import core
from medsegagent.service import Service, ServiceError


def registered_upload(service, principal="alice"):
    upload_id, path = service.reserve_upload(principal, "synthetic.nii")
    path.write_bytes(b"synthetic-private-image")
    service.uploading.discard(principal)
    now = time.time()
    metadata = {"id": upload_id, "created_at": now, "expires_at": now + service.retention_seconds}
    with service.db:
        service.db.execute(
            "INSERT INTO uploads VALUES(?,?,?,?,?,?)",
            (upload_id, principal, str(path), path.stat().st_size, now, json.dumps(metadata)),
        )
    return upload_id, path


def test_active_input_protection_is_not_limited_to_100_recent_tasks(tmp_path, monkeypatch):
    service = Service(tmp_path, "https://medseg.example.org")
    monkeypatch.setattr(service, "launch", lambda task_id: None)
    upload_id, path = registered_upload(service)

    async def scenario():
        first = await service.submit("alice", upload_id, "liver", "CT", "active-first")
        for number in range(101):
            recent = await service.submit("alice", upload_id, "liver", "CT", f"recent-{number}")
            await service.cancel("alice", recent["id"])
        assert service.get("alice", first["id"])["status"] == "queued"
        assert all(task["id"] != first["id"] for task in service.list("alice"))
        with pytest.raises(ServiceError) as error:
            service.delete_upload("alice", upload_id)
        assert error.value.code == "TASK_ACTIVE"
        assert path.is_file()

    try:
        asyncio.run(scenario())
    finally:
        service.db.close()


def test_crashed_unregistered_upload_is_removed_on_start_after_retention(tmp_path):
    service = Service(tmp_path, "https://medseg.example.org")
    _, path = service.reserve_upload("alice", "synthetic.nii")
    path.write_bytes(b"synthetic-private-image-partial-upload")
    expired = time.time() - service.retention_seconds - 3600
    os.utime(path, (expired, expired))
    os.utime(path.parent, (expired, expired))
    # Equivalent persisted state to a process crash before finish_upload inserts a row.
    service.db.close()

    restarted = Service(tmp_path, "https://medseg.example.org")

    async def scenario():
        await restarted.start()
        try:
            assert not path.exists(), "Expired raw image outside SQLite retention survives restart"
        finally:
            await restarted.close()

    asyncio.run(scenario())


def test_cancelled_upload_validation_waits_for_its_reader(tmp_path, monkeypatch):
    """The upload route must not release quota/delete input while a reader survives."""
    service = Service(tmp_path, "https://medseg.example.org")
    upload_id, path = service.reserve_upload("alice", "synthetic.nii")
    path.write_bytes(b"synthetic-image")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def validation(*_args, **_kwargs):
        started.set()
        try:
            release.wait(5)
            return path
        finally:
            finished.set()

    monkeypatch.setattr(core, "validate_input", validation)

    async def scenario():
        worker = asyncio.create_task(
            service.finish_upload("alice", upload_id, path, "synthetic.nii", path.stat().st_size)
        )
        try:
            assert await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=2)
            worker.cancel()
            await asyncio.sleep(0.02)
            assert not worker.done(), (
                "Cancelled upload escaped while its image-validation reader lives"
            )
        finally:
            release.set()
            await asyncio.gather(worker, return_exceptions=True)
        assert finished.is_set()

    try:
        asyncio.run(scenario())
    finally:
        service.db.close()

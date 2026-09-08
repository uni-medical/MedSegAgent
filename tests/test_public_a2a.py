"""Anonymous A2A uses a separate namespace and opaque IDs, never a public listing."""

import asyncio
import json
import time

import nibabel as nib
import numpy as np
import pytest
from starlette.testclient import TestClient
from test_execution import Backend

from medsegagent import agent, core
from medsegagent.auth import SESSION_COOKIE, AuthStore
from medsegagent.service import MAX_UPLOAD_BYTES, PUBLIC_A2A_PRINCIPAL, Service, ServiceError
from medsegagent.upload_sessions import UploadSessions
from medsegagent.web import create_app

PUBLIC = PUBLIC_A2A_PRINCIPAL
ORIGIN = "http://localhost"
VERSION = {"A2A-Version": "1.0"}


def volume():
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm")
    return image.to_bytes()


async def upload(service, principal):
    data = volume()
    upload_id, path = service.reserve_upload(principal, "synthetic.nii", len(data))
    path.write_bytes(data)
    try:
        await service.finish_upload(principal, upload_id, path, "synthetic.nii", len(data))
    finally:
        service.uploading.discard(principal)
    return upload_id


def message(upload_id, message_id="public-request", context_id=None):
    request = {
        "message": {
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [
                {"text": "Segment CT liver"},
                {"data": {"upload_id": upload_id, "modality": "CT"}},
            ],
        },
        "configuration": {"returnImmediately": False},
    }
    if context_id:
        request["message"]["contextId"] = context_id
    return request


def test_public_namespace_never_exposes_old_accounts_or_contexts_and_keeps_idempotency(
    tmp_path, monkeypatch
):
    async def run():
        service = Service(tmp_path, ORIGIN)
        monkeypatch.setattr(service, "launch", lambda task_id: None)
        try:
            old_upload = await upload(service, "legacy-token-user")
            private_upload = await upload(service, "guest:private")
            public_upload = await upload(service, PUBLIC)
            other_upload = await upload(service, PUBLIC)
            old = await service.submit("legacy-token-user", old_upload, "CT liver", "CT", "same-id")
            private = await service.submit(
                "guest:private", private_upload, "CT liver", "CT", "same-id"
            )
            first = await service.submit(PUBLIC, public_upload, "CT liver", "CT", "same-id")
            assert len({old["id"], private["id"], first["id"]}) == 3
            assert (await service.submit(PUBLIC, public_upload, "CT liver", "CT", "same-id"))[
                "id"
            ] == first["id"]
            for foreign, foreign_upload in [(old, old_upload), (private, private_upload)]:
                with pytest.raises(ServiceError) as error:
                    service.get(PUBLIC, foreign["id"])
                assert error.value.status_code == 404
                with pytest.raises(ServiceError):
                    service.upload_path(PUBLIC, foreign_upload)
                with pytest.raises(ServiceError):
                    await service.submit(
                        PUBLIC,
                        public_upload,
                        "CT liver",
                        "CT",
                        "foreign-context",
                        foreign["context_id"],
                    )
            for input_id, context in [(other_upload, None), (public_upload, first["context_id"])]:
                with pytest.raises(ServiceError) as error:
                    await service.submit(PUBLIC, input_id, "CT liver", "CT", "same-id", context)
                assert error.value.code == "IDEMPOTENCY_CONFLICT"
            continued = await service.submit(
                PUBLIC, public_upload, "CT liver", "CT", "next-message", first["context_id"]
            )
            assert continued["context_id"] == first["context_id"]
            with pytest.raises(ServiceError) as error:
                service.list(PUBLIC)
            assert error.value.code == "LIST_UNSUPPORTED"
            assert len(service.list("legacy-token-user")) == 1
        finally:
            await service.close()

    asyncio.run(run())


def test_public_chunk_reservations_keep_existing_namespace_quota_and_ownership(tmp_path):
    async def run():
        service = Service(tmp_path, ORIGIN)
        sessions = UploadSessions(service)
        try:
            first = sessions.start(
                PUBLIC, {"name": "synthetic.nii", "size": MAX_UPLOAD_BYTES, "message_id": "first"}
            )
            assert (
                sessions.start(
                    PUBLIC,
                    {"name": "synthetic.nii", "size": MAX_UPLOAD_BYTES, "message_id": "first"},
                )
                == first
            )
            with pytest.raises(ServiceError):
                sessions.get("legacy-token-user", first["id"])
            for index in range(3):
                sessions.start(
                    PUBLIC,
                    {"name": "synthetic.nii", "size": MAX_UPLOAD_BYTES, "message_id": str(index)},
                )
            with pytest.raises(ServiceError) as error:
                sessions.start(
                    PUBLIC,
                    {"name": "synthetic.nii", "size": MAX_UPLOAD_BYTES, "message_id": "over-quota"},
                )
            assert error.value.code == "QUOTA_EXHAUSTED"
            private = sessions.start(
                "guest:private", {"name": "synthetic.nii", "size": 1024, "message_id": "first"}
            )
            with pytest.raises(ServiceError):
                sessions.get(PUBLIC, private["id"])
            assert service.uploading == set()
        finally:
            await service.close()

    asyncio.run(run())


def test_anonymous_upload_segmentation_files_and_session_isolation(tmp_path, monkeypatch):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)

    async def run(text, modality, execution, **kwargs):
        assert (await execution.call("segment", {"targets": ["liver"]}))["ok"]
        return {"status": "completed", "summary": "Liver complete", "unresolved": []}

    monkeypatch.setattr(agent, "run_agent", run)
    application = create_app(tmp_path, ORIGIN)
    with TestClient(application) as client:
        public_upload = client.post(
            "/a2a/uploads", content=volume(), headers={"X-Filename": "synthetic.nii"}
        )
        assert public_upload.status_code == 201, public_upload.text
        upload_id = public_upload.json()["id"]
        assert not client.cookies
        payload = message(upload_id)
        result = client.post("/a2a/v1/message:send", json=payload, headers=VERSION)
        assert result.status_code == 200, result.text
        task = result.json()["task"]
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert (
            client.post(
                "/a2a/v1/message:send",
                json=payload,
                headers={**VERSION, "Authorization": "Bearer obsolete"},
            ).json()["task"]["id"]
            == task["id"]
        )
        assert len(backend.calls) == 1
        assert client.get("/a2a/v1/tasks/" + task["id"], headers=VERSION).status_code == 200
        assert client.get("/a2a/v1/tasks", headers=VERSION).status_code == 400
        assert client.get("/a2a/tasks").status_code in {404, 405}
        assert client.get("/api/tasks").status_code == 401
        assert client.get("/a2a/uploads/" + upload_id + "/file").content == volume()
        files = task["metadata"]["segmentation"]["files"]
        assert all("/a2a/tasks/" + task["id"] + "/files/" in file["url"] for file in files)
        for file in files:
            assert client.get(file["url"]).status_code == 200
        store = AuthStore(application.state.service.db, ORIGIN)
        private_session = store.create_guest()
        private_headers = {"Cookie": f"{SESSION_COOKIE}={private_session.token}", "Origin": ORIGIN}
        private_upload = client.post(
            "/api/uploads",
            content=volume(),
            headers={**private_headers, "X-Filename": "synthetic.nii"},
        ).json()["id"]
        private = client.post(
            "/a2a/v1/message:send",
            json=message(private_upload, "private-request"),
            headers={**private_headers, **VERSION},
        )
        assert private.status_code == 200, private.text
        private_task = private.json()["task"]
        assert client.get("/a2a/v1/tasks/" + private_task["id"], headers=VERSION).status_code == 404
        assert client.get("/a2a/uploads/" + private_upload + "/file").status_code == 404
        assert (
            client.get(
                "/a2a/tasks/" + private_task["id"] + "/files/segmentation.nii.gz"
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/a2a/v1/tasks/" + task["id"], headers={**private_headers, **VERSION}
            ).status_code
            == 404
        )
        for file in private_task["metadata"]["segmentation"]["files"]:
            assert "/api/tasks/" in file["url"]
            assert client.get(file["url"]).status_code == 401
            assert client.get(file["url"], headers=private_headers).status_code == 200
        application.state.service.update(task["id"], expires_at=time.time() - 1)
        assert client.get(files[0]["url"]).status_code == 404
    assert "legacy" not in json.dumps(task)


@pytest.mark.parametrize("table", ["tasks", "uploads", "upload_sessions", "sessions"])
def test_reserved_public_namespace_never_adopts_legacy_records(tmp_path, monkeypatch, table):
    import sqlite3

    async def prepare():
        service = Service(tmp_path, ORIGIN)
        monkeypatch.setattr(service, "launch", lambda task_id: None)
        if table in {"tasks", "uploads"}:
            upload_id = await upload(service, PUBLIC)
            if table == "tasks":
                await service.submit(PUBLIC, upload_id, "CT liver", "CT", "legacy-record")
                service.db.execute("DELETE FROM uploads")
        elif table == "upload_sessions":
            UploadSessions(service).start(
                PUBLIC, {"name": "legacy.nii", "size": 1024, "message_id": "legacy"}
            )
        else:
            service.db.execute(
                "INSERT INTO sessions VALUES(?,?,?)", ("legacy-hash", PUBLIC, time.time() + 60)
            )
        # Simulate a database predating public A2A with that arbitrary token username.
        service.db.execute("DELETE FROM service_metadata WHERE key='public_a2a_namespace_v1'")
        service.db.commit()
        before = service.db.execute(f"SELECT * FROM {table}").fetchall()
        await service.close()
        return [tuple(row) for row in before]

    before = asyncio.run(prepare())
    with pytest.raises(ValueError, match="collides with existing private"):
        Service(tmp_path, ORIGIN)
    with sqlite3.connect(tmp_path / "state.sqlite3") as db:
        assert db.execute(f"SELECT * FROM {table}").fetchall() == before
        assert not db.execute(
            "SELECT 1 FROM service_metadata WHERE key='public_a2a_namespace_v1'"
        ).fetchone()


def test_public_namespace_marker_preserves_public_objects_across_restart(tmp_path, monkeypatch):
    async def prepare():
        service = Service(tmp_path, ORIGIN)
        monkeypatch.setattr(service, "launch", lambda task_id: None)
        upload_id = await upload(service, PUBLIC)
        task = await service.submit(PUBLIC, upload_id, "CT liver", "CT", "public-marker")
        service.update(task["id"], status="canceled")
        await service.close()
        return upload_id, task["id"]

    upload_id, task_id = asyncio.run(prepare())
    service = Service(tmp_path, ORIGIN)
    try:
        assert service.get(PUBLIC, task_id)["status"] == "canceled"
        assert service.get_upload(PUBLIC, upload_id)["available"]
        assert (
            service.db.execute(
                "SELECT value FROM service_metadata WHERE key='public_a2a_namespace_v1'"
            ).fetchone()[0]
            == PUBLIC
        )
    finally:
        asyncio.run(service.close())


def test_anonymous_chunks_resume_across_clients_and_do_not_open_private_sessions(
    tmp_path, monkeypatch
):
    from medsegagent import upload_sessions

    monkeypatch.setattr(upload_sessions, "CHUNK_BYTES", 512)
    data = volume()
    root = tmp_path / "service"
    application = create_app(root, ORIGIN)
    with TestClient(application) as client:
        body = {"name": "synthetic.nii", "size": len(data), "message_id": "anonymous-chunks"}
        started = client.post("/a2a/upload-sessions", json=body)
        assert started.status_code == 201, started.text
        upload_id = started.json()["id"]
        chunk = client.put(
            "/a2a/upload-sessions/" + upload_id, content=data[:512], headers={"Upload-Offset": "0"}
        )
        assert chunk.status_code == 200 and chunk.json()["offset"] == 512
        assert client.post("/a2a/upload-sessions", json=body).json()["id"] == upload_id
        changed = client.post("/a2a/upload-sessions", json={**body, "size": len(data) + 1})
        assert changed.status_code == 409
        session = AuthStore(application.state.service.db, ORIGIN).create_guest()
        headers = {"Cookie": f"{SESSION_COOKIE}={session.token}", "Origin": ORIGIN}
        private = client.post(
            "/api/upload-sessions", json={**body, "message_id": "private-chunks"}, headers=headers
        ).json()["id"]
        assert client.get("/a2a/upload-sessions/" + private).status_code == 404
        assert client.get("/a2a/upload-sessions/" + upload_id, headers=headers).status_code == 404
    with TestClient(create_app(root, ORIGIN)) as client:
        assert client.get("/a2a/upload-sessions/" + upload_id).json()["offset"] == 512
        chunk = client.put(
            "/a2a/upload-sessions/" + upload_id,
            content=data[512:],
            headers={"Upload-Offset": "512"},
        )
        assert chunk.status_code == 200 and chunk.json()["offset"] == len(data)
        complete = client.post("/a2a/upload-sessions/" + upload_id + "/complete")
        assert complete.status_code == 200, complete.text
        assert client.get("/a2a/uploads/" + upload_id + "/file").content == data
        assert client.delete("/a2a/uploads/" + upload_id).status_code == 200
        assert client.get("/a2a/uploads/" + upload_id + "/file").status_code == 404

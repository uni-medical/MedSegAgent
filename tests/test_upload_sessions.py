"""Chunk commits survive ACK loss/restart and preserve upload ownership and limits."""

import hashlib
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from auth_helpers import ALICE, BOB
from auth_helpers import create_test_app as create_app
from starlette.testclient import TestClient

from medsegagent import upload_sessions


@pytest.fixture
def payload():
    return nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.int16), np.eye(4)).to_bytes()


def start(client, data, name="test.nii", message="upload-one"):
    response = client.post(
        "/api/upload-sessions",
        headers=ALICE,
        json={"name": name, "size": len(data), "message_id": message},
    )
    assert response.status_code == 201, response.text
    return response.json()


def put(client, upload_id, offset, data, headers=ALICE):
    return client.put(
        "/api/upload-sessions/" + upload_id,
        headers={**headers, "Upload-Offset": str(offset)},
        content=data,
    )


def test_chunked_upload_replay_ownership_restart_and_full_validation(
    tmp_path, payload, monkeypatch
):
    monkeypatch.setattr(upload_sessions, "CHUNK_BYTES", 1024)
    with TestClient(create_app(tmp_path, "http://localhost")) as c:
        assert c.post("/api/upload-sessions", json={}).status_code == 401
        session = start(c, payload)
        assert start(c, payload)["id"] == session["id"]
        sid = session["id"]
        assert c.get("/api/upload-sessions/" + sid, headers=BOB).status_code == 404
        assert put(c, sid, 0, payload[:1024], BOB).status_code == 404
        assert put(c, sid, 10, b"x").status_code == 409
        assert (
            c.put(
                "/api/upload-sessions/" + sid,
                headers={**ALICE, "Upload-Offset": b"\xb2"},
                content=payload[:1024],
            ).status_code
            == 400
        )
        assert put(c, sid, 0, payload[:1024]).json()["offset"] == 1024
        assert put(c, sid, 0, payload[:1024]).json()["offset"] == 1024
        assert put(c, sid, 0, b"x" * 1024).status_code == 409
        assert c.post(f"/api/upload-sessions/{sid}/complete", headers=ALICE).status_code == 409
        path = tmp_path / "uploads" / sid / "image.nii"
        # Simulate an interrupted append whose offset was not committed to SQLite.
        with path.open("ab") as f:
            f.write(b"uncommitted")
        (path.parent / ".interrupted.chunk").write_bytes(b"temporary")
        assert c.get("/api/tasks", headers=ALICE).json() == []
    with TestClient(create_app(tmp_path, "http://localhost")) as c:
        assert not (path.parent / ".interrupted.chunk").exists()
        assert path.stat().st_size == 1024
        assert c.get("/api/upload-sessions/" + sid, headers=ALICE).json()["offset"] == 1024
        for offset in range(1024, len(payload), 1024):
            assert put(c, sid, offset, payload[offset : offset + 1024]).status_code == 200
        done = c.post(f"/api/upload-sessions/{sid}/complete", headers=ALICE)
        assert done.status_code == 200, done.text
        assert done.json()["shape"] == [8, 8, 8]
        assert c.post(f"/api/upload-sessions/{sid}/complete", headers=ALICE).json()["id"] == sid
        assert c.get("/api/upload-sessions/" + sid, headers=ALICE).json()["upload"]["id"] == sid
        # A client losing the completion ACK cannot delete a finished source via session cleanup.
        assert c.delete("/api/upload-sessions/" + sid, headers=ALICE).status_code == 409
        source = c.get(f"/api/uploads/{sid}/file", headers=ALICE)
        assert hashlib.sha256(source.content).digest() == hashlib.sha256(payload).digest()
        assert c.get(f"/api/uploads/{sid}/file", headers=BOB).status_code == 404
        assert c.get("/api/tasks", headers=ALICE).json() == []
        assert os.stat(path).st_mode & 0o777 == 0o600
        assert c.delete(f"/api/uploads/{sid}", headers=ALICE).status_code == 200
        assert c.post(f"/api/upload-sessions/{sid}/complete", headers=ALICE).status_code == 404
        assert (
            c.app.state.service.db.execute("SELECT COUNT(*) FROM upload_sessions").fetchone()[0]
            == 0
        )


def test_failed_directory_creation_does_not_keep_identity_busy(tmp_path, payload, monkeypatch):
    with TestClient(create_app(tmp_path, "http://localhost")) as c:
        original = Path.mkdir

        def full_disk(path, *args, **kwargs):
            if path.parent == tmp_path / "uploads":
                raise OSError(28, "No space left on device")
            return original(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "mkdir", full_disk)
            response = c.post(
                "/api/upload-sessions",
                headers=ALICE,
                json={"name": "test.nii", "size": len(payload), "message_id": "upload-one"},
            )
            assert response.status_code == 500
        assert not c.app.state.service.uploading
        assert start(c, payload)["offset"] == 0


@pytest.mark.parametrize("name", ["../private.nii", "/tmp/private.nii", "wrong.zip"])
def test_chunked_names_are_validated_at_reservation(tmp_path, name):
    with TestClient(create_app(tmp_path, "http://localhost")) as c:
        r = c.post(
            "/api/upload-sessions",
            headers=ALICE,
            json={"name": name, "size": 50, "message_id": "bad-name"},
        )
        assert r.status_code == 400
        assert not list((tmp_path / "uploads").iterdir())


def test_upload_500_mib_limit_and_reserved_quota(tmp_path, payload):
    with TestClient(create_app(tmp_path, "http://localhost")) as c:
        config = c.get("/api/config", headers=ALICE).json()
        maximum = 500 * 1024**2
        assert config["max_upload_bytes"] == maximum
        assert config["single_upload_bytes"] == 90 * 1024**2
        for index in range(4):
            r = c.post(
                "/api/upload-sessions",
                headers=ALICE,
                json={"name": "big.nii", "size": maximum, "message_id": str(index)},
            )
            assert r.status_code == 201
        assert (
            c.post(
                "/api/upload-sessions",
                headers=ALICE,
                json={"name": "big.nii", "size": maximum, "message_id": "quota"},
            ).status_code
            == 429
        )
        assert (
            c.post(
                "/api/upload-sessions",
                headers=BOB,
                json={"name": "big.nii", "size": maximum + 1, "message_id": "size"},
            ).status_code
            == 413
        )
        assert (
            c.post(
                "/api/uploads",
                headers={**BOB, "X-Filename": "big.nii", "Content-Length": str(91 * 1024**2)},
                content=b"x",
            ).status_code
            == 413
        )


def test_bad_chunks_and_corrupt_completed_file_do_not_publish(tmp_path):
    with TestClient(create_app(tmp_path, "http://localhost")) as c:
        sid = start(c, b"bad")["id"]
        assert put(c, sid, 0, b"oversize").status_code == 413
        assert put(c, sid, 0, b"").status_code == 400
        assert put(c, sid, 0, b"bad").status_code == 200
        result = c.post(f"/api/upload-sessions/{sid}/complete", headers=ALICE)
        assert result.status_code == 400 and result.json()["error"]["code"] == "INVALID_FILE"
        assert c.get(f"/api/uploads/{sid}/file", headers=ALICE).status_code == 404
        assert not (tmp_path / "uploads" / sid).exists()
        assert not c.app.state.service.uploading


def test_upload_cleanup_reclaims_stale_sessions(tmp_path, payload):
    with TestClient(create_app(tmp_path, "http://localhost")) as c:
        sid = start(c, payload)["id"]
        assert put(c, sid, 0, payload).status_code == 200
        db = c.app.state.service.db
        with db:
            db.execute("UPDATE upload_sessions SET updated=? WHERE id=?", (time.time() - 3601, sid))
        c.app.state.service.cleanup()
        assert c.get("/api/upload-sessions/" + sid, headers=ALICE).status_code == 404
        assert not (tmp_path / "uploads" / sid).exists()


def test_session_delete_and_conflicting_request_key(tmp_path, payload):
    with TestClient(create_app(tmp_path, "http://localhost")) as c:
        sid = start(c, payload)["id"]
        r = c.post(
            "/api/upload-sessions",
            headers=ALICE,
            json={"name": "different.nii", "size": len(payload), "message_id": "upload-one"},
        )
        assert r.status_code == 409
        assert c.delete("/api/upload-sessions/" + sid, headers=BOB).status_code == 404
        assert c.delete("/api/upload-sessions/" + sid, headers=ALICE).status_code == 200
        assert not (tmp_path / "uploads" / sid).exists()


def test_tiny_nonfinal_chunks_cannot_amplify_durable_metadata(tmp_path, payload):
    with TestClient(create_app(tmp_path, "http://localhost")) as c:
        sid = start(c, payload)["id"]
        assert put(c, sid, 0, b"x").status_code == 400
        row = c.app.state.service.db.execute(
            "SELECT received,chunks FROM upload_sessions WHERE id=?", (sid,)
        ).fetchone()
        assert row["received"] == 0 and row["chunks"] == "{}"
        assert not list((tmp_path / "uploads" / sid).iterdir())

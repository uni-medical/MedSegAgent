from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import nibabel as nib
import numpy as np
import pytest
from starlette.testclient import TestClient

from medsegagent import agent, core
from medsegagent.service import Service
from medsegagent.web import create_app

TOKENS = {"alice": "a" * 40, "bob": "b" * 40}
ALICE = {"Authorization": "Bearer " + TOKENS["alice"]}
BOB = {"Authorization": "Bearer " + TOKENS["bob"]}


def nifti_bytes(value=0):
    image = nib.Nifti1Image(np.full((8, 8, 8), value, dtype=np.float32), np.eye(4))
    return image.to_bytes()


def client(tmp_path):
    return TestClient(create_app(tmp_path, "http://localhost", TOKENS))


@pytest.mark.parametrize("bad", [[], [" "], ["liver", ""], ["bogus"], None])
def test_model_arguments_fail_closed(monkeypatch, bad):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "deepseek-v4-flash"
        assert "input_path" not in request.content.decode()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "segment_ct",
                                        "arguments": json.dumps({"targets": bad}),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    with pytest.raises(agent.RoutingError):
        asyncio.run(agent.select_tool("分割肝脏", "CT", transport=httpx.MockTransport(handler)))


def test_provider_projection_and_wrong_modality(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "private-key")

    def handler(request):
        body = json.loads(request.content)
        assert len(body["messages"]) == 2
        assert json.loads(body["messages"][1]["content"]) == {"modality": "CT", "request": "liver"}
        assert all(
            set(t["function"]["parameters"]["properties"]) == {"targets"} for t in body["tools"]
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "segment_mr",
                                        "arguments": '{"targets":["liver"]}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    with pytest.raises(agent.RoutingError, match="modality"):
        asyncio.run(agent.select_tool("liver", "CT", transport=httpx.MockTransport(handler)))


def test_auth_upload_bounds_and_owner_isolation(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/api/tasks").status_code == 401
        assert c.get("/.well-known/agent-card.json").status_code == 200
        assert c.post("/api/uploads", content=b"bad").status_code == 401
        assert (
            c.post(
                "/api/uploads", headers={**ALICE, "Content-Length": str(91 * 1024**2)}
            ).status_code
            == 413
        )
        assert (
            c.post(
                "/api/uploads", headers={**ALICE, "X-Filename": "../x.nii"}, content=b"bad"
            ).status_code
            == 400
        )
        assert (
            c.post(
                "/api/uploads", headers={**ALICE, "X-Filename": "bad.nii"}, content=b"bad"
            ).status_code
            == 400
        )
        assert (
            c.post(
                "/api/uploads",
                headers={**ALICE, "X-Filename": "nan.nii"},
                content=nifti_bytes(float("nan")),
            ).status_code
            == 400
        )
        response = c.post(
            "/api/uploads", headers={**ALICE, "X-Filename": "ct.nii"}, content=nifti_bytes()
        )
        assert response.status_code == 201, response.text
        upload = response.json()
        path = f"/api/uploads/{upload['id']}/file"
        assert c.get(path, headers=ALICE).content == nifti_bytes()
        assert c.get(path, headers=BOB).status_code == 404
        assert c.get(path).status_code == 401
        assert "path" not in upload
        assert len(list((tmp_path / "uploads").iterdir())) == 1


def test_https_cookie_and_csrf(tmp_path):
    app = create_app(tmp_path, "https://testserver", TOKENS)
    with TestClient(app, base_url="https://testserver") as c:
        bad = c.post(
            "/api/session",
            json={"token": TOKENS["alice"]},
            headers={"Origin": "https://evil.invalid"},
        )
        assert bad.status_code == 403
        response = c.post("/api/session", json={"token": TOKENS["alice"]})
        assert response.status_code == 200
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie and "secure" in cookie and "samesite=strict" in cookie
        assert c.get("/api/tasks").status_code == 200
        assert (
            c.post("/api/tasks", json={}, headers={"Origin": "https://evil.invalid"}).status_code
            == 403
        )
        assert c.post("/a2a/v1/message:send", json={}).status_code == 401
        assert c.delete("/api/session").status_code == 200
        assert c.get("/api/tasks").status_code == 401


def test_task_idempotency_cancel_and_restart(tmp_path, monkeypatch):
    async def slow_select(*args, **kwargs):
        await asyncio.sleep(60)

    monkeypatch.setattr(agent, "select_tool", slow_select)
    with client(tmp_path) as c:
        upload = c.post(
            "/api/uploads", headers={**ALICE, "X-Filename": "ct.nii"}, content=nifti_bytes()
        ).json()
        body = {"upload_id": upload["id"], "text": "liver", "modality": "CT", "message_id": "one"}
        first = c.post("/api/tasks", headers=ALICE, json=body)
        assert first.status_code == 202, first.text
        task_id = first.json()["id"]
        replay = c.post("/api/tasks", headers=ALICE, json=body)
        assert replay.json()["id"] == task_id
        assert (
            c.post("/api/tasks", headers=ALICE, json={**body, "text": "spleen"}).status_code == 409
        )
        assert c.get(f"/api/tasks/{task_id}", headers=BOB).status_code == 404
        assert c.post(f"/api/tasks/{task_id}/cancel", headers=BOB).status_code == 404
        assert c.post(f"/api/tasks/{task_id}/cancel", headers=ALICE).json()["status"] == "canceled"
    with client(tmp_path) as c:
        assert c.get(f"/api/tasks/{task_id}", headers=ALICE).json()["status"] == "canceled"
        assert c.post("/api/tasks", headers=ALICE, json=body).json()["id"] == task_id


def test_completed_result_survives_restart_and_files_remain_private(tmp_path, monkeypatch):
    async def select(*args, **kwargs):
        return agent.Selection("segment_ct", "total", ["liver"])

    async def segment(**kwargs):
        parent = Path(kwargs["output_dir"]) / "unique-run"
        parent.mkdir(parents=True)
        path = parent / "segmentation.nii.gz"
        nib.save(nib.Nifti1Image(np.full((8, 8, 8), 5, dtype=np.uint8), np.eye(4)), path)
        return {
            "segmentation_path": str(path),
            "task": "total",
            "device": "cpu",
            "targets": ["liver"],
            "labels": [{"id": 5, "name": "liver", "voxels": 512}],
        }

    monkeypatch.setattr(agent, "select_tool", select)
    monkeypatch.setattr(core, "segment", segment)
    with client(tmp_path) as c:
        upload = c.post(
            "/api/uploads", headers={**ALICE, "X-Filename": "ct.nii"}, content=nifti_bytes()
        ).json()
        task_id = c.post(
            "/api/tasks",
            headers=ALICE,
            json={
                "upload_id": upload["id"],
                "text": "liver",
                "modality": "CT",
                "message_id": "success",
            },
        ).json()["id"]
        for _ in range(50):
            row = c.get(f"/api/tasks/{task_id}", headers=ALICE).json()
            if row["status"] == "completed":
                break
        assert row["status"] == "completed", row
        assert str(tmp_path) not in json.dumps(row)
    with client(tmp_path) as c:
        assert c.get(f"/api/tasks/{task_id}", headers=ALICE).json()["status"] == "completed"
        path = f"/api/tasks/{task_id}/files/segmentation.nii.gz"
        assert c.get(path, headers=ALICE).status_code == 200
        assert c.get(path, headers=BOB).status_code == 404
        assert c.get(path).status_code == 401
        assert c.get(f"/api/tasks/{task_id}/files/process.log", headers=ALICE).status_code == 404
        assert c.get("/runtime/state.sqlite3", headers=ALICE).status_code == 404


def test_single_service_lock(tmp_path):
    async def exercise():
        first, second = Service(tmp_path, "http://localhost"), Service(tmp_path, "http://localhost")
        await first.start()
        try:
            with pytest.raises(RuntimeError, match="Another"):
                await second.start()
        finally:
            await first.close()
            await second.close()

    asyncio.run(exercise())

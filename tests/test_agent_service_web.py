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


def routing_transport(monkeypatch, tool, arguments, *, check_request=None):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    def handler(request):
        if check_request:
            check_request(json.loads(request.content))
        calls = [{"function": {"name": tool, "arguments": json.dumps(arguments)}}] if tool else []
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": calls}}]})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    ("text", "tool", "modality"),
    [
        ("请分割这份CT中的肝脏", "segment_ct", "CT"),
        (
            "Segment the liver in this computed tomography",
            "segment_ct",
            "CT",
        ),
        ("分割磁共振影像中的肝脏", "segment_mr", "MR"),
        ("分割核磁共振中的肝脏", "segment_mr", "MR"),
        ("Segment the liver in this MRI", "segment_mr", "MR"),
    ],
)
def test_text_modality_uses_single_tool_call_and_targets_only(monkeypatch, text, tool, modality):
    requests = []

    def check(payload):
        requests.append(payload)
        assert json.loads(payload["messages"][1]["content"]) == {
            "modality": None,
            "request": text,
        }
        for schema in payload["tools"]:
            assert schema["function"]["parameters"]["required"] == ["targets"]
            assert set(schema["function"]["parameters"]["properties"]) == {"targets"}

    selected = asyncio.run(
        agent.select_tool(
            text,
            transport=routing_transport(
                monkeypatch,
                tool,
                {"targets": ["liver"]},
                check_request=check,
            ),
        )
    )
    assert selected.modality == modality
    assert len(requests) == 1


@pytest.mark.parametrize(
    "text", ["分割肝脏", "分割肺结节", "Segment SPECT liver", "CT or MR liver"]
)
def test_absent_or_ambiguous_modality_is_local_clarification_without_provider(monkeypatch, text):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    def handler(request):
        pytest.fail("Missing or ambiguous modality must not contact the provider")

    with pytest.raises(agent.RoutingError) as error:
        asyncio.run(agent.select_tool(text, transport=httpx.MockTransport(handler)))
    assert error.value.code == "MODALITY_REQUIRED"
    assert "请在描述中注明" in str(error.value)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("segment_ct", {"targets": ["liver"], "unexpected": "CT"}),
        ("segment_ct", {"targets": []}),
        ("segment_mr", {"targets": ["liver"]}),
    ],
)
def test_text_modality_rejects_mismatch_or_invalid_arguments(monkeypatch, tool, arguments):
    with pytest.raises(agent.RoutingError):
        asyncio.run(
            agent.select_tool(
                "分割这份CT中的肝脏", transport=routing_transport(monkeypatch, tool, arguments)
            )
        )


@pytest.mark.parametrize("modality", ["", " ", "PET", [], {}, 1])
def test_invalid_explicit_modality_does_not_become_text_selection(modality):
    with pytest.raises(agent.RoutingError):
        agent.provider_payload("CT liver", modality)


@pytest.mark.parametrize("task", ["lung_nodules", "liver_lesions"])
def test_dedicated_lesion_routing_is_narrow_and_ct_only(task, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    schema = next(
        t["function"] for t in agent.tool_schema() if t["function"]["name"] == "segment_" + task
    )
    assert schema["parameters"]["properties"]["targets"]["items"]["enum"] == [task]

    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "segment_" + task,
                                        "arguments": json.dumps({"targets": [task]}),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    selected = asyncio.run(agent.select_tool(task, "CT", transport=httpx.MockTransport(handler)))
    assert selected.task == task and selected.targets == [task]
    with pytest.raises(agent.RoutingError, match="modality"):
        asyncio.run(agent.select_tool(task, "MR", transport=httpx.MockTransport(handler)))


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


def test_web_text_modality_persists_and_replay_survives_restart(tmp_path, monkeypatch):
    calls = []

    async def select(text, modality):
        calls.append((text, modality))
        return agent.Selection("segment_mr", "total_mr", ["liver"])

    async def segment(**kwargs):
        assert kwargs["task"] == "total_mr"
        parent = Path(kwargs["output_dir"]) / "unique-run"
        parent.mkdir(parents=True)
        path = parent / "segmentation.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), np.eye(4)), path)
        return {"segmentation_path": str(path), "task": "total_mr", "targets": ["liver"]}

    monkeypatch.setattr(agent, "select_tool", select)
    monkeypatch.setattr(core, "segment", segment)
    with client(tmp_path) as c:
        upload = c.post(
            "/api/uploads", headers={**ALICE, "X-Filename": "image.nii"}, content=nifti_bytes()
        ).json()
        body = {"upload_id": upload["id"], "text": "分割磁共振中的肝脏", "message_id": "text-only"}
        response = c.post("/api/tasks", headers=ALICE, json=body)
        assert response.status_code == 202, response.text
        task_id = response.json()["id"]
        for _ in range(50):
            row = c.get(f"/api/tasks/{task_id}", headers=ALICE).json()
            if row["status"] == "completed":
                break
        assert row["status"] == "completed", row
        assert row["modality"] == "MR" and row["modality_source"] == "text"
        assert c.post("/api/tasks", headers=ALICE, json=body).json()["id"] == task_id
        assert (
            c.post("/api/tasks", headers=ALICE, json={**body, "modality": "MR"}).status_code == 409
        )
        assert calls == [(body["text"], None)]
    with client(tmp_path) as c:
        row = c.get(f"/api/tasks/{task_id}", headers=ALICE).json()
        assert row["modality"] == "MR" and row["status"] == "completed"
        assert c.post("/api/tasks", headers=ALICE, json=body).json()["id"] == task_id


def test_web_projects_modality_clarification_without_inference(tmp_path, monkeypatch):
    original_select = agent.select_tool
    transport = routing_transport(monkeypatch, None, None)

    async def select(text, modality):
        return await original_select(text, modality, transport=transport)

    async def segment(**kwargs):
        pytest.fail("Missing modality must never start inference")

    monkeypatch.setattr(agent, "select_tool", select)
    monkeypatch.setattr(core, "segment", segment)
    with client(tmp_path) as c:
        upload = c.post(
            "/api/uploads", headers={**ALICE, "X-Filename": "image.nii"}, content=nifti_bytes()
        ).json()
        response = c.post(
            "/api/tasks",
            headers=ALICE,
            json={"upload_id": upload["id"], "text": "分割肝脏", "message_id": "needs-modality"},
        )
        assert response.status_code == 202
        task_id = response.json()["id"]
        for _ in range(50):
            row = c.get(f"/api/tasks/{task_id}", headers=ALICE).json()
            if row["status"] == "failed":
                break
        assert row["status"] == "failed", row
        assert row["error"]["code"] == "MODALITY_REQUIRED"
        assert "请在描述中注明" in row["error"]["message"]
        assert row["upload_id"] == upload["id"] and row["text"] == "分割肝脏"
        assert c.get(f"/api/uploads/{upload['id']}/file", headers=ALICE).status_code == 200

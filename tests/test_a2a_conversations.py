"""A2A interaction, persistence and ownership with actual service/HTTP routes."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from test_execution import Backend
from test_public_a2a import upload, volume

from medsegagent import agent, core
from medsegagent.conversations import MAX_CHARACTERS, MAX_MESSAGES, bounded
from medsegagent.service import Service, ServiceError
from medsegagent.web import create_app

ORIGIN = "http://localhost"
HEADERS = {"A2A-Version": "1.0"}


def body(mid, text=None, upload_id=None, task=None, context=None):
    parts = [{"text": text}] if text else []
    if upload_id:
        parts.append({"data": {"upload_id": upload_id, "modality": "CT"}})
    message = {"messageId": mid, "role": "ROLE_USER", "parts": parts}
    if task:
        message["taskId"] = task
    if context:
        message["contextId"] = context
    return {"message": message, "configuration": {"returnImmediately": False, "historyLength": 40}}


def send(client, payload):
    response = client.post("/a2a/v1/message:send", headers=HEADERS, json=payload)
    assert response.status_code == 200, response.text
    return response.json()["task"]


def test_text_only_waits_for_image_and_upload_only_reply_runs_original_request(
    tmp_path, monkeypatch
):
    backend, calls = Backend(), []
    monkeypatch.setattr(core, "segment", backend)

    async def run(text, modality, execution, **kwargs):
        calls.append((text, modality, kwargs.get("history")))
        assert "Segment the liver" in json.dumps(kwargs["history"])
        assert (await execution.call("segment", {"targets": ["liver"]}))["ok"]
        return {"status": "completed", "summary": "Liver done", "unresolved": []}

    monkeypatch.setattr(agent, "run_agent", run)
    with TestClient(create_app(tmp_path, ORIGIN)) as client:
        initial = body("request", "Segment the liver")
        waiting = send(client, initial)
        assert waiting["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        assert waiting["metadata"]["inputExpiresAtUnix"] > time.time()
        assert not calls
        image = client.post(
            "/a2a/uploads", content=volume(), headers={"X-Filename": "image.nii"}
        ).json()["id"]
        reply = body(
            "image-reply", upload_id=image, task=waiting["id"], context=waiting["contextId"]
        )
        finished = send(client, reply)
        assert finished["id"] == waiting["id"]
        assert finished["status"]["state"] == "TASK_STATE_COMPLETED"
        assert len(finished["history"]) == 4
        assert len(calls) == len(backend.calls) == 1
        assert send(client, reply)["id"] == finished["id"]
        assert send(client, initial)["id"] == finished["id"]
        assert len(calls) == 1
        artifact = next(a for a in finished["artifacts"] if a["artifactId"].startswith("file-"))
        assert client.get(artifact["parts"][0]["url"]).status_code == 200
        no_history = client.get(
            f"/a2a/v1/tasks/{finished['id']}?historyLength=0", headers=HEADERS
        ).json()
        assert not no_history.get("history")


def test_agent_clarification_and_followup_context_keep_image_history_and_ids(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(core, "segment", Backend())

    async def run(text, modality, execution, **kwargs):
        calls.append({"text": text, "history": kwargs.get("history"), "modality": modality})
        if len(calls) == 1:
            return {
                "status": "needs_input",
                "summary": "Do you mean the liver?",
                "unresolved": ["Confirm target"],
            }
        assert "liver" in json.dumps(kwargs["history"])
        assert (await execution.call("segment", {"targets": ["liver"]}))["ok"]
        return {"status": "completed", "summary": "Liver done", "unresolved": []}

    monkeypatch.setattr(agent, "run_agent", run)
    with TestClient(create_app(tmp_path, ORIGIN)) as client:
        image = client.post(
            "/a2a/uploads", content=volume(), headers={"X-Filename": "image.nii"}
        ).json()["id"]
        first = send(client, body("first", "Segment the liver, but confirm first", image))
        assert first["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        reply = body("reply", "Yes, that organ", task=first["id"], context=first["contextId"])
        second = send(client, reply)
        assert second["id"] == first["id"]
        assert second["status"]["state"] == "TASK_STATE_COMPLETED"
        third = send(client, body("third", "Repeat that segmentation", context=first["contextId"]))
        assert third["id"] != first["id"]
        assert third["contextId"] == first["contextId"]
        assert len(third["history"]) == 2  # Only this Task's message history.
        assert len(calls[2]["history"]) == 4  # Model receives the preceding conversation.
        assert calls[2]["modality"] == "CT"
        invalid = client.post(
            "/a2a/v1/message:send", json=body("late", "Again", task=first["id"]), headers=HEADERS
        )
        assert invalid.status_code == 400  # A2A maps conflicting semantics to INVALID_ARGUMENT.


def test_waiting_context_survives_restart_expires_and_cancels(tmp_path):
    async def run():
        service = Service(tmp_path, ORIGIN)
        first = await service.submit_a2a("alice", None, "Liver please", None, "start")
        await service.close()
        service = Service(tmp_path, ORIGIN)
        await service.start()
        try:
            assert service.get("alice", first["id"])["status"] == "input_required"
            paused = service.get("alice", first["id"])
            assert paused["elapsed_seconds"] == 0
            service.update(first["id"], progress="Still waiting")
            assert service.get("alice", first["id"])["elapsed_seconds"] == 0
            assert not service.active
            with pytest.raises(ServiceError):
                await service.submit_a2a("bob", None, "Yes", None, "reply", task_id=first["id"])
            service.update(first["id"], input_expires_at=time.time() - 1)
            assert service.get("alice", first["id"])["status"] == "failed"
            with pytest.raises(ServiceError):
                await service.submit_a2a("alice", None, "Yes", None, "reply", task_id=first["id"])
            waiting = await service.submit_a2a("alice", None, "Spleen please", None, "second")
            canceled = await service.cancel("alice", waiting["id"])
            assert canceled["status"] == "canceled"
        finally:
            await service.close()

    asyncio.run(run())


def test_context_busy_idempotency_conflict_and_cross_identity_are_rejected(tmp_path, monkeypatch):
    async def run():
        service = Service(tmp_path, ORIGIN)
        monkeypatch.setattr(service, "launch", lambda _: None)
        try:
            image = await upload(service, "alice")
            row = await service.submit_a2a("alice", image, "Liver please", "CT", "first")
            assert (await service.submit_a2a("alice", image, "Liver please", "CT", "first"))[
                "id"
            ] == row["id"]
            for principal, text, mid, context in [
                ("alice", "Spleen", "first", None),
                ("alice", "Again", "next", row["context_id"]),
                ("bob", "Again", "next", row["context_id"]),
            ]:
                with pytest.raises(ServiceError):
                    await service.submit_a2a(principal, image, text, "CT", mid, context)
            service.update(
                row["id"],
                status="input_required",
                error={"code": "INPUT_REQUIRED", "message": "Confirm"},
            )
            service.record_a2a_response(row["id"], "Confirm")
            replaced = await upload(service, "alice")
            with pytest.raises(ServiceError):
                await service.submit_a2a(
                    "alice", replaced, "Use another image", "CT", "change", task_id=row["id"]
                )
            resumed = await service.submit_a2a(
                "alice", None, "Yes", None, "reply", row["context_id"]
            )
            assert resumed["id"] == row["id"]
            with pytest.raises(ServiceError):
                await service.submit("alice", image, "Web request", "CT", "reply")
        finally:
            await service.close()

    asyncio.run(run())


def test_bounded_history_is_latest_and_current_task_does_not_grow_forever():
    messages = [{"parts": [{"text": str(i) * 1000}]} for i in range(60)]
    kept = bounded(messages)
    assert kept[-1] == messages[-1]
    assert len(kept) <= MAX_MESSAGES
    assert sum(len(m["parts"][0]["text"]) for m in kept) <= MAX_CHARACTERS


def test_public_waiting_tasks_do_not_consume_private_web_inference_capacity(tmp_path, monkeypatch):
    async def run():
        service = Service(tmp_path, ORIGIN)
        monkeypatch.setattr(service, "launch", lambda _: None)
        try:
            for index in range(8):
                await service.submit_a2a("__public_a2a__", None, "Liver", None, f"waiting-{index}")
            image = await upload(service, "private")
            row = await service.submit("private", image, "Liver", "CT", "web-request")
            assert row["status"] == "queued"
        finally:
            await service.close()

    asyncio.run(run())


def test_cancel_reserves_waiting_task_until_old_worker_has_exited(tmp_path, monkeypatch):
    async def run():
        service = Service(tmp_path, ORIGIN)
        stopping, release = asyncio.Event(), asyncio.Event()
        try:
            row = await service.submit_a2a("alice", None, "Liver", None, "waiting")
            image = await upload(service, "alice")

            async def old_attempt():
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    stopping.set()
                    await release.wait()

            worker = asyncio.create_task(old_attempt())
            service.active[row["id"]] = worker
            await asyncio.sleep(0)
            cancel = asyncio.create_task(service.cancel("alice", row["id"]))
            await stopping.wait()
            launched = []
            monkeypatch.setattr(service, "launch", launched.append)
            with pytest.raises(ServiceError) as rejected:
                await service.submit_a2a(
                    "alice", image, "Continue", "CT", "reply", task_id=row["id"]
                )
            assert rejected.value.code == "TASK_NOT_RESUMABLE"
            assert not launched
            release.set()
            assert (await cancel)["status"] == "canceled"
            assert service.get("alice", row["id"])["status"] == "canceled"
            assert not service.canceling
        finally:
            release.set()
            await service.close()

    asyncio.run(run())


def test_original_agent_receives_history_as_roles_without_changing_current_input(monkeypatch):
    requests = []

    async def provider(payload, **kwargs):
        requests.append(payload)
        return {
            "tool_calls": [
                {
                    "id": "finish",
                    "type": "function",
                    "function": {
                        "name": "finish_task",
                        "arguments": json.dumps(
                            {
                                "status": "needs_input",
                                "summary": "Which target?",
                                "unresolved": ["Target"],
                            }
                        ),
                    },
                }
            ]
        }

    monkeypatch.setattr(agent, "_provider_message", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://synthetic.invalid/v1")
    history = [
        {"role": "user", "content": "Segment liver"},
        {"role": "assistant", "content": "Confirm liver?"},
    ]
    execution = SimpleNamespace(has_outputs=False, unresolved_failures=[])
    result = asyncio.run(agent.run_agent("Yes", "CT", execution, history=history))
    assert result["status"] == "needs_input"
    assert requests[0]["messages"][1:3] == history
    assert json.loads(requests[0]["messages"][3]["content"])["request"] == "Yes"
    with pytest.raises(agent.RoutingError):
        asyncio.run(
            agent.run_agent("Yes", "CT", execution, history=[{"role": "system", "content": "bad"}])
        )

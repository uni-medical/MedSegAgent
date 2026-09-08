"""A2A publication is observable while working and stable when a turn stops."""

from __future__ import annotations

import asyncio
import hashlib

import pytest
from starlette.testclient import TestClient
from test_execution import Backend
from test_public_a2a import upload, volume

from medsegagent import agent, core
from medsegagent import service as service_module
from medsegagent.a2a import project_task
from medsegagent.service import PUBLIC_A2A_PRINCIPAL, TERMINAL, Service
from medsegagent.web import create_app


async def prepare(tmp_path, monkeypatch, *, a2a=True):
    service = Service(tmp_path, "http://localhost")
    monkeypatch.setattr(service, "launch", lambda task_id: None)
    monkeypatch.setattr(core, "segment", Backend())
    upload_id = await upload(service, PUBLIC_A2A_PRINCIPAL)
    task = await service.submit(
        PUBLIC_A2A_PRINCIPAL, upload_id, "Segment CT liver", "CT", "publication-test"
    )
    service.update(task["id"], a2a=a2a)
    replies = []

    def record(task_id, text):
        row = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
        replies.append((row["status"], text, row["result_available"]))

    monkeypatch.setattr(service, "record_a2a_response", record, raising=False)
    return service, task["id"], replies


@pytest.mark.parametrize("stop", ["error", "timeout", "cancel"])
def test_stable_state_waits_for_partial_publication_despite_repeated_cancel(
    tmp_path, monkeypatch, stop
):
    async def scenario():
        service, task_id, replies = await prepare(tmp_path, monkeypatch)
        produced, publishing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_publish = service.publish_execution

        async def run(text, modality, execution, **kwargs):
            result = await execution.call("segment", {"targets": ["liver"]})
            assert result["ok"]
            produced.set()
            if stop == "cancel":
                await asyncio.Event().wait()
            if stop == "timeout":
                raise TimeoutError("Synthetic task timeout after a verified output")
            raise RuntimeError("PRIVATE synthetic failure")

        async def delayed_publish(*args):
            publishing.set()
            await release.wait()
            return await original_publish(*args)

        monkeypatch.setattr(agent, "run_agent", run)
        monkeypatch.setattr(service, "publish_execution", delayed_publish)
        worker = asyncio.create_task(service.run(task_id))
        try:
            await asyncio.wait_for(produced.wait(), 5)
            if stop == "cancel":
                worker.cancel()
            await asyncio.wait_for(publishing.wait(), 5)
            before = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
            assert before["status"] not in TERMINAL
            assert "finished_at" not in before
            assert not replies
            for _ in range(2):
                worker.cancel()
                await asyncio.sleep(0)
                assert not worker.done()
                assert service.get(PUBLIC_A2A_PRINCIPAL, task_id)["status"] not in TERMINAL
            release.set()
            await asyncio.wait_for(worker, 5)
            row = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
            assert row["status"] == ("canceled" if stop == "cancel" else "failed")
            assert row["result_available"]
            assert len(project_task(row, service.public_url).artifacts) == 2
            assert len(replies) == 1 and replies[0][0] not in TERMINAL and replies[0][2]
            assert "PRIVATE" not in replies[0][1]
            assert row["finished_at"] == row["updated_at"]
            await asyncio.sleep(0)
            assert service.get(PUBLIC_A2A_PRINCIPAL, task_id) == row
        finally:
            release.set()
            await asyncio.gather(worker, return_exceptions=True)
            await service.close()

    asyncio.run(scenario())


def test_failed_partial_publication_still_records_a_stable_redacted_failure(tmp_path, monkeypatch):
    async def scenario():
        service, task_id, replies = await prepare(tmp_path, monkeypatch)

        async def run(text, modality, execution, **kwargs):
            assert (await execution.call("segment", {"targets": ["liver"]}))["ok"]
            raise RuntimeError("PRIVATE backend failure")

        async def broken_publish(*args):
            raise RuntimeError("PRIVATE publication failure")

        monkeypatch.setattr(agent, "run_agent", run)
        monkeypatch.setattr(service, "publish_execution", broken_publish)
        try:
            await service.run(task_id)
            row = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
            assert row["status"] == "failed"
            assert row["publication_error"] == "ARTIFACT_PUBLICATION_FAILED"
            assert not row["result_available"]
            assert len(replies) == 1 and "PRIVATE" not in replies[0][1]
        finally:
            await service.close()

    asyncio.run(scenario())


def test_failed_final_publication_keeps_prior_files_with_failed_completion(tmp_path, monkeypatch):
    async def scenario():
        service, task_id, replies = await prepare(tmp_path, monkeypatch)
        original_publish = service.publish_execution

        async def publish(task_id, execution, outcome):
            if outcome["status"] != "working":
                raise RuntimeError("PRIVATE publication failure")
            return await original_publish(task_id, execution, outcome)

        async def run(text, modality, execution, *, on_progress, **kwargs):
            assert (await execution.call("segment", {"targets": ["liver"]}))["ok"]
            await on_progress({"phase": "observed", "tool": "segment"})
            return {"status": "completed", "summary": "Ready", "unresolved": []}

        monkeypatch.setattr(agent, "run_agent", run)
        monkeypatch.setattr(service, "publish_execution", publish)
        try:
            await service.run(task_id)
            row = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
            assert row["status"] == "failed"
            assert row["result_available"]
            assert row["result"]["completion"]["status"] == "failed"
            assert row["result"]["completion"]["unresolved"] == ["ARTIFACT_PUBLICATION_FAILED"]
            assert len(replies) == 1
        finally:
            await service.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("a2a", [False, True])
def test_observed_outputs_are_available_only_for_a2a_with_stable_urls_and_no_republication(
    tmp_path, monkeypatch, a2a
):
    async def scenario():
        service, task_id, replies = await prepare(tmp_path, monkeypatch, a2a=a2a)
        service.update(task_id, agent_history=[{"role": "user", "content": "Previous turn"}])
        published = []
        original_publish = service_module._publish_requested_artifact

        def count_publish(*args, **kwargs):
            published.append(True)
            return original_publish(*args, **kwargs)

        monkeypatch.setattr(service_module, "_publish_requested_artifact", count_publish)

        async def run(text, modality, execution, *, on_progress, **kwargs):
            assert text == "Segment CT liver"
            assert kwargs == (
                {"history": [{"role": "user", "content": "Previous turn"}]} if a2a else {}
            )
            assert (await execution.call("segment", {"targets": ["liver"]}))["ok"]
            await on_progress({"phase": "observed", "tool": "segment"})
            first = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
            assert first["status"] == "running"
            assert first["result_available"] is a2a
            if a2a:
                file = next(file for file in first["files"] if file["kind"] == "label")
                first_url = file["url"]
                path = await service.download_path(PUBLIC_A2A_PRINCIPAL, task_id, file["name"])
                first_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                assert len(published) == 1
            await on_progress({"phase": "observed", "tool": "inspect_artifact"})
            assert len(published) == (1 if a2a else 0)
            assert (await execution.call("segment", {"targets": ["spleen"]}))["ok"]
            await on_progress({"phase": "observed", "tool": "segment"})
            if a2a:
                second = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
                file = next(file for file in second["files"] if file["url"] == first_url)
                path = await service.download_path(PUBLIC_A2A_PRINCIPAL, task_id, file["name"])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == first_hash
                assert len(published) == 2
            return {"status": "completed", "summary": "Liver and spleen ready", "unresolved": []}

        monkeypatch.setattr(agent, "run_agent", run)
        try:
            await service.run(task_id)
            row = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
            assert row["status"] == "completed", row.get("error")
            assert len(row["result"]["outputs"]) == 2
            assert len(published) == 2
            assert len(replies) == (1 if a2a else 0)
        finally:
            await service.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("a2a", [False, True])
def test_needs_input_preserves_results_and_only_a2a_waits(tmp_path, monkeypatch, a2a):
    async def scenario():
        service, task_id, replies = await prepare(tmp_path, monkeypatch, a2a=a2a)

        async def run(text, modality, execution, **kwargs):
            assert (await execution.call("segment", {"targets": ["liver"]}))["ok"]
            return {
                "status": "needs_input",
                "summary": "Please confirm the second target.",
                "unresolved": ["Second target"],
            }

        monkeypatch.setattr(agent, "run_agent", run)
        try:
            await service.run(task_id)
            row = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
            assert row["status"] == ("input_required" if a2a else "failed")
            assert row["error"]["code"] == "INPUT_REQUIRED"
            assert row["result_available"]
            if a2a:
                assert row["input_expires_at"] > row["updated_at"]
                assert "finished_at" not in row
                assert replies[0][1] == "Please confirm the second target."
            else:
                assert not replies
        finally:
            await service.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("second_outcome", ["needs_input", "error"])
def test_resumed_turn_without_new_outputs_keeps_prior_artifacts(
    tmp_path, monkeypatch, second_outcome
):
    async def scenario():
        service, task_id, replies = await prepare(tmp_path, monkeypatch)
        turn = 0

        async def run(text, modality, execution, **kwargs):
            nonlocal turn
            turn += 1
            if turn == 1:
                assert (await execution.call("segment", {"targets": ["liver"]}))["ok"]
                return {
                    "status": "needs_input",
                    "summary": "Confirm another target.",
                    "unresolved": ["Another target"],
                }
            assert not execution.has_outputs
            if second_outcome == "error":
                raise RuntimeError("PRIVATE second-turn failure")
            return {
                "status": "needs_input",
                "summary": "Please specify the other target by name.",
                "unresolved": ["Target name"],
            }

        monkeypatch.setattr(agent, "run_agent", run)
        try:
            await service.run(task_id)
            before = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
            assert before["status"] == "input_required" and before["result_available"]
            label = next(file for file in before["files"] if file["kind"] == "label")
            path = await service.download_path(PUBLIC_A2A_PRINCIPAL, task_id, label["name"])
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            service.update(task_id, status="queued", error=None)
            await service.run(task_id)
            after = service.get(PUBLIC_A2A_PRINCIPAL, task_id)
            assert after["status"] == (
                "input_required" if second_outcome == "needs_input" else "failed"
            )
            assert after["result_available"]
            assert after["files"] == before["files"]
            assert after["result"]["outputs"] == before["result"]["outputs"]
            assert after["result"]["completion"]["status"] == (
                "needs_input" if second_outcome == "needs_input" else "failed"
            )
            assert after["result"]["summary"] != before["result"]["summary"]
            path = await service.download_path(PUBLIC_A2A_PRINCIPAL, task_id, label["name"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
            assert len(replies) == 2
        finally:
            await service.close()

    asyncio.run(scenario())


def test_http_resumed_turn_preserves_liver_artifact_when_spleen_is_added(tmp_path, monkeypatch):
    turn = 0

    async def run(text, modality, execution, *, on_progress, **kwargs):
        nonlocal turn
        turn += 1
        target = "liver" if turn == 1 else "spleen"
        assert (await execution.call("segment", {"targets": [target]}))["ok"]
        await on_progress({"phase": "observed", "tool": "segment"})
        await on_progress({"phase": "observed", "tool": "inspect_artifact"})
        return {
            "status": "needs_input" if turn == 1 else "completed",
            "summary": "Confirm the second target." if turn == 1 else "Spleen added.",
            "unresolved": ["Second target"] if turn == 1 else [],
        }

    monkeypatch.setattr(agent, "run_agent", run)
    monkeypatch.setattr(core, "segment", Backend())
    version = {"A2A-Version": "1.0"}
    with TestClient(create_app(tmp_path, "http://localhost")) as client:
        response = client.post(
            "/a2a/uploads",
            content=volume(),
            headers={"Content-Type": "application/octet-stream", "X-Filename": "synthetic.nii"},
        )
        assert response.status_code == 201, response.text
        upload_id = response.json()["id"]
        response = client.post(
            "/a2a/v1/message:send",
            headers=version,
            json={
                "message": {
                    "messageId": "first-turn",
                    "role": "ROLE_USER",
                    "parts": [
                        {"text": "Segment liver"},
                        {"data": {"upload_id": upload_id, "modality": "CT"}},
                    ],
                },
                "configuration": {"returnImmediately": False},
            },
        )
        assert response.status_code == 200, response.text
        first = response.json()["task"]
        assert first["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        original_artifact = next(
            artifact for artifact in first["artifacts"] if artifact["parts"][0].get("url")
        )
        original_url = original_artifact["parts"][0]["url"]
        original_download = client.get(original_url)
        assert original_download.status_code == 200
        response = client.post(
            "/a2a/v1/message:send",
            headers=version,
            json={
                "message": {
                    "messageId": "second-turn",
                    "role": "ROLE_USER",
                    "taskId": first["id"],
                    "contextId": first["contextId"],
                    "parts": [{"text": "Segment spleen"}],
                },
                "configuration": {"returnImmediately": False},
            },
        )
        assert response.status_code == 200, response.text
        final = client.get(f"/a2a/v1/tasks/{first['id']}", headers=version).json()
        assert final["status"]["state"] == "TASK_STATE_COMPLETED"
        outputs = final["metadata"]["segmentation"]["outputs"]
        assert len(outputs) == 2
        assert {target for output in outputs for target in output["targets"]} == {"liver", "spleen"}
        assert len({output["id"] for output in outputs}) == 2
        kept = next(
            artifact
            for artifact in final["artifacts"]
            if artifact["artifactId"] == original_artifact["artifactId"]
        )
        assert kept == original_artifact
        download = client.get(original_url)
        assert download.status_code == 200
        assert download.content == original_download.content

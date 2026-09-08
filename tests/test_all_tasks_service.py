"""Producer identity survives publication, restart and both authenticated APIs."""

import asyncio
import hashlib
import json
import time

import pytest
from auth_helpers import ALICE, BOB
from auth_helpers import create_test_app as create_app
from starlette.testclient import TestClient
from test_execution import Backend, make_input

from medsegagent import agent, core
from medsegagent.task_specs import TASK_SPECS

HEADERS = ALICE
OTHER_HEADERS = BOB
CHOICES = [("total", "fast"), ("total_v3", "fast"), ("total_v3", "fastest")]


def test_producer_and_quality_survive_restart_web_a2a_and_file_download(tmp_path, monkeypatch):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)

    async def scripted_agent(text, modality, execution, **kwargs):
        for task, quality in CHOICES:
            response = await execution.call(
                "segment", {"task": task, "quality": quality, "targets": ["liver"]}
            )
            assert response["ok"], response
        return {"status": "completed", "summary": "Three liver outputs.", "unresolved": []}

    monkeypatch.setattr(agent, "run_agent", scripted_agent)
    image = make_input(tmp_path)
    root = tmp_path / "service"
    with TestClient(create_app(root, "http://localhost")) as client:
        upload = client.post(
            "/api/uploads",
            headers={**HEADERS, "X-Filename": "synthetic.nii.gz"},
            content=image.read_bytes(),
        )
        assert upload.status_code == 201, upload.text
        sent = client.post(
            "/api/tasks",
            headers=HEADERS,
            json={
                "upload_id": upload.json()["id"],
                "modality": "CT",
                "message_id": "three-producers",
                "text": "Compare three liver segmentations.",
            },
        )
        assert sent.status_code < 300, sent.text
        task_id = sent.json()["id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            row = client.get(f"/api/tasks/{task_id}", headers=HEADERS).json()
            if row["status"] in {"completed", "failed", "canceled"}:
                break
            time.sleep(0.01)
        else:
            pytest.fail("Synthetic task did not finish")
        assert row["status"] == "completed", row
        before = row["result"]
        assert before["schema_version"] == 5
        overlays = [file for file in before["files"] if file["kind"] == "overlay"]
        classes = [file for file in before["files"] if file["kind"] == "label"]
        assert len(before["outputs"]) == len(overlays) == len(classes) == 3
        assert len({file["name"] for file in before["files"]}) == 6
        assert len({file["sha256"] for file in overlays}) == 1
        assert {(out["task"], out["quality"]) for out in before["outputs"]} == set(CHOICES)
        assert {out["name"] for out in before["outputs"]} == {
            f"liver ({task} / {quality})" for task, quality in CHOICES
        }
        for output in before["outputs"]:
            assert output["usage_license"] == TASK_SPECS[output["task"]].usage_license
            assert output["requirements"] == list(TASK_SPECS[output["task"]].requirements)
            assert output["regions"][0]["task"] == output["task"]
            assert output["regions"][0]["quality"] == output["quality"]
            assert output["labels"][0]["id"] == 1
        assert {(row["task"], row["quality"]) for row in before["model_sources"]} == set(CHOICES)

    # Reopen the persisted service, without submitting or rerunning any model work.
    with TestClient(create_app(root, "http://localhost")) as client:
        response = client.get(f"/api/tasks/{task_id}", headers=HEADERS)
        assert response.status_code == 200
        assert response.json()["result"] == before
        a2a = client.get(f"/a2a/v1/tasks/{task_id}", headers={**HEADERS, "A2A-Version": "1.0"})
        assert a2a.status_code == 200, a2a.text
        task = a2a.json()
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert task["metadata"]["segmentation"] == before
        assert len([a for a in task["artifacts"] if a["artifactId"].startswith("file-")]) == 3
        for file in before["files"]:
            received = client.get(file["url"], headers=HEADERS)
            assert received.status_code == 200
            if file["kind"] == "overlay":
                assert hashlib.sha256(received.content).hexdigest() == file["sha256"]
                assert len(received.content) == file["size_bytes"]
            assert client.get(file["url"], headers=OTHER_HEADERS).status_code == 404
        assert client.get(f"/api/tasks/{task_id}", headers=OTHER_HEADERS).status_code == 404
        assert (
            client.get(
                f"/a2a/v1/tasks/{task_id}", headers={**OTHER_HEADERS, "A2A-Version": "1.0"}
            ).status_code
            == 404
        )
    assert len(backend.calls) == 3
    assert str(tmp_path) not in json.dumps(before)


def test_web_requests_use_configured_gpu_capacity_without_a_second_global_single_slot(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MEDSEGAGENT_DEVICE", "gpu")
    monkeypatch.setenv("MEDSEGAGENT_GPU_IDS", "0,1")
    monkeypatch.setenv("MEDSEGAGENT_MAX_CONCURRENT_INFERENCES", "2")
    monkeypatch.setattr(core, "segment", Backend())
    entered = None
    active = peak = 0

    async def scripted_agent(text, modality, execution, **kwargs):
        nonlocal entered, active, peak
        if entered is None:
            entered = asyncio.Event()
        active += 1
        peak = max(peak, active)
        if active == 2:
            entered.set()
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert (await execution.call("segment", {"targets": ["liver"]}))["ok"]
            return {"status": "completed", "summary": "Liver output.", "unresolved": []}
        finally:
            active -= 1

    monkeypatch.setattr(agent, "run_agent", scripted_agent)
    image = make_input(tmp_path)
    with TestClient(create_app(tmp_path / "service", "http://localhost")) as client:
        upload = client.post(
            "/api/uploads",
            headers={**HEADERS, "X-Filename": "synthetic.nii.gz"},
            content=image.read_bytes(),
        )
        assert upload.status_code == 201
        task_ids = []
        for index in range(2):
            sent = client.post(
                "/api/tasks",
                headers=HEADERS,
                json={
                    "upload_id": upload.json()["id"],
                    "modality": "CT",
                    "message_id": f"capacity-{index}",
                    "text": "Segment liver.",
                },
            )
            assert sent.status_code < 300, sent.text
            task_ids.append(sent.json()["id"])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rows = [
                client.get(f"/api/tasks/{task_id}", headers=HEADERS).json() for task_id in task_ids
            ]
            if all(row["status"] in {"completed", "failed", "canceled"} for row in rows):
                break
            time.sleep(0.01)
        else:
            pytest.fail("Concurrent synthetic tasks did not finish")
        assert [row["status"] for row in rows] == ["completed", "completed"], rows
    assert peak == 2 and active == 0

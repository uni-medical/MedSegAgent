"""Real orchestration and artifact handling with synthetic inference and provider IO."""

import gzip
import json
import time

import httpx
import nibabel as nib
import numpy as np
from auth_helpers import ALICE, BOB
from auth_helpers import create_test_app as create_app
from starlette.testclient import TestClient

from medsegagent import agent, core, weights

HEADERS = ALICE


def tool(name, arguments, index):
    return {
        "tool_calls": [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ]
    }


def install_pipeline(monkeypatch, *, fail_nodules=False):
    model_requests, backends = [], []

    async def inference(command, *, output_dir, **kwargs):
        task = command[command.index("--task") + 1]
        backends.append(task)
        if fail_nodules and task == "lung_nodules":
            raise core.SegmentationError("PRIVATE BACKEND PATH /patient/raw.log")
        source = nib.load(command[command.index("-i") + 1])
        values = np.zeros(source.shape, dtype=np.uint8)
        if task == "total":
            names = command[command.index("--roi_subset") + 1 :]
            labels = {name: value for value, name in core.task_labels(task).items()}
            for index, name in enumerate(names):
                values.flat[2 * index : 2 * index + 2] = labels[name]
        else:
            values.flat[:3] = 2
            values.flat[-1] = 2
        image = nib.Nifti1Image(values, source.affine, source.header)
        image.header.set_data_dtype(np.uint8)
        nib.save(image, output_dir / "segmentation.nii.gz")
        (output_dir / "run_report.json").write_text("{}")

    def provider(request):
        payload = json.loads(request.content)
        model_requests.append(payload)
        index = len(model_requests)
        if index == 1:
            message = tool("segment", {"targets": ["lungs", "lung_nodules"]}, index)
        elif fail_nodules:
            # Deliberately claim completion; host must reject it and keep partial files.
            message = {
                "content": json.dumps({"status": "completed", "summary": "Done", "unresolved": []})
            }
        else:
            feedback = json.loads(
                next(
                    row["content"] for row in reversed(payload["messages"]) if row["role"] == "tool"
                )
            )
            if index == 2:
                ids = [
                    row["region_id"]
                    for row in feedback["regions"]
                    if row["target"] in {"lungs", "lung_nodules"}
                ]
                message = tool("inspect_artifact", {"region_ids": ids}, index)
            elif index == 3:
                initial = json.loads(
                    next(row["content"] for row in payload["messages"] if row["role"] == "tool")
                )
                ids = [
                    row["region_id"]
                    for row in initial["regions"]
                    if row["target"] in {"lungs", "lung_nodules"}
                ]
                message = tool(
                    "compose_masks",
                    {"operation": "intersection", "region_ids": ids, "name": "in_lung_nodules"},
                    index,
                )
            else:
                message = {
                    "content": json.dumps(
                        {
                            "status": "completed",
                            "summary": "双肺和结节已分割，肺内结节区域与体积已生成。",
                            "unresolved": [],
                        }
                    )
                }
        return httpx.Response(200, json={"choices": [{"message": message}]})

    original = agent.run_agent

    async def run(*args, **kwargs):
        return await original(*args, **kwargs, transport=httpx.MockTransport(provider))

    # This suite simulates inference; real model/cache/device acceptance is separate.
    monkeypatch.setattr(weights, "require_weights", lambda *args, **kwargs: {"ready": True})
    monkeypatch.setenv("MEDSEGAGENT_DEVICE", "cpu")
    monkeypatch.setattr(core, "_run_command", inference)
    monkeypatch.setattr(agent, "run_agent", run)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://synthetic.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    return model_requests, backends


def submit(client):
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm")
    upload = client.post(
        "/api/uploads", headers={**HEADERS, "X-Filename": "synthetic.nii"}, content=image.to_bytes()
    ).json()
    body = {
        "upload_id": upload["id"],
        "modality": "CT",
        "message_id": "one-request",
        "text": "分割这份 CT 的双肺和肺结节，计算肺内结节体积。",
    }
    task_id = client.post("/api/tasks", json=body, headers=HEADERS).json()["id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = client.get(f"/api/tasks/{task_id}", headers=HEADERS).json()
        if row["status"] in {"completed", "failed", "canceled"}:
            return row, body
        time.sleep(0.01)
    raise AssertionError("Synthetic task did not finish")


def test_feedback_drives_composition_and_publishes_all_artifacts(tmp_path, monkeypatch):
    requests, backends = install_pipeline(monkeypatch)
    with TestClient(create_app(tmp_path, "http://localhost")) as client:
        row, body = submit(client)
        assert row["status"] == "completed", row
        assert len(requests) == 4
        assert backends == ["total", "lung_nodules"]
        result = row["result"]
        region = next(region for region in result["regions"] if region["name"] == "in_lung_nodules")
        assert region["voxels"] == 3
        assert region["volume_ml"] == 0.003
        assert len(result["outputs"]) >= 3
        assert row["result_available"]
        assert "labels" not in result  # No misleading single-output projection.
        for file in result["files"]:
            response = client.get(file["url"], headers=HEADERS)
            assert response.status_code == 200
            assert nib.Nifti1Image.from_bytes(gzip.decompress(response.content)).shape == (4, 5, 6)
            assert client.get(file["url"], headers=BOB).status_code == 404
        assert client.post("/api/tasks", json=body, headers=HEADERS).json()["id"] == row["id"]
        assert backends == ["total", "lung_nodules"]
        serialized = json.dumps(requests)
        assert str(tmp_path) not in serialized
        assert "synthetic.nii" not in serialized
        assert "run_report" not in serialized
        assert "volume_ml" in serialized  # The Agent actually observes measurements.


def test_partial_outputs_survive_failed_completion_and_are_projected_to_a2a(tmp_path, monkeypatch):
    _, backends = install_pipeline(monkeypatch, fail_nodules=True)
    with TestClient(create_app(tmp_path, "http://localhost")) as client:
        row, _ = submit(client)
        assert row["status"] == "failed"
        assert row["result_available"]
        assert row["result"]["completion"]["status"] == "failed"
        assert backends == ["total", "lung_nodules"]
        response = client.get(
            f"/a2a/v1/tasks/{row['id']}", headers={**HEADERS, "A2A-Version": "1.0"}
        )
        task = response.json()
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert task["metadata"]["segmentation"]["outputs"]
        assert any(artifact["artifactId"].startswith("file-") for artifact in task["artifacts"])
        for file in row["files"]:
            assert client.get(file["url"], headers=HEADERS).status_code == 200
        assert "PRIVATE BACKEND" not in json.dumps(row)

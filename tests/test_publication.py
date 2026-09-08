"""Requested-object publication, verified source bytes and cancellation ownership."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from auth_helpers import ALICE
from auth_helpers import create_test_app as create_app
from starlette.testclient import TestClient

from medsegagent import agent, catalog, core
from medsegagent import service as service_module
from medsegagent.execution import TaskExecution
from medsegagent.result_metadata import result_metadata
from medsegagent.service import Service
from medsegagent.task_specs import TASK_SPECS


class Publisher:
    publish_execution = Service.publish_execution

    def __init__(self, root):
        self.root = root
        self.updates = []

    def update(self, task_id, **fields):
        self.updates.append((task_id, fields))


def setup(tmp_path, monkeypatch, targets, *, modality="CT"):
    root = tmp_path / "service"
    task_id = "synthetic-task"
    parent = root / "tasks" / task_id
    image_path = tmp_path / "input.nii.gz"
    image = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.diag([2.0, 3.0, 4.0, 1.0]))
    image.header.set_xyzt_units("mm")
    nib.save(image, image_path)
    calls = []

    async def backend(**kwargs):
        calls.append(kwargs)
        source = nib.load(kwargs["input_path"])
        data = np.zeros(source.shape, dtype=np.uint8)
        names = {name: index for index, name in catalog.native_labels(kwargs["task"]).items()}
        labels = []
        for index, target in enumerate(kwargs["targets"], 1):
            data.flat[index] = index
            labels.append({"id": index, "name": target, "source_id": names[target]})
        path = Path(kwargs["output_dir"]) / f"run-{len(calls)}" / "segmentation.nii.gz"
        path.parent.mkdir(parents=True)
        output = nib.Nifti1Image(data, source.affine, source.header)
        output.header.set_data_dtype(np.uint8)
        nib.save(output, path)
        return {"segmentation_path": str(path), "labels": labels}

    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(image_path, modality, parent / "execution")
    response = asyncio.run(execution.call("segment", {"targets": targets}))
    assert response["ok"]
    return Publisher(root), execution, task_id, parent


def publish(publisher, execution, task_id):
    return asyncio.run(
        publisher.publish_execution(
            task_id,
            execution,
            {"status": "completed", "summary": "Complete.", "unresolved": []},
        )
    )


def test_lungs_publishes_only_the_requested_union_not_five_source_lobes(tmp_path, monkeypatch):
    publisher, execution, task_id, parent = setup(tmp_path, monkeypatch, ["lungs"])
    exported = execution.export_result()
    assert len(exported["artifacts"]) == 2
    result = publish(publisher, execution, task_id)
    assert len(result["outputs"]) == 1
    assert result["targets"] == ["lungs"]
    assert [row["target"] for row in result["regions"]] == ["lungs"]
    assert result["files"][0]["name"] == "segmentation.nii.gz"
    image = nib.load(parent / "segmentation.nii.gz")
    assert set(np.unique(image.get_fdata())) == {0, 1}
    assert np.count_nonzero(image.get_fdata()) == 5
    assert len(publisher.updates) == 1
    assert "path" not in json.dumps(result) and "provenance" not in json.dumps(result)


def test_mixed_native_artifact_filters_auxiliary_lobes_and_updates_label_values(
    tmp_path, monkeypatch
):
    publisher, execution, task_id, parent = setup(tmp_path, monkeypatch, ["lungs", "liver"])
    exported = execution.export_result()
    source = next(row for row in exported["artifacts"] if row.get("task") == "total")
    original_hash = core._file_digest(Path(source["path"]))
    liver_region = next(row for row in exported["regions"] if row["target"] == "liver")
    assert liver_region["values"] == [6]
    result = publish(publisher, execution, task_id)
    assert [row["targets"] for row in result["outputs"]] == [["lungs"], ["liver"]]
    assert len(result["files"]) == 2
    for output in result["outputs"]:
        image = nib.load(parent / output["files"][0]["name"])
        assert set(np.unique(image.get_fdata())) == {0, 1}
        assert output["labels"][0]["id"] == 1
        assert output["regions"][0]["values"] == [1]
    liver = result["outputs"][1]
    assert liver["nonzero_voxels"] == 1
    assert liver["regions"][0]["id"] == liver_region["id"]
    assert [row["target"] for row in result["regions"]] == ["lungs", "liver"]
    assert core._file_digest(Path(source["path"])) == original_hash
    assert len(np.unique(nib.load(source["path"]).get_fdata())) == 7


def test_native_objects_share_one_public_mask_with_exact_target_mapping(tmp_path, monkeypatch):
    publisher, execution, task_id, parent = setup(tmp_path, monkeypatch, ["spleen", "liver"])
    result = publish(publisher, execution, task_id)
    assert len(result["outputs"]) == 1
    assert result["targets"] == ["spleen", "liver"]
    assert [row["id"] for row in result["labels"]] == [1, 2]
    assert [row["values"] for row in result["regions"]] == [[1], [2]]
    assert set(np.unique(nib.load(parent / "segmentation.nii.gz").get_fdata())) == {0, 1, 2}


def test_same_numeric_id_in_two_backends_remains_two_independent_masks(tmp_path, monkeypatch):
    publisher, execution, task_id, parent = setup(tmp_path, monkeypatch, ["liver", "lung_nodules"])
    result = publish(publisher, execution, task_id)
    assert len(result["outputs"]) == 2 and len(result["files"]) == 2
    assert {row["targets"][0] for row in result["outputs"]} == {"liver", "lung_nodules"}
    assert (
        len({row["name"] for row in result["files"]}) == 2
    )  # Headers encode distinct label tables.
    for file in result["files"]:
        assert set(np.unique(nib.load(parent / file["name"]).get_fdata())) == {0, 1}


def test_identical_masks_from_distinct_producers_and_qualities_keep_separate_files(
    tmp_path, monkeypatch
):
    publisher, execution, task_id, parent = setup(tmp_path, monkeypatch, ["liver"])
    choices = [("total", "fast"), ("total_v3", "standard"), ("total_v3", "fastest")]
    for task, quality in choices[1:]:
        response = asyncio.run(
            execution.call("segment", {"task": task, "targets": ["liver"], "quality": quality})
        )
        assert response["ok"]
    result = publish(publisher, execution, task_id)
    assert result["schema_version"] == 5
    assert len(result["outputs"]) == len(result["files"]) == 3
    # These synthetic backends return exactly the same bytes and native label ID.
    # Content hashes alone must not collapse independently requested producers.
    assert len({row["sha256"] for row in result["files"]}) == 1
    assert len({row["name"] for row in result["files"]}) == 3
    assert len({(parent / row["name"]).read_bytes() for row in result["files"]}) == 1
    assert {(row["task"], row["quality"]) for row in result["outputs"]} == set(choices)
    assert {(row["task"], row["quality"]) for row in result["model_sources"]} == set(choices)
    for output in result["outputs"]:
        assert output["name"] == f"liver ({output['task']} / {output['quality']})"
        assert output["targets"] == ["liver"]
        assert output["labels"][0]["id"] == 1
        assert output["labels"][0]["name"] == "liver"
        assert output["regions"][0]["target"] == "liver"
        assert output["usage_license"] == TASK_SPECS[output["task"]].usage_license
        assert output["requirements"] == list(TASK_SPECS[output["task"]].requirements)
        assert output["regions"][0]["quality"] == output["quality"]
    assert "labels" not in result


@pytest.mark.parametrize("tamper", ["task", "quality", "duplicate"])
def test_producer_identity_tampering_fails_before_publication(tmp_path, monkeypatch, tamper):
    publisher, execution, task_id, parent = setup(tmp_path, monkeypatch, ["liver"])
    exported = execution.export_result()
    if tamper == "duplicate":
        exported["outputs"].append(copy.deepcopy(exported["outputs"][0]))
    else:
        exported["outputs"][0][tamper] = {"task": "total_v3", "quality": "standard"}[tamper]
    monkeypatch.setattr(execution, "export_result", lambda: copy.deepcopy(exported))
    with pytest.raises(ValueError, match="manifest"):
        publish(publisher, execution, task_id)
    assert publisher.updates == []
    assert list(parent.glob("*.nii.gz")) == []


def test_compositions_retain_source_license_and_sequence_requirements(tmp_path, monkeypatch):
    publisher, execution, task_id, _parent = setup(tmp_path, monkeypatch, ["brain"], modality="MR")
    assert asyncio.run(
        execution.call("segment", {"task": "brain_aneurysm", "targets": ["brain_aneurysm"]})
    )["ok"]
    ids = [row["region_id"] for row in execution.export_result()["outputs"]]
    assert asyncio.run(
        execution.call(
            "compose_masks",
            {"operation": "intersection", "region_ids": ids, "name": "intracranial_aneurysm"},
        )
    )["ok"]
    # Follow a second derived region to verify provenance survives nested composition.
    ids = [row["region_id"] for row in execution.export_result()["outputs"]]
    assert asyncio.run(
        execution.call(
            "compose_masks",
            {"operation": "union", "region_ids": ids[-2:], "name": "aneurysm_union"},
        )
    )["ok"]
    exported = execution.export_result()
    for collection in ("artifacts", "regions", "outputs"):
        for row in exported[collection]:
            assert "usage_license" in row and "requirements" in row
            if row["task"] in {"brain_aneurysm", "composition"}:
                assert "CC-BY-NC-4.0" in row["usage_license"]
                assert "TOF MRI only." in row["requirements"]
            if row["task"] == "composition":
                assert {source["task"] for source in row["model_sources"]} == {
                    "total_mr",
                    "brain_aneurysm",
                }
    # Native Agent and MCP feedback associate policy once with each artifact, while
    # every region retains its artifact ID. Paths remain private to the local export.
    observed = agent.safe_feedback(execution.snapshot())
    assert observed["artifacts"]
    aneurysm = next(row for row in observed["artifacts"] if row["task"] == "brain_aneurysm")
    assert aneurysm["usage_license"] == "CC-BY-NC-4.0"
    assert "TOF MRI only." in aneurysm["requirements"]
    assert str(tmp_path) not in json.dumps(observed)
    result = publish(publisher, execution, task_id)
    for output in result["outputs"]:
        if output["task"] != "composition":
            continue
        assert output["quality"] is None
        assert "CC-BY-NC-4.0" in output["usage_license"]
        assert "TOF MRI only." in output["requirements"]
        assert {row["task"] for row in output["model_sources"]} == {"total_mr", "brain_aneurysm"}
        assert output["regions"][0]["model_sources"] == output["model_sources"]
    serialized = json.dumps(result)
    assert "source_region_ids" not in serialized and "provenance" not in serialized
    assert str(tmp_path) not in serialized


def test_public_model_metadata_projection_is_bounded_and_idempotent():
    model = {
        "task": "brain_aneurysm",
        "quality": "standard",
        "usage_license": "CC-BY-NC-4.0",
        "requirements": ["TOF MRI only.", {"private": "/secret/path"}, "invalid\ntext"],
        "private_report": {"path": "/secret/path"},
    }
    value = {
        "model_sources": [model, "invalid"],
        "outputs": [{"id": "one", "name": "aneurysm", **model, "regions": [{"id": "r1", **model}]}],
    }
    result = result_metadata(value)
    assert result_metadata(result) == result
    assert result["outputs"][0]["requirements"] == ["TOF MRI only."]
    assert result["outputs"][0]["regions"][0]["quality"] == "standard"
    assert result["model_sources"][0]["usage_license"] == "CC-BY-NC-4.0"
    assert "/secret" not in json.dumps(result)
    assert result_metadata({"quality": [], "requirements": {}, "task": "/private/task"}) == {}


def test_changed_source_receipt_blocks_publication_and_cleans_temporary_files(
    tmp_path, monkeypatch
):
    publisher, execution, task_id, parent = setup(tmp_path, monkeypatch, ["liver"])
    artifact = execution.export_result()["artifacts"][0]
    Path(artifact["path"]).write_bytes(b"changed after validation")
    with pytest.raises(ValueError, match="changed"):
        publish(publisher, execution, task_id)
    assert publisher.updates == []
    assert not (parent / "segmentation.nii.gz").exists()
    assert list(parent.glob(".publish-*")) == []


def test_inconsistent_output_mapping_fails_before_any_file_is_published(tmp_path, monkeypatch):
    publisher, execution, task_id, parent = setup(tmp_path, monkeypatch, ["liver"])
    exported = execution.export_result()
    exported["outputs"][0]["target"] = "lungs"
    monkeypatch.setattr(execution, "export_result", lambda: copy.deepcopy(exported))
    with pytest.raises(ValueError, match="manifest"):
        publish(publisher, execution, task_id)
    assert publisher.updates == []
    assert not (parent / "segmentation.nii.gz").exists()


def test_cancel_waits_for_publication_worker_cleanup(tmp_path, monkeypatch):
    publisher, execution, task_id, parent = setup(tmp_path, monkeypatch, ["liver"])
    started, drained = threading.Event(), threading.Event()

    def slow_worker(*args, _stop_event=None, **kwargs):
        try:
            started.set()
            while not _stop_event.is_set():
                time.sleep(0.001)
            time.sleep(0.02)  # Cancellation must wait for actual worker cleanup.
            core._check_stop(_stop_event)
        finally:
            drained.set()

    monkeypatch.setattr(service_module, "_publish_requested_artifact", slow_worker)

    async def run():
        worker = asyncio.create_task(
            publisher.publish_execution(
                task_id,
                execution,
                {"status": "completed", "summary": "Complete.", "unresolved": []},
            )
        )
        while not started.is_set():
            await asyncio.sleep(0.001)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        assert drained.is_set()

    asyncio.run(run())
    assert publisher.updates == []
    assert not (parent / "segmentation.nii.gz").exists()


def test_unresolved_request_can_publish_only_verified_partial_targets(tmp_path, monkeypatch):
    publisher, execution, task_id, _parent = setup(tmp_path, monkeypatch, ["liver"])
    result = asyncio.run(
        publisher.publish_execution(
            task_id,
            execution,
            {
                "status": "needs_input",
                "summary": "Clarify the other target.",
                "unresolved": ["Unsupported target"],
            },
        )
    )
    assert result["completion"]["status"] == "needs_input"
    assert result["completion"]["unresolved"] == ["Unsupported target"]
    assert result["targets"] == ["liver"]


_HEADERS = ALICE


def _submit_and_wait(client, image_path):
    uploaded = client.post(
        "/api/uploads",
        headers={**_HEADERS, "X-Filename": "synthetic.nii.gz"},
        content=image_path.read_bytes(),
    )
    assert uploaded.status_code == 201, uploaded.text
    sent = client.post(
        "/api/tasks",
        headers=_HEADERS,
        json={
            "upload_id": uploaded.json()["id"],
            "text": "分割这份 CT 中的肝脏",
            "message_id": "synthetic-publication-test",
        },
    )
    assert sent.status_code < 300, sent.text
    task_id = sent.json()["id"]
    for _ in range(100):
        row = client.get(f"/api/tasks/{task_id}", headers=_HEADERS).json()
        if row["status"] in {"completed", "failed", "canceled"}:
            return row
        client.portal.call(asyncio.sleep, 0.01)
    pytest.fail("The synthetic task did not finish")


@pytest.mark.parametrize("failure", ["missing_output", "unresolved_text", "execution_failure"])
def test_service_normalizes_false_completion_before_publishing(tmp_path, monkeypatch, failure):
    publisher, _execution, _task_id, _parent = setup(tmp_path, monkeypatch, ["liver"])

    async def claim_completion(text, modality, execution, **kwargs):
        assert modality is None  # The Agent chooses the current text declaration.
        if failure != "missing_output":
            assert (await execution.call("segment", {"targets": ["liver"], "modality": "CT"}))["ok"]
        if failure == "execution_failure":
            assert not (await execution.call("segment", {"targets": ["not_supported"]}))["ok"]
        return {
            "status": "completed",
            "summary": "Incorrect completion claim.",
            "unresolved": ["Unmet requirement"] if failure == "unresolved_text" else [],
        }

    monkeypatch.setattr(agent, "run_agent", claim_completion)
    with TestClient(create_app(publisher.root, "http://localhost")) as client:
        row = _submit_and_wait(client, tmp_path / "input.nii.gz")
        assert row["status"] == "failed"
        assert row["error"]["code"] == "INCOMPLETE_TASK"
        assert row["result"]["completion"]["status"] == "failed"
        assert row["result"]["completion"]["unresolved"]
        assert "尚未完整完成" in row["result"]["summary"]
        assert bool(row["result"]["outputs"]) == (failure != "missing_output")


def test_a2a_retains_clarification_metadata_even_without_artifacts(tmp_path, monkeypatch):
    publisher, _execution, _task_id, _parent = setup(tmp_path, monkeypatch, ["liver"])

    async def clarify(*args, **kwargs):
        return {
            "status": "needs_input",
            "summary": "请明确需要分割哪种对象。",
            "unresolved": ["目标尚不明确"],
        }

    monkeypatch.setattr(agent, "run_agent", clarify)
    with TestClient(create_app(publisher.root, "http://localhost")) as client:
        row = _submit_and_wait(client, tmp_path / "input.nii.gz")
        response = client.get(
            f"/a2a/v1/tasks/{row['id']}", headers={**_HEADERS, "A2A-Version": "1.0"}
        )
        assert response.status_code == 200, response.text
        task = response.json()
        assert task["metadata"]["errorCode"] == "INPUT_REQUIRED"
        result = task["metadata"]["segmentation"]
        assert result["completion"]["status"] == "needs_input"
        assert result["completion"]["unresolved"] == ["目标尚不明确"]
        assert result["summary"] == "请明确需要分割哪种对象。"
        assert not task.get("artifacts")

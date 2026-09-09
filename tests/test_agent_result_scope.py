"""Result scope uses real execution guards; provider and inference IO are synthetic."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import nibabel as nib
import numpy as np
import pytest

from medsegagent import agent, catalog, core
from medsegagent.execution import TaskExecution

HEART_TARGETS = [
    "heart",
    "aorta",
    "pulmonary_vein",
    "atrial_appendage_left",
    "superior_vena_cava",
    "inferior_vena_cava",
]
HEART_SUMMARY = "已分割心脏整体、主动脉、肺静脉、左心耳、上腔静脉和下腔静脉。"


@pytest.fixture(autouse=True)
def provider_config(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.delenv("MEDSEGAGENT_MAX_MODEL_REQUESTS", raising=False)
    monkeypatch.delenv("MEDSEGAGENT_MAX_TOOL_CALLS", raising=False)


def action(name, arguments, index):
    return {
        "tool_calls": [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ]
    }


def run_task(tmp_path, monkeypatch, text, steps, final, *, fail_task=None):
    """Use synthetic labelmaps while retaining freezing, geometry and coverage checks."""
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm")
    source = tmp_path / "synthetic.nii.gz"
    nib.save(image, source)
    backends, requests = [], []

    async def inference(**kwargs):
        backends.append(kwargs["task"])
        if kwargs["task"] == fail_task:
            raise core.SegmentationError("Synthetic inference failure")
        original = nib.load(kwargs["input_path"])
        data = np.zeros(original.shape, dtype=np.uint8)
        native = {name: index for index, name in catalog.native_labels(kwargs["task"]).items()}
        labels = []
        for label_id, target in enumerate(kwargs["targets"], 1):
            data.flat[label_id - 1] = label_id
            labels.append({"id": label_id, "source_id": native[target], "name": target})
        output_dir = Path(kwargs["output_dir"]) / f"synthetic-{len(backends)}"
        output_dir.mkdir(parents=True)
        output_path = output_dir / "segmentation.nii.gz"
        output = nib.Nifti1Image(data, original.affine, original.header)
        output.header.set_data_dtype(np.uint8)
        nib.save(output, output_path)
        return {
            "segmentation_path": str(output_path),
            "task": kwargs["task"],
            "targets": kwargs["targets"],
            "labels": labels,
            "runtime_seconds": 0.01,
        }

    monkeypatch.setattr(core, "segment", inference)
    messages = iter(
        [action(name, arguments, index) for index, (name, arguments) in enumerate(steps)]
        + [action("finish_task", final, len(steps))]
    )

    def provider(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": next(messages)}]})

    execution = TaskExecution(source, "CT", tmp_path / "outputs")
    result = asyncio.run(
        agent.run_agent(text, "CT", execution, transport=httpx.MockTransport(provider))
    )
    return result, execution, requests, backends


@pytest.mark.parametrize("scope_note", ["", "本次结果未细分左右心房和心室。"])
def test_broad_heart_request_completes_with_verified_available_structures(
    tmp_path, monkeypatch, scope_note
):
    summary = HEART_SUMMARY + scope_note
    result, execution, requests, backends = run_task(
        tmp_path,
        monkeypatch,
        "分割心脏各个结构",
        [("segment", {"targets": HEART_TARGETS})],
        {"status": "completed", "summary": summary, "unresolved": []},
    )

    assert result["status"] == "completed" and result["unresolved"] == []
    assert result["summary"] == summary
    assert execution.is_complete and backends == ["total"]
    assert {row["target"] for row in execution.export_result()["outputs"]} == set(HEART_TARGETS)
    feedback = json.loads(requests[-1]["messages"][-1]["content"])
    assert feedback["ok"] and len(feedback["artifacts"]) == 1
    assert feedback["artifacts"][0]["geometry_validated"]
    assert {row["target"] for row in feedback["regions"]} == set(HEART_TARGETS)
    assert all(row["voxels"] == 1 for row in feedback["regions"])


@pytest.mark.parametrize("reported_status", ["needs_input", "failed", "completed"])
def test_explicit_unmet_chambers_remain_unresolved_despite_verified_heart_outputs(
    tmp_path, monkeypatch, reported_status
):
    missing = "左心房、右心房、左心室和右心室尚未分别分割。"
    result, execution, _, _ = run_task(
        tmp_path,
        monkeypatch,
        "分割心脏整体及相连大血管，并分别分割左心房、右心房、左心室、右心室。",
        [("segment", {"targets": HEART_TARGETS})],
        {"status": reported_status, "summary": HEART_SUMMARY, "unresolved": [missing]},
    )

    # Valid existing outputs do not establish coverage of an explicit missing target.
    assert execution.is_complete
    assert len(execution.export_result()["outputs"]) == 6
    assert result["status"] == ("failed" if reported_status == "completed" else reported_status)
    assert result["unresolved"] == [missing]


@pytest.mark.parametrize("observe_after_failure", [False, True])
def test_real_inference_failure_blocks_completion_even_after_other_work_succeeds(
    tmp_path, monkeypatch, observe_after_failure
):
    steps = [
        ("segment", {"targets": HEART_TARGETS}),
        ("segment", {"targets": ["lung_nodules"]}),
    ]
    if observe_after_failure:
        steps.append(("get_capabilities", {"modality": "CT", "query": "heart"}))
    result, execution, requests, backends = run_task(
        tmp_path,
        monkeypatch,
        "分割心脏相关结构和肺结节。",
        steps,
        {"status": "completed", "summary": HEART_SUMMARY, "unresolved": []},
        fail_task="lung_nodules",
    )

    assert result["status"] == "failed" and result["unresolved"]
    assert execution.has_outputs and not execution.is_complete
    assert {row["target"] for row in execution.export_result()["outputs"]} == set(HEART_TARGETS)
    assert backends == ["total", "lung_nodules"]
    assert any(row["code"] == "INFERENCE_FAILED" for row in execution.unresolved_failures)
    if observe_after_failure:
        assert json.loads(requests[-1]["messages"][-1]["content"])["ok"]


def test_provider_keeps_modality_evidence_without_routine_user_disclosure():
    payload = agent.provider_payload("分割心脏各个结构", example_modality_hint="CT")
    request = json.loads(payload["messages"][1]["content"])
    assert request["modality_hint"] == {"modality": "CT", "source": "example_manifest"}

    # Evidence remains available for choosing a modality, while communicating it is
    # conditional on the user's request or a material ambiguity. Check that policy
    # boundary, rather than snapshotting the whole prompt or generated wording.
    instructions = payload["messages"][0]["content"].lower()
    routine_policy = re.search(r"routine\s+modality.{0,500}", instructions)
    assert routine_policy is not None
    assert re.search(r"only\s+(?:if|when).{0,80}request", routine_policy.group())
    assert re.search(r"ambigu(?:ity|ous)|uncertain", routine_policy.group())
    assert not re.search(
        r"briefly\s+noting\s+an\s+inferred\s+choice\s+in\s+the\s+summary", instructions
    )

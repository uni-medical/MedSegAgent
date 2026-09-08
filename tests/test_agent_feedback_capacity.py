"""Every accepted semantic batch must survive the actual executor/Agent feedback path."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from medsegagent import agent, catalog, core
from medsegagent.execution import TaskExecution


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_maximum_semantic_batch_finishes_agent_loop_without_losing_ids(
    modality, tmp_path, monkeypatch
):
    source = tmp_path / "synthetic.nii.gz"
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm")
    nib.save(image, source)
    execution = TaskExecution(source, modality, tmp_path / "out")
    targets = sorted(catalog.public_targets(modality))
    backend_tasks, feedback_sizes = [], []

    async def backend(**kwargs):
        backend_tasks.append(kwargs["task"])
        native = {name: index for index, name in catalog.native_labels(kwargs["task"]).items()}
        data = np.zeros(image.shape, dtype=np.uint8)
        labels = []
        for index, name in enumerate(kwargs["targets"], 1):
            data.flat[index - 1] = index
            labels.append({"id": index, "source_id": native[name], "name": name})
        run = Path(kwargs["output_dir"]) / f"synthetic-{len(backend_tasks)}"
        run.mkdir(parents=True)
        mask = run / "segmentation.nii.gz"
        nib.save(nib.Nifti1Image(data, image.affine), mask)
        return {"segmentation_path": str(mask), "labels": labels}

    async def provider(payload, **kwargs):
        tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
        if not tool_messages:
            return {
                "tool_calls": [
                    {
                        "id": "all_native_semantics",
                        "type": "function",
                        "function": {
                            "name": "segment",
                            "arguments": json.dumps({"targets": targets}),
                        },
                    }
                ]
            }
        assert len(tool_messages) == 1
        feedback = json.loads(tool_messages[0]["content"])
        assert feedback["ok"] is True
        assert feedback["requested_targets"] == targets
        assert {row["region_id"] for row in feedback["regions"]} == set(execution._regions)
        assert {row["artifact_id"] for row in feedback["artifacts"]} == set(execution._artifacts)
        assert all(row["region_id"] in execution._regions for row in feedback["regions"])
        feedback_sizes.append(len(tool_messages[0]["content"].encode()))
        return {
            "content": json.dumps(
                {
                    "status": "completed",
                    "summary": "All requested synthetic outputs are ready.",
                    "unresolved": [],
                }
            )
        }

    monkeypatch.setattr(core, "segment", backend)
    monkeypatch.setattr(agent, "_provider_message", provider)
    outcome = asyncio.run(agent.run_agent("Segment every public object.", modality, execution))
    assert outcome["status"] == "completed"
    assert outcome["model_requests"] == 2 and outcome["tool_calls"] == 1
    assert execution.is_complete
    assert len(backend_tasks) > 1
    assert 0 < feedback_sizes[0] <= agent.MAX_FEEDBACK_BYTES <= 512 * 1024
    if modality == "CT":
        assert feedback_sizes[0] > 64 * 1024  # Regression: the previous budget made this fatal.


def test_feedback_keeps_a_finite_upper_bound():
    response = {"ok": True, "regions": [{"target": "x" * 256}] * 1024}
    assert agent.safe_feedback(response)["code"] == "INVALID_TOOL_FEEDBACK"

"""Fresh artifact references survive repeated inference without growing observations forever."""

import asyncio
import json

from test_execution import Backend, make_input

from medsegagent import agent, catalog, core
from medsegagent.execution import TaskExecution


def test_full_anatomy_second_quality_returns_current_region_ids(tmp_path, monkeypatch):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")
    targets = sorted(catalog.anatomical_targets("CT"))

    async def scenario():
        first = await execution.call("segment", {"targets": targets})
        first_ids = {row["region_id"] for row in first["regions"]}
        second = await execution.call("segment", {"targets": targets, "quality": "standard"})
        observed = agent.safe_feedback(second)
        assert observed["ok"]
        assert len(observed["regions"]) == len(targets) == 117
        assert not first_ids & {row["region_id"] for row in observed["regions"]}
        assert {row["target"] for row in observed["regions"]} == set(targets)
        assert len(observed["artifacts"]) == 1
        assert observed["artifacts"][0]["quality"] == "standard"
        assert len(execution.export_result()["regions"]) == 234
        assert str(tmp_path) not in json.dumps(observed)
        assert "private_report" not in json.dumps(observed)

    asyncio.run(scenario())


def test_partial_batch_keeps_new_region_ids_and_omits_old_quality(tmp_path, monkeypatch):
    backend = Backend(fail="lung_nodules")
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")
    targets = sorted(catalog.anatomical_targets("CT"))

    async def scenario():
        first = await execution.call("segment", {"targets": targets})
        previous = {row["region_id"] for row in first["regions"]}
        response = await execution.call(
            "segment",
            {
                "targets": [*targets, "lung_nodules"],
                "quality": "standard",
            },
        )
        observed = agent.safe_feedback(response)
        assert observed["code"] == "INFERENCE_FAILED"
        assert len(observed["regions"]) == 117
        assert not previous & {row["region_id"] for row in observed["regions"]}
        assert observed["unresolved_failures"]
        assert "PRIVATE-BACKEND" not in json.dumps(observed)
        assert str(tmp_path) not in json.dumps(observed)

    asyncio.run(scenario())


def test_safe_feedback_does_not_silently_truncate_valid_region_identifiers():
    regions = [{"region_id": f"region-{index}", "voxels": 0} for index in range(200)]
    observed = agent.safe_feedback({"ok": True, "regions": regions})
    assert observed["ok"]
    assert observed["regions"] == regions


def test_feedback_over_budget_fails_explicitly_without_partial_regions():
    row = {"region_id": "r" * 120, "target": "t" * 120}
    encoded_size = len(json.dumps(row, separators=(",", ":")))
    oversized = [row] * (agent.MAX_FEEDBACK_BYTES // encoded_size + 2)
    assert len(oversized) <= 1024  # Exercise the byte bound independently of the collection bound.
    observed = agent.safe_feedback({"ok": True, "regions": oversized})
    assert observed["ok"] is False
    assert observed["code"] == "INVALID_TOOL_FEEDBACK"
    assert "regions" not in observed
    collection_limit = agent.safe_feedback({"ok": True, "regions": [{}] * 1025})
    assert collection_limit["code"] == "INVALID_TOOL_FEEDBACK"

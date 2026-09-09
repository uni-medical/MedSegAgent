"""All registered producers use the same executor; synthetic masks and no model inference."""

import asyncio
import json

import nibabel as nib
import numpy as np
import pytest
from test_execution import Backend, make_input
from totalsegmentator.registry import TASKS, requires_license

from medsegagent import catalog, core
from medsegagent.execution import TaskExecution
from medsegagent.task_specs import TASK_SPECS

PUBLIC_TASKS = [
    task
    for task in TASKS
    if not requires_license(task) and task not in {"test", "total_highres_test"}
]
DISABLED_TASKS = [task for task in TASKS if task not in PUBLIC_TASKS]


@pytest.fixture
def prepared_weights(monkeypatch, tmp_path):
    from medsegagent import weights

    monkeypatch.setattr(weights, "require_weights", lambda *args, **kwargs: {"ready": True})
    monkeypatch.setenv("MEDSEGAGENT_LOCK_PATH", str(tmp_path / "inference.lock"))
    return weights


@pytest.mark.parametrize("task", PUBLIC_TASKS)
def test_every_public_producer_runs_through_shared_execution(tmp_path, monkeypatch, task):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    targets = list(catalog.native_labels(task).values())
    execution = TaskExecution(make_input(tmp_path), TASK_SPECS[task].modality, tmp_path / "out")
    result = asyncio.run(execution.call("segment", {"task": task, "targets": targets}))
    assert result["ok"] and execution.is_complete
    assert len(backend.calls) == 1 and backend.calls[0]["task"] == task
    exported = execution.export_result()
    assert {row["target"] for row in exported["outputs"]} == set(targets)
    assert all(row["task"] == task for row in exported["outputs"])
    assert all(row["task"] == task for row in exported["regions"])


@pytest.mark.parametrize("task", DISABLED_TASKS)
def test_disabled_producer_fails_before_image_reads_or_inference(tmp_path, monkeypatch, task):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    target = next(iter(catalog.native_labels(task).values()))
    execution = TaskExecution(
        tmp_path / "missing.nii.gz", TASK_SPECS[task].modality, tmp_path / "out"
    )
    result = asyncio.run(execution.call("segment", {"task": task, "targets": [target]}))
    assert not result["ok"] and result["code"] == "TASK_UNAVAILABLE"
    assert execution._frozen is None and not execution.has_outputs
    assert backend.calls == []


def test_capability_discovery_is_read_only_and_never_counts_as_output(tmp_path):
    execution = TaskExecution(tmp_path / "missing.nii.gz", output_dir=tmp_path / "out")
    result = asyncio.run(execution.call("get_capabilities", {}))
    assert result["ok"]
    assert len(result["capabilities"]["tasks"]) == len(PUBLIC_TASKS) == 33
    assert "excluded_task_counts" not in result["capabilities"]
    assert execution.modality is None and execution._frozen is None
    assert not execution.has_outputs and not execution.is_complete
    detail = asyncio.run(execution.call("get_capabilities", {"task": "lung_nodules"}))
    assert detail["ok"]
    assert execution.unresolved_failures == []


@pytest.mark.parametrize("task", DISABLED_TASKS)
def test_unavailable_capability_lookup_returns_no_model_details(tmp_path, task):
    execution = TaskExecution(tmp_path / "missing.nii.gz", output_dir=tmp_path / "out")
    result = asyncio.run(execution.call("get_capabilities", {"task": task}))
    assert not result["ok"] and result["code"] == "INVALID_ARGUMENTS"
    assert "capabilities" not in result
    assert task not in json.dumps(result)
    assert "LICENSE" not in json.dumps(result)
    assert execution._frozen is None and not execution.has_outputs


def test_producers_and_qualities_with_the_same_target_keep_distinct_outputs(tmp_path, monkeypatch):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")
    choices = [("total", "fast"), ("total_v3", "fast"), ("total_v3", "fastest")]

    async def scenario():
        for task, quality in choices:
            result = await execution.call(
                "segment", {"task": task, "targets": ["liver"], "quality": quality}
            )
            assert result["ok"]
            assert result["regions"][0]["task"] == task
            assert result["regions"][0]["quality"] == quality
        repeated = await execution.call(
            "segment", {"task": "total", "targets": ["liver"], "quality": "fast"}
        )
        assert repeated["cached"]
        outputs = execution.export_result()["outputs"]
        assert {(row["task"], row["target"], row["quality"]) for row in outputs} == {
            (task, "liver", quality) for task, quality in choices
        }
        assert len({row["region_id"] for row in outputs}) == len(choices)
        overlap = await execution.call(
            "inspect_artifact", {"region_ids": [row["region_id"] for row in outputs]}
        )
        assert overlap["ok"] and len(overlap["overlaps"]) == 3
        assert execution.is_complete

    asyncio.run(scenario())
    assert len(backend.calls) == len(choices)


def test_failed_producer_or_quality_is_not_resolved_by_a_different_output(tmp_path, monkeypatch):
    successful = Backend()

    async def backend(**kwargs):
        if kwargs["task"] == "total" and kwargs["speed"] == "standard":
            raise RuntimeError("synthetic producer failure")
        return await successful(**kwargs)

    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")

    async def scenario():
        failed = await execution.call(
            "segment", {"task": "total", "targets": ["liver"], "quality": "standard"}
        )
        assert not failed["ok"]
        assert (
            await execution.call(
                "segment", {"task": "total_v3", "targets": ["liver"], "quality": "standard"}
            )
        )["ok"]
        assert (
            await execution.call(
                "segment", {"task": "total", "targets": ["liver"], "quality": "fast"}
            )
        )["ok"]
        assert not execution.is_complete
        assert any(
            row.get("task") == "total" and row.get("quality") == "standard"
            for row in execution.unresolved_failures
        )

    asyncio.run(scenario())


def test_whole_lung_recipes_preserve_their_selected_producer(tmp_path, monkeypatch):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")

    async def scenario():
        for task in ("total", "total_v3"):
            response = await execution.call("segment", {"task": task, "targets": ["lungs"]})
            assert response["ok"]
        result = execution.export_result()
        assert {(row["task"], row["target"]) for row in result["outputs"]} == {
            ("total", "lungs"),
            ("total_v3", "lungs"),
        }
        regions = {row["id"]: row for row in result["regions"]}
        for output in result["outputs"]:
            region = regions[output["region_id"]]
            assert region["task"] == output["task"]
            assert region["voxels"] == 5
            assert len(region["provenance"]["source_region_ids"]) == 5
            assert all(
                regions[index]["task"] == output["task"]
                for index in region["provenance"]["source_region_ids"]
            )
        assert execution.is_complete

    asyncio.run(scenario())
    assert len(backend.calls) == 2


@pytest.mark.parametrize("task", PUBLIC_TASKS)
def test_commands_follow_each_producers_actual_speed_matrix(tmp_path, task):
    for speed in TASK_SPECS[task].speeds:
        command = core._build_command(
            task=task,
            input_path=tmp_path / "image.nii.gz",
            output_dir=tmp_path,
            targets=None,
            speed=speed,
        )
        assert ("--fast" in command) == (speed == "fast")
        assert ("--fastest" in command) == (speed == "fastest")
        assert "--ml" in command and "--save_lowres" not in command
    for speed in {"fast", "fastest", "standard"} - set(TASK_SPECS[task].speeds):
        with pytest.raises(core.SegmentationError) as error:
            core._build_command(
                task=task,
                input_path=tmp_path / "image.nii.gz",
                output_dir=tmp_path,
                targets=None,
                speed=speed,
            )
        assert error.value.code == "UNSUPPORTED_QUALITY"


@pytest.mark.parametrize(
    "task,target", [("lung_vessels", "lung_arteries"), ("total_v3", "vertebrae_L6")]
)
def test_non_roi_execution_filters_native_outputs_after_full_inference(
    tmp_path, monkeypatch, prepared_weights, task, target
):
    source = make_input(tmp_path)
    observed = []
    labels = core.task_labels(task)
    native_id = next(index for index, name in labels.items() if name == target)

    async def predict(command, *, output_dir, on_start, **kwargs):
        observed.append(command)
        reference = nib.load(source)
        data = np.zeros(reference.shape, dtype=np.uint8)
        for offset, index in enumerate(labels):
            data.flat[offset] = index
        nib.save(
            nib.Nifti1Image(data, reference.affine, reference.header),
            output_dir / "segmentation.nii.gz",
        )
        (output_dir / "run_report.json").write_text("{}")

    monkeypatch.setattr(core, "_run_command", predict)
    result = asyncio.run(
        core.segment(
            task=task,
            input_path=str(source),
            output_dir=str(tmp_path / "out"),
            targets=[target],
            speed="standard",
        )
    )
    assert len(observed) == 1 and "--roi_subset" not in observed[0]
    assert [(row["name"], row["source_id"]) for row in result["labels"]] == [(target, native_id)]
    assert set(np.unique(nib.load(result["segmentation_path"]).get_fdata())) <= {0, 1}
    assert result["segmentation_shape"] == [4, 5, 6]


def test_missing_weights_have_a_safe_error_and_can_be_prepared_then_retried(
    tmp_path, monkeypatch, prepared_weights
):
    def missing(*args, **kwargs):
        raise prepared_weights.WeightError("PRIVATE checkpoint path")

    monkeypatch.setattr(prepared_weights, "require_weights", missing)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")

    async def scenario():
        result = await execution.call("segment", {"targets": ["liver"]})
        assert not result["ok"] and result["code"] == "WEIGHTS_MISSING"
        assert "PRIVATE" not in json.dumps(result)
        assert not execution.has_outputs
        backend = Backend()
        monkeypatch.setattr(core, "segment", backend)
        recovered = await execution.call("segment", {"targets": ["liver"]})
        assert recovered["ok"] and execution.is_complete
        assert len(backend.calls) == 1

    asyncio.run(scenario())

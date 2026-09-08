"""Independent producers share the existing capacity and keep ordered verified state."""

import asyncio
import json
import os

import nibabel as nib
import numpy as np
import pytest
from test_execution import Backend, make_input
from test_gpu_scheduler import UUIDS, available, config

from medsegagent import core, gpu_scheduler, weights
from medsegagent.execution import TaskExecution


@pytest.fixture
def two_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSEGAGENT_DEVICE", "gpu")
    monkeypatch.setattr(weights, "require_weights", lambda *args, **kwargs: {"ready": True})
    monkeypatch.setattr(
        core,
        "GPUScheduler",
        lambda *, device: gpu_scheduler.GPUScheduler(
            config(tmp_path / "locks", max_concurrent=2), device=device, probe=available
        ),
    )


def test_independent_core_jobs_overlap_under_distinct_leases_and_keep_result_order(
    tmp_path, monkeypatch, two_devices
):
    active, used, finished = set(), set(), []
    peak = 0
    both_running = asyncio.Event()

    async def inference(command, *, output_dir, lock_fd, on_start, **kwargs):
        nonlocal peak
        task = command[command.index("--task") + 1]
        on_start(os.getpid())
        assert lock_fd.gpu_uuid not in active
        active.add(lock_fd.gpu_uuid)
        used.add(lock_fd.gpu_uuid)
        peak = max(peak, len(active))
        if peak == 2:
            both_running.set()
        try:
            await asyncio.wait_for(both_running.wait(), 2)
            # The second producer finishes first; identifiers still follow request order.
            if task == "total":
                await asyncio.sleep(0.03)
            source = nib.load(command[command.index("-i") + 1])
            data = np.zeros(source.shape, dtype=np.uint8)
            requested = {
                "total": ["liver", "spleen"],
                "lung_nodules": ["lung_nodules"],
                "liver_lesions": ["liver_lesions"],
            }[task]
            labels = {name: value for value, name in core.task_labels(task).items()}
            for i, name in enumerate(requested):
                data.flat[i] = labels[name]
            nib.save(
                nib.Nifti1Image(data, source.affine, source.header),
                output_dir / "segmentation.nii.gz",
            )
            (output_dir / "run_report.json").write_text("{}")
            finished.append(task)
        finally:
            active.remove(lock_fd.gpu_uuid)

    monkeypatch.setattr(core, "_run_command", inference)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")

    async def scenario():
        response = await execution.call(
            "segment", {"targets": ["liver", "spleen", "lung_nodules", "liver_lesions"]}
        )
        assert response["ok"]
        assert execution.is_complete
        count = len(finished)
        assert (await execution.call("segment", {"targets": ["liver", "lung_nodules"]}))["cached"]
        assert len(finished) == count
        regions = [
            row["region_id"]
            for row in response["regions"]
            if row["target"] in {"liver", "lung_nodules"}
        ]
        combined = await execution.call(
            "compose_masks", {"operation": "intersection", "region_ids": regions, "name": "overlap"}
        )
        assert combined["ok"] and combined["regions"][0]["voxels"] == 1

    asyncio.run(scenario())
    manifest = execution.export_result()
    assert peak == 2 and not active
    assert used == set(UUIDS[:2])
    assert finished[0] == "lung_nodules"
    assert [result["task"] for result in manifest["backend_results"]] == [
        "total",
        "lung_nodules",
        "liver_lesions",
    ]
    assert manifest["backend_results"][0]["targets"] == ["liver", "spleen"]
    assert [region["target"] for region in manifest["regions"]] == [
        "spleen",
        "liver",
        "lung_nodules",
        "liver_lesions",
        "overlap",
    ]
    assert len({row["id"] for row in manifest["artifacts"]}) == len(manifest["artifacts"])


@pytest.mark.parametrize("failure", ["inference", "invalid_output"])
def test_first_producer_failure_does_not_discard_later_success(
    tmp_path, monkeypatch, two_devices, failure
):
    normal = Backend()
    both_started = asyncio.Event()
    started = []

    async def inference(**kwargs):
        started.append(kwargs["task"])
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), 2)
        if kwargs["task"] == "total":
            if failure == "inference":
                raise RuntimeError("PRIVATE-BACKEND /private/checkpoint")
            return {"labels": [], "segmentation_path": "/private/nonexistent"}
        return await normal(**kwargs)

    monkeypatch.setattr(core, "segment", inference)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")

    async def scenario():
        response = await execution.call("segment", {"targets": ["liver", "lung_nodules"]})
        assert response["code"] == (
            "INFERENCE_FAILED" if failure == "inference" else "OUTPUT_INVALID"
        )
        assert [row["target"] for row in response["regions"]] == ["lung_nodules"]
        assert execution.has_outputs and not execution.is_complete
        assert "PRIVATE" not in json.dumps(response)
        assert (await execution.call("segment", {"targets": ["lung_nodules"]}))["cached"]
        assert (await execution.call("segment", {"targets": ["liver"]}))[
            "code"
        ] == "PREVIOUS_FAILURE"

    asyncio.run(scenario())
    assert started == ["total", "lung_nodules"]


def test_repeat_cancellation_stops_every_running_job_before_returning(
    tmp_path, monkeypatch, two_devices
):
    started, cleaned = set(), set()
    both_started, cleanup_started, allow_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked(**kwargs):
        task = kwargs["task"]
        started.add(task)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleaned.add(task)

    monkeypatch.setattr(core, "segment", blocked)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")

    async def scenario():
        worker = asyncio.create_task(
            execution.call("segment", {"targets": ["liver", "lung_nodules", "liver_lesions"]})
        )
        await asyncio.wait_for(both_started.wait(), 2)
        worker.cancel()
        await asyncio.wait_for(cleanup_started.wait(), 2)
        worker.cancel()
        await asyncio.sleep(0)
        assert not worker.done()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await worker
        assert cleaned == started == {"total", "lung_nodules"}
        assert not execution.has_outputs

    asyncio.run(scenario())


def test_cancellation_preserves_already_verified_partial_results(
    tmp_path, monkeypatch, two_devices
):
    backend = Backend()
    blocked = asyncio.Event()
    cleaned = []

    async def inference(**kwargs):
        if kwargs["task"] == "total":
            return await backend(**kwargs)
        try:
            blocked.set()
            await asyncio.Event().wait()
        finally:
            cleaned.append(kwargs["task"])

    monkeypatch.setattr(core, "segment", inference)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")
    saved = asyncio.Event()
    original_save = execution._save

    def save():
        original_save()
        if execution.has_outputs:
            saved.set()

    monkeypatch.setattr(execution, "_save", save)

    async def scenario():
        worker = asyncio.create_task(
            execution.call("segment", {"targets": ["liver", "lung_nodules"]})
        )
        await asyncio.wait_for(blocked.wait(), 2)
        await asyncio.wait_for(saved.wait(), 2)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        assert cleaned == ["lung_nodules"]
        assert execution.has_outputs and not execution.is_complete
        assert [row["target"] for row in execution.export_result()["outputs"]] == ["liver"]
        manifest = json.loads((execution._root / "execution.json").read_text())
        assert manifest["outputs"] == execution.export_result()["outputs"]

    asyncio.run(scenario())


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_single_device_capacity_never_overlaps_preprocessing(tmp_path, monkeypatch, device):
    monkeypatch.setenv("MEDSEGAGENT_DEVICE", device)
    backend = Backend()
    active, peak = 0, 0

    async def inference(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.005)
            return await backend(**kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(core, "segment", inference)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")
    response = asyncio.run(
        execution.call("segment", {"targets": ["liver", "lung_nodules", "liver_lesions"]})
    )
    assert response["ok"] and peak == 1 and not active


def test_explicit_total_and_total_v3_compare_same_targets_concurrently(
    tmp_path, monkeypatch, two_devices
):
    started, active = [], set()
    ready = asyncio.Event()

    async def inference(command, *, output_dir, lock_fd, on_start, **kwargs):
        task = command[command.index("--task") + 1]
        on_start(os.getpid())
        started.append(task)
        assert lock_fd.gpu_uuid not in active
        active.add(lock_fd.gpu_uuid)
        if len(active) == 2:
            ready.set()
        try:
            await asyncio.wait_for(ready.wait(), 2)
            source = nib.load(command[command.index("-i") + 1])
            data = np.zeros(source.shape, dtype=np.uint8)
            native = {name: value for value, name in core.task_labels(task).items()}
            data.flat[0] = native["liver"]
            data.flat[1] = native["spleen"]
            nib.save(
                nib.Nifti1Image(data, source.affine, source.header),
                output_dir / "segmentation.nii.gz",
            )
            (output_dir / "run_report.json").write_text("{}")
        finally:
            active.remove(lock_fd.gpu_uuid)

    monkeypatch.setattr(core, "_run_command", inference)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")
    request = {"task": ["total", "total_v3"], "targets": ["liver", "spleen"], "quality": "fast"}

    async def scenario():
        result = await execution.call("segment", request)
        assert result["ok"] and execution.is_complete
        assert result["requested_targets"] == ["liver", "spleen"]
        assert [(row["task"], row["target"]) for row in execution.export_result()["outputs"]] == [
            ("total", "liver"),
            ("total", "spleen"),
            ("total_v3", "liver"),
            ("total_v3", "spleen"),
        ]
        liver = [row["region_id"] for row in result["regions"] if row["target"] == "liver"]
        assert len(liver) == len(set(liver)) == 2
        combined = await execution.call(
            "compose_masks",
            {"operation": "intersection", "region_ids": liver, "name": "agreed_liver"},
        )
        assert combined["ok"] and combined["regions"][0]["voxels"] == 1
        assert (await execution.call("segment", request))["cached"]
        assert (await execution.call("segment", {**request, "task": "total_v3"}))["cached"]

    asyncio.run(scenario())
    assert sorted(started) == ["total", "total_v3"] and not active
    assert {row["gpu_uuid"] for row in execution.export_result()["backend_results"]} == set(
        UUIDS[:2]
    )
    assert all(
        row["targets"] == ["liver", "spleen"]
        for row in execution.export_result()["backend_results"]
    )


@pytest.mark.parametrize(
    "producer,code",
    [
        ([], "INVALID_ARGUMENTS"),
        (["total", "total"], "INVALID_ARGUMENTS"),
        (["total", None], "INVALID_ARGUMENTS"),
        (["total", []], "INVALID_ARGUMENTS"),
        (["total", "unknown"], "UNSUPPORTED_TASK"),
        (["total", "total_mr"], "UNSUPPORTED_TARGET"),
        (["total", "lung_nodules"], "UNSUPPORTED_TARGET"),
        (("total", "total_v3"), "UNSUPPORTED_TASK"),
    ],
)
def test_invalid_producer_groups_do_no_input_reads_or_work(tmp_path, monkeypatch, producer, code):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(tmp_path / "missing.nii.gz", "CT", tmp_path / "out")
    result = asyncio.run(execution.call("segment", {"task": producer, "targets": ["liver"]}))
    assert result["code"] == code
    assert execution._frozen is None and not execution._wanted and not backend.calls


def test_producer_group_is_bounded_and_supersedes_has_one_replacement(tmp_path, monkeypatch):
    from medsegagent import catalog
    from medsegagent.tool_definitions import MAX_SEGMENT_PRODUCERS

    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(tmp_path / "missing.nii.gz", "CT", tmp_path / "out")

    async def scenario():
        oversized = list(catalog.public_task_names("CT"))[: MAX_SEGMENT_PRODUCERS + 1]
        result = await execution.call("segment", {"task": oversized, "targets": ["liver"]})
        assert result["code"] == "INVALID_ARGUMENTS"
        result = await execution.call(
            "segment",
            {
                "task": ["total", "total_v3"],
                "targets": ["liver"],
                "supersedes": [{"task": "total", "target": "liver", "quality": "standard"}],
            },
        )
        assert result["code"] == "INVALID_SUPERSEDES"
        assert execution._frozen is None and not execution._wanted and not backend.calls

    asyncio.run(scenario())


def test_single_producer_array_is_compatible_with_recovery(tmp_path, monkeypatch):
    backend = Backend(fail="total")
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")

    async def scenario():
        assert not (await execution.call("segment", {"task": "total", "targets": ["liver"]}))["ok"]
        recovered = await execution.call(
            "segment",
            {
                "task": ["total_v3"],
                "targets": ["liver"],
                "supersedes": [{"task": "total", "target": "liver", "quality": "fast"}],
            },
        )
        assert recovered["ok"] and execution.is_complete
        assert recovered["resolved_attempts"][0]["task"] == "total"
        assert recovered["regions"][0]["task"] == "total_v3"

    asyncio.run(scenario())


def test_successful_and_failed_core_timings_are_private_and_keep_producer_order(
    tmp_path, monkeypatch, two_devices
):
    backend = Backend()
    ready = asyncio.Event()

    async def inference(**kwargs):
        if kwargs["task"] == "total":
            await ready.wait()
            error = core.SegmentationError("PRIVATE failed")
            error.timings_seconds = {"device_wait": 0.5, "subprocess": 2.0}
            error.inference_engine = "sequential"
            raise error
        result = await backend(**kwargs)
        ready.set()
        return {
            **result,
            "timings_seconds": {"device_wait": 0.1, "subprocess": 1.0},
            "inference_engine": "sequential",
        }

    monkeypatch.setattr(core, "segment", inference)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")
    response = asyncio.run(
        execution.call("segment", {"task": ["total", "total_v3"], "targets": ["liver"]})
    )
    assert not response["ok"]
    timings = execution.export_result()["inference_timings"]
    assert [(row["task"], row["status"]) for row in timings] == [
        ("total", "failed"),
        ("total_v3", "completed"),
    ]
    assert [row["seconds"]["subprocess"] for row in timings] == [2.0, 1.0]
    assert all(row["quality"] == "fast" and row["engine"] == "sequential" for row in timings)
    assert "inference_timings" not in response and "timings_seconds" not in json.dumps(response)
    assert (
        json.loads((execution._root / "execution.json").read_text())["inference_timings"] == timings
    )


def test_cancelled_core_timing_survives_cleanup_and_is_saved(tmp_path, monkeypatch):
    started = asyncio.Event()

    async def inference(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            exc.timings_seconds = {"subprocess": 1.5}
            exc.inference_engine = "sequential"
            raise

    monkeypatch.setattr(core, "segment", inference)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")

    async def scenario():
        worker = asyncio.create_task(execution.call("segment", {"targets": ["liver"]}))
        await started.wait()
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

    asyncio.run(scenario())
    timings = execution.export_result()["inference_timings"]
    assert timings == [
        {
            "task": "total",
            "quality": "fast",
            "engine": "sequential",
            "status": "cancelled",
            "seconds": {"subprocess": 1.5},
        }
    ]
    assert (
        json.loads((execution._root / "execution.json").read_text())["inference_timings"] == timings
    )

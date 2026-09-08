from __future__ import annotations

import asyncio
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from medsegagent import catalog, core
from medsegagent.execution import TaskExecution


def make_input(tmp_path, *, affine=None):
    path = tmp_path / "synthetic.nii.gz"
    image = nib.Nifti1Image(
        np.zeros((4, 5, 6), dtype=np.float32),
        np.diag([2.0, 3.0, 4.0, 1.0]) if affine is None else affine,
    )
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


class Backend:
    def __init__(self, positions=None, fail=None, bad_geometry=False):
        self.positions = positions or {}
        self.fail = fail
        self.bad_geometry = bad_geometry
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["task"] == self.fail:
            raise RuntimeError("PRIVATE-BACKEND-DETAIL /secret/patient.nii.gz")
        source = nib.load(kwargs["input_path"])
        data = np.zeros(source.shape, dtype=np.uint8)
        native = {name: index for index, name in catalog.native_labels(kwargs["task"]).items()}
        labels = []
        for label_id, target in enumerate(kwargs["targets"], 1):
            for offset in self.positions.get(target, [label_id - 1]):
                data.flat[offset] = label_id
            labels.append({"id": label_id, "source_id": native[target], "name": target})
        run = Path(kwargs["output_dir"]) / f"stub-{len(self.calls)}"
        run.mkdir(parents=True)
        path = run / "segmentation.nii.gz"
        affine = source.affine.copy()
        if self.bad_geometry:
            affine[0, 3] += 1
        image = nib.Nifti1Image(data, affine, source.header)
        image.header.set_data_dtype(np.uint8)
        nib.save(image, path)
        return {
            "segmentation_path": str(path),
            "task": kwargs["task"],
            "targets": kwargs["targets"],
            "labels": labels,
            "private_report": "/private/run.log",
            "runtime_seconds": 1.0,
        }


def setup_execution(tmp_path, monkeypatch, backend=None):
    backend = backend or Backend()
    monkeypatch.setattr(core, "segment", backend)
    source = make_input(tmp_path)
    return TaskExecution(source, "CT", tmp_path / "out"), backend, source


def output_region(execution, target):
    result = execution.export_result()
    output = next(row for row in result["outputs"] if row["target"] == target)
    return next(row for row in result["regions"] if row["id"] == output["region_id"])


@pytest.mark.parametrize(
    "args,code",
    [
        ({"targets": ["lungs", "not_a_target"]}, "UNSUPPORTED_TARGET"),
        ({"targets": ["lungs", "lung_nodules"], "quality": "fast"}, "UNSUPPORTED_QUALITY"),
        ({"targets": ["lungs"], "input_path": "/unexpected"}, "INVALID_ARGUMENTS"),
        ({"targets": []}, "UNSUPPORTED_TARGET"),
    ],
)
def test_whole_call_validates_before_any_inference(tmp_path, monkeypatch, args, code):
    execution, backend, _ = setup_execution(tmp_path, monkeypatch)
    response = asyncio.run(execution.call("segment", args))
    assert response["ok"] is False
    assert response["code"] == code
    assert backend.calls == []
    assert execution.has_outputs is False
    assert not list((tmp_path / "out").rglob("image.nii.gz"))


def test_lungs_expand_once_and_union_is_an_independent_verified_mask(tmp_path, monkeypatch):
    execution, backend, _ = setup_execution(tmp_path, monkeypatch)
    response = asyncio.run(execution.call("segment", {"targets": ["lungs", "liver"]}))
    assert response["ok"]
    assert len(backend.calls) == 1
    assert backend.calls[0]["task"] == "total"
    assert set(backend.calls[0]["targets"]) == {
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
        "liver",
    }
    result = execution.export_result()
    assert len(result["artifacts"]) == 2  # Shared 6-class native mask and binary lungs.
    lungs = output_region(execution, "lungs")
    assert lungs["voxels"] == 5
    assert lungs["volume_ml"] == pytest.approx(5 * 24 / 1000)
    assert len(lungs["provenance"]["source_region_ids"]) == 5
    artifact = next(row for row in result["artifacts"] if row["id"] == lungs["artifact_id"])
    assert set(np.unique(nib.load(artifact["path"]).get_fdata())) == {0, 1}
    assert len(result["backend_results"]) == 1
    assert execution.is_complete
    safe = json.dumps(response)
    assert str(tmp_path) not in safe
    assert "private_report" not in safe
    assert "sha256" not in safe


def test_native_targets_share_labelmap_and_zero_counts_are_covered(tmp_path, monkeypatch):
    backend = Backend(positions={"liver": [], "spleen": [3]})
    execution, backend, _ = setup_execution(tmp_path, monkeypatch, backend)

    async def scenario():
        assert (await execution.call("segment", {"targets": ["liver", "spleen"]}))["ok"]
        second = await execution.call("segment", {"targets": ["liver"]})
        assert second["cached"]

    asyncio.run(scenario())
    assert len(backend.calls) == 1
    assert len(execution.export_result()["artifacts"]) == 1
    assert output_region(execution, "liver")["voxels"] == 0
    assert (
        output_region(execution, "liver")["artifact_id"]
        == output_region(execution, "spleen")["artifact_id"]
    )
    assert execution.is_complete


def test_later_targets_reuse_coverage_and_only_infer_missing_labels(tmp_path, monkeypatch):
    execution, backend, _ = setup_execution(tmp_path, monkeypatch)

    async def scenario():
        assert (await execution.call("segment", {"targets": ["lungs"]}))["ok"]
        assert (await execution.call("segment", {"targets": ["lung_left"]}))["cached"]
        assert (await execution.call("segment", {"targets": ["lungs", "spleen"]}))["ok"]

    asyncio.run(scenario())
    assert len(backend.calls) == 2
    assert backend.calls[1]["targets"] == ["spleen"]
    assert output_region(execution, "lung_left")["voxels"] == 2


def test_quality_has_separate_cache_and_does_not_repeat_unchanged_failure(tmp_path, monkeypatch):
    execution, backend, _ = setup_execution(tmp_path, monkeypatch)

    async def scenario():
        await execution.call("segment", {"targets": ["liver"]})
        await execution.call("segment", {"targets": ["liver"], "quality": "standard"})
        await execution.call("segment", {"targets": ["liver"], "quality": "standard"})

    asyncio.run(scenario())
    assert [row["speed"] for row in backend.calls] == ["fast", "standard"]


def test_failure_preserves_success_allows_independent_work_and_blocks_repetition(
    tmp_path, monkeypatch
):
    execution, backend, _ = setup_execution(tmp_path, monkeypatch, Backend(fail="lung_nodules"))

    async def scenario():
        first = await execution.call(
            "segment", {"targets": ["lungs", "lung_nodules", "liver_lesions"]}
        )
        assert first["code"] == "INFERENCE_FAILED"
        assert len(backend.calls) == 3
        assert execution.has_outputs
        assert not execution.is_complete
        assert "PRIVATE-BACKEND" not in json.dumps(first)
        second = await execution.call("segment", {"targets": ["lung_nodules"]})
        assert second["code"] == "PREVIOUS_FAILURE"
        assert len(backend.calls) == 3
        third = await execution.call("segment", {"targets": ["liver_lesions"]})
        assert third["ok"] and third["cached"]
        assert not execution.is_complete

    asyncio.run(scenario())
    assert len(backend.calls) == 3
    assert {row["target"] for row in execution.export_result()["outputs"]} == {
        "lungs",
        "liver_lesions",
    }


def test_cross_backend_overlap_and_set_operations_preserve_original_masks(tmp_path, monkeypatch):
    positions = {"liver": [0, 1, 2], "liver_lesions": [1, 2, 3]}
    execution, backend, _ = setup_execution(tmp_path, monkeypatch, Backend(positions))

    async def scenario():
        assert (await execution.call("segment", {"targets": ["liver", "liver_lesions"]}))["ok"]
        ids = [output_region(execution, target)["id"] for target in ("liver", "liver_lesions")]
        inspection = await execution.call("inspect_artifact", {"region_ids": ids})
        assert inspection["overlaps"][0]["voxels"] == 2
        for operation, count in (("union", 4), ("intersection", 2), ("difference", 1)):
            response = await execution.call(
                "compose_masks",
                {
                    "operation": operation,
                    "region_ids": ids,
                    "name": operation,
                },
            )
            assert response["ok"]
            assert response["regions"][0]["voxels"] == count
        assert output_region(execution, "liver")["voxels"] == 3
        assert output_region(execution, "liver_lesions")["voxels"] == 3

    asyncio.run(scenario())
    assert len(backend.calls) == 2
    assert len(execution.export_result()["artifacts"]) == 5


def test_identical_composition_is_reused_and_empty_difference_is_valid(tmp_path, monkeypatch):
    execution, backend, _ = setup_execution(
        tmp_path, monkeypatch, Backend({"liver": [], "spleen": []})
    )

    async def scenario():
        await execution.call("segment", {"targets": ["liver", "spleen"]})
        ids = [output_region(execution, name)["id"] for name in ("liver", "spleen")]
        arguments = {"operation": "difference", "region_ids": ids, "name": "empty"}
        first = await execution.call("compose_masks", arguments)
        second = await execution.call("compose_masks", arguments)
        assert first["regions"][0]["region_id"] == second["regions"][0]["region_id"]
        assert second["regions"][0]["empty"]

    asyncio.run(scenario())
    assert len(backend.calls) == 1
    assert len(execution.export_result()["artifacts"]) == 2


def test_composition_cannot_redefine_native_or_existing_derived_objects(tmp_path, monkeypatch):
    execution, _, _ = setup_execution(tmp_path, monkeypatch)

    async def scenario():
        await execution.call("segment", {"targets": ["liver", "spleen"]})
        liver = output_region(execution, "liver")
        ids = [liver["id"], output_region(execution, "spleen")["id"]]
        native = await execution.call(
            "compose_masks",
            {
                "operation": "union",
                "region_ids": ids,
                "name": "liver",
            },
        )
        assert native["code"] == "NAME_CONFLICT"
        assert output_region(execution, "liver")["id"] == liver["id"]
        assert len(execution.export_result()["artifacts"]) == 1
        first = await execution.call(
            "compose_masks",
            {
                "operation": "union",
                "region_ids": ids,
                "name": "combined",
            },
        )
        assert first["ok"]
        original = output_region(execution, "combined")
        conflict = await execution.call(
            "compose_masks",
            {
                "operation": "intersection",
                "region_ids": ids,
                "name": "combined",
            },
        )
        assert conflict["code"] == "NAME_CONFLICT"
        assert output_region(execution, "combined")["id"] == original["id"]
        assert len(execution.export_result()["artifacts"]) == 2
        repeated = await execution.call(
            "compose_masks",
            {
                "operation": "union",
                "region_ids": list(reversed(ids)),
                "name": "combined",
            },
        )
        assert repeated["ok"]
        assert execution.is_complete

    asyncio.run(scenario())


def test_backend_geometry_is_checked_even_for_stubbed_inference(tmp_path, monkeypatch):
    execution, _, _ = setup_execution(tmp_path, monkeypatch, Backend(bad_geometry=True))
    response = asyncio.run(execution.call("segment", {"targets": ["liver"]}))
    assert response["code"] == "OUTPUT_INVALID"
    assert not execution.has_outputs
    assert execution.export_result()["artifacts"] == []


def test_source_is_frozen_across_subsequent_actions(tmp_path, monkeypatch):
    execution, backend, source = setup_execution(tmp_path, monkeypatch)

    async def scenario():
        await execution.call("segment", {"targets": ["liver"]})
        changed = nib.Nifti1Image(np.ones((4, 5, 6)), np.diag([2.0, 3.0, 4.0, 1.0]))
        nib.save(changed, source)
        await execution.call("segment", {"targets": ["spleen"]})

    asyncio.run(scenario())
    assert backend.calls[0]["input_path"] == backend.calls[1]["input_path"]
    assert backend.calls[0]["input_path"] != str(source)
    assert not np.any(nib.load(backend.calls[1]["input_path"]).get_fdata())


def test_regions_are_scoped_and_tampering_is_detected(tmp_path, monkeypatch):
    execution, _, _ = setup_execution(tmp_path, monkeypatch)

    async def scenario():
        await execution.call("segment", {"targets": ["liver"]})
        region = output_region(execution, "liver")
        invalid = await execution.call("inspect_artifact", {"region_ids": ["region-other-task-1"]})
        assert invalid["code"] == "UNKNOWN_REGION"
        repaired = await execution.call("inspect_artifact", {"region_ids": [region["id"]]})
        assert repaired["ok"]
        artifact = execution.export_result()["artifacts"][0]
        Path(artifact["path"]).write_bytes(b"changed")
        tampered = await execution.call("inspect_artifact", {"region_ids": [region["id"]]})
        assert tampered["code"] == "ARTIFACT_CHANGED"
        cached = await execution.call("segment", {"targets": ["liver"]})
        assert cached["code"] == "ARTIFACT_CHANGED"

    asyncio.run(scenario())


def test_invalid_arguments_can_be_corrected_without_losing_other_outputs(tmp_path, monkeypatch):
    execution, backend, _ = setup_execution(tmp_path, monkeypatch)

    async def scenario():
        invalid = await execution.call("segment", {"targets": ["liverr"]})
        assert not invalid["ok"]
        assert not execution.is_complete
        repaired = await execution.call("segment", {"targets": ["liver"]})
        assert repaired["ok"]
        assert execution.is_complete

    asyncio.run(scenario())
    assert len(backend.calls) == 1


def test_artifact_ownership_is_checked_even_when_geometry_matches(tmp_path, monkeypatch):
    source = make_input(tmp_path)
    external = tmp_path / "outside.nii.gz"
    image = nib.load(source)
    nib.save(nib.Nifti1Image(np.ones(image.shape, dtype=np.uint8), image.affine), external)

    async def outside(**kwargs):
        return {"segmentation_path": str(external), "labels": [{"id": 1, "name": "liver"}]}

    monkeypatch.setattr(core, "segment", outside)
    execution = TaskExecution(source, "CT", tmp_path / "out")
    response = asyncio.run(execution.call("segment", {"targets": ["liver"]}))
    assert response["code"] == "OUTPUT_INVALID"
    assert not execution.has_outputs


def test_dicom_is_frozen_once_and_following_jobs_reuse_validated_conversion(tmp_path, monkeypatch):
    from test_core import dicom_at

    source = dicom_at(tmp_path / "dicom")
    backend = Backend()
    received = []
    converted = []

    async def convert_then_segment(**kwargs):
        input_path = Path(kwargs["input_path"])
        received.append(input_path)
        if input_path.is_dir():
            assert input_path != source
            assert {path.name for path in input_path.iterdir()} == {
                "000000.dcm",
                "000001.dcm",
                "000002.dcm",
            }
            directory = Path(kwargs["output_dir"]) / "converted"
            directory.mkdir(parents=True)
            converted.append(make_input(directory))
            result = await backend(**{**kwargs, "input_path": str(converted[0])})
            result["converted_input_path"] = str(converted[0])
            return result
        return await backend(**kwargs)

    monkeypatch.setattr(core, "segment", convert_then_segment)
    execution = TaskExecution(source, "CT", tmp_path / "out")
    response = asyncio.run(execution.call("segment", {"targets": ["lungs", "lung_nodules"]}))
    assert response["ok"]
    assert len(converted) == 1
    assert received[0].is_dir()
    assert received[1] == converted[0]
    assert execution.is_complete


def test_cancellation_propagates_and_waits_for_backend_cleanup(tmp_path, monkeypatch):
    started = asyncio.Event()
    cleaned = []

    async def blocked(**kwargs):
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    execution, _, _ = setup_execution(tmp_path, monkeypatch, blocked)

    async def scenario():
        worker = asyncio.create_task(execution.call("segment", {"targets": ["liver"]}))
        await started.wait()
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        assert cleaned == [True]
        assert not execution.has_outputs

    asyncio.run(scenario())


def test_execution_forwards_inference_start_callback(tmp_path, monkeypatch):
    source = make_input(tmp_path)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    callback = lambda: None
    execution = TaskExecution(source, "CT", tmp_path / "runs", on_inference_start=callback)
    result = asyncio.run(execution.call("segment", {"targets": ["liver"]}))
    assert result["ok"]
    assert backend.calls[0]["on_inference_start"] is callback


def test_example_hint_returns_evidence_without_binding_agent_choice(tmp_path, monkeypatch):
    source = make_input(tmp_path)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(source, None, tmp_path / "runs", example_modality_hint="MR")
    observed = asyncio.run(execution.call("detect_modality", {}))
    assert observed["ok"]
    assert execution.modality is None
    assert execution.modality_detection["modality"] == "MR"
    assert execution.modality_detection["source"] == "example_manifest"
    chosen = asyncio.run(execution.call("segment", {"targets": ["liver"], "modality": "CT"}))
    assert chosen["ok"]
    assert execution.modality == "CT"
    assert backend.calls[0]["task"] == "total"


@pytest.mark.parametrize("hint", ["US", {}, ["CT"]])
def test_invalid_example_hint_is_rejected(tmp_path, hint):
    with pytest.raises(ValueError, match="example modality hint"):
        TaskExecution(tmp_path / "image.nii", example_modality_hint=hint)


def test_composition_rejects_unpublishable_label_before_deriving(tmp_path, monkeypatch):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "runs")
    observed = asyncio.run(execution.call("segment", {"targets": ["liver"]}))
    source = next(row for row in observed["regions"] if row["target"] == "liver")
    result = asyncio.run(
        execution.call(
            "compose_masks",
            {
                "operation": "union",
                "region_ids": [source["region_id"]],
                "name": "liver\u202ereview",
            },
        )
    )
    assert result["ok"] is False
    assert result["code"] == "INVALID_ARGUMENTS"
    assert len(execution.export_result()["outputs"]) == 1

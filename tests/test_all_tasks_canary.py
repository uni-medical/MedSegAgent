"""Acceptance evidence must distinguish primary inference from crop shortcuts."""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "all_tasks_canary", Path(__file__).parents[1] / "ops/all_tasks_canary.py"
)
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)


@pytest.mark.parametrize(
    "task,completions,starts",
    [
        ("total", 2, 6),
        ("total_v3", 1, 5),
        ("total_mr", 2, 3),
        ("teeth", 3, 3),
        ("vertebrae_pp_refined", 2, 2),
        ("brain_aneurysm", 1, 1),
    ],
)
def test_all_native_labels_require_every_inference_stage(task, completions, starts):
    targets = list(canary.catalog.native_labels(task).values())
    result = canary.prediction_contract(task, "standard", targets)
    assert result["expected_prediction_completions"] == completions
    assert result["expected_model_starts"] == starts


def test_complete_crop_does_not_prove_primary_inference(tmp_path):
    log = tmp_path / "process.log"
    contract = canary.prediction_contract(
        "cerebral_bleed", "standard", ["intracerebral_hemorrhage"]
    )
    log.write_text("Predicting...\n  Predicted in 3.23s\n")
    assert not canary.inspect_log(log, contract)["real_inference_verified"]
    log.write_text(log.read_text() + "INFO: Crop is empty. Returning empty segmentation.\n")
    assert canary.inspect_log(log, contract)["empty_crop_shortcut"]
    log.write_text("Predicting...\n  Predicted in 3.23s\n" * 2)
    assert canary.inspect_log(log, contract)["real_inference_verified"]


def test_missing_anatomy_part_rejects_completion_marker(tmp_path):
    log = tmp_path / "process.log"
    contract = canary.prediction_contract("total_v3", "standard", None)
    log.write_text(
        "".join(f"Predicting part {i} of 5 ...\n" for i in range(1, 5)) + "  Predicted in 12.00s\n"
    )
    assert not canary.inspect_log(log, contract)["real_inference_verified"]
    log.write_text(log.read_text() + "Predicting part 5 of 5 ...\n")
    assert canary.inspect_log(log, contract)["real_inference_verified"]


def test_resume_requires_intact_artifact_and_log(tmp_path):
    paths = [tmp_path / "process.log", tmp_path / "segmentation.nii.gz"]
    for path in paths:
        path.write_bytes(b"verified evidence")
    report = {
        "status": "success",
        "fingerprint": "same-input-and-runner",
        "real_inference_verified": True,
        "geometry_and_labels_verified": True,
        "log_checks": [{"path": str(paths[0]), "sha256": canary.digest(paths[0])}],
        "artifact_checks": [{"path": str(paths[1]), "sha256": canary.digest(paths[1])}],
    }
    assert canary.reusable(report, "same-input-and-runner")
    assert not canary.reusable(report, "changed-input")
    paths[1].write_bytes(b"changed")
    assert not canary.reusable(report, "same-input-and-runner")


def test_geometry_and_nonfinite_labels_are_rejected(tmp_path):
    source, mask = tmp_path / "input.nii.gz", tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), source)
    affine = np.eye(4)
    affine[0, 3] = 4
    values = np.zeros((2, 2, 2))
    values[0, 0, 0] = 2
    nib.save(nib.Nifti1Image(values, affine), mask)
    result = canary.inspect_artifact(mask, source, [{"id": 1}])
    assert not result["geometry_valid"]
    assert not result["labels_valid"]

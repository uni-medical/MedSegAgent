"""Agent-requested modality detection on synthetic inputs; no real inference/provider calls."""

import asyncio
import json
import threading
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from test_core import dicom_at
from test_execution import Backend, make_input

from medsegagent import core, modality
from medsegagent.execution import TaskExecution


def classification(value="CT", *, status="detected"):
    report = {
        "modality": value if status == "detected" else None,
        "source": "totalseg_intensity",
        "status": status,
        "supported": status == "detected" and value in {"CT", "MR"},
        "limitations": ["INTENSITY_ONLY_NOT_MODALITY_PROOF"],
    }
    if status == "detected":
        report.update(candidate_modality=value, vote_agreement=1.0)
    return report


def test_unbound_segmentation_requires_agent_modality_without_reading_or_inference(
    tmp_path, monkeypatch
):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), output_dir=tmp_path / "out")
    response = asyncio.run(execution.call("segment", {"targets": ["liver"]}))
    assert response["code"] == "MODALITY_REQUIRED"
    assert execution.modality is None
    assert execution._frozen is None
    assert not execution.has_outputs
    assert backend.calls == []


@pytest.mark.parametrize("detected", ["CT", "MR"])
def test_detection_is_evidence_and_agent_choice_binds_while_reusing_frozen_features(
    tmp_path, monkeypatch, detected
):
    calls, reads, all_reads = [], [], []
    original_inspect = core._inspect_nifti

    def inspect(path, **kwargs):
        all_reads.append(str(path))
        if kwargs.get("collect_intensity_stats"):
            reads.append(str(path))
        return original_inspect(path, **kwargs)

    def classify(features):
        calls.append(list(features))
        return {
            **classification(detected),
            "path": "/private/model",
            "intensity_features": features,
            "intensity_statistics": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "patient_identifier": "PRIVATE",
            },
        }

    monkeypatch.setattr(core, "_inspect_nifti", inspect)
    monkeypatch.setattr(modality, "classify_features", classify)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    source = make_input(tmp_path)
    execution = TaskExecution(source, output_dir=tmp_path / "out")

    async def scenario():
        await execution.call("segment", {"targets": ["liver"]})
        detected_reply = await execution.call("detect_modality", {})
        assert detected_reply["ok"]
        assert detected_reply["status"] == "completed"
        assert execution.modality is None
        assert not execution.has_outputs  # Detection alone is not task completion.
        assert not execution.is_complete
        assert execution.unresolved_failures[0]["code"] == "MODALITY_REQUIRED"
        assert detected_reply["modality_detection"]["declared_modality"] is None
        frozen = execution._frozen
        assert all_reads == [str(frozen)]
        nib.save(nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4)), source)
        repeated = await execution.call("detect_modality", {})
        assert repeated["cached"]
        assert repeated["modality_detection"]["cached"]
        assert all_reads == [str(frozen)]
        segmented = await execution.call("segment", {"targets": ["liver"], "modality": detected})
        assert segmented["ok"]
        assert execution.modality == detected
        assert execution.is_complete
        assert backend.calls[0]["input_path"] == str(frozen)
        assert not np.any(nib.load(frozen).get_fdata())
        encoded = json.dumps(detected_reply)
        assert "/private/model" not in encoded and str(tmp_path) not in encoded
        assert "intensity_features" not in encoded
        assert "patient_identifier" not in encoded
        assert detected_reply["modality_detection"]["intensity_statistics"] == {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
        assert "intensity_features" not in json.dumps(execution.export_result())

    asyncio.run(scenario())
    assert len(calls) == len(reads) == 1
    assert len(calls[0]) == 4
    assert len(backend.calls) == 1


@pytest.mark.parametrize("action", ["detect_modality", "segment"])
@pytest.mark.parametrize("limit_kind", ["compressed", "uncompressed"])
def test_oversized_input_never_reaches_detection_or_inference(
    tmp_path, monkeypatch, action, limit_kind
):
    source = make_input(tmp_path)
    inspected, classified = [], []
    original_inspect = core._inspect_nifti

    def inspect(path, **kwargs):
        inspected.append(str(path))
        return original_inspect(path, **kwargs)

    def classify(features):
        classified.append(features)
        return classification()

    monkeypatch.setattr(core, "_inspect_nifti", inspect)
    monkeypatch.setattr(modality, "classify_features", classify)
    variable, size = (
        ("MEDSEGAGENT_MAX_INPUT_BYTES", "10")
        if limit_kind == "compressed"
        else ("MEDSEGAGENT_MAX_UNCOMPRESSED_BYTES", "360")
    )
    monkeypatch.setenv(variable, size)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(source, "CT" if action == "segment" else None, tmp_path / "out")
    arguments = {"targets": ["liver"]} if action == "segment" else {}
    result = asyncio.run(execution.call(action, arguments))
    if action == "segment":
        assert not result["ok"] and result["code"] == "INPUT_INVALID"
    else:
        assert result["ok"] and result["status"] == "completed"
        assert result["modality_detection"]["status"] == "uncertain"
        assert execution.unresolved_failures == []
    assert len(inspected) == (0 if limit_kind == "compressed" else 1)
    assert backend.calls == classified == []
    assert execution._frozen is None
    assert not execution.has_outputs
    assert not list(execution._root.rglob("*.tmp"))


@pytest.mark.parametrize("declared", ["CT", "MR"])
def test_declared_modality_is_returned_without_reading_input_or_classifying(
    tmp_path, monkeypatch, declared
):
    def forbidden(*args, **kwargs):
        raise AssertionError("A declaration must not trigger image reads or modality detection.")

    monkeypatch.setattr(core, "_inspect_nifti", forbidden)
    monkeypatch.setattr(modality, "classify_features", forbidden)
    monkeypatch.setattr(modality, "probe_dicom", forbidden)
    monkeypatch.setattr(modality, "probe_nifti_sidecar", forbidden)
    execution = TaskExecution(tmp_path / "missing.nii.gz", declared, tmp_path / "out")

    async def scenario():
        result = await execution.call("detect_modality", {})
        assert result["ok"] and result["status"] == "completed"
        assert result["modality_detection"] == {
            "modality": declared,
            "source": "user_declaration",
            "status": "provided",
            "supported": True,
            "limitations": [],
            "declared_modality": declared,
            "cached": False,
        }
        assert execution._frozen is None
        assert not execution.has_outputs
        assert execution.unresolved_failures == []
        assert (await execution.call("detect_modality", {}))["cached"]

    asyncio.run(scenario())


@pytest.mark.parametrize("declared", [None, "CT"])
def test_agent_cannot_replace_an_existing_declaration_or_execution_modality(
    tmp_path, monkeypatch, declared
):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), declared, tmp_path / "out")

    async def scenario():
        if declared is None:
            assert (await execution.call("segment", {"targets": ["liver"], "modality": "CT"}))["ok"]
        reply = await execution.call("segment", {"targets": ["spleen"], "modality": "MR"})
        assert reply["code"] == "MODALITY_CONFLICT"
        assert execution.modality == "CT"
        assert len(backend.calls) == int(declared is None)
        assert (await execution.call("segment", {"targets": ["liver"]}))["ok"]
        assert execution.is_complete

    asyncio.run(scenario())


def test_example_manifest_modality_preserves_its_source_without_reading_image(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Verified example modality does not trigger image observation.")

    monkeypatch.setattr(core, "_inspect_nifti", forbidden)
    monkeypatch.setattr(modality, "classify_features", forbidden)
    monkeypatch.setattr(modality, "probe_dicom", forbidden)
    monkeypatch.setattr(modality, "probe_nifti_sidecar", forbidden)
    execution = TaskExecution(
        tmp_path / "missing.nii.gz", "MR", tmp_path / "out", modality_source="example_manifest"
    )
    result = asyncio.run(execution.call("detect_modality", {}))
    assert result["ok"] and result["status"] == "completed"
    assert result["modality_detection"]["source"] == "example_manifest"
    assert result["modality_detection"]["modality"] == "MR"
    assert result["modality_detection"]["status"] == "provided"
    assert execution._frozen is None
    assert execution.unresolved_failures == []


def test_uncertain_detection_does_not_block_the_agents_modality_choice(tmp_path, monkeypatch):
    calls = []

    def classify(features):
        calls.append(True)
        return classification(status="uncertain")

    monkeypatch.setattr(modality, "classify_features", classify)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), output_dir=tmp_path / "out")

    async def scenario():
        result = await execution.call("detect_modality", {})
        assert result["ok"] and result["status"] == "completed"
        assert result["modality_detection"]["status"] == "uncertain"
        assert "candidate_modality" not in result["modality_detection"]
        assert "vote_agreement" not in result["modality_detection"]
        assert execution.modality is None
        assert execution.unresolved_failures == []
        assert not execution.is_complete
        assert (await execution.call("segment", {"targets": ["liver"], "modality": "MR"}))["ok"]
        assert execution.modality == "MR"
        assert execution.is_complete
        assert (await execution.call("detect_modality", {}))["cached"]
        assert execution.is_complete

    asyncio.run(scenario())
    assert len(calls) == 1
    assert len(backend.calls) == 1 and backend.calls[0]["task"] == "total_mr"


def test_dicom_metadata_is_observed_without_binding_or_intensity_classification(
    tmp_path, monkeypatch
):
    source = dicom_at(tmp_path / "dicom")

    def forbidden(_features):
        raise AssertionError("DICOM detection must not run the intensity classifier.")

    monkeypatch.setattr(modality, "classify_features", forbidden)
    execution = TaskExecution(source, output_dir=tmp_path / "out")
    result = asyncio.run(execution.call("detect_modality", {}))
    assert result["ok"]
    assert execution.modality is None
    assert result["modality_detection"]["source"] == "dicom_metadata"
    assert execution._frozen.is_dir() and execution._frozen != source
    assert len(list(execution._frozen.iterdir())) == 3
    assert not execution.has_outputs


def test_detection_after_dicom_conversion_uses_original_frozen_metadata(tmp_path, monkeypatch):
    source = dicom_at(tmp_path / "dicom")
    backend = Backend()

    async def convert_then_segment(**kwargs):
        directory = Path(kwargs["output_dir"]) / "converted"
        directory.mkdir(parents=True)
        converted = make_input(directory)
        result = await backend(**{**kwargs, "input_path": str(converted)})
        result["converted_input_path"] = str(converted)
        return result

    def forbidden(_features):
        raise AssertionError("Converted DICOM must retain its source metadata for detection.")

    monkeypatch.setattr(core, "segment", convert_then_segment)
    monkeypatch.setattr(modality, "classify_features", forbidden)
    execution = TaskExecution(source, output_dir=tmp_path / "out")

    async def scenario():
        assert (await execution.call("segment", {"targets": ["liver"], "modality": "CT"}))["ok"]
        assert execution._frozen.is_file()
        assert execution._dicom_snapshot.is_dir()
        # After execution freezes the input, changes to the original cannot alter detection.
        for entry in source.iterdir():
            entry.unlink()
        result = await execution.call("detect_modality", {})
        assert result["ok"]
        assert result["modality_detection"]["source"] == "dicom_metadata"
        assert result["modality_detection"]["modality"] == "CT"
        assert result["modality_detection"]["declared_modality"] is None
        assert execution.is_complete

    asyncio.run(scenario())
    assert len(backend.calls) == 1


@pytest.mark.parametrize("detect_first", [False, True])
def test_wrong_dicom_modality_can_be_corrected_before_binding(tmp_path, monkeypatch, detect_first):
    source = dicom_at(tmp_path / "dicom")
    backend = Backend()

    async def convert_then_segment(**kwargs):
        directory = Path(kwargs["output_dir"]) / "converted"
        directory.mkdir(parents=True)
        converted = make_input(directory)
        result = await backend(**{**kwargs, "input_path": str(converted)})
        result["converted_input_path"] = str(converted)
        return result

    monkeypatch.setattr(core, "segment", convert_then_segment)
    execution = TaskExecution(source, output_dir=tmp_path / "out")

    async def scenario():
        if detect_first:
            assert (await execution.call("detect_modality", {}))["ok"]
            assert execution.modality is None and execution._frozen.is_dir()
        incorrect = await execution.call("segment", {"targets": ["liver"], "modality": "MR"})
        assert not incorrect["ok"] and incorrect["code"] == "INPUT_INVALID"
        assert execution.modality is None
        assert backend.calls == []
        corrected = await execution.call("segment", {"targets": ["liver"], "modality": "CT"})
        assert corrected["ok"]
        assert execution.modality == "CT" and execution.is_complete

    asyncio.run(scenario())
    assert len(backend.calls) == 1 and backend.calls[0]["task"] == "total"


@pytest.mark.parametrize("value", ["US", "other"])
def test_unsupported_dicom_never_starts_ct_mr_validation_or_conversion(
    tmp_path, monkeypatch, value
):
    source = tmp_path / "ultrasound.dcm"
    source.write_bytes(b"synthetic metadata probe fixture")

    def probe(path, *, _stop_event=None):
        return {
            "modality": value,
            "source": "dicom_metadata",
            "status": "unsupported",
            "supported": False,
            "limitations": ["NO_SEGMENTATION_BACKEND"],
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported modality must not reach CT/MR image preparation.")

    monkeypatch.setattr(modality, "probe_dicom", probe)
    monkeypatch.setattr(core, "_inspect_dicom_directory", forbidden)
    monkeypatch.setattr(core, "validate_input", forbidden)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(source, output_dir=tmp_path / "out")

    async def scenario():
        result = await execution.call("detect_modality", {})
        assert result["ok"] and result["status"] == "completed"
        assert result["modality_detection"]["status"] == "unsupported"
        assert result["modality_detection"]["modality"] == value
        assert execution.modality is None and execution._frozen is None
        assert execution.unresolved_failures == []
        assert (await execution.call("segment", {"targets": ["liver"], "modality": "CT"}))[
            "code"
        ] == "INPUT_INVALID"

    asyncio.run(scenario())
    assert backend.calls == []


def test_detect_has_no_model_supplied_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(modality, "classify_features", lambda features: classification())
    execution = TaskExecution(make_input(tmp_path), output_dir=tmp_path / "out")
    response = asyncio.run(execution.call("detect_modality", {"input_path": "/unexpected"}))
    assert response["code"] == "INVALID_ARGUMENTS"
    assert execution._frozen is None


def test_unsupported_dicom_cannot_become_ct_through_an_agent_argument(tmp_path, monkeypatch):
    source = dicom_at(tmp_path / "dicom", modality="US")
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(source, output_dir=tmp_path / "out")

    async def scenario():
        evidence = await execution.call("detect_modality", {})
        assert evidence["ok"]
        assert evidence["modality_detection"]["modality"] == "US"
        assert evidence["modality_detection"]["status"] == "unsupported"
        result = await execution.call("segment", {"targets": ["liver"], "modality": "CT"})
        assert not result["ok"] and result["code"] == "INPUT_INVALID"
        assert execution.modality is None
        assert not execution.has_outputs

    asyncio.run(scenario())
    assert backend.calls == []


def test_detector_failure_is_cached_safe_uncertainty(tmp_path, monkeypatch):
    calls = []

    def broken(features):
        calls.append(True)
        raise RuntimeError("PRIVATE /models/checkpoint.bin and patient identifier")

    monkeypatch.setattr(modality, "classify_features", broken)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), output_dir=tmp_path / "out")

    async def scenario():
        response = await execution.call("detect_modality", {})
        assert response["ok"] and response["status"] == "completed"
        assert response["modality_detection"]["status"] == "uncertain"
        assert execution.unresolved_failures == []
        assert "PRIVATE" not in json.dumps(response)
        assert str(tmp_path) not in json.dumps(response)
        assert (await execution.call("detect_modality", {}))["cached"]
        assert (await execution.call("segment", {"targets": ["liver"], "modality": "CT"}))["ok"]
        assert execution.is_complete

    asyncio.run(scenario())
    assert len(calls) == 1
    assert len(backend.calls) == 1


@pytest.mark.parametrize("declared", ["CT", "MR"])
def test_declared_segmentation_never_calls_modality_observers(tmp_path, monkeypatch, declared):
    def forbidden(*args, **kwargs):
        raise AssertionError("Segmentation with a declaration does not request modality evidence.")

    monkeypatch.setattr(modality, "classify_features", forbidden)
    monkeypatch.setattr(modality, "probe_dicom", forbidden)
    monkeypatch.setattr(modality, "probe_nifti_sidecar", forbidden)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), declared, tmp_path / "out")
    result = asyncio.run(execution.call("segment", {"targets": ["liver"]}))
    assert result["ok"] and execution.is_complete
    assert len(backend.calls) == 1
    assert execution.modality_detection is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"targets": ["not_a_target"], "modality": "CT"},
        {"targets": ["liver"], "modality": "CT", "quality": "unknown"},
        {"targets": ["liver"], "modality": "US"},
        {"targets": ["liver"], "modality": ["CT"]},
    ],
)
def test_invalid_segment_arguments_never_bind_modality(tmp_path, arguments):
    execution = TaskExecution(tmp_path / "missing.nii.gz", output_dir=tmp_path / "out")
    result = asyncio.run(execution.call("segment", arguments))
    assert not result["ok"]
    assert execution.modality is None and execution._frozen is None
    assert not execution.has_outputs


def test_nifti_sidecar_is_frozen_as_evidence_without_running_classifier(tmp_path, monkeypatch):
    source = make_input(tmp_path)
    sidecar = source.with_name("synthetic.json")
    sidecar.write_text(json.dumps({"Modality": "MR", "PatientName": "PRIVATE"}))

    def forbidden(*args, **kwargs):
        raise AssertionError("A sidecar modality needs no intensity classifier.")

    monkeypatch.setattr(modality, "classify_features", forbidden)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(source, output_dir=tmp_path / "out")

    async def scenario():
        # The Agent can choose directly; freezing keeps the contemporaneous sidecar observation.
        assert (await execution.call("segment", {"targets": ["liver"], "modality": "MR"}))["ok"]
        sidecar.write_text(json.dumps({"Modality": "CT"}))
        result = await execution.call("detect_modality", {})
        assert result["ok"] and result["modality_detection"]["source"] == "nifti_sidecar"
        assert result["modality_detection"]["modality"] == "MR"
        assert "PRIVATE" not in json.dumps(result)
        assert (await execution.call("detect_modality", {}))["cached"]
        assert execution.is_complete

    asyncio.run(scenario())


def test_public_intensity_evidence_only_contains_finite_allowed_numbers():
    result = TaskExecution._public_detection(
        {
            **classification(status="uncertain"),
            "intensity_statistics": {
                "mean": 42,
                "std": float("nan"),
                "min": "PRIVATE",
                "max": True,
                "patient_identifier": "PRIVATE",
            },
        }
    )
    assert result["intensity_statistics"] == {"mean": 42.0}


def test_cancellation_waits_for_detection_worker_and_does_not_bind(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def classify(features):
        started.set()
        release.wait(5)
        return classification()

    monkeypatch.setattr(modality, "classify_features", classify)
    execution = TaskExecution(make_input(tmp_path), output_dir=tmp_path / "out")

    async def scenario():
        worker = asyncio.create_task(execution.call("detect_modality", {}))
        while not started.is_set():
            await asyncio.sleep(0.001)
        worker.cancel()
        await asyncio.sleep(0.005)
        assert not worker.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await worker
        assert execution.modality is None
        assert execution.modality_detection is None
        assert not execution.has_outputs

    asyncio.run(scenario())

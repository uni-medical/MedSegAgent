"""Local modality observations: bundled trees, streamed features and metadata-only DICOM."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    MRImageStorage,
    UltrasoundImageStorage,
    generate_uid,
)

from medsegagent import core, modality


def test_real_bundled_classifier_loads_and_caches_single_thread_models():
    ct = modality.classify_features([0, 600, -1024, 2000])
    mr = modality.classify_features([300, 250, 0, 3000])
    assert ct["modality"] == "CT" and mr["modality"] == "MR"
    assert ct["source"] == "totalseg_intensity" and ct["vote_agreement"] == 1.0
    assert ct["status"] == "detected" and ct["supported"]
    assert "VOTE_AGREEMENT_NOT_CALIBRATED" in ct["limitations"]
    assert "NO_GENERAL_OOD_DETECTION" in ct["limitations"]
    assert modality._classifiers() is modality._classifiers()
    assert len(modality._classifiers()) == 5
    assert all(model.n_jobs == 1 for model in modality._classifiers())


def test_fresh_process_classifies_without_torch_segmentation_imports_or_network():
    code = """
import importlib.abc
import json
import socket
import sys

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("totalsegmentator."):
            raise AssertionError("Heavy segmentation import attempted")

def no_network(*args, **kwargs):
    raise AssertionError("Network access attempted")

sys.meta_path.insert(0, BlockHeavyImports())
socket.create_connection = no_network
socket.socket.connect = no_network
from medsegagent.modality import classify_features
result = classify_features([0, 600, -1024, 2000])
assert result["status"] == "detected"
assert "torch" not in sys.modules
assert "totalsegmentator.python_api" not in sys.modules
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    assert json.loads(result.stdout)["modality"] == "CT"
    assert not result.stderr


def test_real_resampled_mr_counterexample_abstains_despite_unanimous_ct_votes():
    # Public abdomen-mr example, shape (117, 91, 20). Resampling introduces slight
    # negative values; all five shipped models call this CT. Never override to MR.
    result = modality.classify_features([193.6635, 193.7969, -47, 833])
    assert result["status"] == "uncertain" and result["modality"] is None
    assert "candidate_modality" not in result and "vote_agreement" not in result
    assert not result["supported"]
    assert "CT_INTENSITY_EVIDENCE_INSUFFICIENT" in result["limitations"]


class Vote:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        assert len(features) == 1 and len(features[0]) == 4
        return [self.value]


@pytest.mark.parametrize(
    "minimum,accepted", [(-1024, True), (-500, True), (-499, False), (0, False)]
)
def test_ct_intensity_gate_abstains_without_recommending_a_modality(monkeypatch, minimum, accepted):
    monkeypatch.setattr(modality, "_classifiers", lambda: tuple(Vote(0) for _ in range(5)))
    result = modality.classify_features([100, 400, minimum, 3000])
    if accepted:
        assert result["candidate_modality"] == "CT" and result["vote_agreement"] == 1.0
    else:
        assert "candidate_modality" not in result and "vote_agreement" not in result
    assert result["modality"] == ("CT" if accepted else None)
    assert result["status"] == ("detected" if accepted else "uncertain")
    assert result["supported"] is accepted


@pytest.mark.parametrize(
    "votes",
    [
        [0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0],
    ],
)
def test_disagreement_is_an_observation_without_a_candidate_recommendation(monkeypatch, votes):
    monkeypatch.setattr(modality, "_classifiers", lambda: tuple(Vote(value) for value in votes))
    result = modality.classify_features([100, 400, -1024, 3000])
    assert result["modality"] is None and "candidate_modality" not in result
    assert result["status"] == "uncertain" and not result["supported"]
    assert "vote_agreement" not in result
    assert "VOTE_DISAGREEMENT" in result["limitations"]


@pytest.mark.parametrize(
    "features,reason",
    [
        ([0, 0, 0, 0], "CONSTANT_INTENSITY"),
        ([0.5, 0.2, 0, 1], "NORMALIZED_INTENSITY_SUSPECTED"),
        ([0, 0.5, -1, 1], "NORMALIZED_INTENSITY_SUSPECTED"),
        ([0.01, 1.01, -5, 8], "NORMALIZED_INTENSITY_SUSPECTED"),
        ([float("nan"), 1, -1, 1], "INVALID_FEATURES"),
        ([100, -1, -1024, 3000], "INVALID_FEATURES"),
        ([100, 100, 200, 3000], "INVALID_FEATURES"),
        ([1, 2, 3], "INVALID_FEATURES"),
        (None, "INVALID_FEATURES"),
    ],
)
def test_obvious_normalization_constants_and_invalid_features_do_not_run_models(
    monkeypatch, features, reason
):
    def forbidden():
        pytest.fail("Unsupported intensity features must not load models")

    monkeypatch.setattr(modality, "_classifiers", forbidden)
    result = modality.classify_features(features)
    assert result["status"] == "uncertain" and result["modality"] is None
    assert "candidate_modality" not in result and "vote_agreement" not in result
    assert reason in result["limitations"]


def test_classifier_load_failure_is_safe_and_uncertain(monkeypatch):
    def fail():
        raise RuntimeError("/private/secret-model-location")

    monkeypatch.setattr(modality, "_classifiers", fail)
    result = modality.classify_features([100, 500, -1024, 3000])
    assert result["status"] == "uncertain"
    assert "candidate_modality" not in result and "vote_agreement" not in result
    assert "CLASSIFIER_UNAVAILABLE" in result["limitations"]
    assert "private" not in json.dumps(result)
    assert result["intensity_statistics"] == {"mean": 100, "std": 500, "min": -1024, "max": 3000}


@pytest.mark.parametrize(
    "features",
    [
        [0, 0, 0, 0],
        [0.5, 0.2, 0, 1],
        [193.6635, 193.7969, -47, 833],
        [0, 600, -1024, 2000],
        [300, 250, 0, 3000],
    ],
)
def test_finite_statistics_are_observations_even_when_the_classifier_abstains(features):
    result = modality.classify_features(features)
    assert result["intensity_statistics"] == dict(zip(("mean", "std", "min", "max"), features))
    json.dumps(result, allow_nan=False)


def test_invalid_nonfinite_statistics_never_escape_in_reports():
    for value in (float("inf"), float("-inf"), float("nan")):
        result = modality.classify_features([value, 1, -1024, 2000])
        assert "intensity_statistics" not in result
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
@pytest.mark.parametrize(
    "declared,expected,supported",
    [("CT", "CT", True), ("MR", "MR", True), ("US", "US", False), ("PT", "other", False)],
)
def test_sidecar_reads_only_same_stem_explicit_modality(
    tmp_path, suffix, declared, expected, supported
):
    source = tmp_path / ("ct-named-image" + suffix)
    source.touch()  # This observer must not load image pixels or validate the image.
    (tmp_path / "ct-named-image.json").write_text(
        json.dumps(
            {"Modality": declared, "PatientID": "private-id", "ProtocolName": "private-protocol"}
        )
    )
    report = modality.probe_nifti_sidecar(source)
    assert report["modality"] == expected and report["supported"] is supported
    assert report["status"] == ("detected" if supported else "unsupported")
    assert report["source"] == "nifti_sidecar"
    assert "SIDECAR_DECLARATION_NOT_PIXEL_VALIDATION" in report["limitations"]
    encoded = json.dumps(report)
    assert "private" not in encoded and str(tmp_path) not in encoded


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        "null",
        "[]",
        "invalid-json",
        '{"Modality": null}',
        '{"Modality": ["CT"]}',
        '{"Modality": "MR instructions"}',
        '{"Modality": ""}',
    ],
)
def test_unusable_sidecar_returns_none_to_allow_other_observations(tmp_path, content):
    source = tmp_path / "image.nii.gz"
    source.touch()
    (tmp_path / "image.json").write_text(content)
    assert modality.probe_nifti_sidecar(source) is None


def test_sidecar_does_not_infer_from_filename_or_other_metadata(tmp_path):
    source = tmp_path / "ct.nii.gz"
    source.touch()
    assert modality.probe_nifti_sidecar(source) is None
    (tmp_path / "ct.json").write_text(json.dumps({"MagneticFieldStrength": 3, "EchoTime": 0.003}))
    assert modality.probe_nifti_sidecar(source) is None


def test_sidecar_rejects_oversized_nonregular_and_symlink_inputs(tmp_path):
    source = tmp_path / "image.nii"
    source.touch()
    sidecar = tmp_path / "image.json"
    sidecar.write_text('{"Modality":"CT","padding":"' + "x" * 65536 + '"}')
    assert modality.probe_nifti_sidecar(source) is None
    sidecar.unlink()
    sidecar.mkdir()
    assert modality.probe_nifti_sidecar(source) is None
    sidecar.rmdir()
    os.mkfifo(sidecar)
    assert modality.probe_nifti_sidecar(source) is None
    sidecar.unlink()
    target = tmp_path / "metadata.json"
    target.write_text('{"Modality":"MR"}')
    sidecar.symlink_to(target)
    assert modality.probe_nifti_sidecar(source) is None
    sidecar.unlink()
    sidecar.write_text('{"Modality":"MR"}')
    source.unlink()
    source.symlink_to(target)
    assert modality.probe_nifti_sidecar(source) is None


def test_sidecar_stops_without_opening_metadata(tmp_path):
    stop = threading.Event()
    stop.set()
    with pytest.raises(modality._ProbeStopped):
        modality.probe_nifti_sidecar(tmp_path / "image.nii.gz", _stop_event=stop)


@pytest.mark.parametrize(
    "suffix,image_type",
    [
        (".nii", nib.Nifti1Image),
        (".nii.gz", nib.Nifti1Image),
        (".nii.gz", nib.Nifti2Image),
    ],
)
def test_streamed_intensity_features_match_float64_population_reference(
    tmp_path, monkeypatch, suffix, image_type
):
    path = tmp_path / ("synthetic" + suffix)
    values = np.random.default_rng(123).integers(-1000, 3000, size=(7, 9, 11), dtype=np.int16)
    image = image_type(values, np.eye(4))
    image.header.set_slope_inter(2.5, -100)
    nib.save(image, path)
    data = nib.load(path).get_fdata()
    expected = [np.mean(data), np.std(data, ddof=0), np.min(data), np.max(data)]

    def forbid_whole_volume(*args, **kwargs):
        pytest.fail("The inspector must not allocate a get_fdata whole-volume array")

    monkeypatch.setattr(nib.dataobj_images.DataobjImage, "get_fdata", forbid_whole_volume)
    result = core._inspect_nifti(path, collect_intensity_stats=True)
    np.testing.assert_allclose(result["intensity_features"], expected, rtol=1e-12, atol=1e-10)
    assert result["shape"] == list(values.shape)


def test_default_inspection_does_not_compute_intensity_moments(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4)), path)

    def forbidden(*args, **kwargs):
        pytest.fail("Default inspection must not add mean or variance work")

    monkeypatch.setattr(np, "mean", forbidden)
    monkeypatch.setattr(np, "var", forbidden)
    result = core._inspect_nifti(path)
    assert "intensity_features" not in result


def test_intensity_collection_shares_one_gzip_expansion(tmp_path, monkeypatch):
    from contextlib import contextmanager

    path = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(np.arange(120, dtype=np.float32).reshape((4, 5, 6)), np.eye(4)), path)
    original = core._uncompressed_nifti
    calls = []

    @contextmanager
    def observed(*args, **kwargs):
        calls.append(args[0])
        with original(*args, **kwargs) as result:
            yield result

    monkeypatch.setattr(core, "_uncompressed_nifti", observed)
    result = core._inspect_nifti(path, collect_intensity_stats=True)
    assert calls == [path]
    assert result["intensity_features"][0] == 59.5


def make_dicom(path, *, image_modality="CT", study=None, series=None, instance=None):
    sop_class = {"CT": CTImageStorage, "MR": MRImageStorage, "US": UltrasoundImageStorage}.get(
        image_modality,
        CTImageStorage,
    )
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = instance or generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.Modality = image_modality
    dataset.SOPClassUID = sop_class
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = study or generate_uid()
    dataset.SeriesInstanceUID = series or generate_uid()
    dataset.PatientID = "private-patient-identifier"
    dataset.PatientName = "Private^Name"
    dataset.Rows, dataset.Columns = 2, 2
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelData = b"\0" * 8
    dataset.save_as(path, enforce_file_format=True)
    return dataset


@pytest.mark.parametrize(
    "image_modality,status,supported,expected",
    [
        ("CT", "detected", True, "CT"),
        ("MR", "detected", True, "MR"),
        ("US", "unsupported", False, "US"),
        ("PT", "unsupported", False, "other"),
    ],
)
def test_dicom_reads_every_instance_without_pixels_or_patient_identifiers(
    tmp_path,
    monkeypatch,
    image_modality,
    status,
    supported,
    expected,
):
    study, series = generate_uid(), generate_uid()
    for index in range(3):
        make_dicom(
            tmp_path / f"{index}.dcm", image_modality=image_modality, study=study, series=series
        )
    original = pydicom.dcmread
    calls = []

    def observed(*args, **kwargs):
        assert kwargs["stop_before_pixels"] is True
        result = original(*args, **kwargs)
        assert "PixelData" not in result and "PatientName" not in result
        calls.append(result.Modality)
        return result

    monkeypatch.setattr(pydicom, "dcmread", observed)
    result = modality.probe_dicom(tmp_path)
    assert len(calls) == 3
    assert result["modality"] == expected and result["status"] == status
    assert result["supported"] is supported
    assert result["source"] == "dicom_metadata"
    serialized = json.dumps(result)
    for private in (str(tmp_path), "private-patient", "Private", study, series):
        assert private not in serialized


@pytest.mark.parametrize("mismatch", ["modality", "series", "study"])
def test_dicom_consistency_cannot_be_decided_from_the_first_slice(tmp_path, mismatch):
    study, series = generate_uid(), generate_uid()
    make_dicom(tmp_path / "first.dcm", study=study, series=series)
    make_dicom(
        tmp_path / "second.dcm",
        image_modality="MR" if mismatch == "modality" else "CT",
        study=generate_uid() if mismatch == "study" else study,
        series=generate_uid() if mismatch == "series" else series,
    )
    result = modality.probe_dicom(tmp_path)
    assert result["modality"] is None and result["status"] == "uncertain"
    assert "DICOM_SERIES_INCONSISTENT" in result["limitations"]


def test_dicom_duplicate_instance_is_uncertain(tmp_path):
    study, series, instance = generate_uid(), generate_uid(), generate_uid()
    for index in range(2):
        make_dicom(tmp_path / f"{index}.dcm", study=study, series=series, instance=instance)
    result = modality.probe_dicom(tmp_path)
    assert result["status"] == "uncertain" and "DICOM_DUPLICATE_INSTANCE" in result["limitations"]


def test_dicom_does_not_skip_unreadable_files_or_follow_symlinks(tmp_path):
    make_dicom(tmp_path / "valid.dcm")
    (tmp_path / "invalid.txt").write_text("private unrelated content")
    result = modality.probe_dicom(tmp_path)
    assert result["status"] == "uncertain" and "private" not in json.dumps(result)
    (tmp_path / "invalid.txt").unlink()
    (tmp_path / "linked.dcm").symlink_to(tmp_path / "valid.dcm")
    assert "DICOM_INPUT_INVALID" in modality.probe_dicom(tmp_path)["limitations"]


def test_dicom_metadata_probe_obeys_total_file_size_limit(tmp_path, monkeypatch):
    make_dicom(tmp_path / "valid.dcm")
    monkeypatch.setenv("MEDSEGAGENT_MAX_INPUT_BYTES", "10")
    result = modality.probe_dicom(tmp_path)
    assert result["status"] == "uncertain" and "DICOM_LIMIT_EXCEEDED" in result["limitations"]


def test_dicom_metadata_probe_obeys_instance_count_limit(tmp_path):
    for index in range(4097):
        (tmp_path / f"{index}.dcm").touch()
    result = modality.probe_dicom(tmp_path)
    assert "DICOM_LIMIT_EXCEEDED" in result["limitations"]


def test_dicom_probe_propagates_stop_without_converting_it_to_an_observation(tmp_path):
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(RuntimeError, match="stopped"):
        modality.probe_dicom(tmp_path, _stop_event=stopped)

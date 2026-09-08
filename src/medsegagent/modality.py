"""Small local modality observations; no segmentation imports, telemetry or model downloads.

The shipped TotalSegmentator intensity classifier is a closed CT/MR classifier. Its
five-model voting agreement is not a calibrated probability or an OOD detector.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from functools import lru_cache
from importlib import metadata
from pathlib import Path

_INTENSITY_LIMITATIONS = (
    "CT_MR_ONLY",
    "VOTE_AGREEMENT_NOT_CALIBRATED",
    "NO_GENERAL_OOD_DETECTION",
)
_DICOM_LIMITATIONS = ("METADATA_ONLY_NOT_PIXEL_VALIDATION",)
# A conservative routing heuristic, not a calibrated CT/MR decision boundary.
# Air-containing HU CT should have a substantially negative lower tail. Cropped
# CT can lack it and will be left uncertain; negative-valued MR can still pass.
_CT_MINIMUM_INTENSITY = -500.0
_MAX_SIDECAR_BYTES = 64 * 1024


@lru_cache(maxsize=1)
def _classifiers():
    """Load only the five small bundled models, without importing the TotalSeg CLI."""
    from xgboost import XGBClassifier

    distribution = metadata.distribution("TotalSegmentator")
    models = []
    for fold in range(5):
        path = distribution.locate_file(
            f"totalsegmentator/resources/modality_classifiers_2025_02_24.json.{fold}"
        )
        model = XGBClassifier(n_jobs=1, device="cpu")
        # The shipped files have numeric suffixes; loading bytes avoids format-guess
        # warnings that would otherwise print local package paths.
        model.load_model(bytearray(Path(path).read_bytes()))
        model.set_params(n_jobs=1, device="cpu")
        models.append(model)
    return tuple(models)


def _intensity_report(reason, *, candidate=None, agreement=0.0, statistics=None):
    report = {
        "modality": candidate if reason is None else None,
        "source": "totalseg_intensity",
        "status": "detected" if reason is None else "uncertain",
        "supported": reason is None,
        "limitations": [*_INTENSITY_LIMITATIONS, *([reason] if reason is not None else [])],
    }
    # An abstained prediction is not a routing recommendation. Returning its
    # candidate and unanimous votes would contradict the uncertain observation.
    if reason is None and candidate is not None:
        report["candidate_modality"] = candidate
        report["vote_agreement"] = agreement
    if statistics is not None:
        report["intensity_statistics"] = statistics
    return report


def classify_features(features) -> dict:
    """Classify float64 [mean, population std, min, max] using the bundled model votes.

    Unanimity is a conservative routing policy, not a validation claim. Obvious
    constants, min-max normalization and z-score normalization are left uncertain;
    these checks deliberately do not claim to catch every normalized or OOD image.
    A CT vote additionally requires a minimum <= -500 in the input intensity scale.
    This uncalibrated heuristic deliberately abstains on some cropped/shifted CT;
    it cannot establish that intensities are HU or exclude negative-valued MR.
    """
    try:
        if len(features) != 4 or any(isinstance(value, (str, bytes, bool)) for value in features):
            return _intensity_report("INVALID_FEATURES")
        mean, std, minimum, maximum = [float(value) for value in features]
    except (TypeError, ValueError, OverflowError):
        return _intensity_report("INVALID_FEATURES")
    if not all(math.isfinite(value) for value in (mean, std, minimum, maximum)):
        return _intensity_report("INVALID_FEATURES")
    statistics = {"mean": mean, "std": std, "min": minimum, "max": maximum}
    if std < 0 or minimum > maximum or not minimum <= mean <= maximum:
        return _intensity_report("INVALID_FEATURES", statistics=statistics)
    if maximum == minimum or std == 0:
        return _intensity_report("CONSTANT_INTENSITY", statistics=statistics)
    if (minimum >= -1.0 and maximum <= 1.0) or (abs(mean) <= 0.05 and abs(std - 1.0) <= 0.05):
        return _intensity_report("NORMALIZED_INTENSITY_SUSPECTED", statistics=statistics)
    try:
        votes = [
            float(model.predict([[mean, std, minimum, maximum]])[0]) for model in _classifiers()
        ]
    except Exception:  # noqa: BLE001 - dependency/model exceptions must not expose local paths.
        return _intensity_report("CLASSIFIER_UNAVAILABLE", statistics=statistics)
    if len(votes) != 5 or any(vote not in {0, 1} for vote in votes):
        return _intensity_report("CLASSIFIER_UNAVAILABLE", statistics=statistics)
    mr_votes = sum(votes)
    candidate = "CT" if mr_votes < 3 else "MR"
    agreement = max(mr_votes, 5 - mr_votes) / 5
    reason = "VOTE_DISAGREEMENT" if agreement != 1.0 else None
    if reason is None and candidate == "CT" and minimum > _CT_MINIMUM_INTENSITY:
        reason = "CT_INTENSITY_EVIDENCE_INSUFFICIENT"
    return _intensity_report(
        reason, candidate=candidate, agreement=agreement, statistics=statistics
    )


class _ProbeStopped(RuntimeError):
    pass


def _check_stop(event):
    if event is not None and event.is_set():
        raise _ProbeStopped("Modality observation was stopped.")


def probe_nifti_sidecar(path, *, _stop_event=None) -> dict | None:
    """Read only a same-stem conversion JSON's explicit Modality declaration.

    This is source metadata, not a pixel classifier or full BIDS inheritance
    resolution. Missing, malformed or oversized sidecars leave other evidence
    available; neither filenames nor arbitrary header text imply a modality.
    """
    _check_stop(_stop_event)
    directory_fd = None
    try:
        source = Path(path).expanduser()
        suffix = 7 if source.name.lower().endswith(".nii.gz") else 4
        if not source.name.lower().endswith((".nii", ".nii.gz")):
            return None
        if not stat.S_ISREG(source.lstat().st_mode):
            return None
        directory_fd = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptor = os.open(
            source.name[:-suffix] + ".json",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_SIDECAR_BYTES:
                return None
            raw = stream.read(_MAX_SIDECAR_BYTES + 1)
            after = os.fstat(stream.fileno())
            if len(raw) != before.st_size or (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                return None
        _check_stop(_stop_event)
        metadata = json.loads(raw)
        value = metadata.get("Modality") if isinstance(metadata, dict) else None
        if not isinstance(value, str) or re.fullmatch(r"[A-Z][A-Z0-9_]{0,15}", value) is None:
            return None
        modality = value if value in {"CT", "MR", "US"} else "other"
        supported = modality in {"CT", "MR"}
        return {
            "modality": modality,
            "source": "nifti_sidecar",
            "status": "detected" if supported else "unsupported",
            "supported": supported,
            "limitations": [
                "SIDECAR_DECLARATION_NOT_PIXEL_VALIDATION",
                *([] if supported else ["UNSUPPORTED_MODALITY"]),
            ],
        }
    except _ProbeStopped:
        raise
    except (OSError, ValueError, TypeError, RecursionError):
        return None
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _dicom_report(modality=None, reason=None):
    supported = modality in {"CT", "MR"}
    return {
        "modality": modality,
        "source": "dicom_metadata",
        "status": "uncertain" if modality is None else "detected" if supported else "unsupported",
        "supported": supported,
        "limitations": [
            *_DICOM_LIMITATIONS,
            *([reason] if reason is not None else []),
            *(["UNSUPPORTED_MODALITY"] if modality is not None and not supported else []),
        ],
    }


def probe_dicom(path, *, _stop_event=None) -> dict:
    """Observe every file's modality and series identity using headers only.

    This does not replace the core's strict pixel, geometry and format validation.
    A single file can be observed, even though current segmentation accepts a series
    directory. Patient identifiers and UIDs are compared locally and never returned.
    """
    import pydicom
    from pydicom.uid import UID

    from medsegagent import core

    _check_stop(_stop_event)
    source = Path(path).expanduser()
    directory_fd = None
    try:
        if source.is_symlink():
            return _dicom_report(reason="DICOM_INPUT_INVALID")
        single = source.is_file()
        directory = source.parent if single else source
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if single:
            names = [source.name]
        else:
            names = []
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    _check_stop(_stop_event)
                    names.append(entry.name)
                    if len(names) > 4096:
                        return _dicom_report(reason="DICOM_LIMIT_EXCEEDED")
        if not names:
            return _dicom_report(reason="DICOM_INPUT_INVALID")
        limit = core._positive_int("MEDSEGAGENT_MAX_INPUT_BYTES", core.DEFAULT_MAX_INPUT_BYTES)
        total_bytes = 0
        reference = None
        instances = set()
        for name in sorted(names):
            _check_stop(_stop_event)
            entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(entry.st_mode):
                return _dicom_report(reason="DICOM_INPUT_INVALID")
            total_bytes += entry.st_size
            if entry.st_size > min(limit, 64 * 1024 * 1024) or total_bytes > limit:
                return _dicom_report(reason="DICOM_LIMIT_EXCEEDED")
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != entry.st_size:
                    return _dicom_report(reason="DICOM_INPUT_CHANGED")
                ds = pydicom.dcmread(
                    stream,
                    stop_before_pixels=True,
                    force=False,
                    specific_tags=[
                        "Modality",
                        "PatientID",
                        "StudyInstanceUID",
                        "SeriesInstanceUID",
                        "SOPClassUID",
                        "SOPInstanceUID",
                    ],
                )
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    return _dicom_report(reason="DICOM_INPUT_CHANGED")
            modality = str(getattr(ds, "Modality", "")).strip().upper()
            if re.fullmatch(r"[A-Z][A-Z0-9_]{0,15}", modality) is None:
                return _dicom_report(reason="DICOM_METADATA_INCOMPLETE")
            uids = tuple(
                str(getattr(ds, key, ""))
                for key in (
                    "StudyInstanceUID",
                    "SeriesInstanceUID",
                    "SOPClassUID",
                    "SOPInstanceUID",
                )
            )
            if any(not value or not UID(value).is_valid for value in uids):
                return _dicom_report(reason="DICOM_METADATA_INCOMPLETE")
            identity = (modality, str(getattr(ds, "PatientID", "")), *uids[:3])
            if reference is not None and reference != identity:
                return _dicom_report(reason="DICOM_SERIES_INCONSISTENT")
            reference = identity
            if uids[3] in instances:
                return _dicom_report(reason="DICOM_DUPLICATE_INSTANCE")
            instances.add(uids[3])
        _check_stop(_stop_event)
        modality = reference[0]
        return _dicom_report(modality if modality in {"CT", "MR", "US"} else "other")
    except _ProbeStopped:
        raise
    except Exception:  # noqa: BLE001 - malformed metadata and paths are never returned.
        return _dicom_report(reason="DICOM_INPUT_INVALID")
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

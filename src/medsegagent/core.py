"""Local CT/MR inference shared by CLI, MCP, Web and A2A; no network/LLM calls."""

from __future__ import annotations

import asyncio
import colorsys
import fcntl
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import signal
import stat
import struct
import tempfile
import threading
import time
import unicodedata
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from difflib import get_close_matches
from pathlib import Path
from shutil import which
from typing import Literal
from xml.etree import ElementTree

from medsegagent.gpu_scheduler import DeviceLease, GPUScheduler, SchedulerError, SchedulerTimeout
from medsegagent.task_specs import TASK_SPECS, Task, supports_native_roi

DEFAULT_TIMEOUT_SECONDS = 7200
DEFAULT_MAX_INPUT_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
WARNING = "Research use only. Outputs require review; no clinical performance claim is made."
RESULT_SCHEMA_VERSION = 3
LABEL_COLORS = (
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#ffff00",
    "#ff00ff",
    "#00ffff",
    "#ff8000",
    "#ff0080",
    "#80ff80",
    "#0080ff",
    "#808080",
    "#b9aa9b",
)


class SegmentationError(ValueError):
    """Expected failure; the private run directory contains durable audit details."""

    def __init__(self, message: str, *, run_dir: Path | None = None, code: str | None = None):
        super().__init__(message)
        self.run_dir = run_dir
        self.code = code


def task_classes(task: str) -> set[str]:
    return set(task_labels(task).values())


def task_labels(task: str) -> dict[int, str]:
    from totalsegmentator.registry import get_task_classes

    if task not in TASK_SPECS:
        raise SegmentationError(f"task must be one of: {', '.join(TASK_SPECS)}.")
    return get_task_classes(task)


def normalize_targets(task: str, targets: list[str] | None) -> list[str] | None:
    available = task_classes(task)
    if targets is None:
        return None
    if not isinstance(targets, list) or not targets:
        raise SegmentationError(
            "targets must be a non-empty list; omit it only for all structures."
        )
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise SegmentationError("Each target must be a non-empty official class name.")
    normalized = list(dict.fromkeys(target.strip() for target in targets))
    unknown = sorted(set(normalized) - available)
    if unknown:
        suggestions = {
            x: get_close_matches(x, sorted(available), n=3, cutoff=0.45) for x in unknown
        }
        raise SegmentationError(
            f"Unsupported {task} targets: {unknown}. Suggestions: {suggestions}"
        )
    return normalized


def _task_speed(task: str, speed: str | None):
    spec = TASK_SPECS.get(task) if isinstance(task, str) else None
    if spec is None:
        raise SegmentationError("Unsupported segmentation task.", code="UNSUPPORTED_TASK")
    speed = spec.default_speed if speed is None else speed
    if not isinstance(speed, str) or speed not in spec.speeds:
        raise SegmentationError(
            f"{task} supports these quality modes: {', '.join(spec.speeds)}.",
            code="UNSUPPORTED_QUALITY",
        )
    return spec, speed


def validate_task_options(task: str, speed: str | None = None, targets: list[str] | None = None):
    """Validate a producer choice without image reads, weight loading, or inference."""
    spec, speed = _task_speed(task, speed)
    if targets is None and spec.default_targets is not None:
        targets = list(spec.default_targets)
    normalized = normalize_targets(task, targets)
    if spec.availability == "unavailable" or spec.license_required:
        raise SegmentationError(
            "This registered task is excluded by the public service policy.",
            code="TASK_UNAVAILABLE",
        )
    return spec, speed, normalized


def preflight_task(task: str, speed: str | None = None, targets: list[str] | None = None):
    """Require an executable task and its prepared dependencies; never download weights."""
    from medsegagent.weights import WeightError, require_weights

    spec, speed, normalized = validate_task_options(task, speed, targets)
    try:
        require_weights(task, speed, normalized)
    except WeightError as exc:
        raise SegmentationError(
            "Required model weights are not prepared.", code="WEIGHTS_MISSING"
        ) from exc
    return spec, speed, normalized


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
        if value <= 0:
            raise ValueError
        return value
    except ValueError as exc:
        raise SegmentationError(f"{name} must be a positive integer.") from exc


def _check_stop(event):
    if event is not None and event.is_set():
        raise SegmentationError("Input validation was stopped.")


async def _validation(function, *args, timeout_seconds=None, **kwargs):
    """Keep the event loop responsive and retain ownership until its reader has stopped."""
    stop = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(function, *args, _stop_event=stop, **kwargs))
    try:
        return await asyncio.wait_for(asyncio.shield(worker), timeout=timeout_seconds)
    except (asyncio.CancelledError, TimeoutError) as exc:
        stop.set()
        drained = asyncio.gather(worker, return_exceptions=True)
        while not drained.done():
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                continue
        drained.result()  # consume the reader's stop exception; preserve cancellation/timeout
        if isinstance(exc, TimeoutError):
            raise SegmentationError("Input validation exceeded the task timeout.") from exc
        raise


def _inspect_dicom_directory(
    path: Path, task: Task | None, *, snapshot_dir: Path | None = None, _stop_event=None
) -> dict[str, object]:
    """Accept only one regular, uncompressed, single-frame CT/MR image stack.

    Read every byte with O_NOFOLLOW; conversion sees exact validated bytes under private
    generated names. ZIPs, nested folders, mixed series, enhanced/multiframe, compressed
    pixels, duplicate slices, variable spacing and gantry tilt are deliberately rejected.
    """
    import numpy as np
    import pydicom
    from pydicom.uid import CTImageStorage, MRImageStorage

    if task not in TASK_SPECS:
        raise SegmentationError("DICOM requires an explicit supported task with CT or MR modality.")
    modality = TASK_SPECS[task].modality
    sop_class = str(CTImageStorage if modality == "CT" else MRImageStorage)
    limit = _positive_int("MEDSEGAGENT_MAX_INPUT_BYTES", DEFAULT_MAX_INPUT_BYTES)
    decoded_limit = _positive_int(
        "MEDSEGAGENT_MAX_UNCOMPRESSED_BYTES", DEFAULT_MAX_UNCOMPRESSED_BYTES
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    reference = None
    positions = []
    instances = set()
    size = decoded_bytes = 0
    try:
        names = sorted(os.listdir(descriptor))
        if not 2 <= len(names) <= 4096:
            raise SegmentationError("DICOM directory must contain 2 to 4096 image slices.")
        if snapshot_dir is not None:
            snapshot_dir.mkdir(mode=0o700)
        for index, name in enumerate(names):
            _check_stop(_stop_event)
            entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(entry.st_mode):
                raise SegmentationError(
                    "DICOM directory may contain only regular files; no symlinks or nested folders."
                )
            if entry.st_size > min(limit, 64 * 1024 * 1024):
                raise SegmentationError("DICOM instance exceeds the 64 MiB file size limit.")
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise SegmentationError("DICOM input changed to a non-regular file.")
                raw = bytearray()
                while block := stream.read(1024 * 1024):
                    _check_stop(_stop_event)
                    size += len(block)
                    if size > limit or len(raw) + len(block) > 64 * 1024 * 1024:
                        raise SegmentationError("DICOM input exceeds the file size limit.")
                    raw.extend(block)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise SegmentationError("DICOM input changed during validation.")
            try:
                ds = pydicom.dcmread(io.BytesIO(raw), force=False)
                if str(ds.Modality) != modality:
                    raise SegmentationError(
                        f"DICOM modality does not match the selected {modality} tool."
                    )
                if str(ds.SOPClassUID) != sop_class or int(getattr(ds, "NumberOfFrames", 1)) != 1:
                    raise SegmentationError(
                        "Only classic single-frame CT or MR DICOM images are supported."
                    )
                syntax = ds.file_meta.TransferSyntaxUID
                if syntax.is_compressed:
                    raise SegmentationError(
                        "Compressed DICOM pixels are not supported; convert to NIfTI first."
                    )
                for uid in (ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID):
                    if not uid.is_valid:
                        raise SegmentationError(
                            "DICOM contains an invalid instance, study or series UID."
                        )
                if str(ds.file_meta.MediaStorageSOPInstanceUID) != str(ds.SOPInstanceUID):
                    raise SegmentationError("DICOM instance metadata is inconsistent.")
                if str(ds.SOPInstanceUID) in instances:
                    raise SegmentationError("DICOM contains duplicate instances.")
                instances.add(str(ds.SOPInstanceUID))
                rows, columns = int(ds.Rows), int(ds.Columns)
                bits = int(ds.BitsAllocated)
                if (
                    not 0 < rows <= 4096
                    or not 0 < columns <= 4096
                    or bits not in {8, 16, 32}
                    or int(ds.SamplesPerPixel) != 1
                    or ds.PhotometricInterpretation not in {"MONOCHROME1", "MONOCHROME2"}
                ):
                    raise SegmentationError(
                        "DICOM pixels must be one bounded grayscale image per file."
                    )
                expected = rows * columns * (bits // 8)
                decoded_bytes += expected
                if decoded_bytes > decoded_limit:
                    raise SegmentationError(
                        "DICOM decoded pixels exceed the uncompressed size limit."
                    )
                if len(ds.PixelData) not in {expected, expected + expected % 2}:
                    raise SegmentationError(
                        "DICOM pixel data is truncated or has an unexpected length."
                    )
                pixels = ds.pixel_array  # exercise the full data, not only headers
                slope, intercept = (
                    float(getattr(ds, "RescaleSlope", 1)),
                    float(getattr(ds, "RescaleIntercept", 0)),
                )
                if (
                    pixels.shape != (rows, columns)
                    or not math.isfinite(slope)
                    or slope == 0
                    or not math.isfinite(intercept)
                    or not np.isfinite(pixels * slope + intercept).all()
                ):
                    raise SegmentationError("DICOM pixel data or intensity scaling is invalid.")
                spacing = np.asarray(ds.PixelSpacing, dtype=float)
                orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
                position = np.asarray(ds.ImagePositionPatient, dtype=float)
                if (
                    spacing.shape != (2,)
                    or orientation.shape != (6,)
                    or position.shape != (3,)
                    or not np.isfinite(spacing).all()
                    or np.any(spacing <= 0)
                    or not np.isfinite(orientation).all()
                    or not np.isfinite(position).all()
                ):
                    raise SegmentationError("DICOM spatial geometry must be finite and complete.")
                row, column = orientation[:3], orientation[3:]
                if (
                    not np.isclose(np.linalg.norm(row), 1, atol=1e-4)
                    or not np.isclose(np.linalg.norm(column), 1, atol=1e-4)
                    or not np.isclose(np.dot(row, column), 0, atol=1e-4)
                ):
                    raise SegmentationError(
                        "DICOM orientation must contain orthonormal direction cosines."
                    )
                identity = (str(ds.PatientID), str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID))
                if reference is None:
                    reference = (identity, rows, columns, spacing, orientation)
                elif (
                    identity != reference[0]
                    or (rows, columns) != reference[1:3]
                    or not np.allclose(spacing, reference[3], rtol=1e-5, atol=1e-4)
                    or not np.allclose(orientation, reference[4], rtol=1e-5, atol=1e-4)
                ):
                    raise SegmentationError(
                        "DICOM must contain one patient, study and series with consistent geometry."
                    )
                positions.append(position)
            except SegmentationError:
                raise
            except Exception as exc:
                raise SegmentationError(
                    "DICOM contains an unreadable, incomplete or non-image file."
                ) from exc
            if snapshot_dir is not None:
                destination = snapshot_dir / f"{index:06d}.dcm"
                with destination.open("xb") as stream:
                    os.chmod(destination, 0o600)
                    stream.write(raw)
        normal = np.cross(reference[4][:3], reference[4][3:])
        positions = np.asarray(positions)
        locations = positions @ normal
        order = np.argsort(locations)
        differences = np.diff(locations[order])
        if np.any(differences <= 1e-4) or not np.allclose(
            differences, differences[0], rtol=0.01, atol=0.01
        ):
            raise SegmentationError("DICOM slices must have unique positions and uniform spacing.")
        offsets = positions[order] - positions[order][0]
        if not np.allclose(
            offsets, np.outer(locations[order] - locations[order][0], normal), atol=0.1
        ):
            raise SegmentationError(
                "DICOM gantry tilt or changing in-plane position is not supported."
            )
        return {"modality": modality, "instances": len(names)}
    finally:
        os.close(descriptor)


def _dicom_conversion_command(source: Path, output: Path) -> list[str]:
    executable = which("dcm2niix")
    if executable is None:
        candidate = Path(os.sys.executable).parent / "dcm2niix"
        executable = str(candidate) if candidate.is_file() else None
    if executable is None:
        raise SegmentationError(
            "DICOM conversion requires an installed official dcm2niix executable in the uv environment or PATH."
        )
    output.mkdir(mode=0o700)
    return [
        executable,
        "-b",
        "n",
        "-z",
        "y",
        "-f",
        "input",
        "-m",
        "n",
        "-o",
        str(output),
        str(source),
    ]


@contextmanager
def _uncompressed_nifti(path: Path, limit: int, _stop_event=None):
    """Bound expansion in an anonymous file automatically reclaimed even after SIGKILL."""
    if not path.name.lower().endswith(".gz"):
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size > limit:
                raise SegmentationError("NIfTI exceeds the uncompressed size limit.")
            yield stream
        return
    with tempfile.TemporaryFile(mode="w+b") as expanded, gzip.open(path, "rb") as source:
        size = 0
        while block := source.read(1024 * 1024):
            _check_stop(_stop_event)
            size += len(block)
            if size > limit:
                raise SegmentationError("NIfTI exceeds the uncompressed size limit.")
            expanded.write(block)
        expanded.seek(0)
        yield expanded


def _inspect_nifti(
    path: Path,
    *,
    label_map: dict[int, str] | None = None,
    collect_intensity_stats: bool = False,
    _stop_event=None,
) -> dict[str, object]:
    import nibabel as nib
    import numpy as np

    limit = _positive_int("MEDSEGAGENT_MAX_UNCOMPRESSED_BYTES", DEFAULT_MAX_UNCOMPRESSED_BYTES)
    try:
        with _uncompressed_nifti(path, limit, _stop_event) as expanded:
            header_size_bytes = expanded.read(4)
            header_sizes = {
                struct.unpack("<i", header_size_bytes)[0],
                struct.unpack(">i", header_size_bytes)[0],
            }
            image_type = nib.Nifti2Image if 540 in header_sizes else nib.Nifti1Image
            header_type = image_type.header_class
            expanded.seek(0)
            raw_header = header_type.from_fileobj(expanded, check=False)
            if not bytes(raw_header["magic"]).startswith((b"n+1", b"n+2")):
                raise SegmentationError("Input must be a single-file NIfTI image.")
            raw_spacing = np.asarray(raw_header.get_zooms(), dtype=float)
            if not np.isfinite(raw_spacing).all() or np.any(raw_spacing <= 0):
                raise SegmentationError("NIfTI voxel spacing must be finite and positive.")
            expanded.seek(0)
            image = image_type.from_file_map(
                {"image": nib.FileHolder(fileobj=expanded)}, mmap=False
            )
            shape = image.shape
            if len(shape) != 3 or any(size <= 0 for size in shape):
                raise SegmentationError(f"NIfTI must be one non-empty 3D volume, got {shape}.")
            if any(size > 4096 for size in shape):
                raise SegmentationError("NIfTI dimensions exceed the 4096-voxel per-axis limit.")
            spacing = np.asarray(image.header.get_zooms(), dtype=float)
            if not np.isfinite(spacing).all() or np.any(spacing <= 0):
                raise SegmentationError("NIfTI voxel spacing must be finite and positive.")
            affine = image.affine
            if not np.isfinite(affine).all() or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12:
                raise SegmentationError("NIfTI affine must be finite and invertible.")
            dtype = image.get_data_dtype()
            if dtype.kind not in {"u", "i", "f"}:
                raise SegmentationError("NIfTI voxels must be real numeric values.")
            expected_size = math.prod(shape) * dtype.itemsize + int(image.dataobj.offset)
            if expected_size > limit:
                raise SegmentationError(
                    "NIfTI declared dimensions exceed the uncompressed size limit."
                )
            if expected_size > os.fstat(expanded.fileno()).st_size:
                raise SegmentationError("NIfTI voxel data is truncated.")
            counts: dict[int, int] = {}
            if collect_intensity_stats:
                sample_count = 0
                intensity_mean = intensity_m2 = 0.0
                intensity_min, intensity_max = math.inf, -math.inf
            for z in range(shape[2]):
                _check_stop(_stop_event)
                plane = np.asanyarray(image.dataobj[:, :, z])
                if not np.isfinite(plane).all():
                    raise SegmentationError("NIfTI voxel data contains NaN or infinite values.")
                if collect_intensity_stats:
                    # Combine per-plane population moments. This matches get_fdata's
                    # float64, ddof=0 features without holding the whole volume in RAM.
                    values = np.asarray(plane, dtype=np.float64)
                    with np.errstate(over="ignore", invalid="ignore"):
                        plane_mean = float(np.mean(values))
                        plane_m2 = float(np.var(values, ddof=0)) * values.size
                    total = sample_count + values.size
                    delta = plane_mean - intensity_mean
                    intensity_m2 += plane_m2 + delta * delta * sample_count * values.size / total
                    intensity_mean += delta * values.size / total
                    sample_count = total
                    intensity_min = min(intensity_min, float(np.min(values)))
                    intensity_max = max(intensity_max, float(np.max(values)))
                if label_map is not None:
                    if np.any(plane < 0) or np.any(plane != np.floor(plane)):
                        raise SegmentationError("Segmentation contains invalid label values.")
                    labels, voxels = np.unique(plane, return_counts=True)
                    for value, count in zip(labels, voxels, strict=True):
                        label = int(value)
                        if label and label not in label_map:
                            raise SegmentationError(
                                "Segmentation contains labels outside the selected task."
                            )
                        counts[label] = counts.get(label, 0) + int(count)
            result: dict[str, object] = {
                "shape": list(shape),
                "voxel_spacing": spacing.tolist(),
                "affine": affine.tolist(),
            }
            if collect_intensity_stats:
                features = [
                    intensity_mean,
                    math.sqrt(max(0.0, intensity_m2 / sample_count)),
                    intensity_min,
                    intensity_max,
                ]
                if not all(math.isfinite(value) for value in features):
                    raise SegmentationError("NIfTI intensity statistics are not finite.")
                result["intensity_features"] = features
            if label_map is not None:
                result["labels"] = [
                    {"id": index, "name": label_map[index], "voxels": count}
                    for index, count in sorted(counts.items())
                    if index
                ]
                result["nonzero_voxels"] = sum(count for index, count in counts.items() if index)
            return result
    except SegmentationError:
        raise
    except Exception as exc:
        raise SegmentationError("Input is not a readable, complete NIfTI image.") from exc


def _file_digest(path: Path, _stop_event=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            _check_stop(_stop_event)
            digest.update(block)
    return digest.hexdigest()


def _label_color(index: int) -> str:
    if index <= len(LABEL_COLORS):
        return LABEL_COLORS[index - 1]
    # Continue deterministically without repeating the initial categorical palette.
    rgb = colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.65, 0.95)
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)


def _label_extension(rows):
    """Replace native IDs in the embedded Caret label table as well as in the voxels."""
    import nibabel as nib

    root = ElementTree.Element("CaretExtension")
    volume = ElementTree.SubElement(root, "VolumeInformation", Index="0")
    table = ElementTree.SubElement(volume, "LabelTable")
    for row in [{"id": 0, "name": "Background", "color": "#000000"}, *rows]:
        rgb = [int(row["color"][start : start + 2], 16) / 255 for start in (1, 3, 5)]
        entry = ElementTree.SubElement(
            table,
            "Label",
            Key=str(row["id"]),
            Red=str(rgb[0]),
            Green=str(rgb[1]),
            Blue=str(rgb[2]),
            Alpha="1" if row["id"] else "0",
        )
        entry.text = row["name"]
    ElementTree.SubElement(volume, "VolumeType").text = "Label"
    return nib.nifti1.Nifti1Extension(0, ElementTree.tostring(root, encoding="utf-8"))


def label_mask_filename(label: dict) -> str:
    """Keep the stored label ID, including historical IDs, in a safe download name."""
    if not isinstance(label, dict):
        raise SegmentationError("Stored label metadata is invalid.")
    index, name = label.get("id"), label.get("name")
    if (
        type(index) is not int
        or not 0 < index <= 2147483647
        or not isinstance(name, str)
        or not name.strip()
        or len(name) > 120
        or re.search(r"[\\/:]", name)
        or ".." in name
        or any(unicodedata.category(char).startswith("C") for char in name)
    ):
        raise SegmentationError("A label must have a positive integer ID and a safe class name.")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", name) is None:
        # Preserve historical filenames; the dot gives generated names a namespace
        # that cannot collide with any previously accepted ASCII class name.
        name = "label." + hashlib.sha256(name.encode("utf-8")).hexdigest()
    return f"{index}_{name}.nii.gz"


@contextmanager
def _label_export_lock(path: Path, _stop_event=None):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    acquired = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SegmentationError("Class mask lock must be a regular file.")
        while True:
            _check_stop(_stop_event)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(0.025)
        yield
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _export_label_mask(
    segmentation_path: str | Path,
    labels: list[dict],
    label_id: int,
    output_dir: str | Path | None = None,
    *,
    _stop_event=None,
) -> dict:
    """Create one binary mask from a validated merged mask, without rerunning a model.

    Each source scan decompresses gzip once into anonymous storage. Slice reads use the
    resulting uncompressed file, never repeated random seeks through a gzip proxy.
    """
    import nibabel as nib
    import numpy as np

    if type(label_id) is not int or not isinstance(labels, list) or not labels:
        raise SegmentationError("Choose one label from the completed segmentation.")
    label_map = {}
    for row in labels:
        if not isinstance(row, dict):
            raise SegmentationError("Stored label metadata is invalid.")
        label_mask_filename(row)
        if row["id"] in label_map:
            raise SegmentationError("Stored label IDs must be unique.")
        label_map[row["id"]] = row["name"]
    if label_id not in label_map:
        raise SegmentationError("The requested label is not part of this segmentation.")
    filename = label_mask_filename({"id": label_id, "name": label_map[label_id]})
    source = Path(segmentation_path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise SegmentationError("Merged segmentation must be an existing regular file.")
    if source.stat().st_size > _positive_int(
        "MEDSEGAGENT_MAX_INPUT_BYTES", DEFAULT_MAX_INPUT_BYTES
    ):
        raise SegmentationError("Merged segmentation exceeds the file size limit.")
    root = Path(output_dir) if output_dir is not None else source.parent / "class_masks"
    if root.is_symlink():
        raise SegmentationError("Class mask directory must not be a symlink.")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = root.resolve()
    destination, receipt = root / filename, root / f".{filename}.json"
    if destination == source.resolve():
        raise SegmentationError("Class mask output must not replace its merged source.")
    if destination.is_symlink() or receipt.is_symlink():
        raise SegmentationError("Class mask cache must not contain symlinks.")
    temporary = root / f".binary-{uuid.uuid4().hex}.nii.gz"
    with _label_export_lock(root / f".{filename}.lock", _stop_event):
        if source.is_symlink() or destination.is_symlink() or receipt.is_symlink():
            raise SegmentationError("Class mask source and cache must not contain symlinks.")
        source_sha256 = _file_digest(source, _stop_event)
        identity = {
            "schema_version": 1,
            "source_sha256": source_sha256,
            "name": filename,
            "label_id": label_id,
            "label_name": label_map[label_id],
            "mask_value": 1,
            "media_type": "application/gzip",
        }
        try:
            if destination.is_file() and receipt.is_file() and receipt.stat().st_size < 16384:
                cached = json.loads(receipt.read_text())
                if (
                    isinstance(cached, dict)
                    and all(cached.get(key) == value for key, value in identity.items())
                    and type(cached.get("voxels")) is int
                    and cached["voxels"] >= 0
                    and cached.get("size_bytes") == destination.stat().st_size
                    and cached.get("sha256") == _file_digest(destination, _stop_event)
                ):
                    return {
                        **identity,
                        **{key: cached[key] for key in ("sha256", "size_bytes", "voxels")},
                        "path": str(destination),
                    }
        except SegmentationError:
            raise
        except (OSError, ValueError, TypeError):
            # A stale, incomplete or damaged private cache is rebuilt from the merged mask.
            pass
        try:
            geometry = _inspect_nifti(source, label_map=label_map, _stop_event=_stop_event)
            expected_voxels = next(
                (row["voxels"] for row in geometry["labels"] if row["id"] == label_id), 0
            )
            limit = _positive_int(
                "MEDSEGAGENT_MAX_UNCOMPRESSED_BYTES", DEFAULT_MAX_UNCOMPRESSED_BYTES
            )
            with (
                _uncompressed_nifti(source, limit, _stop_event) as expanded,
                tempfile.TemporaryFile(mode="w+b") as binary,
            ):
                prefix = expanded.read(4)
                sizes = {struct.unpack("<i", prefix)[0], struct.unpack(">i", prefix)[0]}
                image_type = nib.Nifti2Image if 540 in sizes else nib.Nifti1Image
                expanded.seek(0)
                image = image_type.from_file_map(
                    {"image": nib.FileHolder(fileobj=expanded)}, mmap=False
                )
                header = image.header.copy()
                header.set_data_dtype(np.uint8)
                header.set_slope_inter(1, 0)
                header.set_intent("label", name="MedSegAgent")
                header.extensions.clear()
                header.extensions.append(
                    _label_extension([{"id": 1, "name": label_map[label_id], "color": "#ff0000"}])
                )
                header.set_data_offset(0)
                header.write_to(binary)
                binary.seek(int(header["vox_offset"]))
                voxels = 0
                for z in range(image.shape[2]):
                    _check_stop(_stop_event)
                    plane = np.asanyarray(image.dataobj[:, :, z])
                    values = np.asarray(plane == label_id, dtype=np.uint8)
                    voxels += int(np.count_nonzero(values))
                    binary.write(values.tobytes(order="F"))
                binary.seek(0)
                with temporary.open("xb") as outgoing:
                    os.chmod(temporary, 0o600)
                    with gzip.GzipFile(
                        filename="", mode="wb", fileobj=outgoing, mtime=0
                    ) as compressed:
                        while block := binary.read(1024 * 1024):
                            _check_stop(_stop_event)
                            compressed.write(block)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
            verified = _inspect_nifti(
                temporary, label_map={1: label_map[label_id]}, _stop_event=_stop_event
            )
            if (
                voxels != expected_voxels
                or verified["nonzero_voxels"] != voxels
                or verified["shape"] != geometry["shape"]
                or verified["voxel_spacing"] != geometry["voxel_spacing"]
                or verified["affine"] != geometry["affine"]
                or _file_digest(source, _stop_event) != source_sha256
            ):
                raise SegmentationError("Class mask geometry or source changed during export.")
            metadata = {
                **identity,
                "voxels": voxels,
                "sha256": _file_digest(temporary, _stop_event),
                "size_bytes": temporary.stat().st_size,
            }
            _check_stop(_stop_event)
            os.replace(temporary, destination)
            _write_json(receipt, metadata)
            return {**metadata, "path": str(destination)}
        finally:
            temporary.unlink(missing_ok=True)


async def export_label_mask(
    segmentation_path: str | Path,
    labels: list[dict],
    label_id: int,
    output_dir: str | Path | None = None,
    *,
    timeout_seconds: float = 180,
) -> dict:
    """Export a completed task's class; callers enforce task ownership and retention."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise SegmentationError("Class mask export timeout must be positive and finite.")
    try:
        return await _validation(
            _export_label_mask,
            segmentation_path,
            labels,
            label_id,
            output_dir,
            timeout_seconds=timeout_seconds,
        )
    except SegmentationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise SegmentationError(
            "Class mask export failed; source or cache is unavailable."
        ) from exc


def _normalize_segmentation(
    path: Path,
    *,
    labels: dict[int, str],
    targets: list[str] | None,
    reference_path: Path | None = None,
    allow_unrequested: bool = False,
    _stop_event=None,
) -> dict[str, object]:
    """Validate, select, relabel and quantify every tool's output without changing regions.

    IDs follow sorted native IDs, including selected-but-empty classes. Native bytes stay
    private; only a fully validated result atomically replaces the public mask filename.
    """
    import nibabel as nib
    import numpy as np

    native_geometry = _inspect_nifti(path, label_map=labels, _stop_event=_stop_event)
    if targets is not None and (not targets or set(targets) - set(labels.values())):
        raise SegmentationError("Normalization requires supported, non-empty targets.")
    selected = sorted(index for index, name in labels.items() if targets is None or name in targets)
    if not selected:
        raise SegmentationError("Normalization requires at least one supported target.")
    if not allow_unrequested and any(
        row["id"] not in selected for row in native_geometry["labels"]
    ):
        raise SegmentationError("Segmentation includes structures outside the requested targets.")
    reference = nib.load(reference_path or path)
    if (
        list(reference.shape) != native_geometry["shape"]
        or not np.allclose(reference.affine, native_geometry["affine"], rtol=1e-5, atol=1e-4)
        or not np.allclose(reference.header.get_zooms(), native_geometry["voxel_spacing"])
    ):
        raise SegmentationError("Segmentation geometry does not match the input volume.")
    unit = reference.header.get_xyzt_units()[0]
    factor = {"meter": 1000.0, "mm": 1.0, "micron": 0.001, "unknown": 1.0}[unit]
    spacing_mm = [float(value) * factor for value in native_geometry["voxel_spacing"]]
    voxel_volume = math.prod(spacing_mm)
    if not math.isfinite(voxel_volume * math.prod(native_geometry["shape"])) or voxel_volume <= 0:
        raise SegmentationError("NIfTI spacing cannot produce a finite physical volume.")
    measurement = {
        "method": "voxel_count_times_spacing_product",
        "spacing_mm": spacing_mm,
        "voxel_volume_mm3": voxel_volume,
        "source_spatial_unit": unit,
        "unit_assumption": "assumed_mm" if unit == "unknown" else None,
    }
    rows = [
        {
            "id": index,
            "source_id": source_id,
            "name": labels[source_id],
            "color": _label_color(index),
        }
        for index, source_id in enumerate(selected, 1)
    ]
    normalized_labels = {row["id"]: row["name"] for row in rows}
    raw_path = path.with_name("segmentation.raw.nii.gz")
    temporary = path.with_name(f".normalized-{uuid.uuid4().hex}.nii.gz")
    if raw_path.exists():
        raise SegmentationError("A private raw mask already exists; refusing to replace it.")
    _check_stop(_stop_event)
    path.rename(raw_path)
    os.chmod(raw_path, 0o600)
    limit = _positive_int("MEDSEGAGENT_MAX_UNCOMPRESSED_BYTES", DEFAULT_MAX_UNCOMPRESSED_BYTES)
    try:
        with (
            _uncompressed_nifti(raw_path, limit, _stop_event) as source,
            tempfile.TemporaryFile(mode="w+b") as normalized,
        ):
            prefix = source.read(4)
            header_sizes = {struct.unpack("<i", prefix)[0], struct.unpack(">i", prefix)[0]}
            image_type = nib.Nifti2Image if 540 in header_sizes else nib.Nifti1Image
            source.seek(0)
            image = image_type.from_file_map({"image": nib.FileHolder(fileobj=source)}, mmap=False)
            if image.dataobj.slope != 1 or image.dataobj.inter != 0:
                raise SegmentationError("Native mask intensity scaling must be identity.")
            header = image.header.copy()
            header.set_data_dtype(np.uint8 if len(rows) <= 255 else np.uint16)
            header.set_slope_inter(1, 0)
            header.set_intent("label", name="MedSegAgent")
            header.set_xyzt_units(*reference.header.get_xyzt_units())
            header.extensions.clear()
            header.extensions.append(_label_extension(rows))
            header.set_data_offset(0)
            header.write_to(normalized)
            offset = int(header["vox_offset"])
            normalized.seek(offset)
            lookup = np.zeros(max(labels) + 1, dtype=header.get_data_dtype())
            lookup[selected] = np.arange(1, len(selected) + 1)
            counts = np.zeros(len(rows) + 1, dtype=np.int64)
            for z in range(image.shape[2]):
                _check_stop(_stop_event)
                plane = np.asanyarray(image.dataobj[:, :, z])
                values = lookup[plane.astype(np.int64)]
                counts += np.bincount(values.ravel(), minlength=len(rows) + 1)
                normalized.write(values.tobytes(order="F"))
            for row in rows:
                _check_stop(_stop_event)
                voxels = int(counts[row["id"]])
                row.update(
                    voxels=voxels,
                    volume_mm3=voxels * voxel_volume,
                    volume_ml=voxels * voxel_volume / 1000,
                )
            normalized.seek(0)
            with temporary.open("xb") as destination:
                os.chmod(temporary, 0o600)
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=destination, mtime=0
                ) as compressed:
                    while block := normalized.read(1024 * 1024):
                        _check_stop(_stop_event)
                        compressed.write(block)
                destination.flush()
                os.fsync(destination.fileno())
        geometry = _inspect_nifti(temporary, label_map=normalized_labels, _stop_event=_stop_event)
        if (
            geometry["shape"] != native_geometry["shape"]
            or not np.allclose(geometry["affine"], native_geometry["affine"], rtol=1e-5, atol=1e-4)
            or geometry["voxel_spacing"] != native_geometry["voxel_spacing"]
            or {row["id"]: row["voxels"] for row in geometry["labels"]}
            != {row["id"]: row["voxels"] for row in rows if row["voxels"]}
        ):
            raise SegmentationError("Normalization changed mask geometry or region sizes.")
        raw_sha256 = _file_digest(raw_path, _stop_event)
        normalized_sha256 = _file_digest(temporary, _stop_event)
        _check_stop(_stop_event)
        os.replace(temporary, path)
        return {
            "geometry": {**geometry, "labels": rows},
            "volume_measurement": measurement,
            "audit": {
                "schema_version": RESULT_SCHEMA_VERSION,
                "operation": "select_relabel_quantify",
                "requested_targets": targets,
                "label_mapping": [
                    {key: row[key] for key in ("id", "source_id", "name", "color")} for row in rows
                ],
                "retained_label_ids": selected,
                "raw_segmentation_path": str(raw_path),
                "raw_sha256": raw_sha256,
                "segmentation_sha256": normalized_sha256,
                "geometry_preserved": True,
                "raw_labels": native_geometry["labels"],
                "raw_nonzero_voxels": native_geometry["nonzero_voxels"],
                "normalized_nonzero_voxels": geometry["nonzero_voxels"],
                "volume_measurement": measurement,
                "postprocessing": "none",
            },
        }
    finally:
        temporary.unlink(missing_ok=True)


def validate_input(input_path: str, task: Task | None = None, *, _stop_event=None) -> Path:
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        if path.is_dir():
            _inspect_dicom_directory(path, task, _stop_event=_stop_event)
            return path
        raise SegmentationError("Input file does not exist or is not a regular file.")
    if not path.name.lower().endswith((".nii", ".nii.gz")):
        raise SegmentationError(
            "Use NIfTI .nii/.nii.gz or a validated DICOM directory; ZIP is rejected."
        )
    if path.stat().st_size > _positive_int("MEDSEGAGENT_MAX_INPUT_BYTES", DEFAULT_MAX_INPUT_BYTES):
        raise SegmentationError("Input exceeds the file size limit.")
    _inspect_nifti(path, _stop_event=_stop_event)
    return path


async def validate_input_async(
    input_path: str, task: Task | None = None, *, timeout_seconds: float | None = None
) -> Path:
    """Validate without blocking the event loop; cancellation waits for reader cleanup."""
    return await _validation(validate_input, input_path, task, timeout_seconds=timeout_seconds)


def device() -> str:
    default = "mps" if platform.system() == "Darwin" else "gpu"
    value = os.environ.get("MEDSEGAGENT_DEVICE", default).lower()
    if value not in {"mps", "gpu", "cpu"}:
        raise SegmentationError("MEDSEGAGENT_DEVICE must be 'mps', 'gpu', or 'cpu'.")
    return value


def _new_run(output_dir: str | None) -> Path:
    root = (
        Path(output_dir or os.environ.get("MEDSEGAGENT_OUTPUT_ROOT", "outputs"))
        .expanduser()
        .resolve()
    )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    while True:
        path = root / uuid.uuid4().hex
        try:
            path.mkdir(mode=0o700)
            return path
        except FileExistsError:
            continue


def _write_json(path: Path, data: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def read_run(run_dir: str | Path) -> dict[str, object]:
    """Read durable state without rerunning inference."""
    return json.loads((Path(run_dir) / "state.json").read_text(encoding="utf-8"))


@asynccontextmanager
async def _inference_lock(deadline: float):
    # Every adapter leases an available device for this inference, never a global env change.
    try:
        async with GPUScheduler(device=device()).acquire(deadline=deadline) as lease:
            yield lease
    except (SchedulerError, SchedulerTimeout) as exc:
        raise SegmentationError(str(exc)) from exc


def _build_command(
    *, task: str, input_path: Path, output_dir: Path, targets: list[str] | None, speed: str
) -> list[str]:
    _spec, speed = _task_speed(task, speed)
    engine = os.environ.get("MEDSEGAGENT_TOTALSEG_ENGINE", "sequential")
    if engine == "sequential":
        command = [os.sys.executable, "-m", "medsegagent.totalseg_worker"]
    elif engine == "cli":
        executable = which("TotalSegmentator")
        if executable is None:
            candidate = Path(os.sys.executable).parent / "TotalSegmentator"
            executable = str(candidate) if candidate.is_file() else None
        if executable is None:
            raise SegmentationError(
                "TotalSegmentator executable was not found in the uv environment."
            )
        command = [executable]
    else:
        raise SegmentationError("MEDSEGAGENT_TOTALSEG_ENGINE must be sequential or cli.")
    command.extend(
        [
            "-i",
            str(input_path),
            "-o",
            str(output_dir / "segmentation.nii.gz"),
            "--ml",
            "--nr_thr_saving",
            "1",
            "--task",
            task,
            "--device",
            device(),
            "--quiet",
            "--report",
            str(output_dir / "run_report.json"),
        ]
    )
    if speed in {"fast", "fastest"}:
        command.extend(["--" + speed, "--higher_order_resampling"])
    if targets is not None and supports_native_roi(task, targets):
        command.extend(["--roi_subset", *targets])
    return command


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    # Stop the entire session even if its group leader already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        pass
    # Allow an already-orphaned group to process SIGTERM before the escalation.
    await asyncio.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    # Repeated cancellation must not interrupt TERM/KILL and release a live child's lease.
    stopping = asyncio.create_task(_terminate_process_group(process))
    while not stopping.done():
        try:
            await asyncio.shield(stopping)
        except asyncio.CancelledError:
            continue
    stopping.result()


def _inference_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
    }


async def _run_command(
    command: list[str],
    *,
    timeout_seconds: float,
    output_dir: Path,
    lock_fd: DeviceLease | int,
    on_start,
) -> None:
    log_path = output_dir / "process.log"
    with log_path.open("ab", buffering=0) as log:
        os.chmod(log_path, 0o600)
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                env=lock_fd.environment(_inference_environment())
                if isinstance(lock_fd, DeviceLease)
                else _inference_environment(),
                pass_fds=lock_fd.pass_fds if isinstance(lock_fd, DeviceLease) else (lock_fd,),
            )
        )
        try:
            process = await asyncio.shield(creation)
        except asyncio.CancelledError:
            # Cancellation can arrive after fork but before the subprocess handle returns.
            while not creation.done():
                try:
                    await asyncio.shield(creation)
                except asyncio.CancelledError:
                    continue
            process = creation.result()
            await _stop_process(process)
            raise
        try:
            on_start(process.pid)
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except TimeoutError as exc:
            await _stop_process(process)
            raise SegmentationError(
                "Local inference exceeded its timeout; the process group was stopped."
            ) from exc
        except BaseException:
            await _stop_process(process)
            raise
        if process.returncode != 0:
            await _stop_process(process)
            raise SegmentationError(
                f"Local inference failed (exit code {process.returncode}); inspect the private run log."
            )


async def segment(
    *,
    task: Task,
    input_path: str,
    output_dir: str | None = None,
    targets: list[str] | None = None,
    speed: Literal["fast", "fastest", "standard"] | None = None,
    on_inference_start=None,
) -> dict[str, object]:
    """Run local inference; output_dir is a parent and omitted targets use task defaults.

    Anatomy tasks default to all native classes; specialized tasks default to their lesion.

    task explicitly declares CT/MR. DICOM directories are strictly preflighted and
    converted from a private snapshot; ZIP is rejected. Cancellation stops readers and
    subprocess groups before persisting terminal state.
    """
    run = _new_run(output_dir)
    started = time.monotonic()
    timings = {}
    timing_stage, stage_started = "validation", started

    def next_stage(name):
        nonlocal timing_stage, stage_started
        now = time.monotonic()
        if timing_stage is not None:
            timings[timing_stage] = timings.get(timing_stage, 0.0) + now - stage_started
        timing_stage, stage_started = name, now
    state: dict[str, object] = {
        "run_id": run.name,
        "status": "validating",
        "task": task,
        "created_at": datetime.now(UTC).isoformat(),
        "owner_pid": os.getpid(),
        "output_dir": str(run),
        "state_path": str(run / "state.json"),
        "timings_seconds": timings,
    }

    def update(**fields):
        state.update(fields, updated_at=datetime.now(UTC).isoformat())
        _write_json(run / "state.json", state)

    update()
    (run / "process.log").touch(mode=0o600)
    try:
        _spec, speed, normalized = preflight_task(task, speed, targets)
        selected_device = device()
        timeout = _positive_int("MEDSEGAGENT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        deadline = time.monotonic() + timeout
        source = Path(input_path).expanduser().resolve()
        dicom = source.is_dir()
        if dicom:
            snapshot = run / "dicom"
            metadata = await _validation(
                _inspect_dicom_directory,
                source,
                task,
                snapshot_dir=snapshot,
                timeout_seconds=deadline - time.monotonic(),
            )
            update(input_format="dicom", dicom_instances=metadata["instances"])
        else:
            source = await _validation(
                validate_input, input_path, timeout_seconds=deadline - time.monotonic()
            )
            update(input_format="nifti")
        next_stage("device_wait")
        update(status="queued", device=selected_device, targets=normalized, speed=speed)
        async with _inference_lock(deadline) as lock_fd:
            next_stage("preparation")
            update(**lock_fd.audit_metadata())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SegmentationError(
                    "Task exceeded its timeout during input validation or queueing."
                )
            if dicom:
                next_stage("dicom_conversion")
                converted = run / "converted"
                command = _dicom_conversion_command(snapshot, converted)
                update(status="converting")
                await _run_command(
                    command,
                    timeout_seconds=remaining,
                    output_dir=run,
                    lock_fd=lock_fd,
                    on_start=lambda pid: update(process_pid=pid),
                )
                outputs = list(converted.glob("*.nii")) + list(converted.glob("*.nii.gz"))
                if len(outputs) != 1:
                    raise SegmentationError(
                        "DICOM conversion must produce exactly one 3D NIfTI volume."
                    )
                source = await _validation(
                    validate_input, str(outputs[0]), timeout_seconds=deadline - time.monotonic()
                )
                update(converted_input_path=str(source))
                next_stage("preparation")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SegmentationError("Task exceeded its timeout before inference.")
            command = _build_command(
                task=task, input_path=source, output_dir=run, targets=normalized, speed=speed
            )
            update(
                status="running",
                inference_engine=os.environ.get("MEDSEGAGENT_TOTALSEG_ENGINE", "sequential"),
            )
            inference_started = time.monotonic()
            next_stage("inference_subprocess")
            if on_inference_start is not None:
                # Persist intent before fork so recovery never duplicates a live child.
                on_inference_start()
            await _run_command(
                command,
                timeout_seconds=remaining,
                output_dir=run,
                lock_fd=lock_fd,
                on_start=lambda pid: update(process_pid=pid),
            )
            inference_seconds = time.monotonic() - inference_started
            next_stage("output_validation")
        segmentation_path = run / "segmentation.nii.gz"
        report_path = run / "run_report.json"
        if not segmentation_path.is_file() or not report_path.is_file():
            raise SegmentationError(
                "Inference finished without its required segmentation and run report."
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise SegmentationError("Inference run report is invalid.")
        update(status="normalizing")
        normalization_started = time.monotonic()
        next_stage("normalization")
        standardized = await _validation(
            _normalize_segmentation,
            segmentation_path,
            labels=task_labels(task),
            targets=normalized,
            reference_path=source,
            allow_unrequested=not supports_native_roi(task, normalized),
            timeout_seconds=deadline - time.monotonic(),
        )
        normalization_seconds = time.monotonic() - normalization_started
        next_stage("result_export")
        geometry = standardized["geometry"]
        _write_json(run / "normalization.json", standardized["audit"])
        result = {
            **state,
            "status": "completed",
            "schema_version": RESULT_SCHEMA_VERSION,
            "volume_measurement": standardized["volume_measurement"],
            "normalization": standardized["audit"],
            "normalization_seconds": normalization_seconds,
            "segmentation_path": str(segmentation_path),
            "run_report": str(report_path),
            "targets": normalized if normalized is not None else "all",
            "segmentation_shape": geometry["shape"],
            "segmentation_voxel_spacing": geometry["voxel_spacing"],
            "segmentation_affine": geometry["affine"],
            "labels": geometry["labels"],
            "nonzero_voxels": geometry["nonzero_voxels"],
            "detection_status": "target_detected"
            if geometry["nonzero_voxels"]
            else "no_target_detected",
            "no_target_detected": not bool(geometry["nonzero_voxels"]),
            "runtime_seconds": inference_seconds,
            "total_seconds": time.monotonic() - started,
            "totalsegmentator_version": importlib.metadata.version("TotalSegmentator"),
            "warning": WARNING
            + (
                " No requested target was detected; this does not rule out disease."
                if not geometry["nonzero_voxels"]
                else ""
            ),
        }
        next_stage(None)
        update(status="completed", result=result)
        return result
    except asyncio.CancelledError as exc:
        next_stage(None)
        exc.timings_seconds = dict(timings)
        exc.inference_engine = os.environ.get("MEDSEGAGENT_TOTALSEG_ENGINE", "sequential")
        update(status="cancelled", error="Task was cancelled; its process group was stopped.")
        raise
    except Exception as exc:
        next_stage(None)
        message = (
            str(exc)
            if isinstance(exc, SegmentationError)
            else "Local task failed; inspect its private run state."
        )
        update(status="failed", error=message, error_type=type(exc).__name__)
        error = SegmentationError(message, run_dir=run, code=getattr(exc, "code", None))
        error.timings_seconds = dict(timings)
        error.inference_engine = os.environ.get("MEDSEGAGENT_TOTALSEG_ENGINE", "sequential")
        raise error from exc


def doctor() -> dict[str, object]:
    """Report installed runtime and hardware without running inference."""
    import torch

    from medsegagent.weights import inventory

    return {
        "python": os.sys.version.split()[0],
        "totalsegmentator": importlib.metadata.version("TotalSegmentator"),
        "executable": which("TotalSegmentator"),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "configured_device": device(),
        "ct_class_count": len(task_classes("total")),
        "mr_class_count": len(task_classes("total_mr")),
        "weights": inventory(),
        "output_root": str(Path(os.environ.get("MEDSEGAGENT_OUTPUT_ROOT", "outputs")).resolve()),
    }

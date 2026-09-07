"""Local CT/MR inference shared by CLI, MCP, Web and A2A; no network/LLM calls."""

from __future__ import annotations

import asyncio
import fcntl
import gzip
import importlib.metadata
import io
import json
import math
import os
import platform
import signal
import stat
import struct
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from difflib import get_close_matches
from pathlib import Path
from shutil import which
from typing import Literal

Task = Literal["total", "total_mr"]
DEFAULT_TIMEOUT_SECONDS = 7200
DEFAULT_MAX_INPUT_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
WARNING = "Research use only. Outputs require review; no clinical performance claim is made."


class SegmentationError(ValueError):
    """Expected failure; the private run directory contains durable audit details."""

    def __init__(self, message: str, *, run_dir: Path | None = None):
        super().__init__(message)
        self.run_dir = run_dir


def task_classes(task: str) -> set[str]:
    return set(task_labels(task).values())


def task_labels(task: str) -> dict[int, str]:
    from totalsegmentator.registry import get_task_classes

    if task not in {"total", "total_mr"}:
        raise SegmentationError("task must be 'total' (CT) or 'total_mr' (MR).")
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

    if task not in {"total", "total_mr"}:
        raise SegmentationError("DICOM requires an explicit CT task 'total' or MR task 'total_mr'.")
    modality = "CT" if task == "total" else "MR"
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
    path: Path, *, label_map: dict[int, str] | None = None, _stop_event=None
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
            for z in range(shape[2]):
                _check_stop(_stop_event)
                plane = np.asanyarray(image.dataobj[:, :, z])
                if not np.isfinite(plane).all():
                    raise SegmentationError("NIfTI voxel data contains NaN or infinite values.")
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
    # Shared by every adapter, regardless of output directory or Python process.
    path = Path(
        os.environ.get(
            "MEDSEGAGENT_LOCK_PATH", str(Path.home() / ".cache/medsegagent/inference.lock")
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SegmentationError("Timed out waiting for the local inference device.")
                await asyncio.sleep(0.1)
        yield descriptor
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _build_command(
    *, task: str, input_path: Path, output_dir: Path, targets: list[str] | None, speed: str
) -> list[str]:
    executable = which("TotalSegmentator")
    if executable is None:
        candidate = Path(os.sys.executable).parent / "TotalSegmentator"
        executable = str(candidate) if candidate.is_file() else None
    if executable is None:
        raise SegmentationError("TotalSegmentator executable was not found in the uv environment.")
    command = [
        executable,
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
    if speed == "fast":
        command.extend(["--fast", "--higher_order_resampling"])
    if targets is not None:
        command.extend(["--roi_subset", *targets])
    return command


async def _stop_process(process: asyncio.subprocess.Process) -> None:
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


def _inference_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
    }


async def _run_command(
    command: list[str], *, timeout_seconds: float, output_dir: Path, lock_fd: int, on_start
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
                env=_inference_environment(),
                pass_fds=(lock_fd,),
            )
        )
        try:
            process = await asyncio.shield(creation)
        except asyncio.CancelledError:
            # Cancellation can arrive after fork but before the subprocess handle returns.
            process = await creation
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
    speed: Literal["fast", "standard"] = "fast",
) -> dict[str, object]:
    """Run local inference; output_dir is a parent, and targets=None requests all anatomy.

    task explicitly declares CT/MR. DICOM directories are strictly preflighted and
    converted from a private snapshot; ZIP is rejected. Cancellation stops readers and
    subprocess groups before persisting terminal state.
    """
    run = _new_run(output_dir)
    started = time.monotonic()
    state: dict[str, object] = {
        "run_id": run.name,
        "status": "validating",
        "task": task,
        "created_at": datetime.now(UTC).isoformat(),
        "owner_pid": os.getpid(),
        "output_dir": str(run),
        "state_path": str(run / "state.json"),
    }

    def update(**fields):
        state.update(fields, updated_at=datetime.now(UTC).isoformat())
        _write_json(run / "state.json", state)

    update()
    (run / "process.log").touch(mode=0o600)
    try:
        if speed not in {"fast", "standard"}:
            raise SegmentationError("speed must be 'fast' or 'standard'.")
        normalized = normalize_targets(task, targets)
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
        update(status="queued", device=selected_device, targets=normalized, speed=speed)
        async with _inference_lock(deadline) as lock_fd:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SegmentationError(
                    "Task exceeded its timeout during input validation or queueing."
                )
            if dicom:
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
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SegmentationError("Task exceeded its timeout before inference.")
            command = _build_command(
                task=task, input_path=source, output_dir=run, targets=normalized, speed=speed
            )
            update(status="running")
            inference_started = time.monotonic()
            await _run_command(
                command,
                timeout_seconds=remaining,
                output_dir=run,
                lock_fd=lock_fd,
                on_start=lambda pid: update(process_pid=pid),
            )
            inference_seconds = time.monotonic() - inference_started
        segmentation_path = run / "segmentation.nii.gz"
        report_path = run / "run_report.json"
        if not segmentation_path.is_file() or not report_path.is_file():
            raise SegmentationError(
                "Inference finished without its required segmentation and run report."
            )
        geometry = await _validation(
            _inspect_nifti,
            segmentation_path,
            label_map=task_labels(task),
            timeout_seconds=deadline - time.monotonic(),
        )
        import nibabel as nib
        import numpy as np

        reference = nib.load(source)
        if list(reference.shape) != geometry["shape"] or not np.allclose(
            reference.affine, geometry["affine"], rtol=1e-5, atol=1e-4
        ):
            raise SegmentationError("Segmentation geometry does not match the input volume.")
        if normalized is not None and any(
            row["name"] not in normalized for row in geometry["labels"]
        ):
            raise SegmentationError(
                "Segmentation includes structures outside the requested targets."
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise SegmentationError("Inference run report is invalid.")
        result = {
            **state,
            "status": "completed",
            "segmentation_path": str(segmentation_path),
            "run_report": str(report_path),
            "targets": normalized if normalized is not None else "all",
            "segmentation_shape": geometry["shape"],
            "segmentation_voxel_spacing": geometry["voxel_spacing"],
            "segmentation_affine": geometry["affine"],
            "labels": geometry["labels"],
            "nonzero_voxels": geometry["nonzero_voxels"],
            "runtime_seconds": inference_seconds,
            "total_seconds": time.monotonic() - started,
            "totalsegmentator_version": importlib.metadata.version("TotalSegmentator"),
            "warning": WARNING,
        }
        _write_json(run / "result.json", result)
        update(status="completed", result=result)
        return result
    except asyncio.CancelledError:
        update(status="cancelled", error="Task was cancelled; its process group was stopped.")
        raise
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, SegmentationError)
            else "Local task failed; inspect its private run state."
        )
        update(status="failed", error=message, error_type=type(exc).__name__)
        raise SegmentationError(message, run_dir=run) from exc


def doctor() -> dict[str, object]:
    """Report installed runtime and hardware without running inference."""
    import torch

    return {
        "python": os.sys.version.split()[0],
        "totalsegmentator": importlib.metadata.version("TotalSegmentator"),
        "executable": which("TotalSegmentator"),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "configured_device": device(),
        "ct_class_count": len(task_classes("total")),
        "mr_class_count": len(task_classes("total_mr")),
        "output_root": str(Path(os.environ.get("MEDSEGAGENT_OUTPUT_ROOT", "outputs")).resolve()),
    }

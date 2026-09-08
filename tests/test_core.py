from __future__ import annotations

import asyncio
import gzip
import json
import multiprocessing
import os
import struct
import sys
import time
from itertools import pairwise
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from medsegagent import core


def image_at(path: Path, values=None) -> Path:
    values = np.zeros((3, 4, 5), dtype=np.float32) if values is None else values
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)
    return path


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    from medsegagent import weights

    monkeypatch.setattr(weights, "require_weights", lambda *args, **kwargs: {"ready": True})
    monkeypatch.setenv("MEDSEGAGENT_DEVICE", "cpu")
    monkeypatch.setenv("MEDSEGAGENT_LOCK_PATH", str(tmp_path / "inference.lock"))
    monkeypatch.setenv("MEDSEGAGENT_OUTPUT_ROOT", str(tmp_path / "outputs"))


@pytest.mark.parametrize("targets", [[], [""], ["  "], ["liver", ""], [None], "liver"])
def test_empty_or_invalid_targets_fail_closed(targets):
    with pytest.raises(core.SegmentationError):
        core.normalize_targets("total", targets)


def test_targets_only_omitted_means_all():
    assert core.normalize_targets("total", None) is None
    assert core.normalize_targets("total", [" liver ", "liver"]) == ["liver"]
    with pytest.raises(core.SegmentationError, match="kidney_left"):
        core.normalize_targets("total", ["kidny_left"])
    with pytest.raises(core.SegmentationError, match="task must"):
        core.normalize_targets("invalid", None)


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_nifti_acceptance_and_complete_read(tmp_path, suffix):
    path = image_at(tmp_path / f"valid{suffix}")
    assert core.validate_input(str(path)) == path
    raw = gzip.decompress(path.read_bytes()) if suffix.endswith("gz") else path.read_bytes()
    truncated = raw[:-4]
    path.write_bytes(gzip.compress(truncated) if suffix.endswith("gz") else truncated)
    with pytest.raises(core.SegmentationError, match="truncated"):
        core.validate_input(str(path))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nan_or_infinite_voxels_rejected(tmp_path, bad):
    data = np.zeros((3, 4, 5), dtype=np.float32)
    data[-1, -1, -1] = bad
    with pytest.raises(core.SegmentationError, match="voxel data"):
        core.validate_input(str(image_at(tmp_path / "bad.nii.gz", data)))


@pytest.mark.parametrize("spacing", [float("nan"), float("inf"), 0.0, -1.0])
def test_raw_header_invalid_spacing_rejected(tmp_path, spacing):
    path = image_at(tmp_path / "spacing.nii")
    data = bytearray(path.read_bytes())
    struct.pack_into("<f", data, 80, spacing)  # raw pixdim[1], before nibabel can repair it
    path.write_bytes(data)
    with pytest.raises(core.SegmentationError, match="spacing"):
        core.validate_input(str(path))


def test_nan_affine_rejected(tmp_path):
    path = image_at(tmp_path / "affine.nii")
    data = bytearray(path.read_bytes())
    struct.pack_into("<f", data, 280, float("nan"))  # srow_x[0]
    path.write_bytes(data)
    with pytest.raises(core.SegmentationError):
        core.validate_input(str(path))


def test_4d_and_unsupported_formats_rejected(tmp_path):
    with pytest.raises(core.SegmentationError, match="3D"):
        core.validate_input(str(image_at(tmp_path / "4d.nii", np.zeros((2, 3, 4, 5)))))
    directory = tmp_path / "dicom"
    directory.mkdir()
    for path in (directory, tmp_path / "study.zip", tmp_path / "file.txt"):
        if path != directory:
            path.touch()
        with pytest.raises(core.SegmentationError, match="DICOM|NIfTI"):
            core.validate_input(str(path))


def test_compressed_and_declared_size_limits(tmp_path, monkeypatch):
    path = image_at(tmp_path / "large.nii.gz")
    monkeypatch.setenv("MEDSEGAGENT_MAX_UNCOMPRESSED_BYTES", "360")
    with pytest.raises(core.SegmentationError, match="size limit"):
        core.validate_input(str(path))
    monkeypatch.setenv("MEDSEGAGENT_MAX_INPUT_BYTES", "10")
    with pytest.raises(core.SegmentationError, match="file size limit"):
        core.validate_input(str(path))


def test_corrupt_gzip_trailer_rejected(tmp_path):
    path = image_at(tmp_path / "checksum.nii.gz")
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(core.SegmentationError, match="complete"):
        core.validate_input(str(path))


def test_command_preserves_task_device_and_subset(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "which", lambda _: "/venv/bin/TotalSegmentator")
    monkeypatch.setenv("MEDSEGAGENT_DEVICE", "gpu")
    command = core._build_command(
        task="total_mr",
        input_path=tmp_path / "mr.nii",
        output_dir=tmp_path,
        targets=["liver"],
        speed="fast",
    )
    assert command[command.index("--task") + 1] == "total_mr"
    assert command[command.index("--device") + 1] == "gpu"
    assert command[-2:] == ["--roi_subset", "liver"]
    assert "--statistics" not in command


async def fake_inference(command, *, output_dir, on_start, **kwargs):
    on_start(os.getpid())
    liver = next(key for key, value in core.task_labels("total").items() if value == "liver")
    image_at(output_dir / "segmentation.nii.gz", np.full((3, 4, 5), liver, dtype=np.uint8))
    (output_dir / "run_report.json").write_text('{"runtime_seconds": 1.25, "status": "evil"}')
    await asyncio.sleep(0.01)


@pytest.mark.parametrize("cancel", [False, True])
def test_current_inference_stage_is_recorded_after_failure_or_cancellation(
    tmp_path, monkeypatch, cancel
):
    source = image_at(tmp_path / "ct.nii.gz")

    async def failing_inference(*args, **kwargs):
        await asyncio.sleep(0.01)
        if cancel:
            raise asyncio.CancelledError()
        raise core.SegmentationError("inference stopped")

    monkeypatch.setattr(core, "_run_command", failing_inference)
    expected = asyncio.CancelledError if cancel else core.SegmentationError
    with pytest.raises(expected) as error:
        asyncio.run(
            core.segment(
                task="total",
                input_path=str(source),
                targets=["liver"],
                output_dir=str(tmp_path / "runs"),
            )
        )
    assert error.value.timings_seconds["inference_subprocess"] >= 0.01
    assert error.value.timings_seconds["validation"] > 0
    assert error.value.inference_engine == "sequential"
    state = core.read_run(next((tmp_path / "runs").iterdir()))
    assert state["timings_seconds"] == error.value.timings_seconds
    assert state["status"] == ("cancelled" if cancel else "failed")


def test_segment_persists_result_and_validation_failures(tmp_path, monkeypatch):
    source = image_at(tmp_path / "ct.nii.gz")
    monkeypatch.setattr(core, "_run_command", fake_inference)
    result = asyncio.run(
        core.segment(
            task="total",
            input_path=str(source),
            output_dir=str(tmp_path / "same-parent"),
            targets=["liver"],
        )
    )
    run = Path(result["output_dir"])
    assert run.parent == tmp_path / "same-parent"
    assert result["status"] == "completed"  # report must not overwrite trusted metadata
    assert result["nonzero_voxels"] == 60
    assert result["labels"][0]["name"] == "liver"
    assert result["labels"][0]["voxels"] == 60
    assert core.read_run(run)["status"] == "completed"
    assert core.read_run(run)["result"]["run_id"] == result["run_id"]
    assert not (run / "result.json").exists()
    assert run.stat().st_mode & 0o777 == 0o700
    assert (run / "state.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(core.SegmentationError) as error:
        asyncio.run(core.segment(task="total", input_path=str(source), targets=[]))
    assert core.read_run(error.value.run_dir)["status"] == "failed"
    assert (error.value.run_dir / "process.log").is_file()


def test_same_parent_concurrent_requests_have_unique_runs(tmp_path, monkeypatch):
    source = image_at(tmp_path / "ct.nii.gz")
    active = 0
    peak = 0

    async def observed(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await fake_inference(*args, **kwargs)
        active -= 1

    monkeypatch.setattr(core, "_run_command", observed)

    async def concurrent():
        return await asyncio.gather(
            *(
                core.segment(task="total", input_path=str(source), targets=["liver"])
                for _ in range(3)
            )
        )

    results = asyncio.run(concurrent())
    assert len({row["output_dir"] for row in results}) == 3
    assert peak == 1


def _process_lock_worker(root: str, queue):
    async def work():
        run = core._new_run(root)
        async with core._inference_lock(time.monotonic() + 10):
            start = time.monotonic()
            await asyncio.sleep(0.2)
            queue.put((str(run), start, time.monotonic()))

    asyncio.run(work())


def test_cross_process_lock_and_atomic_run_allocation(tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_process_lock_worker, args=(str(tmp_path / "parent"), queue))
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert len({row[0] for row in results}) == 3
    intervals = sorted((row[1], row[2]) for row in results)
    assert all(previous[1] <= current[0] for previous, current in pairwise(intervals))


@pytest.mark.parametrize("mode", ["timeout", "cancel", "failure"])
def test_process_group_stopped_and_log_preserved(tmp_path, mode):
    run = core._new_run(str(tmp_path / "runs"))
    marker = tmp_path / "escaped-child"
    child_code = f"import time,pathlib;time.sleep(1);pathlib.Path({str(marker)!r}).touch()"
    parent_code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        "print('started',flush=True);time.sleep(10)"
    )

    if mode == "failure":
        parent_code = parent_code.replace("time.sleep(10)", "sys.exit(7)")

    async def run_and_stop():
        async with core._inference_lock(time.monotonic() + 5) as lock_fd:
            ready = asyncio.Event()
            task = asyncio.create_task(
                core._run_command(
                    [sys.executable, "-c", parent_code],
                    timeout_seconds=0.2 if mode == "timeout" else 10,
                    output_dir=run,
                    lock_fd=lock_fd,
                    on_start=lambda pid: ready.set(),
                )
            )
            await ready.wait()
            if mode == "cancel":
                await asyncio.sleep(0.2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                expected = "timeout" if mode == "timeout" else "exit code 7"
                with pytest.raises(core.SegmentationError, match=expected):
                    await task
        await asyncio.sleep(1)

    asyncio.run(run_and_stop())
    assert not marker.exists()
    assert "started" in (run / "process.log").read_text()


def test_cancelled_run_state_and_expected_process_failure(tmp_path, monkeypatch):
    source = image_at(tmp_path / "ct.nii.gz")

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(core, "_run_command", cancelled)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(core.segment(task="total", input_path=str(source)))
    state = next((tmp_path / "outputs").glob("*/state.json"))
    assert json.loads(state.read_text())["status"] == "cancelled"


def test_inference_process_does_not_receive_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("MEDSEGAGENT_TOKENS_JSON", "synthetic-secret")
    monkeypatch.setenv("MEDSEGAGENT_DEVICE", "mps")
    environment = core._inference_environment()
    assert "OPENAI_API_KEY" not in environment
    assert "MEDSEGAGENT_TOKENS_JSON" not in environment
    assert environment["MEDSEGAGENT_DEVICE"] == "mps"


@pytest.mark.parametrize("failure", ["geometry", "labels", "nan"])
def test_invalid_result_is_rejected_and_audited(tmp_path, monkeypatch, failure):
    source = image_at(tmp_path / "ct.nii.gz")

    async def invalid(command, **kwargs):
        await fake_inference(command, **kwargs)
        run = kwargs["output_dir"]
        values = np.ones((3, 4, 5), dtype=np.float32)
        affine = np.eye(4)
        if failure == "geometry":
            affine[0, 3] = 10
        elif failure == "labels":
            values[:] = 255
        else:
            values[-1, -1, -1] = np.nan
        nib.save(nib.Nifti1Image(values, affine), run / "segmentation.nii.gz")

    monkeypatch.setattr(core, "_run_command", invalid)
    with pytest.raises(core.SegmentationError) as error:
        asyncio.run(core.segment(task="total", input_path=str(source)))
    assert core.read_run(error.value.run_dir)["status"] == "failed"
    assert not (error.value.run_dir / "result.json").exists()


def test_waiting_for_device_has_a_deadline(tmp_path):
    async def timed():
        async with core._inference_lock(time.monotonic() + 5):
            with pytest.raises(core.SegmentationError, match="waiting"):
                async with core._inference_lock(time.monotonic() + 0.1):
                    pytest.fail("Second lock must not enter while first lock is held")

    asyncio.run(timed())


def test_cancellation_during_subprocess_creation_stops_created_child(tmp_path, monkeypatch):
    run = core._new_run(str(tmp_path / "runs"))
    marker = tmp_path / "creation-race"
    command = [
        sys.executable,
        "-c",
        f"import pathlib,time;time.sleep(0.6);pathlib.Path({str(marker)!r}).touch()",
    ]
    original_create = asyncio.create_subprocess_exec

    async def race():
        started = asyncio.Event()

        async def delayed_return(*args, **kwargs):
            process = await original_create(*args, **kwargs)
            started.set()
            await asyncio.sleep(0.1)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_return)
        async with core._inference_lock(time.monotonic() + 5) as lock_fd:
            task = asyncio.create_task(
                core._run_command(
                    command,
                    timeout_seconds=5,
                    output_dir=run,
                    lock_fd=lock_fd,
                    on_start=lambda pid: None,
                )
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await asyncio.sleep(0.7)

    asyncio.run(race())
    assert not marker.exists()


def dicom_at(directory: Path, *, modality="CT") -> Path:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid

    directory.mkdir()
    study, series = generate_uid(), generate_uid()
    for index in range(3):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage if modality == "CT" else MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\x00" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PatientID = "SYNTHETIC"
        ds.PatientName = "Synthetic^Canary"
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series
        ds.Modality = modality
        ds.StudyDate, ds.StudyTime = "20260101", "120000"
        ds.SeriesNumber, ds.AcquisitionNumber, ds.InstanceNumber = 1, 1, index + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.Rows, ds.Columns = 4, 3
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.PixelSpacing, ds.SliceThickness = [1.0, 1.0], 2.0
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, float(index * 2)]
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = np.full((4, 3), index + 1, dtype=np.int16).tobytes()
        ds.save_as(directory / f"slice-{index}.dcm", enforce_file_format=True)
    return directory


@pytest.mark.parametrize("task,modality", [("total", "CT"), ("total_mr", "MR")])
def test_dicom_directory_requires_matching_modality_and_explicit_task(tmp_path, task, modality):
    source = dicom_at(tmp_path / "dicom", modality=modality)
    assert core.validate_input(str(source), task) == source
    with pytest.raises(core.SegmentationError, match="explicit"):
        core.validate_input(str(source))
    other_task = "total_mr" if task == "total" else "total"
    with pytest.raises(core.SegmentationError, match="modality"):
        core.validate_input(str(source), other_task)


@pytest.mark.parametrize(
    "problem",
    [
        "series",
        "study",
        "patient",
        "truncated",
        "position",
        "nan",
        "multiframe",
        "compressed",
        "duplicate",
    ],
)
def test_dicom_mixed_or_invalid_data_fail_before_conversion(tmp_path, problem):
    import pydicom
    from pydicom.encaps import encapsulate
    from pydicom.uid import JPEG2000Lossless, generate_uid

    source = dicom_at(tmp_path / "dicom")
    path = source / "slice-1.dcm"
    ds = pydicom.dcmread(path)
    if problem == "series":
        ds.SeriesInstanceUID = generate_uid()
    elif problem == "study":
        ds.StudyInstanceUID = generate_uid()
    elif problem == "patient":
        ds.PatientID = "ANOTHER-SYNTHETIC-PATIENT"
    elif problem == "truncated":
        ds.PixelData = ds.PixelData[:-2]
    elif problem == "position":
        ds.ImagePositionPatient = [0, 0, 0]
    elif problem == "nan":
        ds.PixelSpacing = [float("nan"), 1]
    elif problem == "multiframe":
        ds.NumberOfFrames = 2
    elif problem == "compressed":
        ds.file_meta.TransferSyntaxUID = JPEG2000Lossless
        ds.PixelData = encapsulate([b"synthetic-compressed-data"])
    elif problem == "duplicate":
        ds.SOPInstanceUID = pydicom.dcmread(source / "slice-0.dcm").SOPInstanceUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(core.SegmentationError):
        core.validate_input(str(source), "total")


@pytest.mark.parametrize("problem", ["text", "symlink", "nested"])
def test_dicom_rejects_non_image_files_and_unsafe_directory_entries(tmp_path, problem):
    source = dicom_at(tmp_path / "dicom")
    if problem == "text":
        (source / "notes.txt").write_text("Not an image")
    elif problem == "symlink":
        (source / "linked.dcm").symlink_to(source / "slice-0.dcm")
    else:
        (source / "nested").mkdir()
    with pytest.raises(core.SegmentationError):
        core.validate_input(str(source), "total")


def test_dicom_snapshot_has_exact_validated_bytes_and_generated_names(tmp_path):
    source = dicom_at(tmp_path / "source")
    snapshot = tmp_path / "private-snapshot"
    metadata = core._inspect_dicom_directory(source, "total", snapshot_dir=snapshot)
    assert metadata == {"modality": "CT", "instances": 3}
    assert [path.name for path in sorted(snapshot.iterdir())] == [
        "000000.dcm",
        "000001.dcm",
        "000002.dcm",
    ]
    for original, copied in zip(sorted(source.iterdir()), sorted(snapshot.iterdir()), strict=True):
        assert original.read_bytes() == copied.read_bytes()
        assert copied.stat().st_mode & 0o777 == 0o600


def test_real_dcm2niix_conversion_then_mock_inference_preserves_geometry(tmp_path, monkeypatch):
    converter = Path(sys.executable).parent / "dcm2niix"
    if not converter.is_file() and core.which("dcm2niix") is None:
        pytest.skip("dcm2niix is not installed")
    source = dicom_at(tmp_path / "source")
    original_run = core._run_command
    seen = []

    async def convert_then_fake(command, **kwargs):
        seen.append(command)
        if Path(command[0]).name == "dcm2niix":
            return await original_run(command, **kwargs)
        input_path = Path(command[command.index("-i") + 1])
        assert input_path.suffix == ".gz" and input_path.parent.name == "converted"
        image = nib.load(input_path)
        liver = next(index for index, name in core.task_labels("total").items() if name == "liver")
        values = np.full(image.shape, liver, dtype=np.uint8)
        nib.save(
            nib.Nifti1Image(values, image.affine), kwargs["output_dir"] / "segmentation.nii.gz"
        )
        (kwargs["output_dir"] / "run_report.json").write_text("{}")

    monkeypatch.setattr(core, "_run_command", convert_then_fake)
    result = asyncio.run(core.segment(task="total", input_path=str(source), targets=["liver"]))
    assert result["status"] == "completed"
    assert result["input_format"] == "dicom"
    assert result["dicom_instances"] == 3
    assert sorted(result["segmentation_shape"]) == [3, 3, 4]
    assert len(seen) == 2
    assert seen[1][seen[1].index("--task") + 1] == "total"
    assert "-b" in seen[0] and seen[0][seen[0].index("-b") + 1] == "n"


def test_async_validation_is_responsive_and_waits_for_cancelled_reader(tmp_path, monkeypatch):
    import threading

    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()

    def reader(*args, **kwargs):
        entered.set()
        release.wait(3)
        stopped.set()
        return tmp_path

    monkeypatch.setattr(core, "validate_input", reader)

    async def scenario():
        task = asyncio.create_task(core.validate_input_async("synthetic.nii"))
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()

    asyncio.run(scenario())


def test_gzip_validation_uses_anonymous_temporary_storage(tmp_path):
    path = image_at(tmp_path / "input.nii.gz")
    with core._uncompressed_nifti(path, 1024 * 1024) as stream:
        # Unlinked inode cannot survive the last descriptor closing, including a hard crash.
        assert os.fstat(stream.fileno()).st_nlink == 0
        assert stream.read(4) == struct.pack("<i", 348)


def test_task_policy_preserves_original_defaults_and_covers_installed_registry():
    from dataclasses import FrozenInstanceError

    from totalsegmentator.registry import TASKS

    from medsegagent.task_specs import TASK_SPECS

    assert set(TASK_SPECS) == set(TASKS)
    assert TASK_SPECS["total"].default_speed == "fast"
    assert TASK_SPECS["total_mr"].modality == "MR"
    assert TASK_SPECS["lung_nodules"].tool == "segment_lung_nodules"
    assert TASK_SPECS["liver_lesions"].default_targets == ("liver_lesions",)
    assert core.task_labels("lung_nodules") == {1: "lung", 2: "lung_nodules"}
    with pytest.raises(FrozenInstanceError):
        TASK_SPECS["lung_nodules"].modality = "MR"
    with pytest.raises(core.SegmentationError):
        core.task_classes("not_a_registered_task")


@pytest.mark.parametrize("task", ["lung_nodules", "liver_lesions"])
def test_specialized_commands_never_pass_fast_or_roi_flags(tmp_path, task):
    command = core._build_command(
        task=task,
        input_path=tmp_path / "scan.nii.gz",
        output_dir=tmp_path,
        targets=[task],
        speed="standard",
    )
    assert command[command.index("--task") + 1] == task
    assert not {"--fast", "--higher_order_resampling", "--roi_subset"}.intersection(command)
    with pytest.raises(core.SegmentationError, match="standard"):
        core._build_command(
            task=task,
            input_path=tmp_path / "scan.nii.gz",
            output_dir=tmp_path,
            targets=[task],
            speed="fast",
        )
    with pytest.raises(core.SegmentationError, match="standard"):
        asyncio.run(
            core.segment(task=task, input_path="unused.nii.gz", targets=[task], speed="fast")
        )
    with pytest.raises(core.SegmentationError, match="non-empty"):
        asyncio.run(core.segment(task=task, input_path="unused.nii.gz", targets=[]))


async def fake_specialized(command, *, output_dir, on_start, **kwargs):
    task = command[command.index("--task") + 1]
    values = np.zeros((3, 4, 5), dtype=np.uint8)
    if task == "lung_nodules":
        values[:] = 1  # native task emits a lung label as well as the requested nodules
        values[0, 0, :2] = 2
    else:
        values[0, 0, :2] = 1
    on_start(os.getpid())
    image_at(output_dir / "segmentation.nii.gz", values)
    (output_dir / "run_report.json").write_text("{}")


@pytest.mark.parametrize("task", ["lung_nodules", "liver_lesions"])
def test_specialized_default_speed_and_audited_normalization(tmp_path, monkeypatch, task):
    source = image_at(tmp_path / "ct.nii.gz")
    monkeypatch.setattr(core, "_run_command", fake_specialized)
    result = asyncio.run(core.segment(task=task, input_path=str(source), targets=[task]))
    run = Path(result["output_dir"])
    assert result["status"] == "completed"
    assert result["speed"] == "standard"
    assert result["detection_status"] == "target_detected"
    assert result["no_target_detected"] is False
    assert result["nonzero_voxels"] == 2
    assert [label["name"] for label in result["labels"]] == [task]
    expected_label = 2 if task == "lung_nodules" else 1
    assert set(np.unique(np.asanyarray(nib.load(result["segmentation_path"]).dataobj))) == {
        0,
        1,
    }
    assert result["labels"][0]["source_id"] == expected_label
    assert result["labels"][0]["color"] == "#ff0000"
    audit = json.loads((run / "normalization.json").read_text())
    assert audit["retained_label_ids"] == [expected_label]
    assert audit["geometry_preserved"] is True
    assert audit["raw_sha256"] == core._file_digest(run / "segmentation.raw.nii.gz")
    assert audit["segmentation_sha256"] == core._file_digest(run / "segmentation.nii.gz")
    assert not list(run.glob(".normalized-*"))
    assert (run / "normalization.json").stat().st_mode & 0o777 == 0o600


def test_empty_nodule_prediction_is_legal_and_does_not_exclude_disease(tmp_path, monkeypatch):
    source = image_at(tmp_path / "ct.nii.gz")

    async def lung_only(command, **kwargs):
        await fake_specialized(command, **kwargs)
        image_at(kwargs["output_dir"] / "segmentation.nii.gz", np.ones((3, 4, 5), dtype=np.uint8))

    monkeypatch.setattr(core, "_run_command", lung_only)
    result = asyncio.run(
        core.segment(task="lung_nodules", input_path=str(source), targets=["lung_nodules"])
    )
    assert result["status"] == "completed"
    assert result["detection_status"] == "no_target_detected"
    assert result["no_target_detected"] is True
    assert len(result["labels"]) == 1
    assert result["labels"][0]["id"] == 1
    assert result["labels"][0]["name"] == "lung_nodules"
    assert result["labels"][0]["voxels"] == 0
    assert result["nonzero_voxels"] == 0
    assert "does not rule out disease" in result["warning"]
    assert result["normalization"]["raw_nonzero_voxels"] == 60
    assert result["normalization"]["normalized_nonzero_voxels"] == 0


def test_native_specialized_mask_is_validated_before_normalization(tmp_path, monkeypatch):
    source = image_at(tmp_path / "ct.nii.gz")

    async def invalid(command, **kwargs):
        await fake_specialized(command, **kwargs)
        image_at(
            kwargs["output_dir"] / "segmentation.nii.gz", np.full((3, 4, 5), 99, dtype=np.uint8)
        )

    monkeypatch.setattr(core, "_run_command", invalid)
    with pytest.raises(core.SegmentationError, match="outside") as error:
        asyncio.run(
            core.segment(task="lung_nodules", input_path=str(source), targets=["lung_nodules"])
        )
    assert not (error.value.run_dir / "segmentation.raw.nii.gz").exists()
    assert not (error.value.run_dir / "result.json").exists()


@pytest.mark.parametrize("version", [1, 2])
def test_normalization_preserves_voxel_locations_and_rotated_geometry(tmp_path, version):
    path = tmp_path / "segmentation.nii.gz"
    values = (np.arange(60).reshape((3, 4, 5)) % 3).astype(np.uint8)
    affine = np.array([[0, -1, 0, 3], [2, 0, 0, 7], [0, 0, 3, -5], [0, 0, 0, 1]], dtype=float)
    constructor = nib.Nifti1Image if version == 1 else nib.Nifti2Image
    nib.save(constructor(values, affine), path)
    original_sha = core._file_digest(path)
    result = core._normalize_segmentation(
        path,
        labels=core.task_labels("lung_nodules"),
        targets=["lung_nodules"],
        allow_unrequested=True,
    )
    filtered = nib.load(path)
    np.testing.assert_array_equal(np.asanyarray(filtered.dataobj), np.where(values == 2, 1, 0))
    np.testing.assert_array_equal(filtered.affine, affine)
    assert result["audit"]["raw_sha256"] == original_sha
    assert result["geometry"]["shape"] == [3, 4, 5]


@pytest.mark.parametrize("mode", ["cancel", "timeout"])
def test_normalization_cancellation_and_timeout_preserve_raw_without_partial_public_mask(
    tmp_path, monkeypatch, mode
):
    import threading
    from contextlib import contextmanager

    source = image_at(tmp_path / "ct.nii.gz")
    monkeypatch.setattr(core, "_run_command", fake_specialized)
    if mode == "timeout":
        monkeypatch.setenv("MEDSEGAGENT_TIMEOUT_SECONDS", "1")
    entered = threading.Event()
    original = core._uncompressed_nifti

    @contextmanager
    def observed(path, limit, _stop_event=None):
        with original(path, limit, _stop_event) as stream:
            if path.name == "segmentation.raw.nii.gz":
                entered.set()
                assert _stop_event is not None
                assert _stop_event.wait(3), "Normalization cancellation did not signal its reader"
            yield stream

    monkeypatch.setattr(core, "_uncompressed_nifti", observed)

    async def scenario():
        operation = asyncio.create_task(
            core.segment(task="lung_nodules", input_path=str(source), targets=["lung_nodules"])
        )
        assert await asyncio.to_thread(entered.wait, 2)
        if mode == "cancel":
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            with pytest.raises(core.SegmentationError, match="timeout"):
                await operation

    asyncio.run(scenario())
    run = next((tmp_path / "outputs").iterdir())
    assert core.read_run(run)["status"] == ("cancelled" if mode == "cancel" else "failed")
    assert (run / "segmentation.raw.nii.gz").is_file()
    assert not (run / "segmentation.nii.gz").exists()
    assert not (run / "result.json").exists()
    assert not list(run.glob(".normalized-*"))


@pytest.mark.parametrize("task", ["lung_nodules", "liver_lesions"])
def test_dicom_preflight_uses_specialized_task_modality(tmp_path, task):
    ct = dicom_at(tmp_path / "ct", modality="CT")
    mr = dicom_at(tmp_path / "mr", modality="MR")
    assert core.validate_input(str(ct), task) == ct
    with pytest.raises(core.SegmentationError, match="modality"):
        core.validate_input(str(mr), task)


@pytest.mark.parametrize(
    "targets,expected_name,expected_voxels",
    [
        (None, "lung_nodules", 2),
        (["lung"], "lung", 58),
    ],
)
def test_lung_defaults_to_nodules_but_core_accepts_explicit_native_class(
    tmp_path, monkeypatch, targets, expected_name, expected_voxels
):
    source = image_at(tmp_path / "ct.nii.gz")
    monkeypatch.setattr(core, "_run_command", fake_specialized)
    result = asyncio.run(core.segment(task="lung_nodules", input_path=str(source), targets=targets))
    assert result["targets"] == [expected_name]
    assert [label["name"] for label in result["labels"]] == [expected_name]
    assert result["nonzero_voxels"] == expected_voxels
    assert Path(result["normalization"]["raw_segmentation_path"]).stat().st_mode & 0o777 == 0o600


def test_normalization_measures_volume_and_preserves_every_selected_voxel(tmp_path):
    values = np.zeros((6, 6, 6), dtype=np.uint8)
    values[0, 0, 0] = values[1, 1, 1] = 7
    values[4, 4, 4] = values[4, 4, 5] = values[4, 5, 5] = 7
    values[0, 5, 0] = 7  # retain this single-voxel region
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm")
    path = tmp_path / "segmentation.nii.gz"
    nib.save(image, path)
    result = core._normalize_segmentation(path, labels={7: "organ"}, targets=["organ"])
    row = result["geometry"]["labels"][0]
    assert row == {
        "id": 1,
        "source_id": 7,
        "name": "organ",
        "color": "#ff0000",
        "voxels": 6,
        "volume_mm3": 144.0,
        "volume_ml": 0.144,
    }
    measurement = result["volume_measurement"]
    assert measurement["source_spatial_unit"] == "mm"
    assert measurement["unit_assumption"] is None
    np.testing.assert_array_equal(nib.load(path).get_fdata(), values == 7)
    assert result["audit"]["postprocessing"] == "none"


@pytest.mark.parametrize("targets", [None, ["right", "left", "center"]])
def test_normalization_mapping_includes_empty_classes_and_is_stable(tmp_path, targets):
    labels = {30: "right", 3: "left", 7: "center"}
    mappings = []
    for run_index, present in enumerate((7, 30)):
        run = tmp_path / str(run_index)
        run.mkdir()
        path = image_at(run / "segmentation.nii.gz", np.full((3, 4, 5), present, dtype=np.uint8))
        result = core._normalize_segmentation(path, labels=labels, targets=targets)
        rows = result["geometry"]["labels"]
        mappings.append(result["audit"]["label_mapping"])
        assert [row["id"] for row in rows] == [1, 2, 3]
        assert [row["source_id"] for row in rows] == [3, 7, 30]
        assert [row["name"] for row in rows] == ["left", "center", "right"]
        assert [row["voxels"] for row in rows] == ([0, 60, 0] if present == 7 else [0, 0, 60])
        assert np.unique(nib.load(path).get_fdata()).tolist() == [2 if present == 7 else 3]
    assert mappings[0] == mappings[1]


@pytest.mark.parametrize(
    "unit,factor", [("mm", 1), ("meter", 1000), ("micron", 0.001), ("unknown", 1)]
)
def test_normalization_respects_spatial_units_and_declares_unknown_assumption(
    tmp_path, unit, factor
):
    path = tmp_path / "segmentation.nii.gz"
    image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.diag([2.0, 3.0, 4.0, 1.0]))
    image.header.set_xyzt_units(unit)
    nib.save(image, path)
    result = core._normalize_segmentation(path, labels={1: "organ"}, targets=None)
    row = result["geometry"]["labels"][0]
    assert row["volume_mm3"] == pytest.approx(8 * 24 * factor**3)
    assert row["volume_ml"] == pytest.approx(row["volume_mm3"] / 1000)
    assert result["volume_measurement"]["spacing_mm"] == [2 * factor, 3 * factor, 4 * factor]
    assert result["volume_measurement"]["unit_assumption"] == (
        "assumed_mm" if unit == "unknown" else None
    )
    assert nib.load(path).header.get_xyzt_units()[0] == unit


@pytest.mark.parametrize("version", [1, 2])
def test_normalization_replaces_native_extension_and_retains_qform_sform(tmp_path, version):
    from xml.etree import ElementTree

    path = tmp_path / "segmentation.nii.gz"
    constructor = nib.Nifti1Image if version == 1 else nib.Nifti2Image
    affine = np.array([[0, -1, 0, 3], [2, 0, 0, 7], [0, 0, 3, -5], [0, 0, 0, 1]], dtype=float)
    image = constructor(np.full((3, 4, 5), 9, dtype=np.float32), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=2)
    image.header.extensions.append(nib.nifti1.Nifti1Extension(0, b'<native-label id="9"/>'))
    nib.save(image, path)
    result = core._normalize_segmentation(path, labels={9: "A & B"}, targets=["A & B"])
    normalized = nib.load(path)
    assert normalized.get_data_dtype() == np.dtype("uint8")
    assert normalized.header.get_intent()[0] == "label"
    for form in ("qform", "sform"):
        expected, expected_code = getattr(image, f"get_{form}")(coded=True)
        actual, actual_code = getattr(normalized, f"get_{form}")(coded=True)
        np.testing.assert_allclose(actual, expected)
        assert actual_code == expected_code
    assert len(normalized.header.extensions) == 1
    xml = ElementTree.fromstring(normalized.header.extensions[0].get_content())
    entries = xml.findall("./VolumeInformation/LabelTable/Label")
    assert [entry.attrib["Key"] for entry in entries] == ["0", "1"]
    assert entries[1].text == "A & B"
    assert [entries[1].attrib[key] for key in ("Red", "Green", "Blue", "Alpha")] == [
        "1.0",
        "0.0",
        "0.0",
        "1",
    ]
    raw = nib.load(result["audit"]["raw_segmentation_path"])
    assert raw.header.extensions[0].get_content() == b'<native-label id="9"/>'
    assert np.unique(raw.get_fdata()).tolist() == [9]


@pytest.mark.parametrize("task", ["total", "total_mr", "lung_nodules", "liver_lesions"])
def test_every_model_uses_the_same_standardized_core_contract(tmp_path, monkeypatch, task):
    target = "liver" if task in {"total", "total_mr"} else task
    native_id = next(index for index, name in core.task_labels(task).items() if name == target)
    source = image_at(tmp_path / "source.nii.gz")

    async def inference(command, *, output_dir, on_start, **kwargs):
        on_start(os.getpid())
        image_at(output_dir / "segmentation.nii.gz", np.full((3, 4, 5), native_id, dtype=np.uint8))
        (output_dir / "run_report.json").write_text("{}")

    monkeypatch.setattr(core, "_run_command", inference)
    result = asyncio.run(core.segment(task=task, input_path=str(source), targets=[target]))
    assert result["schema_version"] == 3
    assert result["normalization_seconds"] > 0
    assert result["total_seconds"] >= result["runtime_seconds"] + result["normalization_seconds"]
    assert result["labels"][0]["id"] == 1
    assert result["labels"][0]["source_id"] == native_id
    assert result["labels"][0]["color"] == "#ff0000"
    assert set(result["labels"][0]) == {
        "id",
        "source_id",
        "name",
        "color",
        "voxels",
        "volume_mm3",
        "volume_ml",
    }
    assert set(result["volume_measurement"]) == {
        "method",
        "spacing_mm",
        "voxel_volume_mm3",
        "source_spatial_unit",
        "unit_assumption",
    }
    assert result["volume_measurement"]["unit_assumption"] == "assumed_mm"
    run = Path(result["output_dir"])
    assert (run / "normalization.json").is_file()
    assert (run / "segmentation.raw.nii.gz").is_file()
    assert not (run / "result.json").exists()
    assert core.read_run(run)["result"]["labels"] == result["labels"]


@pytest.mark.parametrize("invalid", [-1.0, 0.5, float("nan"), float("inf"), 99.0])
def test_invalid_native_labels_fail_before_standardization(tmp_path, invalid):
    path = image_at(tmp_path / "segmentation.nii.gz", np.full((3, 4, 5), invalid, dtype=np.float32))
    original = path.read_bytes()
    with pytest.raises(core.SegmentationError):
        core._normalize_segmentation(path, labels={1: "organ"}, targets=None)
    assert path.read_bytes() == original
    assert not (tmp_path / "segmentation.raw.nii.gz").exists()
    assert not list(tmp_path.glob(".normalized-*"))


def test_unrequested_anatomy_labels_are_not_silently_discarded(tmp_path):
    path = image_at(tmp_path / "segmentation.nii.gz", np.full((3, 4, 5), 2, dtype=np.uint8))
    with pytest.raises(core.SegmentationError, match="outside the requested"):
        core._normalize_segmentation(
            path, labels={1: "requested", 2: "unrequested"}, targets=["requested"]
        )
    assert not (tmp_path / "segmentation.raw.nii.gz").exists()


def test_final_validation_failure_never_publishes_a_partial_mask(tmp_path, monkeypatch):
    path = image_at(tmp_path / "segmentation.nii.gz", np.ones((3, 4, 5), dtype=np.uint8))
    original = path.read_bytes()
    inspect = core._inspect_nifti

    def failed(candidate, **kwargs):
        if candidate.name.startswith(".normalized-"):
            raise core.SegmentationError("Synthetic final validation failure")
        return inspect(candidate, **kwargs)

    monkeypatch.setattr(core, "_inspect_nifti", failed)
    with pytest.raises(core.SegmentationError, match="final validation failure"):
        core._normalize_segmentation(path, labels={1: "organ"}, targets=None)
    assert not path.exists()
    assert (tmp_path / "segmentation.raw.nii.gz").read_bytes() == original
    assert not list(tmp_path.glob(".normalized-*"))


def test_cancellation_waits_for_final_validation_before_reclaiming_files(tmp_path, monkeypatch):
    import threading

    path = image_at(tmp_path / "segmentation.nii.gz", np.ones((3, 4, 5), dtype=np.uint8))
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    inspect = core._inspect_nifti

    def observed(candidate, **kwargs):
        if candidate.name.startswith(".normalized-"):
            entered.set()
            assert release.wait(3)
            stopped.set()
        return inspect(candidate, **kwargs)

    monkeypatch.setattr(core, "_inspect_nifti", observed)

    async def scenario():
        operation = asyncio.create_task(
            core._validation(
                core._normalize_segmentation,
                path,
                labels={1: "organ"},
                targets=None,
            )
        )
        assert await asyncio.to_thread(entered.wait, 2)
        operation.cancel()
        await asyncio.sleep(0.01)
        assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert stopped.is_set()

    asyncio.run(scenario())
    assert not path.exists()
    assert (tmp_path / "segmentation.raw.nii.gz").is_file()
    assert not list(tmp_path.glob(".normalized-*"))


def test_standardization_rejects_nonidentity_native_scaling(tmp_path):
    path = tmp_path / "segmentation.nii.gz"
    image = nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.uint8), np.eye(4))
    image.header.set_slope_inter(2, 0)
    nib.save(image, path)
    with pytest.raises(core.SegmentationError, match="scaling must be identity"):
        core._normalize_segmentation(path, labels={2: "organ"}, targets=None)
    assert not path.exists()
    assert (tmp_path / "segmentation.raw.nii.gz").is_file()


def test_volume_measurement_rejects_overflow_in_convertible_spacing(tmp_path):
    path = tmp_path / "segmentation.nii.gz"
    image = nib.Nifti2Image(np.ones((2, 2, 2), dtype=np.uint8), np.diag([1e100, 1e100, 1e100, 1.0]))
    image.header.set_xyzt_units("meter")
    nib.save(image, path)
    with pytest.raises(core.SegmentationError, match="finite physical volume"):
        core._normalize_segmentation(path, labels={1: "organ"}, targets=None)
    assert not (tmp_path / "segmentation.raw.nii.gz").exists()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("label_id", [3, 8, 12])
def test_class_export_is_binary_and_preserves_stored_ids_geometry_and_units(
    tmp_path, version, label_id
):
    from xml.etree import ElementTree

    values = np.array([0, 3, 8] * 20, dtype=np.uint16).reshape((3, 4, 5))
    labels = [
        {"id": 8, "name": "liver"},
        {"id": 3, "name": "spleen"},
        {"id": 12, "name": "empty_class"},
    ]
    affine = np.array([[0, -1, 0, 3], [2, 0, 0, 7], [0, 0, 3, -5], [0, 0, 0, 1]], dtype=float)
    constructor = nib.Nifti1Image if version == 1 else nib.Nifti2Image
    original = constructor(values, affine)
    original.set_qform(affine, code=1)
    original.set_sform(affine, code=2)
    original.header.set_xyzt_units("micron", "sec")
    original.header.extensions.append(nib.nifti1.Nifti1Extension(0, b"obsolete labels"))
    path = tmp_path / "segmentation.nii.gz"
    nib.save(original, path)
    before = path.read_bytes()
    result = asyncio.run(core.export_label_mask(path, labels, label_id))
    exported = nib.load(result["path"])
    name = next(row["name"] for row in labels if row["id"] == label_id)
    assert result["name"] == f"{label_id}_{name}.nii.gz"
    assert result["label_id"] == label_id
    assert result["mask_value"] == 1
    assert result["voxels"] == np.count_nonzero(values == label_id)
    assert result["size_bytes"] == Path(result["path"]).stat().st_size
    assert result["sha256"] == core._file_digest(Path(result["path"]))
    assert result["source_sha256"] == core._file_digest(path)
    assert type(exported) is constructor
    assert exported.get_data_dtype() == np.dtype("uint8")
    assert exported.header.get_intent()[0] == "label"
    assert exported.header.get_zooms() == original.header.get_zooms()
    assert exported.header.get_xyzt_units() == ("micron", "sec")
    np.testing.assert_array_equal(exported.get_fdata(), values == label_id)
    np.testing.assert_array_equal(exported.affine, original.affine)
    assert nib.aff2axcodes(exported.affine) == nib.aff2axcodes(original.affine)
    for form in ("qform", "sform"):
        expected, expected_code = getattr(original, f"get_{form}")(coded=True)
        actual, actual_code = getattr(exported, f"get_{form}")(coded=True)
        np.testing.assert_array_equal(actual, expected)
        assert actual_code == expected_code
    assert len(exported.header.extensions) == 1
    xml = ElementTree.fromstring(exported.header.extensions[0].get_content())
    entries = xml.findall("./VolumeInformation/LabelTable/Label")
    assert [(entry.attrib["Key"], entry.text) for entry in entries] == [
        ("0", "Background"),
        ("1", name),
    ]
    assert [entries[1].attrib[key] for key in ("Red", "Green", "Blue", "Alpha")] == [
        "1.0",
        "0.0",
        "0.0",
        "1",
    ]
    assert path.read_bytes() == before
    assert Path(result["path"]).stat().st_mode & 0o777 == 0o600
    assert Path(result["path"]).parent.stat().st_mode & 0o777 == 0o700


def test_class_export_cache_skips_decoding_and_ignores_untrusted_receipt_fields(
    tmp_path, monkeypatch
):
    path = image_at(tmp_path / "segmentation.nii.gz", np.ones((3, 4, 5), dtype=np.uint8))
    labels = [{"id": 1, "name": "liver"}]
    first = asyncio.run(core.export_label_mask(path, labels, 1))
    receipt = Path(first["path"]).with_name(f".{first['name']}.json")
    cached = json.loads(receipt.read_text())
    cached.update({"path": "/should/not/be/returned", "untrusted": "extra"})
    receipt.write_text(json.dumps(cached))

    def unexpected(*args, **kwargs):
        pytest.fail("A verified cache hit must not decode the merged or exported mask")

    monkeypatch.setattr(core, "_inspect_nifti", unexpected)
    monkeypatch.setattr(core, "_uncompressed_nifti", unexpected)
    assert asyncio.run(core.export_label_mask(path, labels, 1)) == first


@pytest.mark.parametrize("damage", ["file", "receipt", "source"])
def test_class_export_rebuilds_damaged_or_stale_cache(tmp_path, damage):
    values = np.ones((3, 4, 5), dtype=np.uint8)
    path = image_at(tmp_path / "segmentation.nii.gz", values)
    labels = [{"id": 1, "name": "liver"}]
    first = asyncio.run(core.export_label_mask(path, labels, 1))
    if damage == "file":
        Path(first["path"]).write_bytes(b"broken gzip")
    elif damage == "receipt":
        Path(first["path"]).with_name(f".{first['name']}.json").write_text("{invalid")
    else:
        values[:, :, 0] = 0
        image_at(path, values)
    result = asyncio.run(core.export_label_mask(path, labels, 1))
    np.testing.assert_array_equal(nib.load(result["path"]).get_fdata(), values == 1)
    assert result["voxels"] == np.count_nonzero(values)
    assert result["source_sha256"] == core._file_digest(path)
    assert result["sha256"] == core._file_digest(Path(result["path"]))
    assert not list(Path(result["path"]).parent.glob(".binary-*"))


@pytest.mark.parametrize("invalid", [-1.0, 0.5, float("nan"), float("inf"), 99.0])
def test_class_export_invalid_merged_labels_fail_closed(tmp_path, invalid):
    path = image_at(tmp_path / "segmentation.nii.gz", np.full((3, 4, 5), invalid, dtype=np.float32))
    original = path.read_bytes()
    with pytest.raises(core.SegmentationError):
        asyncio.run(core.export_label_mask(path, [{"id": 1, "name": "liver"}], 1))
    assert path.read_bytes() == original
    assert not list((tmp_path / "class_masks").glob("*.nii.gz"))
    assert not list((tmp_path / "class_masks").glob("*.json"))


@pytest.mark.parametrize(
    "label",
    [
        None,
        {},
        {"id": True, "name": "liver"},
        {"id": 0, "name": "liver"},
        {"id": 1.0, "name": "liver"},
        {"id": 2**31, "name": "liver"},
        {"id": 1, "name": "../liver"},
        {"id": 1, "name": "a/b"},
        {"id": 1, "name": "a\\b"},
        {"id": 1, "name": ""},
        {"id": 1, "name": "a" * 121},
    ],
)
def test_class_export_filename_rejects_unsafe_metadata(label):
    with pytest.raises(core.SegmentationError):
        core.label_mask_filename(label)


@pytest.mark.parametrize(
    "labels,label_id",
    [
        ([], 1),
        ([{"id": 1, "name": "liver"}], 2),
        ([{"id": 1, "name": "liver"}], True),
        ([{"id": 1, "name": "liver"}, {"id": 1, "name": "spleen"}], 1),
    ],
)
def test_class_export_rejects_unknown_duplicate_or_missing_labels(tmp_path, labels, label_id):
    path = image_at(tmp_path / "segmentation.nii.gz")
    with pytest.raises(core.SegmentationError):
        asyncio.run(core.export_label_mask(path, labels, label_id))
    assert not (tmp_path / "class_masks").exists()


@pytest.mark.parametrize("location", ["source", "directory", "destination", "receipt", "lock"])
def test_class_export_rejects_symlinks(tmp_path, location):
    source = image_at(tmp_path / "segmentation.nii.gz")
    root = tmp_path / "class_masks"
    target = image_at(tmp_path / "untouched.nii.gz")
    original = target.read_bytes()
    if location == "source":
        source.unlink()
        source.symlink_to(target)
    elif location == "directory":
        outside = tmp_path / "outside"
        outside.mkdir()
        root.symlink_to(outside)
    else:
        root.mkdir()
        names = {
            "destination": "1_liver.nii.gz",
            "receipt": ".1_liver.nii.gz.json",
            "lock": ".1_liver.nii.gz.lock",
        }
        (root / names[location]).symlink_to(target)
    with pytest.raises(core.SegmentationError):
        asyncio.run(core.export_label_mask(source, [{"id": 1, "name": "liver"}], 1))
    assert target.read_bytes() == original


def test_class_export_cannot_overwrite_merged_source(tmp_path):
    source = image_at(tmp_path / "1_liver.nii.gz", np.ones((3, 4, 5), dtype=np.uint8))
    before = source.read_bytes()
    with pytest.raises(core.SegmentationError, match="replace its merged source"):
        asyncio.run(core.export_label_mask(source, [{"id": 1, "name": "liver"}], 1, tmp_path))
    assert source.read_bytes() == before


def test_class_export_validation_failure_publishes_no_partial_mask(tmp_path, monkeypatch):
    source = image_at(tmp_path / "segmentation.nii.gz", np.ones((3, 4, 5), dtype=np.uint8))
    before = source.read_bytes()
    inspect = core._inspect_nifti

    def failed(candidate, **kwargs):
        if candidate.name.startswith(".binary-"):
            raise core.SegmentationError("Synthetic final validation failure")
        return inspect(candidate, **kwargs)

    monkeypatch.setattr(core, "_inspect_nifti", failed)
    with pytest.raises(core.SegmentationError, match="final validation failure"):
        asyncio.run(core.export_label_mask(source, [{"id": 1, "name": "liver"}], 1))
    assert source.read_bytes() == before
    assert sorted(path.name for path in (tmp_path / "class_masks").iterdir()) == [
        ".1_liver.nii.gz.lock"
    ]


def test_class_export_cancellation_drains_final_validation_before_cleanup(tmp_path, monkeypatch):
    import threading

    source = image_at(tmp_path / "segmentation.nii.gz", np.ones((3, 4, 5), dtype=np.uint8))
    before = source.read_bytes()
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    inspect = core._inspect_nifti

    def observed(candidate, **kwargs):
        if candidate.name.startswith(".binary-"):
            entered.set()
            assert release.wait(3)
            stopped.set()
        return inspect(candidate, **kwargs)

    monkeypatch.setattr(core, "_inspect_nifti", observed)

    async def scenario():
        operation = asyncio.create_task(
            core.export_label_mask(source, [{"id": 1, "name": "liver"}], 1)
        )
        assert await asyncio.to_thread(entered.wait, 2)
        operation.cancel()
        await asyncio.sleep(0.01)
        assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert stopped.is_set()

    asyncio.run(scenario())
    assert source.read_bytes() == before
    assert sorted(path.name for path in (tmp_path / "class_masks").iterdir()) == [
        ".1_liver.nii.gz.lock"
    ]


@pytest.mark.parametrize("mode", ["cancel", "timeout"])
def test_class_export_waiting_lock_is_cancelable_and_reusable(tmp_path, monkeypatch, mode):
    import threading
    from contextlib import contextmanager

    source = image_at(tmp_path / "segmentation.nii.gz", np.ones((3, 4, 5), dtype=np.uint8))
    root = tmp_path / "class_masks"
    root.mkdir()
    entered = threading.Event()
    lock = core._label_export_lock

    @contextmanager
    def observed(*args, **kwargs):
        entered.set()
        with lock(*args, **kwargs):
            yield

    monkeypatch.setattr(core, "_label_export_lock", observed)

    async def scenario():
        with lock(root / ".1_liver.nii.gz.lock"):
            operation = asyncio.create_task(
                core.export_label_mask(
                    source,
                    [{"id": 1, "name": "liver"}],
                    1,
                    timeout_seconds=0.2 if mode == "timeout" else 5,
                )
            )
            assert await asyncio.to_thread(entered.wait, 1)
            if mode == "cancel":
                operation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await operation
            else:
                with pytest.raises(core.SegmentationError, match="timeout"):
                    await operation
            assert not (root / "1_liver.nii.gz").exists()
        return await core.export_label_mask(source, [{"id": 1, "name": "liver"}], 1)

    assert asyncio.run(scenario())["voxels"] == 60


def _class_export_process(source, root, ready, start, result):
    inspect = core._inspect_nifti

    def observed(candidate, **kwargs):
        if candidate == Path(source):
            with (Path(root) / "builds.txt").open("a") as output:
                output.write("built\n")
            time.sleep(0.15)
        return inspect(candidate, **kwargs)

    core._inspect_nifti = observed
    ready.put(True)
    if not start.wait(10):
        raise RuntimeError("Class export process did not start")
    result.put(asyncio.run(core.export_label_mask(source, [{"id": 1, "name": "liver"}], 1, root)))


def test_class_export_cross_process_same_class_builds_once(tmp_path):
    source = image_at(tmp_path / "segmentation.nii.gz", np.ones((3, 4, 5), dtype=np.uint8))
    root = tmp_path / "class_masks"
    root.mkdir()
    context = multiprocessing.get_context("spawn")
    ready, result, start = context.Queue(), context.Queue(), context.Event()
    processes = [
        context.Process(
            target=_class_export_process, args=(str(source), str(root), ready, start, result)
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            assert ready.get(timeout=10)
        start.set()
        first, second = result.get(timeout=10), result.get(timeout=10)
        assert first == second
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert (root / "builds.txt").read_text().splitlines() == ["built"]
        np.testing.assert_array_equal(nib.load(first["path"]).get_fdata(), 1)
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
        for queue in (ready, result):
            queue.close()
            queue.join_thread()


def test_class_export_decodes_source_twice_independent_of_slice_count(tmp_path, monkeypatch):
    from contextlib import contextmanager

    source = image_at(tmp_path / "segmentation.nii.gz", np.ones((2, 2, 80), dtype=np.uint8))
    original = core._uncompressed_nifti
    scans = []

    @contextmanager
    def observed(path, *args, **kwargs):
        scans.append(path)
        with original(path, *args, **kwargs) as stream:
            yield stream

    monkeypatch.setattr(core, "_uncompressed_nifti", observed)
    result = asyncio.run(core.export_label_mask(source, [{"id": 1, "name": "liver"}], 1))
    assert result["voxels"] == 320
    assert scans.count(source) == 2
    assert len(scans) == 3  # source validation + export + one binary output validation


@pytest.mark.parametrize("timeout", [True, None, 0, -1, float("nan"), float("inf")])
def test_class_export_timeout_rejects_invalid_values(tmp_path, timeout):
    with pytest.raises(core.SegmentationError, match="positive and finite"):
        asyncio.run(
            core.export_label_mask(tmp_path / "unused.nii.gz", [], 1, timeout_seconds=timeout)
        )

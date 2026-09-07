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
    assert json.loads((run / "result.json").read_text())["run_id"] == result["run_id"]
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


def test_task_policy_is_an_explicit_four_tool_allowlist():
    from dataclasses import FrozenInstanceError

    from medsegagent.task_specs import TASK_SPECS

    assert set(TASK_SPECS) == {"total", "total_mr", "lung_nodules", "liver_lesions"}
    assert TASK_SPECS["total"].default_speed == "fast"
    assert TASK_SPECS["total_mr"].modality == "MR"
    assert TASK_SPECS["lung_nodules"].tool == "segment_lung_nodules"
    assert TASK_SPECS["liver_lesions"].default_targets == ("liver_lesions",)
    assert core.task_labels("lung_nodules") == {1: "lung", 2: "lung_nodules"}
    with pytest.raises(FrozenInstanceError):
        TASK_SPECS["lung_nodules"].modality = "MR"
    with pytest.raises(core.SegmentationError):
        core.task_classes("brain_structures")


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
def test_specialized_default_speed_and_audited_subset_filter(tmp_path, monkeypatch, task):
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
        expected_label,
    }
    audit = json.loads((run / "filtering.json").read_text())
    assert audit["retained_label_ids"] == [expected_label]
    assert audit["geometry_preserved"] is True
    assert audit["raw_sha256"] == core._file_digest(run / "segmentation.raw.nii.gz")
    assert audit["segmentation_sha256"] == core._file_digest(run / "segmentation.nii.gz")
    assert not list(run.glob(".filtered-*"))
    assert (run / "filtering.json").stat().st_mode & 0o777 == 0o600


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
    assert result["labels"] == []
    assert result["nonzero_voxels"] == 0
    assert "does not rule out disease" in result["warning"]
    assert result["filtering"]["raw_nonzero_voxels"] == 60
    assert result["filtering"]["filtered_nonzero_voxels"] == 0


def test_native_specialized_mask_is_validated_before_filtering(tmp_path, monkeypatch):
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
def test_filter_preserves_voxel_locations_and_rotated_geometry(tmp_path, version):
    path = tmp_path / "segmentation.nii.gz"
    values = (np.arange(60).reshape((3, 4, 5)) % 3).astype(np.uint8)
    affine = np.array([[0, -1, 0, 3], [2, 0, 0, 7], [0, 0, 3, -5], [0, 0, 0, 1]], dtype=float)
    constructor = nib.Nifti1Image if version == 1 else nib.Nifti2Image
    nib.save(constructor(values, affine), path)
    original_sha = core._file_digest(path)
    result = core._filter_segmentation(
        path, labels=core.task_labels("lung_nodules"), targets=["lung_nodules"]
    )
    filtered = nib.load(path)
    np.testing.assert_array_equal(np.asanyarray(filtered.dataobj), np.where(values == 2, 2, 0))
    np.testing.assert_array_equal(filtered.affine, affine)
    assert result["audit"]["raw_sha256"] == original_sha
    assert result["geometry"]["shape"] == [3, 4, 5]


@pytest.mark.parametrize("mode", ["cancel", "timeout"])
def test_filter_cancellation_and_timeout_preserve_raw_without_partial_public_mask(
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
                assert _stop_event.wait(3), "Filtering cancellation did not signal its reader"
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
    assert not list(run.glob(".filtered-*"))


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
    assert Path(result["filtering"]["raw_segmentation_path"]).stat().st_mode & 0o777 == 0o600

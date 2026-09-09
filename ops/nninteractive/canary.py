"""One isolated, synthetic real-checkpoint engineering canary; never a clinical test.

Run the controller with the existing service Python, PYTHONDONTWRITEBYTECODE=1,
and PYTHONPATH pointing at the scratch copy of src. The child uses only the new
nnInteractive virtual environment. All artifacts stay in the exact scratch root.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.util
import io
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

SCRATCH = Path("/mnt/disks/bulk/huangziyan/MedSegAgent-nninteractive-canary-20260909")
PRODUCTION_DB = Path("/mnt/disks/bulk/huangziyan/MedSegAgent/runtime/state.sqlite3")
SHARED_LOCKS = Path("/home/huangziyan/.cache/medsegagent/scheduler")
GPU_UUID = "GPU-4853932d-66a0-200c-035f-fbda7598fd54"
CHECKPOINT_SHA256 = "b3ac4421f85457bbd1aa0d87f5e67bcb7bc8e2ce6b824b6ac45077cc5d630ea9"
RUN = SCRATCH / "canary/synthetic-canary-001"


def timestamp():
    return datetime.now(UTC).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def geometry(image):
    value = {
        "shape": list(image.shape),
        "affine": image.affine.tolist(),
        "spacing": [float(x) for x in image.header.get_zooms()[:3]],
        "units": list(image.header.get_xyzt_units()),
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        "qform": image.get_qform().tolist(),
        "sform": image.get_sform().tolist(),
    }
    value["sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def production_counts():
    # Aggregate statuses only: never select task data, identities, or upload paths.
    with sqlite3.connect(f"file:{PRODUCTION_DB}?mode=ro", uri=True, timeout=2) as db:
        db.execute("PRAGMA query_only=ON")
        return dict(db.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status"))


def has_active_jobs(counts):
    return any(
        count > 0
        for status, count in counts.items()
        if status not in {"completed", "failed", "canceled", "input_required"}
    )


def prepare_synthetic():
    import nibabel as nib
    import numpy as np

    grid = np.indices((128, 128, 96), dtype=np.float32)
    center = np.array([64, 64, 48], dtype=np.float32)[:, None, None, None]
    radius = np.sqrt(np.sum((grid - center) ** 2, axis=0))
    data = np.full((128, 128, 96), -1000, dtype=np.float32)
    data[radius <= 45] = 0
    data[radius <= 25] = 65
    angle = np.deg2rad(13)
    affine = np.array(
        [
            [-1.2 * np.cos(angle), -1.2 * np.sin(angle), 0, 100],
            [-1.2 * np.sin(angle), 1.2 * np.cos(angle), 0, -80],
            [0, 0, 2.0, -40],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    source = nib.Nifti1Image(data, affine)
    source.header.set_xyzt_units("mm")
    source.set_qform(affine, 1)
    source.set_sform(affine, 2)
    image_path = RUN / "synthetic-sphere.nii.gz"
    nib.save(source, image_path)
    source = nib.load(image_path)
    prompts = [
        {"kind": "point", "voxel": [64, 64, 48], "positive": True},
        {"kind": "point", "voxel": [94, 64, 48], "positive": False},
        {"kind": "box", "bounds": [[39, 90], [39, 90], [48, 49]], "positive": True},
    ]
    provenance = {
        "kind": "synthetic_sphere",
        "patient_data": False,
        "public_ct_used": False,
        "reason": "No public CT with verified provenance was provisioned for this canary.",
        "generator": "Deterministic sphere r=25, body r=45; intensities 65/0/-1000.",
        "seed": None,
        "license": "Generated here; no external input data.",
        "image_sha256": sha256(image_path),
        "geometry": geometry(source),
        "prompts": prompts,
        "point_world_mm": [
            (source.affine @ np.array([*p["voxel"], 1]))[:3].tolist() for p in prompts[:2]
        ],
        "limitations": "Engineering/API/geometry/resource check, not medical segmentation quality.",
    }
    write_json(RUN / "source-manifest.json", provenance)
    return image_path, prompts, provenance


def run_instrumented_worker(request_path, response_path):
    """Call the unmodified adapter, recording stage boundaries and CUDA peaks."""
    import nibabel as nib
    import numpy as np
    import torch

    worker_path = SCRATCH / "src/medsegagent/nninteractive_worker.py"
    spec = importlib.util.spec_from_file_location("canary_nninteractive_worker", worker_path)
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)
    request = json.loads(request_path.read_text())
    image = nib.load(request["image_path"])
    original_loader = worker._load_session
    stages = []
    target = None

    def instrumented_loader(device):
        session, version = original_loader(device)
        torch.cuda.reset_peak_memory_stats()

        class SessionProxy:
            def __getattr__(self, name):
                return getattr(session, name)

            def invoke(self, name, *args, **kwargs):
                nonlocal target
                torch.cuda.synchronize()
                started = time.monotonic()
                value = getattr(session, name)(*args, **kwargs)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                if name == "set_target_buffer":
                    target = args[0]
                stage = {
                    "method": name,
                    "seconds": elapsed,
                    "cuda_max_allocated_bytes_cumulative": torch.cuda.max_memory_allocated(),
                    "cuda_max_reserved_bytes_cumulative": torch.cuda.max_memory_reserved(),
                }
                if name in {"add_point_interaction", "add_bbox_interaction"}:
                    index = sum("mask_path" in x for x in stages) + 1
                    mask_path = RUN / f"stage-{index}.nii.gz"
                    worker._save_mask(target.copy(), image, mask_path)
                    stage.update(
                        mask_path=str(mask_path),
                        mask_sha256=sha256(mask_path),
                        voxel_count=int(np.count_nonzero(target)),
                    )
                stages.append(stage)
                write_json(RUN / "worker-metrics.json", {"stages": stages})
                return value

            def initialize_from_trained_model_folder(self, *args, **kwargs):
                return self.invoke("initialize_from_trained_model_folder", *args, **kwargs)

            def set_image(self, *args, **kwargs):
                return self.invoke("set_image", *args, **kwargs)

            def set_target_buffer(self, *args, **kwargs):
                return self.invoke("set_target_buffer", *args, **kwargs)

            def add_point_interaction(self, *args, **kwargs):
                return self.invoke("add_point_interaction", *args, **kwargs)

            def add_bbox_interaction(self, *args, **kwargs):
                return self.invoke("add_bbox_interaction", *args, **kwargs)

        return SessionProxy(), version

    worker._load_session = instrumented_loader
    started = time.monotonic()
    exit_code = worker.main(["--request", str(request_path), "--response", str(response_path)])
    write_json(
        RUN / "worker-metrics.json",
        {
            "adapter_seconds": time.monotonic() - started,
            "stages": stages,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scope": "Instrumented real adapter; stage snapshots add small CPU/file overhead.",
            "timing_notes": [
                "Model initialization includes the upstream warmup network forward.",
                (
                    "set_image submits asynchronous CPU preprocessing; unfinished work is charged "
                    "to the first interaction wait. First-interaction time is not pure GPU inference."
                ),
                (
                    "Three interactions are three public prediction calls, not three total network "
                    "forwards; upstream warmup and autozoom may perform additional forwards."
                ),
            ],
        },
    )
    return exit_code


async def gpu_process_memory(query):
    raw = await query(
        ["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]
    )
    result = []
    for row in csv.reader(io.StringIO(raw)):
        if row and row[0].strip() == GPU_UUID:
            result.append({"pid": int(row[1].strip()), "memory_mib": int(row[2].strip())})
    return result


async def run_controller():
    from medsegagent import core
    from medsegagent.gpu_scheduler import (
        GPUScheduler,
        SchedulerConfig,
        SchedulerTimeout,
        _query_nvidia_smi,
        query_gpu_status,
    )

    if not Path(core.__file__).resolve().is_relative_to(SCRATCH / "src"):
        raise RuntimeError("Controller must import only the scratch source copy.")
    if Path(__file__).resolve() != SCRATCH / "ops/nninteractive/canary.py":
        raise RuntimeError("Run only the isolated scratch copy of this script.")
    os.umask(0o077)
    RUN.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "preparing",
        "started_at": timestamp(),
        "gpu_uuid": GPU_UUID,
        "timeout_seconds": 600,
        "clinical_validation": False,
        "single_job": True,
        "controller_source": str(Path(core.__file__).resolve()),
        "controller_source_sha256": sha256(core.__file__),
        "canary_script_sha256": sha256(__file__),
        "adapter_sha256": sha256(SCRATCH / "src/medsegagent/nninteractive_worker.py"),
        "shared_lock_directory": str(SHARED_LOCKS),
        "scheduler_max_concurrent": 3,
        "admission_min_free_mib": 16384,
        "admission_max_utilization_percent": 5,
    }
    samples = []
    monitor_errors = []
    process_pid = None
    finished = asyncio.Event()

    async def snapshot():
        states = await asyncio.wait_for(query_gpu_status(), 6)
        gpu = next(x for x in states if x.uuid == GPU_UUID)
        return {"timestamp": timestamp(), **asdict(gpu)}

    async def preflight():
        # Last read immediately before launch, repeated after obtaining the lease.
        counts = production_counts()
        gpu = await snapshot()
        ready = (
            not has_active_jobs(counts)
            and gpu["free_memory_mib"] >= 16384
            and gpu["utilization_percent"] <= 5
            and not gpu["compute_pids"]
        )
        return {"production_status_counts": counts, "gpu": gpu, "ready": ready}

    async def monitor():
        while not finished.is_set():
            try:
                item = await snapshot()
                item["compute_memory"] = await asyncio.wait_for(
                    gpu_process_memory(_query_nvidia_smi), 5
                )
                samples.append(item)
                with (RUN / "telemetry.jsonl").open("a") as stream:
                    stream.write(json.dumps(item) + "\n")
            except Exception as exc:  # noqa: BLE001 -- telemetry failure is recorded
                monitor_errors.append(f"{type(exc).__name__}: {exc}")
            try:
                await asyncio.wait_for(finished.wait(), 0.5)
            except TimeoutError:
                pass

    def child_started(pid):
        nonlocal process_pid
        process_pid = pid
        report["worker_pid"] = pid
        report["inference_started_at"] = timestamp()
        write_json(RUN / "report.json", report)

    try:
        report["preparation"] = json.loads((SCRATCH / "preparation.json").read_text())
        if not (
            report["preparation"].get("environment_sync_complete")
            and report["preparation"].get("model_integrity_verified")
        ):
            report.update(status="skipped", reason="isolated_dependencies_not_ready")
            return report
        model = SCRATCH / "models/nnInteractive_v1.0"
        checkpoint = model / "fold_0/checkpoint_final.pth"
        report["model_sha256"] = sha256(checkpoint)
        if report["model_sha256"] != CHECKPOINT_SHA256:
            raise RuntimeError("Official checkpoint SHA256 mismatch.")
        for name in ("verified-manifest.json", "source-manifest.json"):
            report[f"model_{name}"] = json.loads((SCRATCH / "models" / name).read_text())
        report["worker_python"] = str(SCRATCH / "ops/nninteractive/.venv/bin/python")
        image_path, prompts, provenance = prepare_synthetic()
        report["source"] = provenance
        request_path = RUN / "request.json"
        response_path = RUN / "response.json"
        output_path = RUN / "final-mask.nii.gz"
        write_json(
            request_path,
            {
                "image_path": str(image_path),
                "output_path": str(output_path),
                "model_path": str(model),
                "device": "cuda:0",
                "prompts": prompts,
            },
        )
        report["preflight"] = await preflight()
        if not report["preflight"]["ready"]:
            report.update(status="skipped", reason="production_or_gpu_busy")
            return report
        config = SchedulerConfig(
            lock_dir=SHARED_LOCKS,
            allowed_gpus=(GPU_UUID,),
            min_free_memory_mib=16384,
            max_utilization_percent=5,
            max_concurrent=3,
        )
        scheduler = GPUScheduler(config, device="gpu")
        # Do not queue a canary behind production work; bound lease admission to 3s.
        async with scheduler.acquire(deadline=time.monotonic() + 3) as lease:
            report["lease_gpu_uuid"] = lease.gpu_uuid
            report["final_preflight"] = await preflight()
            if not report["final_preflight"]["ready"]:
                report.update(status="skipped", reason="production_or_gpu_became_busy")
                return report
            report["status"] = "running"
            # The fixed exclusive run directory makes accidental reruns fail closed.
            write_json(RUN / "attempt.json", {"started_at": timestamp(), "attempt": 1})
            monitor_task = asyncio.create_task(monitor())
            started = time.monotonic()
            try:
                await core._run_command(
                    [
                        report["worker_python"],
                        str(Path(__file__)),
                        "--worker-request",
                        str(request_path),
                        "--response",
                        str(response_path),
                    ],
                    timeout_seconds=600,
                    output_dir=RUN,
                    lock_fd=lease,
                    on_start=child_started,
                )
            finally:
                report["process_seconds"] = time.monotonic() - started
                finished.set()
                await monitor_task
            report["response"] = json.loads(response_path.read_text())
            if not report["response"].get("ok"):
                raise RuntimeError("Worker did not report a successful inference.")
            import nibabel as nib
            import numpy as np

            source = nib.load(image_path)
            masks = []
            previous_data = None
            for path in [*(RUN / f"stage-{i}.nii.gz" for i in range(1, 4)), output_path]:
                mask = nib.load(path)
                data = np.asarray(mask.dataobj)
                matches = geometry(mask)["sha256"] == geometry(source)["sha256"]
                binary = bool(np.isin(data, [0, 1]).all())
                if not matches or not binary:
                    raise RuntimeError("Output mask failed binary/exact-geometry validation.")
                masks.append(
                    {
                        "path": str(path),
                        "sha256": sha256(path),
                        "geometry_sha256": geometry(mask)["sha256"],
                        "binary": binary,
                        "voxel_count": int(np.count_nonzero(data)),
                        "mask_data_sha256": hashlib.sha256(
                            np.ascontiguousarray(data, dtype=np.uint8).tobytes()
                        ).hexdigest(),
                        "positive_point_value": int(data[tuple(prompts[0]["voxel"])]),
                        "negative_point_value": int(data[tuple(prompts[1]["voxel"])]),
                        "changed_voxels_from_previous_stage": (
                            int(np.count_nonzero(data != previous_data))
                            if previous_data is not None
                            else None
                        ),
                    }
                )
                previous_data = data.copy()
            if masks[-1]["mask_data_sha256"] != masks[-2]["mask_data_sha256"]:
                raise RuntimeError("Final mask differs from the third interaction snapshot.")
            if sha256(image_path) != provenance["image_sha256"]:
                raise RuntimeError("Synthetic source changed during the job.")
            if report["response"]["predictions_run"] != 3:
                raise RuntimeError("Expected exactly three interaction prediction calls.")
            report["masks"] = masks
            report["worker_metrics"] = json.loads((RUN / "worker-metrics.json").read_text())
            report["status"] = "completed"
    except SchedulerTimeout:
        report.update(status="skipped", reason="shared_scheduler_busy")
    except Exception as exc:  # noqa: BLE001 -- preserve a failed canary receipt
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if (RUN / "response.json").exists():
            report["response"] = json.loads((RUN / "response.json").read_text())
    finally:
        report["finished_at"] = timestamp()
        metrics_path = RUN / "worker-metrics.json"
        if metrics_path.exists():
            try:
                report["worker_metrics"] = json.loads(metrics_path.read_text())
            except (OSError, ValueError) as exc:
                report["worker_metrics_error"] = f"{type(exc).__name__}: {exc}"
        report["telemetry_samples"] = len(samples)
        report["telemetry_errors"] = monitor_errors
        report["sampled_gpu_peak_used_mib"] = max(
            (x["total_memory_mib"] - x["free_memory_mib"] for x in samples), default=None
        )
        report["sampled_worker_peak_used_mib"] = max(
            (
                p["memory_mib"]
                for x in samples
                for p in x.get("compute_memory", [])
                if p["pid"] == process_pid
            ),
            default=None,
        )
        report["sampled_peak_scope"] = "0.5s polling plus command latency; sampled lower bound."
        try:
            report["postflight"] = await preflight()
            report["worker_pid_absent_from_gpu_after_job"] = (
                process_pid not in report["postflight"]["gpu"]["compute_pids"]
            )
        except Exception as exc:  # noqa: BLE001 -- retain main result if postflight fails
            report["postflight_error"] = f"{type(exc).__name__}: {exc}"
        write_json(RUN / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = str(SCRATCH / "src")
    for key, relative in (
        ("XDG_CACHE_HOME", "cache"),
        ("TORCH_HOME", "cache/torch"),
        ("MPLCONFIGDIR", "cache/matplotlib"),
        ("TMPDIR", "tmp"),
    ):
        location = SCRATCH / relative
        location.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(location)
    if args.worker_request:
        if not args.response:
            parser.error("--response is required with --worker-request")
        return run_instrumented_worker(args.worker_request, args.response)
    report = asyncio.run(run_controller())
    print(json.dumps({"status": report["status"], "report": str(RUN / "report.json")}))
    return 0 if report["status"] in {"completed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

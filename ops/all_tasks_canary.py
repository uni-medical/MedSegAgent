"""Run real producer inference on explicitly mapped public research images.

The matrix is {task: {input, modality, quality?, notes?}} (or a document with
that mapping under ``tasks``). Relative input paths resolve beside the matrix.
This is an engineering acceptance runner, not a clinical accuracy evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from medsegagent import catalog, core
from medsegagent.execution import TaskExecution
from medsegagent.task_specs import TASK_SPECS, supports_native_roi, task_config


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prediction_contract(task: str, quality: str, targets: list[str] | None) -> dict:
    """Count nnUNet_predict_image completions, not weight files or image tiles.

    In TotalSegmentator 2.18, one multi-part anatomy call prints one completion
    after all primary models. A crop/refinement call adds its own completion.
    Teeth recursively runs craniofacial structures and that producer's crop.
    """
    config = task_config(task, quality)
    primary = config["task_id"]
    stages = []
    crop = config.get("crop")
    if crop or config.get("cascade") or supports_native_roi(task, targets):
        crop_task = config.get("crop_model")
        if crop_task:
            stages.extend(prediction_contract(crop_task, "standard", None)["stages"])
        else:
            stages.append({"stage": "crop", "model_count": 1})
    if task == "vertebrae_pp_refined":
        stages.append({"stage": "vertebrae_body_refinement", "model_count": 1})
    stages.append({"stage": task, "model_count": len(primary) if isinstance(primary, list) else 1})
    return {
        "stages": stages,
        "expected_prediction_completions": len(stages),
        "expected_model_starts": sum(stage["model_count"] for stage in stages),
    }


def inspect_log(path: Path, contract: dict) -> dict:
    content = path.read_text(errors="replace")
    completions = len(re.findall(r"\bPredicted in \d+(?:\.\d+)?s", content))
    starts = len(re.findall(r"(?m)^\s*Predicting(?:\.\.\.| part \d+ of \d+ \.\.\.)", content))
    empty_crop = "Crop is empty. Returning empty segmentation." in content
    reference_shortcut = "Using reference seg instead of prediction" in content
    verified = (
        not empty_crop
        and not reference_shortcut
        and completions == contract["expected_prediction_completions"]
        and starts == contract["expected_model_starts"]
    )
    return {
        "path": str(path),
        "sha256": digest(path),
        "prediction_completions": completions,
        "model_starts": starts,
        "empty_crop_shortcut": empty_crop,
        "reference_shortcut": reference_shortcut,
        "real_inference_verified": verified,
        **contract,
    }


def inspect_artifact(path: Path, source: Path, labels: list[dict]) -> dict:
    image, mask = nib.load(source), nib.load(path)
    values = np.asanyarray(mask.dataobj)
    unique = np.unique(values)
    geometry = (
        image.shape == mask.shape
        and np.allclose(image.affine, mask.affine, rtol=1e-5, atol=1e-4)
        and np.allclose(image.header.get_zooms(), mask.header.get_zooms(), rtol=1e-5, atol=1e-6)
    )
    label_valid = bool(
        np.isfinite(unique).all()
        and np.equal(unique, np.floor(unique)).all()
        and set(unique.tolist()) <= {0, *(label["id"] for label in labels)}
    )
    return {
        "path": str(path),
        "sha256": digest(path),
        "geometry_valid": bool(geometry),
        "labels_valid": label_valid,
        "nonzero_voxels": int(np.count_nonzero(values)),
        "present_label_ids": [int(value) for value in unique if value > 0],
    }


def reusable(report: dict, fingerprint: str) -> bool:
    if (
        report.get("status") != "success"
        or report.get("fingerprint") != fingerprint
        or report.get("real_inference_verified") is not True
        or report.get("geometry_and_labels_verified") is not True
    ):
        return False
    records = report.get("log_checks", []) + report.get("artifact_checks", [])
    if not report.get("log_checks") or not report.get("artifact_checks"):
        return False
    try:
        return all(digest(Path(record["path"])) == record["sha256"] for record in records)
    except (OSError, KeyError, TypeError):
        return False


async def run_one(task: str, row: dict, args, semaphore: asyncio.Semaphore) -> dict:
    output = args.output / task
    result_path = output / "result.json"
    report = {
        "schema_version": 1,
        "task": task,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "input_spec": row,
        "limitation": "Public reference engineering canary; no clinical accuracy claim.",
    }
    start = time.monotonic()
    try:
        spec = TASK_SPECS[task]
        if not spec.public_service_supported:
            raise ValueError("Task excluded by public-service policy.")
        if row.get("modality") != spec.modality:
            raise ValueError("Matrix modality must match the task's declared modality.")
        if not row.get("input"):
            report.update(status="input_unavailable", reason=row.get("notes"))
            return report
        source = Path(row["input"]).expanduser()
        if not source.is_absolute():
            source = args.input_matrix.parent / source
        source = source.resolve(strict=True)
        targets = list(catalog.native_labels(task).values())
        quality = row.get("quality", "standard")
        core.validate_task_options(task, quality, targets)
        input_sha = digest(source)
        if row.get("sha256") and row["sha256"] != input_sha:
            raise ValueError("Input digest does not match its public reference manifest.")
        version = importlib.metadata.version("TotalSegmentator")
        if version != "2.18.0":
            raise ValueError("Log acceptance contract is pinned to TotalSegmentator 2.18.0.")
        fingerprint = hashlib.sha256(
            json.dumps(
                [task, quality, targets, input_sha, version, digest(Path(__file__))],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        report.update(
            input=str(source),
            input_sha256=input_sha,
            quality=quality,
            targets=targets,
            totalsegmentator_version=version,
            fingerprint=fingerprint,
        )
        if result_path.is_file():
            try:
                previous = json.loads(result_path.read_text())
                if reusable(previous, fingerprint):
                    print(
                        json.dumps({"task": task, "status": "reused_verified_success"}), flush=True
                    )
                    return previous
            except (ValueError, OSError):
                pass
        write_json(result_path, report)
        async with semaphore:
            print(json.dumps({"task": task, "status": "running", "quality": quality}), flush=True)
            execution = TaskExecution(
                input_path=source,
                modality=spec.modality,
                output_dir=output / "runs",
                modality_source="example_manifest",
            )
            response = await execution.call(
                "segment",
                {"task": task, "modality": spec.modality, "quality": quality, "targets": targets},
            )
            result = execution.export_result()
            write_json(output / "execution-result.json", result)
            report["response"] = response
            report["execution_result"] = str(output / "execution-result.json")
            contract = prediction_contract(task, quality, targets)
            logs = sorted(execution._root.glob("runs/*/process.log"))
            report["log_checks"] = [inspect_log(path, contract) for path in logs]
            report["artifact_checks"] = [
                inspect_artifact(Path(artifact["path"]), source, artifact["labels"])
                for artifact in result["artifacts"]
            ]
            report["real_inference_verified"] = bool(logs) and all(
                check["real_inference_verified"] for check in report["log_checks"]
            )
            report["geometry_and_labels_verified"] = bool(result["artifacts"]) and all(
                check["geometry_valid"] and check["labels_valid"]
                for check in report["artifact_checks"]
            )
            report["status"] = (
                "success"
                if execution.is_complete
                and report["real_inference_verified"]
                and report["geometry_and_labels_verified"]
                else "failed"
            )
    except asyncio.CancelledError:
        report["status"] = "cancelled"
        raise
    except Exception as exc:  # noqa: BLE001 - Preserve other tasks after a producer failure.
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        report["seconds"] = time.monotonic() - start
        report["finished_at"] = datetime.now(UTC).isoformat()
        # A resumed success is returned without replacing its original evidence.
        if report.get("status") != "running":
            write_json(result_path, report)
    print(
        json.dumps({"task": task, "status": report["status"], "seconds": report["seconds"]}),
        flush=True,
    )
    return report


async def run(args) -> int:
    matrix = json.loads(args.input_matrix.read_text())
    matrix = matrix.get("tasks", matrix)
    tasks = list(dict.fromkeys(args.task or matrix))
    if any(task not in TASK_SPECS or task not in matrix for task in tasks):
        raise ValueError("Every selected task must exist in both catalog and matrix.")
    original = core._build_command

    def verbose_command(**kwargs):
        command = [part for part in original(**kwargs) if part != "--quiet"]
        return command + ["--verbose"]

    core._build_command = verbose_command
    try:
        semaphore = asyncio.Semaphore(args.workers)
        results = await asyncio.gather(
            *(run_one(task, matrix[task], args, semaphore) for task in tasks)
        )
    finally:
        core._build_command = original
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "input_matrix": str(args.input_matrix),
        "workers": args.workers,
        "results": {
            row["task"]: {
                key: row.get(key)
                for key in (
                    "status",
                    "real_inference_verified",
                    "geometry_and_labels_verified",
                    "seconds",
                )
            }
            for row in results
        },
    }
    write_json(args.output / "summary.json", summary)
    return 0 if all(row["status"] == "success" for row in results) else 1


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-matrix", required=True, type=Path)
    parser.add_argument("--task", action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.input_matrix = args.input_matrix.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()

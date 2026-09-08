"""Local weight readiness; model downloads are explicit preparation, never inference."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from medsegagent.task_specs import TASK_SPECS, model_record, required_model_ids


class WeightError(ValueError):
    code = "WEIGHTS_MISSING"


def weights_root(root=None) -> Path:
    if root is not None:
        return Path(root).expanduser()
    # The upstream config module imports torch. Mirror its documented path precedence
    # here so capability discovery never loads an inference runtime or license config.
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        return Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
    fallback = Path("/tmp") if str(Path.home()) == "/" else Path.home()
    return (
        Path(os.environ.get("TOTALSEG_HOME_DIR", str(fallback / ".totalsegmentator")))
        / "nnunet/results"
    )


def _json_file(path: Path) -> bool:
    try:
        return 0 < path.stat().st_size <= 16 * 1024 * 1024 and isinstance(
            json.loads(path.read_text()), dict
        )
    except (OSError, ValueError, UnicodeError):
        return False


def inspect_model(model_id: int, *, root=None) -> dict:
    record = model_record(model_id)
    base = weights_root(root) / (record.get("rel_path") or "") / record["foldername"]
    missing, files = [], []
    for config in record["expected_configs"]:
        directory = base / f"{config['trainer']}__{config['plans']}__{config['model']}"
        for name in ("dataset.json", "plans.json"):
            path = directory / name
            if not _json_file(path):
                missing.append(str(path.relative_to(base)))
            else:
                files.append(path)
        folds = config["folds"]
        if folds is None:
            # Match nnUNet's numeric-fold auto discovery, not an invented fold count.
            folds = sorted(
                int(p.name[5:])
                for p in directory.glob("fold_*")
                if p.is_dir() and p.name[5:].isdigit()
            )
        if not folds:
            missing.append(str(directory.relative_to(base) / "fold_*"))
        for fold in folds:
            path = directory / f"fold_{fold}" / "checkpoint_final.pth"
            if not path.is_file() or path.stat().st_size <= 4096:
                missing.append(str(path.relative_to(base)))
            else:
                files.append(path)
    if not record["expected_configs"]:
        missing.append("runtime_configuration")
    return {
        "model_id": model_id,
        "ready": not missing,
        "missing_files": sorted(set(missing)),
        "files": sorted({str(path.relative_to(base)) for path in files}),
        "folder": str(base.relative_to(weights_root(root))),
    }


def inspect_weights(task: str, quality: str = "standard", targets=None, *, root=None) -> dict:
    model_ids = required_model_ids(task, quality, targets)
    models = [inspect_model(model_id, root=root) for model_id in model_ids]
    return {
        "task": task,
        "quality": quality,
        "ready": all(row["ready"] for row in models),
        "model_ids": list(model_ids),
        "missing_model_ids": [row["model_id"] for row in models if not row["ready"]],
    }


def require_weights(task: str, quality: str = "standard", targets=None) -> dict:
    result = inspect_weights(task, quality, targets)
    if not result["ready"]:
        raise WeightError(
            "Required model weights are missing or incomplete; run weight preparation."
        )
    return result


def inventory(*, root=None) -> dict:
    tasks = []
    for task, spec in TASK_SPECS.items():
        row = {
            "task": task,
            "availability": spec.availability,
            "reason": spec.availability_reason,
            "usage_license": spec.usage_license,
        }
        if spec.availability == "available":
            row["modes"] = [inspect_weights(task, speed, root=root) for speed in spec.speeds]
            row["ready"] = all(mode["ready"] for mode in row["modes"])
        else:
            row["ready"] = False
        tasks.append(row)
    return {
        "tasks": tasks,
        "available_count": sum(t["availability"] == "available" for t in tasks),
        "ready_count": sum(t["ready"] for t in tasks),
    }


def verify_model(model_id: int, *, root=None, expected=None) -> dict:
    """Hash runtime files once during preparation; never silently bless changed pins."""
    status = inspect_model(model_id, root=root)
    if not status["ready"]:
        raise WeightError(f"Model {model_id} is incomplete.")
    if expected is not None and set(status["files"]) != set(expected):
        raise WeightError(f"Model {model_id} differs from the pinned runtime file set.")
    base = weights_root(root) / status["folder"]
    files = []
    for name in status["files"]:
        path = base / name
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        row = {"path": name, "bytes": path.stat().st_size, "sha256": checksum}
        if (
            expected
            and name in expected
            and any(row[key] != expected[name][key] for key in ("bytes", "sha256"))
        ):
            raise WeightError(f"Model {model_id} differs from the pinned weight manifest.")
        files.append(row)
    record = model_record(model_id)
    return {
        "model_id": model_id,
        "folder": status["folder"],
        "source_url": record["download_url"],
        "files": files,
    }

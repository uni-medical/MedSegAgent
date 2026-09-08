"""Prepare every public noncommercial TotalSegmentator task, then pin runtime bytes.

Download preparation is separate from inference. Each download uses its own temporary
TotalSegmentator home, so upstream's shared ZIP name cannot corrupt parallel downloads.
Existing valid models are reused; existing incomplete directories are retained as backups.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from medsegagent.task_specs import TASK_SPECS, model_ids, model_record
from medsegagent.weights import WeightError, inspect_model, verify_model, weights_root


def download_model(model_id, root, *, expected=None):
    record = model_record(model_id)
    if record["license_required"] or not record["download_url"]:
        raise ValueError("This model is outside public weight preparation.")
    lockdir = root.parent / ".medseg-download-locks"
    lockdir.mkdir(exist_ok=True, mode=0o700)
    with (lockdir / f"{model_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if inspect_model(model_id, root=root)["ready"]:
            return
        with tempfile.TemporaryDirectory(
            prefix=f".medseg-download-{model_id}-", dir=root.parent
        ) as stage:
            env = dict(os.environ, TOTALSEG_HOME_DIR=stage)
            env.pop("TOTALSEG_WEIGHTS_PATH", None)
            Path(stage, "config.json").write_text(
                json.dumps(
                    {
                        "send_usage_stats": False,
                        "statistics_disclaimer_shown": True,
                    }
                )
            )
            subprocess.run(
                [
                    os.sys.executable,
                    "-c",
                    (
                        "from totalsegmentator.libs import download_pretrained_weights; "
                        f"download_pretrained_weights({model_id})"
                    ),
                ],
                env=env,
                check=True,
                timeout=2400,
            )
            prepared = Path(stage) / "nnunet/results"
            # Validate pins before promotion; a rejected download must never become usable.
            verify_model(model_id, root=prepared, expected=expected)
            relative = Path(record.get("rel_path") or "") / record["foldername"]
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = destination.with_name(destination.name + ".previous-" + uuid.uuid4().hex)
            if destination.exists():
                destination.rename(backup)
            try:
                (prepared / relative).rename(destination)
            except BaseException:
                if backup.exists():
                    backup.rename(destination)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--task", action="append", choices=list(TASK_SPECS))
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("open-model-manifest.json")
    )
    args = parser.parse_args()
    os.umask(0o077)
    root = weights_root(args.root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = args.task or [t for t, s in TASK_SPECS.items() if s.availability == "available"]
    if any(TASK_SPECS[t].availability != "available" for t in tasks):
        parser.error("Only public noncommercial tasks with released weights can be prepared.")
    ids = sorted(
        {
            i
            for t in tasks
            for speed in TASK_SPECS[t].speeds
            for i in model_ids(t, speed, roi=TASK_SPECS[t].supports_roi)
        }
    )
    pinned = {}
    for path in (Path(__file__).with_name("model-manifest.json"), args.manifest):
        if path.exists():
            for row in json.loads(path.read_text())["models"]:
                pinned[row.get("model_id", row.get("task_id"))] = {
                    f["path"]: f for f in row["files"]
                }

    def prepare(model_id):
        if args.download:
            download_model(model_id, root, expected=pinned.get(model_id))
        return verify_model(model_id, root=root, expected=pinned.get(model_id))

    models, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare, i): i for i in ids}
        for future in concurrent.futures.as_completed(futures):
            model_id = futures[future]
            try:
                models.append(future.result())
                print(json.dumps({"model_id": model_id, "status": "verified"}), flush=True)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors.append({"model_id": model_id, "error": str(exc)})
                print(
                    json.dumps(
                        {"model_id": model_id, "status": "failed", "error_type": type(exc).__name__}
                    ),
                    flush=True,
                )
    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        raise SystemExit(1)
    manifest = {
        "schema": 2,
        "totalsegmentator_version": "2.18.0",
        "usage_policy": "public_noncommercial",
        "verified_at": time.time(),
        "tasks": {
            t: {
                "license": TASK_SPECS[t].usage_license,
                "modes": {
                    q: list(model_ids(t, q, roi=TASK_SPECS[t].supports_roi))
                    for q in TASK_SPECS[t].speeds
                },
            }
            for t in tasks
        },
        "models": sorted(models, key=lambda row: row["model_id"]),
    }
    publish_manifest(args.manifest, manifest)
    print(json.dumps({"tasks": len(tasks), "models": len(models), "status": "verified"}))


def publish_manifest(path, manifest):
    """Merge subset preparation under a lock without dropping existing pinned models."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads(path.read_text()) if path.exists() else {}
        models = {r.get("model_id", r.get("task_id")): r for r in previous.get("models", [])}
        for row in manifest["models"]:
            model_id = row["model_id"]
            if model_id in models:
                before = {f["path"]: (f["bytes"], f["sha256"]) for f in models[model_id]["files"]}
                after = {f["path"]: (f["bytes"], f["sha256"]) for f in row["files"]}
                if before != after:
                    raise WeightError(f"Model {model_id} changed while preparing the manifest.")
            models[model_id] = row
        result = {
            **previous,
            **manifest,
            "tasks": {**previous.get("tasks", {}), **manifest["tasks"]},
            "models": [models[i] for i in sorted(models)],
        }
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

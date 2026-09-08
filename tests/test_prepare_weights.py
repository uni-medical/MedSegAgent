"""Preparation never publishes unverified downloads or loses existing model pins."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_weights import prepare

from medsegagent.weights import WeightError, verify_model

spec = importlib.util.spec_from_file_location(
    "prepare_weights", Path(__file__).parents[1] / "ops/prepare_weights.py"
)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def test_rejected_download_never_replaces_existing_directory(tmp_path, monkeypatch):
    root = tmp_path / "live"
    root.mkdir()
    reference = tmp_path / "reference"
    prepare(reference, 297)
    expected = {f["path"]: f for f in verify_model(297, root=reference)["files"]}
    # Resolve the installed folder name instead of relying on an upstream nickname.
    destination = root / preparation.model_record(297)["foldername"]
    destination.mkdir()
    (destination / "preserve.txt").write_text("old incomplete model")

    def changed_download(command, *, env, **kwargs):
        base = prepare(Path(env["TOTALSEG_HOME_DIR"]) / "nnunet/results", 297)
        checkpoint = next(base.rglob("checkpoint_final.pth"))
        checkpoint.write_bytes(b"changed" * 2000)

    monkeypatch.setattr(preparation.subprocess, "run", changed_download)
    with pytest.raises(WeightError, match="pinned"):
        preparation.download_model(297, root, expected=expected)
    assert (destination / "preserve.txt").read_text() == "old incomplete model"
    assert not list(destination.rglob("checkpoint_final.pth"))


def manifest(model_id, *, checksum="original"):
    return {
        "schema": 2,
        "tasks": {f"task_{model_id}": {"modes": {"standard": [model_id]}}},
        "models": [
            {
                "model_id": model_id,
                "files": [
                    {"path": "fold_0/checkpoint_final.pth", "bytes": 10000, "sha256": checksum}
                ],
            }
        ],
    }


def test_concurrent_subset_updates_preserve_every_task_and_pin(tmp_path):
    path = tmp_path / "manifest.json"
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda i: preparation.publish_manifest(path, manifest(i)), [297, 315, 615]))
    result = json.loads(path.read_text())
    assert [r["model_id"] for r in result["models"]] == [297, 315, 615]
    assert set(result["tasks"]) == {"task_297", "task_315", "task_615"}
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_race_cannot_replace_another_preparations_pins(tmp_path):
    path = tmp_path / "manifest.json"
    preparation.publish_manifest(path, manifest(297))
    before = path.read_bytes()
    with pytest.raises(WeightError, match="changed"):
        preparation.publish_manifest(path, manifest(297, checksum="different"))
    assert path.read_bytes() == before

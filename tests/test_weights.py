"""Weight preparation/readiness is independent of model invocation and Agent claims."""

import json
import shutil

import pytest

from medsegagent import weights
from medsegagent.task_specs import model_record


def prepare(root, model_id):
    record = model_record(model_id)
    base = root / (record.get("rel_path") or "") / record["foldername"]
    for config in record["expected_configs"]:
        directory = base / f"{config['trainer']}__{config['plans']}__{config['model']}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "dataset.json").write_text('{"labels":{"background":0,"target":1}}')
        (directory / "plans.json").write_text('{"configurations":{}}')
        for fold in config["folds"] if config["folds"] is not None else [0, 2]:
            path = directory / f"fold_{fold}" / "checkpoint_final.pth"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"synthetic-checkpoint" * 512)
    return base


def test_crop_weights_are_required_even_with_primary_checkpoint(tmp_path):
    prepare(tmp_path, 913)
    result = weights.inspect_weights("lung_nodules", root=tmp_path)
    assert result["missing_model_ids"] == [298] and not result["ready"]
    prepare(tmp_path, 298)
    assert weights.inspect_weights("lung_nodules", root=tmp_path)["ready"]


def test_all_explicit_folds_are_required(tmp_path):
    base = prepare(tmp_path, 713)
    assert weights.inspect_model(713, root=tmp_path)["ready"]
    next(base.rglob("fold_3/checkpoint_final.pth")).unlink()
    assert not weights.inspect_model(713, root=tmp_path)["ready"]


def test_auto_discovery_requires_real_numeric_folds(tmp_path):
    base = prepare(tmp_path, 315)
    assert weights.inspect_model(315, root=tmp_path)["ready"]
    for p in base.rglob("checkpoint_final.pth"):
        p.unlink()
    assert not weights.inspect_model(315, root=tmp_path)["ready"]


def test_missing_and_invalid_configuration_never_count_ready(tmp_path):
    base = prepare(tmp_path, 297)
    plans = next(base.rglob("plans.json"))
    plans.write_text("invalid-json")
    assert not weights.inspect_model(297, root=tmp_path)["ready"]
    plans.write_text("[]")
    assert not weights.inspect_model(297, root=tmp_path)["ready"]


def test_pinned_digest_mismatch_is_not_silently_replaced(tmp_path):
    base = prepare(tmp_path, 297)
    first = weights.verify_model(297, root=tmp_path)
    expected = {r["path"]: {**r, "kind": "legacy-extra-field"} for r in first["files"]}
    assert weights.verify_model(297, root=tmp_path, expected=expected)["files"] == first["files"]
    path = next(base.rglob("checkpoint_final.pth"))
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(weights.WeightError, match="pinned"):
        weights.verify_model(297, root=tmp_path, expected=expected)


def test_auto_discovered_ensemble_cannot_silently_lose_a_pinned_fold(tmp_path):
    base = prepare(tmp_path, 315)
    expected = {r["path"]: r for r in weights.verify_model(315, root=tmp_path)["files"]}
    shutil.rmtree(next(base.rglob("fold_2")))
    assert weights.inspect_model(315, root=tmp_path)["ready"]
    with pytest.raises(weights.WeightError, match="file set"):
        weights.verify_model(315, root=tmp_path, expected=expected)


def test_path_precedence_matches_upstream_without_importing_its_runtime(tmp_path, monkeypatch):
    from totalsegmentator.config import get_weights_dir

    monkeypatch.delenv("TOTALSEG_HOME_DIR", raising=False)
    monkeypatch.delenv("TOTALSEG_WEIGHTS_PATH", raising=False)
    assert weights.weights_root() == get_weights_dir()
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path / "home"))
    assert weights.weights_root() == get_weights_dir()
    monkeypatch.setenv("TOTALSEG_WEIGHTS_PATH", str(tmp_path / "weights"))
    assert weights.weights_root() == get_weights_dir()


def test_require_weights_returns_safe_error_without_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "weights_root", lambda root=None: tmp_path)
    with pytest.raises(weights.WeightError) as caught:
        weights.require_weights("total", "fast", ["liver"])
    assert caught.value.code == "WEIGHTS_MISSING"
    assert str(tmp_path) not in str(caught.value)


def test_inventory_does_not_advertise_unavailable_tasks_as_ready(tmp_path):
    result = weights.inventory(root=tmp_path)
    assert result["available_count"] == 33 and result["ready_count"] == 0
    assert len(result["tasks"]) == 53
    encoded = json.dumps(result)
    assert str(tmp_path) not in encoded
    assert (
        next(r for r in result["tasks"] if r["task"] == "brain_aneurysm")["usage_license"]
        == "CC-BY-NC-4.0"
    )

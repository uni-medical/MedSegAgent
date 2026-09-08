"""The verification command checks current public weights unless explicitly given old pins."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "verify_weights", Path(__file__).parents[1] / "ops/verify_weights.py"
)
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)


def model(root, model_id, *, legacy=False):
    folder = f"Dataset{model_id}"
    path = root / folder / "fold_0/checkpoint_final.pth"
    path.parent.mkdir(parents=True)
    data = f"synthetic checkpoint {model_id}".encode()
    path.write_bytes(data)
    return {
        **({"task_id": model_id, "task": "legacy_task"} if legacy else {"model_id": model_id}),
        "folder": folder,
        "files": [
            {
                "path": str(path.relative_to(root / folder)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(verification, "load_dotenv", lambda: False)
    monkeypatch.setattr(verification, "__file__", str(tmp_path / "verify_weights.py"))
    root = tmp_path / "weights"
    current = [model(root, 8), model(root, 113)]
    (tmp_path / "open-model-manifest.json").write_text(json.dumps({"models": current}))
    (tmp_path / "model-manifest.json").write_text(json.dumps({"models": current[:1]}))
    return root, current


def test_default_manifest_verifies_models_beyond_the_legacy_subset(setup, tmp_path, capsys):
    root, current = setup
    extra = root / current[1]["folder"] / current[1]["files"][0]["path"]
    original = extra.read_bytes()
    extra.unlink()
    with pytest.raises(SystemExit, match="Missing or incomplete weights: Dataset113"):
        verification.main(["--root", str(root)])
    extra.write_bytes(original)
    verification.main(["--root", str(root)])
    assert "Verified Dataset113" in capsys.readouterr().out


def test_legacy_manifest_is_supported_when_explicitly_selected(setup, tmp_path, capsys):
    root, _ = setup
    legacy = model(root, 913, legacy=True)
    path = tmp_path / "old-pins.json"
    path.write_text(json.dumps({"models": [legacy]}))
    # The explicit legacy path is independent of the default manifest's newer entries.
    (tmp_path / "open-model-manifest.json").write_text("invalid default, intentionally unused")
    verification.main(["--root", str(root), "--manifest", str(path)])
    assert "Verified legacy_task (Dataset913)" in capsys.readouterr().out


def test_same_size_checkpoint_corruption_fails_hash_verification(setup):
    root, current = setup
    entry = current[1]
    path = root / entry["folder"] / entry["files"][0]["path"]
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(SystemExit, match="Weight checksum mismatch: Dataset113"):
        verification.main(["--root", str(root)])


def test_default_cache_root_matches_runtime_configuration(setup, monkeypatch, capsys):
    root, _ = setup
    monkeypatch.setenv("TOTALSEG_WEIGHTS_PATH", str(root))
    verification.main([])
    assert "Verified Dataset113" in capsys.readouterr().out

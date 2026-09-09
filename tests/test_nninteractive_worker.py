"""Geometry/API contract tests with a fake session, NOT model-inference acceptance."""

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from medsegagent import nninteractive_worker as worker


class FakeSession:
    """Deterministic edits expose coordinate/order errors without a GPU/model."""

    license = "CC BY-NC-SA 4.0"

    def __init__(self):
        self.calls = []

    def initialize_from_trained_model_folder(self, model_path, *, use_fold):
        self.calls.append(("load", model_path, use_fold))

    def set_image(self, image):
        self.image = image.copy()
        self.calls.append(("image", image.shape))

    def set_target_buffer(self, target):
        self.target = target
        self.calls.append(("target", target.shape))

    def add_initial_seg_interaction(self, initial_seg, *, run_prediction):
        assert run_prediction is False
        self.target[:] = initial_seg
        self.calls.append(("initial",))

    def add_point_interaction(self, coordinates, *, include_interaction, run_prediction):
        assert run_prediction is True
        self.calls.append(("point", coordinates, include_interaction))
        self.target[coordinates] = include_interaction

    def add_bbox_interaction(self, bounds, *, include_interaction, run_prediction):
        assert run_prediction is True
        self.calls.append(("box", bounds, include_interaction))
        self.target[tuple(slice(lo, hi) for lo, hi in bounds)] = include_interaction


@pytest.fixture
def case(tmp_path, monkeypatch):
    # All axes differ, so x/z reversal cannot accidentally pass.
    data = np.arange(5 * 7 * 11, dtype=np.float32).reshape(5, 7, 11) - 900
    image = nib.Nifti1Image(data, np.diag([0.7, 1.3, 4.1, 1.0]))
    image.header.set_xyzt_units("mm")
    path = tmp_path / "source.nii.gz"
    nib.save(image, path)
    model = tmp_path / "nnInteractive_v1.0"
    (model / "fold_0").mkdir(parents=True)
    for name in ("dataset.json", "plans.json", "inference_session_class.json"):
        (model / name).write_text("{}")
    (model / "fold_0/checkpoint_final.pth").write_bytes(b"fake checkpoint for API-only test")
    session = FakeSession()
    monkeypatch.setattr(worker, "_load_session", lambda device: (session, "2.5.1"))
    request = {
        "image_path": str(path),
        "output_path": str(tmp_path / "result.nii.gz"),
        "model_path": str(model),
        "device": "cpu",
        "prompts": [{"kind": "point", "voxel": [2, 3, 8], "positive": True}],
    }
    return request, session


@pytest.mark.parametrize(
    "affine",
    [
        np.array([[-0.7, 0, 0, 95], [0, 1.3, 0, -25], [0, 0, 4.1, 17], [0, 0, 0, 1]]),
        np.array([[0, -1.3, 0.4, 35], [0.7, 0, 0.2, -14], [0, 0, 4.1, 8], [0, 0, 0, 1]]),
    ],
    ids=["left-handed-translated", "permuted-oblique-sheared"],
)
def test_source_xyz_and_world_geometry_survive_without_reorientation(case, affine):
    request, session = case
    original = nib.load(request["image_path"])
    source = nib.Nifti1Image(np.asanyarray(original.dataobj), affine, original.header.copy())
    source.set_qform(None, 0)
    source.set_sform(affine, 4)
    nib.save(source, request["image_path"])
    source = nib.load(request["image_path"])
    response = worker.run_request(request)
    output = nib.load(request["output_path"])

    assert response["shape"] == [5, 7, 11]
    np.testing.assert_array_equal(session.image[0], np.asanyarray(source.dataobj))
    assert session.image.shape == (1, 5, 7, 11)
    expected = np.zeros(source.shape, dtype=np.uint8)
    expected[2, 3, 8] = 1
    np.testing.assert_array_equal(output.get_fdata(), expected)
    np.testing.assert_array_equal(output.affine, source.affine)
    np.testing.assert_allclose(
        nib.affines.apply_affine(output.affine, [2, 3, 8]),
        nib.affines.apply_affine(source.affine, [2, 3, 8]),
    )
    assert output.get_qform(coded=True)[1] == 0
    assert output.get_sform(coded=True)[1] == 4
    assert output.header.get_xyzt_units()[0] == "mm"
    assert output.get_data_dtype() == np.dtype("uint8")
    assert response["voxel_count"] == 1
    assert response["coordinate_space"] == "source_voxel_xyz"


def test_nifti_scaled_intensities_are_preserved_and_not_windowed(case):
    request, session = case
    data = np.arange(5 * 7 * 11, dtype=np.int16).reshape(5, 7, 11)
    source = nib.Nifti1Image(data, np.eye(4))
    source.header.set_slope_inter(3.5, -1200)
    nib.save(source, request["image_path"])
    worker.run_request(request)
    np.testing.assert_array_equal(session.image[0], data * 3.5 - 1200)
    assert session.image.min() == -1200
    output = nib.load(request["output_path"])
    assert set(np.unique(output.get_fdata())) == {0, 1}


def test_seed_then_ordered_positive_negative_point_and_box_replay(case):
    request, session = case
    image = nib.load(request["image_path"])
    seed = np.zeros(image.shape, dtype=np.uint8)
    seed[4, 6, 10] = 1
    seed_path = Path(request["output_path"]).with_name("seed.nii.gz")
    nib.save(nib.Nifti1Image(seed, image.affine, image.header), seed_path)
    request["initial_mask_path"] = str(seed_path)
    box = [[1, 4], [2, 5], [8, 9]]
    request["prompts"] = [
        {"kind": "box", "bounds": box, "positive": True},
        {"kind": "point", "voxel": [2, 3, 8], "positive": False},
        {"kind": "point", "voxel": [0, 1, 10], "positive": True},
    ]
    response = worker.run_request(request)
    assert [call[0] for call in session.calls] == [
        "load",
        "image",
        "target",
        "initial",
        "box",
        "point",
        "point",
    ]
    assert session.calls[4:] == [
        ("box", box, True),
        ("point", (2, 3, 8), False),
        ("point", (0, 1, 10), True),
    ]
    expected = seed.copy()
    expected[1:4, 2:5, 8:9] = 1
    expected[2, 3, 8] = 0
    expected[0, 1, 10] = 1
    np.testing.assert_array_equal(nib.load(request["output_path"]).get_fdata(), expected)
    assert response["voxel_count"] == 10
    assert response["initial_mask_used"] is True
    assert response["predictions_run"] == 3


@pytest.mark.parametrize(
    "prompt",
    [
        {"kind": "point", "voxel": [5, 0, 0], "positive": True},
        {"kind": "point", "voxel": [-1, 0, 0], "positive": True},
        {"kind": "point", "voxel": [1.0, 2, 3], "positive": True},
        {"kind": "point", "voxel": [True, 2, 3], "positive": True},
        {"kind": "point", "voxel": [1, 2, 3], "positive": "false"},
        {"kind": "box", "bounds": [[0, 2], [0, 2], [0, 2]], "positive": True},
        {"kind": "box", "bounds": [[0, 1], [0, 1], [0, 2]], "positive": True},
        {"kind": "box", "bounds": [[0, 2], [0, 2], [11, 12]], "positive": True},
        {"kind": "box", "bounds": [[2, 2], [0, 2], [0, 1]], "positive": True},
        {"kind": "box", "bounds": [[3, 2], [0, 2], [0, 1]], "positive": True},
        {"kind": "scribble", "positive": True},
    ],
)
def test_bad_prompts_rejected_before_loading_model(case, prompt):
    request, session = case
    request["prompts"] = [prompt]
    with pytest.raises(worker.WorkerError, match="prompts") as error:
        worker.run_request(request)
    assert error.value.code == "invalid_prompt"
    assert session.calls == []
    assert not Path(request["output_path"]).exists()


@pytest.mark.parametrize("problem", ["shift", "shape", "nonbinary", "units"])
def test_initial_mask_requires_binary_same_physical_grid(case, problem):
    request, session = case
    image = nib.load(request["image_path"])
    affine = image.affine.copy()
    shape = image.shape
    if problem == "shift":
        affine[0, 3] += 1
    if problem == "shape":
        shape = (7, 5, 11)
    seed = np.zeros(shape, dtype=np.float32)
    if problem == "nonbinary":
        seed[1, 2, 3] = 0.7
    mask = nib.Nifti1Image(seed, affine, image.header.copy())
    if problem == "units":
        mask.header.set_xyzt_units("meter")
    path = Path(request["output_path"]).with_name("seed.nii.gz")
    nib.save(mask, path)
    request["initial_mask_path"] = str(path)
    with pytest.raises(worker.WorkerError) as error:
        worker.run_request(request)
    assert error.value.code == "invalid_initial_mask"
    assert session.calls == []


def test_missing_weights_never_initializes_runtime(case):
    request, session = case
    Path(request["model_path"], "fold_0/checkpoint_final.pth").unlink()
    with pytest.raises(worker.WorkerError, match="never downloads") as error:
        worker.run_request(request)
    assert error.value.code == "model_missing"
    assert session.calls == []


def test_existing_output_is_not_clobbered(case):
    request, session = case
    output = Path(request["output_path"])
    output.write_bytes(b"prior artifact")
    with pytest.raises(worker.WorkerError) as error:
        worker.run_request(request)
    assert error.value.code == "output_exists"
    assert output.read_bytes() == b"prior artifact"
    assert session.calls == []


@pytest.mark.parametrize("problem", ["4d", "nonfinite"])
def test_bad_source_image_is_rejected_before_runtime(case, problem):
    request, session = case
    data = np.zeros((5, 7, 11, 2) if problem == "4d" else (5, 7, 11), dtype=np.float32)
    if problem == "nonfinite":
        data[2, 3, 8] = np.nan
    nib.save(nib.Nifti1Image(data, np.eye(4)), request["image_path"])
    with pytest.raises(worker.WorkerError) as error:
        worker.run_request(request)
    assert error.value.code == "invalid_image"
    assert session.calls == []


def test_initial_mask_without_edits_reports_no_prediction(case):
    request, session = case
    image = nib.load(request["image_path"])
    seed = np.zeros(image.shape, dtype=np.uint8)
    seed[2, 3, 8] = 1
    path = Path(request["output_path"]).with_name("seed.nii.gz")
    nib.save(nib.Nifti1Image(seed, image.affine, image.header), path)
    request.update(initial_mask_path=str(path), prompts=[])
    response = worker.run_request(request)
    np.testing.assert_array_equal(nib.load(request["output_path"]).get_fdata(), seed)
    assert response["predictions_run"] == 0
    assert [call[0] for call in session.calls] == ["load", "image", "target", "initial"]


def test_nonbinary_backend_output_is_not_published(case, monkeypatch):
    request, session = case

    def invalid_prediction(*args, **kwargs):
        session.target[2, 3, 8] = 2

    monkeypatch.setattr(session, "add_point_interaction", invalid_prediction)
    with pytest.raises(worker.WorkerError) as error:
        worker.run_request(request)
    assert error.value.code == "invalid_output"
    assert not Path(request["output_path"]).exists()


def test_cli_reports_error_json_without_importing_nninteractive(case, tmp_path):
    request, _ = case
    Path(request["model_path"], "fold_0/checkpoint_final.pth").unlink()
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    request_path.write_text(json.dumps(request))
    result = subprocess.run(
        [
            sys.executable,
            worker.__file__,
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    response = json.loads(response_path.read_text())
    assert response["ok"] is False
    assert response["backend"] == "nninteractive"
    assert response["error"]["code"] == "model_missing"
    assert not Path(request["output_path"]).exists()


def test_cli_will_not_write_json_over_source_image(case, tmp_path):
    request, _ = case
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    before = Path(request["image_path"]).read_bytes()
    with pytest.raises(SystemExit) as error:
        worker.main(["--request", str(request_path), "--response", request["image_path"]])
    assert error.value.code == 2
    assert Path(request["image_path"]).read_bytes() == before


def test_unsupported_runtime_version_fails_before_torch_import(monkeypatch):
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda _: "99.0")
    with pytest.raises(worker.WorkerError, match="requires nnInteractive==2.5.1") as error:
        worker._load_session("cpu")
    assert error.value.code == "unsupported_version"

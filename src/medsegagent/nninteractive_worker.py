"""One-request nnInteractive process with an explicit source-voxel XYZ contract.

Run this file with the isolated nnInteractive interpreter, not the web service's
Python. No model download, reorientation, windowing or resampling happens here.
The caller owns cancellation, GPU exclusivity, and publication of the result.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

BACKEND = "nninteractive"
SUPPORTED_VERSION = "2.5.1"


class WorkerError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("invalid_request", f"{field} must be a nonempty path string")
    return Path(value).expanduser().resolve()


def _integer(value: Any, field: str) -> int:
    # bool is a Python int subclass but is never a valid voxel coordinate.
    if type(value) is not int:
        raise WorkerError("invalid_prompt", f"{field} must be an integer voxel index")
    return value


def _validate_prompts(prompts: Any, shape: tuple[int, ...]) -> list[dict[str, Any]]:
    if not isinstance(prompts, list):
        raise WorkerError("invalid_prompt", "prompts must be an ordered list")
    normalized = []
    for index, prompt in enumerate(prompts):
        prefix = f"prompts[{index}]"
        if not isinstance(prompt, dict) or type(prompt.get("positive")) is not bool:
            raise WorkerError("invalid_prompt", f"{prefix}.positive must be a boolean")
        kind = prompt.get("kind")
        if kind == "point":
            voxel = prompt.get("voxel")
            if not isinstance(voxel, list) or len(voxel) != 3:
                raise WorkerError("invalid_prompt", f"{prefix}.voxel must be [x, y, z]")
            coords = [_integer(v, f"{prefix}.voxel[{i}]") for i, v in enumerate(voxel)]
            if any(not 0 <= v < shape[i] for i, v in enumerate(coords)):
                raise WorkerError("invalid_prompt", f"{prefix} point is outside source image")
            normalized.append({"kind": kind, "voxel": coords, "positive": prompt["positive"]})
        elif kind == "box":
            bounds = prompt.get("bounds")
            if not isinstance(bounds, list) or len(bounds) != 3:
                raise WorkerError("invalid_prompt", f"{prefix}.bounds must contain three ranges")
            parsed = []
            for axis, pair in enumerate(bounds):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise WorkerError("invalid_prompt", f"{prefix}.bounds[{axis}] needs [lo, hi]")
                lo, hi = [_integer(v, f"{prefix}.bounds[{axis}]") for v in pair]
                if not 0 <= lo < hi <= shape[axis]:
                    raise WorkerError(
                        "invalid_prompt",
                        f"{prefix} box must have nonempty in-bounds half-open ranges",
                    )
                parsed.append([lo, hi])
            if sum(hi - lo == 1 for lo, hi in parsed) != 1:
                raise WorkerError(
                    "invalid_prompt", f"{prefix} box must have exactly one single-voxel-thick axis"
                )
            normalized.append({"kind": kind, "bounds": parsed, "positive": prompt["positive"]})
        else:
            raise WorkerError("invalid_prompt", f"{prefix}.kind must be point or box")
    return normalized


def _validate_model(model_path: Path) -> None:
    if not model_path.is_dir():
        raise WorkerError("model_missing", f"model_path is not a local model folder: {model_path}")
    required = ["dataset.json", "plans.json", "fold_0/checkpoint_final.pth"]
    if not any(
        (model_path / name).is_file()
        for name in ("inference_info.json", "inference_session_class.json")
    ):
        required.append("inference_info.json or inference_session_class.json")
    missing = [
        name
        for name in required
        if not (model_path / name).is_file() or (model_path / name).stat().st_size == 0
    ]
    if missing:
        raise WorkerError(
            "model_missing",
            f"Incomplete local nnInteractive model: {', '.join(missing)}. "
            "Provision the official checkpoint separately; this worker never downloads weights.",
        )


def _load_session(device: str):
    try:
        version = importlib.metadata.version("nnInteractive")
    except importlib.metadata.PackageNotFoundError as exc:
        raise WorkerError(
            "dependency_missing", "nnInteractive is missing; use the ops/nninteractive interpreter"
        ) from exc
    if version != SUPPORTED_VERSION:
        raise WorkerError(
            "unsupported_version",
            f"This adapter requires nnInteractive=={SUPPORTED_VERSION}; found {version}",
        )
    try:
        import torch
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    except ImportError as exc:
        raise WorkerError(
            "dependency_missing", f"Cannot import nnInteractive runtime: {exc}"
        ) from exc
    if device.startswith("cuda:"):
        ordinal = int(device.split(":")[1])
        if not torch.cuda.is_available() or ordinal >= torch.cuda.device_count():
            raise WorkerError("device_unavailable", f"Requested device {device} is unavailable")
    session = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=min(os.cpu_count() or 1, 8),
        do_autozoom=True,
        enable_undo=False,
    )
    return session, version


def _load_image(image_path: Path):
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise WorkerError("dependency_missing", f"Missing image dependency: {exc}") from exc
    if not image_path.is_file():
        raise WorkerError("invalid_image", f"Source image does not exist: {image_path}")
    try:
        image = nib.load(image_path)
        if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
            raise TypeError("expected a NIfTI image")
        if len(image.shape) != 3 or any(size < 1 for size in image.shape):
            raise ValueError("expected a nonempty 3D image")
        affine = image.affine
        if (
            affine is None
            or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12
        ):
            raise ValueError("source voxel-to-world affine must be finite and invertible")
        # ArrayProxy applies NIfTI slope/intercept. This preserves physical image
        # intensities, unlike reading unscaled storage or converting to uint8.
        data = np.asanyarray(image.dataobj)
        if data.dtype.kind not in "iuf" or not np.isfinite(data).all():
            raise ValueError("image must contain finite real numerical intensities")
    except Exception as exc:
        raise WorkerError("invalid_image", f"Invalid source NIfTI: {exc}") from exc
    return image, np.ascontiguousarray(data)


def _load_initial_mask(mask_path: Path, image):
    import nibabel as nib
    import numpy as np

    try:
        mask = nib.load(mask_path)
        if not isinstance(mask, (nib.Nifti1Image, nib.Nifti2Image)):
            raise TypeError("expected a NIfTI mask")
        if mask.shape != image.shape or not np.allclose(
            mask.affine, image.affine, rtol=0, atol=1e-5
        ):
            raise ValueError("initial mask shape and affine must match the source image grid")
        if mask.header.get_xyzt_units()[0] != image.header.get_xyzt_units()[0]:
            raise ValueError("initial mask and source must use the same spatial units")
        data = np.asanyarray(mask.dataobj)
        if not np.isfinite(data).all() or not np.isin(data, [0, 1]).all():
            raise ValueError("initial mask must be binary, containing only 0 and 1")
    except Exception as exc:
        raise WorkerError("invalid_initial_mask", str(exc)) from exc
    return np.ascontiguousarray(data, dtype=np.uint8)


def _save_mask(mask, image, output_path: Path) -> None:
    import nibabel as nib
    import numpy as np

    if mask.shape != image.shape or not np.isin(mask, [0, 1]).all():
        raise WorkerError("invalid_output", "Model target buffer must be binary on the source grid")
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    header.set_intent("label")
    header["cal_min"], header["cal_max"] = 0, 1
    result = image.__class__(mask.astype(np.uint8, copy=False), image.affine.copy(), header)
    # Preserve both forms/codes, including oblique/left-handed sforms and qform=0.
    for name in ("qform", "sform"):
        transform, code = getattr(image, f"get_{name}")(coded=True)
        getattr(result, f"set_{name}")(transform, int(code))
    result.header.set_slope_inter(1, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".nninteractive-", suffix=".nii.gz", dir=output_path.parent
    )
    os.close(fd)
    try:
        nib.save(result, temporary)
        # Refuse to clobber another result if the destination appeared during inference.
        os.link(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_request(request: Any) -> dict[str, Any]:
    """Validate before loading Torch, then replay prompts in their given order."""
    if not isinstance(request, dict):
        raise WorkerError("invalid_request", "request must be a JSON object")
    image_path = _path(request.get("image_path"), "image_path")
    output_path = _path(request.get("output_path"), "output_path")
    model_path = _path(request.get("model_path"), "model_path")
    initial_path = (
        _path(request["initial_mask_path"], "initial_mask_path")
        if request.get("initial_mask_path") is not None
        else None
    )
    if not str(output_path).endswith(".nii.gz"):
        raise WorkerError("invalid_request", "output_path must end with .nii.gz")
    if output_path in (image_path, initial_path) or output_path.exists():
        raise WorkerError("output_exists", "output_path must be a new file, distinct from inputs")
    device = request.get("device", "cuda:0")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise WorkerError("invalid_request", "device must be cpu or cuda:<integer>")
    image, data = _load_image(image_path)
    prompts = _validate_prompts(request.get("prompts"), image.shape)
    initial_mask = _load_initial_mask(initial_path, image) if initial_path else None
    if not prompts and initial_mask is None:
        raise WorkerError("invalid_prompt", "Provide at least one prompt or an initial_mask_path")
    _validate_model(model_path)
    session, version = _load_session(device)
    session.initialize_from_trained_model_folder(str(model_path), use_fold=0)
    # nibabel array indices already match request voxel=[x,y,z]. No transpose.
    session.set_image(data[None])
    import numpy as np

    target = np.zeros(image.shape, dtype=np.uint8)
    session.set_target_buffer(target)
    if initial_mask is not None:
        # This upstream operation resets prior interactions, so it must run FIRST.
        session.add_initial_seg_interaction(initial_mask, run_prediction=False)
    for prompt in prompts:
        if prompt["kind"] == "point":
            session.add_point_interaction(
                tuple(prompt["voxel"]), include_interaction=prompt["positive"], run_prediction=True
            )
        else:
            session.add_bbox_interaction(
                prompt["bounds"], include_interaction=prompt["positive"], run_prediction=True
            )
    _save_mask(target, image, output_path)
    return {
        "ok": True,
        "backend": BACKEND,
        "version": version,
        "model_path": str(model_path),
        "model_license": getattr(session, "license", None),
        "device": device,
        "output_path": str(output_path),
        "shape": list(image.shape),
        "affine": image.affine.tolist(),
        "coordinate_space": "source_voxel_xyz",
        "voxel_count": int(np.count_nonzero(target)),
        "prompts_applied": len(prompts),
        "predictions_run": len(prompts),
        "initial_mask_used": initial_mask is not None,
    }


def _write_json(path: Path, response: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".nninteractive-response-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(response, stream, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.response.resolve() == args.request.resolve():
        parser.error("--response must be distinct from --request")
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        # The response is a separate artifact; never overwrite the request, image,
        # initial mask, or the requested segmentation with JSON.
        protected = [args.request.resolve()]
        if isinstance(request, dict):
            protected.extend(
                Path(request[key]).expanduser().resolve()
                for key in ("image_path", "initial_mask_path", "output_path")
                if isinstance(request.get(key), str) and request[key].strip()
            )
        if args.response.resolve() in protected:
            parser.error("--response must be distinct from the request and image/mask paths")
        response = run_request(request)
    except Exception as exc:  # noqa: BLE001 -- process boundary must report third-party failures
        response = {
            "ok": False,
            "backend": BACKEND,
            "version": SUPPORTED_VERSION,
            "error": {"code": getattr(exc, "code", "worker_failed"), "message": str(exc)},
        }
    _write_json(args.response.resolve(), response)
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

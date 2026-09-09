# Optional nnInteractive worker

This runtime is separate from the MedSegAgent/TotalSegmentator environment. It is
an opt-in backend, not a production install or deployment. The adapter's geometry
and API tests pass with a fake session; real checkpoint output, CPU operation,
RTX 3090 memory and latency still need acceptance testing on the target host.

## Dependency isolation

The worker targets the released **nnInteractive 2.5.1** API. On a separate Linux
GPU checkout, create only this subproject's environment:

```sh
uv sync --project ops/nninteractive --python 3.12
```

This creates `ops/nninteractive/.venv`, not the service's `.venv`. The runtime pins
nnInteractive 2.5.1 (which itself pins nninteractive-client 2.5.1), nnU-Net 2.8.1,
PyTorch 2.8.0 with CUDA 12.6 wheels, nibabel 5.4.2, and NumPy 2.5.3. CUDA 12.6
needs a compatible NVIDIA driver. The PyTorch 2.8 choice avoids the upstream
explicit exclusion of `torch==2.9.*`; CUDA 12.6 follows the upstream documented
installation route. The combined set was resolved successfully with Python
3.12.13 into this subproject's `uv.lock` (109 packages), **without installing a
runtime or downloading model weights**. Use `uv sync --locked --project
ops/nninteractive --python 3.12` to install the resolved set, and record
`uv pip freeze --python ops/nninteractive/.venv/bin/python` during host acceptance.
Do not run pip against the production environment or add these dependencies to
the main project.

No installation or weight download is performed by the worker. A configured
worker interpreter missing nnInteractive, or running a different version,
returns an explicit error. `device="cpu"` is supported by the adapter/upstream
session API, but CPU speed and checkpoint operation have not been measured here.

## Provision weights separately

The official model is available from
[MIC-DKFZ/nnInteractive](https://huggingface.co/MIC-DKFZ/nnInteractive). Download it
through a separate, deliberate provisioning step, then verify the checkpoint hash
and record the license. `model_path` must point to the **model subfolder**, not a
checkpoint file or the Hugging Face cache root:

```text
nnInteractive_v1.0/
  dataset.json
  plans.json
  inference_session_class.json   # or newer inference_info.json
  LICENSE
  fold_0/
    checkpoint_final.pth
```

The code is Apache-2.0; the official checkpoint is **CC BY-NC-SA 4.0**. Shipping
the platform's source does not grant commercial-use rights to this checkpoint.
The worker returns the license reported by the loaded model as `model_license`.
Only provision a trusted checkpoint: upstream uses PyTorch's checkpoint loading
with `weights_only=False`.

## Command and request

Call the standalone script with the isolated interpreter; installing the main
MedSegAgent package into this runtime is unnecessary:

```sh
ops/nninteractive/.venv/bin/python src/medsegagent/nninteractive_worker.py \
  --request /path/to/request.json --response /path/to/response.json
```

```json
{
  "image_path": "/path/to/original.nii.gz",
  "output_path": "/path/to/new-mask.nii.gz",
  "model_path": "/path/to/nnInteractive_v1.0",
  "device": "cuda:0",
  "initial_mask_path": "/path/to/optional-binary-seed.nii.gz",
  "prompts": [
    {"kind": "box", "bounds": [[30, 80], [40, 100], [10, 11]], "positive": true},
    {"kind": "point", "voxel": [50, 70, 10], "positive": false}
  ]
}
```

- `image_path` is an original, finite, real-valued 3D NIfTI. NIfTI intensity
  slope/intercept is applied on reading, without windowing, uint8 conversion,
  normalization, axis permutation, canonical reorientation, or resampling.
- Coordinates are integer **source array voxel XYZ**, the indices returned by
  nibabel. They are not screen coordinates, millimetres, or SimpleITK's ZYX array
  indexing. A prompt `[x,y,z]` addresses `source[x,y,z]` directly. The upstream
  session documents its input as `[C,X,Y,Z]` and currently ignores spacing.
- Box ranges are half-open `[lo,hi)` and stay inside the source image. Exactly
  one axis must have extent one; this is a **2D box on one source slice**, not a
  volumetric box. The worker rejects degenerate line/point boxes.
- The optional seed must be binary and match the source's shape, effective
  affine (absolute tolerance `1e-5`), and spatial units. It is never resampled.
- Each request replays the complete prompt history in supplied order. If using
  a seed, that seed is the base **before** this history; do not seed with the last
  edited mask and also replay interactions already represented in it.
- `initial_mask_path` can be omitted. With no seed, at least one prompt is
  required. A seed-only request returns the seed with `predictions_run=0`, rather
  than claiming a model prediction occurred.
- `output_path` must be a new `.nii.gz` file. Existing results and input images
  are not overwritten. Output is a binary uint8 NIfTI with the original shape,
  affine, spatial units, qform/sform and form codes. No resizing back is needed.

Successful response (exit 0) includes:

```json
{
  "ok": true,
  "backend": "nninteractive",
  "version": "2.5.1",
  "shape": [512, 512, 280],
  "coordinate_space": "source_voxel_xyz",
  "voxel_count": 12700,
  "prompts_applied": 2,
  "predictions_run": 2,
  "initial_mask_used": true,
  "output_path": "/path/to/new-mask.nii.gz"
}
```

The complete response additionally has `affine`, `device`, `model_path`, and
`model_license`. These illustrative numbers are not inference measurements.
Failures return exit 1 and
`{"ok":false,"backend":"nninteractive","version":"2.5.1","error":{"code":"...","message":"..."}}`.
The error response's version identifies the adapter's target version; a version
mismatch message reports the actual installed version. The response JSON is the
machine-readable channel; upstream progress output can appear on stdout/stderr.
CLI misuse, including a response path aliasing an input, exits 2 without writing
over that input. Callers must check exit code and `ok`, not a leftover response.

## Verified upstream sequence

The released wheel (not just the GitHub development branch) was inspected for
these APIs, without importing or installing the package:

```python
session = nnInteractiveInferenceSession(
    device=torch.device(device), use_torch_compile=False,
    verbose=False, torch_n_threads=8, do_autozoom=True, enable_undo=False,
)
session.initialize_from_trained_model_folder(model_path, use_fold=0)
session.set_image(source_xyz[None])
session.set_target_buffer(binary_target_xyz)
session.add_initial_seg_interaction(seed_xyz, run_prediction=False)  # optional, FIRST
session.add_point_interaction((x, y, z), include_interaction=True, run_prediction=True)
session.add_bbox_interaction(bounds_xyz, include_interaction=False, run_prediction=True)
```

`add_initial_seg_interaction` exists in 2.5.1 and resets all existing interactions.
The adapter leaves capability checks enabled; unsupported model capabilities
fail rather than being silently overridden. It does not use target labels to
generate point/box prompts.

## Process lifetime and acceptance

The first implementation loads the model per request and replays the history.
This costs cold-start/preprocessing time but makes GPU release, cancellation,
isolation from TotalSegmentator, and reproducible requests straightforward. The
parent service owns the GPU lease and kills the whole worker process on cancel.
Compilation and undo snapshots are disabled because neither amortizes here.

After real acceptance, the smallest latency improvement is to keep one dedicated
worker alive per interactive session under an exclusive GPU lease: load once,
set the image once, then append new prompts. Bind the session to image/model/seed
hashes and the exact prompt-history prefix; changed history must reconstruct the
session. Apply an idle TTL and terminate the process before another backend gets
the lease. The upstream HTTP server is another supported future route. Do not
add a second scheduler inside this worker.

Acceptance must separately measure initial model load, image preparation,
per-prompt latency, full replay, peak GPU allocated/reserved and process VRAM,
host RAM, cancellation/recovery, and actual segmentation quality. The official
recommendation of roughly 10 GB VRAM is not a measured guarantee for this
adapter or every volume. Include anisotropic CT/MR, oblique/left-handed geometry,
long volumes, positive/negative prompts and an initial mask. Fake-session tests
verify coordinate and API contracts only; they are not a GPU or clinical test.

Sources checked 2026-09-09:

- [Official README](https://github.com/MIC-DKFZ/nnInteractive)
- [Released 2.5.1 metadata](https://pypi.org/pypi/nnInteractive/2.5.1/json)
- [Upstream inference session](https://github.com/MIC-DKFZ/nnInteractive/blob/master/nnInteractive/inference/inference_session.py)
- [Official checkpoint](https://huggingface.co/MIC-DKFZ/nnInteractive)

The inspected PyPI 2.5.1 wheel SHA-256 is
`19517439f040497729ad8c4649e122695ec905cea02d759222253f646ac19922`.

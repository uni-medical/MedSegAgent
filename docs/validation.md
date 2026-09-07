# Engineering acceptance

Observed on 2026-09-07. These are execution and integration canaries, not clinical
validation, independent benchmark estimates, or medical-device claims. Source images,
masks, credentials, private reports and browser captures are excluded from Git.

## Base CT/MR closure

The base implementation was tested at commit
`8df35465db35ecedfdf77dd47223272d5816cbef` on Mac and pjoffice. Both hosts passed
106 automated tests. Tests cover full NIfTI integrity, nonfinite voxels/header spacing,
DICOM geometry/modality, cross-process locking, subprocess cancellation, identity
isolation, idempotency, timeouts, upload quotas and restart/retention behavior.

Real function calling used `deepseek-v4-flash`. The provider request contained text,
declared modality and allowed targets; images and file metadata stayed local.

| Test | Mac M5 Pro / MPS | pjoffice / NVIDIA GPU |
| --- | --- | --- |
| FLARE22 training case 0014, liver + both kidneys | 13.63 s; 1,092,389 foreground voxels | 51.30 s; 1,091,829 foreground voxels |
| Liver Dice, TotalSegmentator 5 → FLARE22 1 | 0.97279 | 0.97298 |
| Right kidney Dice, 2 → 2 | 0.94317 | 0.94286 |
| Left kidney Dice, 3 → 13 | 0.93068 | 0.93056 |
| Official small MR example, liver | 9.54 s; 17,658 voxels | 24.23 s; 17,660 voxels |

CT shape was 512 × 512 × 71; MR shape was 117 × 91 × 20. Output shape and affine
matched the input on both hosts. Dice used direct label mapping without resampling.
Runtime is the inference subprocess wall time, including local preprocessing; it is
not a throughput guarantee. Small numerical differences across backends are recorded.

## Public API and browser

`https://medseg.huangziyan97.com` was reached through the existing named Cloudflare
Tunnel with an additional hostname rule; the other ingress rules were preserved.
The public Agent Card returned 200 and parsed with the official A2A protobuf types.
Unauthenticated tasks/downloads returned 401; cross-identity tasks/files returned 404.

`ops/acceptance.py` completed real public send and stream tasks, intentionally
disconnected the stream, recovered through GetTask, replayed messageId without a
duplicate task, and downloaded authenticated masks. Both masks were nonempty and
geometrically consistent. Task identifiers for the private evidence:

- send: `bd4167d5-ecac-403a-adfd-fe38139f6434`, inference 34.10 s.
- stream/recover: `441297a0-825c-43e6-b5da-58edb609905b`, inference 32.77 s.
- Web upload/text: `89c094bc-2aa8-4525-b62a-b293fd06f4d0`, inference 32.60 s.

Actual Chromium desktop and 390 × 844 mobile viewport checks showed source CT,
colored liver/kidney overlays in three planes and 3D, label hiding, opacity 55% → 80%,
download, and task/overlay restoration after refresh. Mobile WebGL2 initialized and
the page had no horizontal overflow. This is browser viewport testing, not physical
iPhone/Android certification. A 91 MiB fixture was rejected before upload; corrupt
NIfTI was rejected by the public server and displayed an error. An unsupported brain
tumor request produced a visible failed task without starting inference.

Cloudflare's zone-level analytics script injection was blocked by the application's
same-origin CSP. Viewer/application functionality passed with that script blocked;
the CSP was not loosened to permit third-party analytics.

Private reproducibility evidence lives in `outputs/acceptance/` and
`output/playwright/` on the operator's machine. Service uploads and results follow the
24-hour retention policy, so these task identifiers are audit references rather than
permanent public file links.

## Restart correction

A real running inference initially failed with `INFERENCE_FAILED` during a systemd
restart because simultaneous process-group TERM arrived before the application shutdown
handler. The old failure report was preserved. Switching the unit to `KillMode=mixed`
was verified with an isolated uv signal-forwarding probe and a new real inference:
task `7c420687-bf25-4fb6-8463-654f80228270` became FAILED / `SERVER_RESTART`, and its
inference process group disappeared. Ten checks passed, including unchanged completed
mask/result checksums, preserved login session and unchanged GMAI/Tunnel process IDs.

## Complementary CT lesion models

Two dedicated TotalSegmentator weights were selected after the comparison in
`model-candidates.md`; no MONAI/nnInteractive/SAM model was silently substituted.
Mac and pjoffice each passed **130 automated tests**, covering specialized-task default speed,
native-mask filtering, cancellation and private-file/empty-result projection.
Separate real stdio MCP discovery returned four tools, and empty targets produced ToolError.

The real provider initially truncated a two-tool request to one task. Clarifying that
no tool may execute a partial request fixed the observed case: a new eight-request
provider canary accepted CT lung nodules, CT liver lesions and MR anatomy, and refused
MR lesions, malignancy classification and mixed-tool requests in Chinese and English.
This is observed routing behavior, not a guarantee that a probabilistic model cannot
misinterpret other wording; every selection is recorded for review.

| Mac MPS canary | Full source shape | Lesion voxels | End-to-end seconds | Reference comparison |
| --- | --- | ---: | ---: | --- |
| Lung nodules, Dataset913 | 512 × 512 × 133 | 4,558 (label 2) | 49.22 | Dice 0.81869 with one reader's Nodule 1 |
| Liver lesions, Dataset591 | 212 × 182 × 142 | 54,911 (label 1) | 28.62 | Dice 0.89424 with provided training label |

Both masks preserved full input geometry. The lung source was the complete LIDC-IDRI
0001 CT; DICOM-to-NIfTI conversion was checked against every original voxel after HU
rescale, with zero error. The reference SEG was aligned by referenced SOP UID and frame
position, not filename ordering. It is one annotation, not an exhaustive scan label.
The liver sample came from the publisher's training archive, retrieved with bounded
HTTP Range reads and CRC checks. Neither reference was used for cropping or inference.
These comparisons have possible training overlap and must not be used as clinical
performance or independent validation estimates.

The same inputs and copied, checksum-matched weights then completed on pjoffice's
NVIDIA RTX 3090 (24 GiB, GPU 1):

| GPU canary | Lesion voxels | Inference / pipeline seconds | Reference Dice |
| --- | ---: | ---: | ---: |
| Lung nodules | 4,518 | 62.78 / 68.01 | 0.81627 |
| Liver lesions | 55,274 | 34.00 / 36.63 | 0.89553 |

Mac command-level maximum RSS was 3.89 GB for lung and 5.14 GB for liver; Linux was
4.97 GB and 2.90 GB respectively, measured with native `time`. These are process RSS
measurements, not the device's minimum required memory. Actual tested hosts were an
M5 Pro with 64 GB unified memory and RTX 3090 with 24 GiB VRAM. The artifacts recorded
MPS/CUDA execution; no CPU fallback or unsupported-operator error was observed.
Periodic sampling during the public lesion canaries captured 144 active-process
observations and reached 4,886 MiB of GPU memory; this is an observed value, not a
guaranteed peak or a minimum-VRAM specification.

At implementation commit `421c39a1bc8779eeae7d85f27dc5e9972195f683`, public HTTPS
acceptance was repeated for both lesion tools, including messageId replay, identity
isolation, authenticated files, streaming disconnect and GetTask recovery:

- Lung send `4e248d6d-3879-40b4-b31a-35a4001a13ea`, stream `2dd151e2-646f-4ad6-81e6-f0f194b85c82`: both completed with 4,518 nodule voxels.
- Liver send `1202800e-c19c-4788-b9ec-5710b8e0ff61`, stream `7f835147-98a8-4349-8060-f22ad2329be8`: completed with 55,274 / 55,273 lesion voxels.

Both public lesion masks were viewed against the original image in NiiVue, including
colored lesion overlays in all three planes and 3D. Small CUDA output variation is
recorded; the test does not assert byte-identical repeated model predictions.

The additional Pixel 7 Chromium emulation used Android mobile UA, touch events,
DPR 2.625 and a 412 × 839 viewport. Tap-based labels, opacity, 3D and refresh recovery
passed; portrait and 839px landscape layouts had no horizontal overflow. This remains
device emulation, not testing on a physical phone. Screenshots and full evidence stay private.

`ops/model-manifest.json` records the actual six installed weight sets for the four
tools, including crop dependencies, checkpoint/config SHA256, source URL and license.
`uv run python ops/verify_weights.py` checks local bytes without downloading. The final
release record adds this manifest and validation evidence without changing inference code.

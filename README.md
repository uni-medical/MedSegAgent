# MIA-Omni

An agent workspace for medical imaging, built on MedSegAgent. MIA-Omni brings
specialized models and interactive tools into one workflow, starting with
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) CT/MR segmentation
and [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) point-and-box editing.

[nnInteractive setup](docs/nninteractive.md) ·
[Stable TotalSeg demo](https://medseg.huangziyan97.com) · [Deployment](docs/deployment.md) ·
[CLI & MCP](docs/simple-local-tools.md) · [A2A API](docs/a2a.md)

## Features

- **Interactive region segmentation** — foreground/background points and slice
  boxes in the Web viewer, with saved mask versions, volume comparisons and downloads.
- **Agent-guided editing** — apply your recorded marks and summarize verified
  measurements for the selected region.
- **Natural-language segmentation** — choose from 33 CT/MR tasks for anatomy,
  selected lesions and specialist structures.
- **TotalSeg workflows** — combine models, inspect region volumes and create
  unions, intersections or differences from their masks.
- **Interactive viewing** — linked axial, coronal and sagittal slices, 3D rendering,
  label controls and individual NIfTI downloads.
- **TotalSeg across four interfaces** — Web, CLI, MCP and A2A, with local model
  inference. The nnInteractive extension currently runs through the Web interface.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone --branch mia-omni https://github.com/uni-medical/MedSegAgent.git
cd MedSegAgent
uv sync --frozen
cp .env.example .env
```

Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` in `.env` for a provider serving
`deepseek-v4-flash`. Apple Silicon uses `MEDSEGAGENT_DEVICE=mps`; on NVIDIA Linux,
set it to `gpu`.

Prepare the CT/MR anatomy models and start the app:

```bash
uv run --env-file .env python ops/prepare_weights.py --download --task total --task total_mr
uv run medsegagent doctor
uv run medsegagent serve --host 127.0.0.1 --port 8767
```

Open [localhost:8767](http://127.0.0.1:8767) to enter automatically as a guest and upload a
`.nii` or `.nii.gz` image. Try “Segment the liver and kidneys.” The
[deployment guide](docs/deployment.md) covers the full model catalog and GitHub login.

nnInteractive is **disabled by default** and uses a separate model environment.
Follow the [interactive setup and validation guide](docs/nninteractive.md) to enable
it. The first version edits one region at a time and reloads the model for each
submitted edit; performance measurements and remaining validation are documented there.

## Command line

```bash
uv run medsegagent run --modality CT --text "Segment the liver and kidneys" \
  --input /path/to/scan.nii.gz --output outputs
uv run medsegagent catalog --query lung --modality CT
uv run --env-file .env medsegagent-mcp
```

See [Agent workflow](docs/autonomous-segmentation.md),
[tool reference](docs/tool-management.md) and [development checks](docs/validation.md).

## About

`mia-omni` is an independent development branch. The public demo continues to run
[totalseg-agent](https://github.com/uni-medical/MedSegAgent/tree/totalseg-agent),
without this branch's extensions. The MedSegAgent paper and original experiments
are on [main](https://github.com/uni-medical/MedSegAgent/tree/main).

[Apache-2.0](LICENSE) · [Software, model and example attribution](docs/third-party-notices.md)

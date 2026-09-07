# MedSegAgent

A small research segmentation agent: natural language + a 3D volume → an ordinary
function call → local inference → a NIfTI mask and structured result. The language model
receives only the request, declared modality and allowed targets. Image bytes, paths,
headers and masks stay on the inference host.

This branch replaces the original orchestration and prompt pipeline. The published paper
and its original experiments remain available on [main](https://github.com/uni-medical/MedSegAgent/tree/main).
The historical dataset metadata in `dataset/` is reference material; executable targets
come from the pinned TotalSegmentator registry.

## Start

```bash
uv sync --frozen --group dev
cp .env.example .env
chmod 600 .env
# Configure the HTTPS OpenAI-compatible endpoint, API key and private access tokens.
uv run medsegagent doctor
uv run medsegagent run --modality CT --text 'Segment the liver and kidneys' \
  --input /path/to/scan.nii.gz --output outputs
uv run medsegagent serve --host 127.0.0.1 --port 8767
```

The provider model is fixed to `deepseek-v4-flash`. `.env` is local and must never be
committed. Default compute is Apple MPS on macOS and NVIDIA CUDA on Linux; choose a GPU
with `CUDA_VISIBLE_DEVICES`. All environments use `uv.lock`.

## Interfaces

- **CLI:** `medsegagent run`; `route` tests actual function calling without an image.
- **MCP:** `uv run medsegagent-mcp` over stdio, with `segment_ct` and `segment_mr`.
- **Web:** NIfTI upload, natural language, durable progress, NiiVue 3D/slice overlays,
  label visibility, opacity and authenticated downloads.
- **A2A 1.0:** public `/.well-known/agent-card.json`, authenticated HTTP+JSON at `/a2a/v1`.
  See [the integration contract](docs/a2a.md).

All four adapters share `src/medsegagent/core.py`. `agent.py` makes one direct
OpenAI-compatible HTTP request and validates the proposed tool call. `service.py` adds
one durable SQLite task store; it does not implement another inference pipeline.

## Models and limits

| Tool | Official task | Input | Scope |
| --- | --- | --- | --- |
| `segment_ct` | TotalSegmentator `total` | CT | 117 anatomical structures |
| `segment_mr` | TotalSegmentator `total_mr` | MR | 50 anatomical structures |

TotalSegmentator is pinned to 2.18.0. Both base tasks use the Apache-2.0 fast models.
An explicit empty or invalid target list fails; omit targets in the local tool only when
requesting all structures. NIfTI cannot reliably establish modality: callers must declare
CT or MR. Unsupported lesions or ambiguous requests fail without substituting anatomy.

Web/A2A accept single-volume `.nii` and `.nii.gz`, at most 90 MiB compressed/file size
and 2 GiB expanded. Remote URLs, inline base64 and DICOM/ZIP uploads are rejected.
Local DICOM directory conversion has a separate strict boundary in [local tools](docs/simple-local-tools.md).

## Persistence and access

Each inference gets an atomically created run directory with durable state, a private
process log and an output manifest. An OS file lock serializes GPU work across processes,
including CLI and MCP. Timeout/cancel terminates the process group. Server jobs use
per-identity message idempotency, a bounded queue and durable task states. Restart preserves
completed tasks and explicitly fails interrupted jobs; it never silently redoes inference.

Web uses a 12-hour HttpOnly SameSite=Strict session; HTTPS sessions are Secure. A2A uses
separate Bearer identities. Uploads, tasks and files are owner scoped. There is no public
data directory, arbitrary filesystem route or remote URL fetch. Input/result files expire
under the 24-hour cleanup policy; task records remain as audit/idempotency tombstones.
Keep research inputs de-identified; user-written text is sent to the configured LLM provider.

## Verification and operations

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

[Deployment](docs/deployment.md) · [A2A](docs/a2a.md) · [Viewer decision](docs/viewer-decision.md)

**Research use only.** This implementation is not clinical validation, a medical device,
or evidence of patient benefit. Masks need independent review. Sample canary results
establish execution and file geometry only.

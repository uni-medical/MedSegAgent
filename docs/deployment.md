# Deployment

Run one service process per data directory. The app serves Web and A2A on port 8767.

## Install and configure

```bash
uv sync --frozen
cp .env.example .env
chmod 600 .env
```

Set these values in `.env`:

| Setting | Purpose |
| --- | --- |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | OpenAI-compatible HTTPS provider serving `deepseek-v4-flash` |
| `MEDSEGAGENT_PUBLIC_URL` | Public HTTPS origin, or `http://127.0.0.1:8767` locally |
| `MEDSEGAGENT_DATA_ROOT` | Runtime directory for uploads, results and SQLite; default `runtime` |
| `MEDSEGAGENT_DEVICE` | `mps` on Apple Silicon, `gpu` on NVIDIA Linux, or `cpu` |
| `TOTALSEG_HOME_DIR` | Optional TotalSegmentator configuration and weight cache location |
| `TMPDIR` | Optional existing private directory with space for NIfTI expansion and exports |

The current Agent model is `deepseek-v4-flash`. Keep credentials, image data and weights
in private storage. For a custom temporary directory, create it with mode 700 before
starting the app. Concurrent validation and export can use about 8 GiB of temporary space.

Guest access works immediately. For GitHub login, set `MEDSEGAGENT_GITHUB_CLIENT_ID`
and `MEDSEGAGENT_GITHUB_CLIENT_SECRET` together and register this OAuth callback:

```text
<MEDSEGAGENT_PUBLIC_URL>/api/auth/github/callback
```

GitHub login uses public profile information. Guest and GitHub accounts have separate
histories. Web sessions use HttpOnly, SameSite=Lax cookies, with Secure enabled on HTTPS.

## Prepare models

Prepare the models you need before inference. For CT and MR anatomy:

```bash
uv run --env-file .env python ops/prepare_weights.py --download --task total --task total_mr
uv run medsegagent weights
```

Omit `--task` to prepare all 33 public tasks and their supported quality modes:

```bash
uv run --env-file .env python ops/prepare_weights.py --download
```

The utility isolates downloads, verifies model files against the bundled manifests and
reuses installed models. CLI, MCP and Web can share one weight cache. The
[third-party notices](third-party-notices.md) describe software, model and data licenses.
To disable TotalSegmentator usage telemetry, set `send_usage_stats` to `false` in its
`config.json`.

## Start the service

```bash
uv run medsegagent doctor
uv run medsegagent serve --host 127.0.0.1 --port 8767
```

For Linux systemd, edit the paths in [the example unit](../ops/medsegagent.service)
to match your checkout, `.env` and uv executable, then install it:

```bash
mkdir -p ~/.config/systemd/user
install -m 644 ops/medsegagent.service ~/.config/systemd/user/medsegagent.service
systemctl --user daemon-reload
systemctl --user enable --now medsegagent.service
systemctl --user status medsegagent.service
```

Point an HTTPS reverse proxy or tunnel at `http://127.0.0.1:8767`. Enable streaming for
A2A responses and keep the Agent Card and `/a2a` paths accessible to API clients.

## Resources and retention

NVIDIA scheduling defaults to three concurrent inferences, one per eligible physical GPU.
All visible GPUs are considered. The defaults require 8192 MiB free memory and utilization
at most 20%; configure `MEDSEGAGENT_GPU_MIN_FREE_MIB` and
`MEDSEGAGENT_GPU_MAX_UTILIZATION` to adjust admission. Use `MEDSEGAGENT_GPU_IDS` to
restrict the pool by physical indices or full UUIDs. An explicit setting overrides
`CUDA_VISIBLE_DEVICES`; otherwise its existing restriction is honored.

Independent models can use different GPUs. MPS and CPU run one inference at a time.
Processes on one host share `MEDSEGAGENT_SCHEDULER_LOCK_DIR`, which defaults to
`~/.cache/medsegagent/scheduler`. The example service limits aggregate memory to 28 GiB,
CPU to eight cores and tasks/threads to 768; size these for your workload.

| Limit | Default |
| --- | --- |
| NIfTI upload / expanded volume | 500 MiB / 2 GiB |
| Raw upload / resumable chunk | 90 MiB / 8 MiB |
| Source storage per identity | 16 uploads or reservations, 2 GiB combined |
| Pending tasks | 8 globally, 4 per identity |
| Agent task timeout | 7200 seconds |
| Model requests / work-tool calls | 24 / 64 per task |
| Input and result retention | 24 hours |
| Incomplete upload session | 1 hour since its last committed chunk |

Source quotas cover uploaded files and pending reservations; budget separately for
models, results and temporary files. File cleanup runs every 15 minutes. Active work
keeps its files until it stops; expired files become inaccessible at their deadline.
Task history and message IDs persist for recovery. Apply the same access and retention
policy to runtime backups.

Anonymous A2A uses a shared namespace; resource IDs and file URLs grant access.
A valid Web session selects its account's namespace. See [A2A integration](a2a.md)
for upload, recovery and identity handling.

## Check and update

After starting or updating the service, open the viewer and run a segmentation through
your public origin. The integration check exercises upload, inference, streaming recovery,
task replay and mask downloads:

```bash
uv run python ops/acceptance.py --url https://your-service.example \
  --input /path/to/scan.nii.gz --output outputs/acceptance
```

Check active tasks before restarting. Completed tasks and message IDs survive a restart;
interrupted inference is marked failed, while queued work can resume within its deadline.
Retry failed work with a new message ID. Preserve `KillMode=mixed` in the example unit
so the application records shutdown before terminating its inference processes.

For source changes, run the [development checks](validation.md), update the host to the
tested commit, run `uv sync --frozen`, restart the service and repeat the integration check.

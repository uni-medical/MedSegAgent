# Deployment and recovery

Use one uv environment, one systemd user service and one SQLite data directory per host.
The example unit in `ops/medsegagent.service` binds loopback port 8767. Adjust its explicit
working directory for another machine. Do not share this data directory with another worker.

## Required settings

Keep `.env` mode 600 and outside Git. Copy only OPENAI_BASE_URL / OPENAI_API_KEY from the
approved provider configuration and set OPENAI_MODEL=deepseek-v4-flash. The model is also
fixed in code. MEDSEGAGENT_TOKENS_JSON maps each principal to a distinct random token of
at least 32 characters. Generate tokens locally; never place them in docs, URLs or logs.

On NVIDIA Linux set MEDSEGAGENT_DEVICE=gpu, CUDA_VISIBLE_DEVICES to an available GPU,
MEDSEGAGENT_PUBLIC_URL to the HTTPS origin, MEDSEGAGENT_DATA_ROOT to the private runtime
directory and TOTALSEG_HOME_DIR to the private weight/config directory. Disable anonymous
TotalSegmentator telemetry with send_usage_stats=false in its config.json. Limit BLAS/OpenMP
threads when sharing the server. macOS defaults to MPS.

```bash
uv sync --frozen --group dev
uv run medsegagent doctor
install -m 644 ops/medsegagent.service ~/.config/systemd/user/medsegagent.service
systemctl --user daemon-reload
systemctl --user enable --now medsegagent.service
systemctl --user status medsegagent.service
```

Only the service's public login shell, static viewer assets, Agent Card and health/readiness
are unauthenticated. Every upload, task and download requires identity. Web sessions are
HttpOnly/SameSite=Strict and Secure on HTTPS; A2A requires explicit Bearer. Cookies are
never accepted as A2A authentication. No query-string tokens, public data mounts or CORS.

The named Cloudflare tunnel should map only the new hostname to `http://127.0.0.1:8767`.
Preserve all other ingress rules and the final 404 fallback. A remotely managed tunnel can
receive its updated ingress configuration without restarting the connector. The application
handles Web identity protection and A2A Bearer separately; an Access login redirect must not
be placed in front of the public Agent Card or A2A paths. Verify each path through HTTPS.

Cloudflare Free currently allows 100 MB request bodies; this application chooses 90 MiB.
The server enforces the same limit with or without Content-Length. NIfTI expansion is bounded
to 2 GiB. Four concurrent uploads globally and one per identity are allowed, with 16 retained
uploads / 512 MiB per identity. Queue capacity is 8 globally / 2 per identity. One local
inference runs at a time, protected by a cross-process device lock. Overall server task timeout
is 7200 seconds including queue/routing/inference; LLM selection has a 90-second total limit.

## Real acceptance

`readyz` checks the process, LLM configuration presence and SQLite only. Run the real canary:

```bash
uv run python ops/acceptance.py --url https://medseg.huangziyan97.com \\
  --input /path/to/deidentified-research-ct.nii.gz --output outputs/acceptance/public
```

This exercises public Card, authentication refusal, upload, real model send and stream,
deliberate stream disconnect, GetTask, idempotency, identity isolation and actual mask
download/geometry/nonempty voxels. Run real browser upload/overlay/label/opacity/download,
refresh, failure and mobile checks as well. Do not substitute synthetic tests for GPU evidence.

Restart only after inspecting active tasks. Completed Tasks and original message IDs survive
restart. Interrupted routing/inference gets an explicit failed state; retry requires a new
messageId. Queued work can resume within its original time limit. Cancel terminates the whole
inference process group. systemd KillMode=control-group also contains abrupt shutdowns.

Input/result files are removed after the 24-hour retention window (cleanup runs every 15
minutes), except inputs currently referenced by active work. Unregistered upload directories
left by a crash are removed on the next exclusive service start. Task audit/idempotency records
persist, with expired files explicitly marked. Operators can inspect private run state/logs;
HTTP errors expose only bounded, redacted explanations. Backups must receive the same access
protection and retention policy as runtime data.

For releases: local checks and real canary → explicit branch commit/push → server fast-forward
to the exact commit → uv sync → service restart → public real-task acceptance. Record all
three Git SHAs and the active interpreter. Do not include `.env`, weight caches, sample data,
results, screenshots or logs in the public branch. No PR, main merge or force push is needed.

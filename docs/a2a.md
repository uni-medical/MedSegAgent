# MedSegAgent A2A integration

Research use only. CT `total` and MR `total_mr` produce anatomical masks through
the same local inference core as CLI, MCP and Web. Image bytes never enter the LLM
request. This document defines the interface; consult the acceptance report for
the current deployment and real-model test status.

| Item | Value |
| --- | --- |
| Public interface | `https://medseg.huangziyan97.com/a2a/v1` |
| Public Agent Card | `https://medseg.huangziyan97.com/.well-known/agent-card.json` |
| Protocol | A2A 1.0 HTTP+JSON; SDK pinned to PyPI release 1.1.2 |
| Required headers | `A2A-Version: 1.0`, `Authorization: Bearer <token>` |
| Content type | `application/a2a+json` preferred; `application/json` also accepted |
| Skill | `local-anatomical-segmentation` |
| File input | `.nii` or `.nii.gz`, at most 90 MiB, via authenticated upload |
| Natural language | 1–4000 characters; total A2A JSON envelope at most 512 KiB |
| File lifetime | 24 hours by default; current Card reports configured hours |

1. Upload the NIfTI with `POST /api/uploads`, an `X-Filename: image.nii.gz`
   header and the file as the raw request body. Send the same Bearer token.
   HTTP 201 returns JSON containing `id` and `expires_at`.
2. Pass that `id` as `upload_id` in a message. `modality` must be `CT` or `MR`.
   The server does not infer modality from a NIfTI header.
3. Recover the resulting Task with the same identity and download its Artifact
   URL with Bearer authentication.

`POST /a2a/v1/message:send`:

```json
{
  "message": {
    "messageId": "caller-request-001",
    "role": "ROLE_USER",
    "parts": [
      {"text": "Segment the liver and spleen", "mediaType": "text/plain"},
      {"data": {"upload_id": "<upload-response-id>", "modality": "CT"},
       "mediaType": "application/json"}
    ]
  },
  "configuration": {
    "returnImmediately": true,
    "acceptedOutputModes": ["text/plain", "application/json", "application/gzip"]
  }
}
```

The response is `{"task": {...}}`. Read `task.id`, then call
`GET /a2a/v1/tasks/{id}` until `status.state` is terminal. GetTask returns the
Task directly, without a `task` wrapper. A successful terminal Task has
`TASK_STATE_COMPLETED`; failure and cancellation use `TASK_STATE_FAILED` and
`TASK_STATE_CANCELED`. The structured Artifact contains allowlisted geometry,
targets, label and runtime fields when available. The mask Artifact uses a
same-origin authenticated URL and includes checksum and byte size metadata.

For streaming, send the same body to `POST /a2a/v1/message:stream`.
SSE `data:` records contain `task`, `statusUpdate`, or `artifactUpdate`.
These are task-level updates, not model tokens. After a broken connection, poll
GetTask; work continues independently. An active Task can also be subscribed to
with `GET /a2a/v1/tasks/{id}:subscribe`.

`POST /a2a/v1/tasks/{id}:cancel` with `{}` requests cancellation and returns the
terminal canceled Task after the local worker stops. Repeating cancellation of
a canceled Task succeeds. Cancellation cannot undo finished work.

`messageId` (1–128 characters) is idempotent within one caller identity.
Retrying the same semantic input reuses its Task; changing text, upload, modality
or context under that ID returns `INVALID_PARAMS`. Use a new ID for a new run.
Service restart preserves durable Tasks. Work interrupted after execution began
becomes failed; queued work may resume. File expiration leaves a historical Task
with `metadata.filesExpired: true` and no mask download.

Safe URL references are also accepted as a `url` Part, but only an exact
`https://medseg.huangziyan97.com/api/uploads/{id}/file` URL owned by the same
principal, with a separate `data: {"modality":"CT"}` Part. No remote URL is
fetched. Inline base64, DICOM upload, arbitrary paths, remote file fetching,
push notifications and task-list queries are unsupported.

Before admission, missing/invalid auth returns 401, missing/foreign Task or
upload returns 404, full capacity returns 429 with `Retry-After`, and malformed
or unsupported content returns 400. Inspect `error.details[].reason` for the
machine-readable cause. In the pinned current SDK, `TASK_NOT_CANCELABLE` is
HTTP 400; older GMAI SDK examples used 409. A failure after admission remains a
recoverable failed Task, without provider bodies, credentials or private paths.

Clients must support the v1 protobuf JSON shape (`ROLE_USER`, `mediaType`, direct
`url`/`data` Part fields) and authenticated Artifact downloads. Agent Card discovery
and `readyz` do not prove that model routing or segmentation has passed acceptance.

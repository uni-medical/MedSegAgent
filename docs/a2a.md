# A2A integration

Submit natural-language CT/MR segmentation requests and retrieve NIfTI masks through
TotalSeg Agent's A2A interface. It uses the same Agent and inference core as Web and CLI.

| Item | Value |
| --- | --- |
| Public interface | `https://medseg.huangziyan97.com/a2a/v1` |
| Public Agent Card | `https://medseg.huangziyan97.com/.well-known/agent-card.json` |
| Protocol | A2A 1.0 HTTP+JSON; SDK pinned to PyPI release 1.1.2 |
| Required headers | `A2A-Version: 1.0`; no login or Authorization header required |
| Content type | `application/a2a+json` preferred; `application/json` also accepted |
| Skill | `3d-medical-image-segmentation` |
| File input | `.nii` or `.nii.gz`, at most 500 MiB total / 2 GiB expanded, via `/a2a/uploads` or resumable sessions |
| Upload transport | Up to 90 MiB raw body; otherwise resumable 8 MiB chunks |
| Identity source quota | 2 GiB across completed source uploads and pending reservations; at most 16 combined |
| Natural language | 1–4000 characters; total A2A JSON envelope at most 512 KiB |
| File lifetime | 24 hours by default |

Without a Web cookie, A2A uses a shared public namespace. Retain the opaque upload,
task and context IDs returned by the service for recovery. Anyone holding an anonymous
resource's ID or file URL can access it; there is no public task-list endpoint. Use globally
unique `messageId` values, such as UUIDs, because anonymous callers share idempotency and quota
limits. A valid guest or GitHub Web cookie selects that identity's private namespace instead;
keep the same cookie for upload, submission, polling and downloads. Guest and GitHub histories
are separate.

`GET /a2a/config` lists the installed examples. `POST /a2a/examples/{id}` opens an example
as an upload in the current namespace; its returned `id` can be used directly as `upload_id`.
Preview and license files use `/a2a/examples/{id}/preview` and `/a2a/examples/{id}/license`.
These routes work without login, as do the upload and upload-session routes below.

The Card examples link directly to three NIfTI test images from Jakob Wasserthal and
the TotalSegmentator contributors at the fixed `v2.18.0` release:
`example_ct.nii.gz`, `example_mr_sm.nii.gz`, and `aorta_report/example_ct.nii.gz`
under [`tests/reference_files`](https://github.com/wasserth/TotalSegmentator/tree/v2.18.0/tests/reference_files).
These bundled files use the repository's [Apache-2.0 license](https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/LICENSE).
Download the linked image and upload it using the steps below before submitting the
segmentation request. The URL in an example is a download source, not an `upload_id`.

1. Upload the NIfTI. Files up to 90 MiB can use
   `POST /a2a/uploads`, an `X-Filename: image.nii.gz` header and the file as the raw
   request body. HTTP 201 returns JSON containing `id` and `expires_at`.
   For larger files, or resumable transfer, use the upload-session sequence below.
2. Pass that `id` as `upload_id` in a message. `modality` is optional; when supplied it
   must be `CT` or `MR`. A user declaration is used directly without modality detection.
   With no declaration the Agent can call `detect_modality` locally and use the returned
   evidence, available intensity statistics and metadata to choose CT/MR through
   `segment(modality=...)`. Detection itself does not bind the modality, and an uncertain
   observation does not block a later choice. The first valid segmentation choice binds
   the execution; the Agent asks the user only when it cannot reasonably determine CT/MR.
3. Recover the resulting Task by its ID and download the returned Artifact URL.
   If the task was created with a Web cookie, retain that same session for these requests.

For a resumable upload, send `POST /a2a/upload-sessions` with
`{"name":"image.nii.gz","size":123456789,"message_id":"upload-request-001"}`.
`size` is the exact compressed file size for `.nii.gz`, or file size for `.nii`.
HTTP 201 returns `id`, `offset`, `total_bytes` and `chunk_bytes` (8 MiB). This reserves the
full declared size against the identity's quota before receiving image bytes.

Send successive raw blocks with `PUT /a2a/upload-sessions/{id}` and
`Upload-Offset: <offset>`. Each block must be exactly `chunk_bytes`, except the last,
which must contain exactly the remaining bytes. After each HTTP 200, use the returned
`offset` as the next position. `GET /a2a/upload-sessions/{id}` recovers that durable
position after a lost response or service restart. Replaying a committed block at the
same boundary succeeds only if its size and SHA-256 match; different bytes return 409.

When all bytes are committed, call `POST /a2a/upload-sessions/{id}/complete`. Full 3D NIfTI
validation must succeed before HTTP 200 returns the upload metadata usable in an A2A
message. Completion is replayable while the session exists. Creating a session with the
same identity and upload `message_id` reuses it when name and size agree; a different
input returns 409. This upload key is separate from the A2A message's `messageId`.

Sessions expire one hour after creation or the last committed chunk; status queries do
not renew them. Each chunk and completion validation has a 180-second timeout. An incomplete
session can be discarded with `DELETE /a2a/upload-sessions/{id}`. That endpoint rejects
completed uploads with 409; use `DELETE /a2a/uploads/{id}` to remove a completed source,
provided no active task uses it. Uploading and completing a file never starts segmentation.

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
    "acceptedOutputModes": ["text/plain", "application/gzip"]
  }
}
```

The response is `{"task": {...}}`. Read `task.id`, then call
`GET /a2a/v1/tasks/{id}` until execution finishes or needs user input. GetTask returns
the Task directly, without a `task` wrapper. A successful terminal Task has
`TASK_STATE_COMPLETED`; failure and cancellation use `TASK_STATE_FAILED` and
`TASK_STATE_CANCELED`. `TASK_STATE_INPUT_REQUIRED` pauses the Task: read its
`status.message`, retain both IDs, and send a user reply as described below. It is
not a successful or failed terminal result, and polling alone cannot supply the answer. `task.metadata.segmentation` contains allowlisted geometry,
targets, labels (IDs/colors/volumes) and runtime fields
when available. New tasks also contain independent `outputs[]`, `regions[]`, `summary`
and `completion` with unresolved requirements. A failed task can retain verified partial
outputs without being marked completed. Each output preserves its producer and quality.
The viewer's merged mask uses background 0 and consecutive class IDs from 1;
a single class is 1/red. Historical records keep their original IDs. Each downloadable
class Artifact contains a separate binary NIfTI: background 0 and target 1/red, preserving
the merged mask's shape, affine, spacing, qform/sform and spatial units. An empty known
class produces an all-zero file. A filename such as `5_liver.nii.gz` retains label ID 5
from the merged mask; it does not mean the binary file contains value 5.
Volumes convert the stored spatial unit to millimeters. If that unit is unknown, the
header remains unchanged and `volume_measurement.unit_assumption` declares `assumed_mm`.

Class Artifacts use same-origin URLs and metadata `label_id`, `label_name`
and `mask_value: 1`. Multiple outputs also carry `output_id`; their filenames
include the output identifier so equal label IDs from different models cannot collide. They are generated and cached on first download, without another model
run; clients must not require a checksum or byte size in a lazy Artifact descriptor. The
merged `segmentation.nii.gz` serves the viewer, while class Artifacts provide the downloads.
Anonymous task files use `/a2a/tasks/{id}/files/{name}`; session-owned files retain their
private `/api/tasks/{id}/files/{name}` URLs. Class exports share
the parent task's expiration and identity checks, including when a request waits for an
export slot. At most two exports run concurrently per service.
There is no separate `result-json` Artifact or `/files/result.json` download. Clients
requesting only `application/json` as an output mode receive `CONTENT_TYPE_NOT_SUPPORTED`;
the A2A envelope and input data Parts still use JSON.

For streaming, send the same body to `POST /a2a/v1/message:stream`.
SSE `data:` records contain `task`, `statusUpdate`, or `artifactUpdate`.
These are task-level updates, not model tokens. Verified intermediate outputs can
become Artifact updates while work continues. A failed or canceled Task can retain
verified partial outputs; these do not establish successful completion. After a broken
connection, poll GetTask; work continues independently. An active Task can also be subscribed to
with `GET /a2a/v1/tasks/{id}:subscribe`.
Use GetTask as the authoritative snapshot of available Artifacts and measurements.
A newly opened subscription starts with a Task snapshot, so it can include outputs
that were published before the subscription began. Clients should identify Artifacts
by `artifactId` rather than treating every replayed descriptor as a new output.

`POST /a2a/v1/tasks/{id}:cancel` with `{}` requests cancellation and returns the
terminal canceled Task after the local worker stops. Repeating cancellation of
a canceled Task succeeds. Cancellation cannot undo finished work.

`messageId` (1–128 characters) is idempotent within the selected namespace.
Retrying the same semantic input reuses its Task; changing text, upload, modality
or context under that ID returns `INVALID_PARAMS`. Use a new ID for a new run or
a new user reply. A transport retry must keep the original body unchanged, including
the original presence or absence of `contextId`; do not add IDs from a later response.
Service restart preserves durable Tasks. Work interrupted after execution began
becomes failed; queued work may resume. File expiration leaves a historical Task
with `metadata.filesExpired: true` and no mask download.

## Context and user clarification

Omit `contextId` on the first request. For a later related request, send the returned
`contextId`; the service supplies a bounded history of earlier user requests and
assistant summaries to the original Agent: at most 40 user/assistant messages
(about 20 turns), with a combined 16,000-character limit, dropping the oldest entries
first. The same context can reuse its current image when no new `upload_id` is supplied. An explicit new upload selects a new image
for the request and clears the previous image’s history; its modality is not inherited.
Without a replacement image, only an explicitly recorded modality is inherited;
the Agent can determine modality again from the available evidence. A new request
without `taskId` creates a new Task in that context after its previous Task is terminal.
A context-only reply to an `INPUT_REQUIRED` Task continues the waiting Task; supplying
both IDs explicitly is recommended. A new turn while the context is actively running
is rejected; wait for its result before submitting another turn.

When a Task enters `TASK_STATE_INPUT_REQUIRED`, the Agent's question is in
`status.message.parts`. Reply through `message:send` or `message:stream` using the
**same `taskId` and `contextId`**, a fresh `messageId`, and the additional text:

```json
{
  "message": {
    "messageId": "new-clarification-reply-uuid",
    "taskId": "<paused-task-id>",
    "contextId": "<returned-context-id>",
    "role": "ROLE_USER",
    "parts": [{"text": "The image is MR; segment the liver.", "mediaType": "text/plain"}]
  },
  "configuration": {"returnImmediately": true}
}
```

The retained upload is reused; a `data` Part may supply an explicit `modality`.
A paused Task’s existing image cannot be replaced: start a new context for another
image. Continuation retains the external Task and Context IDs and re-enters the
original Agent with bounded history. It may repeat operations; it does not restore
a frozen model/tool process at the exact interrupted instruction.
Previously published outputs remain part of that Task when a continuation adds more
results. Their advertised URLs stay available for the Task's result-retention period.

A waiting Task survives service restart. Its `metadata.inputExpiresAtUnix` gives
the actual deadline for a reply, governed by the configured retention period
(default 24 hours). Work that was actively executing at restart follows the existing
interrupted-work recovery policy. Completed, failed and canceled Tasks cannot be
continued; create a new Task instead.
`historyLength` on SendMessage or GetTask controls the returned `Task.history`
window, independently of the bounded history used internally. A zero value omits
returned history. Prior image bytes, raw model/tool transcripts and private paths
are not included in the text history.

## Image-aware reference client

From the repository root, run:

```bash
uv run python ops/a2a_client.py \
  --url https://medseg.huangziyan97.com \
  --image /path/to/research-image.nii.gz \
  --prompt "Segment the liver" --modality MR
```

`--url` accepts the service root or its `/a2a/v1` interface. `--modality` is optional.
The client uploads a local file of at most 90 MiB as a raw body, sends its ID in
`data.upload_id`, consumes SSE, and recovers with GetTask after a disconnect.
If the first Task frame was lost, it retries the original message unchanged with
the same `messageId`. Larger images require the upload-session flow; use the
resulting ID with `--upload-id` instead of `--image`.

It prints the authoritative Task and Artifact list as JSON to stdout and recovery
IDs to stderr. It does not download Artifact URLs or run another inference to inspect
them. Exit codes are `0` for completed, `2` for failed/canceled/rejected, `3` for user
input or authentication required, and `1` for a client/admission error or timeout.
Client timeout does not cancel the remote Task. The default total deadline is two
hours; `--timeout` and `--poll-interval` can be set explicitly.

Poll a known Task without submitting another request:

```bash
uv run python ops/a2a_client.py --url https://medseg.huangziyan97.com \
  --task-id "<returned-task-id>"
```

Reply to a clarification without uploading the image again:

```bash
uv run python ops/a2a_client.py --url https://medseg.huangziyan97.com \
  --task-id "<paused-task-id>" --context-id "<returned-context-id>" \
  --prompt "The image is MR; segment the liver."
```

Use `--context-id` with a new prompt, omitting `--task-id`, for another Task in the
same context. `--message-id` and `--upload-id` allow an uncertain submission to be
retried with the original IDs and unchanged prompt/modality/context.

Safe URL references are also accepted as a `url` Part, but only an exact
`https://medseg.huangziyan97.com/a2a/uploads/{id}/file` URL in the same namespace,
optionally with a separate `data: {"modality":"CT"}` Part. The corresponding private
`/api/uploads/{id}/file` form is also accepted for session-owned uploads. No remote URL is
fetched. Inline base64, DICOM upload, arbitrary paths, remote file fetching,
push notifications and task-list queries are unsupported.

Before admission, a missing or differently owned Task or upload returns 404,
full capacity returns 429 with `Retry-After`, and malformed
or unsupported content returns 400. Inspect `error.details[].reason` for the
machine-readable cause. In the pinned current SDK, `TASK_NOT_CANCELABLE` is
HTTP 400. A failure after admission remains a
recoverable failed Task, without provider bodies, credentials or private paths.

Clients use the v1 protobuf JSON shape (`ROLE_USER`, `mediaType`, direct
`url`/`data` Part fields) and the namespace rules for Artifact downloads.

The internal Agent can use several segmentation, inspection and composition actions.
Only image-free, allowlisted tool feedback reaches the language provider, including
region measurements. Source images,
mask arrays, local paths and raw logs remain on the inference host.

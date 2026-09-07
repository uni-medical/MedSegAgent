# A2A protocol audit — 2026-09-07

This is a source audit and protocol-test record, not evidence of a deployed or
model-tested public service. Live acceptance belongs in the deployment report.

## Current primary sources

- Official protocol: [A2A v1.0.1](https://github.com/a2aproject/A2A/tree/v1.0.1),
  tag commit `3303592588e388e62e0f69f701af531d2f4e3991`.
- Canonical wire authority:
  [specification/a2a.proto](https://github.com/a2aproject/A2A/blob/v1.0.1/specification/a2a.proto).
  Retrieved SHA-256: `e195bf96ab630c69797851970203e1b2b6b19528f2e9803b7d904b91a5104016`.
- `git ls-remote --tags --refs` showed no newer protocol release. The current
  main proto differs from v1.0.1 only in a gRPC address explanatory comment;
  its retrieved SHA-256 was `945df6e34001b2bfd0fd62d9484b63094dfad9d78705e41e2873441c419ae2d1`.
- Latest observed official Python SDK tag:
  [v1.1.3](https://github.com/a2aproject/a2a-python/tree/v1.1.3),
  `4e71245bf2bf4b31f6429f12d97991f1f3d4b3f4`, released 2026-08-18 per the
  [official changelog](https://github.com/a2aproject/a2a-python/blob/v1.1.3/CHANGELOG.md).
  This tag is not yet published on PyPI: official
  [PyPI JSON](https://pypi.org/pypi/a2a-sdk/json) reports `1.1.2` as latest and has
  no `1.1.3` release. The installable pin is therefore **a2a-sdk==1.1.2**, not the
  newer Git tag. Tagged 1.1.2 and 1.1.3 error-mapping files compare byte-identical.
  SDK package version and A2A wire protocol version are different quantities.
- GMAI-Seeker reference checkout inspected at
  `22e92378d48274b97807b3ce26260a7ffd6bf49f`; its `docs/a2a.md`, untracked
  `docs/a2a-handoff.md`, `app.py`, `auth.py`, `agent_card.py`, `handler.py`,
  `executor.py`, `projection.py`, `store.py`, and installed SDK were read.
  It pins SDK `1.0.1`, so its older HTTP error codes must not be mistaken for
  the current official error mapping.

The GitHub releases API returned HTTP 403; official Git tags, tagged files and
changelogs supplied the release evidence instead.

## Actual GMAI protocol integration

GMAI uses the official SDK as a protocol adapter, independently of its agent:

```python
from a2a.server.routes import create_agent_card_routes, create_rest_routes
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater

handler = DefaultRequestHandlerV2(
    agent_executor=executor,
    task_store=store,
    agent_card=card,
    request_context_builder=context_builder,
)
routes = [
    *create_agent_card_routes(card),
    *create_rest_routes(
        request_handler=handler,
        context_builder=server_call_context_builder,
        enable_v0_3_compat=False,
        path_prefix="/a2a/v1",
    ),
]
```

GMAI subclasses this handler to atomically reserve `(principal, messageId)`,
validate semantic fingerprints and admission, and inject reserved Task/Context
IDs. Its SQLite store scopes reads to `ServerCallContext.user.user_name`.
`execute(context: RequestContext, event_queue: EventQueue)` emits a Task,
status updates and Artifact updates. The runtime owns the work independently of
HTTP/SSE lifetime. Its current implementation has cooperative cancellation;
historical notes saying that cancellation is unsupported are stale.

MedSegAgent already has a single durable job service shared by Web and inference.
Adding `DefaultRequestHandlerV2` plus a second TaskStore would duplicate its state
machine. The thinner integration in `src/medsegagent/a2a.py` therefore uses the
same official protobuf types and error mappings while projecting the existing
service's `submit/get/cancel` methods through Starlette. No agent framework or
separate inference implementation is introduced.

## Exact v1 HTTP+JSON contract

Base interface: `https://medseg.huangziyan97.com/a2a/v1`.
Canonical public Card: `/.well-known/agent-card.json`; `/agent-card.json` is a
compatibility alias. All operations require `A2A-Version: 1.0` and
`Authorization: Bearer <caller-token>`; browser cookies do not authenticate A2A.
`application/a2a+json` is preferred by v1.0.1; `application/json` remains accepted.

| Operation | Method and suffix | Response |
| --- | --- | --- |
| SendMessage | POST `/message:send` | `{"task": Task}` |
| SendStreamingMessage | POST `/message:stream` | SSE `StreamResponse` records |
| GetTask | GET `/tasks/{id}` | bare `Task`, without `task` wrapper |
| SubscribeToTask | GET `/tasks/{id}:subscribe` | SSE; reject already terminal Task |
| CancelTask | POST `/tasks/{id}:cancel` | bare terminal canceled `Task` |

`configuration.returnImmediately: true` returns after durable admission. With
false or omitted, SendMessage waits until terminal. An SSE disconnect does not
cancel the shared job. GetTask is the recovery authority and must be polled to
terminal state; one read is not proof that the run finished. No exact SSE event
replay cursor is promised. A stream starts with the current Task snapshot; new
completion emits Artifacts before its terminal status update.

The protocol uses protobuf JSON enum names: `ROLE_USER`, `ROLE_AGENT`,
`TASK_STATE_SUBMITTED`, `TASK_STATE_WORKING`, `TASK_STATE_COMPLETED`,
`TASK_STATE_FAILED`, `TASK_STATE_CANCELED`. The wire uses `messageId`, `contextId`,
`artifactId`, `mediaType`, `lastChunk`, `statusUpdate`, `artifactUpdate`.
There are no v0.3 `kind: "text"`, `file: {uri: ...}` or `final` fields.

Part has one content field: `text`, `data`, `url`, or `raw` (base64 bytes).
Supporting a protocol field does not require accepting every content type;
MedSegAgent deliberately rejects `raw` and arbitrary remote URLs.

Official constructors are protobuf messages, not Pydantic models:

```python
from a2a.types import AgentCard, Part, Artifact, Task, TaskStatus, TaskState
from google.protobuf.json_format import ParseDict, MessageToDict
from google.protobuf.struct_pb2 import Value

part = Part(data=ParseDict({"schema": "medsegagent_result_v1"}, Value()),
            media_type="application/json")
artifact = Artifact(artifact_id="result-json", parts=[part])
task = Task(id="server-id", context_id="context-id",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[artifact])
wire = MessageToDict(task)
```

Agent Card authentication uses the canonical v1 wrappers:

```json
{
  "supportedInterfaces": [{
    "url": "https://medseg.huangziyan97.com/a2a/v1",
    "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"
  }],
  "securitySchemes": {"bearerAuth": {
    "httpAuthSecurityScheme": {"scheme": "bearer", "bearerFormat": "opaque"}
  }},
  "securityRequirements": [{"schemes": {"bearerAuth": {"list": []}}}]
}
```

## File transfer and lifetime

The authenticated upload endpoint accepts `.nii`/`.nii.gz` with a 90 MiB limit.
The A2A JSON envelope is independently capped at 512 KiB; joined natural language
text must contain 1–4000 characters and fit within 64 KiB. The Agent Card reports the actual configured retention
hours; default input/output retention is 24 hours. Service cleanup determines
removal time. Persisted Task status may remain available after files expire;
GetTask then marks `metadata.filesExpired` and does not return downloadable masks.

Recommended request after uploading and receiving an opaque upload ID:

```json
{
  "message": {
    "messageId": "caller-generated-id-001",
    "role": "ROLE_USER",
    "parts": [
      {"text": "Segment the liver and spleen", "mediaType": "text/plain"},
      {"data": {"upload_id": "returned-upload-id", "modality": "CT"},
       "mediaType": "application/json"}
    ]
  },
  "configuration": {
    "returnImmediately": true,
    "acceptedOutputModes": ["text/plain", "application/json", "application/gzip"]
  }
}
```

Alternatively, replace the upload data Part with a URL Part whose URL is exactly
`https://medseg.huangziyan97.com/api/uploads/{id}/file`, and a separate data Part
containing `{"modality":"CT"}`. The adapter extracts the ID and asks the service
for its principal-scoped upload; it never performs an HTTP fetch. Reject credentials
in URLs, different scheme/host/port, query strings, fragments, traversal,
percent-encoded paths, multiple file references and absent/ambiguous modality.

Mask Artifact Parts contain an authenticated same-origin
`/api/tasks/{id}/files/segmentation.nii.gz` URL. Tokens are never embedded in the
URL, Card, JSON artifact, logs or user-visible error. The caller must carry the
same Bearer identity when downloading. A platform that only performs unauthenticated
artifact fetches needs an explicit integration change; it cannot use these URLs
as public files. Signed public URLs are not implicitly enabled.

## Error projection and current compatibility difference

Errors before admission are HTTP JSON errors; inference failures after admission
are persisted `FAILED` Tasks. Error messages never serialize provider bodies,
parse input, raw exceptions, full filesystem paths or subprocess logs.

Installed SDK 1.1.2 uses the following
[official error table](https://github.com/a2aproject/a2a-python/blob/v1.1.2/src/a2a/utils/errors.py):

| Reason | HTTP | Status |
| --- | --- | --- |
| `INVALID_PARAMS`, `INVALID_REQUEST` | 400 | `INVALID_ARGUMENT` |
| `CONTENT_TYPE_NOT_SUPPORTED` | 400 | `INVALID_ARGUMENT` |
| `VERSION_NOT_SUPPORTED` | 400 | `FAILED_PRECONDITION` |
| `TASK_NOT_FOUND` | 404 | `NOT_FOUND` |
| `TASK_NOT_CANCELABLE` | 400 | `FAILED_PRECONDITION` |
| `UNSUPPORTED_OPERATION`, `PUSH_NOTIFICATION_NOT_SUPPORTED` | 400 | `FAILED_PRECONDITION` |
| adapter `UNAUTHENTICATED` | 401 | `UNAUTHENTICATED` |
| adapter `CAPACITY_EXCEEDED` | 429 + Retry-After | `RESOURCE_EXHAUSTED` |

The SDK's older 1.0.1 returns 409 for `TASK_NOT_CANCELABLE` and 415 for
`CONTENT_TYPE_NOT_SUPPORTED`; GMAI's older handoff examples reflect that version.
Clients should interpret the typed ErrorInfo reason. Do not copy these outdated
status codes into claims about current spec conformance.

```json
{"error":{"code":404,"status":"NOT_FOUND","message":"Task not found",
 "details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo",
 "reason":"TASK_NOT_FOUND","domain":"a2a-protocol.org","metadata":{}}]}}
```

## Implementation and acceptance boundaries

The shared service is responsible for atomic principal/messageId claims, semantic
conflict detection, owner-scoped Tasks/uploads/files, one inference worker, queue
limits, cancellation, timeouts, startup recovery and retention. A2A admission
passes only principal, upload_id, text, modality, message_id and context_id.
Response-only settings do not alter the inference fingerprint. Do not call a
principal `tenant`: tenant is a distinct optional protocol routing dimension.

The initial protocol regression suite covers canonical SDK parsing, missing auth
and version, owner isolation, malicious URLs, traversal, raw/base64 rejection,
modality/type validation, redacted parsing failures, body limits, output modes,
SSE artifact ordering, polling recovery and cancel/terminal behavior. It uses a
fake shared service and is not inference, durability, Cloudflare or 端砚 acceptance.

Before claiming deployment, verify the public Card, 401 rejection, a real
function-calling plus segmentation SendMessage/stream, stream disconnect then
GetTask to terminal, authenticated mask download, service restart and retained
task/file ownership. Verify 端砚 actually sends A2A-Version and Bearer, supports v1
protobuf JSON and can authenticate Artifact URL downloads. The inspected GMAI
handoff is evidence of a compatible reference shape, not proof of a new MedSegAgent
integration inside 端砚.

Currently unsupported: remote-file fetching, inline base64, push notifications,
task-list queries, `INPUT_REQUIRED`/`AUTH_REQUIRED`, model-token streaming,
multi-process replicas and clinical-use claims.

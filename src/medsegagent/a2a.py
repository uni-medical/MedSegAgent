"""A2A 1.0 HTTP+JSON projection of the shared durable segmentation service.

The official SDK's protobuf types define the wire contract. This module owns
neither inference nor a second task store. File references are opaque identifiers
for uploads already owned by the authenticated principal; no URL is fetched.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Artifact,
    HTTPAuthSecurityScheme,
    Message,
    Part,
    Role,
    SecurityRequirement,
    SecurityScheme,
    SendMessageRequest,
    StringList,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.error_handlers import build_rest_error_payload
from a2a.utils.errors import (
    A2AError,
    ContentTypeNotSupportedError,
    InvalidParamsError,
    InvalidRequestError,
    PushNotificationNotSupportedError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
    VersionNotSupportedError,
)
from google.protobuf.json_format import MessageToDict, ParseDict, ParseError
from google.protobuf.struct_pb2 import Value
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import __version__

logger = logging.getLogger(__name__)
_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MODES = ("text/plain", "application/json", "application/gzip")
_INPUT_FILE_MODES = frozenset({"application/gzip", "application/nifti", "application/octet-stream"})
_TERMINAL = frozenset({"completed", "failed", "canceled"})
_STATES = {
    "queued": TaskState.TASK_STATE_SUBMITTED,
    "routing": TaskState.TASK_STATE_WORKING,
    "running": TaskState.TASK_STATE_WORKING,
    "completed": TaskState.TASK_STATE_COMPLETED,
    "failed": TaskState.TASK_STATE_FAILED,
    "canceled": TaskState.TASK_STATE_CANCELED,
}
_RESULT_FIELDS = frozenset(
    {
        "tool",
        "task",
        "modality",
        "targets",
        "device",
        "duration_seconds",
        "elapsed_seconds",
        "labels",
        "geometry",
        "voxel_counts",
        "warning",
        "research_use_only",
        "segmentation_shape",
        "segmentation_voxel_spacing",
        "nonzero_voxels",
        "runtime_seconds",
        "model",
    }
)
_PHASES = {
    "queued": "Segmentation queued.",
    "routing": "Selecting a local segmentation tool.",
    "running": "Local segmentation is running.",
    "completed": "Segmentation completed. Research use only.",
    "failed": "Segmentation failed. See the task error code.",
    "canceled": "Segmentation canceled.",
}


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _json(value: Any, *, status_code: int = 200, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        value,
        status_code=status_code,
        media_type="application/a2a+json",
        headers={"Cache-Control": "no-store", "A2A-Version": "1.0", **(headers or {})},
    )


def _error(reason: str, message: str, code: int, status: str, **headers: str) -> Response:
    return _json(
        {
            "error": {
                "code": code,
                "status": status,
                "message": message,
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": reason,
                        "domain": "medsegagent",
                        "metadata": {},
                    }
                ],
            }
        },
        status_code=code,
        headers=headers,
    )


def build_agent_card(
    public_url: str, *, max_upload_bytes: int = 90 * 1024 * 1024, retention_hours: int = 24
) -> AgentCard:
    """Advertise only the implemented upload-reference and task contract."""
    return AgentCard(
        name="MedSegAgent",
        version=__version__,
        description=(
            "Research-only local 3D anatomical segmentation using TotalSegmentator CT total "
            "and MR total_mr. Upload a .nii or .nii.gz through authenticated POST /api/uploads "
            "first, then send natural language plus its upload_id and explicit CT/MR modality. "
            "Only same-service, caller-owned upload references are accepted; remote URLs, "
            "inline base64, DICOM uploads and image data sent to the LLM are unsupported. "
            f"The upload limit is {max_upload_bytes} bytes; input and result files expire "
            f"after {retention_hours} hours under the service cleanup policy. Downloads "
            "require the same Bearer identity. Not a medical device or clinical claim."
        ),
        supported_interfaces=[
            AgentInterface(
                url=f"{public_url.rstrip('/')}/a2a/v1",
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True, push_notifications=False, extended_agent_card=False
        ),
        security_schemes={
            "bearerAuth": SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer",
                    bearer_format="opaque",
                    description="Per-principal API token. Also required for uploads and artifact files.",
                )
            )
        },
        security_requirements=[SecurityRequirement(schemes={"bearerAuth": StringList(list=[])})],
        default_input_modes=[
            "text/plain",
            "application/json",
            "application/gzip",
            "application/nifti",
        ],
        default_output_modes=list(_MODES),
        skills=[
            AgentSkill(
                id="local-anatomical-segmentation",
                name="Local CT/MR anatomical segmentation",
                description=(
                    "Provide one text request and one data Part with upload_id and modality CT/MR, "
                    "or a same-service /api/uploads/{id}/file URL and a modality data Part. "
                    "Supported targets are restricted by the CT total or MR total_mr tool. "
                    "No lesion detection, diagnosis or clinical performance is claimed."
                ),
                tags=["research", "3d", "nifti", "ct", "mr", "segmentation"],
                examples=["Segment the liver and spleen in this uploaded CT volume."],
            )
        ],
    )


def _output_modes(params: SendMessageRequest) -> tuple[str, ...]:
    requested = set(params.configuration.accepted_output_modes)
    modes = tuple(mode for mode in _MODES if not requested or mode in requested)
    if not modes:
        raise ContentTypeNotSupportedError(
            "Accept text/plain, application/json or application/gzip."
        )
    return modes


def _input(params: SendMessageRequest, public_url: str) -> tuple[str, str, str]:
    message = params.message
    if message.role != Role.ROLE_USER:
        raise InvalidParamsError("message.role must be ROLE_USER.")
    if not message.message_id or message.message_id != message.message_id.strip():
        raise InvalidParamsError("messageId must be nonempty with no surrounding whitespace.")
    if len(message.message_id) > 128 or len(message.context_id) > 128:
        raise InvalidParamsError("messageId and contextId must be at most 128 characters.")
    if message.context_id and message.context_id != message.context_id.strip():
        raise InvalidParamsError("contextId cannot have surrounding whitespace.")
    if message.task_id or message.reference_task_ids:
        raise InvalidParamsError(
            "Each request creates a Task; taskId and referenceTaskIds are unsupported."
        )
    if message.extensions or params.tenant or params.metadata or message.metadata:
        raise InvalidParamsError("Extensions, tenant and message/request metadata are unsupported.")
    config = params.configuration
    if config.HasField("task_push_notification_config"):
        raise PushNotificationNotSupportedError
    if config.HasField("history_length") and config.history_length < 0:
        raise InvalidParamsError("historyLength cannot be negative.")
    blocks: list[str] = []
    upload_ids: list[str] = []
    modalities: list[str] = []
    for part in message.parts:
        kind = part.WhichOneof("content")
        if part.metadata:
            raise InvalidParamsError(
                "Part metadata is unsupported; supply modality in a data Part."
            )
        if kind == "text":
            if part.media_type not in {"", "text/plain"}:
                raise ContentTypeNotSupportedError("Text Parts must use text/plain.")
            blocks.append(part.text)
        elif kind == "data":
            if part.media_type not in {"", "application/json"}:
                raise ContentTypeNotSupportedError("Data Parts must use application/json.")
            data = MessageToDict(part.data)
            if not isinstance(data, dict) or not data or set(data) - {"upload_id", "modality"}:
                raise InvalidParamsError("Data Part supports only upload_id and modality.")
            if "upload_id" in data:
                upload_ids.append(data["upload_id"])
            if "modality" in data:
                modalities.append(data["modality"])
        elif kind == "url":
            if part.media_type not in _INPUT_FILE_MODES:
                raise ContentTypeNotSupportedError(
                    "File references must declare a supported NIfTI media type."
                )
            supplied, base = urlsplit(part.url), urlsplit(public_url.rstrip("/"))
            prefix = base.path.rstrip("/") + "/api/uploads/"
            match = re.fullmatch(re.escape(prefix) + r"([A-Za-z0-9_-]{1,128})/file", supplied.path)
            if (
                supplied.scheme != base.scheme
                or supplied.netloc != base.netloc
                or supplied.username
                or supplied.password
                or supplied.query
                or supplied.fragment
                or not match
            ):
                raise InvalidParamsError("Only an exact same-service upload file URL is accepted.")
            upload_ids.append(match.group(1))
        else:
            raise ContentTypeNotSupportedError(
                "Inline raw/base64 and arbitrary files are unsupported."
            )
    text = "\n".join(blocks)
    if not text.strip() or len(text) > 4000 or len(text.encode("utf-8")) > 64 * 1024:
        raise InvalidParamsError(
            "Supply natural language text of 1–4000 characters (at most 64 KiB)."
        )
    if (
        len(upload_ids) != 1
        or not isinstance(upload_ids[0], str)
        or not _ID.fullmatch(upload_ids[0])
    ):
        raise InvalidParamsError("Supply exactly one valid upload_id or same-service upload URL.")
    if (
        len(modalities) != 1
        or not isinstance(modalities[0], str)
        or modalities[0] not in {"CT", "MR"}
    ):
        raise InvalidParamsError("Supply exactly one explicit modality: CT or MR.")
    return upload_ids[0], text.strip(), modalities[0]


def _data(value: dict) -> Value:
    return ParseDict(value, Value())


def project_task(row: dict, public_url: str, modes: tuple[str, ...] = _MODES) -> Task:
    """Project public task metadata and authenticated files, never local paths/logs."""
    phase = row["status"]
    task = Task(
        id=row["id"],
        context_id=row["context_id"],
        status=TaskStatus(
            state=_STATES[phase],
            message=Message(
                message_id=f"{row['id']}-status",
                task_id=row["id"],
                context_id=row["context_id"],
                role=Role.ROLE_AGENT,
                parts=[Part(text=_PHASES[phase], media_type="text/plain")],
            ),
        ),
    )
    updated = row.get("updated_at")
    if updated:
        try:
            if isinstance(updated, (int, float)):
                task.status.timestamp.FromMilliseconds(int(updated * 1000))
            else:
                task.status.timestamp.FromJsonString(str(updated))
        except (ValueError, TypeError):
            pass
    metadata = {"phase": phase, "researchUseOnly": True}
    if row.get("error"):
        metadata["errorCode"] = str(row["error"].get("code", "EXECUTION_FAILED"))
    if row.get("files_expired"):
        metadata["filesExpired"] = True
    if row.get("expires_at"):
        metadata["filesExpireAtUnix"] = row["expires_at"]
    task.metadata.update(metadata)
    if phase != "completed":
        return task
    result = row.get("result") or {}
    public_result = {key: value for key, value in result.items() if key in _RESULT_FIELDS}
    public_result.update(
        {"schema": "medsegagent_result_v1", "taskId": row["id"], "researchUseOnly": True}
    )
    if "text/plain" in modes:
        task.artifacts.append(
            Artifact(
                artifact_id="summary",
                name="Segmentation summary",
                parts=[
                    Part(
                        text=(
                            "Local segmentation completed, but its result files have expired. "
                            "Submit a new upload and message to rerun. Research use only."
                            if row.get("files_expired")
                            else "Local segmentation completed. Research use only. Download the mask "
                            "using the same Bearer identity."
                        ),
                        media_type="text/plain",
                    )
                ],
            )
        )
    if "application/json" in modes:
        task.artifacts.append(
            Artifact(
                artifact_id="result-json",
                name="Structured segmentation result",
                parts=[Part(data=_data(public_result), media_type="application/json")],
            )
        )
    files = row.get("files", result.get("files", []))
    for index, file in enumerate(files):
        if not isinstance(file, dict) or file.get("media_type") not in modes:
            continue
        url = file.get("url", "")
        base = public_url.rstrip("/")
        if url.startswith("/") and not url.startswith("//"):
            url = base + url
        parsed, origin = urlsplit(url), urlsplit(base)
        expected = origin.path.rstrip("/") + f"/api/tasks/{row['id']}/files/"
        name = str(file.get("name", "segmentation.nii.gz"))
        if (
            parsed.scheme != origin.scheme
            or parsed.netloc != origin.netloc
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith(expected)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", parsed.path[len(expected) :]) is None
        ):
            continue
        task.artifacts.append(
            Artifact(
                artifact_id=f"file-{index}",
                name=name,
                description="NIfTI output; the same Bearer identity is required for download.",
                parts=[Part(url=url, filename=name, media_type=file["media_type"])],
                metadata={key: file[key] for key in ("sha256", "size_bytes") if key in file},
            )
        )
    return task


def routes(service: Any, authenticate: Callable[[Request], Any]) -> list[Route]:
    """Mount the public Card and identity-protected A2A binding on the Web app."""
    public_url = service.public_url.rstrip("/")

    async def owner(request: Request) -> str:
        principal = await _resolve(authenticate(request))
        if not isinstance(principal, str) or not principal:
            raise HTTPException(401, "Bearer authentication is required.")
        version = request.headers.get("a2a-version", "")
        if re.fullmatch(r"1\.0(?:\.\d+)?", version) is None:
            raise VersionNotSupportedError("Supply A2A-Version: 1.0.")
        return principal

    def boundary(endpoint: Callable) -> Callable:
        async def wrapped(request: Request) -> Response:
            try:
                return await endpoint(request)
            except A2AError as exc:
                payload = build_rest_error_payload(exc)
                return _json(payload, status_code=payload["error"]["code"])
            except (ParseError, json.JSONDecodeError, UnicodeDecodeError):
                return _error(
                    "INVALID_REQUEST", "Invalid A2A 1.0 request envelope.", 400, "INVALID_ARGUMENT"
                )
            except (KeyError, FileNotFoundError):
                return _error("TASK_NOT_FOUND", "Task or upload not found.", 404, "NOT_FOUND")
            except HTTPException as exc:
                if exc.status_code in {401, 403}:
                    return _error(
                        "UNAUTHENTICATED",
                        "Valid Bearer authentication is required.",
                        401,
                        "UNAUTHENTICATED",
                        **{"WWW-Authenticate": "Bearer"},
                    )
                return _error(
                    "INVALID_PARAMS",
                    "Request cannot be processed.",
                    exc.status_code,
                    "INVALID_ARGUMENT",
                )
            except Exception as exc:  # noqa: BLE001 — never expose unknown service/provider exceptions.
                code = getattr(exc, "status_code", None)
                if code in {401, 403}:
                    return _error(
                        "UNAUTHENTICATED",
                        "Valid Bearer authentication is required.",
                        401,
                        "UNAUTHENTICATED",
                        **{"WWW-Authenticate": "Bearer"},
                    )
                if code == 404:
                    return _error("TASK_NOT_FOUND", "Task or upload not found.", 404, "NOT_FOUND")
                if code == 409:
                    if getattr(exc, "code", None) == "TASK_NOT_CANCELABLE":
                        payload = build_rest_error_payload(TaskNotCancelableError())
                        return _json(payload, status_code=payload["error"]["code"])
                    return _error(
                        "INVALID_PARAMS", "Conflicting request semantics.", 400, "INVALID_ARGUMENT"
                    )
                if code == 429:
                    return _error(
                        "CAPACITY_EXCEEDED",
                        "Task capacity is full; retry later.",
                        429,
                        "RESOURCE_EXHAUSTED",
                        **{"Retry-After": "5"},
                    )
                if code in {400, 413, 415}:
                    return _error(
                        "INVALID_PARAMS",
                        "Request violates the service input limits.",
                        code,
                        "INVALID_ARGUMENT",
                    )
                if isinstance(exc, ValueError):
                    return _error(
                        "INVALID_PARAMS", "Invalid segmentation request.", 400, "INVALID_ARGUMENT"
                    )
                logger.error("A2A request failed (%s)", type(exc).__name__)
                return _error(
                    "INTERNAL_ERROR", "The service could not process this request.", 500, "INTERNAL"
                )

        return wrapped

    async def read_params(request: Request) -> SendMessageRequest:
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type not in {"application/json", "application/a2a+json"}:
            raise ContentTypeNotSupportedError("Use application/a2a+json or application/json.")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > 512 * 1024:
                raise HTTPException(413, "A2A request exceeds 512 KiB.")
            body.extend(chunk)
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise InvalidRequestError("Request body must be an object.")
        params = ParseDict(payload, SendMessageRequest())
        if not params.HasField("message"):
            raise InvalidParamsError("message is required.")
        return params

    async def lookup(principal: str, task_id: str) -> dict:
        if not _ID.fullmatch(task_id):
            raise TaskNotFoundError
        row = await _resolve(service.get(principal, task_id))
        if row is None:
            raise TaskNotFoundError
        return row

    async def submit(request: Request) -> tuple[str, dict, tuple[str, ...]]:
        principal = await owner(request)
        params = await read_params(request)
        upload_id, text, modality = _input(params, public_url)
        modes = _output_modes(params)
        row = await _resolve(
            service.submit(
                principal=principal,
                upload_id=upload_id,
                text=text,
                modality=modality,
                message_id=params.message.message_id,
                context_id=params.message.context_id or None,
            )
        )
        request.state.a2a_return_immediately = params.configuration.return_immediately
        return principal, row, modes

    async def send(request: Request) -> Response:
        principal, row, modes = await submit(request)
        while not request.state.a2a_return_immediately and row["status"] not in _TERMINAL:
            if await request.is_disconnected():
                return Response(status_code=499)
            await asyncio.sleep(0.25)
            row = await lookup(principal, row["id"])
        return _json({"task": MessageToDict(project_task(row, public_url, modes))})

    async def events(principal: str, row: dict, modes: tuple[str, ...]):
        task = MessageToDict(project_task(row, public_url, modes))
        yield "data: " + json.dumps({"task": task}, separators=(",", ":")) + "\n\n"
        last_status = task["status"]
        heartbeat = time.monotonic()
        try:
            while row["status"] not in _TERMINAL:
                await asyncio.sleep(0.25)
                row = await lookup(principal, row["id"])
                task = MessageToDict(project_task(row, public_url, modes))
                if row["status"] == "completed":
                    for artifact in task.get("artifacts", []):
                        event = {
                            "artifactUpdate": {
                                "taskId": task["id"],
                                "contextId": task["contextId"],
                                "artifact": artifact,
                                "append": False,
                                "lastChunk": True,
                            }
                        }
                        yield "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"
                if task["status"] != last_status:
                    event = {
                        "statusUpdate": {
                            "taskId": task["id"],
                            "contextId": task["contextId"],
                            "status": task["status"],
                            "metadata": task.get("metadata", {}),
                        }
                    }
                    yield "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"
                    last_status = task["status"]
                if time.monotonic() - heartbeat >= 15:
                    yield ": keep-alive\n\n"
                    heartbeat = time.monotonic()
        except asyncio.CancelledError:
            raise  # Disconnect only stops this subscription, never the shared job.
        except Exception:  # noqa: BLE001 — an already-open SSE stream needs a redacted error event.
            yield 'event: error\ndata: {"error":{"code":500,"status":"INTERNAL","message":"Stream interrupted; recover with GetTask."}}\n\n'

    async def stream(request: Request) -> Response:
        principal, row, modes = await submit(request)
        return StreamingResponse(
            events(principal, row, modes),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def get(request: Request) -> Response:
        principal = await owner(request)
        if set(request.query_params) - {"historyLength"}:
            raise InvalidParamsError("Only historyLength is supported by GetTask.")
        if "historyLength" in request.query_params:
            value = request.query_params["historyLength"]
            if not value.isdecimal() or int(value) > 2**31 - 1:
                raise InvalidParamsError("historyLength must be a nonnegative int32.")
        row = await lookup(principal, request.path_params["id"])
        return _json(MessageToDict(project_task(row, public_url)))

    async def subscribe(request: Request) -> Response:
        principal = await owner(request)
        row = await lookup(principal, request.path_params["id"])
        if row["status"] in _TERMINAL:
            raise UnsupportedOperationError("Cannot subscribe to a terminal Task; use GetTask.")
        return StreamingResponse(
            events(principal, row, _MODES),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def cancel(request: Request) -> Response:
        principal = await owner(request)
        row = await lookup(principal, request.path_params["id"])
        if row["status"] == "canceled":
            return _json(MessageToDict(project_task(row, public_url)))
        if row["status"] in _TERMINAL:
            raise TaskNotCancelableError
        row = await _resolve(service.cancel(principal, row["id"]))
        while row["status"] not in _TERMINAL:
            await asyncio.sleep(0.25)
            row = await lookup(principal, row["id"])
        if row["status"] != "canceled":
            raise TaskNotCancelableError("Task completed before cancellation.")
        return _json(MessageToDict(project_task(row, public_url)))

    async def unsupported(request: Request) -> Response:
        await owner(request)
        raise UnsupportedOperationError

    async def card(_request: Request) -> Response:
        return _json(
            MessageToDict(
                build_agent_card(
                    public_url,
                    max_upload_bytes=getattr(service, "max_upload_bytes", 90 * 1024 * 1024),
                    retention_hours=getattr(service, "retention_seconds", 86400) // 3600,
                )
            )
        )

    return [
        Route("/.well-known/agent-card.json", card, methods=["GET"]),
        Route("/agent-card.json", card, methods=["GET"]),
        Route("/a2a/v1/message:send", boundary(send), methods=["POST"]),
        Route("/a2a/v1/message:stream", boundary(stream), methods=["POST"]),
        Route("/a2a/v1/tasks/{id}:subscribe", boundary(subscribe), methods=["GET"]),
        Route("/a2a/v1/tasks/{id}:cancel", boundary(cancel), methods=["POST"]),
        Route("/a2a/v1/tasks/{id}", boundary(get), methods=["GET"]),
        Route("/a2a/v1/tasks", boundary(unsupported), methods=["GET"]),
        Route("/a2a/v1/extendedAgentCard", boundary(unsupported), methods=["GET"]),
    ]

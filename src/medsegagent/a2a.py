"""A2A 1.0 HTTP+JSON projection of the shared durable segmentation service.

The official SDK's protobuf types define the wire contract. This module owns
neither inference nor a second task store. File references are opaque identifiers
for uploads in the caller's storage namespace; no URL is fetched. Anonymous
requests use a dedicated public namespace; existing private records stay private.
"""

from __future__ import annotations

import asyncio
import hashlib
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
    Message,
    Part,
    Role,
    SendMessageRequest,
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
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import __version__
from .agent import unsupported_request
from .result_metadata import result_metadata
from .service import PUBLIC_A2A_PRINCIPAL

logger = logging.getLogger(__name__)
_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MODES = ("text/plain", "application/gzip")
_INPUT_FILE_MODES = frozenset({"application/gzip", "application/nifti", "application/octet-stream"})
_TERMINAL = frozenset({"completed", "failed", "canceled"})
_TURN_FINISHED = _TERMINAL | {"input_required"}
_STATES = {
    "queued": TaskState.TASK_STATE_SUBMITTED,
    "routing": TaskState.TASK_STATE_WORKING,
    "running": TaskState.TASK_STATE_WORKING,
    "input_required": TaskState.TASK_STATE_INPUT_REQUIRED,
    "completed": TaskState.TASK_STATE_COMPLETED,
    "failed": TaskState.TASK_STATE_FAILED,
    "canceled": TaskState.TASK_STATE_CANCELED,
}
_PHASES = {
    "queued": "Segmentation queued.",
    "routing": "Checking the requested segmentation.",
    "running": "Local segmentation is running.",
    "input_required": "Please supply the requested information to continue this Task.",
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


def build_agent_card(public_url: str) -> AgentCard:
    """Describe the segmentation capability and its A2A interface."""
    base = public_url.rstrip("/")
    return AgentCard(
        name="MedSegAgent",
        version=__version__,
        description=(
            "Segment a variety of anatomical structures and common lesions in 3D medical images, "
            "including CT and MR. Specify the segmentation target; modality may be supplied or detected. Use "
            "natural language. A2A requires no login or Authorization header. Anonymous uploads "
            "and tasks are accessible to anyone holding their opaque IDs; keep these IDs private. "
            "There is no public task list. An optional GitHub or guest browser session retains "
            "that session's private storage namespace."
        ),
        supported_interfaces=[
            AgentInterface(
                url=f"{base}/a2a/v1",
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True, push_notifications=False, extended_agent_card=False
        ),
        default_input_modes=[
            "text/plain",
            "application/json",
            "application/gzip",
            "application/nifti",
        ],
        default_output_modes=list(_MODES),
        skills=[
            AgentSkill(
                id="3d-medical-image-segmentation",
                name="3D medical image segmentation",
                description=(
                    "Segment anatomical structures and common lesions in a 3D medical image. "
                    "Describe what to segment; imaging modality may be supplied or detected locally, "
                    "in natural language.\n\n"
                    "Quick start for a valid 3D .nii or .nii.gz file up to 90 MiB:\n"
                    "1. Download the image locally, then POST its raw file bytes to "
                    f"{base}/a2a/uploads with Content-Type: application/octet-stream and "
                    "X-Filename set to the actual filename, for example image.nii.gz. "
                    "Do not use multipart. HTTP 201 returns JSON; its id is the upload ID. "
                    "Example image URLs are download sources only. A2A messages accept "
                    "same-service upload references, not external URLs or inline raw/base64 files.\n"
                    f"2. POST {base}/a2a/v1/message:send with A2A-Version: 1.0 and "
                    "Content-Type: application/a2a+json. Example JSON for MR liver segmentation "
                    "(replace the placeholders and adapt text/modality to your image):\n"
                    '{"message":{"messageId":"<new UUID>","role":"ROLE_USER","parts":'
                    '[{"text":"Segment the liver"},{"data":{"upload_id":"<upload id>",'
                    '"modality":"MR"}}]},"configuration":{"returnImmediately":true,'
                    '"acceptedOutputModes":["text/plain","application/gzip"]}}\n'
                    "modality is optional for detection; explicit values must be CT or MR. "
                    "Use a fresh UUID messageId for each new request. Omit contextId for the first "
                    "request; on transport retries keep the original message body and messageId "
                    "unchanged, without adding the returned contextId.\n"
                    "3. Read task.id from the send response, then GET "
                    f"{base}/a2a/v1/tasks/{{id}} with A2A-Version: 1.0. "
                    "GetTask returns the Task directly, without a task wrapper. Poll every 2 seconds "
                    "until status.state is TASK_STATE_COMPLETED, TASK_STATE_FAILED or "
                    "TASK_STATE_CANCELED, or pause polling on TASK_STATE_INPUT_REQUIRED. "
                    "Only COMPLETED means success; other states may retain partial outputs. "
                    "Download file URLs from artifacts[].parts[].url.\n"
                    "4. A text-only request is accepted. If an image or clarification is needed, "
                    "the Task returns TASK_STATE_INPUT_REQUIRED with the question in "
                    "status.message. Supply the answer with a new messageId and the original "
                    "taskId and contextId. If an image was requested, upload it as in step 1 and "
                    "include data.upload_id in that continuation. Waiting Tasks remain resumable "
                    "for 24 hours by default; metadata.inputExpiresAtUnix is the actual deadline. "
                    "A new request with the returned contextId retains recent conversation "
                    "history and can reuse its image. After its previous Task is terminal, "
                    "omit taskId to create a new Task; a context-only reply to a waiting Task "
                    "continues that Task. Wait for running work before sending another turn. "
                    "Task responses include up to 20 of that Task's messages by default; "
                    "configuration.historyLength or GetTask ?historyLength controls the count "
                    "(0 omits history).\n"
                    "For SSE, POST the same body to /a2a/v1/message:stream. It emits an initial "
                    "Task followed by statusUpdate and artifactUpdate records, then closes on "
                    "completion, failure, cancellation or INPUT_REQUIRED. Stream closure alone "
                    "does not establish success: always GET the Task afterward to recover its "
                    "current status and complete published artifacts. A disconnected stream "
                    "does not cancel the Task."
                ),
                tags=["3d", "nifti", "ct", "mr", "anatomy", "lesions", "segmentation"],
                examples=[
                    (
                        "Download and upload this CT image, then segment the liver and spleen: "
                        "https://raw.githubusercontent.com/wasserth/TotalSegmentator/v2.18.0/"
                        "tests/reference_files/example_ct.nii.gz"
                    ),
                    (
                        "Download and upload this MR image, then segment the liver: "
                        "https://raw.githubusercontent.com/wasserth/TotalSegmentator/v2.18.0/"
                        "tests/reference_files/example_mr_sm.nii.gz"
                    ),
                    (
                        "Download and upload this CT image, then segment the aorta: "
                        "https://raw.githubusercontent.com/wasserth/TotalSegmentator/v2.18.0/"
                        "tests/reference_files/aorta_report/example_ct.nii.gz"
                    ),
                ],
            ),
        ],
    )


def _output_modes(params: SendMessageRequest) -> tuple[str, ...]:
    requested = set(params.configuration.accepted_output_modes)
    modes = tuple(mode for mode in _MODES if not requested or mode in requested)
    if not modes:
        raise ContentTypeNotSupportedError(
            "Accept text/plain or application/gzip. Segmentation metadata is included in the Task."
        )
    return modes


def _input(params: SendMessageRequest, public_url: str) -> tuple[str | None, str, str | None]:
    message = params.message
    if message.role != Role.ROLE_USER:
        raise InvalidParamsError("message.role must be ROLE_USER.")
    if not message.message_id or message.message_id != message.message_id.strip():
        raise InvalidParamsError("messageId must be nonempty with no surrounding whitespace.")
    if len(message.message_id) > 128 or len(message.context_id) > 128:
        raise InvalidParamsError("messageId and contextId must be at most 128 characters.")
    if message.context_id and message.context_id != message.context_id.strip():
        raise InvalidParamsError("contextId cannot have surrounding whitespace.")
    if message.task_id and not _ID.fullmatch(message.task_id):
        raise InvalidParamsError("taskId must identify an existing Task.")
    if message.reference_task_ids:
        raise InvalidParamsError("referenceTaskIds are unsupported; use taskId to continue a Task.")
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
            prefix = re.escape(base.path.rstrip("/")) + r"/(?:api|a2a)/uploads/"
            match = re.fullmatch(prefix + r"([A-Za-z0-9_-]{1,128})/file", supplied.path)
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
    if (
        len(upload_ids) > 1
        or upload_ids
        and (not isinstance(upload_ids[0], str) or not _ID.fullmatch(upload_ids[0]))
    ):
        raise InvalidParamsError("Supply at most one valid upload_id or same-service upload URL.")
    text = "\n".join(blocks)
    if len(text) > 4000 or len(text.encode("utf-8")) > 64 * 1024:
        raise InvalidParamsError("Supply text of at most 4000 characters (at most 64 KiB).")
    if not text.strip():
        if not upload_ids:
            raise InvalidParamsError("Supply natural language text or an uploaded image.")
        text = (
            "I have supplied the image. Continue the segmentation request, "
            "or ask me which structures to segment."
        )
    if len(modalities) > 1 or (
        modalities and (not isinstance(modalities[0], str) or modalities[0] not in {"CT", "MR"})
    ):
        raise InvalidParamsError(
            "Supply at most one explicit modality: CT or MR, or omit it for detection."
        )
    return (
        upload_ids[0] if upload_ids else None,
        text.strip(),
        modalities[0] if modalities else None,
    )


def project_task(
    row: dict,
    public_url: str,
    modes: tuple[str, ...] = _MODES,
    history_length: int | None = None,
) -> Task:
    """Project public task metadata and authenticated files, never local paths/logs."""
    phase = row["status"]
    status_text = _PHASES[phase]
    error_code = (row.get("error") or {}).get("code")
    if phase == "failed":
        status_text = {
            "UNSUPPORTED_REQUEST": str(unsupported_request()),
            "MODALITY_REQUIRED": "请在请求中说明影像是 CT 还是 MR；本次未开始分割。",
            "INPUT_REQUIRED": (row.get("error") or {}).get(
                "message", "Please clarify the requested outputs."
            ),
            "MODALITY_CONFLICT": "影像模态与请求不一致，请确认 CT 或 MR 后重新提交。",
        }.get(error_code, status_text)
    elif phase == "input_required":
        status_text = (row.get("error") or {}).get("message") or status_text
    status_message_id = f"{row['id']}-status-{phase}"
    if phase == "input_required":
        status_message_id += "-" + hashlib.sha256(status_text.encode()).hexdigest()[:16]
    task = Task(
        id=row["id"],
        context_id=row["context_id"],
        status=TaskStatus(
            state=_STATES[phase],
            message=Message(
                message_id=status_message_id,
                task_id=row["id"],
                context_id=row["context_id"],
                role=Role.ROLE_AGENT,
                parts=[Part(text=status_text, media_type="text/plain")],
            ),
        ),
    )
    history_limit = 20 if history_length is None else max(0, history_length)
    if history_limit:
        for message in (row.get("a2a_history") or [])[-history_limit:]:
            task.history.append(ParseDict(message, Message()))
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
    if row.get("publication_error") == "ARTIFACT_PUBLICATION_FAILED":
        metadata["publicationErrorCode"] = "ARTIFACT_PUBLICATION_FAILED"
    if row.get("files_expired"):
        metadata["filesExpired"] = True
    if row.get("expires_at"):
        metadata["filesExpireAtUnix"] = row["expires_at"]
    if row.get("input_expires_at"):
        metadata["inputExpiresAtUnix"] = row["input_expires_at"]
    task.metadata.update(metadata)
    result = row.get("result") or {}
    public_result = result_metadata(result)
    if result:
        task.metadata.update({"segmentation": public_result})
    if phase != "completed" and not result.get("outputs"):
        return task
    if "text/plain" in modes:
        task.artifacts.append(
            Artifact(
                artifact_id="summary",
                name="Segmentation summary",
                parts=[
                    Part(
                        text=(
                            "The request is incomplete. Verified partial results are available; "
                            "see metadata.segmentation.completion for unresolved requirements."
                            if phase != "completed"
                            else "Local segmentation completed, but its result files have expired. "
                            "Submit a new upload and message to rerun. Research use only."
                            if row.get("files_expired")
                            else "Local segmentation completed. No requested target was detected; "
                            "this does not rule out disease. Research use only. Download the mask "
                            "using its access URL; session-owned files require the same browser session."
                            if result.get("detection_status") == "no_target_detected"
                            else "Local segmentation completed. Research use only. Download the mask "
                            "using its access URL; session-owned files require the same browser session."
                        ),
                        media_type="text/plain",
                    )
                ],
            )
        )
    files = row.get("files", result.get("files", []))
    class_outputs = {
        file.get("output_id")
        for file in files
        if isinstance(file, dict) and file.get("kind") == "label"
    }
    for file in files:
        if not isinstance(file, dict) or file.get("media_type") not in modes:
            continue
        if file.get("output_id") in class_outputs and (
            file.get("kind") == "overlay" or file.get("name") == "segmentation.nii.gz"
        ):
            continue
        url = file.get("url", "")
        base = public_url.rstrip("/")
        if url.startswith("/") and not url.startswith("//"):
            url = base + url
        parsed, origin = urlsplit(url), urlsplit(base)
        root_path = origin.path.rstrip("/")
        expected = next(
            (
                prefix
                for prefix in (
                    root_path + f"/api/tasks/{row['id']}/files/",
                    root_path + f"/a2a/tasks/{row['id']}/files/",
                )
                if parsed.path.startswith(prefix)
            ),
            None,
        )
        name = str(file.get("name", "segmentation.nii.gz"))
        if (
            parsed.scheme != origin.scheme
            or parsed.netloc != origin.netloc
            or parsed.query
            or parsed.fragment
            or expected is None
            or re.fullmatch(r"[A-Za-z0-9_.-]+", parsed.path[len(expected) :]) is None
        ):
            continue
        identity = (
            [file.get(key) for key in ("output_id", "kind", "label_id", "label_name")]
            if any(file.get(key) is not None for key in ("output_id", "label_id", "label_name"))
            else [name, url]
        )
        artifact_id = (
            "file-"
            + hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()[:20]
        )
        task.artifacts.append(
            Artifact(
                artifact_id=artifact_id,
                name=name,
                description=(
                    "NIfTI output. Anonymous files are accessible by possession of this URL; "
                    "session-owned files require the same browser session."
                ),
                parts=[Part(url=url, filename=name, media_type=file["media_type"])],
                metadata={
                    key: file[key]
                    for key in (
                        "sha256",
                        "size_bytes",
                        "output_id",
                        "label_id",
                        "label_name",
                        "mask_value",
                    )
                    if key in file
                },
            )
        )
    return task


def routes(service: Any, authenticate: Callable[[Request], Any] | None = None) -> list[Route]:
    """Mount public A2A; an optional browser-session resolver preserves private ownership."""
    public_url = service.public_url.rstrip("/")

    async def owner(request: Request) -> str:
        principal = await _resolve(authenticate(request)) if authenticate is not None else None
        if principal is None:
            principal = PUBLIC_A2A_PRINCIPAL
        if not isinstance(principal, str) or not principal:
            raise HTTPException(401, "The browser session is invalid.")
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
                        "The browser session is invalid.",
                        401,
                        "UNAUTHENTICATED",
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
                        "The browser session is invalid.",
                        401,
                        "UNAUTHENTICATED",
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
        arguments = {
            "principal": principal,
            "upload_id": upload_id,
            "text": text,
            "modality": modality,
            "message_id": params.message.message_id,
            "context_id": params.message.context_id or None,
        }
        submit_a2a = getattr(service, "submit_a2a", None)
        if submit_a2a is not None:
            row = await _resolve(submit_a2a(**arguments, task_id=params.message.task_id or None))
        else:
            if params.message.task_id:
                raise UnsupportedOperationError("Task continuation is unavailable.")
            row = await _resolve(service.submit(**arguments))
        request.state.a2a_return_immediately = params.configuration.return_immediately
        request.state.a2a_history_length = (
            params.configuration.history_length
            if params.configuration.HasField("history_length")
            else None
        )
        return principal, row, modes

    async def send(request: Request) -> Response:
        principal, row, modes = await submit(request)
        while not request.state.a2a_return_immediately and row["status"] not in _TURN_FINISHED:
            if await request.is_disconnected():
                return Response(status_code=499)
            await asyncio.sleep(0.25)
            row = await lookup(principal, row["id"])
        return _json(
            {
                "task": MessageToDict(
                    project_task(row, public_url, modes, request.state.a2a_history_length)
                )
            }
        )

    async def events(
        principal: str, row: dict, modes: tuple[str, ...], history_length: int | None = None
    ):
        task = MessageToDict(project_task(row, public_url, modes, history_length))
        yield "data: " + json.dumps({"task": task}, separators=(",", ":")) + "\n\n"
        artifacts_seen = {
            artifact["artifactId"]: json.dumps(artifact, sort_keys=True, separators=(",", ":"))
            for artifact in task.get("artifacts", [])
        }
        last_status = task["status"]
        heartbeat = time.monotonic()
        try:
            while row["status"] not in _TURN_FINISHED:
                await asyncio.sleep(0.25)
                row = await lookup(principal, row["id"])
                task = MessageToDict(project_task(row, public_url, modes, 0))
                for artifact in task.get("artifacts", []):
                    fingerprint = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
                    if artifacts_seen.get(artifact["artifactId"]) == fingerprint:
                        continue
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
                    artifacts_seen[artifact["artifactId"]] = fingerprint
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
            events(principal, row, modes, request.state.a2a_history_length),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    def query_history_length(request: Request) -> int | None:
        if set(request.query_params) - {"historyLength"}:
            raise InvalidParamsError("Only historyLength is supported by GetTask.")
        if "historyLength" in request.query_params:
            value = request.query_params["historyLength"]
            if not value.isdecimal() or int(value) > 2**31 - 1:
                raise InvalidParamsError("historyLength must be a nonnegative int32.")
            return int(value)
        return None

    async def get(request: Request) -> Response:
        principal = await owner(request)
        history_length = query_history_length(request)
        row = await lookup(principal, request.path_params["id"])
        return _json(MessageToDict(project_task(row, public_url, history_length=history_length)))

    async def subscribe(request: Request) -> Response:
        principal = await owner(request)
        history_length = query_history_length(request)
        row = await lookup(principal, request.path_params["id"])
        if row["status"] in _TERMINAL:
            raise UnsupportedOperationError("Cannot subscribe to a terminal Task; use GetTask.")
        return StreamingResponse(
            events(principal, row, _MODES, history_length),
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
        return _json(MessageToDict(build_agent_card(public_url)))

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

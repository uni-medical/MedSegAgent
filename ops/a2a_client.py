"""Small MedSegAgent client: upload an image, stream a Task, recover through GetTask.

Artifact URLs are reported, never fetched. Uses anonymous access and one HTTPX
cookie jar; this is an image-aware client fixture, not platform acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx

VERSION = {"A2A-Version": "1.0"}
ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
STOPPED = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_AUTH_REQUIRED",
}
MAX_RAW_UPLOAD = 90 * 1024**2


class ClientError(RuntimeError):
    """A transport, admission or response error; no segmentation success is implied."""


def identifier(value, field):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ClientError(f"Invalid {field}; expected an opaque ID of 1–128 safe characters.")
    return value


def endpoints(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ClientError(
            "--url must be an HTTP(S) service URL without credentials, query or fragment."
        )
    base = url.rstrip("/")
    base = base.removesuffix("/a2a/v1")
    return base + "/a2a", base + "/a2a/v1"


def response_json(response):
    try:
        value = response.json()
    except ValueError:
        raise ClientError(f"HTTP {response.status_code}: expected a JSON response.") from None
    if not response.is_success:
        error = value.get("error", {}) if isinstance(value, dict) else {}
        if not isinstance(error, dict):
            error = {}
        details = error.get("details", [])
        if not isinstance(details, list):
            details = []
        reason = next(
            (
                item.get("reason")
                for item in details
                if isinstance(item, dict) and item.get("reason")
            ),
            None,
        )
        reason = reason or error.get("code") or error.get("status") or "REQUEST_FAILED"
        message = str(error.get("message", "Request rejected."))[:1000]
        raise ClientError(f"HTTP {response.status_code} {reason}: {message}")
    if not isinstance(value, dict):
        raise ClientError("Expected a JSON object.")
    return value


def task_object(value, expected_id=None):
    if not isinstance(value, dict):
        raise ClientError("Expected an A2A Task object.")
    task_id = identifier(value.get("id"), "Task.id")
    identifier(value.get("contextId"), "Task.contextId")
    if expected_id and task_id != expected_id:
        raise ClientError("GetTask returned a different Task ID.")
    status = value.get("status")
    if not isinstance(status, dict) or not isinstance(status.get("state"), str):
        raise ClientError("Task.status.state is missing.")
    return value


def run(
    *,
    url,
    prompt=None,
    image=None,
    modality=None,
    task_id=None,
    context_id=None,
    message_id=None,
    upload_id=None,
    timeout=7200,
    poll_interval=2,
    transport=None,
    notify=None,
):
    """Return the authoritative stopped Task; timeout does not cancel remote work."""
    upload_base, interface = endpoints(url)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ClientError("timeout must be a positive finite number.")
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ClientError("poll_interval must be a positive finite number.")
    for field, value in (
        ("task_id", task_id),
        ("context_id", context_id),
        ("message_id", message_id),
        ("upload_id", upload_id),
    ):
        if value is not None:
            identifier(value, field)
    if modality not in {None, "CT", "MR"}:
        raise ClientError("modality must be CT or MR.")
    if prompt is not None and (not prompt.strip() or len(prompt) > 4000):
        raise ClientError("prompt must contain 1–4000 characters.")
    if prompt is None and (not task_id or image or upload_id or modality or context_id):
        raise ClientError("Supply --prompt to submit, or only --task-id to poll an existing Task.")
    if prompt and not (image or upload_id or task_id or context_id):
        raise ClientError("A first request needs --image or --upload-id.")
    if image and upload_id:
        raise ClientError("Use either --image or --upload-id.")
    deadline = time.monotonic() + timeout
    notify = notify or (lambda _event: None)

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise ClientError(
                f"Client timeout; remote work is not canceled. Recover taskId={task_id!r}."
            )
        return seconds

    def request_timeout():
        return httpx.Timeout(min(30, remaining()), connect=min(10, remaining()))

    def pause(retry_after=None):
        delay = poll_interval
        if retry_after is not None:
            try:
                delay = max(delay, min(30, float(retry_after)))
            except (TypeError, ValueError):
                pass
        time.sleep(min(delay, remaining()))

    with httpx.Client(transport=transport, follow_redirects=False, trust_env=False) as client:
        if image:
            path = Path(image)
            if not path.name.endswith((".nii", ".nii.gz")):
                raise ClientError("--image must be a .nii or .nii.gz file.")
            size = path.stat().st_size
            if not 0 < size <= MAX_RAW_UPLOAD:
                raise ClientError(
                    "Raw upload supports 1 byte–90 MiB; use upload sessions for larger files."
                )
            try:
                with path.open("rb") as source:
                    uploaded = response_json(
                        client.post(
                            upload_base + "/uploads",
                            content=source,
                            headers={
                                "Content-Type": "application/octet-stream",
                                "X-Filename": path.name,
                                "Content-Length": str(size),
                            },
                            timeout=request_timeout(),
                        )
                    )
            except httpx.TransportError as exc:
                raise ClientError(
                    f"Upload transport failed ({type(exc).__name__}); no task submitted."
                ) from None
            upload_id = identifier(uploaded.get("id"), "upload id")
            notify({"upload_id": upload_id})

        if prompt is not None:
            message_id = message_id or str(uuid.uuid4())
            message = {
                "messageId": message_id,
                "role": "ROLE_USER",
                "parts": [{"text": prompt, "mediaType": "text/plain"}],
            }
            if task_id:
                message["taskId"] = task_id
            if context_id:
                message["contextId"] = context_id
            data = {}
            if upload_id:
                data["upload_id"] = upload_id
            if modality:
                data["modality"] = modality
            if data:
                message["parts"].append({"data": data, "mediaType": "application/json"})
            body = {
                "message": message,
                "configuration": {
                    "returnImmediately": True,
                    "acceptedOutputModes": ["text/plain", "application/gzip"],
                },
            }
            headers = {**VERSION, "Content-Type": "application/a2a+json"}
            notify({"message_id": message_id, "upload_id": upload_id})
            admitted = False
            try:
                with client.stream(
                    "POST",
                    interface + "/message:stream",
                    json=body,
                    headers=headers,
                    timeout=request_timeout(),
                ) as response:
                    if not response.is_success:
                        response.read()
                        response_json(response)
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        raise ClientError("SendStreamingMessage did not return text/event-stream.")
                    lines = []
                    event_bytes = 0
                    for line in response.iter_lines():
                        remaining()
                        if line.startswith("data:"):
                            lines.append(line[5:].lstrip(" "))
                            event_bytes += len(line.encode())
                            if event_bytes > 4 * 1024**2:
                                raise ClientError("SSE record exceeds 4 MiB.")
                        elif not line and lines:
                            event = json.loads("\n".join(lines))
                            lines, event_bytes = [], 0
                            if not isinstance(event, dict):
                                raise ClientError("Invalid A2A SSE event.")
                            if "error" in event:
                                break  # Recover the persisted Task below.
                            if "task" in event:
                                snapshot = task_object(event["task"], task_id)
                                task_id = snapshot["id"]
                                admitted = True
                                notify({"task_id": task_id, "context_id": snapshot["contextId"]})
                                if snapshot["status"]["state"] in STOPPED:
                                    break
                            elif "statusUpdate" in event:
                                update = event["statusUpdate"]
                                if not isinstance(update, dict) or not isinstance(
                                    update.get("status"), dict
                                ):
                                    raise ClientError("Invalid A2A statusUpdate.")
                                observed_id = identifier(
                                    update.get("taskId"), "statusUpdate.taskId"
                                )
                                if task_id and observed_id != task_id:
                                    raise ClientError("SSE update changed Task ID.")
                                task_id, admitted = observed_id, True
                                if update.get("status", {}).get("state") in STOPPED:
                                    break
            except (httpx.TransportError, json.JSONDecodeError):
                pass  # HTTP/SSE lifetime never determines task lifetime.
            if not admitted:
                # The initial Task frame may be lost after admission. Retry exactly
                # the original message, including its original contextId (or absence).
                for attempt in range(3):
                    try:
                        sent = response_json(
                            client.post(
                                interface + "/message:send",
                                json=body,
                                headers=headers,
                                timeout=request_timeout(),
                            )
                        )
                        recovered = task_object(sent.get("task"), task_id)
                        task_id = recovered["id"]
                        notify({"task_id": task_id, "context_id": recovered["contextId"]})
                        break
                    except httpx.TransportError as exc:
                        if attempt == 2:
                            raise ClientError(
                                f"Admission response lost ({type(exc).__name__}); retry the same "
                                f"messageId={message_id} and upload_id={upload_id!r} with unchanged inputs."
                            ) from None
                        pause()
        while True:
            try:
                response = client.get(
                    interface + "/tasks/" + task_id, headers=VERSION, timeout=request_timeout()
                )
                if response.status_code in {429, 502, 503, 504}:
                    pause(response.headers.get("Retry-After"))
                    continue
                task = task_object(response_json(response), task_id)
                if task["status"]["state"] in STOPPED:
                    return task
            except httpx.TransportError:
                pass
            pause()


def report(task):
    """Keep complete Task data and provide a compact Artifact/continuation index."""
    state = task["status"]["state"]
    value = {"task": task, "artifacts": task.get("artifacts", [])}
    if state == "TASK_STATE_INPUT_REQUIRED":
        value["continuation"] = {
            "task_id": task["id"],
            "context_id": task["contextId"],
            "instruction": "Reply with --task-id and --context-id above plus --prompt. "
            "The existing image is reused. To use a different image, start a new context. "
            "Use a new --message-id for this new user reply.",
        }
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="MedSegAgent root URL or /a2a/v1 interface")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--modality", choices=["CT", "MR"])
    parser.add_argument("--task-id", help="Resume with --prompt, or poll without --prompt")
    parser.add_argument("--context-id")
    parser.add_argument(
        "--message-id", help="Stable request UUID when retrying an uncertain submission"
    )
    parser.add_argument("--upload-id", help="Reuse an already uploaded image")
    parser.add_argument(
        "--timeout", type=float, default=7200, help="Total client deadline in seconds"
    )
    parser.add_argument("--poll-interval", type=float, default=2)
    args = vars(parser.parse_args(argv))
    try:
        task = run(
            **args, notify=lambda event: print(json.dumps(event), file=sys.stderr, flush=True)
        )
        print(json.dumps(report(task), ensure_ascii=False, indent=2))
        state = task["status"]["state"]
        return (
            0
            if state == "TASK_STATE_COMPLETED"
            else 3
            if state in {"TASK_STATE_INPUT_REQUIRED", "TASK_STATE_AUTH_REQUIRED"}
            else 2
        )
    except (ClientError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

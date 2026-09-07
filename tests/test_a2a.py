"""Protocol and security contract tests for the shared-service A2A projection."""

from __future__ import annotations

import copy
import json

import pytest
from a2a.types import AgentCard, StreamResponse, Task
from google.protobuf.json_format import ParseDict
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.testclient import TestClient

from medsegagent.a2a import routes


class FakeService:
    public_url = "https://medseg.example.org"

    def __init__(self):
        self.calls = []
        self.rows = {}
        self.complete_after_get = False

    async def submit(self, **params):
        self.calls.append(params)
        owner, message_id = params["principal"], params["message_id"]
        task_id = f"task-{owner}-{message_id}"
        if task_id not in self.rows:
            self.rows[task_id] = {
                "id": task_id,
                "context_id": params["context_id"] or f"ctx-{owner}",
                "principal": owner,
                "status": "queued",
                "updated_at": "2026-09-07T12:00:00Z",
                "result": None,
                "error": None,
            }
        return copy.deepcopy(self.rows[task_id])

    def get(self, principal, task_id):
        row = self.rows.get(task_id)
        if not row or row["principal"] != principal:
            return None
        if self.complete_after_get and row["status"] not in {"canceled", "completed", "failed"}:
            self.complete(task_id)
        return copy.deepcopy(row)

    async def cancel(self, principal, task_id):
        row = self.get(principal, task_id)
        if row is None:
            raise KeyError(task_id)
        self.rows[task_id]["status"] = "canceled"
        return self.get(principal, task_id)

    def complete(self, task_id):
        self.rows[task_id].update(
            {
                "status": "completed",
                "updated_at": "2026-09-07T12:00:01Z",
                "result": {
                    "tool": "segment_ct",
                    "modality": "CT",
                    "targets": ["liver"],
                    "elapsed_seconds": 1.5,
                    "segmentation_path": "/private/patient/mask.nii.gz",
                    "input_path": "/private/patient/input.nii.gz",
                    "provider_key": "must-not-leak",
                },
                "files": [
                    {
                        "name": "segmentation.nii.gz",
                        "media_type": "application/gzip",
                        "url": f"/api/tasks/{task_id}/files/segmentation.nii.gz",
                        "size_bytes": 123,
                        "sha256": "a" * 64,
                    }
                ],
            }
        )


@pytest.fixture
def fixture():
    service = FakeService()

    def auth(request):
        token = request.headers.get("authorization")
        if token not in {"Bearer alice", "Bearer bob"}:
            raise HTTPException(401)
        return token.split()[1]

    client = TestClient(Starlette(routes=routes(service, auth)))
    return service, client


def headers(principal="alice"):
    return {"Authorization": f"Bearer {principal}", "A2A-Version": "1.0"}


def request(message_id="one", **configuration):
    return {
        "message": {
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [
                {"text": "Segment the liver", "mediaType": "text/plain"},
                {
                    "data": {"upload_id": "upload-1", "modality": "CT"},
                    "mediaType": "application/json",
                },
            ],
        },
        "configuration": {"returnImmediately": True, **configuration},
    }


def reason(response):
    return response.json()["error"]["details"][0]["reason"]


def test_public_card_uses_v1_sdk_schema_and_declares_file_boundary(fixture):
    _, client = fixture
    response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = ParseDict(response.json(), AgentCard())
    assert card.supported_interfaces[0].protocol_version == "1.0"
    assert card.supported_interfaces[0].protocol_binding == "HTTP+JSON"
    assert card.capabilities.streaming and not card.capabilities.push_notifications
    assert card.security_schemes["bearerAuth"].http_auth_security_scheme.scheme == "bearer"
    assert "base64" in card.description and "Research" in card.description
    assert "kind" not in response.text


@pytest.mark.parametrize("path", ["/a2a/v1/message:send", "/a2a/v1/message:stream"])
def test_auth_and_version_required_before_admission(fixture, path):
    service, client = fixture
    assert client.post(path, json=request()).status_code == 401
    response = client.post(path, json=request(), headers={"Authorization": "Bearer alice"})
    assert response.status_code == 400 and reason(response) == "VERSION_NOT_SUPPORTED"
    assert not service.calls


def test_submit_get_owner_isolation_and_same_id_forwarded(fixture):
    service, client = fixture
    first = client.post("/a2a/v1/message:send", json=request(), headers=headers()).json()["task"]
    repeated = client.post("/a2a/v1/message:send", json=request(), headers=headers()).json()["task"]
    assert first["id"] == repeated["id"]
    ParseDict(first, Task())
    assert first["status"]["state"] == "TASK_STATE_SUBMITTED"
    assert service.calls[0] == {
        "principal": "alice",
        "upload_id": "upload-1",
        "text": "Segment the liver",
        "modality": "CT",
        "message_id": "one",
        "context_id": None,
    }
    assert client.get(f"/a2a/v1/tasks/{first['id']}", headers=headers()).status_code == 200
    denied = client.get(f"/a2a/v1/tasks/{first['id']}", headers=headers("bob"))
    assert denied.status_code == 404 and reason(denied) == "TASK_NOT_FOUND"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9000/api/uploads/upload-1/file",
        "https://evil.example.org/api/uploads/upload-1/file",
        "https://medseg.example.org@evil.example.org/api/uploads/upload-1/file",
        "https://medseg.example.org/api/uploads/../secret/file",
        "https://medseg.example.org/api/uploads/%2e%2e/file",
        "https://medseg.example.org/api/uploads/upload-1/file?token=secret",
        "https://medseg.example.org/api/uploads/upload-1/file#fragment",
        "file:///private/patient/image.nii.gz",
        "https://medseg.example.org/api/uploads/upload-1/file/",
    ],
)
def test_file_uri_ssrf_and_path_confusion_rejected_before_admission(fixture, url):
    service, client = fixture
    payload = request()
    payload["message"]["parts"][1:] = [
        {"url": url, "mediaType": "application/gzip"},
        {"data": {"modality": "CT"}},
    ]
    response = client.post("/a2a/v1/message:send", json=payload, headers=headers())
    assert response.status_code == 400
    assert not service.calls


def test_same_service_url_is_an_id_reference_not_a_fetch(fixture):
    service, client = fixture
    payload = request()
    payload["message"]["parts"][1:] = [
        {
            "url": "https://medseg.example.org/api/uploads/upload-1/file",
            "mediaType": "application/gzip",
        },
        {"data": {"modality": "MR"}},
    ]
    assert client.post("/a2a/v1/message:send", json=payload, headers=headers()).status_code == 200
    assert service.calls[0]["upload_id"] == "upload-1"
    assert service.calls[0]["modality"] == "MR"


@pytest.mark.parametrize(
    "part",
    [
        {"raw": "YWJjZA==", "mediaType": "application/gzip"},
        {"data": {"upload_id": "upload-1", "modality": "CT", "path": "/private/patient"}},
        {"data": {"upload_id": "../private", "modality": "CT"}},
        {"data": {"upload_id": "upload-1", "modality": "ct"}},
        {"data": {"upload_id": "upload-1", "modality": ["CT"]}},
        {"data": []},
        {"data": {"upload_id": "upload-1"}},
        {"data": {"upload_id": "upload-1", "modality": "CT"}, "metadata": {"secret": "value"}},
    ],
)
def test_invalid_input_boundaries_reject_without_inference(fixture, part):
    service, client = fixture
    payload = request()
    payload["message"]["parts"][1] = part
    response = client.post("/a2a/v1/message:stream", json=payload, headers=headers())
    assert response.status_code in {400, 415}
    assert not service.calls


def test_raw_parse_errors_are_redacted(fixture):
    service, client = fixture
    payload = request()
    payload["message"]["private-secret-123"] = "patient-private-value"
    response = client.post("/a2a/v1/message:send", json=payload, headers=headers())
    assert response.status_code == 400
    assert "private" not in response.text
    assert not service.calls


def test_envelope_text_limit_and_negotiation_fail_before_admission(fixture):
    service, client = fixture
    payload = request()
    payload["message"]["parts"][0]["text"] = "x" * 4001
    assert client.post("/a2a/v1/message:send", json=payload, headers=headers()).status_code == 400
    payload["message"]["parts"][0]["text"] = "x" * (64 * 1024 + 1)
    assert client.post("/a2a/v1/message:send", json=payload, headers=headers()).status_code == 400
    payload["message"]["parts"][0]["text"] = "x" * (512 * 1024 + 1)
    assert client.post("/a2a/v1/message:send", json=payload, headers=headers()).status_code == 413
    payload = request(acceptedOutputModes=["image/png"])
    response = client.post("/a2a/v1/message:send", json=payload, headers=headers())
    assert response.status_code == 400 and reason(response) == "CONTENT_TYPE_NOT_SUPPORTED"
    assert not service.calls


def test_sse_is_canonical_and_emits_artifacts_before_terminal_status(fixture):
    service, client = fixture
    service.complete_after_get = True
    response = client.post("/a2a/v1/message:stream", json=request(), headers=headers())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    for event in events:
        ParseDict(event, StreamResponse())
    assert events[0]["task"]["status"]["state"] == "TASK_STATE_SUBMITTED"
    assert events[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    artifacts = [
        event["artifactUpdate"]["artifact"] for event in events if "artifactUpdate" in event
    ]
    assert len(artifacts) == 3
    assert artifacts[-1]["parts"][0]["url"].endswith("/files/segmentation.nii.gz")
    assert "/private" not in response.text and "must-not-leak" not in response.text
    task_id = events[0]["task"]["id"]
    recovered = client.get(f"/a2a/v1/tasks/{task_id}", headers=headers()).json()
    assert recovered["artifacts"] == artifacts


def test_json_only_and_blocking_send_obeys_output_and_wait_contract(fixture):
    service, client = fixture
    service.complete_after_get = True
    payload = request(returnImmediately=False, acceptedOutputModes=["application/json"])
    response = client.post("/a2a/v1/message:send", json=payload, headers=headers())
    task = response.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert [artifact["artifactId"] for artifact in task["artifacts"]] == ["result-json"]


def test_cancellation_owner_terminal_and_replay_contract(fixture):
    service, client = fixture
    task_id = client.post("/a2a/v1/message:send", json=request(), headers=headers()).json()["task"][
        "id"
    ]
    assert (
        client.post(f"/a2a/v1/tasks/{task_id}:cancel", json={}, headers=headers("bob")).status_code
        == 404
    )
    canceled = client.post(f"/a2a/v1/tasks/{task_id}:cancel", json={}, headers=headers())
    assert canceled.json()["status"]["state"] == "TASK_STATE_CANCELED"
    assert (
        client.post(f"/a2a/v1/tasks/{task_id}:cancel", json={}, headers=headers()).status_code
        == 200
    )
    assert client.get(f"/a2a/v1/tasks/{task_id}:subscribe", headers=headers()).status_code == 400
    service.complete(task_id)
    response = client.post(f"/a2a/v1/tasks/{task_id}:cancel", json={}, headers=headers())
    assert response.status_code == 400 and reason(response) == "TASK_NOT_CANCELABLE"


def test_invalid_history_and_unsupported_push_are_explicit(fixture):
    service, client = fixture
    payload = request(historyLength=-1)
    assert client.post("/a2a/v1/message:send", json=payload, headers=headers()).status_code == 400
    payload = request(taskPushNotificationConfig={"url": "https://callback.example.org"})
    response = client.post("/a2a/v1/message:send", json=payload, headers=headers())
    assert response.status_code == 400
    assert reason(response) == "PUSH_NOTIFICATION_NOT_SUPPORTED"
    assert not service.calls


@pytest.mark.parametrize(
    "status_code,error_code,expected_status,expected_reason",
    [
        (401, "UNAUTHENTICATED", 401, "UNAUTHENTICATED"),
        (404, "FILE_NOT_FOUND", 404, "TASK_NOT_FOUND"),
        (409, "IDEMPOTENCY_CONFLICT", 400, "INVALID_PARAMS"),
        (429, "CAPACITY_EXCEEDED", 429, "CAPACITY_EXCEEDED"),
    ],
)
def test_service_error_valueerror_subclasses_preserve_http_boundary(
    fixture,
    status_code,
    error_code,
    expected_status,
    expected_reason,
):
    service, client = fixture

    class ServiceError(ValueError):
        def __init__(self):
            super().__init__("private-error-contents")
            self.status_code, self.code = status_code, error_code

    async def fail(**_params):
        raise ServiceError()

    service.submit = fail
    response = client.post("/a2a/v1/message:send", json=request(), headers=headers())
    assert response.status_code == expected_status and reason(response) == expected_reason
    assert "private-error" not in response.text
    if expected_status == 429:
        assert response.headers["retry-after"] == "5"
    if expected_status == 401:
        assert response.headers["www-authenticate"] == "Bearer"


def test_failed_task_preserves_state_without_private_error_projection(fixture):
    service, client = fixture
    task_id = client.post("/a2a/v1/message:send", json=request(), headers=headers()).json()["task"][
        "id"
    ]
    service.rows[task_id].update(
        status="failed",
        error={
            "code": "INFERENCE_FAILED",
            "message": "private-provider-key",
            "trace": "sensitive-trace",
        },
    )
    response = client.get(f"/a2a/v1/tasks/{task_id}", headers=headers())
    assert response.json()["status"]["state"] == "TASK_STATE_FAILED"
    assert response.json()["metadata"]["errorCode"] == "INFERENCE_FAILED"
    assert "private-provider" not in response.text and "sensitive" not in response.text


def test_expired_results_do_not_pretend_to_be_downloadable(fixture):
    service, client = fixture
    task_id = client.post("/a2a/v1/message:send", json=request(), headers=headers()).json()["task"][
        "id"
    ]
    service.complete(task_id)
    service.rows[task_id].update(files_expired=True, files=[], result=None)
    response = client.get(f"/a2a/v1/tasks/{task_id}", headers=headers())
    assert response.json()["metadata"]["filesExpired"] is True
    assert "have expired" in response.text
    assert "/files/segmentation.nii.gz" not in response.text


def test_real_shared_service_upload_idempotency_owner_and_restart(tmp_path, monkeypatch):
    """Exercise Web upload plus A2A against real SQLite, without model/GPU work."""
    import nibabel as nib
    import numpy as np

    from medsegagent.service import Service
    from medsegagent.web import create_app

    monkeypatch.setattr(Service, "launch", lambda self, task_id: None)
    alice = {"Authorization": "Bearer " + "a" * 32, "A2A-Version": "1.0"}
    bob = {"Authorization": "Bearer " + "b" * 32, "A2A-Version": "1.0"}
    tokens = {"alice": "a" * 32, "bob": "b" * 32}
    volume = nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.int16), np.eye(4)).to_bytes()

    def app():
        return create_app(tmp_path, "https://medseg.example.org", tokens=tokens)

    with TestClient(app()) as client:
        assert client.post("/a2a/v1/message:send", json=request()).status_code == 401
        uploaded = client.post(
            "/api/uploads", content=volume, headers={**alice, "X-Filename": "synthetic.nii"}
        )
        assert uploaded.status_code == 201
        upload_id = uploaded.json()["id"]
        payload = request()
        payload["message"]["parts"][1]["data"]["upload_id"] = upload_id
        first = client.post("/a2a/v1/message:send", json=payload, headers=alice)
        assert first.status_code == 200
        task_id = first.json()["task"]["id"]
        replay = client.post("/a2a/v1/message:send", json=payload, headers=alice)
        assert replay.json()["task"]["id"] == task_id
        changed = copy.deepcopy(payload)
        changed["message"]["parts"][0]["text"] = "Segment the spleen"
        conflict = client.post("/a2a/v1/message:send", json=changed, headers=alice)
        assert conflict.status_code == 400 and reason(conflict) == "INVALID_PARAMS"
        assert client.post("/a2a/v1/message:send", json=payload, headers=bob).status_code == 404
        assert client.get(f"/a2a/v1/tasks/{task_id}", headers=bob).status_code == 404

    with TestClient(app()) as client:
        recovered = client.get(f"/a2a/v1/tasks/{task_id}", headers=alice)
        assert recovered.status_code == 200
        assert recovered.json()["id"] == task_id
        replay = client.post("/a2a/v1/message:send", json=payload, headers=alice)
        assert replay.json()["task"]["id"] == task_id
        assert client.get(f"/api/uploads/{upload_id}/file", headers=alice).content == volume
        assert client.get(f"/api/uploads/{upload_id}/file", headers=bob).status_code == 404

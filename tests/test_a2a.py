"""Protocol and security contract tests for the shared-service A2A projection."""

from __future__ import annotations

import copy
import json

import pytest
from a2a.types import AgentCard, StreamResponse, Task
from google.protobuf.json_format import ParseDict
from starlette.applications import Starlette
from starlette.testclient import TestClient

from medsegagent.a2a import project_task, routes
from medsegagent.service import PUBLIC_A2A_PRINCIPAL


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
        principal = request.cookies.get("test_identity")
        return principal if principal in {"alice", "bob"} else None

    client = TestClient(Starlette(routes=routes(service, auth)))
    return service, client


def headers(principal="alice"):
    return {"Cookie": f"test_identity={principal}", "A2A-Version": "1.0"}


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


def test_status_messages_have_unique_ids_for_distinct_phase_content():
    messages = []
    for phase in ("queued", "routing", "running", "completed", "failed", "canceled"):
        row = {"id": "task-1", "context_id": "context-1", "status": phase, "result": None}
        message = project_task(row, "https://medseg.example.org").status.message
        repeated = project_task(row, "https://medseg.example.org").status.message
        assert message.message_id == repeated.message_id
        assert message == repeated
        messages.append(message)
    assert len({message.parts[0].text for message in messages}) == len(messages)
    assert len({message.message_id for message in messages}) == len(messages)


def test_public_card_uses_v1_sdk_schema_and_declares_one_segmentation_skill(fixture):
    _, client = fixture
    response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = ParseDict(response.json(), AgentCard())
    assert card.supported_interfaces[0].protocol_version == "1.0"
    assert card.supported_interfaces[0].protocol_binding == "HTTP+JSON"
    assert card.capabilities.streaming and not card.capabilities.push_notifications
    assert not card.security_schemes and not card.security_requirements
    assert "opaque IDs" in card.description and "GitHub" in card.description
    assert len(card.skills) == 1
    assert "kind" not in response.text


@pytest.mark.parametrize("path", ["/a2a/v1/message:send", "/a2a/v1/message:stream"])
def test_version_is_required_but_credentials_are_not(fixture, path):
    service, client = fixture
    response = client.post(path, json=request())
    assert response.status_code == 400 and reason(response) == "VERSION_NOT_SUPPORTED"
    assert not service.calls
    service.complete_after_get = True
    response = client.post(path, json=request(), headers={"A2A-Version": "1.0"})
    assert response.status_code == 200
    assert service.calls[0]["principal"] == PUBLIC_A2A_PRINCIPAL


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


@pytest.mark.parametrize("prefix", ["api", "a2a"])
def test_same_service_url_is_an_id_reference_not_a_fetch(fixture, prefix):
    service, client = fixture
    payload = request()
    payload["message"]["parts"][1:] = [
        {
            "url": f"https://medseg.example.org/{prefix}/uploads/upload-1/file",
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
    assert len(artifacts) == 2
    assert artifacts[-1]["parts"][0]["url"].endswith("/files/segmentation.nii.gz")
    assert "/private" not in response.text and "must-not-leak" not in response.text
    task_id = events[0]["task"]["id"]
    recovered = client.get(f"/a2a/v1/tasks/{task_id}", headers=headers()).json()
    assert recovered["artifacts"] == artifacts
    assert (
        events[-1]["statusUpdate"]["metadata"]["segmentation"]
        == recovered["metadata"]["segmentation"]
    )


def test_mask_only_and_blocking_send_obeys_output_and_wait_contract(fixture):
    service, client = fixture
    service.complete_after_get = True
    payload = request(returnImmediately=False, acceptedOutputModes=["application/gzip"])
    response = client.post("/a2a/v1/message:send", json=payload, headers=headers())
    task = response.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert len(task["artifacts"]) == 1
    assert task["artifacts"][0]["artifactId"].startswith("file-")
    assert "segmentation" in task["metadata"]


def test_retired_json_artifact_is_not_advertised_or_negotiated(fixture):
    service, client = fixture
    card = client.get("/.well-known/agent-card.json").json()
    assert "application/json" not in card["defaultOutputModes"]
    response = client.post(
        "/a2a/v1/message:send",
        json=request(acceptedOutputModes=["application/json"]),
        headers=headers(),
    )
    assert response.status_code == 400 and reason(response) == "CONTENT_TYPE_NOT_SUPPORTED"
    assert not service.calls


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
        assert "www-authenticate" not in response.headers


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
    from medsegagent.auth import SESSION_COOKIE, AuthStore

    volume = nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.int16), np.eye(4)).to_bytes()

    def app():
        return create_app(tmp_path, "https://medseg.example.org")

    application = app()
    with TestClient(application) as client:
        store = AuthStore(application.state.service.db, "https://medseg.example.org")
        sessions = [store.create_guest(), store.create_guest()]
        alice, bob = [
            {
                "Cookie": f"{SESSION_COOKIE}={session.token}",
                "A2A-Version": "1.0",
                "Origin": "https://medseg.example.org",
            }
            for session in sessions
        ]
        assert client.post("/a2a/v1/message:send", json=request()).status_code == 400
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


@pytest.mark.parametrize("task_name", ["lung_nodules", "liver_lesions"])
@pytest.mark.parametrize("nonzero_voxels", [0, 2])
def test_shared_lesion_results_preserve_detection_semantics_without_raw_paths(
    tmp_path, monkeypatch, task_name, nonzero_voxels
):
    """Real upload, SQLite, Web/A2A projection and restart with only inference mocked."""
    import gzip
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    from medsegagent import agent, core
    from medsegagent.web import create_app

    empty = nonzero_voxels == 0
    detection_status = "no_target_detected" if empty else "target_detected"
    private_marker = "private-filtering-audit-must-not-leak"
    values = np.zeros((4, 5, 6), dtype=np.uint8)
    values.ravel()[:nonzero_voxels] = 1

    async def select(text, modality, execution, **kwargs):
        await execution.call("segment", {"targets": [task_name]})
        return {
            "status": "completed",
            "summary": "Requested lesion segmentation finished",
            "unresolved": [],
        }

    async def segment(**kwargs):
        run = Path(kwargs["output_dir"]) / "unique-run"
        run.mkdir(parents=True)
        mask = run / "segmentation.nii.gz"
        raw_mask = run / "segmentation.raw.nii.gz"
        nib.save(nib.Nifti1Image(values, np.eye(4)), mask)
        raw_mask.write_bytes(mask.read_bytes())
        return {
            "task": task_name,
            "targets": [task_name],
            "speed": "standard",
            "schema_version": 3,
            "volume_measurement": {"spatial_unit": "mm"},
            "normalization_seconds": 0.02,
            "labels": [
                {
                    "id": 1,
                    "source_id": 2 if task_name == "lung_nodules" else 1,
                    "name": task_name,
                    "color": "#ff0000",
                    "voxels": nonzero_voxels,
                    "volume_mm3": float(nonzero_voxels),
                    "volume_ml": nonzero_voxels / 1000,
                }
            ],
            "nonzero_voxels": nonzero_voxels,
            "detection_status": detection_status,
            "no_target_detected": empty,
            "segmentation_path": str(mask),
            "filtering": {"raw_segmentation_path": str(raw_mask), "private": private_marker},
            "warning": f"{private_marker}: {raw_mask}",
        }

    monkeypatch.setattr(agent, "run_agent", select)
    monkeypatch.setattr(core, "segment", segment)
    from medsegagent.auth import SESSION_COOKIE, AuthStore

    volume = nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.int16), np.eye(4)).to_bytes()

    def app():
        return create_app(tmp_path, "https://medseg.example.org")

    application = app()
    with TestClient(application) as client:
        session = AuthStore(
            application.state.service.db, "https://medseg.example.org"
        ).create_guest()
        auth = {
            "Cookie": f"{SESSION_COOKIE}={session.token}",
            "A2A-Version": "1.0",
            "Origin": "https://medseg.example.org",
        }
        upload = client.post(
            "/api/uploads", content=volume, headers={**auth, "X-Filename": "synthetic.nii"}
        )
        assert upload.status_code == 201
        payload = request(returnImmediately=False)
        payload["message"]["parts"][0]["text"] = f"Segment {task_name}"
        payload["message"]["parts"][1]["data"]["upload_id"] = upload.json()["id"]
        sent = client.post("/a2a/v1/message:send", json=payload, headers=auth)
        assert sent.status_code == 200, sent.text
        task = sent.json()["task"]
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        task_id = task["id"]
        result = task["metadata"]["segmentation"]
        assert all(a["artifactId"] != "result-json" for a in task["artifacts"])
        assert result["detection_status"] == detection_status
        assert result["no_target_detected"] is empty
        assert result["nonzero_voxels"] == nonzero_voxels
        assert not {"speed", "tool", "device", "model"} & result.keys()
        assert result["task"] == task_name
        assert result["quality"] == "standard"
        assert result["schema_version"] == 5
        assert result["labels"][0]["id"] == 1
        assert result["labels"][0]["color"] == "#ff0000"
        assert set(result["labels"][0]) == {
            "id",
            "source_id",
            "name",
            "color",
            "voxels",
            "volume_mm3",
            "volume_ml",
        }
        assert result["labels"][0]["voxels"] == nonzero_voxels
        assert result["outputs"][0]["labels"][0]["name"] == task_name
        assert ("does not rule out disease" in result["warning"]) is empty
        summary = next(a for a in task["artifacts"] if a["artifactId"] == "summary")
        assert ("does not rule out disease" in summary["parts"][0]["text"]) is empty

        web = client.get(f"/api/tasks/{task_id}", headers=auth)
        downloaded = client.get(f"/api/tasks/{task_id}/files/result.json", headers=auth)
        assert downloaded.status_code == 404
        assert not (tmp_path / "tasks" / task_id / "result.json").exists()
        for public in (web.json()["result"], result):
            assert public["no_target_detected"] is empty
            assert public["detection_status"] == detection_status
            assert public["nonzero_voxels"] == nonzero_voxels
            assert ("does not rule out disease" in public["warning"]) is empty
        for serialized in (sent.text, web.text, downloaded.text):
            assert str(tmp_path) not in serialized
            assert private_marker not in serialized
            assert "segmentation.raw" not in serialized
            assert "filtering" not in serialized
        overlay = next(file for file in web.json()["files"] if file["kind"] == "overlay")
        mask = client.get(overlay["url"], headers=auth)
        assert mask.status_code == 200
        image = nib.Nifti1Image.from_bytes(gzip.decompress(mask.content))
        assert np.count_nonzero(np.asanyarray(image.dataobj)) == nonzero_voxels
        for private_name in ("segmentation.raw.nii.gz", "filtering.json", "process.log"):
            assert (
                client.get(f"/api/tasks/{task_id}/files/{private_name}", headers=auth).status_code
                == 404
            )

    with TestClient(app()) as client:
        recovered = client.get(f"/a2a/v1/tasks/{task_id}", headers=auth)
        assert recovered.status_code == 200
        assert recovered.json()["artifacts"] == task["artifacts"]


def test_omitted_modality_reaches_the_agent_for_local_detection(fixture):
    service, client = fixture
    payload = request()
    payload["message"]["parts"][0]["text"] = "请分割肝脏，先判断影像模态。"
    payload["message"]["parts"][1] = {"data": {"upload_id": "upload-1"}}
    response = client.post("/a2a/v1/message:send", json=payload, headers=headers())
    assert response.status_code == 200, response.text
    assert service.calls[0]["modality"] is None


def test_anonymous_namespace_shares_ids_but_cannot_read_session_owned_tasks(fixture):
    service, client = fixture
    public_headers = {"A2A-Version": "1.0"}
    first = client.post("/a2a/v1/message:send", json=request(), headers=public_headers).json()[
        "task"
    ]
    assert service.calls[-1]["principal"] == PUBLIC_A2A_PRINCIPAL
    copied = client.post(
        "/a2a/v1/message:send",
        json=request(),
        headers={**public_headers, "Authorization": "Bearer obsolete-token"},
    ).json()["task"]
    assert copied["id"] == first["id"]
    assert client.get("/a2a/v1/tasks/" + first["id"], headers=public_headers).status_code == 200
    assert client.get("/a2a/v1/tasks", headers=public_headers).status_code == 400
    private = client.post("/a2a/v1/message:send", json=request(), headers=headers()).json()["task"]
    assert private["id"] != first["id"]
    assert client.get("/a2a/v1/tasks/" + private["id"], headers=public_headers).status_code == 404
    assert client.get("/a2a/v1/tasks/" + first["id"], headers=headers()).status_code == 404


def test_routes_without_identity_resolver_are_public_by_default():
    service = FakeService()
    with TestClient(Starlette(routes=routes(service))) as client:
        response = client.post(
            "/a2a/v1/message:send", json=request(), headers={"A2A-Version": "1.0"}
        )
        assert response.status_code == 200
        assert service.calls[0]["principal"] == PUBLIC_A2A_PRINCIPAL


@pytest.mark.parametrize("prefix", ["api", "a2a"])
def test_artifact_projection_accepts_only_this_task_same_origin_file_namespace(prefix):
    service = FakeService()
    service.rows["task-one"] = {"id": "task-one", "context_id": "context-one", "status": "queued"}
    service.complete("task-one")
    row = service.rows["task-one"]
    row["files"][0]["url"] = f"/{prefix}/tasks/task-one/files/segmentation.nii.gz"
    assert len(project_task(row, service.public_url).artifacts) == 2
    row["files"][0]["url"] = f"/{prefix}/tasks/other-task/files/segmentation.nii.gz"
    assert len(project_task(row, service.public_url).artifacts) == 1

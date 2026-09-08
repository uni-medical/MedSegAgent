"""A2A conversation, interruption and incremental publication contracts."""

from __future__ import annotations

import copy
import json

import pytest
from a2a.types import StreamResponse, Task
from google.protobuf.json_format import MessageToDict, ParseDict
from starlette.applications import Starlette
from starlette.testclient import TestClient

from medsegagent.a2a import build_agent_card, project_task, routes

URL = "https://medseg.example.org"
HEADERS = {"A2A-Version": "1.0"}


def history(count=3):
    return [
        {
            "messageId": f"message-{index}",
            "role": "ROLE_USER" if index % 2 == 0 else "ROLE_AGENT",
            "taskId": "task-one",
            "contextId": "context-one",
            "parts": [{"text": f"Conversation message {index}", "mediaType": "text/plain"}],
        }
        for index in range(count)
    ]


def row(status="queued", *, files=None, question=None):
    value = {
        "id": "task-one",
        "context_id": "context-one",
        "status": status,
        "updated_at": "2026-09-08T12:00:00Z",
        "a2a_history": history(),
        "error": {"code": "INPUT_REQUIRED", "message": question} if question else None,
    }
    if files:
        value.update(result={"outputs": [{"id": "output-one"}]}, files=files)
    if status == "input_required":
        value["input_expires_at"] = 1788868800
    return value


def output(name="liver", checksum="a"):
    return {
        "name": f"{name}.nii.gz",
        "url": f"/a2a/tasks/task-one/files/{name}.nii.gz",
        "media_type": "application/gzip",
        "output_id": f"output-{name}",
        "kind": "label",
        "label_id": 1,
        "label_name": name,
        "sha256": checksum * 64,
    }


class SequenceService:
    """Advance durable snapshots only when the protocol polls its service boundary."""

    public_url = URL

    def __init__(self, snapshots):
        self.snapshots = copy.deepcopy(list(snapshots))
        self.current = self.snapshots.pop(0)
        self.calls = []
        self.cancel_calls = []
        self.read_started = False

    async def submit_a2a(self, **params):
        self.calls.append(params)
        self.read_started = True
        return copy.deepcopy(self.current)

    def get(self, principal, task_id):
        if task_id != self.current["id"]:
            return None
        if self.read_started and self.snapshots:
            self.current = self.snapshots.pop(0)
        self.read_started = True
        return copy.deepcopy(self.current)

    async def cancel(self, principal, task_id):
        self.cancel_calls.append((principal, task_id))
        self.current["status"] = "canceled"
        self.snapshots.clear()
        return copy.deepcopy(self.current)


def client_for(*snapshots):
    service = SequenceService(snapshots)
    return service, TestClient(Starlette(routes=routes(service)))


def request(*, text="Segment the liver", upload_id=None, task_id=None, **configuration):
    parts = [{"text": text}] if text is not None else []
    if upload_id is not None:
        parts.append({"data": {"upload_id": upload_id}})
    message = {"messageId": "request-one", "role": "ROLE_USER", "parts": parts}
    if task_id is not None:
        message.update(taskId=task_id, contextId="context-one")
    return {"message": message, "configuration": configuration}


def stream_events(response):
    assert response.status_code == 200
    values = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    for event in values:
        ParseDict(event, StreamResponse())
    return values


@pytest.mark.parametrize("path", ["message:send", "message:stream"])
@pytest.mark.parametrize("initially_waiting", [False, True])
def test_text_only_request_stops_on_input_required(path, initially_waiting):
    waiting = row("input_required", question="Please upload the image.")
    snapshots = [waiting] if initially_waiting else [row(), waiting]
    service, client = client_for(*snapshots)
    response = client.post(f"/a2a/v1/{path}", json=request(), headers=HEADERS)
    assert response.status_code == 200
    assert service.calls[0]["upload_id"] is None
    assert service.calls[0]["task_id"] is None
    if path == "message:send":
        task = response.json()["task"]
        ParseDict(task, Task())
        assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        assert task["metadata"]["inputExpiresAtUnix"] == waiting["input_expires_at"]
        assert task["status"]["message"]["parts"][0]["text"] == "Please upload the image."
    else:
        events = stream_events(response)
        status = events[-1].get("statusUpdate", events[-1].get("task"))["status"]
        assert status["state"] == "TASK_STATE_INPUT_REQUIRED"


@pytest.mark.parametrize("upload_id", [None, "upload-one"])
def test_continuation_forwards_original_task_context_and_optional_image(upload_id):
    service, client = client_for(row("completed"))
    payload = request(task_id="task-one", text="It is MR.", upload_id=upload_id)
    payload["message"]["messageId"] = "new-continuation-message"
    response = client.post("/a2a/v1/message:send", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert service.calls[0] == {
        "principal": "__public_a2a__",
        "upload_id": upload_id,
        "text": "It is MR.",
        "modality": None,
        "message_id": "new-continuation-message",
        "context_id": "context-one",
        "task_id": "task-one",
    }


def test_image_only_continuation_supplies_clear_text_but_empty_message_is_rejected():
    service, client = client_for(row("input_required"))
    payload = request(text=None, upload_id="upload-one", task_id="task-one")
    response = client.post("/a2a/v1/message:send", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert "supplied the image" in service.calls[0]["text"]
    response = client.post("/a2a/v1/message:send", json=request(text=None), headers=HEADERS)
    assert response.status_code == 400
    assert len(service.calls) == 1


@pytest.mark.parametrize("extra", [{"taskId": "../private"}, {"referenceTaskIds": ["task-one"]}])
def test_invalid_task_and_reference_tasks_never_reach_the_service(extra):
    service, client = client_for(row("input_required"))
    payload = request()
    payload["message"].update(extra)
    response = client.post("/a2a/v1/message:send", json=payload, headers=HEADERS)
    assert response.status_code == 400
    assert not service.calls


def test_input_required_can_be_canceled_and_subscription_returns_its_snapshot():
    service, client = client_for(row("input_required"))
    events = stream_events(client.get("/a2a/v1/tasks/task-one:subscribe", headers=HEADERS))
    assert len(events) == 1
    assert events[0]["task"]["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    response = client.post("/a2a/v1/tasks/task-one:cancel", json={}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["status"]["state"] == "TASK_STATE_CANCELED"
    assert len(service.cancel_calls) == 1


@pytest.mark.parametrize("limit,count", [(None, 20), (0, 0), (2, 2), (100, 25)])
def test_history_projection_preserves_standard_message_shape_and_latest_count(limit, count):
    source = row("input_required")
    source["a2a_history"] = history(25)
    task = project_task(source, URL, history_length=limit)
    assert len(task.history) == count
    if count:
        assert task.history[-1].message_id == "message-24"
        assert task.history[0].message_id == f"message-{25 - count}"
        assert MessageToDict(task.history[-1]) == source["a2a_history"][-1]


@pytest.mark.parametrize("limit", [0, 2])
@pytest.mark.parametrize("endpoint", ["send", "stream", "get", "subscribe"])
def test_history_length_applies_to_send_get_and_initial_stream_snapshot(limit, endpoint):
    _, client = client_for(row("input_required"))
    if endpoint in {"send", "stream"}:
        response = client.post(
            f"/a2a/v1/message:{endpoint}", json=request(historyLength=limit), headers=HEADERS
        )
    else:
        path = "/a2a/v1/tasks/task-one" + (":subscribe" if endpoint == "subscribe" else "")
        response = client.get(path, params={"historyLength": limit}, headers=HEADERS)
    assert response.status_code == 200
    task = (
        stream_events(response)[0]["task"]
        if endpoint in {"stream", "subscribe"}
        else response.json()["task"]
        if endpoint == "send"
        else response.json()
    )
    assert len(task.get("history", [])) == limit


def test_clarification_status_message_id_changes_only_when_question_changes():
    first = row("input_required", question="Please upload the image.")
    second = row("input_required", question="Is the image CT or MR?")
    first_message = project_task(first, URL).status.message
    assert first_message == project_task(copy.deepcopy(first), URL).status.message
    assert first_message.message_id != project_task(second, URL).status.message.message_id


def test_file_artifact_ids_survive_reordering_and_content_updates():
    liver, spleen = output(), output("spleen")
    first = project_task(row("running", files=[liver, spleen]), URL)
    changed = project_task(row("running", files=[spleen, output(checksum="b")]), URL)
    first_ids = {artifact.name: artifact.artifact_id for artifact in first.artifacts}
    changed_ids = {artifact.name: artifact.artifact_id for artifact in changed.artifacts}
    assert first_ids == changed_ids
    assert first_ids["liver.nii.gz"] != first_ids["spleen.nii.gz"]
    assert first_ids["liver.nii.gz"].startswith("file-")


def test_stream_publishes_running_changed_and_failed_partial_artifacts_once():
    initial_liver, changed_liver, spleen = output(), output(checksum="b"), output("spleen")
    service, client = client_for(
        row(),
        row("running", files=[initial_liver]),
        row("running", files=[initial_liver]),
        row("running", files=[changed_liver]),
        row("failed", files=[spleen, changed_liver]),
    )
    events = stream_events(
        client.post("/a2a/v1/message:stream", json=request(upload_id="upload-one"), headers=HEADERS)
    )
    updates = [event["artifactUpdate"] for event in events if "artifactUpdate" in event]
    assert [value["artifact"]["name"] for value in updates] == [
        "Segmentation summary",
        "liver.nii.gz",
        "liver.nii.gz",
        "spleen.nii.gz",
    ]
    liver_updates = [
        value["artifact"] for value in updates if value["artifact"]["name"] == "liver.nii.gz"
    ]
    assert liver_updates[0]["artifactId"] == liver_updates[1]["artifactId"]
    assert [value["metadata"]["sha256"] for value in liver_updates] == ["a" * 64, "b" * 64]
    assert all(value["append"] is False and value["lastChunk"] is True for value in updates)
    assert events[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_FAILED"
    assert service.current["status"] == "failed"


@pytest.mark.parametrize("state", ["failed", "canceled", "input_required"])
def test_partial_artifacts_precede_noncompleted_final_status(state):
    _, client = client_for(row(), row(state, files=[output()]))
    events = stream_events(
        client.post("/a2a/v1/message:stream", json=request(upload_id="upload-one"), headers=HEADERS)
    )
    assert any("artifactUpdate" in event for event in events[:-1])
    assert events[-1]["statusUpdate"]["status"]["state"] == f"TASK_STATE_{state.upper()}"


def test_initial_snapshot_artifacts_are_not_repeated_in_later_updates():
    _, client = client_for(row("running", files=[output()]), row("failed", files=[output()]))
    events = stream_events(
        client.get("/a2a/v1/tasks/task-one:subscribe?historyLength=0", headers=HEADERS)
    )
    assert events[0]["task"]["artifacts"]
    assert not any("artifactUpdate" in event for event in events)
    assert events[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_FAILED"


def test_card_explains_missing_image_continuation_history_and_stream_recovery():
    description = build_agent_card(URL).skills[0].description
    for phrase in (
        "TASK_STATE_INPUT_REQUIRED",
        "taskId and contextId",
        "new messageId",
        "24 hours",
        "inputExpiresAtUnix",
        "historyLength",
        "always GET the Task",
        "not external URLs or inline raw/base64 files",
    ):
        assert phrase in description

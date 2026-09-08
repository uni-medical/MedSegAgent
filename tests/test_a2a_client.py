"""Image-aware external client contracts; no model, network or platform acceptance."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from a2a.types import SendMessageRequest, Task
from google.protobuf.json_format import ParseDict

MODULE = Path(__file__).parents[1] / "ops" / "a2a_client.py"
spec = importlib.util.spec_from_file_location("image_a2a_client", MODULE)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

BASE = "https://medseg.example.org"


def task(state="TASK_STATE_COMPLETED", **fields):
    value = {"id": "task-1", "contextId": "context-1", "status": {"state": state}, **fields}
    ParseDict(value, Task())
    return value


def frame(value):
    return ("data: " + json.dumps(value) + "\n\n").encode()


class BrokenStream(httpx.SyncByteStream):
    def __init__(self, initial=b""):
        self.initial = initial

    def __iter__(self):
        if self.initial:
            yield self.initial
        raise httpx.ReadError("connection lost")


def test_upload_stream_and_authoritative_get_task_without_downloading_artifacts(tmp_path):
    source = tmp_path / "example.nii.gz"
    source.write_bytes(b"supplied-nifti-bytes")
    artifacts = [
        {
            "artifactId": "mask-1",
            "name": "Liver",
            "parts": [
                {"url": "https://untrusted.example/mask.nii.gz", "mediaType": "application/gzip"}
            ],
        }
    ]
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "medseg.example.org"
        if request.url.path == "/a2a/uploads":
            assert request.content == source.read_bytes()
            assert request.headers["x-filename"] == source.name
            assert request.headers["content-type"] == "application/octet-stream"
            return httpx.Response(201, json={"id": "upload-1"})
        assert request.headers["A2A-Version"] == "1.0"
        if request.url.path.endswith("message:stream"):
            body = json.loads(request.content)
            ParseDict(body, SendMessageRequest())
            assert body["message"]["parts"][1]["data"] == {
                "upload_id": "upload-1",
                "modality": "CT",
            }
            assert body["message"]["messageId"] == "message-1"
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=b"".join(
                    [
                        frame({"task": task("TASK_STATE_SUBMITTED")}),
                        frame(
                            {
                                "artifactUpdate": {
                                    "taskId": "task-1",
                                    "contextId": "context-1",
                                    "artifact": artifacts[0],
                                    "lastChunk": True,
                                }
                            }
                        ),
                        frame(
                            {
                                "statusUpdate": {
                                    "taskId": "task-1",
                                    "contextId": "context-1",
                                    "status": {"state": "TASK_STATE_COMPLETED"},
                                }
                            }
                        ),
                    ]
                ),
            )
        assert request.url.path == "/a2a/v1/tasks/task-1"
        return httpx.Response(200, json=task(artifacts=artifacts, metadata={"authoritative": True}))

    result = client.run(
        url=BASE + "/a2a/v1",
        image=source,
        prompt="Segment liver",
        modality="CT",
        message_id="message-1",
        transport=httpx.MockTransport(handler),
    )
    assert result["metadata"]["authoritative"] is True
    assert client.report(result)["artifacts"] == artifacts
    assert [r.method for r in requests] == ["POST", "POST", "GET"]


def test_stream_disconnect_after_task_frame_recovers_without_resubmitting():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("message:stream"):
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                stream=BrokenStream(frame({"task": task("TASK_STATE_WORKING")})),
            )
        return httpx.Response(200, json=task())

    result = client.run(
        url=BASE,
        upload_id="upload-1",
        prompt="Segment liver",
        transport=httpx.MockTransport(handler),
    )
    assert result["status"]["state"] == "TASK_STATE_COMPLETED"
    assert paths == ["/a2a/v1/message:stream", "/a2a/v1/tasks/task-1"]


def test_lost_admission_frame_replays_original_message_without_inserting_context():
    bodies = []

    def handler(request):
        if request.method == "POST":
            bodies.append(json.loads(request.content))
        if request.url.path.endswith("message:stream"):
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, stream=BrokenStream()
            )
        if request.url.path.endswith("message:send"):
            return httpx.Response(200, json={"task": task("TASK_STATE_WORKING")})
        return httpx.Response(200, json=task())

    client.run(
        url=BASE,
        upload_id="upload-1",
        prompt="Segment liver",
        message_id="stable-id",
        transport=httpx.MockTransport(handler),
    )
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert "contextId" not in bodies[1]["message"]
    assert bodies[1]["message"]["messageId"] == "stable-id"


def test_input_required_returns_continuation_and_text_only_resume_reuses_task_context():
    observed = []
    waiting = task(
        "TASK_STATE_INPUT_REQUIRED",
        status={
            "state": "TASK_STATE_INPUT_REQUIRED",
            "message": {
                "messageId": "clarification",
                "role": "ROLE_AGENT",
                "parts": [{"text": "Which organ?"}],
            },
        },
    )

    def handler(request):
        observed.append(request)
        if request.method == "POST":
            body = json.loads(request.content)
            ParseDict(body, SendMessageRequest())
            assert body["message"]["taskId"] == "task-1"
            assert body["message"]["contextId"] == "context-1"
            assert body["message"]["parts"] == [{"text": "Liver", "mediaType": "text/plain"}]
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=frame({"task": waiting})
            )
        return httpx.Response(200, json=waiting)

    result = client.run(
        url=BASE,
        prompt="Liver",
        task_id="task-1",
        context_id="context-1",
        transport=httpx.MockTransport(handler),
    )
    assert client.report(result)["continuation"]["task_id"] == "task-1"
    assert len(observed) == 2 and all("uploads" not in str(r.url) for r in observed)


def test_continued_context_can_omit_image_and_supply_modality():
    def handler(request):
        if request.method == "POST":
            message = json.loads(request.content)["message"]
            assert message["contextId"] == "context-1" and "taskId" not in message
            assert message["parts"][1]["data"] == {"modality": "MR"}
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=frame({"task": task()})
            )
        return httpx.Response(200, json=task())

    client.run(
        url=BASE,
        prompt="Segment spleen too",
        context_id="context-1",
        modality="MR",
        transport=httpx.MockTransport(handler),
    )


def test_typed_admission_error_is_reported_and_never_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": "Unknown upload",
                    "details": [{"reason": "INVALID_PARAMS"}],
                }
            },
        )

    with pytest.raises(client.ClientError, match="HTTP 400 INVALID_PARAMS: Unknown upload"):
        client.run(
            url=BASE,
            prompt="Segment liver",
            upload_id="upload-1",
            transport=httpx.MockTransport(handler),
        )
    assert len(calls) == 1


def test_poll_retries_transient_response_and_returns_failed_task_without_claiming_success(
    monkeypatch,
):
    monkeypatch.setattr(client.time, "sleep", lambda _seconds: None)
    replies = iter(
        [
            httpx.Response(503, text="temporarily unavailable"),
            httpx.Response(200, json=task("TASK_STATE_WORKING")),
            httpx.Response(200, json=task("TASK_STATE_FAILED")),
        ]
    )
    result = client.run(
        url=BASE, task_id="task-1", transport=httpx.MockTransport(lambda _r: next(replies))
    )
    assert result["status"]["state"] == "TASK_STATE_FAILED"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"task": task()}),  # GetTask is a bare Task.
        httpx.Response(200, json=task(id="different-task")),
    ],
)
def test_invalid_get_task_responses_fail_explicitly(response):
    with pytest.raises(client.ClientError):
        client.run(url=BASE, task_id="task-1", transport=httpx.MockTransport(lambda _r: response))


def test_poll_deadline_leaves_recovery_id_and_does_not_cancel(monkeypatch):
    seconds = [0]
    monkeypatch.setattr(client.time, "monotonic", lambda: seconds[0])
    monkeypatch.setattr(
        client.time, "sleep", lambda value: seconds.__setitem__(0, seconds[0] + value)
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=task("TASK_STATE_WORKING"))

    with pytest.raises(client.ClientError, match="remote work is not canceled.*task-1"):
        client.run(url=BASE, task_id="task-1", timeout=0.1, transport=httpx.MockTransport(handler))
    assert all(r.method == "GET" for r in requests)


@pytest.mark.parametrize(
    "state,expected",
    [
        ("TASK_STATE_COMPLETED", 0),
        ("TASK_STATE_FAILED", 2),
        ("TASK_STATE_CANCELED", 2),
        ("TASK_STATE_INPUT_REQUIRED", 3),
    ],
)
def test_cli_exit_status_distinguishes_completion_from_pause_and_failure(
    monkeypatch, capsys, state, expected
):
    monkeypatch.setattr(client, "run", lambda **_kwargs: task(state))
    assert client.main(["--url", BASE, "--task-id", "task-1"]) == expected
    assert json.loads(capsys.readouterr().out)["task"]["status"]["state"] == state


def test_first_request_requires_image_or_uploaded_reference():
    with pytest.raises(client.ClientError, match="first request needs"):
        client.run(url=BASE, prompt="Segment liver")


def test_poll_respects_retry_after_within_deadline(monkeypatch):
    delays = []
    monkeypatch.setattr(client.time, "sleep", delays.append)
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "8"}),
            httpx.Response(200, json=task()),
        ]
    )
    client.run(
        url=BASE, task_id="task-1", transport=httpx.MockTransport(lambda _r: next(responses))
    )
    assert delays == [8]


def test_invalid_sse_update_returns_readable_error():
    response = httpx.Response(
        200, headers={"Content-Type": "text/event-stream"}, content=frame({"statusUpdate": None})
    )
    with pytest.raises(client.ClientError, match="Invalid A2A statusUpdate"):
        client.run(
            url=BASE,
            prompt="Segment liver",
            upload_id="upload-1",
            transport=httpx.MockTransport(lambda _r: response),
        )


def test_client_runs_upload_clarification_and_same_task_resume_against_local_service(
    tmp_path, monkeypatch
):
    from starlette.testclient import TestClient
    from test_execution import Backend, make_input

    from medsegagent import agent, core, web

    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    turns = []

    async def run_agent(text, modality, execution, **kwargs):
        turns.append(text)
        if len(turns) == 1:
            return {
                "status": "needs_input",
                "summary": "Please confirm the organ.",
                "unresolved": ["organ confirmation"],
            }
        result = await execution.call("segment", {"targets": ["liver"], "modality": "CT"})
        assert result["ok"], result
        return {"status": "completed", "summary": "Liver segmented.", "unresolved": []}

    monkeypatch.setattr(agent, "run_agent", run_agent)
    source = make_input(tmp_path)
    observed = []
    with TestClient(web.create_app(tmp_path / "server", "http://localhost")) as server:

        def bridge(request):
            observed.append((request.method, request.url.path))
            response = server.request(
                request.method,
                request.url.path,
                headers=dict(request.headers),
                content=request.read(),
            )
            return httpx.Response(
                response.status_code, headers=response.headers, content=response.content
            )

        transport = httpx.MockTransport(bridge)
        waiting = client.run(
            url="http://localhost",
            image=source,
            prompt="Segment the liver?",
            modality="CT",
            transport=transport,
        )
        assert waiting["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        completed = client.run(
            url="http://localhost",
            task_id=waiting["id"],
            context_id=waiting["contextId"],
            prompt="Yes, the liver.",
            transport=transport,
        )
    assert completed["id"] == waiting["id"]
    assert completed["contextId"] == waiting["contextId"]
    assert completed["status"]["state"] == "TASK_STATE_COMPLETED"
    assert completed["artifacts"]
    assert len(backend.calls) == 1
    assert observed.count(("POST", "/a2a/uploads")) == 1

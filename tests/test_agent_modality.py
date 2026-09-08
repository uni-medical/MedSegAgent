"""The feedback loop owns modality choice and can act on advisory observations."""

import asyncio
import json

import httpx
import pytest
from test_execution import Backend, make_input

from medsegagent import agent, core, modality
from medsegagent.execution import TaskExecution


def report(value="CT", status="detected"):
    return {
        "modality": value if status == "detected" else None,
        "source": "totalseg_intensity",
        "status": status,
        "supported": status == "detected",
        **({"candidate_modality": value, "vote_agreement": 1.0} if status == "detected" else {}),
        "limitations": ["CT_MR_ONLY", "UNCALIBRATED_VOTE"],
    }


def tool(name, args):
    return {
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }
        ]
    }


@pytest.mark.parametrize(
    "identified,request_text",
    [
        ("CT", "请分割肝脏。"),
        ("MR", "请分割肝脏。"),
        ("CT", "分割肝脏，不要使用 MR 模型。"),
        ("CT", "分割肝脏，患者此前做过MRI。"),
    ],
)
def test_agent_detects_then_selects_modality_and_binds_schema(
    tmp_path, monkeypatch, identified, request_text
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    classifier_calls = []
    monkeypatch.setattr(
        modality,
        "classify_features",
        lambda features: classifier_calls.append(features) or report(identified),
    )
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), None, tmp_path / "outputs")
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert json.loads(payload["messages"][1]["content"]) == {
            "modality": None,
            "request": request_text,
        }
        if len(requests) == 1:
            assert not backend.calls
            assert not execution.has_outputs
            assert execution.modality is None
            schema = next(
                t["function"] for t in payload["tools"] if t["function"]["name"] == "segment"
            )
            assert "modality" in schema["parameters"]["required"]
            assert schema["parameters"]["properties"]["modality"]["enum"] == ["CT", "MR"]
            message = tool("detect_modality", {})
        elif len(requests) == 2:
            feedback = json.loads(payload["messages"][-1]["content"])
            assert feedback["modality_detection"]["modality"] == identified
            assert feedback["modality_detection"]["vote_agreement"] == 1.0
            schema = next(
                t["function"] for t in payload["tools"] if t["function"]["name"] == "segment"
            )
            assert execution.modality is None
            assert schema["parameters"]["properties"]["modality"]["enum"] == ["CT", "MR"]
            message = tool("segment", {"targets": ["liver"], "modality": identified})
        else:
            schema = next(
                t["function"] for t in payload["tools"] if t["function"]["name"] == "segment"
            )
            tasks = schema["parameters"]["properties"]["task"]["anyOf"][0]["enum"]
            assert ("lung_nodules" in tasks) == (identified == "CT")
            message = {
                "content": json.dumps(
                    {"status": "completed", "summary": "已识别模态并分割肝脏。", "unresolved": []}
                )
            }
        return httpx.Response(200, json={"choices": [{"message": message}]})

    result = asyncio.run(
        agent.run_agent(
            request_text,
            None,
            execution,
            transport=httpx.MockTransport(handler),
        )
    )
    assert result["status"] == "completed"
    assert result["tool_calls"] == 2 and len(classifier_calls) == 1
    assert backend.calls[0]["task"] == ("total" if identified == "CT" else "total_mr")
    serialized = json.dumps(requests)
    assert str(tmp_path) not in serialized and "synthetic.nii" not in serialized
    assert "intensity_features" not in serialized and "PatientID" not in serialized


def test_agent_can_choose_mr_after_intensity_abstention(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    detected = {
        **report("CT", "uncertain"),
        "intensity_statistics": {"mean": 193.7, "std": 193.8, "min": -47.0, "max": 833.0},
    }
    monkeypatch.setattr(modality, "classify_features", lambda features: detected)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), None, tmp_path / "outputs")
    messages = iter(
        [
            tool("detect_modality", {}),
            tool("segment", {"targets": ["liver"], "modality": "MR"}),
            {"content": json.dumps({"status": "completed", "summary": "Done", "unresolved": []})},
        ]
    )
    observations = []

    def provider(request):
        payload = json.loads(request.content)
        if payload["messages"][-1]["role"] == "tool":
            observations.append(json.loads(payload["messages"][-1]["content"]))
        return httpx.Response(200, json={"choices": [{"message": next(messages)}]})

    transport = httpx.MockTransport(provider)
    result = asyncio.run(agent.run_agent("分割肝脏", None, execution, transport=transport))
    assert result["status"] == "completed"
    assert backend.calls[0]["task"] == "total_mr"
    assert execution.modality == "MR" and not execution.unresolved_failures
    assert observations[0]["modality_detection"]["intensity_statistics"]["min"] == -47
    assert "candidate_modality" not in observations[0]["modality_detection"]
    assert "vote_agreement" not in observations[0]["modality_detection"]


@pytest.mark.parametrize(
    "declared,text",
    [
        ("CT", "分割这份CT中的肝脏"),
        ("MR", "分割这份MR中的肝脏"),
        ("CT", "这份是CT，不是MR，分割肝脏"),
        ("CT", "用CT分割肝脏，不要使用MR模型"),
        ("CT", "分割这份CT肝脏，患者以前做过MRI"),
        ("MR", "这份是MR，曾经做过CT，分割肝脏"),
        ("CT", "MRI liver"),
        ("MR", "CT肝脏"),
    ],
)
def test_user_declaration_needs_only_segment(tmp_path, monkeypatch, declared, text):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")

    def forbidden(*args):
        pytest.fail("A declared modality must not invoke detection")

    monkeypatch.setattr(modality, "classify_features", forbidden)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), declared, tmp_path / "outputs")
    messages = iter(
        [
            tool("segment", {"targets": ["liver"]}),
            {"content": json.dumps({"status": "completed", "summary": "Done", "unresolved": []})},
        ]
    )
    requests = []

    def provider(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert json.loads(payload["messages"][1]["content"]) == {
            "modality": declared,
            "request": text,
        }
        schema = next(t["function"] for t in payload["tools"] if t["function"]["name"] == "segment")
        tasks = schema["parameters"]["properties"]["task"]["anyOf"][0]["enum"]
        assert ("total" in tasks) == (declared == "CT")
        assert ("total_mr" in tasks) == (declared == "MR")
        return httpx.Response(200, json={"choices": [{"message": next(messages)}]})

    result = asyncio.run(
        agent.run_agent(text, declared, execution, transport=httpx.MockTransport(provider))
    )
    assert result["status"] == "completed" and result["tool_calls"] == 1
    assert len(requests) == 2
    assert execution.modality_detection is None
    assert backend.calls[0]["task"] == ("total" if declared == "CT" else "total_mr")


@pytest.mark.parametrize(
    "text,chosen,hint",
    [
        ("分割这份CT中的肝脏", "CT", None),
        ("分割这份MR中的肝脏", "MR", None),
        ("分割这份CT中的肝脏", "CT", "MR"),
        ("分割肝脏", "MR", "MR"),
        ("分割肝脏，不要使用 MR 模型", "CT", "CT"),
        ("分割肝脏，患者此前做过MRI", "CT", "CT"),
    ],
)
def test_agent_chooses_from_text_and_example_hint_without_forced_detection(
    tmp_path, monkeypatch, text, chosen, hint
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")

    def forbidden(*args):
        pytest.fail("An evidence-supported choice must not force classification")

    monkeypatch.setattr(modality, "classify_features", forbidden)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(
        make_input(tmp_path), None, tmp_path / "outputs", example_modality_hint=hint
    )
    requests = []

    def provider(request):
        payload = json.loads(request.content)
        requests.append(payload)
        context = {"modality": None, "request": text}
        if hint is not None:
            context["modality_hint"] = {"modality": hint, "source": "example_manifest"}
        assert json.loads(payload["messages"][1]["content"]) == context
        schema = next(t["function"] for t in payload["tools"] if t["function"]["name"] == "segment")
        properties = schema["parameters"]["properties"]
        if len(requests) == 1:
            assert execution.modality is None and not backend.calls
            assert properties["modality"]["enum"] == ["CT", "MR"]
            assert "modality" in schema["parameters"]["required"]
            message = tool("segment", {"targets": ["liver"], "modality": chosen})
        else:
            assert execution.modality == chosen
            assert "modality" not in properties
            assert ("total" in properties["task"]["anyOf"][0]["enum"]) == (chosen == "CT")
            assert ("total_mr" in properties["task"]["anyOf"][0]["enum"]) == (chosen == "MR")
            message = {
                "content": json.dumps({"status": "completed", "summary": "Done", "unresolved": []})
            }
        return httpx.Response(200, json={"choices": [{"message": message}]})

    outcome = asyncio.run(
        agent.run_agent(text, None, execution, transport=httpx.MockTransport(provider))
    )
    assert outcome["status"] == "completed" and outcome["tool_calls"] == 1
    assert len(requests) == 2 and len(backend.calls) == 1
    assert execution.modality_detection is None
    assert backend.calls[0]["task"] == ("total" if chosen == "CT" else "total_mr")


def test_web_autodetection_persists_and_a2a_replays_the_resolved_modality(tmp_path, monkeypatch):
    import time

    import nibabel as nib
    import numpy as np
    from auth_helpers import ALICE
    from auth_helpers import create_test_app as create_app
    from starlette.testclient import TestClient

    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    counts = []
    monkeypatch.setattr(
        modality, "classify_features", lambda features: counts.append(1) or report("MR")
    )
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    requests = []
    messages = iter(
        [
            tool("detect_modality", {}),
            tool("segment", {"targets": ["liver"], "modality": "MR"}),
            {
                "content": json.dumps(
                    {"status": "completed", "summary": "识别为MR，肝脏已分割。", "unresolved": []}
                )
            },
        ]
    )

    def provider(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": next(messages)}]})

    original = agent.run_agent

    async def run(*args, **kwargs):
        return await original(*args, **kwargs, transport=httpx.MockTransport(provider))

    monkeypatch.setattr(agent, "run_agent", run)
    headers = ALICE
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    with TestClient(create_app(tmp_path, "http://localhost")) as client:
        upload = client.post(
            "/api/uploads", headers={**headers, "X-Filename": "image.nii"}, content=image.to_bytes()
        ).json()
        body = {
            "upload_id": upload["id"],
            "text": "请分割肝脏。",
            "message_id": "automatic-modality",
        }
        admitted = client.post("/api/tasks", headers=headers, json=body)
        assert admitted.status_code == 202, admitted.text
        task_id = admitted.json()["id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            row = client.get(f"/api/tasks/{task_id}", headers=headers).json()
            if row["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert row["status"] == "completed", row
        assert row["modality"] == "MR" and row["modality_source"] == "agent"
        assert row["result"]["modality_detection"]["modality"] == "MR"
        assert client.post("/api/tasks", headers=headers, json=body).json()["id"] == task_id
        a2a = client.get(f"/a2a/v1/tasks/{task_id}", headers={**headers, "A2A-Version": "1.0"})
        assert a2a.status_code == 200, a2a.text
        assert a2a.json()["metadata"]["segmentation"]["modality_detection"]["modality"] == "MR"
    assert len(counts) == 1 and len(backend.calls) == 1
    assert backend.calls[0]["task"] == "total_mr"
    assert str(tmp_path) not in json.dumps(requests)


def test_malformed_final_does_not_invent_a_modality_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    monkeypatch.setattr(modality, "classify_features", lambda features: report("MR"))
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "outputs")
    messages = iter(
        [tool("detect_modality", {}), {"content": "not JSON"}, {"content": "still not JSON"}]
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": next(messages)}]})
    )
    result = asyncio.run(
        agent.run_agent("请核对CT模态再分割肝脏", "CT", execution, transport=transport)
    )
    assert result["status"] == "failed"
    assert "冲突" not in json.dumps(result, ensure_ascii=False)
    assert not execution.has_outputs

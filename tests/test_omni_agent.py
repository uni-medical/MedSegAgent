"""Ordinary provider tool-loop tests; the fake service performs no model inference."""

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest

from medsegagent import agent, omni_agent

BASE_ID = "b" * 32
NEW_ID = "a" * 32


def call(name, arguments=None):
    return {
        "tool_calls": [
            {
                "id": "provider_call",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments or {})},
            }
        ]
    }


def finish(status="completed", summary="已完成记录标记的处理。", unresolved=None):
    return call("finish", {"status": status, "summary": summary, "unresolved": unresolved or []})


class FakeService:
    def __init__(self):
        self.calls = []
        self.revision = {
            "id": NEW_ID,
            "parent_revision": BASE_ID,
            "voxel_count": 125,
            "volume_mm3": 500.0,
            "volume_ml": 0.5,
            "name": "Private patient name",
            "source_sha256": "private image digest",
            "prompts": [{"world": [9123.25, 4567.25, 8989.25]}],
            "private_path": "/sensitive/subject/mask.nii.gz",
        }
        self.parent_revision = {
            **copy.deepcopy(self.revision),
            "id": BASE_ID,
            "parent_revision": None,
            "voxel_count": 80,
            "volume_mm3": 320.0,
            "volume_ml": 0.32,
        }
        self.operation_error = None
        self.inspect_error = None

    async def run_operation(self, workspace, operation, body, jobdir):
        self.calls.append(("apply", copy.deepcopy(body)))
        if self.operation_error is not None:
            raise self.operation_error
        return {"revision": copy.deepcopy(self.revision)}

    def refresh_workspace(self, workspace):
        return {**workspace, "latest_revision": NEW_ID}

    async def inspect_revision(self, workspace, revision_id=None):
        self.calls.append(("inspect", revision_id))
        if self.inspect_error is not None:
            raise self.inspect_error
        revision = (
            None
            if revision_id is None
            else copy.deepcopy(self.parent_revision if revision_id == BASE_ID else self.revision)
        )
        if revision is not None:
            revision["id"] = revision_id
        return {"revision": revision, "geometry": copy.deepcopy(workspace["geometry"])}


@pytest.fixture
def setup_case(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://planner.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    workspace = {
        "id": "w" * 32,
        "latest_revision": BASE_ID,
        "source_file": "/sensitive/patient/original.nii.gz",
        "_principal": "secret owner",
        "geometry": {
            "shape": [5, 7, 11],
            "spacing": [0.7, 1.3, 4.1],
            "unit": "mm",
            "affine": [[1234.987, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "header": "private header",
            "pixels": [787878.125],
        },
    }
    body = {
        "instruction": "使用已记录的标记修正并检查体积 /sensitive/patient/original.nii.gz",
        "base_revision": BASE_ID,
        "expected_revision": BASE_ID,
        "name": "Private patient name",
        "prompts": [{"kind": "point", "world": [9123.25, 4567.25, 8989.25], "positive": True}],
    }
    return workspace, body, FakeService()


def provider(messages, requests):
    queue = iter(messages)

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={"choices": [{"message": next(queue)}]})

    return httpx.MockTransport(handle)


def run(case, tmp_path, messages):
    workspace, body, service = case
    requests = []
    result = asyncio.run(
        omni_agent.run_agent(
            service, workspace, body, tmp_path, transport=provider(messages, requests)
        )
    )
    return result, requests


def test_real_native_tool_loop_applies_host_marks_inspects_and_finishes(setup_case, tmp_path):
    workspace, body, service = setup_case
    original_body = copy.deepcopy(body)
    result, requests = run(
        setup_case, tmp_path, [call("apply_recorded_prompts"), call("inspect_revision"), finish()]
    )
    assert result["status"] == "completed"
    assert result["revision"]["id"] == NEW_ID
    assert result["model_requests"] == 3
    assert service.calls[0] == (
        "apply",
        {
            key: original_body[key]
            for key in ("base_revision", "expected_revision", "prompts", "name")
        },
    )
    assert service.calls[1:] == [
        ("inspect", BASE_ID),
        ("inspect", NEW_ID),
        ("inspect", BASE_ID),
    ]
    assert body == original_body
    assert workspace["latest_revision"] == BASE_ID
    assert Path(tmp_path, result["inspect_artifact"]).is_file()
    observed = json.loads(Path(tmp_path, result["inspect_artifact"]).read_text())
    assert observed["revision"]["voxel_count"] == 125
    assert json.loads((tmp_path / "agent-result.json").read_text())["status"] == "completed"

    transmitted = json.dumps(requests)
    for forbidden in [
        "/sensitive/",
        "private header",
        "secret owner",
        "private image digest",
        "Private patient name",
        "1234.987",
        "787878.125",
        "9123.25",
        "4567.25",
        "8989.25",
        '"world"',
        '"affine"',
        '"source_file"',
    ]:
        assert forbidden not in transmitted
    assert "0.5" in transmitted
    schemas = requests[0]["tools"]
    assert {tool["function"]["name"] for tool in schemas} == {
        "apply_recorded_prompts",
        "inspect_revision",
        "finish",
    }
    assert schemas[0]["function"]["parameters"]["properties"] == {}
    assert requests[0]["parallel_tool_calls"] is False


def test_premature_completion_requires_actual_edit_not_provider_assertion(setup_case, tmp_path):
    result, requests = run(
        setup_case,
        tmp_path,
        [finish(summary="我已分割完成。"), call("apply_recorded_prompts"), finish()],
    )
    assert result["status"] == "completed"
    rejected = requests[1]["messages"][-1]
    assert json.loads(rejected["content"])["code"] == "NOT_COMPLETE"
    assert [item[0] for item in setup_case[2].calls] == ["apply", "inspect"]


def test_mutation_can_only_be_attempted_once(setup_case, tmp_path):
    result, requests = run(
        setup_case,
        tmp_path,
        [call("apply_recorded_prompts"), call("apply_recorded_prompts"), finish()],
    )
    assert result["status"] == "completed"
    assert [item[0] for item in setup_case[2].calls] == ["apply", "inspect"]
    assert "ALREADY_ATTEMPTED" in json.dumps(requests)


def test_coordinates_from_provider_cannot_replace_host_recorded_marks(setup_case, tmp_path):
    result, requests = run(
        setup_case,
        tmp_path,
        [
            call("apply_recorded_prompts", {"voxel": [0, 0, 0]}),
            call("apply_recorded_prompts"),
            finish(),
        ],
    )
    assert result["status"] == "completed"
    assert len(setup_case[2].calls) == 2
    assert setup_case[2].calls[0][1]["prompts"] == setup_case[1]["prompts"]
    assert '"voxel"' not in json.dumps(requests)


def test_inspection_of_existing_revision_does_not_satisfy_unapplied_marks(setup_case, tmp_path):
    result, _ = run(setup_case, tmp_path, [call("inspect_revision"), *[finish()] * 5])
    assert result["status"] == "failed"
    assert result["model_requests"] == omni_agent.MAX_ROUNDS == 6
    assert [item[0] for item in setup_case[2].calls] == ["inspect"]
    assert result["revision"]["id"] == BASE_ID


def test_measurement_only_task_completes_with_real_inspection_artifact(setup_case, tmp_path):
    setup_case[1].update(prompts=[], instruction="读取当前区域的体积。")
    result, _ = run(setup_case, tmp_path, [call("inspect_revision"), finish()])
    assert result["status"] == "completed"
    assert result["revision"]["id"] == BASE_ID
    assert Path(tmp_path, result["inspect_artifact"]).is_file()
    assert setup_case[2].calls == [("inspect", BASE_ID)]


def test_shape_only_inspection_has_artifact_but_no_invented_revision(setup_case, tmp_path):
    setup_case[0]["latest_revision"] = None
    setup_case[1].update(
        prompts=[],
        base_revision=None,
        expected_revision=None,
        instruction="当前图像的体素尺寸是多少？",
    )
    result, _ = run(
        setup_case, tmp_path, [call("inspect_revision"), finish(summary="图像数组为5×7×11。")]
    )
    assert result["status"] == "completed"
    assert "revision" not in result
    assert json.loads(Path(tmp_path, result["inspect_artifact"]).read_text())["shape"] == [5, 7, 11]


def test_failed_edit_cannot_be_hidden_by_inspection_or_completion(setup_case, tmp_path):
    setup_case[2].operation_error = RuntimeError("/sensitive/model/path failed with private header")
    result, requests = run(
        setup_case,
        tmp_path,
        [
            call("apply_recorded_prompts"),
            call("apply_recorded_prompts"),
            call("inspect_revision"),
            finish(),
            finish(),
            finish(),
        ],
    )
    assert result["status"] == "failed"
    assert [item[0] for item in setup_case[2].calls] == ["apply", "inspect"]
    assert "/sensitive/" not in json.dumps(requests)
    assert "private header" not in json.dumps(requests)


def test_no_marks_requests_input_and_never_runs_a_model(setup_case, tmp_path):
    setup_case[1]["prompts"] = []
    result, _ = run(
        setup_case,
        tmp_path,
        [call("apply_recorded_prompts"), finish("needs_input", "请在目标上添加一个标记。")],
    )
    assert result["status"] == "needs_input"
    assert setup_case[2].calls == []
    assert "revision" not in result


def test_plain_text_or_multiple_calls_never_run_tools_or_claim_completion(setup_case, tmp_path):
    doubled = call("apply_recorded_prompts")
    doubled["tool_calls"] *= 2
    result, _ = run(
        setup_case, tmp_path, [{"content": "已经完成"}, doubled, *[{"tool_calls": []}] * 4]
    )
    assert result["status"] == "failed"
    assert result["model_requests"] == 6
    assert setup_case[2].calls == []


def test_cancellation_propagates_from_local_operation(setup_case, tmp_path):
    setup_case[2].operation_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        run(setup_case, tmp_path, [call("apply_recorded_prompts")])
    assert not (tmp_path / "agent-result.json").exists()
    assert json.loads((tmp_path / "agent-trace.json").read_text())["status"] == "canceled"


def test_cancellation_propagates_from_provider(setup_case, tmp_path, monkeypatch):
    async def canceled(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(agent, "_provider_message", canceled)
    with pytest.raises(asyncio.CancelledError):
        run(setup_case, tmp_path, [])
    assert setup_case[2].calls == []


def test_missing_revision_in_tool_result_cannot_establish_completion(setup_case, tmp_path):
    setup_case[2].revision = {}
    result, _ = run(setup_case, tmp_path, [call("apply_recorded_prompts"), *[finish()] * 5])
    assert result["status"] == "failed"
    assert "revision" not in result


def test_unknown_volume_units_remain_null_not_zero(setup_case, tmp_path):
    setup_case[2].revision.update(volume_mm3=None, volume_ml=None)
    result, requests = run(setup_case, tmp_path, [call("apply_recorded_prompts"), finish()])
    assert result["status"] == "completed"
    feedback = json.loads(requests[1]["messages"][-1]["content"])
    assert feedback["revision"]["volume_ml"] is None


@pytest.mark.parametrize(
    ("current_ml", "parent_ml", "expected_delta"),
    [(0.5, 0.32, 0.18), (None, 0.32, None), (0.5, None, None), (None, None, None)],
)
def test_apply_then_inspect_returns_verified_parent_and_null_safe_deltas(
    setup_case, tmp_path, current_ml, parent_ml, expected_delta
):
    service = setup_case[2]
    service.revision.update(
        volume_ml=current_ml, volume_mm3=None if current_ml is None else current_ml * 1000
    )
    service.parent_revision.update(
        volume_ml=parent_ml, volume_mm3=None if parent_ml is None else parent_ml * 1000
    )
    result, requests = run(
        setup_case, tmp_path, [call("apply_recorded_prompts"), call("inspect_revision"), finish()]
    )
    assert result["status"] == "completed"
    for payload in requests[1:]:
        feedback = json.loads(payload["messages"][-1]["content"])
        assert feedback["revision"]["id"] == NEW_ID
        assert feedback["parent_revision"] == {
            "id": BASE_ID,
            "parent_revision": None,
            "voxel_count": 80,
            "volume_ml": parent_ml,
            "volume_mm3": None if parent_ml is None else parent_ml * 1000,
        }
        assert feedback["delta_voxels"] == 45
        if expected_delta is None:
            assert feedback["delta_ml"] is None
        else:
            assert feedback["delta_ml"] == pytest.approx(expected_delta)
        assert "private" not in json.dumps(feedback).lower()
    assert json.loads(Path(tmp_path, result["inspect_artifact"]).read_text())["delta_voxels"] == 45
    assert service.calls == [
        ("apply", service.calls[0][1]),
        ("inspect", BASE_ID),
        ("inspect", NEW_ID),
        ("inspect", BASE_ID),
    ]


def test_parent_verification_failure_retains_edit_and_can_recover_by_inspection(
    setup_case, tmp_path, monkeypatch
):
    service = setup_case[2]
    original_inspect = service.inspect_revision
    inspections = 0

    async def transient_failure(workspace, revision_id=None):
        nonlocal inspections
        inspections += 1
        if inspections == 1:
            raise RuntimeError("/sensitive/parent/mask.nii.gz is temporarily unavailable")
        return await original_inspect(workspace, revision_id)

    monkeypatch.setattr(service, "inspect_revision", transient_failure)
    result, requests = run(
        setup_case,
        tmp_path,
        [call("apply_recorded_prompts"), finish(), call("inspect_revision"), finish()],
    )
    assert result["status"] == "completed"
    failed_comparison = json.loads(requests[1]["messages"][-1]["content"])
    assert failed_comparison["code"] == "PARENT_INSPECTION_FAILED"
    assert failed_comparison["revision"]["id"] == NEW_ID
    assert "NOT_COMPLETE" in requests[2]["messages"][-1]["content"]
    assert sum(name == "apply" for name, _ in service.calls) == 1
    assert "/sensitive/" not in json.dumps(requests)


def test_provider_failure_after_edit_retains_actual_revision(setup_case, tmp_path):
    workspace, body, service = setup_case
    count = 0

    def handle(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(
                200, json={"choices": [{"message": call("apply_recorded_prompts")}]}
            )
        return httpx.Response(503)

    result = asyncio.run(
        omni_agent.run_agent(
            service, workspace, body, tmp_path, transport=httpx.MockTransport(handle)
        )
    )
    assert result["status"] == "failed"
    assert result["revision"]["id"] == NEW_ID
    assert len(service.calls) == 2

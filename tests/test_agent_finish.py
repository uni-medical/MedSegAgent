"""Native completion shares host guards and reuses one bounded provider session."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from medsegagent import agent
from medsegagent.tool_definitions import WORK_TOOLS


@pytest.fixture(autouse=True)
def provider_config(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.delenv("MEDSEGAGENT_MAX_MODEL_REQUESTS", raising=False)
    monkeypatch.delenv("MEDSEGAGENT_MAX_TOOL_CALLS", raising=False)


def call(name="segment", arguments=None):
    if arguments is None:
        arguments = {"targets": ["liver"]}
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def finish(status="completed", unresolved=None):
    return call(
        "finish_task",
        {
            "status": status,
            "summary": "请求的输出已生成。" if status == "completed" else "还有未完成的要求。",
            "unresolved": [] if unresolved is None else unresolved,
        },
    )


def transport(messages, requests):
    sequence = iter(messages)

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": next(sequence)}]})

    return httpx.MockTransport(handler)


class Execution:
    def __init__(self, feedback=None):
        self.calls = []
        self.has_outputs = False
        self.unresolved_failures = []
        self.feedback = iter(feedback) if feedback is not None else None

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        result = next(self.feedback) if self.feedback is not None else {"ok": True}
        if isinstance(result, BaseException):
            raise result
        if result.get("ok") and name in {"segment", "compose_masks"}:
            self.has_outputs = True
        self.unresolved_failures = result.get("unresolved_failures", self.unresolved_failures)
        return result


def run(messages, execution=None, progress=None):
    requests = []
    execution = Execution() if execution is None else execution
    outcome = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            execution,
            on_progress=progress,
            transport=transport(messages, requests),
        )
    )
    return outcome, requests, execution


def test_native_completion_finishes_in_two_requests_without_json_repair():
    result, requests, execution = run([{"tool_calls": [call()]}, {"tool_calls": [finish()]}])
    assert result["status"] == "completed"
    assert result["model_requests"] == len(requests) == 2
    assert result["tool_calls"] == len(execution.calls) == 1
    assert all("response_format" not in request for request in requests)
    for request in requests:
        names = {tool["function"]["name"] for tool in request["tools"]}
        assert names == WORK_TOOLS | {"finish_task"}
    assert {tool["function"]["name"] for tool in agent.tool_schema("CT")} == WORK_TOOLS


@pytest.mark.parametrize(
    "feedback,unresolved",
    [
        ([], []),
        ([{"ok": False, "unresolved_failures": ["failed"]}], []),
        ([{"ok": True}], ["未支持的病灶"]),
        ([{"ok": True}, {"ok": False}], []),
        ([RuntimeError("/private/secret"), {"ok": True}], []),
    ],
)
def test_native_completion_cannot_bypass_host_completion_guards(feedback, unresolved):
    messages = [{"tool_calls": [call()]} for _ in feedback]
    messages.append({"tool_calls": [finish(unresolved=unresolved)]})
    result, requests, _ = run(messages, Execution(feedback))
    assert result["status"] == "failed"
    assert result["unresolved"]
    assert "/private" not in json.dumps(requests)


@pytest.mark.parametrize("status", ["needs_input", "failed"])
def test_native_partial_completion_preserves_outputs_and_unresolved(status):
    result, _, execution = run(
        [
            {"tool_calls": [call()]},
            {"tool_calls": [finish(status, ["未支持的病灶"])]},
        ]
    )
    assert result["status"] == status
    assert result["unresolved"] == ["未支持的病灶"]
    assert execution.has_outputs


def test_native_completion_remains_available_after_work_budget_is_used(monkeypatch):
    monkeypatch.setenv("MEDSEGAGENT_MAX_TOOL_CALLS", "1")
    result, _, _ = run([{"tool_calls": [call()]}, {"tool_calls": [finish()]}])
    assert result["status"] == "completed" and result["tool_calls"] == 1


@pytest.mark.parametrize(
    "bad_finish",
    [
        {"status": "completed", "summary": "Done", "unresolved": [], "extra": True},
        {"status": "completed", "summary": " ", "unresolved": []},
        {"status": "done", "summary": "Done", "unresolved": []},
        {"status": "completed", "summary": "Done", "unresolved": "none"},
    ],
)
def test_invalid_finish_is_correctable_with_native_tools(bad_finish):
    result, requests, execution = run(
        [
            {"tool_calls": [call()]},
            {"tool_calls": [call("finish_task", bad_finish)]},
            {"tool_calls": [finish()]},
        ]
    )
    assert result["status"] == "completed"
    assert len(execution.calls) == 1
    assert "INVALID_FINISH_ARGUMENTS" in requests[-1]["messages"][-1]["content"]
    assert all("response_format" not in request for request in requests)


def test_finish_cannot_claim_unobserved_work_in_same_turn():
    result, requests, execution = run(
        [
            {"tool_calls": [call(), finish()]},
            {"tool_calls": [call()]},
            {"tool_calls": [finish()]},
        ]
    )
    assert result["status"] == "completed"
    assert len(execution.calls) == 1
    rejected = requests[1]["messages"][-2:]
    assert all("FINISH_REQUIRES_SEPARATE_TURN" in item["content"] for item in rejected)
    assert len({item["tool_call_id"] for item in rejected}) == 2


def test_native_repair_is_bounded_by_request_budget(monkeypatch):
    monkeypatch.setenv("MEDSEGAGENT_MAX_MODEL_REQUESTS", "2")
    invalid = {"tool_calls": [call("finish_task", {"status": "completed"})]}
    result, requests, execution = run([invalid, invalid])
    assert result["status"] == "failed" and len(requests) == 2
    assert execution.calls == []


def test_stage_timings_report_observed_durations_without_private_payloads(monkeypatch):
    ticks = iter([10, 11.5, 11.5, 15.5, 15.5, 17])
    monkeypatch.setattr(agent, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    events = []
    result, _, _ = run(
        [{"tool_calls": [call()]}, {"tool_calls": [finish()]}], progress=events.append
    )
    assert [row["stage"] for row in result["timings"]] == [
        "model_request",
        "tool",
        "model_request",
    ]
    assert [row["duration_seconds"] for row in result["timings"]] == [1.5, 4.0, 1.5]
    assert result["timings"] == [event["timing"] for event in events if "timing" in event]
    assert all(row["status"] == "completed" for row in result["timings"])
    assert result["timings"][1]["tool"] == "segment"


def test_canceled_execution_records_timing_and_propagates():
    events = []
    with pytest.raises(asyncio.CancelledError):
        run(
            [{"tool_calls": [call()]}],
            Execution([asyncio.CancelledError()]),
            progress=events.append,
        )
    assert events[-1]["timing"]["stage"] == "tool"
    assert events[-1]["timing"]["status"] == "canceled"


def test_failed_provider_records_timing_without_provider_error_body():
    events = []
    with pytest.raises(agent.RoutingError):
        asyncio.run(
            agent.run_agent(
                "CT肝脏",
                "CT",
                Execution(),
                on_progress=events.append,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(500, text="private-error")
                ),
            )
        )
    assert events[-1]["timing"]["status"] == "failed"
    assert "private-error" not in json.dumps(events)


def test_provider_client_is_reused_within_task_and_closed_on_completion(monkeypatch):
    clients = []
    original = agent._provider_client

    def factory(**kwargs):
        client = original(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(agent, "_provider_client", factory)
    for _ in range(2):
        result, requests, _ = run([{"tool_calls": [call()]}, {"tool_calls": [finish()]}])
        assert result["status"] == "completed" and len(requests) == 2
    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert all(client.is_closed for client in clients)


def test_provider_client_closes_when_execution_is_canceled(monkeypatch):
    clients = []
    original = agent._provider_client

    def factory(**kwargs):
        client = original(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(agent, "_provider_client", factory)
    with pytest.raises(asyncio.CancelledError):
        run([{"tool_calls": [call()]}], Execution([asyncio.CancelledError()]))
    assert len(clients) == 1 and clients[0].is_closed


def test_preview_keeps_a_single_producer_schema():
    requests = []
    selected = asyncio.run(
        agent.select_tool(
            "CT肝脏",
            "CT",
            transport=transport(
                [{"tool_calls": [call(arguments={"targets": ["liver"], "task": "total"})]}],
                requests,
            ),
        )
    )
    assert selected.task == "total"
    segment = next(
        tool["function"] for tool in requests[0]["tools"] if tool["function"]["name"] == "segment"
    )
    assert segment["parameters"]["properties"]["task"]["type"] == "string"
    assert "anyOf" not in segment["parameters"]["properties"]["task"]


@pytest.mark.parametrize("task", [["total", "total_v3"], ["total"], None, 1])
def test_preview_rejects_non_string_producers(task):
    with pytest.raises(agent.RoutingError):
        asyncio.run(
            agent.select_tool(
                "CT肝脏",
                "CT",
                transport=transport(
                    [{"tool_calls": [call(arguments={"targets": ["liver"], "task": task})]}],
                    [],
                ),
            )
        )

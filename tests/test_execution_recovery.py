"""Explicit same-target recovery preserves failed attempts and producer identities."""

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest
from test_execution import Backend, make_input

from medsegagent import agent, catalog, core
from medsegagent.execution import TaskExecution

OLD = {"task": "total", "target": "liver", "quality": "standard"}


class Attempts(Backend):
    def __init__(self):
        super().__init__()
        self.failures = {("total", "standard")}
        self.attempts = []
        self.failure_code = None

    async def __call__(self, **kwargs):
        key = (kwargs["task"], kwargs["speed"])
        self.attempts.append(key)
        if key in self.failures:
            if self.failure_code:
                raise core.SegmentationError("PRIVATE CHECKPOINT PATH", code=self.failure_code)
            raise RuntimeError("PRIVATE BACKEND PATH")
        return await super().__call__(**kwargs)


def setup(tmp_path, monkeypatch):
    backend = Attempts()
    monkeypatch.setattr(core, "segment", backend)
    execution = TaskExecution(make_input(tmp_path), "CT", tmp_path / "out")
    return execution, backend


def request(task, quality, targets=("liver",), supersedes=None):
    result = {"task": task, "quality": quality, "targets": list(targets)}
    if supersedes is not None:
        result["supersedes"] = supersedes
    return result


@pytest.mark.parametrize("task,quality", [("total_v3", "standard"), ("total", "fast")])
def test_explicit_recovery_preserves_failure_audit_and_actual_producer(
    tmp_path, monkeypatch, task, quality
):
    execution, backend = setup(tmp_path, monkeypatch)

    async def scenario():
        failed = await execution.call("segment", request("total", "standard"))
        assert not failed["ok"]
        recovered = await execution.call("segment", request(task, quality, supersedes=[OLD]))
        assert recovered["ok"] and execution.is_complete
        observed = agent.safe_feedback(recovered)
        assert observed["resolved_attempts"] == [
            {**OLD, "replacement_region_id": observed["regions"][0]["region_id"]}
        ]
        assert observed["regions"][0]["task"] == task
        assert observed["regions"][0]["quality"] == quality
        exported = execution.export_result()
        assert len(exported["failed_attempts"]) == 1
        assert exported["failed_attempts"][0]["code"] == "INFERENCE_FAILED"
        assert exported["resolutions"] == observed["resolved_attempts"]
        assert len(exported["outputs"]) == 1
        assert (exported["outputs"][0]["task"], exported["outputs"][0]["quality"]) == (
            task,
            quality,
        )
        assert ("total", "liver", "standard") not in execution._semantic
        saved = json.loads((execution._root / "execution.json").read_text())
        assert saved["resolutions"] == exported["resolutions"]
        assert saved["failed_attempts"] == exported["failed_attempts"]
        assert "PRIVATE" not in json.dumps(observed)

    asyncio.run(scenario())
    assert len(backend.attempts) == 2


def test_cached_result_can_explicitly_resolve_a_failure_and_replay_idempotently(
    tmp_path, monkeypatch
):
    execution, backend = setup(tmp_path, monkeypatch)

    async def scenario():
        assert not (await execution.call("segment", request("total", "standard")))["ok"]
        assert (await execution.call("segment", request("total_v3", "fast")))["ok"]
        assert not execution.is_complete
        action = request("total_v3", "fast", supersedes=[OLD])
        first = await execution.call("segment", action)
        second = await execution.call("segment", action)
        assert first["ok"] and first["cached"] and second["ok"] and second["cached"]
        assert first["resolved_attempts"] == second["resolved_attempts"]
        assert execution.is_complete and len(execution._resolutions) == 1

    asyncio.run(scenario())
    assert len(backend.attempts) == 2


def test_replacing_five_lobe_lungs_does_not_swallow_other_failed_targets(tmp_path, monkeypatch):
    execution, backend = setup(tmp_path, monkeypatch)

    async def scenario():
        assert not (
            await execution.call("segment", request("total", "standard", ["lungs", "liver"]))
        )["ok"]
        audit = copy.deepcopy(execution.export_result()["failed_attempts"])
        lungs = {**OLD, "target": "lungs"}
        recovered = await execution.call("segment", request("total_v3", "fast", ["lungs"], [lungs]))
        assert recovered["ok"] and not execution.is_complete
        failures = execution.unresolved_failures
        assert failures and all(row["requested_targets"] == ["liver"] for row in failures)
        assert execution.export_result()["failed_attempts"] == audit
        assert (await execution.call("segment", request("total_v3", "fast", supersedes=[OLD])))[
            "ok"
        ]
        assert execution.is_complete
        assert len(execution._resolutions) == 2
        assert execution.export_result()["failed_attempts"] == audit

    asyncio.run(scenario())
    assert len(backend.attempts) == 3


@pytest.mark.parametrize(
    "declarations",
    [
        [],
        None,
        "total",
        [OLD, OLD],
        [{**OLD, "quality": []}],
        [{**OLD, "extra": True}],
        [{"task": "total", "target": "liver"}],
        [{**OLD, "quality": "fastest"}],
        [{**OLD, "task": "nonexistent"}],
        [{**OLD, "target": "spleen"}],
        [OLD, {**OLD, "target": "spleen"}],
    ],
)
def test_invalid_replacement_changes_no_requests_outputs_or_resolutions(
    tmp_path, monkeypatch, declarations
):
    execution, backend = setup(tmp_path, monkeypatch)

    async def scenario():
        assert not (await execution.call("segment", request("total", "standard")))["ok"]
        wanted, outputs, audit = copy.deepcopy(
            (execution._wanted, execution._outputs, execution._failed_jobs)
        )
        arguments = request("total_v3", "fast")
        arguments["supersedes"] = declarations
        response = await execution.call("segment", arguments)
        assert not response["ok"]
        assert response["code"] in {"INVALID_ARGUMENTS", "INVALID_SUPERSEDES"}
        assert execution._wanted == wanted and execution._outputs == outputs
        assert execution._failed_jobs == audit and execution._resolutions == {}

    asyncio.run(scenario())
    assert len(backend.attempts) == 1


@pytest.mark.parametrize("state", ["successful", "not_attempted", "same_producer"])
def test_only_failed_uncovered_requests_can_be_replaced(tmp_path, monkeypatch, state):
    execution, backend = setup(tmp_path, monkeypatch)

    async def scenario():
        old = OLD
        task, quality = "total_v3", "fast"
        if state == "successful":
            backend.failures.clear()
            assert (await execution.call("segment", request("total", "standard")))["ok"]
        elif state == "not_attempted":
            # An interrupted batch may record demand without a failed producer attempt.
            backend.failures = {("lung_nodules", "standard")}
            assert not (
                await execution.call(
                    "segment", {"targets": ["lung_nodules"], "quality": "standard"}
                )
            )["ok"]
            execution._wanted[("total", "liver", "standard")] = ("liver",)
        else:
            assert not (await execution.call("segment", request("total", "standard")))["ok"]
            task, quality = "total", "standard"
        count = len(backend.attempts)
        result = await execution.call("segment", request(task, quality, [old["target"]], [old]))
        assert not result["ok"] and result["code"] == "INVALID_SUPERSEDES"
        assert not execution._resolutions and len(backend.attempts) == count

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["inference", "output_validation", "cached_bytes"])
def test_failed_alternative_never_commits_a_replacement(tmp_path, monkeypatch, failure):
    execution, backend = setup(tmp_path, monkeypatch)

    async def scenario():
        assert not (await execution.call("segment", request("total", "standard")))["ok"]
        if failure == "inference":
            backend.failures.add(("total_v3", "fast"))
        elif failure == "output_validation":
            backend.bad_geometry = True
        else:
            assert (await execution.call("segment", request("total_v3", "fast")))["ok"]
            artifact = execution.export_result()["artifacts"][0]
            Path(artifact["path"]).write_bytes(b"changed after verification")
        response = await execution.call("segment", request("total_v3", "fast", supersedes=[OLD]))
        assert not response["ok"] and not execution.is_complete
        assert execution._resolutions == {}
        assert any(row["task"] == "total" for row in execution.export_result()["failed_attempts"])
        assert any(row.get("task") == "total" for row in execution.unresolved_failures)

    asyncio.run(scenario())


def test_missing_weights_can_be_replaced_but_remain_retryable_in_the_original_producer(
    tmp_path, monkeypatch
):
    execution, backend = setup(tmp_path, monkeypatch)
    backend.failure_code = "WEIGHTS_MISSING"

    async def scenario():
        first = await execution.call("segment", request("total", "standard"))
        assert first["code"] == "WEIGHTS_MISSING"
        second = await execution.call("segment", request("total", "standard"))
        assert second["code"] == "WEIGHTS_MISSING"  # Never PREVIOUS_FAILURE.
        recovered = await execution.call("segment", request("total_v3", "fast", supersedes=[OLD]))
        assert recovered["ok"] and execution.is_complete
        assert execution.export_result()["failed_attempts"][0]["code"] == "WEIGHTS_MISSING"

    asyncio.run(scenario())


def test_native_agent_can_observe_explicit_recovery_and_complete(tmp_path, monkeypatch):
    execution, _backend = setup(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://synthetic.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    calls = []

    def provider(incoming):
        payload = json.loads(incoming.content)
        calls.append(payload)
        index = len(calls)
        if index < 3:
            arguments = (
                request("total", "standard")
                if index == 1
                else request("total_v3", "fast", supersedes=[OLD])
            )
            message = {
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "segment", "arguments": json.dumps(arguments)},
                    }
                ]
            }
        else:
            observed = json.loads(
                next(
                    row["content"] for row in reversed(payload["messages"]) if row["role"] == "tool"
                )
            )
            assert observed["resolved_attempts"][0]["task"] == "total"
            assert (
                observed["resolved_attempts"][0]["replacement_region_id"]
                == observed["regions"][0]["region_id"]
            )
            assert not observed["unresolved_failures"]
            message = {
                "content": json.dumps(
                    {"status": "completed", "summary": "Recovered.", "unresolved": []}
                )
            }
        return httpx.Response(200, json={"choices": [{"message": message}]})

    result = asyncio.run(
        agent.run_agent("Segment liver.", "CT", execution, transport=httpx.MockTransport(provider))
    )
    assert result["status"] == "completed" and len(calls) == 3


def test_full_anatomy_recovery_feedback_preserves_all_regions_and_replacements(
    tmp_path, monkeypatch
):
    execution, _backend = setup(tmp_path, monkeypatch)
    targets = list(catalog.native_labels("total").values())

    async def scenario():
        assert not (await execution.call("segment", request("total", "standard", targets)))["ok"]
        recovered = await execution.call(
            "segment",
            request("total", "fast", targets, [{**OLD, "target": target} for target in targets]),
        )
        assert recovered["ok"] and execution.is_complete
        observed = agent.safe_feedback(recovered)
        assert observed["ok"], len(json.dumps(recovered).encode())
        assert len(observed["regions"]) == len(observed["resolved_attempts"]) == 117

    asyncio.run(scenario())

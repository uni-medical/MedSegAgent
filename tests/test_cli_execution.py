import asyncio
import json
import sys
from dataclasses import dataclass

import pytest

from medsegagent import agent, cli, core, execution


@pytest.mark.parametrize("command", ["weights", "weights-status"])
def test_cli_weights_status_is_read_only(monkeypatch, capsys, command):
    from medsegagent import weights

    status = {"available_count": 33, "ready_count": 0, "tasks": []}
    monkeypatch.setattr(weights, "inventory", lambda: status)
    monkeypatch.setattr(sys, "argv", ["medsegagent", command])
    cli.main()
    assert json.loads(capsys.readouterr().out) == status


@pytest.mark.parametrize("command", ["catalog", "tasks"])
def test_cli_catalog_lists_capabilities_without_input_or_model(monkeypatch, capsys, command):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Catalog browsing cannot construct an image execution or contact a model"
        )

    monkeypatch.setattr(execution, "TaskExecution", forbidden)
    monkeypatch.setattr(agent, "run_agent", forbidden)
    monkeypatch.setattr(agent, "select_tool", forbidden)
    monkeypatch.setattr(sys, "argv", ["medsegagent", command, "--task", "total_v3"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["task"] == "total_v3" and result["labels"]


def test_cli_route_never_constructs_an_execution(monkeypatch, capsys):
    @dataclass
    class Selection:
        tool: str = "segment"
        targets: tuple = ("lungs",)

    async def route(text, modality):
        assert (text, modality) == ("Segment both lungs", "CT")
        return Selection()

    def unexpected(**kwargs):
        raise AssertionError("route must not execute or validate image input")

    monkeypatch.setattr(agent, "select_tool", route)
    monkeypatch.setattr(execution, "TaskExecution", unexpected)
    monkeypatch.setattr(core, "validate_input", unexpected)
    monkeypatch.setattr(
        sys, "argv", ["medsegagent", "route", "--text", "Segment both lungs", "--modality", "CT"]
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"tool": "segment", "targets": ["lungs"]}


def test_cli_route_accepts_a_text_declared_modality_without_the_flag(monkeypatch, capsys):
    @dataclass
    class Selection:
        tool: str = "segment"
        targets: tuple = ("liver",)

    async def route(text, modality):
        assert (text, modality) == ("Segment CT liver", "CT")
        return Selection()

    monkeypatch.setattr(agent, "select_tool", route)
    monkeypatch.setattr(sys, "argv", ["medsegagent", "route", "--text", "Segment CT liver"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["targets"] == ["liver"]


def test_cli_route_requires_modality_before_contacting_the_provider(monkeypatch, capsys):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Image-free routing cannot detect an unknown modality")

    monkeypatch.setattr(agent, "select_tool", forbidden)
    monkeypatch.setattr(sys, "argv", ["medsegagent", "route", "--text", "Segment liver"])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2
    assert "route has no image to inspect" in capsys.readouterr().err


@pytest.mark.parametrize(
    "text,expected_modality",
    [
        ("Segment liver", None),
        ("Segment CT liver", None),
        ("Segment liver; do not use MR models", None),
        ("Segment liver; previous MRI study", None),
    ],
)
def test_cli_run_accepts_optional_modality_and_leaves_input_validation_to_execution(
    monkeypatch, capsys, text, expected_modality
):
    class Execution:
        def __init__(self, **kwargs):
            assert kwargs["modality"] == expected_modality
            assert kwargs["input_path"] == "/synthetic/unknown.nii"

        def export_result(self):
            return {"modality": expected_modality, "outputs": []}

    async def run(request_text, modality, runner):
        assert request_text == text and modality == expected_modality
        assert isinstance(runner, Execution)
        return {"status": "needs_input", "summary": "Please confirm CT or MR", "unresolved": []}

    def forbidden(*args, **kwargs):
        raise AssertionError("CLI must not force an unknown image through MR validation")

    monkeypatch.setattr(execution, "TaskExecution", Execution)
    monkeypatch.setattr(agent, "run_agent", run)
    monkeypatch.setattr(core, "validate_input", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["medsegagent", "run", "--text", text, "--input", "/synthetic/unknown.nii"]
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["completion"]["status"] == "needs_input"
    assert result["result"] == {"modality": expected_modality, "outputs": []}


def test_cli_run_delivers_agent_completion_and_every_host_output(monkeypatch, capsys):
    observed = []

    class Execution:
        def __init__(self, **kwargs):
            observed.append(kwargs)

        def export_result(self):
            return {"outputs": [{"target": "lungs"}, {"target": "lung_nodules"}]}

    async def run(text, modality, runner):
        assert isinstance(runner, Execution)
        return {
            "status": "failed",
            "summary": "Partial result retained",
            "unresolved": ["A requested output is unavailable"],
        }

    monkeypatch.setattr(execution, "TaskExecution", Execution)
    monkeypatch.setattr(agent, "run_agent", run)
    monkeypatch.setattr(core, "validate_input", lambda path, task: observed.append((path, task)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medsegagent",
            "run",
            "--text",
            "Segment lungs and nodules",
            "--modality",
            "CT",
            "--input",
            "/synthetic/input.nii",
            "--output",
            "/synthetic/results",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["completion"]["status"] == "failed"
    assert len(result["result"]["outputs"]) == 2
    assert observed == [
        {
            "input_path": "/synthetic/input.nii",
            "modality": "CT",
            "output_dir": "/synthetic/results",
        },
    ]


@pytest.mark.parametrize(
    "has_outputs,failures,unresolved,expected_exit",
    [
        (True, [], [], 0),
        (False, [], [], 1),
        (True, [{"code": "TARGETS_PENDING"}], [], 1),
        (True, [], ["A requested object is unresolved"], 1),
    ],
)
def test_cli_only_exits_successfully_for_verified_complete_outputs(
    monkeypatch, capsys, has_outputs, failures, unresolved, expected_exit
):
    class Execution:
        def __init__(self, **kwargs):
            self.has_outputs = has_outputs
            self.unresolved_failures = failures

        def export_result(self):
            return {"outputs": [{"target": "lungs"}] if has_outputs else []}

    async def run(*args):
        return {"status": "completed", "summary": "Finished", "unresolved": unresolved}

    monkeypatch.setattr(execution, "TaskExecution", Execution)
    monkeypatch.setattr(agent, "run_agent", run)
    monkeypatch.setattr(core, "validate_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medsegagent",
            "run",
            "--text",
            "Segment lungs",
            "--modality",
            "CT",
            "--input",
            "/synthetic/input.nii",
        ],
    )
    if expected_exit:
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert stopped.value.code == expected_exit
    else:
        cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["completion"]["status"] == ("failed" if expected_exit else "completed")


@pytest.mark.parametrize("has_partial_outputs", [False, True])
def test_cli_total_deadline_cancels_agent_and_retains_verified_outputs(
    monkeypatch, capsys, has_partial_outputs
):
    canceled = False
    outputs = [{"target": "lungs", "region_id": "region_lungs"}] if has_partial_outputs else []

    class Execution:
        def __init__(self, **kwargs):
            pass

        def export_result(self):
            assert canceled
            return {"outputs": outputs}

    async def run(*args):
        nonlocal canceled
        try:
            # Two individually short operations must share the same total deadline.
            await asyncio.sleep(0.6)
            await asyncio.sleep(0.6)
        except asyncio.CancelledError:
            canceled = True
            raise
        raise AssertionError("The complete Agent run must not outlive its deadline")

    monkeypatch.setenv("MEDSEGAGENT_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(execution, "TaskExecution", Execution)
    monkeypatch.setattr(agent, "run_agent", run)
    monkeypatch.setattr(core, "validate_input", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medsegagent",
            "run",
            "--text",
            "Segment lungs and nodules",
            "--modality",
            "CT",
            "--input",
            "/synthetic/input.nii",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["completion"]["status"] == "failed"
    assert result["completion"]["unresolved"] == ["TASK_TIMEOUT"]
    assert result["result"]["outputs"] == outputs

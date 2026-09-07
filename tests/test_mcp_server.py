from __future__ import annotations

import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from medsegagent import core, mcp_server


def test_mcp_surface_contains_exactly_two_tools():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert [tool.name for tool in tools] == ["segment_ct", "segment_mr"]
    for tool in tools:
        assert set(tool.input_schema["properties"]) == {"input_path", "output_dir", "targets"}


def test_ct_and_mr_class_counts_follow_installed_registry():
    assert len(core.task_classes("total")) == 117
    assert len(core.task_classes("total_mr")) == 50


@pytest.mark.parametrize("tool", ["segment_ct", "segment_mr"])
def test_expected_core_failure_is_llm_readable_tool_error(tool, monkeypatch):
    async def fail(**kwargs):
        raise core.SegmentationError("targets must be a non-empty list")

    monkeypatch.setattr(core, "segment", fail)
    with pytest.raises(ToolError, match="non-empty") as error:
        asyncio.run(mcp_server.mcp.call_tool(tool, {"input_path": "/synthetic.nii", "targets": []}))
    assert not isinstance(error.value, UnexpectedToolError)


def test_adapter_routes_to_shared_core(monkeypatch):
    calls = []

    async def record(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(core, "segment", record)
    asyncio.run(mcp_server.segment_mr("/synthetic.nii", "/private/runs", ["liver"]))
    assert calls == [
        {
            "task": "total_mr",
            "input_path": "/synthetic.nii",
            "output_dir": "/private/runs",
            "targets": ["liver"],
        }
    ]

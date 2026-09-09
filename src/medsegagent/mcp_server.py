"""MCP adapter for the common local inference core."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

from medsegagent import catalog, core
from medsegagent.tool_definitions import MAX_SEGMENT_PRODUCERS, tool_schema

MAX_EXECUTIONS = 8
PublicTaskName = Literal[catalog.public_task_names()]
Target = Annotated[str, Field(min_length=1, max_length=128)]
_TOOL_DESCRIPTIONS = {
    row["function"]["name"]: row["function"]["description"] for row in tool_schema()
}
RegionIDs = Annotated[
    list[str], Field(min_length=1, max_length=128, json_schema_extra={"uniqueItems": True})
]


class SupersededTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: PublicTaskName
    target: Target
    quality: Literal["fastest", "fast", "standard"]


@dataclass
class ExecutionBinding:
    execution: object
    input_path: str
    modality: str | None
    output_dir: str | None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class SessionState:
    executions: dict[str, ExecutionBinding] = field(default_factory=dict)


@asynccontextmanager
async def _lifespan(server):
    # stdio serves one client per process; references never survive this connection.
    state = SessionState()
    try:
        yield state
    finally:
        state.executions.clear()


def _session(ctx: Context) -> SessionState:
    try:
        state = ctx.request_context.lifespan_context
    except ValueError as exc:
        raise ToolError("An active MCP session is required.") from exc
    if not isinstance(state, SessionState):
        raise ToolError("An active MCP session is required.")
    return state


def _binding(ctx: Context, execution_id: str) -> ExecutionBinding:
    binding = _session(ctx).executions.get(execution_id)
    if binding is None:
        raise ToolError("Unknown execution_id in this MCP session.")
    return binding


def _execution_binding(ctx, input_path, modality, output_dir, execution_id):
    from medsegagent.execution import TaskExecution

    state = _session(ctx)
    source = str(Path(input_path).expanduser().resolve()) if input_path is not None else None
    destination = str(Path(output_dir).expanduser().resolve()) if output_dir else None
    if execution_id is not None:
        binding = _binding(ctx, execution_id)
        if source is not None and binding.input_path != source:
            raise ToolError("execution_id is bound to a different input or modality.")
        if destination is not None and binding.output_dir != destination:
            raise ToolError("output_dir cannot change within an execution.")
        return binding, execution_id
    if source is None:
        raise ToolError("input_path is required when creating an execution.")
    if len(state.executions) >= MAX_EXECUTIONS:
        raise ToolError("This session reached its input limit; start a new MCP session.")
    execution_id = uuid.uuid4().hex
    binding = ExecutionBinding(
        TaskExecution(input_path=source, modality=modality, output_dir=destination),
        source,
        modality,
        destination,
    )
    state.executions[execution_id] = binding
    return binding, execution_id


async def _execute(binding, execution_id, name, arguments, *, modality=None):
    async with binding.lock:
        known_modality = getattr(binding.execution, "modality", None) or binding.modality
        if modality is not None and known_modality is not None and known_modality != modality:
            raise ToolError("execution_id is bound to a different input or modality.")
        if name == "detect_modality" and modality is not None and known_modality is None:
            raise ToolError(
                "Use segment with modality to choose CT/MR for an existing unknown execution."
            )
        try:
            result = await binding.execution.call(name, arguments)
            exported = binding.execution.export_result()
            regions = {item["id"]: item for item in exported.get("regions", [])}
            requested = {}
            for item in exported.get("outputs", []):
                region = regions.get(item["region_id"])
                if region is not None:
                    requested.setdefault(item["artifact_id"], []).append(
                        {
                            "region_id": region["id"],
                            "target": item["target"],
                            "label_ids": region["values"],
                            **{
                                key: item.get(key, region.get(key))
                                for key in (
                                    "task",
                                    "quality",
                                    "usage_license",
                                    "requirements",
                                    "model_sources",
                                )
                                if key in item or key in region
                            },
                        }
                    )
            outputs = [
                {
                    **{
                        key: item[key]
                        for key in (
                            "id",
                            "name",
                            "path",
                            "labels",
                            "task",
                            "quality",
                            "usage_license",
                            "requirements",
                            "model_sources",
                        )
                        if key in item
                    },
                    "requested_regions": requested[item["id"]],
                }
                for item in exported.get("artifacts", [])
                if item["id"] in requested
            ]
            return {"execution_id": execution_id, **result, "local_outputs": outputs}
        except (ValueError, OSError) as exc:
            message = str(exc) if isinstance(exc, ValueError) else "Local file operation failed."
            raise ToolError(message) from exc


mcp = MCPServer(
    "MedSegAgent",
    instructions=(
        "Research use only. Query get_capabilities for task labels, producer choices and supported "
        "quality before specialist segmentation. Describe unsupported requests in terms of the "
        "requested output, without silently changing an explicitly requested producer. "
        "Use a user-declared CT/MR modality directly without detecting it again. "
        "For an unknown modality, detect_modality returns local evidence without binding a modality. "
        "Use its observations, intensity statistics and metadata to choose CT/MR via segment's "
        "modality argument; ask the user only when the available evidence does not support a choice. "
        "The first valid segmentation choice binds the modality; later segment calls may omit it. "
        "A segment call may produce multiple model outputs. lungs is the bilateral lung union; "
        "CT lung_left and lung_right combine their corresponding lobes. Preserve region IDs. "
        "Use inspect_artifact to check produced regions and compose_masks only for requested "
        "union, intersection or difference outputs. Empty masks do not rule out disease. "
        "Reuse execution_id only for the same input and established modality; IDs are private to this "
        "stdio session, with at most eight input executions per session. "
        "Local 3D NIfTI or a strict single-series CT/MR DICOM directory is accepted. "
        "ZIP and enhanced/compressed DICOM are rejected. This MCP server does not call an LLM. "
        "External callers receive local output paths and structured measurements, never image pixels. "
        "A local output file may contain multiple labels; requested_regions specifies the "
        "exact label_ids belonging to each requested object."
    ),
    lifespan=_lifespan,
)


@mcp.tool(description=_TOOL_DESCRIPTIONS["get_capabilities"])
async def get_capabilities(
    query: Annotated[str, Field(min_length=1, max_length=200)] | None = None,
    task: PublicTaskName | None = None,
    modality: Literal["CT", "MR"] | None = None,
) -> dict[str, object]:
    """Read task summaries or exact labels without opening an image or creating an execution."""
    try:
        return {
            "ok": True,
            "status": "completed",
            "capabilities": catalog.get_agent_capabilities(
                query=query, task=task, modality=modality
            ),
        }
    except (TypeError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(description=_TOOL_DESCRIPTIONS["detect_modality"])
async def detect_modality(
    ctx: Context,
    input_path: str | None = None,
    execution_id: str | None = None,
    modality: Literal["CT", "MR"] | None = None,
    output_dir: str | None = None,
) -> dict[str, object]:
    """Return local modality evidence, or the existing user declaration without detection.

    A new execution requires input_path, with an optional modality and output_dir.
    Reuse an execution by supplying only execution_id. Detection returns structured
    observations without binding a modality. Use segment's modality argument to choose
    CT/MR from the evidence; uncertain observations do not block a later choice.
    Source pixels and paths are not sent by this server to a language model.
    """
    binding, execution_id = _execution_binding(ctx, input_path, modality, output_dir, execution_id)
    return await _execute(binding, execution_id, "detect_modality", {}, modality=modality)


@mcp.tool(description=_TOOL_DESCRIPTIONS["segment"])
async def segment(
    input_path: str,
    modality: Literal["CT", "MR"] | None = None,
    *,
    targets: Annotated[
        list[Target],
        Field(min_length=1, max_length=512, json_schema_extra={"uniqueItems": True}),
    ],
    ctx: Context,
    output_dir: str | None = None,
    execution_id: str | None = None,
    quality: Literal["fastest", "fast", "standard"] | None = None,
    task: PublicTaskName
    | Annotated[
        list[PublicTaskName],
        Field(
            min_length=1, max_length=MAX_SEGMENT_PRODUCERS, json_schema_extra={"uniqueItems": True}
        ),
    ]
    | None = None,
    supersedes: Annotated[
        list[SupersededTarget],
        Field(min_length=1, max_length=512, json_schema_extra={"uniqueItems": True}),
    ]
    | None = None,
) -> dict[str, object]:
    """Return local region references bound to an opaque execution_id.

    Reuse execution_id to add targets on the same input, without repeating completed work.
    output_dir is a parent directory. This MCP server does not call an LLM; external
    callers receive local output paths and measurements, never image pixels.
    """
    binding, execution_id = _execution_binding(ctx, input_path, modality, output_dir, execution_id)
    arguments = {"targets": targets}
    if modality is not None:
        arguments["modality"] = modality
    if task is not None:
        arguments["task"] = task
    if quality is not None:
        arguments["quality"] = quality
    if supersedes is not None:
        arguments["supersedes"] = [
            row.model_dump() if isinstance(row, SupersededTarget) else row for row in supersedes
        ]
    return await _execute(binding, execution_id, "segment", arguments, modality=modality)


@mcp.tool(description=_TOOL_DESCRIPTIONS["inspect_artifact"])
async def inspect_artifact(
    execution_id: str, region_ids: RegionIDs, ctx: Context
) -> dict[str, object]:
    """Inspect this execution's region metadata, geometry, volumes and provenance.

    Pass only region IDs returned by segment or compose_masks in this MCP session.
    Returns structured observations, never image pixels or a clinical diagnosis.
    """
    return await _execute(
        _binding(ctx, execution_id), execution_id, "inspect_artifact", {"region_ids": region_ids}
    )


@mcp.tool(description=_TOOL_DESCRIPTIONS["compose_masks"])
async def compose_masks(
    execution_id: str,
    operation: Literal["union", "intersection", "difference"],
    region_ids: RegionIDs,
    name: Annotated[str, Field(min_length=1, max_length=80)],
    ctx: Context,
) -> dict[str, object]:
    """Derive a requested region from compatible regions without changing source masks.

    Difference subtracts all later regions from the first. Operands must belong to this
    execution and share input geometry. Overlapping regions remain independent outputs.
    """
    return await _execute(
        _binding(ctx, execution_id),
        execution_id,
        "compose_masks",
        {"operation": operation, "region_ids": region_ids, "name": name},
    )


async def _call(
    task: core.Task, input_path: str, output_dir: str | None, targets: list[str] | None
) -> dict[str, object]:
    try:
        return await core.segment(
            task=task, input_path=input_path, output_dir=output_dir, targets=targets
        )
    except (core.SegmentationError, OSError) as exc:
        message = (
            str(exc) if isinstance(exc, core.SegmentationError) else "Local file operation failed."
        )
        raise ToolError(message) from exc


async def segment_ct(
    input_path: str, output_dir: str | None = None, targets: list[str] | None = None
) -> dict[str, object]:
    """Segment a local 3D CT NIfTI with TotalSegmentator total in fast mode.

    targets contains exact CT anatomy names (e.g. liver, spleen, kidney_left).
    Omit it for all 117 structures; empty lists or blank names are rejected.
    output_dir is a parent directory; each run creates its own private subdirectory.
    A DICOM directory must contain one uncompressed classic CT/MR series with regular
    geometry. It is validated, privately copied, then converted with dcm2niix. ZIP,
    compressed/enhanced DICOM, mixed series and irregular geometry are rejected.
    Research use only.
    """
    return await _call("total", input_path, output_dir, targets)


async def segment_mr(
    input_path: str, output_dir: str | None = None, targets: list[str] | None = None
) -> dict[str, object]:
    """Segment a local 3D MRI NIfTI with TotalSegmentator total_mr in fast mode.

    targets contains exact MR anatomy names (e.g. liver, spleen, kidney_left).
    Omit it for all 50 structures; empty lists or blank names are rejected.
    output_dir is a parent directory; each run creates its own private subdirectory.
    A DICOM directory must contain one uncompressed classic CT/MR series with regular
    geometry. It is validated, privately copied, then converted with dcm2niix. ZIP,
    compressed/enhanced DICOM, mixed series and irregular geometry are rejected.
    Research use only.
    """
    return await _call("total_mr", input_path, output_dir, targets)


def _doctor() -> dict[str, object]:
    return core.doctor()


async def _lesion_call(task, input_path, output_dir, targets):
    if targets is None:
        targets = [task]
    if not isinstance(targets, list) or targets != [task]:
        raise ToolError(
            f"targets must contain only '{task}'; empty or other targets are unsupported."
        )
    return await _call(task, input_path, output_dir, targets)


async def segment_lung_nodules(
    input_path: str, output_dir: str | None = None, targets: list[str] | None = None
) -> dict[str, object]:
    """Segment CT lung nodules with the dedicated standard-resolution model.

    Omitted targets means ['lung_nodules']; no other targets are accepted. Accepts
    local 3D NIfTI or the same strict CT DICOM directory boundary as segment_ct.
    An empty result means no target was detected, not absence of disease. Does not
    classify malignancy or support MR, arbitrary tumors, boxes or points. Research only.
    """
    return await _lesion_call("lung_nodules", input_path, output_dir, targets)


async def segment_liver_lesions(
    input_path: str, output_dir: str | None = None, targets: list[str] | None = None
) -> dict[str, object]:
    """Segment CT liver lesions with the dedicated standard-resolution model.

    Omitted targets means ['liver_lesions']; no other targets are accepted. Accepts
    local 3D NIfTI or the same strict CT DICOM directory boundary as segment_ct.
    An empty result means no target was detected, not absence of disease. Does not
    classify lesion subtype or malignancy and does not support MR lesions. Research only.
    """
    return await _lesion_call("liver_lesions", input_path, output_dir, targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="MedSegAgent local segmentation MCP server")
    parser.add_argument("--transport", choices=["stdio"], default="stdio")
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args()
    if args.doctor:
        print(json.dumps(_doctor(), indent=2))
        return
    mcp.run()


if __name__ == "__main__":
    main()

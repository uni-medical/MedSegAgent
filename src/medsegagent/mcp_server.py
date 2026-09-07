"""MCP adapter for the common local inference core."""

from __future__ import annotations

import argparse
import json

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from medsegagent import core

mcp = MCPServer(
    "MedSegAgent",
    instructions=(
        "Research use only. Choose segment_ct for CT or segment_mr for MR. "
        "Translate anatomy names to exact TotalSegmentator class names. "
        "Omit targets only when the user requests all supported structures. "
        "Local 3D NIfTI or a strict single-series CT/MR DICOM directory is accepted. "
        "ZIP and enhanced/compressed DICOM are rejected. No image data is sent to an LLM."
    ),
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


@mcp.tool()
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


@mcp.tool()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-tool MedSegAgent MCP server")
    parser.add_argument("--transport", choices=["stdio"], default="stdio")
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args()
    if args.doctor:
        print(json.dumps(_doctor(), indent=2))
        return
    mcp.run()


if __name__ == "__main__":
    main()

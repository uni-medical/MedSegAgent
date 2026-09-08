"""Exercise all five MCP operations over stdio with real local segmentation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from medsegagent.weights import inspect_weights, weights_root


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def backend_receipts(directory):
    receipts = []
    for path in directory.rglob("state.json"):
        state = json.loads(path.read_text())
        result = state.get("result", {})
        receipts.append(
            {
                **{key: state[key] for key in ("status", "task", "speed", "targets", "device")},
                **{
                    key: result[key]
                    for key in ("runtime_seconds", "total_seconds", "totalsegmentator_version")
                },
            }
        )
    return receipts


def safe_result(result):
    """Keep protocol observations while excluding local selectors and private logs."""
    return {
        key: value
        for key, value in result.items()
        if key
        in {
            "ok",
            "status",
            "code",
            "execution_id",
            "modality",
            "modality_detection",
            "regions",
            "artifacts",
            "overlaps",
            "overlaps_checked",
            "cached",
            "requested_targets",
            "unresolved_failures",
            "resolved_attempts",
            "summary",
        }
    }


async def run(args, directory, report):
    source = Path(args.input).expanduser().resolve(strict=True)
    initial_digest = sha256(source)
    report["input_sha256"] = initial_digest
    report["weight_readiness"] = inspect_weights("total", "fast", ["liver", "spleen"])
    if not report["weight_readiness"]["ready"]:
        raise RuntimeError("WEIGHTS_MISSING")
    configuration = directory / "totalseg-config"
    configuration.mkdir(mode=0o700)
    (configuration / "config.json").write_text(
        json.dumps(
            {
                "send_usage_stats": False,
                "statistics_disclaimer_shown": True,
            }
        )
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"HOME", "PATH", "TMPDIR", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES"}
    }
    environment.update(
        {
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "MEDSEGAGENT_DEVICE": args.device,
            "MEDSEGAGENT_TIMEOUT_SECONDS": str(args.timeout),
            "TOTALSEG_HOME_DIR": str(configuration),
            "TOTALSEG_WEIGHTS_PATH": str(weights_root().resolve()),
        }
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "medsegagent.mcp_server"],
        env=environment,
        cwd=Path(__file__).resolve().parents[1],
    )
    with (directory / "server.stderr.log").open("w") as stderr:
        async with stdio_client(parameters, errlog=stderr) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=args.timeout) as session:
                started = time.monotonic()
                initialized = await session.initialize()
                report["protocol"] = {
                    "version": initialized.protocol_version,
                    "server": initialized.server_info.model_dump(by_alias=True),
                }
                listing = await session.list_tools()
                report["tools"] = [tool.name for tool in listing.tools]
                assert report["tools"] == [
                    "get_capabilities",
                    "detect_modality",
                    "segment",
                    "inspect_artifact",
                    "compose_masks",
                ]

                async def call(name, arguments):
                    before = time.monotonic()
                    response = await session.call_tool(name, arguments)
                    data = response.structured_content
                    if response.is_error or not isinstance(data, dict) or not data.get("ok"):
                        code = (
                            data.get("code", "INVALID_RESULT")
                            if isinstance(data, dict)
                            else "INVALID_RESULT"
                        )
                        report["operations"].append({"tool": name, "ok": False, "code": code})
                        raise RuntimeError(code)
                    entry = {
                        "tool": name,
                        "elapsed_seconds": time.monotonic() - before,
                        "result": safe_result(data),
                    }
                    if name == "get_capabilities":
                        entry["catalog"] = {
                            key: data["capabilities"][key]
                            for key in (
                                "task",
                                "label_count",
                                "speeds",
                                "availability",
                                "weight_readiness",
                            )
                        }
                    report["operations"].append(entry)
                    print(
                        json.dumps(
                            {
                                "tool": name,
                                "ok": True,
                                "elapsed_seconds": round(entry["elapsed_seconds"], 3),
                            }
                        ),
                        flush=True,
                    )
                    return data

                await call("get_capabilities", {"task": "total"})
                detection = await call(
                    "detect_modality",
                    {
                        "input_path": str(source),
                        "output_dir": str(directory / "results"),
                    },
                )
                execution_id = detection["execution_id"]
                segmentation = await call(
                    "segment",
                    {
                        "execution_id": execution_id,
                        "input_path": str(source),
                        "modality": "CT",
                        "targets": ["liver", "spleen"],
                        "task": "total",
                        "quality": "fast",
                    },
                )
                regions = {row["target"]: row for row in segmentation["regions"]}
                assert {"liver", "spleen"} <= set(regions)
                selected_ids = [regions[target]["region_id"] for target in ("liver", "spleen")]
                await call(
                    "inspect_artifact", {"execution_id": execution_id, "region_ids": selected_ids}
                )
                composed = await call(
                    "compose_masks",
                    {
                        "execution_id": execution_id,
                        "operation": "union",
                        "region_ids": selected_ids,
                        "name": "liver_and_spleen",
                    },
                )
                reference = nib.load(source)
                selected_masks = {}
                geometry = []
                for output in composed["local_outputs"]:
                    path = Path(output["path"]).resolve(strict=True)
                    assert path.is_relative_to(directory)
                    image = nib.load(path)
                    assert image.shape == reference.shape
                    assert np.allclose(image.affine, reference.affine, rtol=1e-5, atol=1e-4)
                    assert np.allclose(image.header.get_zooms(), reference.header.get_zooms())
                    values = np.asanyarray(image.dataobj)
                    geometry.append(
                        {
                            "artifact_id": output["id"],
                            "shape": list(image.shape),
                            "spacing": [float(x) for x in image.header.get_zooms()],
                            "geometry_matches_input": True,
                            "sha256": sha256(path),
                        }
                    )
                    for selector in output["requested_regions"]:
                        selected_masks[selector["target"]] = np.isin(values, selector["label_ids"])
                union = selected_masks["liver"] | selected_masks["spleen"]
                assert np.array_equal(selected_masks["liver_and_spleen"], union)
                assert all(
                    np.count_nonzero(selected_masks[target]) > 0 for target in ("liver", "spleen")
                )
                assert sha256(source) == initial_digest
                report["verification"] = {
                    "input_unchanged": True,
                    "output_geometry": geometry,
                    "union_matches_voxels": True,
                    "voxel_counts": {
                        target: int(np.count_nonzero(mask))
                        for target, mask in selected_masks.items()
                    },
                }
                report["backend_runs"] = backend_receipts(directory)
                assert len(report["backend_runs"]) == 1
                assert all(
                    row["device"] == args.device and row["status"] == "completed"
                    for row in report["backend_runs"]
                )
                report["elapsed_seconds"] = time.monotonic() - started
                report["ok"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/acceptance/mcp-real")
    parser.add_argument("--device", choices=["mps", "gpu", "cpu"], default="mps")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    directory = Path(args.output).expanduser().resolve() / datetime.now(UTC).strftime(
        "run-%Y%m%dT%H%M%S%fZ"
    )
    directory.mkdir(parents=True, mode=0o700)
    report = {"ok": False, "device": args.device, "external_llm_calls": 0, "operations": []}
    try:
        asyncio.run(run(args, directory, report))
    except Exception as exc:  # noqa: BLE001 - exception text and private paths stay out of reports.
        report["error_type"] = type(exc).__name__
    finally:
        report_path = directory / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"ok": report["ok"], "report": str(report_path)}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

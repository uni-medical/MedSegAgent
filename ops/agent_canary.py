"""Run an actual provider Agent against local inference on a public research input."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from medsegagent import agent
from medsegagent.execution import TaskExecution


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    execution = TaskExecution(args.input, args.modality, output_dir=args.output)
    calls = []
    original = execution.call

    async def recorded(name, arguments):
        response = await original(name, arguments)
        row = {"tool": name, "arguments": arguments, "ok": response.get("ok")}
        if "code" in response:
            row["code"] = response["code"]
        calls.append(row)
        return response

    async def progress(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    execution.call = recorded
    result = await asyncio.wait_for(
        agent.run_agent(args.request, args.modality, execution, progress),
        timeout=args.timeout,
    )
    exported = execution.export_result() if execution.has_outputs else None
    missing = sorted(set(args.require_tool) - {r["tool"] for r in calls if r["ok"]})
    report = {
        "request": args.request,
        "agent": result,
        "calls": calls,
        "execution_complete": execution.is_complete,
        "required_tools_missing": missing,
        "result": exported,
        "verified": result["status"] == "completed" and execution.is_complete and not missing,
    }
    (args.output / "agent-canary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"verified": report["verified"], "agent": result}, ensure_ascii=False))
    return report["verified"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--modality", choices=["CT", "MR"])
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--require-tool", action="append", default=[])
    args = parser.parse_args()
    os.umask(0o077)
    if args.env_file:
        load_dotenv(args.env_file)
    raise SystemExit(0 if asyncio.run(run(args)) else 1)


if __name__ == "__main__":
    main()

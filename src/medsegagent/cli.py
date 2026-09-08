"""CLI, server launcher, and an image-free real function-calling canary."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict

from dotenv import load_dotenv


def main():
    os.umask(0o077)
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="MedSegAgent — research use only")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8767)
    for name in ("run", "route"):
        run = sub.add_parser(name)
        run.add_argument("--text", required=True)
        run.add_argument("--modality", choices=["CT", "MR"])
        if name == "run":
            run.add_argument("--input", required=True)
            run.add_argument("--output", default="outputs")
    sub.add_parser("doctor")
    sub.add_parser("weights", aliases=["weights-status"])
    capabilities = sub.add_parser("catalog", aliases=["tasks"])
    capabilities.add_argument("--query")
    capabilities.add_argument("--task")
    capabilities.add_argument("--modality", choices=["CT", "MR"])
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        from medsegagent.web import create_app

        uvicorn.run(
            create_app(),
            host=args.host,
            port=args.port,
            access_log=False,
            timeout_graceful_shutdown=30,
            proxy_headers=False,
        )
    elif args.command == "doctor":
        from medsegagent.core import doctor

        print(json.dumps(doctor(), indent=2))
    elif args.command in {"catalog", "tasks"}:
        from medsegagent.catalog import get_capabilities

        print(
            json.dumps(
                get_capabilities(query=args.query, task=args.task, modality=args.modality),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command in {"weights", "weights-status"}:
        from medsegagent.weights import inventory

        print(json.dumps(inventory(), ensure_ascii=False, indent=2))
    else:

        async def execute():
            from medsegagent.agent import RoutingError, resolve_modality, run_agent, select_tool
            from medsegagent.core import DEFAULT_TIMEOUT_SECONDS, _positive_int
            from medsegagent.execution import TaskExecution

            if args.command == "route":
                try:
                    modality = resolve_modality(args.text, args.modality)
                except RoutingError as exc:
                    if exc.code == "MODALITY_REQUIRED":
                        parser.error(
                            "route has no image to inspect; pass --modality CT|MR or state "
                            "CT/MR explicitly in --text."
                        )
                    raise
                return asdict(await select_tool(args.text, modality))
            declared_modality = args.modality
            execution = TaskExecution(
                input_path=args.input,
                modality=declared_modality,
                output_dir=args.output,
            )
            timeout = _positive_int("MEDSEGAGENT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
            try:
                async with asyncio.timeout(timeout):
                    outcome = await run_agent(args.text, declared_modality, execution)
            except TimeoutError:
                outcome = {
                    "status": "failed",
                    "summary": "The task exceeded its total Agent execution time limit.",
                    "unresolved": ["TASK_TIMEOUT"],
                }
            if outcome["status"] == "completed" and (
                not execution.has_outputs
                or execution.unresolved_failures
                or outcome.get("unresolved")
            ):
                outcome = {
                    **outcome,
                    "status": "failed",
                    "unresolved": list(
                        dict.fromkeys(
                            [
                                *outcome.get("unresolved", []),
                                "The task did not produce every required verified output.",
                            ]
                        )
                    ),
                }
            return {"completion": outcome, "result": execution.export_result()}

        result = asyncio.run(execute())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "run" and result["completion"]["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()

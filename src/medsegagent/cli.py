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
        run.add_argument("--modality", required=True, choices=["CT", "MR"])
        if name == "run":
            run.add_argument("--input", required=True)
            run.add_argument("--output", default="outputs")
    sub.add_parser("doctor")
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
    else:

        async def execute():
            from medsegagent.agent import select_tool
            from medsegagent.core import segment, validate_input

            if args.command == "run":
                validate_input(args.input, task="total" if args.modality == "CT" else "total_mr")
            selected = await select_tool(args.text, args.modality)
            if args.command == "route":
                return asdict(selected)
            result = await segment(
                task=selected.task,
                input_path=args.input,
                output_dir=args.output,
                targets=selected.targets,
            )
            return {"selection": asdict(selected), "result": result}

        print(json.dumps(asyncio.run(execute()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

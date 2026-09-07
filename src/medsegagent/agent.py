"""One ordinary function-calling request, then one local Python tool.

Only the user's text, declared modality and allowed anatomical names enter the provider
request. Paths, filenames, headers, image bytes, masks and measurements never enter it.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

import httpx

from medsegagent.core import task_classes

MODEL = "deepseek-v4-flash"
LLM_TIMEOUT_SECONDS = 90
TOOLS = {"segment_ct": ("total", "CT"), "segment_mr": ("total_mr", "MR")}


class RoutingError(ValueError):
    """An invalid request, unavailable provider, or unusable tool selection."""


@dataclass(frozen=True)
class Selection:
    tool: str
    task: str
    targets: list[str]
    model: str = MODEL


def tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Local {modality} anatomical segmentation. Research use only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": sorted(task_classes(task))},
                        },
                    },
                    "required": ["targets"],
                    "additionalProperties": False,
                },
            },
        }
        for name, (task, modality) in TOOLS.items()
    ]


def provider_payload(text: str, modality: str) -> dict:
    if modality not in {"CT", "MR"}:
        raise RoutingError("Modality must be CT or MR; it cannot be inferred from NIfTI.")
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        raise RoutingError("Provide a segmentation request of 1–4000 characters.")
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Choose one segmentation tool matching the declared modality and requested anatomy. "
                    "Return exact supported targets. For all structures, list all allowed targets. "
                    "If the request is unsupported, ambiguous, or conflicts with the declared modality, "
                    "explain briefly without calling a tool. Do not substitute anatomy for lesions."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"modality": modality, "request": text}, ensure_ascii=False),
            },
        ],
        "tools": tool_schema(),
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "thinking": {"type": "disabled"},
        "max_tokens": 2048,
    }


async def select_tool(text: str, modality: str, *, transport=None) -> Selection:
    payload = provider_payload(text, modality)
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not base.startswith("https://") or not key:
        raise RoutingError("LLM configuration requires an HTTPS OPENAI_BASE_URL and API key.")
    try:
        async with (
            asyncio.timeout(LLM_TIMEOUT_SECONDS),
            httpx.AsyncClient(
                timeout=90, transport=transport, follow_redirects=False, trust_env=False
            ) as client,
            client.stream(
                "POST",
                base + "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
            ) as response,
        ):
            if response.status_code != 200:
                raise RoutingError(
                    f"LLM provider returned HTTP {response.status_code}; retry later."
                )
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > 1024 * 1024:
                    raise RoutingError("The model response exceeded the allowed size.")
                body.extend(chunk)
        message = json.loads(body)["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            raise RoutingError(
                "Specify one supported anatomical segmentation request and modality."
            )
        function = calls[0]["function"]
        name = function["name"]
        if name not in TOOLS or TOOLS[name][1] != modality:
            raise RoutingError("The selected tool does not match the declared modality.")
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict) or set(arguments) != {"targets"}:
            raise RoutingError("The model returned invalid tool arguments.")
        targets = arguments["targets"]
        allowed = task_classes(TOOLS[name][0])
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(t, str) or t not in allowed for t in targets)
        ):
            raise RoutingError("The model returned empty or unsupported targets.")
        return Selection(name, TOOLS[name][0], list(dict.fromkeys(targets)))
    except RoutingError:
        raise
    except (TimeoutError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise RoutingError("LLM selection failed; no inference was started.") from exc

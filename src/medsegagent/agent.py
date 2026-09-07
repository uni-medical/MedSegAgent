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
from medsegagent.task_specs import TASK_SPECS

MODEL = "deepseek-v4-flash"
LLM_TIMEOUT_SECONDS = 90
TOOLS = {spec.tool: (task, spec.modality) for task, spec in TASK_SPECS.items()}
DESCRIPTIONS = {
    "total": "CT anatomical structures; no lesion segmentation.",
    "total_mr": "MR anatomical structures; no lesion segmentation.",
    "lung_nodules": "CT lung nodules only. No malignancy classification or general lung tumor claim.",
    "liver_lesions": "CT liver lesions only. No lesion subtype or malignancy classification.",
}


def allowed_targets(task: str) -> set[str]:
    # Dedicated lesion tools expose the lesion, not their internal cropping anatomy.
    if task in {"lung_nodules", "liver_lesions"}:
        return {task}
    return task_classes(task)


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
                "description": f"Local {modality} segmentation. {DESCRIPTIONS[task]} Research use only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": sorted(allowed_targets(task))},
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
                    "Choose one segmentation tool matching the declared modality and requested target. "
                    "Return exact supported targets. For all structures, list all allowed targets. "
                    "If the request is unsupported, ambiguous, or conflicts with the declared modality, "
                    "explain briefly without calling a tool. Do not substitute anatomy for lesions. "
                    "Do not call any tool if satisfying the entire request requires more than one tool. "
                    "Never execute just part of a request. "
                    "Lesion tools support only CT lung nodules or CT liver lesions, not diagnosis."
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
            raise RoutingError("Specify one supported segmentation request and modality.")
        function = calls[0]["function"]
        name = function["name"]
        if name not in TOOLS or TOOLS[name][1] != modality:
            raise RoutingError("The selected tool does not match the declared modality.")
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict) or set(arguments) != {"targets"}:
            raise RoutingError("The model returned invalid tool arguments.")
        targets = arguments["targets"]
        allowed = allowed_targets(TOOLS[name][0])
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

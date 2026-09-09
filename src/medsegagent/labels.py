"""Canonical anatomical display names shared by model feedback and the Web UI."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def _label_names() -> dict[str, dict[str, str]]:
    return json.loads(files("medsegagent").joinpath("label_names.json").read_text(encoding="utf-8"))


def label_display_names(name: str) -> dict[str, str]:
    """Return display-only metadata; unknown labels receive no guessed translation."""
    if not isinstance(name, str):
        return {}
    entry = _label_names().get(name)
    if entry is None:
        return {}
    return {"display_name_zh": entry["zh"], "display_name_en": entry["en"]}

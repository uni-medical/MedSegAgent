"""Discoverable upstream tasks and deterministic public target recipes.

General requests keep stable default producers. Explicit task selection exposes the
chosen model's native labels, including clearly identified auxiliary outputs.
"""

from __future__ import annotations

import re
from collections import Counter

from medsegagent.task_specs import TASK_SPECS, model_ids


class CatalogError(ValueError):
    """An unsupported modality, target, or inconsistent installed label registry."""


_CT_LEFT = ("lung_upper_lobe_left", "lung_lower_lobe_left")
_CT_RIGHT = ("lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right")
COMPOSITES = {
    "CT": {"lung_left": _CT_LEFT, "lung_right": _CT_RIGHT, "lungs": _CT_LEFT + _CT_RIGHT},
    "MR": {"lungs": ("lung_left", "lung_right")},
}
_DEFAULT_PRODUCERS = ("total", "total_mr", "lung_nodules", "liver_lesions", "liver_lesions_mr")


def native_labels(task: str) -> dict[int, str]:
    from totalsegmentator.registry import get_task_classes

    if task not in TASK_SPECS:
        raise CatalogError("Unsupported segmentation task.")
    return dict(get_task_classes(task))


def public_native_targets(task: str) -> set[str]:
    spec = TASK_SPECS.get(task)
    if spec is None:
        raise CatalogError("Unsupported segmentation task.")
    native = set(native_labels(task).values())
    public = set(spec.default_targets) if spec.default_targets is not None else native
    if not public or not public <= native:
        raise CatalogError("The installed label registry does not match the public capability.")
    return public


def _modality(modality: str) -> str:
    if not isinstance(modality, str) or modality not in {"CT", "MR"}:
        raise CatalogError("Modality must be CT or MR.")
    return modality


def public_task_names(modality: str | None = None) -> tuple[str, ...]:
    if modality is not None:
        _modality(modality)
    return tuple(
        task
        for task, spec in TASK_SPECS.items()
        if spec.public_service_supported and (modality is None or spec.modality == modality)
    )


def _task_index(task: str) -> dict[str, dict]:
    native = native_labels(task)
    result = {
        name: {"target": name, "task": task, "native_targets": [name]} for name in native.values()
    }
    for target, members in COMPOSITES[TASK_SPECS[task].modality].items():
        if target not in result and set(members) <= set(native.values()):
            result[target] = {"target": target, "task": task, "native_targets": list(members)}
    return result


def _index(modality: str) -> dict[str, dict]:
    modality = _modality(modality)
    available_tasks = public_task_names(modality)
    ordered = [task for task in _DEFAULT_PRODUCERS if task in available_tasks]
    ordered.extend(task for task in available_tasks if task not in ordered)
    result = {}
    for task in ordered:
        for target in sorted(public_native_targets(task)):
            result.setdefault(target, {"target": target, "task": task, "native_targets": [target]})
    anatomy_task = "total" if modality == "CT" else "total_mr"
    available = public_native_targets(anatomy_task)
    for target, members in COMPOSITES[modality].items():
        if target in result or not set(members) <= available:
            raise CatalogError("A composite target conflicts with the installed label registry.")
        result[target] = {"target": target, "task": anatomy_task, "native_targets": list(members)}
    return result


def public_targets(modality: str) -> set[str]:
    return set(_index(modality))


def anatomical_targets(modality: str) -> set[str]:
    """All anatomy retains the default general model, without every specialist task."""
    return public_native_targets("total" if _modality(modality) == "CT" else "total_mr")


def target_descriptions(modality: str) -> dict[str, str]:
    return {
        target: f"Union of {', '.join(members)}."
        for target, members in COMPOSITES[_modality(modality)].items()
    }


def resolve_targets(modality: str, targets: list[str], task: str | None = None) -> list[dict]:
    """Resolve the entire request before execution; explicit producers remain identifiable."""
    _modality(modality)
    if task is not None:
        if (
            not isinstance(task, str)
            or task not in TASK_SPECS
            or TASK_SPECS[task].modality != modality
        ):
            raise CatalogError("The task is unsupported for the declared modality.")
        index = _task_index(task)
    else:
        index = _index(modality)
    if not isinstance(targets, list) or not targets or len(targets) > len(index):
        raise CatalogError("targets must be a bounded non-empty list of public target names.")
    if any(not isinstance(target, str) or target not in index for target in targets):
        raise CatalogError("One or more targets are unsupported for the declared modality or task.")
    return [
        dict(index[target], native_targets=list(index[target]["native_targets"]))
        for target in dict.fromkeys(targets)
    ]


def _summary(task: str) -> dict:
    spec = TASK_SPECS[task]
    return {
        "task": task,
        "modality": spec.modality,
        "description": spec.description,
        "label_count": len(native_labels(task)),
        "speeds": list(spec.speeds),
        "default_speed": spec.default_speed,
        "supports_roi": spec.supports_roi,
        "license_required": spec.license_required,
        "usage_license": spec.usage_license,
        "license_policy": spec.license_policy,
        "public_service_supported": spec.public_service_supported,
        "availability": spec.availability,
        "availability_reason": spec.availability_reason,
        "requirements": list(spec.requirements),
    }


def _query_terms(query: str) -> list[str]:
    # Keep native identifiers such as kidney_left intact. Only standalone
    # conjunctions are syntax; matching "and" inside "gland" adds unrelated organs.
    return list(
        dict.fromkeys(
            term for term in re.split(r"[\s,+]+", query.casefold()) if term and term != "and"
        )
    )


def _query_matches(index: dict[str, dict], terms: list[str]) -> list[str]:
    if not terms:
        return []
    targets = {target: target.casefold() for target in index}
    matches = [
        target
        for target, normalized in targets.items()
        if all(term in normalized for term in terms)
    ]
    if not matches and len(terms) > 1:
        # Prefer a single-label phrase (lung nodule); otherwise require every
        # requested object to be covered by this producer (liver spleen).
        per_term = [
            [target for target, normalized in targets.items() if term in normalized]
            for term in terms
        ]
        if all(per_term):
            matches = list(dict.fromkeys(target for group in per_term for target in group))
    return sorted(matches)


def get_capabilities(
    modality: str | None = None, task: str | None = None, query: str | None = None
) -> dict:
    """Administrative registry, including unavailable producers for explicit queries."""
    return _get_capabilities(modality, task, query, available_only=False)


def project_agent_capabilities(value: dict) -> dict:
    """Keep service-supported producers and omit deployment-policy diagnostics."""
    available = set(public_task_names())
    if not isinstance(value, dict) or (
        "task" in value and (not isinstance(value["task"], str) or value["task"] not in available)
    ):
        raise CatalogError("Unsupported segmentation task.")
    administrative = {
        "availability",
        "availability_reason",
        "license_required",
        "usage_license",
        "license_policy",
        "public_service_supported",
        "excluded_task_counts",
    }

    def summary(row):
        return {key: item for key, item in row.items() if key not in administrative}

    result = summary(value)
    if "tasks" in value:
        if not isinstance(value["tasks"], list):
            raise CatalogError("Invalid segmentation directory.")
        result["tasks"] = [
            summary(row)
            for row in value["tasks"]
            if isinstance(row, dict)
            and isinstance(row.get("task"), str)
            and row["task"] in available
        ]
    return result


def get_agent_capabilities(
    modality: str | None = None, task: str | None = None, query: str | None = None
) -> dict:
    """Discover supported service capabilities without revealing excluded models.

    Local weight readiness remains a separate observation: a supported task may
    still need preparation before inference. Clinical input requirements remain.
    """
    return project_agent_capabilities(_get_capabilities(modality, task, query, available_only=True))


def _get_capabilities(
    modality: str | None, task: str | None, query: str | None, *, available_only: bool
) -> dict:
    from totalsegmentator.registry import package_version

    if modality is not None:
        _modality(modality)
    if query is not None and (not isinstance(query, str) or not query.strip() or len(query) > 200):
        raise CatalogError("query must be a non-empty string of at most 200 characters.")
    if task is not None:
        if not isinstance(task, str) or task not in TASK_SPECS:
            raise CatalogError("Unsupported segmentation task.")
        spec = TASK_SPECS[task]
        if available_only and not spec.public_service_supported:
            raise CatalogError("Unsupported segmentation task.")
        if modality is not None and spec.modality != modality:
            raise CatalogError("The task is unsupported for the declared modality.")
        public = public_native_targets(task)
        index = _task_index(task)
        # Only explicit task details inspect disk; directory/search stay lightweight.
        from medsegagent.weights import inspect_weights

        result = {
            **_summary(task),
            "weight_readiness": {speed: inspect_weights(task, speed) for speed in spec.speeds}
            if spec.public_service_supported
            else {},
            "roi_weight_readiness": {
                speed: inspect_weights(task, speed, list(spec.roi_targets)) for speed in spec.speeds
            }
            if spec.public_service_supported and spec.supports_roi
            else {},
            "labels": [
                {"id": key, "name": value, "auxiliary": value not in public}
                for key, value in native_labels(task).items()
            ],
            "default_targets": list(spec.default_targets)
            if spec.default_targets is not None
            else None,
            "native_roi_targets": list(spec.roi_targets),
            "composites": {
                name: recipe["native_targets"]
                for name, recipe in index.items()
                if len(recipe["native_targets"]) > 1
            },
            "models_by_speed": {speed: list(model_ids(task, speed)) for speed in spec.speeds},
            "roi_models_by_speed": {
                speed: list(model_ids(task, speed, roi=True)) for speed in spec.speeds
            }
            if spec.supports_roi
            else {},
        }
        if query is not None:
            matches = _query_matches(index, _query_terms(query))
            native = {target for name in matches for target in index[name]["native_targets"]}
            result["query"] = query.strip()
            result["matches"] = matches
            result["labels"] = [row for row in result["labels"] if row["name"] in native]
            if result["default_targets"] is not None:
                result["default_targets"] = [
                    target for target in result["default_targets"] if target in native
                ]
            result["native_roi_targets"] = [
                target for target in result["native_roi_targets"] if target in native
            ]
            result["composites"] = {
                name: members for name, members in result["composites"].items() if name in matches
            }
        return result
    relevant = [
        name
        for name, spec in TASK_SPECS.items()
        if (modality is None or spec.modality == modality)
        and (not available_only or spec.public_service_supported)
    ]
    excluded = Counter(
        TASK_SPECS[name].license_policy
        for name in relevant
        if not TASK_SPECS[name].public_service_supported
    )
    result = {
        "totalsegmentator_version": package_version(),
        "tasks": [],
        "excluded_task_counts": dict(excluded),
    }
    if query is None:
        result["tasks"] = [_summary(name) for name in public_task_names(modality)]
    else:
        result["query"] = query.strip()
        terms = _query_terms(query)
        for name in relevant:
            matches = _query_matches(_task_index(name), terms)
            if matches or (terms and all(term in name.casefold() for term in terms)):
                result["tasks"].append({**_summary(name), "matches": matches})
    return result

"""Shared provider-independent declarations for local Agent and MCP operations."""

from __future__ import annotations

from medsegagent import catalog
from medsegagent.task_specs import TASK_SPECS

MAX_SEGMENT_PRODUCERS = 8

WORK_TOOLS = frozenset(
    {"get_capabilities", "detect_modality", "segment", "inspect_artifact", "compose_masks"}
)


def tool_schema(modality: str | None = None) -> list[dict]:
    modalities = [modality] if modality is not None else ["CT", "MR"]
    tasks = [task for task, spec in TASK_SPECS.items() if spec.modality in modalities]
    public_tasks = list(catalog.public_task_names(modality))
    region_ids = {
        "type": "array",
        "minItems": 1,
        "maxItems": 128,
        "uniqueItems": True,
        "items": {"type": "string"},
    }

    def function(name, description, properties, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    return [
        function(
            "get_capabilities",
            "Read the installed segmentation catalog without inference. With no query, list "
            "task summaries. Search an anatomical target or task name with query; inspect an "
            "exact task for native labels, supported speeds, prerequisites, availability and "
            "local weight readiness. "
            "Choose a producer before requesting specialist or overlapping labels. Registered "
            "tasks outside this public, noncommercial deployment are unavailable; explain the limitation "
            "without requesting a private license or silently replacing the producer. "
            "Returned label IDs belong to its native "
            "model, not the normalized output masks.",
            {
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "task": {"type": "string", "enum": tasks},
                "modality": {"type": "string", "enum": modalities},
            },
            [],
        ),
        function(
            "detect_modality",
            "Gather local modality evidence when the user has not specified it. Declared "
            "modality is returned directly without detection. Prefer DICOM or same-name NIfTI "
            "JSON Modality metadata; otherwise report a lightweight CT/MR classifier vote and "
            "intensity statistics. The vote is advisory, not calibrated confidence or an "
            "ultrasound detector; uncertainty does not prevent an evidence-based choice. "
            "Choose the modality in segment after considering the observations and request. "
            "Results are cached; repeating the tool adds no evidence.",
            {},
            [],
        ),
        function(
            "segment",
            "Segment requested public objects from the task's image. The host chooses and groups "
            "the default backends when task is omitted; common anatomy, lungs, lung_left, "
            "lung_right, lung_nodules and liver_lesions retain their semantic defaults. "
            "Specify task to choose a producer, or an array such as [total, total_v3] to "
            "compare the same targets across producers in one device-scheduled parallel call. "
            "Use exact catalog labels supported by every chosen producer. Query "
            "get_capabilities for specialist labels and supported quality; do not invent names. "
            "Omit quality for the backend default. Different producers may provide the same "
            "target as independent outputs; compare their returned region IDs explicitly. "
            "Dependent union/intersection/difference operations follow after results return. "
            "To recover a failed attempt, supersedes can explicitly replace an old "
            "task/target/quality request with this call's verified result for the same target; "
            "supersedes requires a single replacement producer. "
            "Failures are preserved in the audit; replacement is never automatic. "
            "Identical input, producer, targets and effective settings reuse prior results. "
            "If the image modality is not yet established, supply your chosen modality. "
            "A user declaration takes precedence. Subsequent calls reuse that modality. "
            "For all anatomy, query the selected anatomical task and request its native labels; "
            "do not silently add lesions or duplicate composite objects. Whole-lung defaults "
            "combine all corresponding native lung lobes.",
            {
                "targets": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 512,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                },
                "task": {
                    "anyOf": [
                        {"type": "string", "enum": public_tasks},
                        {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_SEGMENT_PRODUCERS,
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": public_tasks},
                        },
                    ]
                },
                "quality": {"type": "string", "enum": ["fastest", "fast", "standard"]},
                "supersedes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 512,
                    "uniqueItems": True,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "task": {"type": "string", "enum": public_tasks},
                            "target": {"type": "string", "minLength": 1, "maxLength": 128},
                            "quality": {"type": "string", "enum": ["fastest", "fast", "standard"]},
                        },
                        "required": ["task", "target", "quality"],
                    },
                },
                **(
                    {"modality": {"type": "string", "enum": modalities}} if modality is None else {}
                ),
            },
            ["targets"] if modality is not None else ["targets", "modality"],
        ),
        function(
            "inspect_artifact",
            "Inspect existing region IDs: measured volume, empty status and pairwise spatial "
            "overlap. Use observations to check requested outputs or decide a justified next "
            "action. Statistics alone do not establish anatomical accuracy or diagnose disease.",
            {"region_ids": region_ids},
            ["region_ids"],
        ),
        function(
            "compose_masks",
            "Create a requested derived region from existing region IDs. union combines masks; "
            "intersection keeps shared voxels; difference subtracts all later masks from the "
            "first. Preserve the intended object; do not invent anatomical or clinical "
            "equivalence. Inputs must share the task's image geometry. Name is a short display "
            "label, never a path. lungs already has a complete built-in recipe in segment.",
            {
                "operation": {"type": "string", "enum": ["union", "intersection", "difference"]},
                "region_ids": region_ids,
                "name": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            ["operation", "region_ids", "name"],
        ),
    ]

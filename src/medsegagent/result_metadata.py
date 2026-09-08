"""Public measurements shared by Web and A2A, including historical task records."""

import math
import re

from medsegagent.task_specs import TASK_SPECS

_RESULT_FIELDS = frozenset(
    {
        "modality",
        "targets",
        "duration_seconds",
        "elapsed_seconds",
        "labels",
        "geometry",
        "voxel_counts",
        "warning",
        "research_use_only",
        "segmentation_shape",
        "segmentation_voxel_spacing",
        "nonzero_voxels",
        "detection_status",
        "no_target_detected",
        "runtime_seconds",
        "schema_version",
        "volume_measurement",
        "total_seconds",
        "normalization_seconds",
        "summary",
        "completion",
        "agent",
    }
)
_LABEL_FIELDS = frozenset({"id", "source_id", "name", "color", "voxels", "volume_mm3", "volume_ml"})
_VOLUME_FIELDS = frozenset(
    {
        "method",
        "source_spatial_unit",
        "spatial_unit",
        "unit_assumption",
        "spacing_mm",
        "voxel_volume_mm3",
    }
)
_FILE_FIELDS = frozenset(
    {
        "name",
        "url",
        "media_type",
        "sha256",
        "size_bytes",
        "kind",
        "output_id",
        "label_id",
        "label_name",
        "mask_value",
    }
)
_REGION_FIELDS = frozenset(
    {"id", "target", "name", "artifact_id", "values", "voxels", "volume_mm3", "volume_ml"}
)
_MODALITY_FIELDS = frozenset(
    {
        "modality",
        "candidate_modality",
        "source",
        "status",
        "supported",
        "vote_agreement",
        "limitations",
        "declared_modality",
        "conflict",
        "cached",
        "intensity_statistics",
    }
)


def model_provenance(region, regions, *, visiting=None):
    """Shared registry policy for a verified object and its source region identifiers."""
    producer, quality = region.get("task"), region.get("quality")
    if producer in TASK_SPECS:
        spec = TASK_SPECS[producer]
        if quality not in spec.speeds:
            raise ValueError("The requested-output manifest has an invalid quality.")
        return {
            "task": producer,
            "quality": quality,
            "usage_license": spec.usage_license,
            "requirements": list(spec.requirements),
        }
    if producer != "composition" or quality is not None:
        raise ValueError("The requested-output manifest has an invalid producer.")
    visiting = set() if visiting is None else set(visiting)
    if region["id"] in visiting:
        raise ValueError("The requested-output manifest contains a source cycle.")
    visiting.add(region["id"])
    sources = {}
    for source_id in region.get("provenance", {}).get("source_region_ids", []):
        if source_id not in regions:
            raise ValueError("The requested-output manifest has an unknown source region.")
        observed = model_provenance(regions[source_id], regions, visiting=visiting)
        for item in observed.get("model_sources", [observed]):
            sources[(item["task"], item["quality"])] = item
    if not sources:
        raise ValueError("A composed region must retain its model sources.")
    sources = list(sources.values())
    return {
        "task": "composition",
        "quality": None,
        "usage_license": "; ".join(dict.fromkeys(row["usage_license"] for row in sources)),
        "requirements": list(
            dict.fromkeys(value for row in sources for value in row["requirements"])
        ),
        "model_sources": sources,
    }


def _model_metadata(value: dict) -> dict:
    """Only compact model provenance; paths and arbitrary nested report fields stay private."""
    result = {}
    task = value.get("task")
    if isinstance(task, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", task):
        result["task"] = task
    quality = value.get("quality")
    if "quality" in value and (
        quality is None or isinstance(quality, str) and quality in {"standard", "fast", "fastest"}
    ):
        result["quality"] = quality
    usage = value.get("usage_license")
    if isinstance(usage, str) and re.fullmatch(r"[A-Za-z0-9_.; +\-]{1,256}", usage):
        result["usage_license"] = usage
    requirements = value.get("requirements")
    if isinstance(requirements, list):
        result["requirements"] = [
            item
            for item in requirements
            if isinstance(item, str)
            and 0 < len(item) <= 512
            and not any(char in item for char in "\x00\r\n")
        ][:64]
    return result


def _model_provenance(value: dict) -> dict:
    result = _model_metadata(value)
    sources = value.get("model_sources")
    if isinstance(sources, list):
        result["model_sources"] = [
            _model_metadata(item) for item in sources[:1024] if isinstance(item, dict)
        ]
    return result


def result_metadata(result: dict) -> dict:
    """Project supported fields without rewriting stored records or mask values."""
    public = {key: value for key, value in result.items() if key in _RESULT_FIELDS}
    public.update(_model_provenance(result))
    if isinstance(result.get("modality_detection"), dict):
        public["modality_detection"] = {
            key: value
            for key, value in result["modality_detection"].items()
            if key in _MODALITY_FIELDS
        }
        statistics = public["modality_detection"].get("intensity_statistics")
        if isinstance(statistics, dict):
            public["modality_detection"]["intensity_statistics"] = {
                key: value
                for key, value in statistics.items()
                if key in {"mean", "std", "min", "max"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            }
        else:
            public["modality_detection"].pop("intensity_statistics", None)
    if isinstance(public.get("labels"), list):
        public["labels"] = [
            {key: value for key, value in label.items() if key in _LABEL_FIELDS}
            for label in public["labels"]
            if isinstance(label, dict)
        ]
    if isinstance(public.get("volume_measurement"), dict):
        public["volume_measurement"] = {
            key: value
            for key, value in public["volume_measurement"].items()
            if key in _VOLUME_FIELDS
        }
    if isinstance(result.get("files"), list):
        public["files"] = [
            {key: value for key, value in row.items() if key in _FILE_FIELDS}
            for row in result["files"]
            if isinstance(row, dict)
        ]
    if isinstance(result.get("outputs"), list):
        public["outputs"] = []
        for output in result["outputs"]:
            if not isinstance(output, dict):
                continue
            # Deliberately do not recurse into arbitrary backend results or paths.
            projected = result_metadata({k: v for k, v in output.items() if k != "outputs"})
            projected.update({k: output[k] for k in ("id", "name") if k in output})
            public["outputs"].append(projected)
    if isinstance(result.get("regions"), list):
        public["regions"] = [
            {
                **{key: value for key, value in row.items() if key in _REGION_FIELDS},
                **_model_provenance(row),
            }
            for row in result["regions"]
            if isinstance(row, dict)
        ]
    return public

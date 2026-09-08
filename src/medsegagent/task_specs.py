"""Installed metadata and the public, noncommercial service policy.

Availability is permission/configuration policy, not a claim of installed weights
or tested inference. Upstream pure-data maps remain the source of task parameters.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from totalsegmentator.map_tasks_config import DEFAULT_CONFIG, TASK_CONFIGS, TASK_ID_WEIGHTS_CONFIGS
from totalsegmentator.map_to_binary import commercial_models
from totalsegmentator.registry import TASKS, get_task_classes, requires_license, task_modality

TASK_NAMES = tuple(TASKS)
Task = Literal[TASK_NAMES]
Speed = Literal["fast", "standard", "fastest"]

CAPABILITIES = {
    "summary": "支持公开可下载的 CT / MR 分割任务，包括解剖结构、特定病灶和精细结构；按需查询任务和标签。",
    "limits": "按各模型适用范围执行；支持多工具分割、区域检查及集合运算。需专门许可或未发布权重的任务不可执行。脑动脉瘤仅适用于 TOF MRI 和非商业用途。",
}
_EXPERIMENTAL = {
    "test": "UPSTREAM_WEIGHT_MAPPING_MISSING",
    "total_highres_test": "UPSTREAM_WEIGHTS_UNPUBLISHED",
}
_LEGACY_TOOLS = {"total": "segment_ct", "total_mr": "segment_mr"}
_DEFAULT_TARGETS = {"lung_nodules": ("lung_nodules",), "liver_lesions": ("liver_lesions",)}
_REQUIREMENTS = {
    "brain_aneurysm": ("TOF MRI only.", "Noncommercial use; preserve CC-BY-NC-4.0 attribution."),
    "abdominal_muscles": ("Segments muscles only within T4-L4.",),
    "tissue_types_mr": (
        "For DIXON, upstream recommends fat images for fat and water images for muscle.",
    ),
}
_DESCRIPTIONS = {
    "total": "117 CT anatomical labels; default general anatomy model.",
    "total_v3": "117 CT anatomical labels; newer model with vertebrae_L6 replacing vertebrae_S1.",
    "total_mr": "50 MR anatomical labels; default general anatomy model.",
    "lung_nodules": "CT lung nodules; the native lung label is auxiliary anatomy.",
    "liver_lesions": "CT liver lesions; does not classify lesion subtype or malignancy.",
    "liver_lesions_mr": "MR liver lesions; does not classify lesion subtype or malignancy.",
    "brain_aneurysm": "Brain aneurysm segmentation on TOF MRI only, under CC-BY-NC-4.0.",
    "vertebrae_pp_refined": "Per-vertebra CT bodies, refined using a second vertebral-body mask.",
    "teeth": "CT/CBCT dental structures; first crops with the craniofacial model.",
    "abdominal_muscles": "CT muscle segmentation restricted to the T4-L4 region.",
}


@dataclass(frozen=True)
class TaskSpec:
    modality: Literal["CT", "MR"]
    tool: str
    default_speed: Speed
    supports_roi: bool
    capability: str
    scope: str
    default_targets: tuple[str, ...] | None = None
    speeds: tuple[Speed, ...] = ("standard",)
    license_required: bool = False
    availability: str = "available"
    availability_reason: str | None = None
    public_service_supported: bool = True
    license_policy: str = "permissive"
    usage_license: str = "Apache-2.0"
    requirements: tuple[str, ...] = ()
    description: str = ""
    roi_targets: tuple[str, ...] = ()


def _make_spec(task: str) -> TaskSpec:
    raw = TASK_CONFIGS[task]
    modes = raw.get("sub_modes", {"default": raw})
    speeds = tuple(
        speed
        for speed, mode in (("standard", "default"), ("fast", "fast"), ("fastest", "fastest"))
        if mode in modes
    )
    licensed = requires_license(task)
    reason = _EXPERIMENTAL.get(task) or ("LICENSE_GATED_MODEL" if licensed else None)
    policy = (
        "experimental"
        if task in _EXPERIMENTAL
        else "license_gated"
        if licensed
        else "noncommercial"
        if task == "brain_aneurysm"
        else "permissive"
    )
    usage = (
        "unpublished"
        if policy == "experimental"
        else "upstream license-server terms"
        if licensed
        else "CC-BY-NC-4.0"
        if task == "brain_aneurysm"
        else "Apache-2.0"
    )
    description = _DESCRIPTIONS.get(
        task, f"{task_modality(task)} {task.replace('_', ' ')} segmentation."
    )
    roi = task.startswith("total")
    crop_task = "total_mr" if task.endswith("_mr") else "total"
    roi_targets = (
        tuple(
            sorted(set(get_task_classes(task).values()) & set(get_task_classes(crop_task).values()))
        )
        if roi
        else ()
    )
    return TaskSpec(
        modality=task_modality(task),
        tool=_LEGACY_TOOLS.get(task, "segment_" + task),
        default_speed="fast" if "fast" in speeds else "standard",
        supports_roi=roi,
        capability=description,
        scope=description,
        default_targets=_DEFAULT_TARGETS.get(task),
        speeds=speeds,
        license_required=licensed,
        availability="available" if reason is None else "unavailable",
        availability_reason=reason,
        public_service_supported=reason is None,
        license_policy=policy,
        usage_license=usage,
        requirements=_REQUIREMENTS.get(task, ()),
        description=description,
        roi_targets=roi_targets,
    )


TASK_SPECS: dict[str, TaskSpec] = {task: _make_spec(task) for task in TASK_NAMES}


def task_config(task: str, quality: str = "standard") -> dict:
    """Effective public-mode configuration without inference imports or key checks."""
    if task not in TASK_SPECS or quality not in TASK_SPECS[task].speeds:
        raise ValueError("Unsupported task or quality mode.")
    raw = TASK_CONFIGS[task]
    mode = "default" if quality == "standard" else quality
    selected = raw.get("sub_modes", {}).get(mode)
    if selected is None:
        selected = {
            key: value
            for key, value in raw.items()
            if key not in {"disallow_fast", "commercial", "info_msg"}
        }
    return deepcopy({**DEFAULT_CONFIG, "plans": "nnUNetPlans", **selected})


def supports_native_roi(task: str, targets: list[str] | tuple[str, ...] | None) -> bool:
    """Total v3 L6 has no label in upstream's older rough crop model."""
    spec = TASK_SPECS.get(task)
    return bool(spec and spec.supports_roi and targets and set(targets) <= set(spec.roi_targets))


def model_ids(task: str, quality: str = "standard", roi: bool = False) -> tuple[int, ...]:
    """All weights used by a mode, including default crop/refinement dependencies."""
    config = task_config(task, quality)
    primary = config["task_id"]
    result = list(primary) if isinstance(primary, list) else [primary]
    if task == "vertebrae_pp_refined":
        result.append(305)
    crop = config.get("crop")
    if isinstance(crop, list) or config.get("cascade") or roi:
        if config.get("crop_model"):
            result.extend(model_ids(config["crop_model"], "standard"))
        elif crop and ("body_trunc" in crop or "body_extremities" in crop):
            result.append(300)
        else:
            result.append(
                852 if task.endswith("_mr") else 297 if config.get("robust_crop") else 298
            )
    return tuple(dict.fromkeys(result))


def required_model_ids(
    task: str,
    speed: str = "standard",
    targets: list[str] | None = None,
    *,
    robust_crop: bool = False,
    body_seg: bool = False,
) -> tuple[int, ...]:
    result = list(model_ids(task, speed, roi=supports_native_roi(task, targets)))
    if robust_crop and 298 in result:
        result[result.index(298)] = 297
    if body_seg and not task.endswith("_mr") and not TASK_CONFIGS[task].get("crop") and not targets:
        result.append(300)
    return tuple(dict.fromkeys(result))


def model_record(model_id: int) -> dict:
    """Download metadata and runtime layouts, without inspecting local license keys."""
    if model_id not in TASK_ID_WEIGHTS_CONFIGS:
        raise ValueError("Upstream has no download mapping for this model.")
    record = deepcopy(TASK_ID_WEIGHTS_CONFIGS[model_id])
    inverse = {value: key for key, value in commercial_models.items()}
    licensed = model_id in inverse or bool(record.get("commercial"))
    version = record.get("version")
    record.update(
        model_id=model_id,
        license_required=licensed,
        download_task=inverse.get(model_id, model_id if licensed else None),
        download_url=(
            f"https://github.com/wasserth/TotalSegmentator/releases/download/{version}/{record['foldername']}.zip"
            if not licensed and version not in (None, "TODO")
            else None
        ),
    )
    layouts = []
    for task, spec in TASK_SPECS.items():
        for speed in spec.speeds:
            config = task_config(task, speed)
            primary = config["task_id"]
            if model_id not in (primary if isinstance(primary, list) else [primary]):
                continue
            layout = {key: config[key] for key in ("trainer", "model", "plans", "folds")}
            if layout not in layouts:
                layouts.append(layout)
    record["expected_configs"] = layouts
    return record

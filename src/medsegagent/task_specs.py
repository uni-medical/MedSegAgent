"""Explicit local tool policy; installed registry still defines native label values."""

from dataclasses import dataclass
from typing import Literal

Task = Literal["total", "total_mr", "lung_nodules", "liver_lesions"]
Speed = Literal["fast", "standard"]


@dataclass(frozen=True)
class TaskSpec:
    modality: Literal["CT", "MR"]
    tool: str
    default_speed: Speed
    supports_roi: bool
    default_targets: tuple[str, ...] | None = None


TASK_SPECS: dict[str, TaskSpec] = {
    "total": TaskSpec("CT", "segment_ct", "fast", True),
    "total_mr": TaskSpec("MR", "segment_mr", "fast", True),
    "lung_nodules": TaskSpec("CT", "segment_lung_nodules", "standard", False, ("lung_nodules",)),
    "liver_lesions": TaskSpec("CT", "segment_liver_lesions", "standard", False, ("liver_lesions",)),
}

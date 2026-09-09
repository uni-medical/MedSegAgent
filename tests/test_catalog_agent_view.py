"""Agent discovery excludes inaccessible producers before inspecting their labels."""

from __future__ import annotations

import json

import pytest

from medsegagent import catalog, weights
from medsegagent.task_specs import TASK_SPECS

ADMINISTRATIVE_FIELDS = {
    "availability",
    "availability_reason",
    "license_required",
    "usage_license",
    "license_policy",
    "public_service_supported",
    "excluded_task_counts",
}
DISABLED_TASKS = [task for task, spec in TASK_SPECS.items() if not spec.public_service_supported]


@pytest.mark.parametrize("query", [None, "heart", "kidney_cyst_left"])
def test_agent_discovery_never_reads_excluded_producers(monkeypatch, query):
    native_labels = catalog.native_labels

    def supported_labels(task):
        assert task not in DISABLED_TASKS
        return native_labels(task)

    monkeypatch.setattr(catalog, "native_labels", supported_labels)
    result = catalog.get_agent_capabilities(modality="CT", query=query)
    assert result["tasks"]
    assert {row["task"] for row in result["tasks"]} <= set(catalog.public_task_names("CT"))
    assert not ADMINISTRATIVE_FIELDS.intersection(result)
    assert all(not ADMINISTRATIVE_FIELDS.intersection(row) for row in result["tasks"])
    serialized = json.dumps(result)
    assert all(json.dumps(task) not in serialized for task in DISABLED_TASKS)


@pytest.mark.parametrize("task", DISABLED_TASKS)
def test_explicit_excluded_task_is_rejected_without_reading_labels(monkeypatch, task):
    def no_labels(task):
        pytest.fail("Excluded model labels must not be inspected.")

    monkeypatch.setattr(catalog, "native_labels", no_labels)
    with pytest.raises(catalog.CatalogError, match="^Unsupported segmentation task\\.$"):
        catalog.get_agent_capabilities(task=task)


def test_agent_projection_filters_administrative_search_without_mutating_it():
    source = catalog.get_capabilities(modality="CT", query="heart")
    original = json.dumps(source, sort_keys=True)
    assert any(row["task"] in DISABLED_TASKS for row in source["tasks"])
    result = catalog.project_agent_capabilities(source)
    assert all(row["task"] in catalog.public_task_names("CT") for row in result["tasks"])
    assert not ADMINISTRATIVE_FIELDS.intersection(result)
    assert all(not ADMINISTRATIVE_FIELDS.intersection(row) for row in result["tasks"])
    assert json.dumps(source, sort_keys=True) == original
    with pytest.raises(catalog.CatalogError):
        catalog.project_agent_capabilities(catalog.get_capabilities(task="heartchambers_highres"))


def test_agent_details_preserve_input_requirements_and_truthful_weight_readiness(monkeypatch):
    monkeypatch.setattr(
        weights,
        "inspect_weights",
        lambda task, speed, *args: {"task": task, "quality": speed, "ready": False},
    )
    result = catalog.get_agent_capabilities(task="brain_aneurysm")
    assert result["requirements"] == ["TOF MRI only."]
    assert result["weight_readiness"]["standard"]["ready"] is False
    assert not ADMINISTRATIVE_FIELDS.intersection(result)
    assert "CC-BY" not in json.dumps(result)
    assert "Noncommercial" not in json.dumps(result)
    assert catalog.get_capabilities(task="brain_aneurysm")["usage_license"] == "CC-BY-NC-4.0"


def test_agent_cardiac_discovery_preserves_available_native_targets():
    result = catalog.get_agent_capabilities(task="total")
    expected = {
        "heart",
        "aorta",
        "pulmonary_vein",
        "atrial_appendage_left",
        "superior_vena_cava",
        "inferior_vena_cava",
    }
    assert expected <= {row["name"] for row in result["labels"]}
    assert "models_by_speed" in result and "native_roi_targets" in result

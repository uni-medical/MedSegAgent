"""Source-backed task discovery, service policy and complete weight dependencies."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from totalsegmentator import map_tasks_config
from totalsegmentator.registry import TASKS, get_task_classes, requires_license, task_modality

from medsegagent import catalog
from medsegagent.task_specs import (
    TASK_SPECS,
    model_ids,
    model_record,
    required_model_ids,
    supports_native_roi,
    task_config,
)


def test_full_registry_is_discoverable_without_claiming_every_task_executable():
    assert list(TASK_SPECS) == TASKS and len(TASK_SPECS) == 53
    assert len(catalog.public_task_names()) == 33
    for task, spec in TASK_SPECS.items():
        assert spec.modality == task_modality(task)
        assert spec.license_required == requires_license(task)
        details = catalog.get_capabilities(task=task)
        assert {row["id"]: row["name"] for row in details["labels"]} == get_task_classes(task)
        assert details["public_service_supported"] == (spec.availability == "available")
    public = catalog.get_capabilities()
    assert len(public["tasks"]) == 33
    assert public["excluded_task_counts"] == {"license_gated": 18, "experimental": 2}
    assert all("labels" not in row and "models_by_speed" not in row for row in public["tasks"])
    assert all(row["public_service_supported"] for row in public["tasks"])
    json.dumps(public, allow_nan=False)


def test_noncommercial_open_weights_are_distinct_from_license_server_models():
    brain = catalog.get_capabilities(task="brain_aneurysm")
    assert brain["modality"] == "MR"
    assert brain["availability"] == "available" and not brain["license_required"]
    assert brain["license_policy"] == "noncommercial" and brain["usage_license"] == "CC-BY-NC-4.0"
    assert "TOF MRI only." in brain["requirements"]
    for task, spec in TASK_SPECS.items():
        if spec.license_required:
            assert not spec.public_service_supported and spec.availability == "unavailable"
            assert spec.availability_reason == "LICENSE_GATED_MODEL"
            assert task not in catalog.public_task_names()
    assert (
        catalog.get_capabilities(task="test")["availability_reason"]
        == "UPSTREAM_WEIGHT_MAPPING_MISSING"
    )
    assert (
        catalog.get_capabilities(task="total_highres_test")["availability_reason"]
        == "UPSTREAM_WEIGHTS_UNPUBLISHED"
    )


def test_speed_modes_match_real_upstream_configs_instead_of_silent_ignored_flags():
    source = Path(map_tasks_config.__file__).with_name("python_api.py").read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_task_config"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "DEFAULT_CONFIG": map_tasks_config.DEFAULT_CONFIG,
        "TASK_CONFIGS": map_tasks_config.TASK_CONFIGS,
        "show_license_info": lambda: None,
    }
    exec(compile(module, "<isolated-upstream-task-config>", "exec"), namespace)  # noqa: S102 - isolate installed config function from inference imports.
    for task, spec in TASK_SPECS.items():
        raw = map_tasks_config.TASK_CONFIGS[task]
        expected_modes = tuple(
            speed
            for speed, mode in [("standard", "default"), ("fast", "fast"), ("fastest", "fastest")]
            if mode in raw.get("sub_modes", {"default": raw})
        )
        assert spec.speeds == expected_modes
        for speed in spec.speeds:
            actual = namespace["get_task_config"](
                task,
                fast=speed == "fast",
                fastest=speed == "fastest",
                quiet=True,
                plans="nnUNetPlans",
                robust_crop=False,
            )
            ours = task_config(task, speed)
            for key in actual:
                assert ours[key] == actual[key], (task, speed, key)
    assert TASK_SPECS["lung_vessels"].speeds == ("standard",)
    assert TASK_SPECS["body_mr"].speeds == ("standard", "fast")
    with pytest.raises(ValueError):
        task_config("lung_vessels", "fast")


def test_default_resolution_stays_stable_while_queries_reveal_all_producers():
    targets = ["liver", "lungs", "lung_nodules", "liver_lesions", "kidney_cyst_left"]
    assert [row["task"] for row in catalog.resolve_targets("CT", targets)] == [
        "total",
        "total",
        "lung_nodules",
        "liver_lesions",
        "total",
    ]
    rows = catalog.get_capabilities(modality="CT", query="kidney_cyst_left")["tasks"]
    assert {row["task"] for row in rows} == {
        "total",
        "total_v3",
        "kidney_cysts",
        "total_highres_test",
    }
    assert all("kidney_cyst_left" in row["matches"] for row in rows)
    explicit = catalog.resolve_targets("CT", ["kidney_cyst_left"], task="kidney_cysts")
    assert explicit == [
        {
            "target": "kidney_cyst_left",
            "task": "kidney_cysts",
            "native_targets": ["kidney_cyst_left"],
        }
    ]
    assert catalog.resolve_targets("MR", ["liver_lesions"])[0]["task"] == "liver_lesions_mr"


def test_explicit_task_exposes_auxiliary_labels_without_changing_default_semantics():
    assert "lung" not in catalog.public_targets("CT")
    details = catalog.get_capabilities(task="lung_nodules")
    assert {row["name"]: row["auxiliary"] for row in details["labels"]} == {
        "lung": True,
        "lung_nodules": False,
    }
    assert catalog.resolve_targets("CT", ["lung"], task="lung_nodules")[0]["native_targets"] == [
        "lung"
    ]
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_targets("CT", ["lung"])


def test_composite_identity_is_preserved_across_producers_and_l6_avoids_broken_crop():
    total = catalog.resolve_targets("CT", ["lungs"], task="total")[0]
    newer = catalog.resolve_targets("CT", ["lungs"], task="total_v3")[0]
    assert total["native_targets"] == newer["native_targets"] and len(newer["native_targets"]) == 5
    assert total["task"] != newer["task"] and total["target"] == newer["target"]
    assert supports_native_roi("total_v3", newer["native_targets"])
    assert not supports_native_roi("total_v3", ["vertebrae_L6"])
    assert 298 not in required_model_ids("total_v3", "fast", ["vertebrae_L6"])
    assert 298 in required_model_ids("total_v3", "fast", ["liver"])


@pytest.mark.parametrize(
    "task,quality,expected",
    [
        ("teeth", "standard", (113, 115, 298)),
        ("abdominal_muscles", "standard", (952, 300)),
        ("vertebrae_pp_refined", "standard", (803, 305)),
        ("lung_vessels", "standard", (117, 297)),
        ("liver_lesions", "standard", (591, 297)),
        ("liver_lesions_mr", "standard", (589, 852)),
        ("liver_segments_mr", "standard", (576, 852)),
        ("headneck_muscles", "standard", (778, 779, 298)),
        ("total", "standard", (291, 292, 293, 294, 295)),
        ("total_mr", "standard", (850, 851)),
    ],
)
def test_required_weights_include_the_entire_inference_chain(task, quality, expected):
    assert model_ids(task, quality) == expected


def test_all_public_mode_weights_have_public_download_mappings_and_runtime_layouts():
    ids = {
        identifier
        for task in catalog.public_task_names()
        for speed in TASK_SPECS[task].speeds
        for identifier in model_ids(task, speed)
    }
    assert len(ids) == 50 and {305, 615, 803} <= ids
    for identifier in ids:
        row = model_record(identifier)
        assert not row["license_required"] and row["download_task"] is None
        assert row["download_url"].startswith(
            "https://github.com/wasserth/TotalSegmentator/releases/download/"
        )
        assert row["expected_configs"]
        for config in row["expected_configs"]:
            assert set(config) == {"trainer", "plans", "model", "folds"}
    assert model_record(615)["expected_configs"][0]["folds"] is None
    assert model_record(713)["expected_configs"][0]["folds"] == [0, 1, 2, 3, 4]
    assert model_record(509)["license_required"] and model_record(509)["download_url"] is None
    assert model_record(957)["download_url"] is None
    with pytest.raises(ValueError):
        model_record(517)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": ""},
        {"query": []},
        {"query": "x" * 201},
        {"task": "unknown"},
        {"task": "total", "modality": "MR"},
        {"modality": "PET"},
    ],
)
def test_invalid_catalog_queries_do_not_fall_back_to_unrelated_tasks(kwargs):
    with pytest.raises(catalog.CatalogError):
        catalog.get_capabilities(**kwargs)


def test_discovery_imports_neither_torch_nor_network_nor_license_configuration():
    source = """
import importlib.abc, sys, socket
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('torch', 'totalsegmentator.python_api', 'totalsegmentator.config'):
            raise AssertionError('Discovery imported inference or license configuration')
def blocked(*a, **kw): raise AssertionError('Discovery attempted a network connection')
sys.meta_path.insert(0, BlockHeavy())
socket.socket.connect = blocked
from medsegagent.catalog import get_capabilities
assert len(get_capabilities()['tasks']) == 33
assert len(get_capabilities(task='total')['labels']) == 117
"""
    subprocess.run(
        [sys.executable, "-c", source], check=True, timeout=15, capture_output=True, text=True
    )


def test_roi_weight_readiness_includes_crop_without_blocking_full_l6(monkeypatch):
    from medsegagent import weights

    monkeypatch.setattr(
        weights,
        "inspect_model",
        lambda model_id, **kwargs: {"model_id": model_id, "ready": model_id != 298},
    )
    for task in ("total", "total_v3"):
        details = catalog.get_capabilities(task=task)
        full = details["weight_readiness"]["fast"]
        roi = details["roi_weight_readiness"]["fast"]
        assert full["ready"] is True and 298 not in full["model_ids"]
        assert roi["ready"] is False and roi["missing_model_ids"] == [298]
        assert "vertebrae_L6" not in details["native_roi_targets"]
    l6 = weights.inspect_weights("total_v3", "fast", ["vertebrae_L6"])
    assert l6["ready"] is True and 298 not in l6["model_ids"]
    assert catalog.get_capabilities(task="brain_aneurysm")["roi_weight_readiness"] == {}


@pytest.mark.parametrize(
    "query", ["liver spleen", "liver, spleen", "liver+spleen", "liver and spleen"]
)
def test_multi_object_query_requires_all_words_within_one_producer(query):
    response = catalog.get_capabilities(query=query)
    matches = {row["task"]: row["matches"] for row in response["tasks"]}
    assert {"total", "total_v3", "total_mr"} <= matches.keys()
    assert matches["total"] == ["liver", "spleen"]
    assert "liver_lesions" not in matches and "liver_vessels" not in matches
    assert catalog.get_capabilities(query="liver unsupported_target_xyz")["tasks"] == []


@pytest.mark.parametrize("task", ["total", "total_v3"])
def test_task_query_returns_precise_labels_and_preserves_model_metadata(task):
    full = catalog.get_capabilities(task=task)
    focused = catalog.get_capabilities(task=task, query="liver")
    assert len(full["labels"]) == focused["label_count"] == 117
    assert focused["query"] == "liver" and focused["matches"] == ["liver"]
    assert focused["labels"] == [{"id": 5, "name": "liver", "auxiliary": False}]
    assert focused["composites"] == {} and focused["native_roi_targets"] == ["liver"]
    label_fields = {"labels", "default_targets", "native_roi_targets", "composites"}
    assert {key: value for key, value in full.items() if key not in label_fields} == {
        key: value
        for key, value in focused.items()
        if key not in label_fields | {"query", "matches"}
    }


@pytest.mark.parametrize("task", ["total", "total_v3", "total_mr"])
def test_task_query_preserves_only_requested_composite_and_all_its_members(task):
    details = catalog.get_capabilities(task=task, query="lungs")
    modality = TASK_SPECS[task].modality
    members = list(catalog.COMPOSITES[modality]["lungs"])
    assert details["matches"] == ["lungs"]
    assert details["composites"] == {"lungs": members}
    assert {row["name"] for row in details["labels"]} == set(members)
    assert set(details["native_roi_targets"]) <= set(members)


@pytest.mark.parametrize(
    "query", ["liver spleen", "liver, spleen", "liver+spleen", "liver and spleen"]
)
def test_task_query_uses_same_multi_object_matching_as_directory(query):
    details = catalog.get_capabilities(task="total", query=query)
    assert details["matches"] == ["liver", "spleen"]
    assert {row["name"] for row in details["labels"]} == {"liver", "spleen"}
    assert details["composites"] == {}
    assert set(details["native_roi_targets"]) == {"liver", "spleen"}


@pytest.mark.parametrize("query", ["unsupported_target_xyz", "and", ", +"])
def test_task_query_without_matches_returns_no_unrelated_labels(query):
    assert catalog.get_capabilities(query=query)["tasks"] == []
    details = catalog.get_capabilities(task="lung_nodules", query=query)
    for key in ("matches", "labels", "default_targets", "native_roi_targets"):
        assert details[key] == []
    assert details["composites"] == {}
    assert details["task"] == "lung_nodules" and details["label_count"] == 2


def test_query_preserves_native_identifier_and_auxiliary_label_semantics():
    rows = catalog.get_capabilities(query="kidney_left")["tasks"]
    assert rows and all(row["matches"] == ["kidney_left"] for row in rows)
    details = catalog.get_capabilities(task="lung_nodules", query="lung_nodules")
    assert details["labels"] == [{"id": 2, "name": "lung_nodules", "auxiliary": False}]
    assert details["default_targets"] == ["lung_nodules"]
    assert catalog.get_capabilities(task="lung_nodules", query="lung")["labels"] == [
        {"id": 1, "name": "lung", "auxiliary": True},
        {"id": 2, "name": "lung_nodules", "auxiliary": False},
    ]


def test_multi_word_phrase_keeps_exact_label_matches():
    response = catalog.get_capabilities(modality="CT", query="lung nodule")
    assert [(row["task"], row["matches"]) for row in response["tasks"]] == [
        ("lung_nodules", ["lung_nodules"])
    ]

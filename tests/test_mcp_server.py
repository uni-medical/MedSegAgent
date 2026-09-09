from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from medsegagent import catalog, core, mcp_server


def test_mcp_surface_contains_supported_tools():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert [tool.name for tool in tools] == [
        "get_capabilities",
        "detect_modality",
        "segment",
        "inspect_artifact",
        "compose_masks",
    ]
    schemas = {tool.name: tool.input_schema for tool in tools}
    detection = schemas["detect_modality"]
    assert not detection.get("required")
    assert set(detection["properties"]) == {"input_path", "execution_id", "modality", "output_dir"}
    segment = schemas["segment"]
    assert set(segment["required"]) == {"input_path", "targets"}
    assert "ctx" not in segment["properties"]
    assert "enum" not in segment["properties"]["targets"]["items"]
    choices = schemas["get_capabilities"]["properties"]["task"]["anyOf"]
    assert next(row["enum"] for row in choices if "enum" in row) == list(
        catalog.public_task_names()
    )
    assert segment["properties"]["targets"]["minItems"] == 1


def test_mcp_and_native_tools_share_names_descriptions_and_action_parameter_contracts():
    from medsegagent.tool_definitions import WORK_TOOLS, tool_schema

    native = {row["function"]["name"]: row["function"] for row in tool_schema()}
    exposed = {row.name: row for row in asyncio.run(mcp_server.mcp.list_tools())}
    assert set(exposed) == set(native) == WORK_TOOLS
    context_fields = {
        "get_capabilities": set(),
        "detect_modality": {"input_path", "execution_id", "output_dir", "modality"},
        "segment": {"input_path", "execution_id", "output_dir"},
        "inspect_artifact": {"execution_id"},
        "compose_masks": {"execution_id"},
    }

    def normalize(schema, definitions=None):
        # SDK local arguments may be nullable/defaulted; the action's value constraints agree.
        if isinstance(schema, dict) and "$ref" in schema:
            assert schema["$ref"].startswith("#/$defs/")
            return normalize(definitions[schema["$ref"].split("/")[-1]], definitions)
        if isinstance(schema, dict) and "anyOf" in schema:
            choices = [row for row in schema["anyOf"] if row.get("type") != "null"]
            if len(choices) == 1:
                return normalize(choices[0], definitions)
            return {"anyOf": [normalize(choice, definitions) for choice in choices]}
        if isinstance(schema, dict):
            return {
                key: sorted(value) if key == "enum" else normalize(value, definitions)
                for key, value in schema.items()
                if key
                not in {"title", "description", "default", "required", "additionalProperties"}
            }
        if isinstance(schema, list):
            return [normalize(row, definitions) for row in schema]
        return schema

    for name, function in native.items():
        mcp_tool = exposed[name]
        assert mcp_tool.description == function["description"]
        local = {
            key: row
            for key, row in mcp_tool.input_schema["properties"].items()
            if key not in context_fields[name]
        }
        assert normalize(local, mcp_tool.input_schema.get("$defs", {})) == normalize(
            function["parameters"]["properties"]
        )


def test_mcp_import_does_not_load_provider_routing_logic():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import medsegagent.mcp_server; "
                "assert 'medsegagent.agent' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_mcp_catalog_needs_no_input_or_session():
    result = asyncio.run(mcp_server.get_capabilities(task="total_v3"))
    assert result["ok"] and result["capabilities"]["task"] == "total_v3"
    assert any(row["name"] == "liver" for row in result["capabilities"]["labels"])


def test_mcp_catalog_hides_unavailable_models_and_policy_details():
    result = asyncio.run(mcp_server.get_capabilities(query="heart"))
    assert all(
        row["task"] in catalog.public_task_names() for row in result["capabilities"]["tasks"]
    )
    assert "excluded_task_counts" not in result["capabilities"]
    with pytest.raises(ToolError, match="^Unsupported segmentation task\\.$"):
        asyncio.run(mcp_server.get_capabilities(task="heartchambers_highres"))


@pytest.mark.parametrize("producer", ["total", ["total", "total_v3"]])
def test_mcp_passes_explicit_producer_and_quality_without_remapping(
    monkeypatch, tmp_path, producer
):
    from medsegagent import execution

    monkeypatch.setattr(execution, "TaskExecution", FakeExecution)
    state = mcp_server.SessionState()
    result = asyncio.run(
        mcp_server.segment(
            str(tmp_path / "input.nii"),
            "CT",
            targets=["liver"],
            task=producer,
            quality="fastest",
            ctx=session_context(state),
        )
    )
    assert state.executions[result["execution_id"]].execution.calls == [
        (
            "segment",
            {"targets": ["liver"], "modality": "CT", "task": producer, "quality": "fastest"},
        )
    ]


def test_mcp_keeps_same_target_from_two_producers_and_reuses_each_separately(monkeypatch, tmp_path):
    from test_execution import Backend, make_input

    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    source = str(make_input(tmp_path))

    async def run():
        state = mcp_server.SessionState()
        ctx = session_context(state)
        first = await mcp_server.segment(
            source, "CT", targets=["liver"], ctx=ctx, output_dir=str(tmp_path / "out")
        )
        eid = first["execution_id"]
        for _ in range(2):
            result = await mcp_server.segment(
                source, targets=["liver"], task="total_v3", ctx=ctx, execution_id=eid
            )
            assert result["ok"]
        assert len(result["local_outputs"]) == 2
        selections = [
            row for output in result["local_outputs"] for row in output["requested_regions"]
        ]
        assert {row["task"] for row in selections} == {"total", "total_v3"}
        assert {row["target"] for row in selections} == {"liver"}
        assert len({row["region_id"] for row in selections}) == 2
        assert len({output["path"] for output in result["local_outputs"]}) == 2

    asyncio.run(run())
    assert [row["task"] for row in backend.calls] == ["total", "total_v3"]


def test_ct_and_mr_class_counts_follow_installed_registry():
    assert len(core.task_classes("total")) == 117
    assert len(core.task_classes("total_mr")) == 50


@pytest.mark.parametrize("tool", ["segment_ct", "segment_mr"])
def test_expected_core_failure_is_llm_readable_tool_error(tool, monkeypatch):
    async def fail(**kwargs):
        raise core.SegmentationError("targets must be a non-empty list")

    monkeypatch.setattr(core, "segment", fail)
    with pytest.raises(ToolError, match="non-empty") as error:
        asyncio.run(getattr(mcp_server, tool)(input_path="/synthetic.nii", targets=[]))
    assert not isinstance(error.value, UnexpectedToolError)


def test_adapter_routes_to_shared_core(monkeypatch):
    calls = []

    async def record(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(core, "segment", record)
    asyncio.run(mcp_server.segment_mr("/synthetic.nii", "/private/runs", ["liver"]))
    assert calls == [
        {
            "task": "total_mr",
            "input_path": "/synthetic.nii",
            "output_dir": "/private/runs",
            "targets": ["liver"],
        }
    ]


@pytest.mark.parametrize("task", ["lung_nodules", "liver_lesions"])
def test_lesion_adapter_limits_targets_and_defaults_to_lesion(task, monkeypatch):
    calls = []

    async def record(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(core, "segment", record)
    tool = getattr(mcp_server, "segment_" + task)
    asyncio.run(tool("/synthetic.nii"))
    assert calls[0]["task"] == task and calls[0]["targets"] == [task]
    for bad in ([], [" "], ["lung"], [task, "liver"]):
        with pytest.raises(ToolError):
            asyncio.run(tool("/synthetic.nii", targets=bad))
    assert len(calls) == 1


class FakeExecution:
    def __init__(self, **kwargs):
        self.context = kwargs
        self.modality = kwargs["modality"]
        self.calls = []

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        return {"ok": True, "regions": [{"region_id": "synthetic-region"}]}

    def export_result(self):
        return {
            "outputs": [
                {"target": "lungs", "region_id": "synthetic-region", "artifact_id": "mask"}
            ],
            "regions": [{"id": "synthetic-region", "values": [1]}],
            "artifacts": [
                {
                    "id": "mask",
                    "name": "lungs",
                    "path": "/synthetic/output.nii.gz",
                    "labels": [],
                    "private": "hidden",
                },
                {
                    "id": "unrequested",
                    "name": "intermediate",
                    "path": "/private/intermediate.nii.gz",
                    "labels": [],
                },
            ],
            "backend_results": [{"process_log": "/private/log"}],
        }


def session_context(state):
    return Context(request_context=SimpleNamespace(lifespan_context=state))


@pytest.mark.parametrize("derived", [False, True])
def test_mcp_local_selectors_keep_shared_model_usage_metadata(monkeypatch, tmp_path, derived):
    from medsegagent import execution

    metadata = {
        "task": "brain_aneurysm",
        "quality": "standard",
        "usage_license": "CC-BY-NC-4.0",
        "requirements": ["TOF MRI only.", "Noncommercial use; preserve attribution."],
    }
    if derived:
        metadata = {**metadata, "task": "composition", "quality": None, "model_sources": [metadata]}

    class AttributedExecution(FakeExecution):
        def export_result(self):
            result = super().export_result()
            result["artifacts"][0].update(metadata)
            result["regions"][0].update(metadata)
            result["outputs"][0].update(metadata)
            return result

    monkeypatch.setattr(execution, "TaskExecution", AttributedExecution)

    async def run():
        return await mcp_server.segment(
            str(tmp_path / "source.nii"),
            "MR",
            targets=["brain_aneurysm"],
            task="brain_aneurysm",
            quality="standard",
            ctx=session_context(mcp_server.SessionState()),
        )

    result = asyncio.run(run())
    output = result["local_outputs"][0]
    selector = output["requested_regions"][0]
    assert {key: output[key] for key in metadata} == metadata
    assert {key: selector[key] for key in metadata} == metadata
    assert "private" not in output


def test_mcp_sdk_preserves_explicit_failed_attempt_replacement(monkeypatch, tmp_path):
    from medsegagent import execution

    monkeypatch.setattr(execution, "TaskExecution", FakeExecution)
    prior = {"task": "total", "target": "liver", "quality": "fast"}

    async def run():
        state = mcp_server.SessionState()
        response = await mcp_server.mcp.call_tool(
            "segment",
            {
                "input_path": str(tmp_path / "input.nii"),
                "modality": "CT",
                "targets": ["liver"],
                "task": "total_v3",
                "quality": "standard",
                "supersedes": [prior],
            },
            context=session_context(state),
        )
        binding = state.executions[response.structured_content["execution_id"]]
        assert binding.execution.calls == [
            (
                "segment",
                {
                    "modality": "CT",
                    "targets": ["liver"],
                    "task": "total_v3",
                    "quality": "standard",
                    "supersedes": [prior],
                },
            )
        ]

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [{"task": "total", "target": "liver"}],
        [{"task": "total", "target": "liver", "quality": "fast", "path": "/private"}],
    ],
)
def test_mcp_sdk_rejects_invalid_failed_attempt_selectors(monkeypatch, tmp_path, bad):
    from medsegagent import execution

    def forbidden(**kwargs):
        pytest.fail("Invalid SDK arguments must not construct an execution")

    monkeypatch.setattr(execution, "TaskExecution", forbidden)

    async def run():
        with pytest.raises(ToolError):
            await mcp_server.mcp.call_tool(
                "segment",
                {
                    "input_path": str(tmp_path / "input.nii"),
                    "targets": ["liver"],
                    "modality": "CT",
                    "supersedes": bad,
                },
                context=session_context(mcp_server.SessionState()),
            )

    asyncio.run(run())


def test_mcp_detection_leaves_choice_to_segment_then_reuses_only_matching_modality(
    monkeypatch, tmp_path
):
    from medsegagent import execution

    class DetectingExecution(FakeExecution):
        async def call(self, name, arguments):
            if name == "detect_modality":
                assert arguments == {}
            elif name == "segment" and self.modality is None:
                self.modality = arguments["modality"]
            return await super().call(name, arguments)

    monkeypatch.setattr(execution, "TaskExecution", DetectingExecution)

    async def run():
        state = mcp_server.SessionState()
        ctx = session_context(state)
        source = str(tmp_path / "input.nii")
        response = await mcp_server.mcp.call_tool(
            "detect_modality", {"input_path": source}, context=ctx
        )
        eid = response.structured_content["execution_id"]
        binding = state.executions[eid]
        assert binding.execution.context["modality"] is None
        assert binding.execution.modality is None
        await mcp_server.mcp.call_tool("detect_modality", {"execution_id": eid}, context=ctx)
        for declaration in ("CT", None, "CT"):
            result = await mcp_server.segment(
                source, declaration, targets=["liver"], ctx=ctx, execution_id=eid
            )
            assert result["execution_id"] == eid
        assert len(state.executions) == 1
        assert [name for name, _ in binding.execution.calls] == [
            "detect_modality",
            "detect_modality",
            "segment",
            "segment",
            "segment",
        ]
        assert binding.execution.calls[2][1] == {"targets": ["liver"], "modality": "CT"}
        assert binding.execution.calls[3][1] == {"targets": ["liver"]}
        with pytest.raises(ToolError, match="different input or modality"):
            await mcp_server.detect_modality(ctx=ctx, execution_id=eid, modality="MR")
        with pytest.raises(ToolError, match="different input"):
            await mcp_server.detect_modality(
                ctx=ctx, execution_id=eid, input_path=str(tmp_path / "other.nii")
            )
        with pytest.raises(ToolError, match="output_dir cannot change"):
            await mcp_server.detect_modality(
                ctx=ctx, execution_id=eid, output_dir=str(tmp_path / "elsewhere")
            )
        with pytest.raises(ToolError, match="Unknown execution_id"):
            await mcp_server.detect_modality(
                ctx=session_context(mcp_server.SessionState()), execution_id=eid
            )
        assert len(binding.execution.calls) == 5

    asyncio.run(run())


def test_mcp_detection_requires_input_only_for_a_new_execution(monkeypatch):
    from medsegagent import execution

    def forbidden(**kwargs):
        raise AssertionError("Missing input must not create an execution")

    monkeypatch.setattr(execution, "TaskExecution", forbidden)
    state = mcp_server.SessionState()
    with pytest.raises(ToolError, match="input_path is required"):
        asyncio.run(mcp_server.detect_modality(ctx=session_context(state)))
    assert not state.executions


def test_mcp_unknown_segmentation_preserves_execution_for_detection(monkeypatch, tmp_path):
    from medsegagent import execution

    class UnknownExecution(FakeExecution):
        async def call(self, name, arguments):
            self.calls.append((name, arguments))
            return {"ok": False, "error": {"code": "MODALITY_REQUIRED"}}

        def export_result(self):
            return {}

    monkeypatch.setattr(execution, "TaskExecution", UnknownExecution)

    async def run():
        state = mcp_server.SessionState()
        ctx = session_context(state)
        result = await mcp_server.mcp.call_tool(
            "segment",
            {"input_path": str(tmp_path / "input.nii"), "targets": ["liver"]},
            context=ctx,
        )
        data = result.structured_content
        eid = data["execution_id"]
        assert data["ok"] is False
        assert data["error"]["code"] == "MODALITY_REQUIRED"
        assert data["local_outputs"] == []
        assert state.executions[eid].execution.modality is None
        await mcp_server.detect_modality(ctx=ctx, execution_id=eid)
        await mcp_server.segment(
            str(tmp_path / "input.nii"),
            "CT",
            targets=["liver"],
            ctx=ctx,
            execution_id=eid,
        )
        assert state.executions[eid].execution.calls == [
            ("segment", {"targets": ["liver"]}),
            ("detect_modality", {}),
            ("segment", {"targets": ["liver"], "modality": "CT"}),
        ]

    asyncio.run(run())


@pytest.mark.parametrize("detection_status", ["detected", "uncertain"])
def test_mcp_real_execution_uses_detection_as_evidence_then_accepts_explicit_choice(
    monkeypatch, tmp_path, detection_status
):
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    from medsegagent import modality

    source = tmp_path / "synthetic.nii.gz"
    image = nib.Nifti1Image(np.arange(64, dtype=np.float32).reshape(4, 4, 4), np.eye(4))
    image.header.set_xyzt_units("mm")
    nib.save(image, source)
    classifier_calls = []

    def classify(features):
        classifier_calls.append(features)
        return {
            "modality": "CT" if detection_status == "detected" else None,
            "candidate_modality": "CT",
            "source": "totalseg_intensity",
            "status": detection_status,
            "supported": True,
            "limitations": ["ct_mr_only"],
            "vote_agreement": 1.0,
            "private": "must never leave the detector",
        }

    inference_calls = []

    async def synthetic_backend(**kwargs):
        inference_calls.append(kwargs)
        reference = nib.load(kwargs["input_path"])
        mask = np.zeros(reference.shape, dtype=np.uint8)
        mask[0, 0, 0] = 1
        destination = Path(kwargs["output_dir"]) / "synthetic-mask.nii.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(mask, reference.affine, reference.header), destination)
        return {
            "segmentation_path": str(destination),
            "labels": [{"id": 1, "source_id": 5, "name": "liver"}],
        }

    monkeypatch.setattr(modality, "classify_features", classify)
    monkeypatch.setattr(core, "segment", synthetic_backend)

    async def run():
        state = mcp_server.SessionState()
        ctx = session_context(state)
        pending = await mcp_server.segment(
            str(source), targets=["liver"], ctx=ctx, output_dir=str(tmp_path / "out")
        )
        assert pending["ok"] is False and pending["code"] == "MODALITY_REQUIRED"
        eid = pending["execution_id"]
        detected = await mcp_server.detect_modality(ctx=ctx, execution_id=eid)
        assert detected["ok"] is True
        assert detected["modality_detection"]["status"] == detection_status
        assert detected["modality_detection"]["declared_modality"] is None
        assert "private" not in detected["modality_detection"]
        assert detected["local_outputs"] == []
        binding = state.executions[eid]
        assert binding.execution._frozen.is_file()
        assert binding.execution.modality is None
        cached = await mcp_server.detect_modality(ctx=ctx, execution_id=eid)
        assert cached["cached"] is True and cached["modality_detection"]["cached"] is True
        assert len(classifier_calls) == 1
        assert not inference_calls
        chosen = await mcp_server.segment(
            str(source), "CT", targets=["liver"], ctx=ctx, execution_id=eid
        )
        assert chosen["ok"] is True
        assert binding.execution.modality == "CT"
        assert not binding.execution.unresolved_failures
        assert len(inference_calls) == 1 and inference_calls[0]["task"] == "total"
        reused = await mcp_server.segment(str(source), targets=["liver"], ctx=ctx, execution_id=eid)
        assert reused["ok"] is True and len(inference_calls) == 1
        with pytest.raises(ToolError, match="different input or modality"):
            await mcp_server.segment(
                str(source), "MR", targets=["liver"], ctx=ctx, execution_id=eid
            )

    asyncio.run(run())


def test_mcp_declared_modality_detection_returns_the_declaration_without_a_classifier(
    monkeypatch, tmp_path
):
    from medsegagent import modality

    def forbidden(*args, **kwargs):
        raise AssertionError("A user declaration must not invoke modality classification")

    monkeypatch.setattr(modality, "classify_features", forbidden)
    monkeypatch.setattr(modality, "probe_dicom", forbidden)

    async def run():
        state = mcp_server.SessionState()
        result = await mcp_server.detect_modality(
            ctx=session_context(state),
            input_path=str(tmp_path / "declared.nii"),
            modality="CT",
            output_dir=str(tmp_path / "out"),
        )
        report = result["modality_detection"]
        assert result["ok"] is True
        assert report["source"] == "user_declaration"
        assert report["status"] == "provided"
        assert report["modality"] == "CT"
        assert state.executions[result["execution_id"]].execution.modality == "CT"

    asyncio.run(run())


def test_mcp_region_tools_share_only_the_bound_execution(monkeypatch, tmp_path):
    from medsegagent import execution

    monkeypatch.setattr(execution, "TaskExecution", FakeExecution)

    async def run():
        async with mcp_server._lifespan(mcp_server.mcp) as state:
            ctx = session_context(state)
            result = await mcp_server.mcp.call_tool(
                "segment",
                {"input_path": str(tmp_path / "input.nii"), "modality": "CT", "targets": ["lungs"]},
                context=ctx,
            )
            data = result.structured_content
            eid = data["execution_id"]
            binding = state.executions[eid]
            assert data["local_outputs"] == [
                {
                    "id": "mask",
                    "name": "lungs",
                    "path": "/synthetic/output.nii.gz",
                    "labels": [],
                    "requested_regions": [
                        {"region_id": "synthetic-region", "target": "lungs", "label_ids": [1]}
                    ],
                }
            ]
            assert "backend_results" not in data
            await mcp_server.segment(
                str(tmp_path / "input.nii"),
                "CT",
                targets=["lung_nodules"],
                ctx=ctx,
                execution_id=eid,
            )
            await mcp_server.inspect_artifact(eid, ["synthetic-region"], ctx)
            await mcp_server.compose_masks(eid, "union", ["synthetic-region"], "combined", ctx)
            assert [call[0] for call in binding.execution.calls] == [
                "segment",
                "segment",
                "inspect_artifact",
                "compose_masks",
            ]
            assert binding.execution.calls[0][1] == {"targets": ["lungs"], "modality": "CT"}
            with pytest.raises(ToolError, match="different input"):
                await mcp_server.segment(
                    str(tmp_path / "other.nii"), "CT", targets=["lungs"], ctx=ctx, execution_id=eid
                )
            with pytest.raises(ToolError, match="different input"):
                await mcp_server.segment(
                    str(tmp_path / "input.nii"), "MR", targets=["lungs"], ctx=ctx, execution_id=eid
                )
            with pytest.raises(ToolError, match="Unknown execution_id"):
                await mcp_server.inspect_artifact(
                    eid, ["synthetic-region"], session_context(mcp_server.SessionState())
                )
        assert not state.executions

    asyncio.run(run())


def test_mcp_session_capacity_is_bounded_and_does_not_evict_live_references(monkeypatch, tmp_path):
    from medsegagent import execution

    monkeypatch.setattr(execution, "TaskExecution", FakeExecution)
    monkeypatch.setattr(mcp_server, "MAX_EXECUTIONS", 1)

    async def run():
        state = mcp_server.SessionState()
        ctx = session_context(state)
        first = await mcp_server.segment(str(tmp_path / "a.nii"), "CT", targets=["lungs"], ctx=ctx)
        with pytest.raises(ToolError, match="input limit"):
            await mcp_server.segment(str(tmp_path / "b.nii"), "CT", targets=["lungs"], ctx=ctx)
        assert list(state.executions) == [first["execution_id"]]

    asyncio.run(run())


def test_mcp_requested_selectors_match_real_shared_and_composite_file_voxels(monkeypatch, tmp_path):
    """The MCP file table remains truthful when only one of its labels is an output."""
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    from medsegagent import catalog

    source = tmp_path / "synthetic.nii.gz"
    image = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm")
    nib.save(image, source)
    calls = []

    async def synthetic_backend(**kwargs):
        calls.append(kwargs)
        reference = nib.load(kwargs["input_path"])
        native = {name: index for index, name in catalog.native_labels(kwargs["task"]).items()}
        ordered = sorted(kwargs["targets"], key=native.__getitem__)
        values = np.zeros(reference.shape, dtype=np.uint8)
        labels = []
        for index, target in enumerate(ordered, 1):
            values.flat[index] = index
            labels.append({"id": index, "source_id": native[target], "name": target})
        path = Path(kwargs["output_dir"]) / "synthetic-mask.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(values, reference.affine, reference.header), path)
        return {"segmentation_path": str(path), "labels": labels}

    monkeypatch.setattr(core, "segment", synthetic_backend)

    async def run():
        ctx = session_context(mcp_server.SessionState())
        return await mcp_server.segment(
            str(source), "CT", targets=["lungs", "liver"], ctx=ctx, output_dir=str(tmp_path / "out")
        )

    result = asyncio.run(run())
    assert result["ok"] is True
    assert len(calls) == 1
    assert len(result["local_outputs"]) == 2
    requested = {}
    for output in result["local_outputs"]:
        mask = np.asarray(nib.load(output["path"]).dataobj)
        assert set(np.unique(mask)) - {0} == {label["id"] for label in output["labels"]}
        for selected in output["requested_regions"]:
            requested[selected["target"]] = int(np.isin(mask, selected["label_ids"]).sum())
            if selected["target"] == "liver":
                assert len(output["labels"]) == 6
                assert selected["label_ids"] == [1]
                assert (
                    next(label for label in output["labels"] if label["name"] == "liver")[
                        "source_id"
                    ]
                    == 5
                )
                assert np.count_nonzero(mask) == 6  # Selecting every nonzero voxel would be wrong.
    assert requested == {"liver": 1, "lungs": 5}

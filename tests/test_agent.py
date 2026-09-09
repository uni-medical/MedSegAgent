"""Public-object routing and the bounded model/tool feedback loop; all providers are mocked."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from medsegagent import agent, catalog


@pytest.fixture(autouse=True)
def provider_config(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.delenv("MEDSEGAGENT_MAX_MODEL_REQUESTS", raising=False)
    monkeypatch.delenv("MEDSEGAGENT_MAX_TOOL_CALLS", raising=False)


def call(name="segment", arguments=None, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps({"targets": ["liver"]} if arguments is None else arguments),
        },
    }


def final(status="completed", summary="请求的输出已生成。", unresolved=None):
    return {
        "content": json.dumps(
            {
                "status": status,
                "summary": summary,
                "unresolved": [] if unresolved is None else unresolved,
            }
        )
    }


def respond(messages, requests):
    sequence = iter(messages)

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": next(sequence)}]})

    return httpx.MockTransport(handler)


class Execution:
    def __init__(self, feedback=None):
        self.calls = []
        self.has_outputs = False
        self.unresolved_failures = []
        self.feedback = iter(feedback) if feedback is not None else None

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        result = (
            next(self.feedback)
            if self.feedback is not None
            else {
                "ok": True,
                "status": "completed",
                "regions": [
                    {
                        "region_id": "r_1",
                        "artifact_id": "a_1",
                        "target": "liver",
                        "voxels": 8,
                        "volume_ml": 0.008,
                        "empty": False,
                    }
                ],
            }
        )
        if isinstance(result, BaseException):
            raise result
        if result.get("ok") and name in {"segment", "compose_masks"}:
            self.has_outputs = True
        self.unresolved_failures = result.get("unresolved_failures", self.unresolved_failures)
        return result


def test_catalog_full_lung_is_a_fixed_complete_recipe_and_preserves_output_identity():
    resolved = catalog.resolve_targets("CT", ["lungs", "lung_left", "lungs"])
    assert [row["target"] for row in resolved] == ["lungs", "lung_left"]
    assert resolved[0]["task"] == "total"
    assert resolved[0]["native_targets"] == [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ]
    assert resolved[1]["native_targets"] == resolved[0]["native_targets"][:2]
    assert catalog.resolve_targets("MR", ["lungs"])[0]["native_targets"] == [
        "lung_left",
        "lung_right",
    ]


def test_catalog_mixed_targets_and_all_anatomy_do_not_expose_internal_crop():
    resolved = catalog.resolve_targets("CT", ["lungs", "lung_nodules", "liver_lesions"])
    assert [row["task"] for row in resolved] == ["total", "lung_nodules", "liver_lesions"]
    assert "lung" in catalog.native_labels("lung_nodules").values()
    assert catalog.public_native_targets("lung_nodules") == {"lung_nodules"}
    assert "lung" not in catalog.public_targets("CT")
    assert len(catalog.anatomical_targets("CT")) == 117
    assert len(catalog.anatomical_targets("MR")) == 50
    assert not {"lungs", "lung_nodules", "liver_lesions"} & catalog.anatomical_targets("CT")


@pytest.mark.parametrize(
    "modality,targets",
    [
        ("MR", ["lung_nodules"]),
        ("CT", ["lung"]),
        ("CT", ["liver", "pancreatic_tumor"]),
        ("CT", []),
        ("CT", None),
        ("CT", "liver"),
        ("CT", [None]),
        ("PET", ["liver"]),
    ],
)
def test_catalog_rejects_entire_invalid_request(modality, targets):
    with pytest.raises(catalog.CatalogError):
        catalog.resolve_targets(modality, targets)


def test_catalog_rejects_missing_recipe_component(monkeypatch):
    native = catalog.native_labels

    def missing(task):
        return {
            key: value for key, value in native(task).items() if value != "lung_middle_lobe_right"
        }

    monkeypatch.setattr(catalog, "native_labels", missing)
    with pytest.raises(catalog.CatalogError, match="composite"):
        catalog.public_targets("CT")


def test_schema_discovers_labels_on_demand_and_constrains_public_producers():
    ct = {row["function"]["name"]: row["function"] for row in agent.tool_schema("CT")}
    mr = {row["function"]["name"]: row["function"] for row in agent.tool_schema("MR")}
    assert set(ct) == {
        "get_capabilities",
        "detect_modality",
        "segment",
        "inspect_artifact",
        "compose_masks",
    }
    assert "enum" not in ct["segment"]["parameters"]["properties"]["targets"]["items"]
    assert ct["segment"]["parameters"]["properties"]["task"]["anyOf"][0]["enum"] == list(
        catalog.public_task_names("CT")
    )
    assert (
        "lung_nodules" not in mr["segment"]["parameters"]["properties"]["task"]["anyOf"][0]["enum"]
    )
    assert "quality" not in ct["segment"]["parameters"]["required"]
    assert "all anatomy" in ct["segment"]["description"]
    assert "get_capabilities" in ct["segment"]["description"]
    assert (
        "heartchambers_highres"
        not in ct["get_capabilities"]["parameters"]["properties"]["task"]["enum"]
    )
    assert (
        "heartchambers_highres"
        not in ct["segment"]["parameters"]["properties"]["task"]["anyOf"][0]["enum"]
    )
    assert "brain_aneurysm" in mr["segment"]["parameters"]["properties"]["task"]["anyOf"][0]["enum"]
    for function in ct.values():
        assert "input_path" not in function["parameters"]["properties"]
        assert "output_dir" not in function["parameters"]["properties"]


@pytest.mark.parametrize("task", ["total", "total_v3"])
@pytest.mark.parametrize("query", [None, "liver"])
def test_preview_can_query_labels_then_select_a_producer_without_image_execution(task, query):
    requests = []
    arguments = {"task": task, **({"query": query} if query is not None else {})}
    selected = asyncio.run(
        agent.select_tool(
            f"用{task}分割CT肝脏",
            "CT",
            transport=respond(
                [
                    {"tool_calls": [call("get_capabilities", arguments)]},
                    {
                        "tool_calls": [
                            call(
                                arguments={
                                    "task": task,
                                    "targets": ["liver"],
                                    "quality": "standard",
                                }
                            )
                        ]
                    },
                ],
                requests,
            ),
        )
    )
    assert selected.task == task and selected.targets == ["liver"]
    assert len(requests) == 2
    assert {row["function"]["name"] for row in requests[0]["tools"]} == {
        "get_capabilities",
        "segment",
    }
    feedback = json.loads(requests[1]["messages"][-1]["content"])
    assert feedback["capabilities"]["task"] == task
    assert any(row["name"] == "liver" for row in feedback["capabilities"]["labels"])
    if query is not None:
        assert feedback["capabilities"]["matches"] == ["liver"]
        assert feedback["capabilities"]["labels"] == [
            {
                "id": 5,
                "name": "liver",
                "auxiliary": False,
                "display_name_zh": "肝脏",
                "display_name_en": "Liver",
            }
        ]
    else:
        assert len(feedback["capabilities"]["labels"]) == 117


def test_catalog_feedback_keeps_controlled_labels_and_acquisition_requirements_only():
    feedback = agent.safe_capabilities(
        {
            "ok": True,
            "status": "completed",
            "process_log": "/private/log",
            "capabilities": {
                "task": "total",
                "description": "Use CT/MR metadata.",
                "usage_license": "CC-BY-NC-4.0",
                "requirements": ["TOF MRI only"],
                "labels": [
                    {"id": 1, "name": "region", "auxiliary": False, "patient_name": "PRIVATE"}
                ],
                "models_by_speed": {"standard": [12]},
                "weight_readiness": {
                    "standard": {
                        "task": "total",
                        "quality": "standard",
                        "ready": False,
                        "model_ids": [12, 13],
                        "missing_model_ids": [13],
                        "path": "/private/weights",
                    }
                },
                "roi_weight_readiness": {
                    "standard": {
                        "task": "total",
                        "quality": "standard",
                        "ready": False,
                        "model_ids": [12, 13, 14],
                        "missing_model_ids": [14],
                        "path": "/private/roi-weights",
                    }
                },
                "composites": {"lungs": ["lung_left", "lung_right"]},
                "path": "/private/model",
                "private_config": {"token": "SECRET"},
            },
        }
    )
    assert feedback["capabilities"]["description"] == "Use CT/MR metadata."
    assert feedback["capabilities"]["models_by_speed"] == {"standard": [12]}
    assert feedback["capabilities"]["weight_readiness"] == {
        "standard": {
            "task": "total",
            "quality": "standard",
            "ready": False,
            "model_ids": [12, 13],
            "missing_model_ids": [13],
        }
    }
    assert feedback["capabilities"]["labels"] == [{"id": 1, "name": "region", "auxiliary": False}]
    assert feedback["capabilities"]["roi_weight_readiness"] == {
        "standard": {
            "task": "total",
            "quality": "standard",
            "ready": False,
            "model_ids": [12, 13, 14],
            "missing_model_ids": [14],
        }
    }
    assert not any(value in json.dumps(feedback) for value in ("PRIVATE", "SECRET", "/private"))


def test_default_catalog_feedback_omits_policy_but_keeps_acquisition_requirements():
    response = agent.safe_capabilities(
        {"ok": True, "status": "completed", "capabilities": catalog.get_capabilities()}
    )
    directory = response["capabilities"]
    assert "excluded_task_counts" not in directory
    assert {row["task"] for row in directory["tasks"]} == set(catalog.public_task_names())
    assert all("public_service_supported" not in row for row in directory["tasks"])
    brain = next(row for row in directory["tasks"] if row["task"] == "brain_aneurysm")
    assert "usage_license" not in brain
    assert brain["requirements"]


def test_agent_catalog_feedback_drives_an_explicit_producer_selection():
    requests = []
    execution = Execution(
        [
            {
                "ok": True,
                "status": "completed",
                "capabilities": catalog.get_capabilities(task="total_v3"),
            },
            {
                "ok": True,
                "status": "completed",
                "regions": [
                    {
                        "region_id": "liver_v3",
                        "artifact_id": "v3",
                        "target": "liver",
                        "task": "total_v3",
                        "quality": "standard",
                        "voxels": 8,
                    }
                ],
            },
        ]
    )
    outcome = asyncio.run(
        agent.run_agent(
            "用total_v3分割CT肝脏",
            "CT",
            execution,
            transport=respond(
                [
                    {"tool_calls": [call("get_capabilities", {"task": "total_v3"})]},
                    {"tool_calls": [call(arguments={"targets": ["liver"], "task": "total_v3"})]},
                    final(),
                ],
                requests,
            ),
        )
    )
    assert outcome["status"] == "completed"
    assert execution.calls[1] == ("segment", {"targets": ["liver"], "task": "total_v3"})
    assert json.loads(requests[1]["messages"][-1]["content"])["capabilities"]["labels"]
    assert json.loads(requests[2]["messages"][-1]["content"])["regions"][0]["task"] == "total_v3"


def test_catalog_discovery_alone_cannot_establish_completed_segmentation():
    execution = Execution(
        [
            {
                "ok": True,
                "status": "completed",
                "capabilities": catalog.get_capabilities(task="total"),
            }
        ]
    )
    result = asyncio.run(
        agent.run_agent(
            "分割CT肝脏",
            "CT",
            execution,
            transport=respond(
                [{"tool_calls": [call("get_capabilities", {"task": "total"})]}, final()], []
            ),
        )
    )
    assert result["status"] == "failed"
    assert not execution.has_outputs


@pytest.mark.parametrize(
    "text,modality",
    [
        ("分割CT肝脏，不分割病灶。", "CT"),
        ("这份 CT 来自肿瘤患者，请只分割肝脏。", "CT"),
        ("Segment MR liver only, not a lesion.", "MR"),
    ],
)
def test_preview_does_not_reject_background_or_negation_by_global_keyword(text, modality):
    requests = []
    selected = asyncio.run(
        agent.select_tool(
            text,
            transport=respond([{"tool_calls": [call()]}], requests),
        )
    )
    assert selected.tool == "segment" and selected.targets == ["liver"]
    assert selected.modality == modality
    assert "negated instructions" in requests[0]["messages"][0]["content"]


def test_preview_mixed_backend_request_is_still_one_public_tool():
    requests = []
    selected = asyncio.run(
        agent.select_tool(
            "CT双肺和结节",
            transport=respond(
                [
                    {"tool_calls": [call(arguments={"targets": ["lungs", "lung_nodules"]})]},
                ],
                requests,
            ),
        )
    )
    assert selected.tool == "segment" and selected.task is None
    assert selected.targets == ["lungs", "lung_nodules"]
    assert selected.modality == "CT"


@pytest.mark.parametrize(
    "message",
    [
        {"tool_calls": []},
        {
            "tool_calls": [call()],
            "refusal": "The requested operation is unsupported.",
        },
        {"tool_calls": [call("segment_ct")]},
        {"tool_calls": [call(arguments={"targets": ["liver", "bogus"]})]},
        {"tool_calls": [call(arguments={"targets": ["lung_nodules"], "quality": "fast"})]},
        {"tool_calls": [call(arguments={"targets": ["liver"], "input_path": "/private/scan.nii"})]},
    ],
)
def test_preview_invalid_or_partial_selection_starts_nothing(message):
    with pytest.raises(agent.RoutingError) as error:
        asyncio.run(agent.select_tool("CT分割目标", transport=respond([message], [])))
    assert error.value.code == "UNSUPPORTED_REQUEST"


def test_preview_does_not_mistake_quoted_interface_limits_for_a_refusal():
    # A real total_v3 reply quoted our preview instruction while making the correct call.
    result = asyncio.run(
        agent.select_tool(
            "用total_v3分割CT肝脏",
            "CT",
            transport=respond(
                [
                    {
                        "content": "The liver target is supported. Image inspection, modality detection "
                        "and segmentation execution are unavailable in this preview. "
                        "Let me propose the segment action.",
                        "tool_calls": [call(arguments={"task": "total_v3", "targets": ["liver"]})],
                    }
                ],
                [],
            ),
        )
    )
    assert result.task == "total_v3" and result.targets == ["liver"]


def test_preview_serially_answers_a_batch_of_read_only_catalog_queries():
    # A real provider emitted these two queries despite parallel_tool_calls=False.
    requests = []
    result = asyncio.run(
        agent.select_tool(
            "TOF MRI脑动脉瘤分割，非商业科研",
            "MR",
            transport=respond(
                [
                    {
                        "tool_calls": [
                            call("get_capabilities", {"task": "brain_aneurysm", "modality": "MR"}),
                            call(
                                "get_capabilities",
                                {"query": "aneurysm", "modality": "MR"},
                                "query_2",
                            ),
                        ]
                    },
                    {
                        "tool_calls": [
                            call(
                                arguments={"task": "brain_aneurysm", "targets": ["brain_aneurysm"]}
                            )
                        ]
                    },
                ],
                requests,
            ),
        )
    )
    assert result.task == "brain_aneurysm"
    assert len(requests) == 2
    replies = [message for message in requests[1]["messages"] if message["role"] == "tool"]
    assert len(replies) == 2
    assert len({row["tool_call_id"] for row in replies}) == 2
    assert all(json.loads(row["content"])["ok"] for row in replies)


def test_operator_interaction_budgets_have_explicit_defaults_and_hard_upper_bounds(monkeypatch):
    assert agent.interaction_limits() == (24, 64)
    monkeypatch.setenv("MEDSEGAGENT_MAX_MODEL_REQUESTS", "128")
    monkeypatch.setenv("MEDSEGAGENT_MAX_TOOL_CALLS", "256")
    assert agent.interaction_limits() == (128, 256)


@pytest.mark.parametrize(
    "name,value",
    [
        ("MEDSEGAGENT_MAX_MODEL_REQUESTS", "0"),
        ("MEDSEGAGENT_MAX_MODEL_REQUESTS", "129"),
        ("MEDSEGAGENT_MAX_TOOL_CALLS", "257"),
        ("MEDSEGAGENT_MAX_TOOL_CALLS", "not-an-integer"),
    ],
)
def test_invalid_operator_budget_fails_before_provider_contact(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    def forbidden(request):
        pytest.fail("Invalid operator limits must not contact the provider")

    with pytest.raises(agent.RoutingError, match=name):
        asyncio.run(
            agent.run_agent("CT肝脏", "CT", Execution(), transport=httpx.MockTransport(forbidden))
        )


@pytest.mark.parametrize("modality", ["PET", "", [], 1])
def test_invalid_modality_fails_locally_before_provider(modality):
    def forbidden(request):
        pytest.fail("Local modality errors must not contact the provider")

    with pytest.raises(agent.RoutingError, match="Modality must be CT or MR"):
        asyncio.run(
            agent.run_agent(
                "CT肝脏", modality, Execution(), transport=httpx.MockTransport(forbidden)
            )
        )


def test_feedback_drives_inspection_composition_and_completion_with_ordinary_messages():
    requests, progress = [], []
    execution = Execution()
    messages = [
        {"tool_calls": [call(arguments={"targets": ["lungs", "lung_nodules"]})]},
        {"tool_calls": [call("inspect_artifact", {"region_ids": ["r_1"]}, "call_2")]},
        {
            "tool_calls": [
                call(
                    "compose_masks",
                    {"operation": "union", "region_ids": ["r_1"], "name": "Requested region"},
                    "call_3",
                )
            ]
        },
        final(),
    ]

    async def on_progress(event):
        progress.append(event)

    result = asyncio.run(
        agent.run_agent(
            "分割CT双肺与结节并检查和生成组合结果",
            "CT",
            execution,
            on_progress,
            transport=respond(messages, requests),
        )
    )
    assert result["status"] == "completed"
    assert result["model_requests"] == 4 and result["tool_calls"] == 3
    assert [name for name, _ in execution.calls] == [
        "segment",
        "inspect_artifact",
        "compose_masks",
    ]
    history = requests[-1]["messages"]
    assert [row["role"] for row in history] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert [row["tool_call_id"] for row in history if row["role"] == "tool"] == [
        "call_1",
        "call_2",
        "call_3",
    ]
    assert json.loads(history[3]["content"])["regions"][0]["volume_ml"] == 0.008
    assert any(item["phase"] == "observed" for item in progress)


def test_provider_feedback_contains_observations_but_no_paths_names_logs_or_secrets():
    feedback = {
        "ok": True,
        "status": "completed",
        "segmentation_path": "/private/patient/scan.nii",
        "api_key": "never-send-this-key",
        "message": "Private patient details",
        "regions": [
            {
                "region_id": "r_1",
                "target": "liver",
                "voxels": 20,
                "volume_ml": 0.02,
                "input_path": "/private/input.nii",
                "name": "Patient identifying file name",
            }
        ],
        "process_log": "secret stderr",
        "image_bytes": "private bytes",
    }
    requests = []
    result = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            Execution([feedback]),
            transport=respond([{"tool_calls": [call()]}, final()], requests),
        )
    )
    assert result["status"] == "completed"
    serialized = json.dumps(requests[-1])
    for forbidden in (
        "/private",
        "never-send",
        "Patient identifying",
        "Private patient",
        "secret stderr",
    ):
        assert forbidden not in serialized
    assert "volume_ml" in serialized and "r_1" in serialized


def test_known_tool_failure_can_be_revised_using_feedback():
    requests = []
    execution = Execution(
        [
            {"ok": False, "status": "failed", "code": "INVALID_QUALITY", "retryable": False},
            {"ok": True, "status": "completed", "unresolved_failures": []},
        ]
    )
    result = asyncio.run(
        agent.run_agent(
            "CT肺结节",
            "CT",
            execution,
            transport=respond(
                [
                    {
                        "tool_calls": [
                            call(arguments={"targets": ["lung_nodules"], "quality": "fast"})
                        ]
                    },
                    {
                        "tool_calls": [
                            call(arguments={"targets": ["lung_nodules"]}, call_id="call_2")
                        ]
                    },
                    final(),
                ],
                requests,
            ),
        )
    )
    assert result["status"] == "completed"
    assert "INVALID_QUALITY" in requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize(
    "feedback",
    [
        None,
        [
            {
                "ok": False,
                "status": "failed",
                "code": "INFERENCE_FAILED",
                "unresolved_failures": [{"code": "INFERENCE_FAILED"}],
            }
        ],
    ],
)
def test_completed_text_cannot_override_absent_or_failed_execution(feedback):
    messages = [final()] if feedback is None else [{"tool_calls": [call()]}, final()]
    result = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            Execution(feedback),
            transport=respond(messages, []),
        )
    )
    assert result["status"] == "failed" and result["unresolved"]


def test_partial_results_require_needs_input_and_keep_unresolved_requirement():
    result = asyncio.run(
        agent.run_agent(
            "CT分割肝脏和不支持的病灶",
            "CT",
            Execution(),
            transport=respond(
                [
                    {"tool_calls": [call()]},
                    final("needs_input", "肝脏结果已生成，另一个对象需要澄清。", ["未支持的病灶"]),
                ],
                [],
            ),
        )
    )
    assert result["status"] == "needs_input" and result["unresolved"] == ["未支持的病灶"]


def test_unknown_exception_is_not_leaked_and_cannot_be_hidden_by_later_success():
    requests = []
    execution = Execution([RuntimeError("/private/patient API_KEY=secret"), {"ok": True}])
    result = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            execution,
            transport=respond(
                [
                    {"tool_calls": [call()]},
                    {"tool_calls": [call(call_id="call_2")]},
                    final(),
                ],
                requests,
            ),
        )
    )
    assert result["status"] == "failed"
    assert "TOOL_FAILED" in requests[1]["messages"][-1]["content"]
    assert "API_KEY" not in json.dumps(requests) and "/private" not in json.dumps(requests)


def test_cancel_propagates_without_another_model_call():
    requests = []
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            agent.run_agent(
                "CT肝脏",
                "CT",
                Execution([asyncio.CancelledError()]),
                transport=respond([{"tool_calls": [call()]}], requests),
            )
        )
    assert len(requests) == 1


def test_action_budget_prevents_unbounded_tool_execution(monkeypatch):
    monkeypatch.setattr(agent, "MAX_TOOL_CALLS", 2)
    execution = Execution()
    result = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            execution,
            transport=respond(
                [
                    {"tool_calls": [call()]},
                    {"tool_calls": [call()]},
                    {"tool_calls": [call()]},
                ],
                [],
            ),
        )
    )
    assert result["status"] == "failed" and result["tool_calls"] == 2
    assert len(execution.calls) == 2


def test_model_budget_bounds_invalid_final_repair(monkeypatch):
    monkeypatch.setattr(agent, "MAX_MODEL_REQUESTS", 2)
    requests = []
    result = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            Execution(),
            transport=respond(
                [
                    {"content": "not json"},
                    {"content": "still not json"},
                ],
                requests,
            ),
        )
    )
    assert result["status"] == "failed" and result["model_requests"] == 2
    assert len(requests) == 2


def test_provider_http_errors_never_return_provider_body():
    def handler(request):
        return httpx.Response(500, text="/private/provider secret configuration")

    with pytest.raises(agent.RoutingError) as error:
        asyncio.run(
            agent.run_agent("CT肝脏", "CT", Execution(), transport=httpx.MockTransport(handler))
        )
    assert "HTTP 500" in str(error.value) and "private" not in str(error.value)


def test_final_json_mode_is_separate_from_native_tool_calls():
    requests = []
    result = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            Execution(),
            transport=respond(
                [
                    {"tool_calls": [call()]},
                    {"content": "已完成。\n" + final()["content"]},
                    final(),
                ],
                requests,
            ),
        )
    )
    assert result["status"] == "completed"
    assert result["model_requests"] == 3 and result["tool_calls"] == 1
    assert all("tools" in item and "response_format" not in item for item in requests[:2])
    assert requests[-1]["response_format"] == {"type": "json_object"}
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & requests[-1].keys()
    assert "were not executed" in requests[-1]["messages"][-1]["content"]


def test_textual_tool_markup_never_executes_or_proves_completion():
    execution = Execution()
    result = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            execution,
            transport=respond(
                [
                    {"content": '<tool_calls><invoke name="segment">liver</invoke></tool_calls>'},
                    final(),
                ],
                [],
            ),
        )
    )
    assert result["status"] == "failed"
    assert execution.calls == []


def test_final_response_repair_does_not_spend_all_turns_on_malformed_text():
    result = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            Execution(),
            transport=respond([{"content": "not JSON"}, {"content": "still not JSON"}], []),
        )
    )
    assert result["status"] == "failed" and result["model_requests"] == 2


def test_textual_tool_markup_cannot_complete_after_a_successful_earlier_action():
    result = asyncio.run(
        agent.run_agent(
            "CT双肺并检查结果",
            "CT",
            Execution(),
            transport=respond(
                [
                    {"tool_calls": [call(arguments={"targets": ["lungs"]})]},
                    {"content": '<｜｜DSML｜｜tool_calls><invoke name="inspect_artifact">'},
                ],
                [],
            ),
        )
    )
    assert result["status"] == "failed" and result["tool_calls"] == 1


def test_final_json_repair_keeps_execution_failure_guard():
    result = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            Execution([{"ok": False, "unresolved_failures": ["failed"]}]),
            transport=respond(
                [
                    {"tool_calls": [call()]},
                    {"content": "Done."},
                    final(),
                ],
                [],
            ),
        )
    )
    assert result["status"] == "failed" and result["model_requests"] == 3


def test_no_extra_json_request_beyond_model_budget(monkeypatch):
    monkeypatch.setattr(agent, "MAX_MODEL_REQUESTS", 1)
    requests = []
    result = asyncio.run(
        agent.run_agent(
            "CT肝脏",
            "CT",
            Execution(),
            transport=respond([{"content": "Done."}], requests),
        )
    )
    assert result["status"] == "failed" and len(requests) == 1

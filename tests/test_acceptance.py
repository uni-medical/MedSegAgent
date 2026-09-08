"""Run the external canary against real HTTP contracts with synthetic model output."""

import importlib.util
import json
from pathlib import Path

import httpx
import nibabel as nib
import pytest
from starlette.testclient import TestClient
from test_execution import Backend, make_input

from medsegagent import agent, core, upload_sessions, web

MODULE = Path(__file__).parents[1] / "ops" / "acceptance.py"
spec = importlib.util.spec_from_file_location("acceptance_canary", MODULE)
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


@pytest.mark.parametrize("access", ["public", "guest"])
@pytest.mark.parametrize("chunked", [False, True])
def test_canary_uses_no_token_and_checks_multisource_and_named_composition(
    tmp_path, monkeypatch, access, chunked
):
    monkeypatch.delenv("MEDSEGAGENT_TOKENS_JSON", raising=False)
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    if chunked:
        monkeypatch.setattr(web, "SINGLE_UPLOAD_BYTES", 300)
        monkeypatch.setattr(upload_sessions, "CHUNK_BYTES", 512)

    async def run(text, modality, execution, **kwargs):
        initial = await execution.call("segment", {"targets": ["liver", "spleen"]})
        assert initial["ok"]
        assert (
            await execution.call(
                "segment", {"task": "total_v3", "quality": "standard", "targets": ["liver"]}
            )
        )["ok"]
        assert (
            await execution.call(
                "compose_masks",
                {
                    "operation": "union",
                    "name": "肝脾 合并",
                    "region_ids": [region["region_id"] for region in initial["regions"]],
                },
            )
        )["ok"]
        return {"status": "completed", "summary": "Three verified outputs", "unresolved": []}

    monkeypatch.setattr(agent, "run_agent", run)
    compressed = make_input(tmp_path)
    source = tmp_path / "synthetic.nii"
    nib.save(nib.load(compressed), source)
    requests = []
    with TestClient(web.create_app(tmp_path / "service", "http://localhost")) as service:

        def bridge(request):
            assert "authorization" not in request.headers
            requests.append((request.method, request.url.path, "cookie" in request.headers))
            # The outer HTTPX clients own their separate jars; this ASGI transport
            # must not lend an earlier guest's cookie to an anonymous request.
            service.cookies.clear()
            response = service.request(
                request.method,
                request.url.path,
                headers=dict(request.headers),
                content=request.read(),
            )
            return httpx.Response(
                response.status_code, headers=response.headers, content=response.content
            )

        report = acceptance.run_canary(
            url="http://localhost",
            input_path=source,
            output_dir=tmp_path / "results",
            access=access,
            transport=httpx.MockTransport(bridge),
        )
    assert report["access"] == access
    assert report["upload_mode"] == ("chunked" if chunked else "single")
    assert report["checks"]["get_task_after_disconnect"]
    assert report["checks"]["access_boundary"] and report["checks"]["idempotency"]
    assert report["checks"]["verified_tasks"] == 2
    assert len(report["tasks"]) == 2 and len(backend.calls) == 4
    for task in report["tasks"]:
        assert len(task["outputs"]) == 3
        assert sum(len(output["class_files"]) for output in task["outputs"]) == 4
        assert any(output["targets"] == ["肝脾 合并"] for output in task["outputs"])
        assert all(output["geometry_consistent"] for output in task["outputs"])
    prefix = "/a2a" if access == "public" else "/api"
    upload_endpoint = prefix + ("/upload-sessions" if chunked else "/uploads")
    assert any(method == "POST" and path == upload_endpoint for method, path, _ in requests)
    if access == "public":
        assert not any(cookie for _, _, cookie in requests)
        assert not any(path == "/api/auth/guest" for _, path, _ in requests)
    else:
        assert sum(path == "/api/auth/guest" for _, path, _ in requests) == 2
    persisted = json.loads((tmp_path / "results" / "report.json").read_text())
    assert persisted == report
    assert "medseg_session" not in json.dumps(report)

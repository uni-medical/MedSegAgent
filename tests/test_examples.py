"""Shared examples become private uploads without starting segmentation."""

import asyncio
import hashlib
import json
import time

import httpx
import nibabel as nib
import numpy as np
import pytest
from auth_helpers import ALICE, BOB
from auth_helpers import create_test_app as create_app
from starlette.testclient import TestClient
from test_execution import Backend

from medsegagent import agent, core, examples


@pytest.fixture
def configured(tmp_path):
    directory = tmp_path / "examples"
    directory.mkdir()
    cases = []
    for index in range(3):
        name = f"case-{index}"
        path = directory / (name + ".nii.gz")
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.int16), np.eye(4)), path)
        (directory / (name + ".png")).write_bytes(b"preview")
        (directory / (name + "-LICENSE.txt")).write_text("Example data license: CC BY 4.0")
        cases.append(
            {
                "id": name,
                "title": "Example CT",
                "modality": "CT",
                "description": "Small example",
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "filename": path.name,
                "preview": name + ".png",
                "license_file": name + "-LICENSE.txt",
                "prompts": [{"label": "Liver", "text": "Segment the liver in this CT"}],
                "attribution": {
                    "label": "Example author",
                    "url": "https://example.org/data",
                    "license": "CC BY 4.0",
                },
            }
        )
    (directory / "manifest.json").write_text(json.dumps({"cases": cases}))
    return tmp_path, directory, cases


def test_all_identities_can_open_examples_as_owned_reusable_uploads(configured, monkeypatch):
    root, _, cases = configured
    calls = []

    async def forbidden_inference(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Examples must not start inference")

    monkeypatch.setattr(core, "segment", forbidden_inference)
    with TestClient(create_app(root, "http://localhost")) as client:
        assert client.get("/api/config").status_code == 401
        assert client.get("/api/examples/case-0/preview").status_code == 401
        assert client.get("/api/examples/case-0/license").status_code == 401
        assert client.post("/api/examples/case-0").status_code == 401
        config = client.get("/api/config", headers=ALICE).json()
        assert len(config["examples"]) == 3
        assert not {"tools", "model"} & config.keys()
        assert "filename" not in json.dumps(config) and "sha256" not in json.dumps(config)
        assert config["capabilities"]["summary"]
        assert client.get(config["examples"][0]["preview_url"], headers=BOB).status_code == 200
        first = client.post("/api/examples/case-0", headers=ALICE)
        assert first.status_code == 201
        first = first.json()
        assert first["example_id"] == "case-0"
        assert first["shape"] == [8, 8, 8]
        repeated = client.post("/api/examples/case-0", headers=ALICE).json()
        second = client.post("/api/examples/case-0", headers=BOB).json()
        assert repeated["id"] == first["id"] != second["id"]
        for user, record in ((ALICE, first), (BOB, second)):
            response = client.get(f"/api/uploads/{record['id']}/file", headers=user)
            assert hashlib.sha256(response.content).hexdigest() == cases[0]["sha256"]
            assert response.headers["Link"] == (
                '<http://localhost/api/examples/case-0/license>; rel="license"'
            )
            license = client.get("/api/examples/case-0/license", headers=user)
            assert license.status_code == 200
            assert license.text == "Example data license: CC BY 4.0"
            assert 'filename="case-0-LICENSE.txt"' in license.headers["content-disposition"]
            assert client.get("/api/tasks", headers=user).json() == []
        assert client.get(f"/api/uploads/{first['id']}/file", headers=BOB).status_code == 404
        assert not calls
        assert client.delete(f"/api/uploads/{first['id']}", headers=ALICE).status_code == 200
        reopened = client.post("/api/examples/case-0", headers=ALICE).json()
        assert reopened["id"] != first["id"]


@pytest.mark.parametrize(
    ("text", "parameter", "expected", "source"),
    [
        ("分割肝脏", None, "MR", "agent"),
        ("分割这份CT的肝脏", None, "CT", "agent"),
        ("上一份是MR，本次上传的图像请先判断后分割肝脏", None, "CT", "agent"),
        ("这份影像不是MR，请分割肝脏", None, "CT", "agent"),
        ("分割肝脏", "CT", "CT", "parameter"),
    ],
)
def test_example_source_is_preserved_but_user_declaration_takes_precedence(
    configured, monkeypatch, text, parameter, expected, source
):
    from medsegagent import modality

    root, directory, cases = configured
    cases[0]["modality"] = "MR"
    (directory / "manifest.json").write_text(json.dumps({"cases": cases}))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    detector_calls = []
    monkeypatch.setattr(modality, "classify_features", lambda values: detector_calls.append(values))
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    requests = []

    def provider(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            initial = json.loads(payload["messages"][1]["content"])
            assert initial["modality"] == parameter
            if parameter is None:
                assert initial["modality_hint"] == {"modality": "MR", "source": "example_manifest"}
            else:
                assert "modality_hint" not in initial
            message = {
                "tool_calls": [
                    {
                        "id": "segment_1",
                        "type": "function",
                        "function": {
                            "name": "segment",
                            "arguments": json.dumps({"targets": ["liver"], "modality": expected}),
                        },
                    }
                ]
            }
        else:
            message = {
                "content": json.dumps(
                    {
                        "status": "completed",
                        "summary": "肝脏已分割",
                        "unresolved": [],
                    }
                )
            }
        return httpx.Response(200, json={"choices": [{"message": message}]})

    original = agent.run_agent

    async def run(*args, **kwargs):
        return await original(*args, **kwargs, transport=httpx.MockTransport(provider))

    monkeypatch.setattr(agent, "run_agent", run)
    with TestClient(create_app(root, "http://localhost")) as client:
        upload = client.post("/api/examples/case-0", headers=ALICE).json()
        assert upload["source_modality"] == "MR"
        sent = client.post(
            "/api/tasks",
            headers=ALICE,
            json={
                "upload_id": upload["id"],
                "text": text,
                "modality": parameter,
                "message_id": "example-source-modality",
            },
        )
        assert sent.status_code == 202
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = client.get("/api/tasks/" + sent.json()["id"], headers=ALICE).json()
            if row["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert row["status"] == "completed", row
        assert row["modality"] == expected and row["modality_source"] == source
        assert row["result"].get("modality_detection") is None
        assert not detector_calls
        assert backend.calls[0]["task"] == ("total" if expected == "CT" else "total_mr")


@pytest.mark.parametrize("damage", ["hash", "symlink", "missing"])
def test_invalid_example_does_not_leave_an_upload(configured, damage):
    root, directory, cases = configured
    path = directory / cases[0]["filename"]
    if damage == "hash":
        path.write_bytes(b"corrupt")
    elif damage == "missing":
        path.unlink()
    else:
        private = root / "private.txt"
        private.write_text("private-marker")
        path.unlink()
        path.symlink_to(private)
    with TestClient(create_app(root, "http://localhost")) as client:
        response = client.post("/api/examples/case-0", headers=ALICE)
        assert response.status_code == 503
        assert "private-marker" not in response.text
        assert not list((root / "uploads").iterdir())
        service = client.app.state.service
        assert service.db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0] == 0
        assert not service.uploading
        assert client.post("/api/examples/case-1", headers=ALICE).status_code == 201


@pytest.mark.parametrize("path", ["unknown", "..%2Fprivate", "%2Fetc%2Fpasswd"])
def test_example_ids_cannot_select_arbitrary_files(configured, path):
    root, _, _ = configured
    with TestClient(create_app(root, "http://localhost")) as client:
        assert client.post("/api/examples/" + path, headers=ALICE).status_code == 404
        assert client.get("/api/examples/" + path + "/license", headers=ALICE).status_code == 404


@pytest.mark.parametrize("damage", ["traversal", "symlink"])
def test_example_license_rejects_unsafe_installed_paths(configured, damage):
    root, directory, cases = configured
    private = root / "private.txt"
    private.write_text("private-marker")
    if damage == "traversal":
        cases[0]["license_file"] = "../private.txt"
        (directory / "manifest.json").write_text(json.dumps({"cases": cases}))
    else:
        license = directory / cases[0]["license_file"]
        license.unlink()
        license.symlink_to(private)
    with TestClient(create_app(root, "http://localhost")) as client:
        response = client.get("/api/examples/case-0/license", headers=ALICE)
        assert response.status_code == 503
        assert "private-marker" not in response.text


def test_example_copy_rejects_extra_bytes_before_writing(configured, monkeypatch):
    root, directory, cases = configured
    source = directory / cases[0]["filename"]
    source.write_bytes(source.read_bytes() + b"x" * (1024 * 1024))
    copied_sizes = []
    remove = examples.shutil.rmtree

    def record_cleanup(path, **kwargs):
        copied_sizes.extend(file.stat().st_size for file in path.iterdir())
        remove(path, **kwargs)

    monkeypatch.setattr(examples.shutil, "rmtree", record_cleanup)
    with TestClient(create_app(root, "http://localhost")) as client:
        response = client.post("/api/examples/case-0", headers=ALICE)
        assert response.status_code == 503
        assert copied_sizes == [0]
        assert not client.app.state.service.uploading
        assert not list((root / "uploads").iterdir())


def test_example_timeout_reclaims_reservation_and_allows_retry(configured, monkeypatch):
    root, _, _ = configured
    monkeypatch.setattr(examples, "OPEN_TIMEOUT_SECONDS", 0.05)
    with TestClient(create_app(root, "http://localhost")) as client:
        service = client.app.state.service
        finish = service.finish_upload

        async def stalled(*args):
            await asyncio.sleep(5)

        monkeypatch.setattr(service, "finish_upload", stalled)
        response = client.post("/api/examples/case-0", headers=ALICE)
        assert response.status_code == 408
        assert response.json()["error"]["code"] == "EXAMPLE_TIMEOUT"
        assert not service.uploading
        assert service.db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0] == 0
        assert not list((root / "uploads").iterdir())
        monkeypatch.setattr(service, "finish_upload", finish)
        monkeypatch.setattr(examples, "OPEN_TIMEOUT_SECONDS", 180)
        assert client.post("/api/examples/case-0", headers=ALICE).status_code == 201


def test_unsupported_request_has_no_inference_or_artifact(configured, monkeypatch):
    root, _, _ = configured
    calls = []

    async def refuse(*args, **kwargs):
        raise agent.unsupported_request()

    async def inference(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(agent, "run_agent", refuse)
    monkeypatch.setattr(core, "segment", inference)
    with TestClient(create_app(root, "http://localhost")) as client:
        upload = client.post("/api/examples/case-0", headers=ALICE).json()
        sent = client.post(
            "/api/tasks",
            headers=ALICE,
            json={
                "upload_id": upload["id"],
                "text": "分割 CT 中的脑肿瘤",
                "message_id": "refusal-one",
            },
        ).json()
        for _ in range(30):
            task = client.get("/api/tasks/" + sent["id"], headers=ALICE).json()
            if task["status"] == "failed":
                break
            client.portal.call(asyncio.sleep, 0.01)
        assert task["error"]["code"] == "UNSUPPORTED_REQUEST"
        assert task["input"]["example_id"] == "case-0"
        assert task["result"] is None and task.get("files", []) == []
        assert not calls
        a2a = client.get(
            "/a2a/v1/tasks/" + sent["id"], headers={**ALICE, "A2A-Version": "1.0"}
        ).json()
        assert a2a["status"]["state"] == "TASK_STATE_FAILED"
        assert "没有启动分割" in a2a["status"]["message"]["parts"][0]["text"]
        assert "不会替换目标" in a2a["status"]["message"]["parts"][0]["text"]
        assert not a2a.get("artifacts")
        assert "total" not in a2a["status"]["message"]["parts"][0]["text"].lower()


@pytest.mark.parametrize("stored_modality", [None, "MR"])
@pytest.mark.parametrize("damage_source", [False, True])
def test_example_reopens_legacy_or_changed_declaration_only_through_verified_copy(
    configured, stored_modality, damage_source
):
    root, directory, cases = configured
    with TestClient(create_app(root, "http://localhost")) as client:
        old = client.post("/api/examples/case-0", headers=ALICE).json()
        service = client.app.state.service
        row = service.db.execute("SELECT data FROM uploads WHERE id=?", (old["id"],)).fetchone()
        legacy = json.loads(row["data"])
        if stored_modality is None:
            legacy.pop("source_modality")
        else:
            legacy["source_modality"] = stored_modality
        with service.db:
            service.db.execute(
                "UPDATE uploads SET data=? WHERE id=?", (json.dumps(legacy), old["id"])
            )
        if damage_source:
            source = directory / cases[0]["filename"]
            original = source.read_bytes()
            source.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

        response = client.post("/api/examples/case-0", headers=ALICE)
        if damage_source:
            assert response.status_code == 503
            assert service.db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0] == 1
        else:
            assert response.status_code == 201
            refreshed = response.json()
            assert refreshed["id"] != old["id"]
            assert refreshed["source_modality"] == "CT"
            assert (
                client.post("/api/examples/case-0", headers=ALICE).json()["id"] == refreshed["id"]
            )
            assert service.db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0] == 2

        preserved = json.loads(
            service.db.execute("SELECT data FROM uploads WHERE id=?", (old["id"],)).fetchone()[
                "data"
            ]
        )
        assert preserved.get("source_modality") == stored_modality
        original_response = client.get(f"/api/uploads/{old['id']}/file", headers=ALICE)
        assert original_response.status_code == 200
        assert hashlib.sha256(original_response.content).hexdigest() == cases[0]["sha256"]


@pytest.mark.parametrize(
    "text",
    [
        "上一份是MR，本次上传的图像请先判断后分割肝脏",
        "这份影像不是MR，请分割肝脏",
    ],
)
def test_plain_upload_leaves_historical_or_negated_modality_for_agent(
    configured, monkeypatch, text
):
    root, _, _ = configured
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)
    calls = []

    async def run(request, modality, execution, **kwargs):
        calls.append((request, modality, execution.modality, execution.example_modality_hint))
        assert (await execution.call("segment", {"modality": "CT", "targets": ["liver"]}))["ok"]
        return {"status": "completed", "summary": "肝脏已分割", "unresolved": []}

    monkeypatch.setattr(agent, "run_agent", run)
    with TestClient(create_app(root, "http://localhost")) as client:
        image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
        uploaded = client.post(
            "/api/uploads", headers={**ALICE, "X-Filename": "source.nii"}, content=image.to_bytes()
        ).json()
        sent = client.post(
            "/api/tasks",
            headers=ALICE,
            json={
                "upload_id": uploaded["id"],
                "text": text,
                "message_id": "historical-modality",
            },
        )
        assert sent.status_code == 202, sent.text
        assert sent.json()["modality_source"] == "unknown"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = client.get("/api/tasks/" + sent.json()["id"], headers=ALICE).json()
            if row["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert row["status"] == "completed", row
        assert calls == [(text, None, None, None)]
        assert row["modality"] == "CT" and row["modality_source"] == "agent"
        assert backend.calls[0]["task"] == "total"

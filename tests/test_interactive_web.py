"""HTTP integration and opt-out compatibility; fake worker only, no model claim."""

import asyncio
import io
import time

import nibabel as nib
import numpy as np
import pytest
from starlette.testclient import TestClient

from medsegagent.web import create_app


def image_bytes():
    image = nib.Nifti1Image(np.ones((8, 9, 10), dtype=np.int16), np.diag([2, 3, 4, 1]))
    image.header.set_xyzt_units("mm")
    stream = io.BytesIO()
    image.to_file_map({"image": nib.FileHolder(fileobj=stream)})
    return stream.getvalue()


def test_disabled_feature_has_no_new_runtime_or_public_capability(tmp_path, monkeypatch):
    monkeypatch.delenv("MEDSEGAGENT_OMNI_ENABLED", raising=False)
    app = create_app(tmp_path, "https://testserver")
    assert app.state.omni is None and not (tmp_path / "omni").exists()
    with TestClient(app, base_url="https://testserver") as client:
        client.post("/api/auth/guest")
        assert "interactive" not in client.get("/api/config").json()
        assert "interactive" not in client.get("/a2a/config").json()
        assert client.post("/api/omni/workspaces", json={}).status_code == 404


@pytest.fixture
def enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSEGAGENT_OMNI_ENABLED", "1")
    app = create_app(tmp_path, "https://testserver")

    async def worker(backend, request, run):
        await asyncio.sleep(0)
        source = nib.load(request["image_path"])
        mask = np.zeros(source.shape, dtype=np.uint8)
        for prompt in request["prompts"]:
            mask[tuple(prompt["voxel"])] = int(prompt["positive"])
        header = source.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(mask, source.affine, header), request["output_path"])
        return {"ok": True, "version": "contract-test"}

    monkeypatch.setattr(app.state.omni, "_worker", worker)
    return app


def test_enabled_authenticated_jobs_revisions_and_same_origin(enabled):
    with TestClient(enabled, base_url="https://testserver") as client:
        assert client.post("/api/omni/workspaces", json={}).status_code == 401
        client.post("/api/auth/guest")
        assert client.get("/api/config").json()["interactive"] == {"enabled": True}
        assert "interactive" not in client.get("/a2a/config").json()
        uploaded = client.post("/api/uploads", content=image_bytes(), headers={"X-Filename": "test.nii"})
        assert uploaded.status_code == 201, uploaded.text
        source = uploaded.json()["id"]
        assert client.post("/api/omni/workspaces", json={"upload_id": source},
                           headers={"Origin": "https://other.invalid"}).status_code == 403
        opened = client.post("/api/omni/workspaces", json={"upload_id": source})
        assert opened.status_code == 201, opened.text
        ws = opened.json()
        path = f"/api/omni/workspaces/{ws['id']}"
        body = {"operation": "refine", "message_id": "repeat-me", "base_revision": None,
                "prompts": [{"kind": "point", "world": [4, 9, 16], "positive": True}]}
        submitted = client.post(path + "/jobs", json=body)
        assert submitted.status_code == 202, submitted.text
        job_path = "/api/omni/jobs/" + submitted.json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = client.get(job_path).json()
            if result["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert result["status"] == "completed", result
        assert client.post(path + "/jobs", json=body).json()["id"] == result["id"]
        revision = result["result"]["revision"]
        assert revision["voxel_count"] == 1
        assert client.get(path).json()["latest_revision"] == revision["id"]
        listing = client.get("/api/omni/workspaces").json()
        assert len(listing) == 1 and listing[0]["id"] == ws["id"]
        assert listing[0]["revision_count"] == 1
        assert client.get(path + "/source").content == image_bytes()
        download = path + f"/revisions/{revision['id']}/file"
        assert client.get(download).status_code == 200
        assert client.post(path + "/jobs", json={**body, "message_id": "stale"}).status_code == 409
        client.delete("/api/session")
        client.post("/api/auth/guest")
        assert client.get("/api/omni/workspaces").json() == []
        for private in [path, job_path, download, path + "/source"]:
            assert client.get(private).status_code == 404


def test_enabled_does_not_expose_medgemma(enabled):
    with TestClient(enabled, base_url="https://testserver") as client:
        client.post("/api/auth/guest")
        source = client.post("/api/uploads", content=image_bytes(), headers={"X-Filename": "test.nii"}).json()["id"]
        ws = client.post("/api/omni/workspaces", json={"upload_id": source}).json()
        assert set(ws["capabilities"]) == {"nninteractive"}
        for operation in ("detect_modality", "generate_report"):
            assert client.post(f"/api/omni/workspaces/{ws['id']}/jobs", json={
                "operation": operation, "message_id": operation,
            }).status_code == 400

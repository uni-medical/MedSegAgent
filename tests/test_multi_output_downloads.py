"""Published requested masks keep private class downloads distinct across producers."""

import gzip
import time

import nibabel as nib
import numpy as np
import pytest
from auth_helpers import ALICE, BOB
from auth_helpers import create_test_app as create_app
from starlette.testclient import TestClient
from test_execution import Backend, make_input

from medsegagent import agent, core


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("target", ["liver", "lungs"])
def test_requested_binary_downloads_preserve_output_identity_and_partial_retention(
    tmp_path, monkeypatch, partial, target
):
    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)

    async def run(text, modality, execution, **kwargs):
        if target == "liver":
            for task, quality in [
                ("total", "fast"),
                ("total_v3", "standard"),
                ("total_v3", "fastest"),
            ]:
                assert (
                    await execution.call(
                        "segment", {"task": task, "quality": quality, "targets": [target]}
                    )
                )["ok"]
        else:
            assert (await execution.call("segment", {"targets": [target]}))["ok"]
        return {
            "status": "failed" if partial else "completed",
            "summary": "Synthetic result",
            "unresolved": ["Requested additional output unavailable"] if partial else [],
        }

    monkeypatch.setattr(agent, "run_agent", run)
    source = make_input(tmp_path)
    root = tmp_path / "service"
    app = create_app(root, "http://localhost")
    with TestClient(app) as client:
        upload = client.post(
            "/api/uploads",
            headers={**ALICE, "X-Filename": "synthetic.nii.gz"},
            content=source.read_bytes(),
        ).json()
        response = client.post(
            "/api/tasks",
            headers=ALICE,
            json={
                "upload_id": upload["id"],
                "text": "CT " + target,
                "modality": "CT",
                "message_id": "synthetic-multi",
            },
        )
        assert response.status_code == 202, response.text
        task_id = response.json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = client.get("/api/tasks/" + task_id, headers=ALICE).json()
            if row["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert row["status"] == ("failed" if partial else "completed")
        assert row["result_available"]
        expected_count = 3 if target == "liver" else 1
        classes = [file for file in row["files"] if file["kind"] == "label"]
        assert len(classes) == expected_count
        assert len({file["name"] for file in classes}) == expected_count
        assert len({file["output_id"] for file in classes}) == expected_count
        assert all(file["label_id"] == 1 and file["label_name"] == target for file in classes)
        if target == "lungs":
            assert classes[0]["name"] == "1_lungs.nii.gz"
            assert len(backend.calls[0]["targets"]) == 5
        originals = {}
        for output in row["result"]["outputs"]:
            assert {file["kind"] for file in output["files"]} == {"overlay", "label"}
            assert all(file["output_id"] == output["id"] for file in output["files"])
            overlay = next(file for file in output["files"] if file["kind"] == "overlay")
            originals[overlay["name"]] = (root / "tasks" / task_id / overlay["name"]).read_bytes()
        if target == "liver":
            outside = tmp_path / "outside-cache"
            outside.mkdir()
            cache = root / "tasks" / task_id / "class_masks"
            cache.symlink_to(outside, target_is_directory=True)
            assert client.get(classes[0]["url"], headers=ALICE).status_code == 503
            assert list(outside.iterdir()) == []
            cache.unlink()
        for file in classes:
            assert client.get(file["url"]).status_code == 401
            assert client.get(file["url"], headers=BOB).status_code == 404
            response = client.get(file["url"], headers=ALICE)
            assert response.status_code == 200, response.text
            image = nib.Nifti1Image.from_bytes(gzip.decompress(response.content))
            assert set(np.unique(image.get_fdata())) == {0, 1}
            assert np.count_nonzero(image.get_fdata()) == (5 if target == "lungs" else 1)
            np.testing.assert_allclose(image.affine, nib.load(source).affine)
        paths = list((root / "tasks" / task_id / "class_masks").rglob("*.nii.gz"))
        assert len(paths) == expected_count
        assert len({str(path.parent) for path in paths}) == expected_count
        a2a = client.get("/a2a/v1/tasks/" + task_id, headers={**ALICE, "A2A-Version": "1.0"}).json()
        parts = [
            part for artifact in a2a["artifacts"] for part in artifact["parts"] if "url" in part
        ]
        assert {part["filename"] for part in parts} == {file["name"] for file in classes}
        for name, before in originals.items():
            assert (root / "tasks" / task_id / name).read_bytes() == before
        app.state.service.update(task_id, expires_at=time.time() - 1)
        assert client.get(classes[0]["url"], headers=ALICE).status_code == 404
        assert client.get("/api/tasks/" + task_id, headers=ALICE).json()["files"] == []


@pytest.mark.parametrize("name", ["双肺 合并", "Whole lungs (reviewed)"])
def test_human_composition_names_keep_binary_download_cache_and_a2a(tmp_path, monkeypatch, name):
    """Exercise composition, publication and both APIs with no provider or GPU IO."""
    from xml.etree import ElementTree

    backend = Backend()
    monkeypatch.setattr(core, "segment", backend)

    async def run(text, modality, execution, **kwargs):
        result = await execution.call("segment", {"targets": ["lungs"]})
        assert result["ok"]
        lungs = next(row for row in result["regions"] if row["target"] == "lungs")
        composed = await execution.call(
            "compose_masks",
            {
                "operation": "union",
                "region_ids": [lungs["region_id"]],
                "name": name,
            },
        )
        assert composed["ok"]
        return {
            "status": "completed",
            "summary": "Requested composition complete",
            "unresolved": [],
        }

    monkeypatch.setattr(agent, "run_agent", run)
    source = make_input(tmp_path)
    root = tmp_path / "service"
    with TestClient(create_app(root, "http://localhost")) as client:
        uploaded = client.post(
            "/api/uploads",
            headers={**ALICE, "X-Filename": "synthetic.nii.gz"},
            content=source.read_bytes(),
        ).json()
        sent = client.post(
            "/api/tasks",
            headers=ALICE,
            json={
                "upload_id": uploaded["id"],
                "text": "分割双肺并合并",
                "modality": "CT",
                "message_id": "human-composition-name",
            },
        )
        assert sent.status_code == 202, sent.text
        task_id = sent.json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = client.get("/api/tasks/" + task_id, headers=ALICE).json()
            if row["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert row["status"] == "completed", row
        output = next(output for output in row["result"]["outputs"] if output["targets"] == [name])
        assert output["labels"][0]["name"] == name
        assert {file["kind"] for file in output["files"]} == {"overlay", "label"}
        file = next(file for file in output["files"] if file["kind"] == "label")
        assert file["label_name"] == name and file["label_id"] == 1
        assert file["name"].startswith(
            core.label_mask_filename({"id": 1, "name": name}).removesuffix(".nii.gz")
        )
        assert file["name"].isascii()
        response = client.get(file["url"], headers=ALICE)
        assert response.status_code == 200, response.text
        image = nib.Nifti1Image.from_bytes(gzip.decompress(response.content))
        assert set(np.unique(image.get_fdata())) == {0, 1}
        assert np.count_nonzero(image.get_fdata()) == 5
        np.testing.assert_allclose(image.affine, nib.load(source).affine)
        xml = ElementTree.fromstring(image.header.extensions[0].get_content())
        assert xml.find("./VolumeInformation/LabelTable/Label[@Key='1']").text == name
        cache = list((root / "tasks" / task_id / "class_masks").rglob("*.nii.gz"))
        assert len(cache) == 1
        cached_at = cache[0].stat().st_mtime_ns
        assert client.get(file["url"], headers=ALICE).content == response.content
        assert cache[0].stat().st_mtime_ns == cached_at
        a2a = client.get("/a2a/v1/tasks/" + task_id, headers={**ALICE, "A2A-Version": "1.0"}).json()
        artifact = next(
            artifact
            for artifact in a2a["artifacts"]
            if artifact.get("metadata", {}).get("label_name") == name
        )
        assert artifact["metadata"]["label_id"] == 1
        assert artifact["parts"][0]["filename"] == file["name"]
        assert all(
            part.get("filename") != output["files"][0]["name"]
            for artifact in a2a["artifacts"]
            for part in artifact["parts"]
        )
        public = row["result"]
    with TestClient(create_app(root, "http://localhost")) as client:
        assert client.get("/api/tasks/" + task_id, headers=ALICE).json()["result"] == public
        assert client.get(file["url"], headers=ALICE).content == response.content
        assert cache[0].stat().st_mtime_ns == cached_at
    assert len(backend.calls) == 1

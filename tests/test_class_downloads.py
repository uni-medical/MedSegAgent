"""Existing completed tasks expose private, lazy per-class binary files."""

import asyncio
import gzip
import hashlib
import time

import nibabel as nib
import numpy as np
import pytest
from auth_helpers import ALICE, BOB
from auth_helpers import create_test_app as create_app
from starlette.testclient import TestClient

from medsegagent import core


def test_private_class_downloads_and_a2a_work_for_existing_records(tmp_path, monkeypatch):
    app = create_app(tmp_path, "http://localhost")
    with TestClient(app) as client:
        service = app.state.service
        monkeypatch.setattr(service, "launch", lambda task_id: None)
        data = np.zeros((8, 9, 10), dtype=np.uint8)
        data[1:4] = 1
        data[5:7] = 2
        affine = np.diag([0.8, 1.2, 2.5, 1])
        source = nib.Nifti1Image(data, affine)
        uploaded = client.post(
            "/api/uploads", headers={**ALICE, "X-Filename": "source.nii"}, content=source.to_bytes()
        ).json()
        task = asyncio.run(
            service.submit("alice", uploaded["id"], "CT liver and spleen", None, "classes-one")
        )
        tid = task["id"]
        root = tmp_path / "tasks" / tid
        root.mkdir()
        merged = root / "segmentation.nii.gz"
        nib.save(source, merged)
        old_hash = hashlib.sha256(merged.read_bytes()).digest()
        service.update(
            tid,
            status="completed",
            result={
                "labels": [
                    {"id": 2, "name": "spleen", "voxels": int((data == 2).sum())},
                    {"id": 1, "name": "liver", "voxels": int((data == 1).sum())},
                ]
            },
            files=[
                {
                    "name": "segmentation.nii.gz",
                    "url": f"http://localhost/api/tasks/{tid}/files/segmentation.nii.gz",
                    "media_type": "application/gzip",
                }
            ],
            expires_at=time.time() + 60,
        )
        task = client.get("/api/tasks/" + tid, headers=ALICE).json()
        classes = [f for f in task["files"] if f.get("kind") == "label"]
        assert [f["name"] for f in classes] == ["1_liver.nii.gz", "2_spleen.nii.gz"]
        assert not (root / "class_masks").exists()
        for item in classes:
            assert client.get(item["url"]).status_code == 401
            assert client.get(item["url"], headers=BOB).status_code == 404
            result = client.get(item["url"], headers=ALICE)
            assert result.status_code == 200, result.text
            binary = nib.Nifti1Image.from_bytes(gzip.decompress(result.content))
            assert np.array_equal(np.asarray(binary.dataobj), data == item["label_id"])
            assert np.allclose(binary.affine, affine)
            assert set(np.unique(binary.dataobj)) == {0, 1}
            assert client.get(item["url"], headers=ALICE).content == result.content
        a2a = client.get("/a2a/v1/tasks/" + tid, headers={**ALICE, "A2A-Version": "1.0"}).json()
        files = [p for a in a2a["artifacts"] for p in a["parts"] if "url" in p]
        assert [p["filename"] for p in files] == ["1_liver.nii.gz", "2_spleen.nii.gz"]
        assert hashlib.sha256(merged.read_bytes()).digest() == old_hash
        assert (
            client.get(f"/api/tasks/{tid}/files/3_pancreas.nii.gz", headers=ALICE).status_code
            == 404
        )
        service.update(tid, expires_at=time.time() - 1)
        assert client.get(classes[0]["url"], headers=ALICE).status_code == 404
        assert client.get("/api/tasks/" + tid, headers=ALICE).json()["files"] == []


@pytest.mark.parametrize("name", ["liver", "lung_upper_lobe_left", "label-abc123", "A_2-b"])
def test_ascii_class_download_names_remain_unchanged(name):
    assert core.label_mask_filename({"id": 8, "name": name}) == f"8_{name}.nii.gz"


def test_human_label_filenames_are_stable_ascii_and_preserve_distinct_names():
    names = ["双肺合并", "双肺 合并", "Whole lungs", "Whole lungs (reviewed)"]
    filenames = [core.label_mask_filename({"id": 8, "name": name}) for name in names]
    assert len(set(filenames)) == len(names)
    for name, filename in zip(names, filenames, strict=True):
        expected = "8_label." + hashlib.sha256(name.encode()).hexdigest() + ".nii.gz"
        assert filename == expected and filename.isascii()
        assert core.label_mask_filename({"id": 8, "name": name}) == filename
    generated_stem = filenames[0].removeprefix("8_").removesuffix(".nii.gz")
    assert core.label_mask_filename({"id": 8, "name": generated_stem}) not in filenames


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "a" * 121,
        "../肝脏",
        "肝脏/区域",
        "肝脏\\区域",
        "肝脏..区域",
        "肝脏:区域",
        "肝脏\n区域",
        "肝脏\x00",
        "肝脏\x7f",
        "肝脏\x85",
        "肝脏\u202e",
        "肝脏\ud800",
    ],
)
def test_human_label_names_still_reject_paths_controls_and_excessive_length(name):
    with pytest.raises(core.SegmentationError):
        core.label_mask_filename({"id": 1, "name": name})

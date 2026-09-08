"""Reopenable records retain source and result together without becoming a file archive."""

from __future__ import annotations

import asyncio
import json
from email.message import Message
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from auth_helpers import ALICE, BOB
from auth_helpers import create_test_app as create_app
from starlette.testclient import TestClient

from medsegagent import service as service_module
from medsegagent.service import Service, ServiceError


@pytest.fixture
def clock(monkeypatch):
    now = SimpleNamespace(value=1_800_000_000.0)
    monkeypatch.setattr(service_module, "time", SimpleNamespace(time=lambda: now.value))
    return now


@pytest.fixture
def service(tmp_path, monkeypatch, clock):
    instance = Service(tmp_path, "https://medseg.example.org")
    monkeypatch.setattr(instance, "launch", lambda task_id: None)
    yield instance
    instance.db.close()


def upload(service):
    upload_id, path = service.reserve_upload("alice", "abdominal CT.nii")
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.int16), np.eye(4)), path)
    try:
        data = asyncio.run(
            service.finish_upload("alice", upload_id, path, "abdominal CT.nii", path.stat().st_size)
        )
    finally:
        service.uploading.discard("alice")
    return data, path


def submit(service, upload_id, message_id="request-one"):
    return asyncio.run(service.submit("alice", upload_id, "分割 CT 中的肝脏", None, message_id))


def complete(service, task_id):
    directory = service.root / "tasks" / task_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "segmentation.nii.gz").write_bytes(b"private-mask")
    service.update(
        task_id,
        status="completed",
        result={"targets": ["liver"]},
        files=[{"name": "segmentation.nii.gz"}],
    )


def test_record_contract_and_identity_isolation(service):
    source, _ = upload(service)
    task = submit(service, source["id"])
    assert task["upload_name"] == "abdominal CT.nii"
    assert task["input"]["id"] == source["id"]
    assert task["input"]["shape"] == [8, 8, 8]
    assert task["input"]["spacing"] == [1.0, 1.0, 1.0]
    assert task["input"]["available"] is True
    assert task["input"]["expires_at"] is None
    assert task["input_available"] is True
    assert task["result_available"] is False
    assert "principal" not in task and "selection" not in task
    assert str(service.root) not in json.dumps(task)
    assert service.list("alice")[0] == task
    for operation in (
        lambda: service.get("bob", task["id"]),
        lambda: service.get_upload("bob", source["id"]),
        lambda: service.upload_path("bob", source["id"]),
    ):
        with pytest.raises(ServiceError) as error:
            operation()
        assert error.value.status_code == 404


def test_historical_measurements_are_projected_without_rewriting_record_or_mask(service):
    from google.protobuf.json_format import MessageToDict

    from medsegagent.a2a import project_task

    source, _ = upload(service)
    task = submit(service, source["id"])
    complete(service, task["id"])
    label = {
        "id": 5,
        "name": "liver",
        "voxels": 12,
        "volume_ml": 0.012,
        "component_count": 3,
        "largest_component_voxels": 10,
        "largest_component_volume_ml": 0.01,
    }
    result = {
        "schema_version": 2,
        "component_connectivity": 26,
        "labels": [label],
        "volume_measurement": {
            "source_spatial_unit": "mm",
            "component_connectivity": 26,
            "component_meaning": "Historical measurement",
        },
    }
    service.update(task["id"], result=result)
    stored = service.db.execute("SELECT data FROM tasks WHERE id=?", (task["id"],)).fetchone()[0]
    mask = service.root / "tasks" / task["id"] / "segmentation.nii.gz"
    before = mask.read_bytes()
    record = service.get("alice", task["id"])
    a2a = MessageToDict(project_task(service._task(task["id"]), service.public_url))
    for public in (record["result"], a2a["metadata"]["segmentation"]):
        assert public["schema_version"] == 2
        assert public["labels"] == [{"id": 5, "name": "liver", "voxels": 12, "volume_ml": 0.012}]
        assert public["volume_measurement"] == {"source_spatial_unit": "mm"}
        assert "component" not in json.dumps(public)
    assert service.list("alice")[0] == record
    assert (
        service.db.execute("SELECT data FROM tasks WHERE id=?", (task["id"],)).fetchone()[0]
        == stored
    )
    assert mask.read_bytes() == before


def test_active_source_and_terminal_result_share_retention(service, clock):
    source, path = upload(service)
    clock.value = source["expires_at"] - 1
    task = submit(service, source["id"])
    clock.value += 60
    service.cleanup()
    assert path.is_file()
    assert service.get_upload("alice", source["id"])["expires_at"] is None
    with pytest.raises(ServiceError, match="Cancel the active task"):
        service.delete_upload("alice", source["id"])

    complete(service, task["id"])
    record = service.get("alice", task["id"])
    assert record["result_available"] is True
    assert record["input"]["expires_at"] == record["expires_at"]
    deadline = record["expires_at"]
    clock.value = deadline - 1
    assert service.upload_path("alice", source["id"]).is_file()
    assert service.file_path("alice", task["id"], "segmentation.nii.gz").is_file()
    with pytest.raises(ServiceError):
        service.file_path("alice", task["id"], "result.json")

    clock.value = deadline
    expired = service.get("alice", task["id"])
    assert expired["files_expired"] is True
    assert expired["input_available"] is False
    assert expired["result_available"] is False
    assert expired["result"] is None and expired["files"] == []
    # HTTP capability disappears at the deadline even before periodic file deletion.
    assert path.is_file()
    for operation in (
        lambda: service.get_upload("alice", source["id"]),
        lambda: service.upload_path("alice", source["id"]),
        lambda: service.file_path("alice", task["id"], "segmentation.nii.gz"),
    ):
        with pytest.raises(ServiceError) as error:
            operation()
        assert error.value.status_code == 404
    service.cleanup()
    assert not path.exists()
    record = service.get("alice", task["id"])
    assert record["status"] == "completed"
    assert record["input"]["name"] == "abdominal CT.nii"
    assert record["input"]["shape"] == [8, 8, 8]
    assert record["expires_at"] == deadline
    # Idempotency restores the audit record rather than launching an expired input again.
    assert submit(service, source["id"])["id"] == task["id"]


def test_reused_source_lives_until_latest_record_expires(service, clock):
    source, path = upload(service)
    first = submit(service, source["id"])
    complete(service, first["id"])
    first_deadline = service.get("alice", first["id"])["expires_at"]
    clock.value += 3600
    second = submit(service, source["id"], "request-two")
    complete(service, second["id"])
    second_deadline = service.get("alice", second["id"])["expires_at"]
    clock.value = first_deadline
    service.cleanup()
    assert path.is_file()
    first_record = service.get("alice", first["id"])
    assert first_record["input_available"] is True
    assert first_record["input"]["expires_at"] == second_deadline
    assert first_record["result_available"] is False
    assert service.get("alice", second["id"])["result_available"] is True
    clock.value = second_deadline
    service.cleanup()
    assert not path.exists()


@pytest.mark.parametrize("status", ["failed", "canceled"])
def test_failed_or_canceled_records_can_reuse_source_for_one_retention_window(
    service, clock, status
):
    source, path = upload(service)
    task = submit(service, source["id"])
    clock.value += 120
    service.update(task["id"], status=status)
    record = service.get("alice", task["id"])
    assert record["input"]["expires_at"] == clock.value + service.retention_seconds
    assert record["input_available"] is True
    assert record["result_available"] is False
    clock.value = record["expires_at"]
    service.cleanup()
    assert not path.exists()
    assert service.get("alice", task["id"])["status"] == status


def test_explicit_source_deletion_does_not_remove_result(service):
    source, path = upload(service)
    task = submit(service, source["id"])
    complete(service, task["id"])
    service.delete_upload("alice", source["id"])
    assert not path.exists()
    record = service.get("alice", task["id"])
    assert record["input_available"] is False
    assert record["input"]["name"] == "abdominal CT.nii"
    assert record["result_available"] is True
    assert service.file_path("alice", task["id"], "segmentation.nii.gz").is_file()


def test_legacy_records_derive_source_deadline_without_renewing_expired_data(service, clock):
    source, path = upload(service)
    task = submit(service, source["id"])
    clock.value += 300
    complete(service, task["id"])
    legacy = service._task(task["id"])
    legacy.pop("input")
    deadline = legacy.pop("expires_at")
    with service.db:
        service.db.execute("UPDATE tasks SET data=? WHERE id=?", (json.dumps(legacy), task["id"]))
    assert service.get("alice", task["id"])["input"]["expires_at"] == deadline
    clock.value = deadline
    service.cleanup()
    assert not path.exists()
    clock.value += 3600
    service.cleanup()
    assert service.get("alice", task["id"])["expires_at"] == deadline


def test_owned_upload_metadata_and_named_download_reject_expiry_immediately(tmp_path, clock):
    app = create_app(
        tmp_path,
        public_url="http://localhost",
    )
    with TestClient(app) as client:
        volume = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.int16), np.eye(4)).to_bytes()
        response = client.post(
            "/api/uploads",
            headers={**ALICE, "X-Filename": "abdominal CT.nii"},
            content=volume,
        )
        assert response.status_code == 201, response.text
        source = response.json()
        endpoint = f"/api/uploads/{source['id']}"
        assert client.get(endpoint).status_code == 401
        assert client.get(endpoint, headers=BOB).status_code == 404
        metadata = client.get(endpoint, headers=ALICE).json()
        assert metadata["available"] is True
        assert metadata["name"] == "abdominal CT.nii"
        assert "path" not in metadata
        response = client.get(endpoint + "/file", headers=ALICE)
        assert response.content == volume
        disposition = Message()
        disposition["Content-Disposition"] = response.headers["content-disposition"]
        assert disposition.get_content_disposition() == "attachment"
        assert disposition.get_filename() == "abdominal CT.nii"
        assert client.get(endpoint + "/file", headers=BOB).status_code == 404
        clock.value = source["expires_at"]
        assert client.get(endpoint, headers=ALICE).status_code == 404
        assert client.get(endpoint + "/file", headers=ALICE).status_code == 404

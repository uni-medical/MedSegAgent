"""Public A2A canary, optionally using a private guest browser session.

Uploads the explicitly supplied research image and verifies its real results.
No API token, provider credential, or environment file is required or printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

import httpx
import nibabel as nib
import numpy as np
from a2a.types import AgentCard, StreamResponse, Task
from google.protobuf.json_format import ParseDict

VERSION = {"A2A-Version": "1.0"}


def upload_image(client, prefix, path):
    config = client.get(prefix + "/config")
    config.raise_for_status()
    config = config.json()
    size = path.stat().st_size
    assert 0 < size <= config["max_upload_bytes"], "Input exceeds the service upload limit"
    if size <= config["single_upload_bytes"]:
        with path.open("rb") as stream:
            response = client.post(
                prefix + "/uploads",
                content=stream,
                headers={"X-Filename": path.name, "Content-Type": "application/octet-stream"},
            )
        response.raise_for_status()
        return response.json()["id"], "single"
    started = client.post(
        prefix + "/upload-sessions",
        json={
            "name": path.name,
            "size": size,
            "message_id": str(uuid.uuid4()),
        },
    )
    started.raise_for_status()
    session = started.json()
    endpoint = prefix + "/upload-sessions/" + session["id"]
    offset = session["offset"]
    with path.open("rb") as stream:
        stream.seek(offset)
        while offset < size:
            data = stream.read(min(session["chunk_bytes"], size - offset))
            response = client.put(endpoint, content=data, headers={"Upload-Offset": str(offset)})
            response.raise_for_status()
            offset += len(data)
            assert response.json()["offset"] == offset
    completed = client.post(endpoint + "/complete")
    completed.raise_for_status()
    # Completion returns the same upload ID; its byte geometry was validated by the service.
    assert completed.json()["id"] == session["id"]
    return session["id"], "chunked"


def verify_outputs(
    client, observer, anonymous, *, base, prefix, access, task, source, output, method
):
    """Check each published output against its own binary class files and measurements."""
    task_id = task["id"]
    public_result = task["metadata"]["segmentation"]
    assert public_result["schema_version"] == 5
    assert public_result["completion"] == {"status": "completed", "unresolved": []}
    outputs = public_result["outputs"]
    assert outputs and len({item["id"] for item in outputs}) == len(outputs)
    artifacts = {
        part["filename"]: (artifact, part)
        for artifact in task.get("artifacts", [])
        for part in artifact.get("parts", [])
        if "url" in part
    }
    expected_classes = {
        file["name"] for item in outputs for file in item["files"] if file["kind"] == "label"
    }
    assert set(artifacts) == expected_classes and expected_classes
    assert len(expected_classes) == sum(
        file["kind"] == "label" for item in outputs for file in item["files"]
    )
    original = nib.load(source)
    checked = []
    for index, item in enumerate(outputs, 1):
        labels = item["labels"]
        assert [label["id"] for label in labels] == list(range(1, len(labels) + 1))
        assert labels and labels[0]["color"] == "#ff0000"
        overlays = [file for file in item["files"] if file["kind"] == "overlay"]
        classes = [file for file in item["files"] if file["kind"] == "label"]
        assert len(overlays) == 1 and len(classes) == len(labels)
        assert {file["label_id"] for file in classes} == {label["id"] for label in labels}
        assert all(file["output_id"] == item["id"] for file in item["files"])
        overlay = overlays[0]
        for file in item["files"]:
            name = file["name"]
            assert name == Path(name).name and re.fullmatch(r"[A-Za-z0-9_.-]+\.nii\.gz", name)
            assert file["url"] == base + f"{prefix}/tasks/{task_id}/files/{name}"
            if access == "public":
                assert observer.get(file["url"]).status_code == 200
            else:
                assert anonymous.get(file["url"]).status_code == 401
                assert observer.get(file["url"]).status_code == 404
        downloaded = client.get(overlay["url"])
        downloaded.raise_for_status()
        assert hashlib.sha256(downloaded.content).hexdigest() == overlay["sha256"]
        assert len(downloaded.content) == overlay["size_bytes"]
        mask_path = output / f"{method}-output-{index}-mask.nii.gz"
        mask_path.write_bytes(downloaded.content)
        mask = nib.load(mask_path)
        assert original.shape == mask.shape and np.allclose(original.affine, mask.affine)
        values = np.asarray(mask.dataobj)
        assert np.isfinite(values).all()
        assert set(np.unique(values)) <= {0, *(label["id"] for label in labels)}
        nonzero = int(np.count_nonzero(values))
        assert item["nonzero_voxels"] == nonzero
        assert item["no_target_detected"] is (nonzero == 0)
        spatial_unit = original.header.get_xyzt_units()[0]
        assert mask.header.get_xyzt_units()[0] == spatial_unit
        factor = {"meter": 1000, "mm": 1, "micron": 0.001, "unknown": 1}[spatial_unit]
        spacing_mm = np.asarray(mask.header.get_zooms()[:3], dtype=float) * factor
        voxel_mm3 = float(np.prod(spacing_mm))
        measurement = item["volume_measurement"]
        assert measurement["method"] == "voxel_count_times_spacing_product"
        assert measurement["source_spatial_unit"] == spatial_unit
        assert measurement["unit_assumption"] == (
            "assumed_mm" if spatial_unit == "unknown" else None
        )
        assert np.allclose(measurement["spacing_mm"], spacing_mm)
        assert np.isclose(measurement["voxel_volume_mm3"], voxel_mm3)
        by_id = {label["id"]: label for label in labels}
        class_files = []
        for file in classes:
            label = by_id[file["label_id"]]
            assert file["label_name"] == label["name"]
            expected = values == label["id"]
            assert int(expected.sum()) == label["voxels"]
            assert np.isclose(label["volume_mm3"], label["voxels"] * voxel_mm3)
            assert np.isclose(label["volume_ml"], label["voxels"] * voxel_mm3 / 1000)
            # Native outputs have source_id; compositions retain their own source manifest.
            assert {"id", "name", "color", "voxels", "volume_mm3", "volume_ml"} <= label.keys()
            assert set(label) <= {
                "id",
                "source_id",
                "name",
                "color",
                "voxels",
                "volume_mm3",
                "volume_ml",
            }
            artifact, part = artifacts[file["name"]]
            assert part["url"] == file["url"]
            metadata = artifact["metadata"]
            assert metadata["output_id"] == item["id"]
            assert metadata["label_id"] == label["id"] and metadata["label_name"] == label["name"]
            assert metadata["mask_value"] == 1
            received = client.get(part["url"])
            received.raise_for_status()
            class_path = output / f"{method}-{file['name']}"
            class_path.write_bytes(received.content)
            binary_image = nib.load(class_path)
            binary = np.asarray(binary_image.dataobj)
            assert binary_image.shape == mask.shape and np.allclose(
                binary_image.affine, mask.affine
            )
            assert binary_image.header.get_zooms() == mask.header.get_zooms()
            assert binary_image.header.get_xyzt_units() == mask.header.get_xyzt_units()
            assert binary_image.get_data_dtype() == np.dtype("uint8")
            assert binary_image.header.get_intent()[0] == "label"
            for form in ("qform", "sform"):
                expected_form, expected_code = getattr(mask, f"get_{form}")(coded=True)
                actual, actual_code = getattr(binary_image, f"get_{form}")(coded=True)
                assert actual_code == expected_code
                assert (
                    actual is None if expected_form is None else np.allclose(actual, expected_form)
                )
            assert np.isfinite(binary).all() and set(np.unique(binary)) <= {0, 1}
            assert np.array_equal(binary, expected)
            checksum = hashlib.sha256(received.content).hexdigest()
            if "sha256" in metadata:
                assert metadata["sha256"] == checksum
            if "size_bytes" in metadata:
                assert metadata["size_bytes"] == len(received.content)
            class_files.append(
                {
                    "filename": file["name"],
                    "label_id": label["id"],
                    "label_name": label["name"],
                    "mask_value": 1,
                    "voxels": int(binary.sum()),
                    "size_bytes": len(received.content),
                    "sha256": checksum,
                    "matches_merged_label": True,
                    "geometry_consistent": True,
                }
            )
        checked.append(
            {
                "output_id": item["id"],
                "task": item["task"],
                "quality": item["quality"],
                "targets": item["targets"],
                "nonzero_voxels": nonzero,
                "sha256": overlay["sha256"],
                "labels": labels,
                "class_files": class_files,
                "normalized_mask_and_measurements": True,
                "geometry_consistent": True,
            }
        )
    assert client.get(f"{prefix}/tasks/{task_id}/files/result.json").status_code == 404
    return checked


def run_canary(
    *,
    url,
    input_path,
    output_dir,
    access="public",
    modality="CT",
    text="请分割肝脏和左右肾。",
    transport=None,
):
    if access not in {"public", "guest"}:
        raise ValueError("access must be public or guest")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = url.rstrip("/")
    prefix = "/a2a" if access == "public" else "/api"
    report = {
        "url": base,
        "access": access,
        "modality": modality,
        "time": time.time(),
        "checks": {},
        "tasks": [],
    }
    started = time.monotonic()
    options = {
        "base_url": base,
        "timeout": 180,
        "follow_redirects": False,
        "trust_env": False,
        "transport": transport,
    }
    with (
        httpx.Client(**options, headers={"Origin": base}) as client,
        httpx.Client(**options) as anonymous,
        httpx.Client(**options, headers={"Origin": base}) as observer,
    ):
        card = client.get("/.well-known/agent-card.json")
        card.raise_for_status()
        parsed = ParseDict(card.json(), AgentCard())
        assert parsed.supported_interfaces[0].url == base + "/a2a/v1"
        assert not parsed.security_schemes and not parsed.security_requirements
        assert anonymous.get("/api/tasks").status_code == 401
        assert anonymous.get("/a2a/v1/tasks/missing", headers=VERSION).status_code == 404
        assert anonymous.get("/a2a/v1/tasks", headers=VERSION).status_code == 400
        report["checks"].update(agent_card=card.status_code, no_public_task_listing=True)
        if access == "guest":
            for caller in (client, observer):
                login = caller.post("/api/auth/guest")
                login.raise_for_status()
                assert login.json()["identity"]["kind"] == "guest" and caller.cookies
        else:
            assert not client.cookies and not observer.cookies
        upload_id, upload_mode = upload_image(client, prefix, input_path)
        report.update(upload_id=upload_id, upload_mode=upload_mode)
        for method in ("send", "stream"):
            body = {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "ROLE_USER",
                    "parts": [
                        {"text": text, "mediaType": "text/plain"},
                        {
                            "data": {"upload_id": upload_id, "modality": modality},
                            "mediaType": "application/json",
                        },
                    ],
                },
                "configuration": {"returnImmediately": True},
            }
            before = time.monotonic()
            if method == "send":
                sent = client.post("/a2a/v1/message:send", json=body, headers=VERSION)
                sent.raise_for_status()
                task = sent.json()["task"]
            else:
                task = None
                with client.stream(
                    "POST", "/a2a/v1/message:stream", json=body, headers=VERSION
                ) as response:
                    response.raise_for_status()
                    assert response.headers["content-type"].startswith("text/event-stream")
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            event = json.loads(line[6:])
                            ParseDict(event, StreamResponse())
                            if "task" in event:
                                task = event["task"]
                                break  # Disconnect after the first task event, then recover by ID.
                assert task is not None, "Stream returned no initial task"
                report["checks"]["stream_disconnected"] = True
            task_id = task["id"]
            ParseDict(task, Task())
            replay = client.post("/a2a/v1/message:send", json=body, headers=VERSION)
            replay.raise_for_status()
            assert replay.json()["task"]["id"] == task_id
            while time.monotonic() - before < 600:
                recovered = client.get(f"/a2a/v1/tasks/{task_id}", headers=VERSION)
                recovered.raise_for_status()
                task = recovered.json()
                ParseDict(task, Task())
                if task["status"]["state"] in {
                    "TASK_STATE_COMPLETED",
                    "TASK_STATE_FAILED",
                    "TASK_STATE_CANCELED",
                }:
                    break
                time.sleep(1)
            assert task["status"]["state"] == "TASK_STATE_COMPLETED", task["status"]
            if access == "public":
                assert observer.get(f"/a2a/v1/tasks/{task_id}", headers=VERSION).status_code == 200
                assert not client.cookies
            else:
                assert anonymous.get(f"/a2a/v1/tasks/{task_id}", headers=VERSION).status_code == 404
                assert observer.get(f"/a2a/v1/tasks/{task_id}", headers=VERSION).status_code == 404
            outputs = verify_outputs(
                client,
                observer,
                anonymous,
                base=base,
                prefix=prefix,
                access=access,
                task=task,
                source=input_path,
                output=output_dir,
                method=method,
            )
            report["tasks"].append(
                {
                    "method": method,
                    "task_id": task_id,
                    "status": task["status"]["state"],
                    "wall_seconds": time.monotonic() - before,
                    "outputs": outputs,
                }
            )
            (output_dir / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False)
            )
            print(json.dumps(report["tasks"][-1], ensure_ascii=False), flush=True)
        report["checks"].update(
            idempotency=True,
            access_boundary=True,
            per_class_binary_artifacts=True,
            get_task_after_disconnect=True,
            verified_tasks=2,
        )
    report["wall_seconds"] = time.monotonic() - started
    report["limitation"] = "Execution canaries only; not clinical validation."
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--access", choices=["public", "guest"], default="public")
    parser.add_argument("--modality", choices=["CT", "MR"], default="CT")
    parser.add_argument("--text", default="请分割肝脏和左右肾。")
    args = parser.parse_args()
    run_canary(
        url=args.url,
        input_path=args.input,
        output_dir=args.output,
        access=args.access,
        modality=args.modality,
        text=args.text,
    )
    print("Acceptance passed; report: " + str(args.output / "report.json"))


if __name__ == "__main__":
    main()

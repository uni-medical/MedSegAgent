"""Real authenticated HTTP/A2A canary. Never prints tokens or image data.

Run with uv run python ops/acceptance.py --url ... --input ... --output outputs/acceptance/...
The input is uploaded to the explicitly selected service; use de-identified research data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import httpx
import nibabel as nib
import numpy as np
from a2a.types import AgentCard, StreamResponse, Task
from dotenv import load_dotenv
from google.protobuf.json_format import ParseDict


def main():
    os.umask(0o077)
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--principal", default="duanyan")
    p.add_argument("--modality", choices=["CT", "MR"], default="CT")
    p.add_argument("--text", default="请分割肝脏和左右肾。")
    args = p.parse_args()
    tokens = json.loads(os.environ["MEDSEGAGENT_TOKENS_JSON"])
    args.output.mkdir(parents=True, exist_ok=True)
    auth = {"Authorization": "Bearer " + tokens[args.principal], "A2A-Version": "1.0"}
    base = args.url.rstrip("/")
    report = {
        "url": base,
        "modality": args.modality,
        "time": time.time(),
        "checks": {},
        "tasks": [],
    }
    started = time.monotonic()
    with httpx.Client(
        base_url=base, timeout=180, follow_redirects=False, trust_env=False
    ) as client:
        card = client.get("/.well-known/agent-card.json")
        card.raise_for_status()
        parsed = ParseDict(card.json(), AgentCard())
        assert parsed.supported_interfaces[0].url == base + "/a2a/v1"
        report["checks"]["agent_card"] = card.status_code
        for endpoint in ("/api/tasks", "/a2a/v1/tasks/missing"):
            assert client.get(endpoint).status_code == 401
        report["checks"]["unauthorized"] = 401
        with args.input.open("rb") as stream:
            upload = client.post(
                "/api/uploads",
                headers={
                    **auth,
                    "X-Filename": args.input.name,
                    "Content-Type": "application/octet-stream",
                },
                content=stream,
            )
        upload.raise_for_status()
        upload_id = upload.json()["id"]
        report["upload_id"] = upload_id
        for method in ("send", "stream"):
            body = {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "ROLE_USER",
                    "parts": [
                        {"text": args.text, "mediaType": "text/plain"},
                        {
                            "data": {"upload_id": upload_id, "modality": args.modality},
                            "mediaType": "application/json",
                        },
                    ],
                },
                "configuration": {"returnImmediately": True},
            }
            before = time.monotonic()
            if method == "send":
                sent = client.post("/a2a/v1/message:send", json=body, headers=auth)
                sent.raise_for_status()
                task = sent.json()["task"]
            else:
                with client.stream(
                    "POST", "/a2a/v1/message:stream", json=body, headers=auth
                ) as response:
                    response.raise_for_status()
                    assert response.headers["content-type"].startswith("text/event-stream")
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            event = json.loads(line[6:])
                            ParseDict(event, StreamResponse())
                            task = event["task"]
                            break  # Deliberately disconnect before inference completes.
                report["checks"]["stream_disconnected"] = True
            task_id = task["id"]
            ParseDict(task, Task())
            replay = client.post("/a2a/v1/message:send", json=body, headers=auth)
            replay.raise_for_status()
            assert replay.json()["task"]["id"] == task_id
            while time.monotonic() - before < 600:
                recovered = client.get(f"/a2a/v1/tasks/{task_id}", headers=auth)
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
            files = [
                part
                for artifact in task.get("artifacts", [])
                for part in artifact.get("parts", [])
                if "url" in part
            ]
            assert files
            file = next(part for part in files if part["filename"] == "segmentation.nii.gz")
            assert client.get(file["url"]).status_code == 401
            for principal, token in tokens.items():
                if principal != args.principal:
                    other = {"Authorization": "Bearer " + token, "A2A-Version": "1.0"}
                    assert client.get(f"/a2a/v1/tasks/{task_id}", headers=other).status_code == 404
                    assert client.get(file["url"], headers=other).status_code == 404
            result = client.get(file["url"], headers=auth)
            result.raise_for_status()
            mask_path = args.output / f"{method}-mask.nii.gz"
            mask_path.write_bytes(result.content)
            original, mask = nib.load(args.input), nib.load(mask_path)
            assert original.shape == mask.shape and np.allclose(original.affine, mask.affine)
            values = np.asarray(mask.dataobj)
            assert np.isfinite(values).all() and np.count_nonzero(values)
            public_result = client.get(
                f"/api/tasks/{task_id}/files/result.json", headers=auth
            ).json()
            report["tasks"].append(
                {
                    "method": method,
                    "task_id": task_id,
                    "status": task["status"]["state"],
                    "wall_seconds": time.monotonic() - before,
                    "runtime_seconds": public_result.get("runtime_seconds"),
                    "geometry_consistent": True,
                    "nonzero_voxels": int(np.count_nonzero(values)),
                    "sha256": hashlib.sha256(result.content).hexdigest(),
                    "labels": public_result.get("labels"),
                }
            )
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report["tasks"][-1]), flush=True)
        report["checks"].update(
            idempotency=True,
            identity_isolation=True,
            artifact_auth=True,
            get_task_after_disconnect=True,
            real_model_tasks=2,
        )
    report["wall_seconds"] = time.monotonic() - started
    report["limitation"] = "Execution canaries only; not clinical validation."
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("Acceptance passed; private report: " + str(args.output / "report.json"))


if __name__ == "__main__":
    main()

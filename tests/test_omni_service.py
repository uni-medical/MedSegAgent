"""Workspace contracts with synthetic masks; no pretrained-model inference claims."""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from medsegagent import core
from medsegagent.omni import OmniService, world_prompts
from medsegagent.service import ServiceError


class Base:
    def __init__(self, root, unit="mm", affine=None):
        self.root = root
        root.mkdir()
        self.retention_seconds = 3600
        self.source = root / "source.nii.gz"
        self.affine = np.diag([2.0, 3.0, 4.0, 1.0]) if affine is None else affine
        image = nib.Nifti1Image(np.ones((8, 9, 10), dtype=np.int16), self.affine)
        image.header.set_xyzt_units(unit)
        nib.save(image, self.source)
        self.metadata = {
            "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "expires_at": time.time() + 3600,
        }

    def get_upload(self, principal, upload_id):
        if principal not in {"alice", "bob", "carol"} or upload_id != "upload":
            raise ServiceError("FILE_NOT_FOUND", "Not found", 404)
        return self.metadata.copy()

    def upload_path(self, principal, upload_id):
        self.get_upload(principal, upload_id)
        return self.source


def world(base, voxel):
    return (base.affine @ np.array([*voxel, 1.0]))[:3].tolist()


def mark(base, voxel=(2, 3, 4), positive=True):
    return {"kind": "point", "world": world(base, voxel), "positive": positive}


def request(base, message="one", base_revision=None, **extra):
    return {"message_id": message, "base_revision": base_revision, "prompts": [mark(base)], **extra}


async def drain(service, job_id):
    pending = service.active.get(job_id)
    if pending:
        await pending
    # Allow task done callbacks to finish before closing the database.
    await asyncio.sleep(0)


async def synthetic_worker(backend, request, run):
    assert backend == "nninteractive"
    source = nib.load(request["image_path"])
    values = np.zeros(source.shape, dtype=np.uint8)
    for prompt in request["prompts"]:
        if prompt["kind"] == "point":
            values[tuple(prompt["voxel"])] = int(prompt["positive"])
        else:
            slices = tuple(slice(*bound) for bound in prompt["bounds"])
            values[slices] = int(prompt["positive"])
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(values, source.affine, header), request["output_path"])
    return {"ok": True, "version": "synthetic-contract-test"}


def test_world_prompt_geometry_and_oblique_box_rejection():
    affine = np.array([[0, -2, 0, 10], [3, 0, 0, -5], [0, 0, 4, 8], [0, 0, 0, 1]], dtype=float)
    base = SimpleNamespace(affine=affine)
    geometry = {"shape": [8, 9, 10], "affine": affine.tolist()}
    prompts = [
        mark(base),
        {
            "kind": "box",
            "world_start": world(base, (1, 2, 4)),
            "world_end": world(base, (4, 6, 4)),
            "positive": False,
        },
    ]
    assert world_prompts(prompts, geometry) == [
        {"kind": "point", "voxel": [2, 3, 4], "positive": True},
        {"kind": "box", "bounds": [[1, 5], [2, 7], [4, 5]], "positive": False},
    ]
    # Half coordinates match JavaScript floor(v + .5), not NumPy ties-to-even.
    assert world_prompts([mark(base, (2.5, 3, 4))], geometry)[0]["voxel"] == [3, 3, 4]
    for invalid in [
        mark(base, (-2, 0, 0)),
        {**prompts[1], "world_end": world(base, (4, 6, 4.2))},
        {**prompts[1], "world_end": world(base, (1, 6, 4))},
        {**prompts[0], "positive": 1},
        {**prompts[0], "world": [float("nan"), 0, 0]},
    ]:
        with pytest.raises(ServiceError) as error:
            world_prompts([invalid], geometry)
        assert error.value.code == "PROMPT_INVALID"


def test_frozen_workspace_ownership_restart_and_unknown_units(tmp_path, monkeypatch):
    async def run():
        base = Base(tmp_path / "base", unit="unknown")
        service = OmniService(base)
        ws = await service.open_workspace("alice", "upload")
        assert await service.open_workspace("alice", "upload") == ws
        assert ws["geometry"]["unit"] == "unknown"
        assert "source_file" not in ws and "_principal" not in ws
        with pytest.raises(ServiceError) as error:
            service.get_workspace("bob", ws["id"])
        assert error.value.status_code == 404
        monkeypatch.setattr(service, "_worker", synthetic_worker)
        job = await service.submit("alice", ws["id"], "refine", request(base))
        await drain(service, job["id"])
        result = service.get_job("alice", job["id"])
        assert result["status"] == "completed"
        revision = result["result"]["revision"]
        assert revision["voxel_count"] == 1
        assert revision["volume_ml"] is None
        assert revision["volume_measurement"]["unit_assumption"] is None
        base.source.unlink()  # The workspace owns a previously verified private snapshot.
        assert service.download_path("alice", ws["id"], revision["id"]).is_file()
        with pytest.raises(ServiceError):
            service.download_path("bob", ws["id"], revision["id"])
        await service.close()
        restarted = OmniService(base)
        assert restarted.get_workspace("alice", ws["id"])["latest_revision"] == revision["id"]
        assert restarted.get_job("alice", job["id"])["status"] == "completed"
        await restarted.close()

    asyncio.run(run())


def test_prompt_replay_revision_cas_idempotence_and_explicit_branch(tmp_path, monkeypatch):
    async def run():
        base = Base(tmp_path / "base")
        service = OmniService(base)
        monkeypatch.setattr(service, "_worker", synthetic_worker)
        ws = await service.open_workspace("alice", "upload")
        first_body = request(base)
        first = await service.submit("alice", ws["id"], "refine", first_body)
        await drain(service, first["id"])
        r1 = service.get_job("alice", first["id"])["result"]["revision"]
        assert r1["volume_ml"] == pytest.approx(0.024)
        assert (await service.submit("alice", ws["id"], "refine", first_body))["id"] == first["id"]
        with pytest.raises(ServiceError) as error:
            await service.submit("alice", ws["id"], "refine", request(base, "stale"))
        assert error.value.code == "REVISION_CONFLICT"
        second = await service.submit(
            "alice",
            ws["id"],
            "refine",
            request(base, "two", r1["id"], prompts=[mark(base, (5, 4, 3))]),
        )
        await drain(service, second["id"])
        r2 = service.get_job("alice", second["id"])["result"]["revision"]
        assert r2["voxel_count"] == 2 and len(r2["prompts"]) == 2
        branch = await service.submit(
            "alice",
            ws["id"],
            "refine",
            request(
                base,
                "branch",
                r1["id"],
                expected_revision=r2["id"],
                prompts=[mark(base, positive=False)],
            ),
        )
        await drain(service, branch["id"])
        r3 = service.get_job("alice", branch["id"])["result"]["revision"]
        assert r3["parent_revision"] == r1["id"] and r3["voxel_count"] == 0
        assert len(r3["prompts"]) == 2
        assert service.get_workspace("alice", ws["id"])["revisions"][1]["sha256"] == r2["sha256"]
        with pytest.raises(ServiceError):
            await service.submit("alice", ws["id"], "refine", request(base, "branch", r3["id"]))
        await service.close()

    asyncio.run(run())


def test_concurrency_cancellation_and_no_legacy_table_changes(tmp_path, monkeypatch):
    async def run():
        base = Base(tmp_path / "base")
        sentinel = base.root / "legacy-marker"
        sentinel.write_bytes(b"unchanged")
        service = OmniService(base)
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def waiting(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        monkeypatch.setattr(service, "_worker", waiting)
        workspaces = {
            who: await service.open_workspace(who, "upload") for who in ["alice", "bob", "carol"]
        }
        a = await service.submit("alice", workspaces["alice"]["id"], "refine", request(base))
        await started.wait()
        b = await service.submit("bob", workspaces["bob"]["id"], "refine", request(base))
        with pytest.raises(ServiceError) as error:
            await service.submit("carol", workspaces["carol"]["id"], "refine", request(base))
        assert error.value.status_code == 429
        with pytest.raises(ServiceError) as error:
            await service.submit(
                "alice", workspaces["alice"]["id"], "refine", request(base, "again")
            )
        assert error.value.code == "WORKSPACE_BUSY"
        with pytest.raises(ServiceError):
            await service.cancel_job("bob", a["id"])
        assert (await service.cancel_job("alice", a["id"]))["status"] == "canceled"
        assert stopped.is_set()
        await service.close()
        assert sentinel.read_bytes() == b"unchanged"
        restarted = OmniService(base)
        assert restarted.get_job("bob", b["id"])["status"] == "failed"
        assert restarted.get_job("bob", b["id"])["error"]["code"] == "SERVICE_RESTARTED"
        await restarted.close()

    asyncio.run(run())


def test_bad_output_and_source_mutation_never_publish_revision(tmp_path, monkeypatch):
    async def run():
        base = Base(tmp_path / "base")
        service = OmniService(base)
        ws = await service.open_workspace("alice", "upload")

        async def invalid(backend, request, run):
            image = nib.load(request["image_path"])
            nib.save(
                nib.Nifti1Image(
                    np.full(image.shape, 2, dtype=np.uint8), image.affine, image.header
                ),
                request["output_path"],
            )
            return {"ok": True}

        monkeypatch.setattr(service, "_worker", invalid)
        job = await service.submit("alice", ws["id"], "refine", request(base))
        await drain(service, job["id"])
        assert service.get_job("alice", job["id"])["error"]["code"] == "OUTPUT_INVALID"
        assert service.get_workspace("alice", ws["id"])["revisions"] == []
        source = service._directory(ws["id"]) / "image.nii.gz"
        source.write_bytes(b"mutated private image")
        again = await service.submit("alice", ws["id"], "refine", request(base, "two"))
        await drain(service, again["id"])
        assert service.get_job("alice", again["id"])["error"]["code"] == "INPUT_CHANGED"
        assert str(tmp_path) not in json.dumps(service.get_job("alice", again["id"]))
        await service.close()

    asyncio.run(run())


def test_missing_backend_fails_without_download_and_ttl_cleans_only_own_files(
    tmp_path, monkeypatch
):
    async def run():
        monkeypatch.delenv("MEDSEGAGENT_NNINTERACTIVE_PYTHON", raising=False)
        monkeypatch.delenv("MEDSEGAGENT_NNINTERACTIVE_MODEL_PATH", raising=False)
        base = Base(tmp_path / "base")
        service = OmniService(base)
        ws = await service.open_workspace("alice", "upload")
        assert service.capabilities() == {
            "nninteractive": {"configured": False, "inference_verified": False}
        }
        for operation in ["generate_report", "detect_modality"]:
            with pytest.raises(ServiceError):
                await service.submit("alice", ws["id"], operation, request(base, operation))
        job = await service.submit("alice", ws["id"], "refine", request(base))
        await drain(service, job["id"])
        assert service.get_job("alice", job["id"])["error"]["code"] == "BACKEND_NOT_CONFIGURED"
        with service.db:
            service.db.execute(
                "UPDATE workspaces SET expires=? WHERE id=?", (time.time() - 1, ws["id"])
            )
        with pytest.raises(ServiceError) as error:
            service.get_workspace("alice", ws["id"])
        assert error.value.status_code == 404
        assert not service._directory(ws["id"]).exists()
        assert base.source.is_file()
        await service.close()

    asyncio.run(run())


def test_worker_uses_existing_scheduler_process_runner_and_preserves_venv_path(
    tmp_path, monkeypatch
):
    async def run():
        base = Base(tmp_path / "base")
        service = OmniService(base)
        model = tmp_path / "model"
        model.mkdir()
        executable = tmp_path / "venv" / "bin" / "python"
        executable.parent.mkdir(parents=True)
        executable.symlink_to(Path("/usr/bin/true"))
        monkeypatch.setenv("MEDSEGAGENT_NNINTERACTIVE_PYTHON", str(executable))
        monkeypatch.setenv("MEDSEGAGENT_NNINTERACTIVE_MODEL_PATH", str(model))
        monkeypatch.setenv("MEDSEGAGENT_DEVICE", "cpu")
        monkeypatch.setenv("MEDSEGAGENT_SCHEDULER_LOCK_DIR", str(tmp_path / "locks"))
        recorded = {}

        async def child(command, **kwargs):
            recorded.update(command=command, lease=kwargs["lock_fd"])
            assert kwargs["lock_fd"].device == "cpu"
            assert kwargs["lock_fd"].pass_fds
            response = Path(command[command.index("--response") + 1])
            response.write_text('{"ok":true}')

        monkeypatch.setattr(core, "_run_command", child)
        directory = tmp_path / "job"
        directory.mkdir()
        assert await service._worker("nninteractive", {"prompts": []}, directory) == {"ok": True}
        assert recorded["command"][0] == str(executable)
        assert recorded["lease"]._closed
        await service.close()

    asyncio.run(run())


def test_agent_completion_requires_real_artifact_and_preserves_failed_outcome(tmp_path):
    async def run():
        base = Base(tmp_path / "base")
        service = OmniService(base)
        ws = await service.open_workspace("alice", "upload")
        for index, status in enumerate(["failed", "needs_input", "completed"]):

            async def runner(omni, workspace, body, jobdir, outcome=status):
                return {"status": outcome, "summary": "example", "unresolved": []}

            service.agent_runner = runner
            job = await service.submit(
                "alice", ws["id"], "agent", {"message_id": str(index), "instruction": "inspect"}
            )
            await drain(service, job["id"])
            observed = service.get_job("alice", job["id"])
            expected = {"failed": "failed", "needs_input": "input_required", "completed": "failed"}[
                status
            ]
            assert observed["status"] == expected
            if status == "completed":
                assert observed["error"]["code"] == "OUTPUT_INVALID"
            else:
                assert observed["result"]["status"] == status

        async def inspector(omni, workspace, body, jobdir):
            filename = "agent-inspection-1.json"
            (jobdir / filename).write_text(
                json.dumps({"ok": True, "shape": workspace["geometry"]["shape"], "revision": None})
            )
            return {
                "status": "completed",
                "summary": "shape inspected",
                "inspect_artifact": filename,
            }

        service.agent_runner = inspector
        valid = await service.submit(
            "alice", ws["id"], "agent", {"message_id": "inspect", "instruction": "shape"}
        )
        await drain(service, valid["id"])
        assert service.get_job("alice", valid["id"])["status"] == "completed"
        invalid = await service.submit(
            "alice",
            ws["id"],
            "agent",
            {"message_id": "unapplied", "instruction": "apply", "prompts": [mark(base)]},
        )
        await drain(service, invalid["id"])
        assert service.get_job("alice", invalid["id"])["error"]["code"] == "OUTPUT_INVALID"
        await service.close()

    asyncio.run(run())


def test_service_constructed_before_asgi_loop_can_run_in_loop_thread(tmp_path):
    base = Base(tmp_path / "base")
    service = OmniService(base)

    async def elsewhere():
        workspace = await service.open_workspace("alice", "upload")
        assert service.get_workspace("alice", workspace["id"])["upload_id"] == "upload"
        await service.close()

    # Mirrors create_app on the calling thread and ASGI lifespan on TestClient's loop.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(asyncio.run, elsewhere()).result(timeout=5)


def test_agent_cannot_claim_old_revision_fulfills_new_marks(tmp_path, monkeypatch):
    async def run():
        base = Base(tmp_path / "base")
        service = OmniService(base)
        monkeypatch.setattr(service, "_worker", synthetic_worker)
        ws = await service.open_workspace("alice", "upload")
        first = await service.submit("alice", ws["id"], "refine", request(base))
        await drain(service, first["id"])
        revision = service.get_job("alice", first["id"])["result"]["revision"]

        async def incorrect(omni, workspace, body, jobdir):
            return {"status": "completed", "revision": revision, "summary": "untrue completion"}

        service.agent_runner = incorrect
        job = await service.submit(
            "alice",
            ws["id"],
            "agent",
            request(
                base,
                "new",
                revision["id"],
                instruction="apply new mark",
                prompts=[mark(base, (1, 1, 1))],
            ),
        )
        await drain(service, job["id"])
        assert service.get_job("alice", job["id"])["error"]["code"] == "OUTPUT_INVALID"
        await service.close()

    asyncio.run(run())

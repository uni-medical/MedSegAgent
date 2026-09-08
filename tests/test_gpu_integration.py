"""Exercise scheduler/core boundaries with real subprocesses, without GPU inference."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time

import nibabel as nib
import numpy as np
import pytest

from medsegagent import agent, core, gpu_scheduler, weights
from medsegagent.service import Service, ServiceError

UUIDS = tuple(f"GPU-{index:08x}-1234-1234-1234-123456789abc" for index in range(1, 4))


@pytest.fixture
def gpu_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "require_weights", lambda *args, **kwargs: {"ready": True})
    config = gpu_scheduler.SchedulerConfig(
        lock_dir=tmp_path / "locks", allowed_gpus=("1", "2", "3"), poll_seconds=0.01
    )

    async def probe():
        return tuple(
            gpu_scheduler.GPUStatus(index, uuid, 24576, 24000, 0)
            for index, uuid in enumerate(UUIDS, 1)
        )

    def factory(*, device):
        return gpu_scheduler.GPUScheduler(config, device=device, probe=probe)

    monkeypatch.setenv("MEDSEGAGENT_DEVICE", "gpu")
    monkeypatch.setenv("MEDSEGAGENT_GPU_IDS", "1,2,3")
    monkeypatch.setenv("MEDSEGAGENT_SCHEDULER_LOCK_DIR", str(config.lock_dir))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(core, "GPUScheduler", factory)
    return factory(device="gpu")


async def wait_until(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def test_core_parallel_children_receive_distinct_gpu_and_all_lease_descriptors(
    tmp_path, monkeypatch, gpu_pool
):
    monkeypatch.setenv("GPU_TEST_API_KEY", "must-not-reach-inference")

    async def one(index):
        output = tmp_path / str(index)
        output.mkdir()
        async with core._inference_lock(time.monotonic() + 5) as lease:
            command = [
                sys.executable,
                "-c",
                (
                    "import json,os,time;"
                    f"fds={lease.pass_fds!r};"
                    "[os.fstat(fd) for fd in fds];"
                    "print(json.dumps({'visible':os.environ['CUDA_VISIBLE_DEVICES'],"
                    "'fd_count':len(fds),'secret_absent':'GPU_TEST_API_KEY' not in os.environ}),"
                    "flush=True);time.sleep(.2)"
                ),
            ]
            await core._run_command(
                command,
                timeout_seconds=3,
                output_dir=output,
                lock_fd=lease,
                on_start=lambda pid: None,
            )
            record = json.loads((output / "process.log").read_text())
            assert record == {
                "visible": lease.gpu_uuid,
                "fd_count": 2,
                "secret_absent": True,
            }
            return lease.gpu_uuid

    async def run():
        return await asyncio.gather(*(one(index) for index in range(3)))

    assert set(asyncio.run(run())) == set(UUIDS)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_inference_start_is_persisted_before_the_child_can_exist(tmp_path, monkeypatch, gpu_pool):
    source = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), np.eye(4)), source)
    started = False

    def on_inference_start():
        nonlocal started
        started = True

    async def launch(command, *, output_dir, on_start, **kwargs):
        # After this boundary, a fork may succeed even if the service dies before
        # receiving the PID. A durable queued task must not be replayable then.
        assert started, "Service still considers the job queued at the subprocess launch boundary"
        on_start(os.getpid())
        liver = next(k for k, v in core.task_labels("total").items() if v == "liver")
        nib.save(
            nib.Nifti1Image(np.full((3, 4, 5), liver, dtype=np.uint8), np.eye(4)),
            output_dir / "segmentation.nii.gz",
        )
        (output_dir / "run_report.json").write_text("{}")

    monkeypatch.setattr(core, "_run_command", launch)
    result = asyncio.run(
        core.segment(
            task="total",
            input_path=str(source),
            output_dir=str(tmp_path / "runs"),
            targets=["liver"],
            on_inference_start=on_inference_start,
        )
    )
    assert result["gpu_uuid"] in UUIDS
    assert core.read_run(result["output_dir"])["gpu_uuid"] == result["gpu_uuid"]


def test_repeated_cancellation_drains_child_before_releasing_lease(tmp_path, gpu_pool):
    ready = tmp_path / "ready"
    terminating = tmp_path / "terminating"
    output = tmp_path / "run"
    output.mkdir()
    code = (
        "import pathlib,signal,time;"
        f"signal.signal(signal.SIGTERM,lambda *_:pathlib.Path({str(terminating)!r}).touch());"
        f"pathlib.Path({str(ready)!r}).touch();time.sleep(60)"
    )

    async def run():
        pid = None

        def capture_pid(value):
            nonlocal pid
            pid = value

        async def infer():
            async with core._inference_lock(time.monotonic() + 10) as lease:
                await core._run_command(
                    [sys.executable, "-c", code],
                    timeout_seconds=60,
                    output_dir=output,
                    lock_fd=lease,
                    on_start=capture_pid,
                )

        task = asyncio.create_task(infer())
        try:
            await wait_until(ready.exists)
            task.cancel()
            await wait_until(terminating.exists)
            task.cancel()  # e.g. user cancel racing with service shutdown/total timeout
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 8)
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            async with gpu_pool.acquire(deadline=time.monotonic() + 1) as lease:
                assert lease.gpu_uuid == UUIDS[0]
        finally:
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_repeated_cancellation_during_spawn_keeps_handle_until_child_stops(
    tmp_path, monkeypatch, gpu_pool
):
    output = tmp_path / "run"
    output.mkdir()
    original_create = asyncio.create_subprocess_exec

    async def run():
        created, return_handle = asyncio.Event(), asyncio.Event()
        process = None

        async def delayed_handle(*args, **kwargs):
            nonlocal process
            process = await original_create(*args, **kwargs)
            created.set()
            await return_handle.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_handle)

        async def infer():
            async with core._inference_lock(time.monotonic() + 10) as lease:
                await core._run_command(
                    [sys.executable, "-c", "import time;time.sleep(60)"],
                    timeout_seconds=60,
                    output_dir=output,
                    lock_fd=lease,
                    on_start=lambda pid: pytest.fail("Canceled spawn must not signal running"),
                )

        task = asyncio.create_task(infer())
        try:
            await asyncio.wait_for(created.wait(), 3)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            return_handle.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 8)
            assert process.returncode is not None, "Repeated cancel lost the spawned child handle"
            async with gpu_pool.acquire(deadline=time.monotonic() + 1) as lease:
                assert lease.gpu_uuid == UUIDS[0]
        finally:
            return_handle.set()
            if process is not None and process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


async def synthetic_upload(service):
    upload_id, path = service.reserve_upload("alice", "ct.nii", size_bytes=4096)
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), np.eye(4)), path)
    await service.finish_upload("alice", upload_id, path, "ct.nii", path.stat().st_size)
    service.uploading.discard("alice")
    return upload_id


def test_service_admits_three_workers_queues_fourth_and_reclaims_canceled_slot(
    tmp_path, monkeypatch, gpu_pool
):
    async def run():
        calls = 0
        blocked = asyncio.Event()

        async def select(*args, **kwargs):
            nonlocal calls
            calls += 1
            await blocked.wait()

        monkeypatch.setattr(agent, "run_agent", select)
        service = Service(tmp_path / "service", "http://localhost")
        await service.start()
        try:
            upload_id = await synthetic_upload(service)
            jobs = [
                await service.submit("alice", upload_id, "CT liver", "CT", str(index))
                for index in range(4)
            ]
            with pytest.raises(ServiceError) as error:
                await service.submit("alice", upload_id, "CT liver", "CT", "fifth")
            assert error.value.code == "CAPACITY_EXCEEDED"
            await wait_until(lambda: calls == 3)
            assert service.get("alice", jobs[3]["id"])["status"] == "queued"
            await service.cancel("alice", jobs[0]["id"])
            await wait_until(lambda: calls == 4)
            assert service.get("alice", jobs[3]["id"])["status"] == "routing"
            assert service.get("alice", jobs[0]["id"])["status"] == "canceled"
        finally:
            await service.close()

    asyncio.run(run())


def test_service_restart_resumes_queued_only_and_never_replays_running(
    tmp_path, monkeypatch, gpu_pool
):
    async def run():
        service = Service(tmp_path / "service", "http://localhost")
        monkeypatch.setattr(service, "launch", lambda task_id: None)
        upload_id = await synthetic_upload(service)
        jobs = [
            await service.submit("alice", upload_id, "CT liver", "CT", str(index))
            for index in range(3)
        ]
        service.update(jobs[1]["id"], status="routing")
        service.update(jobs[2]["id"], status="running")
        await service.close()
        restarted = Service(tmp_path / "service", "http://localhost")
        resumed = []
        monkeypatch.setattr(restarted, "launch", resumed.append)
        await restarted.start()
        try:
            assert resumed == [jobs[0]["id"]]
            for previous in jobs[1:]:
                job = restarted.get("alice", previous["id"])
                assert job["status"] == "failed"
                assert job["error"]["code"] == "SERVER_RESTART"
        finally:
            await restarted.close()

    asyncio.run(run())

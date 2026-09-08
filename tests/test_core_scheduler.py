"""The inference entry point and real subprocess boundary use device leases."""

import asyncio
import json
import os
import sys
import time

import pytest
from test_core import fake_inference, image_at
from test_gpu_scheduler import UUIDS, available, config

from medsegagent import core, gpu_scheduler, weights


@pytest.fixture
def scheduled_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSEGAGENT_DEVICE", "gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(weights, "require_weights", lambda *args, **kwargs: {"ready": True})
    monkeypatch.setattr(
        core,
        "GPUScheduler",
        lambda *, device: gpu_scheduler.GPUScheduler(
            config(tmp_path / "locks", max_concurrent=2), device=device, probe=available
        ),
    )


def test_core_segments_use_distinct_gpu_leases_and_record_private_audit(
    tmp_path, monkeypatch, scheduled_runtime
):
    source = image_at(tmp_path / "synthetic.nii.gz")
    active, used = set(), set()
    peak = 0

    async def scenario():
        entered = asyncio.Event()

        async def inference(*args, lock_fd, **kwargs):
            nonlocal peak
            assert isinstance(lock_fd, gpu_scheduler.DeviceLease)
            assert lock_fd.gpu_uuid not in active
            active.add(lock_fd.gpu_uuid)
            used.add(lock_fd.gpu_uuid)
            peak = max(peak, len(active))
            if len(active) == 2:
                entered.set()
            try:
                await asyncio.wait_for(entered.wait(), 2)
                await fake_inference(*args, **kwargs)
            finally:
                active.remove(lock_fd.gpu_uuid)

        monkeypatch.setattr(core, "_run_command", inference)
        return await asyncio.gather(
            *(
                core.segment(
                    task="total",
                    input_path=str(source),
                    targets=["liver"],
                    output_dir=str(tmp_path / "runs"),
                )
                for _ in range(3)
            )
        )

    results = asyncio.run(scenario())
    assert peak == 2 and used == set(UUIDS[:2]) and not active
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    for result in results:
        state = core.read_run(result["output_dir"])
        assert state["gpu_uuid"] in used and state["gpu_index"] in {0, 1}
        assert state["status"] == "completed"


def test_inference_child_receives_selected_uuid_and_all_lock_descriptors(
    tmp_path, scheduled_runtime
):
    run = core._new_run(str(tmp_path / "runs"))

    async def scenario():
        async with core._inference_lock(time.monotonic() + 2) as lease:
            command = [
                sys.executable,
                "-c",
                (
                    "import json,os,sys; "
                    "print(json.dumps({'visible':os.environ.get('CUDA_VISIBLE_DEVICES'), "
                    "'locks':[os.fstat(int(fd)).st_ino for fd in sys.argv[1:]]}))"
                ),
                *(str(fd) for fd in lease.pass_fds),
            ]
            inodes = [os.fstat(fd).st_ino for fd in lease.pass_fds]
            await core._run_command(
                command, timeout_seconds=2, output_dir=run, lock_fd=lease, on_start=lambda pid: None
            )
            return lease.gpu_uuid, inodes

    expected_uuid, expected_locks = asyncio.run(scenario())
    observed = json.loads((run / "process.log").read_text())
    assert observed == {"visible": expected_uuid, "locks": expected_locks}
    assert len(expected_locks) == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

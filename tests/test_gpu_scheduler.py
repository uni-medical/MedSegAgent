"""Admission/ownership invariants; never require GPU hardware or run inference."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace

import pytest

from medsegagent import gpu_scheduler as scheduler

UUIDS = tuple(f"GPU-{index:08x}-1234-1234-1234-123456789abc" for index in range(4))
GPUS = tuple(scheduler.GPUStatus(i, uuid, 24576, 24000, 0) for i, uuid in enumerate(UUIDS))


def config(tmp_path, **kwargs):
    return scheduler.SchedulerConfig(lock_dir=tmp_path, poll_seconds=0.01, **kwargs)


async def available():
    return GPUS


def pool(tmp_path, **kwargs):
    return scheduler.GPUScheduler(config(tmp_path, **kwargs), device="gpu", probe=available)


def test_environment_policy_and_capacity(tmp_path):
    env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "MEDSEGAGENT_GPU_IDS": "1,2,3",
        "MEDSEGAGENT_GPU_MIN_FREE_MIB": "12000",
        "MEDSEGAGENT_GPU_MAX_UTILIZATION": "10",
        "MEDSEGAGENT_MAX_CONCURRENT_INFERENCES": "2",
        "MEDSEGAGENT_SCHEDULER_LOCK_DIR": str(tmp_path),
    }
    cfg = scheduler.SchedulerConfig.from_environment(env)
    assert cfg.allowed_gpus == ("1", "2", "3")
    assert cfg.min_free_memory_mib == 12000
    assert cfg.max_utilization_percent == 10
    assert scheduler.GPUScheduler(cfg, device="gpu").capacity == 2
    assert scheduler.GPUScheduler(cfg, device="mps").capacity == 1
    assert scheduler.GPUScheduler(cfg, device="cpu").capacity == 1
    assert scheduler.SchedulerConfig.from_environment(
        {"CUDA_VISIBLE_DEVICES": "2,3"}
    ).allowed_gpus == ("2", "3")
    assert scheduler.SchedulerConfig.from_environment({}).allowed_gpus is None


@pytest.mark.parametrize(
    "values",
    [
        {"MEDSEGAGENT_GPU_IDS": "1,"},
        {"MEDSEGAGENT_GPU_IDS": "1,1"},
        {"MEDSEGAGENT_GPU_IDS": "../1"},
        {"MEDSEGAGENT_GPU_IDS": "MIG-invalid"},
        {"MEDSEGAGENT_GPU_MIN_FREE_MIB": "0"},
        {"MEDSEGAGENT_GPU_MAX_UTILIZATION": "101"},
        {"MEDSEGAGENT_MAX_CONCURRENT_INFERENCES": "0"},
        {"MEDSEGAGENT_MAX_CONCURRENT_INFERENCES": "1.5"},
    ],
)
def test_invalid_configuration_is_rejected(values):
    with pytest.raises(scheduler.SchedulerError):
        scheduler.SchedulerConfig.from_environment(values)


def test_legacy_lock_path_gives_common_isolated_directory(tmp_path):
    path = tmp_path / "test.lock"
    cfg = scheduler.SchedulerConfig.from_environment({"MEDSEGAGENT_LOCK_PATH": str(path)})
    assert cfg.lock_dir == tmp_path / "test.lock.scheduler"


def test_csv_parsing_includes_foreign_compute_processes():
    result = scheduler.parse_nvidia_smi(
        f"0, {UUIDS[0]}, 24576, 22000, 0\n1, {UUIDS[1]}, 24576, 24000, 5\n",
        f"{UUIDS[0]}, 4216\n",
    )
    assert result[0].compute_pids == (4216,)
    assert result[1].compute_pids == ()
    assert result[1].free_memory_mib == 24000


@pytest.mark.parametrize(
    "gpu_csv,process_csv",
    [
        ("", ""),
        (f"0, {UUIDS[0]}, 24576, [N/A], 0", ""),
        (f"0, {UUIDS[0]}, 24576, 24000, nan", ""),
        (f"0, {UUIDS[0]}, 24576, 25000, 0", ""),
        (f"0, {UUIDS[0]}, 24576, 24000, 0\n0, {UUIDS[1]}, 24576, 24000, 0", ""),
        (f"0, {UUIDS[0]}, 24576, 24000, 0", f"{UUIDS[1]}, 123"),
        (f"0, {UUIDS[0]}, 24576, 24000, 0", f"{UUIDS[0]}, [N/A]"),
    ],
)
def test_unsupported_or_inconsistent_telemetry_fails_closed(gpu_csv, process_csv):
    with pytest.raises(scheduler.SchedulerError):
        scheduler.parse_nvidia_smi(gpu_csv, process_csv)


def test_selection_excludes_small_and_busy_gpus_and_prefers_more_free_memory(tmp_path):
    async def probe():
        return (
            replace(GPUS[0], free_memory_mib=22000, compute_pids=(4216,)),
            replace(GPUS[1], free_memory_mib=4000),
            replace(GPUS[2], utilization_percent=95),
            GPUS[3],
        )

    async def run():
        p = scheduler.GPUScheduler(config(tmp_path), device="gpu", probe=probe)
        async with p.acquire(deadline=time.monotonic() + 1) as lease:
            assert lease.gpu_index == 3
            assert lease.gpu_uuid == UUIDS[3]
            assert len(lease.pass_fds) == 2
            assert lease.audit_metadata()["device"] == "gpu"
            base = {"CUDA_VISIBLE_DEVICES": "1", "PATH": "unchanged"}
            assert lease.environment(base) == {
                "CUDA_VISIBLE_DEVICES": UUIDS[3],
                "PATH": "unchanged",
            }
            assert base["CUDA_VISIBLE_DEVICES"] == "1"

    asyncio.run(run())


def test_allowlist_uses_physical_uuid_and_prefers_idle_then_free_memory(tmp_path):
    async def probe():
        return (
            replace(GPUS[0], free_memory_mib=24500),
            replace(GPUS[1], free_memory_mib=23000),
            replace(GPUS[2], utilization_percent=5),
            replace(GPUS[3], free_memory_mib=24000),
        )

    async def run():
        p = scheduler.GPUScheduler(
            config(tmp_path, allowed_gpus=("1", "2", UUIDS[3])), device="gpu", probe=probe
        )
        async with p.acquire(deadline=time.monotonic() + 1) as lease:
            assert lease.gpu_index == 3

    asyncio.run(run())


def test_rechecks_after_locking_and_avoids_new_memory_pressure(tmp_path):
    calls = 0

    async def probe():
        nonlocal calls
        calls += 1
        return (
            GPUS
            if calls == 1
            else (replace(GPUS[0], free_memory_mib=2000, compute_pids=(9000,)), *GPUS[1:])
        )

    async def run():
        p = scheduler.GPUScheduler(config(tmp_path), device="gpu", probe=probe)
        async with p.acquire(deadline=time.monotonic() + 1) as lease:
            assert lease.gpu_index == 1
        # GPU 0 wasn't left locked during the failed first admission.
        async with pool(tmp_path).acquire(deadline=time.monotonic() + 1) as lease:
            assert lease.gpu_index == 0

    asyncio.run(run())


def test_resident_compute_process_is_eligible_when_memory_and_utilization_allow(tmp_path):
    async def probe():
        return (
            replace(GPUS[0], free_memory_mib=22000, compute_pids=(4216,)),
            *(replace(gpu, utilization_percent=90) for gpu in GPUS[1:]),
        )

    async def run():
        p = scheduler.GPUScheduler(config(tmp_path), device="gpu", probe=probe)
        async with p.acquire(deadline=time.monotonic() + 1) as lease:
            assert lease.gpu_index == 0
            assert lease.gpu_uuid == UUIDS[0]
            # A resident application is allowed, but a second MedSegAgent lease is not.
            with pytest.raises(scheduler.SchedulerTimeout):
                async with p.acquire(deadline=time.monotonic() + 0.05):
                    pytest.fail("An occupied MedSegAgent GPU lease was reused")
        async with p.acquire(deadline=time.monotonic() + 1) as lease:
            assert lease.gpu_index == 0

    asyncio.run(run())


def test_independent_schedulers_use_distinct_gpus_and_global_capacity(tmp_path):
    async def run():
        p = pool(tmp_path, max_concurrent=2)
        q = pool(tmp_path, max_concurrent=2)
        async with (
            p.acquire(deadline=time.monotonic() + 1) as first,
            q.acquire(deadline=time.monotonic() + 1) as second,
        ):
            assert (first.gpu_index, second.gpu_index) == (0, 1)
            with pytest.raises(scheduler.SchedulerTimeout):
                async with p.acquire(deadline=time.monotonic() + 0.05):
                    pytest.fail("Third lease bypassed the global capacity")
        async with p.acquire(deadline=time.monotonic() + 1) as lease:
            assert lease.gpu_index == 0

    asyncio.run(run())


def test_queued_cancellation_leaks_no_gpu_or_slot(tmp_path):
    async def run():
        p = pool(tmp_path, max_concurrent=1)
        async with p.acquire(deadline=time.monotonic() + 1):

            async def wait():
                async with p.acquire(deadline=time.monotonic() + 10):
                    pytest.fail("Queued task should never acquire the occupied slot")

            waiting = asyncio.create_task(wait())
            await asyncio.sleep(0.03)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        async with p.acquire(deadline=time.monotonic() + 1):
            pass

    asyncio.run(run())


def test_cancellation_during_second_probe_releases_all_locks(tmp_path):
    entered = None
    calls = 0

    async def run():
        nonlocal entered
        entered = asyncio.Event()

        async def probe():
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                await asyncio.sleep(60)
            return GPUS

        p = scheduler.GPUScheduler(config(tmp_path), device="gpu", probe=probe)

        async def acquire():
            async with p.acquire(deadline=time.monotonic() + 10):
                pytest.fail("Acquisition should be cancelled before yielding")

        pending = asyncio.create_task(acquire())
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        async with pool(tmp_path).acquire(deadline=time.monotonic() + 1):
            pass

    asyncio.run(run())


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_non_cuda_fallback_is_one_slot_and_never_queries_nvidia(tmp_path, device):
    async def forbidden():
        pytest.fail("CPU/MPS must not query NVIDIA")

    async def run():
        p = scheduler.GPUScheduler(config(tmp_path), device=device, probe=forbidden)
        async with p.acquire(deadline=time.monotonic() + 1) as lease:
            assert lease.device == device
            assert lease.gpu_index is None
            assert lease.environment({"A": "B"}) == {"A": "B"}
            assert len(lease.pass_fds) == 1
            with pytest.raises(scheduler.SchedulerTimeout):
                async with p.acquire(deadline=time.monotonic() + 0.05):
                    pytest.fail("Non-CUDA mode admitted multiple jobs")
        async with p.acquire(deadline=time.monotonic() + 1):
            pass

    asyncio.run(run())


@pytest.mark.parametrize("value", ["", "-1"])
def test_cuda_visibility_disabling_devices_does_not_expand_to_all(tmp_path, value):
    async def run():
        cfg = scheduler.SchedulerConfig.from_environment(
            {"CUDA_VISIBLE_DEVICES": value, "MEDSEGAGENT_SCHEDULER_LOCK_DIR": str(tmp_path)}
        )
        p = scheduler.GPUScheduler(cfg, device="gpu", probe=available)
        with pytest.raises(scheduler.SchedulerError, match="disables"):
            async with p.acquire(deadline=time.monotonic() + 1):
                pytest.fail("Explicitly disabled CUDA must not schedule a card")

    asyncio.run(run())


def test_unknown_allowlisted_gpu_fails_without_fallback(tmp_path):
    async def run():
        with pytest.raises(scheduler.SchedulerError, match="not present"):
            async with pool(tmp_path, allowed_gpus=("99",)).acquire(deadline=time.monotonic() + 1):
                pytest.fail("Unknown physical GPU must not select a different device")

    asyncio.run(run())


def test_inherited_child_holds_both_gpu_and_capacity_after_parent_closes(tmp_path):
    async def run():
        p = pool(tmp_path, max_concurrent=1)
        child = None
        try:
            async with p.acquire(deadline=time.monotonic() + 1) as lease:
                child = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    "import time; time.sleep(60)",
                    pass_fds=lease.pass_fds,
                )
            # The parent's normal close is insufficient while the child owns the FDs.
            with pytest.raises(scheduler.SchedulerTimeout):
                async with p.acquire(deadline=time.monotonic() + 0.05):
                    pytest.fail("Inherited lease was unlocked while the child was alive")
            child.terminate()
            await asyncio.wait_for(child.wait(), 5)
            async with p.acquire(deadline=time.monotonic() + 1):
                pass
        finally:
            if child is not None and child.returncode is None:
                child.kill()
                await child.wait()

    asyncio.run(run())


def test_parent_crash_preserves_lease_until_inherited_child_exits(tmp_path):
    code = f"""
import asyncio,json,os,subprocess,sys,time
from pathlib import Path
from medsegagent.gpu_scheduler import GPUScheduler,SchedulerConfig,GPUStatus
async def probe():
 return (GPUStatus(0,{UUIDS[0]!r},24576,24000,0),)
async def run():
 p=GPUScheduler(SchedulerConfig(lock_dir=Path({str(tmp_path)!r}),max_concurrent=1),device="gpu",probe=probe)
 async with p.acquire(deadline=time.monotonic()+2) as lease:
  child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],pass_fds=lease.pass_fds,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  print(json.dumps({{"child_pid":child.pid}}),flush=True)
  os._exit(0)
asyncio.run(run())
"""
    parent = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10, check=False
    )
    assert parent.returncode == 0, parent.stderr
    child_pid = json.loads(parent.stdout)["child_pid"]

    async def run():
        p = pool(tmp_path, max_concurrent=1)
        try:
            with pytest.raises(scheduler.SchedulerTimeout):
                async with p.acquire(deadline=time.monotonic() + 0.05):
                    pytest.fail("Parent crash prematurely released a child's lease")
        finally:
            os.kill(child_pid, signal.SIGTERM)
        # Wait for inherited descriptors to close, including orphan reaping latency.
        async with p.acquire(deadline=time.monotonic() + 3):
            pass

    asyncio.run(run())


def test_probe_timeout_kills_telemetry_child_and_releases_guard(tmp_path, monkeypatch):
    binary = tmp_path / "nvidia-smi"
    pid_file = tmp_path / "telemetry.pid"
    binary.write_text(
        f"#!{sys.executable}\nimport os,time\nopen({str(pid_file)!r},'w').write(str(os.getpid()))\ntime.sleep(60)\n"
    )
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))

    async def run():
        p = scheduler.GPUScheduler(
            config(tmp_path / "locks", query_timeout_seconds=1), device="gpu"
        )
        with pytest.raises(scheduler.SchedulerError, match="telemetry timed out"):
            async with p.acquire(deadline=time.monotonic() + 5):
                pytest.fail("Hung telemetry must not admit a job")
        assert pid_file.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
        async with pool(tmp_path / "locks").acquire(deadline=time.monotonic() + 1):
            pass

    asyncio.run(run())


def test_lock_symlink_is_not_followed(tmp_path):
    target = tmp_path / "untouched"
    target.write_text("original")
    (tmp_path / "scheduler.lock").symlink_to(target)

    async def run():
        with pytest.raises(OSError):
            async with pool(tmp_path).acquire(deadline=time.monotonic() + 1):
                pytest.fail("Scheduler followed an untrusted lock symlink")

    asyncio.run(run())
    assert target.read_text() == "original"

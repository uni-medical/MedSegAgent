"""Task wall time survives later publication, cleanup and legacy record reads."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from medsegagent import service as service_module
from medsegagent.service import Service


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


@pytest.fixture
def task(service):
    upload_id, path = service.reserve_upload("alice", "synthetic.nii")
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.int16), np.eye(4)), path)
    try:
        asyncio.run(
            service.finish_upload("alice", upload_id, path, "synthetic.nii", path.stat().st_size)
        )
    finally:
        service.uploading.discard("alice")
    return asyncio.run(service.submit("alice", upload_id, "分割肝脏", "CT", "timing-test"))


def stored(service, task_id):
    return service.db.execute("SELECT data FROM tasks WHERE id=?", (task_id,)).fetchone()[0]


def make_legacy(service, task_id):
    row = service._task(task_id)
    row.pop("finished_at", None)
    with service.db:
        service.db.execute("UPDATE tasks SET data=? WHERE id=?", (json.dumps(row), task_id))


def test_live_elapsed_uses_server_time_and_includes_queue(service, task, clock):
    assert task["elapsed_seconds"] == 0
    clock.value += 20
    queued = service.get("alice", task["id"])
    assert queued["elapsed_seconds"] == 20
    assert "finished_at" not in queued
    service.update(task["id"], status="routing")
    clock.value += 10.5
    assert service.get("alice", task["id"])["elapsed_seconds"] == 30.5
    assert service.list("alice")[0]["elapsed_seconds"] == 30.5
    assert service._task(task["id"])["created_at"] == task["created_at"]
    assert "elapsed_seconds" not in service._task(task["id"])


@pytest.mark.parametrize("status", ["completed", "failed", "canceled"])
def test_first_terminal_time_is_immutable(service, task, clock, status):
    clock.value += 37.25
    terminal = service.update(task["id"], status=status)
    assert terminal["finished_at"] == clock.value
    clock.value += 50
    service.update(task["id"], status=status, publication_error="EXAMPLE", finished_at=clock.value)
    row = service.get("alice", task["id"])
    assert row["finished_at"] == terminal["finished_at"]
    assert row["updated_at"] == clock.value
    assert row["elapsed_seconds"] == 37.25


def test_cleanup_and_retention_changes_do_not_extend_elapsed(service, task, clock):
    clock.value += 42
    terminal = service.update(task["id"], status="completed")
    clock.value = terminal["expires_at"] + 900
    service.retention_seconds *= 2
    service.cleanup()
    row = service.get("alice", task["id"])
    assert row["files_expired"] is True
    assert row["updated_at"] == clock.value
    assert row["finished_at"] == terminal["finished_at"]
    assert row["elapsed_seconds"] == 42


def test_legacy_finished_time_comes_from_first_terminal_event_without_writing(service, task, clock):
    clock.value += 12
    service.update(task["id"], status="running")
    clock.value += 30
    service.update(task["id"], status="completed")
    finished_at = clock.value
    make_legacy(service, task["id"])
    before = stored(service, task["id"])
    events_before = service.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    row = service.get("alice", task["id"])
    assert row["finished_at"] == finished_at
    assert row["elapsed_seconds"] == 42
    assert stored(service, task["id"]) == before
    assert "finished_at" not in json.loads(before)
    assert service.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == events_before
    clock.value += service.retention_seconds + 900
    service.cleanup()
    service.retention_seconds *= 2
    row = service.get("alice", task["id"])
    assert row["files_expired"] is True
    assert row["finished_at"] == finished_at
    assert row["elapsed_seconds"] == 42


def test_legacy_missing_terminal_evidence_has_unknown_duration(service, task, clock):
    clock.value += 42
    service.update(task["id"], status="completed")
    make_legacy(service, task["id"])
    with service.db:
        service.db.execute("DELETE FROM events WHERE task_id=?", (task["id"],))
    clock.value += service.retention_seconds + 900
    before = stored(service, task["id"])
    row = service.get("alice", task["id"])
    assert row["files_expired"] is True
    assert row["finished_at"] is None
    assert row["elapsed_seconds"] is None
    assert stored(service, task["id"]) == before
    service.cleanup()
    service.update(task["id"], publication_error="EXAMPLE")
    assert service.get("alice", task["id"])["elapsed_seconds"] is None
    assert service._task(task["id"])["finished_at"] is None


def test_clock_correction_does_not_produce_negative_elapsed(service, task, clock):
    clock.value -= 10
    assert service.get("alice", task["id"])["elapsed_seconds"] == 0
    service.update(task["id"], status="completed")
    assert service.get("alice", task["id"])["elapsed_seconds"] == 0


def install_execution(monkeypatch):
    class Execution:
        has_outputs = True
        unresolved_failures = ()

        def __init__(self, *, modality, **kwargs):
            self.modality = modality

        def snapshot(self):
            return {}

    monkeypatch.setattr(service_module, "TaskExecution", Execution)


def test_stage_times_are_persisted_without_counting_nested_work_twice(
    service, task, clock, monkeypatch
):
    install_execution(monkeypatch)
    monotonic = SimpleNamespace(value=100.0)
    monkeypatch.setattr(service_module, "perf_counter", lambda: monotonic.value)
    monkeypatch.setattr(
        service_module.TaskExecution,
        "export_result",
        lambda self: {
            "backend_results": [
                {
                    "task": "total",
                    "speed": "fast",
                    "inference_engine": "sequential",
                    "timings_seconds": {"device_wait": 0.2, "inference_subprocess": 4.0},
                },
                {
                    "task": "total_v3",
                    "speed": "fast",
                    "inference_engine": "sequential",
                    "timings_seconds": {"device_wait": 0.1, "inference_subprocess": 5.0},
                },
            ]
        },
        raising=False,
    )

    async def run_agent(text, modality, execution, *, on_progress):
        for phase, tool, stage, duration in (
            ("reasoning", None, "model_request", 1.25),
            ("observed", "segment", "tool", 6.0),
        ):
            monotonic.value += duration
            clock.value += duration
            await on_progress(
                {
                    "phase": phase,
                    "tool": tool,
                    "model_requests": 1,
                    "timing": {
                        "stage": stage,
                        "tool": tool,
                        "duration_seconds": duration,
                        "status": "completed",
                    },
                }
            )
        return {"status": "completed", "unresolved": [], "model_requests": 2, "tool_calls": 1}

    async def publish(*args):
        monotonic.value += 0.75
        clock.value += 0.75

    monkeypatch.setattr(service_module.agent, "run_agent", run_agent)
    monkeypatch.setattr(service, "publish_execution", publish)
    asyncio.run(service.run(task["id"]))
    result = service.get("alice", task["id"])
    timing = result["timings"]
    assert result["elapsed_seconds"] == 8
    assert timing["model_seconds"] == 1.25
    assert timing["tool_seconds"] == 6
    assert timing["publication_seconds"] == 0.75
    assert len(timing["events"]) == 2
    assert len(timing["inferences"]) == 2
    assert [row["engine"] for row in timing["inferences"]] == ["sequential"] * 2


def test_success_includes_publication_and_preserves_agent_counters(
    service, task, clock, monkeypatch
):
    install_execution(monkeypatch)

    async def run_agent(text, modality, execution, *, on_progress):
        clock.value += 30
        await on_progress({"phase": "observed", "tool": "segment", "step": 2, "model_requests": 2})
        return {"status": "completed", "unresolved": [], "model_requests": 3, "tool_calls": 1}

    async def publish(task_id, execution, outcome):
        progress = service.get("alice", task_id)["agent_progress"]
        assert progress == {
            "phase": "publishing",
            "tool": "segment",
            "step": 2,
            "model_requests": 3,
            "tool_calls": 1,
        }
        clock.value += 5
        service.update(task_id, result={"summary": "Complete", "files": []}, files=[])

    monkeypatch.setattr(service_module.agent, "run_agent", run_agent)
    monkeypatch.setattr(service, "publish_execution", publish)
    asyncio.run(service.run(task["id"]))
    row = service.get("alice", task["id"])
    assert row["status"] == "completed"
    assert row["elapsed_seconds"] == 35
    assert row["finished_at"] == clock.value


def test_partial_publication_after_failure_does_not_change_terminal_time(
    service, task, clock, monkeypatch
):
    install_execution(monkeypatch)

    async def run_agent(*args, **kwargs):
        clock.value += 30
        raise ValueError("Synthetic failure after producing an artifact")

    async def publish(task_id, execution, outcome):
        assert service.get("alice", task_id)["elapsed_seconds"] == 30
        clock.value += 5
        service.update(task_id, result={"summary": "Partial", "files": []}, files=[])

    monkeypatch.setattr(service_module.agent, "run_agent", run_agent)
    monkeypatch.setattr(service, "publish_execution", publish)
    asyncio.run(service.run(task["id"]))
    row = service.get("alice", task["id"])
    assert row["status"] == "failed"
    assert row["elapsed_seconds"] == 35
    assert row["updated_at"] == row["finished_at"]
    assert row["result"]["summary"] == "Partial"


def test_restart_freezes_interruption_time_and_preserves_terminal_time(service, task, clock):
    clock.value += 10
    service.update(task["id"], status="running")
    clock.value += 90

    async def restart():
        await service.start()
        service.cleanup_task.cancel()
        await asyncio.gather(service.cleanup_task, return_exceptions=True)
        service.lock.close()

    asyncio.run(restart())
    row = service.get("alice", task["id"])
    assert row["status"] == "failed"
    assert row["error"]["code"] == "SERVER_RESTART"
    assert row["elapsed_seconds"] == 100  # Task wall time includes the interruption.
    clock.value += 60
    asyncio.run(restart())
    assert service.get("alice", task["id"])["finished_at"] == row["finished_at"]
    assert service.get("alice", task["id"])["elapsed_seconds"] == 100

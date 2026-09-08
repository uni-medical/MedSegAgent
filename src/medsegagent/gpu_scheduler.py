"""Host-local inference leases shared by CLI, MCP, Web and A2A processes.

A CUDA lease reserves one physical GPU and one global concurrency slot. Only the
inference child's environment changes: its logical cuda:0 names the selected UUID.
All lease descriptors MUST be passed to that child. Closing the parent's descriptors
(not explicitly unlocking them) preserves exclusion if the parent dies first.

An external application does not participate in these locks. Admission uses current
free memory and utilization, including cards with resident applications. We recheck
after locking, but cannot reserve hardware against an unrelated application's growth.
"""

from __future__ import annotations

import asyncio
import csv
import fcntl
import io
import math
import os
import platform
import re
import stat
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


class SchedulerError(ValueError):
    """Invalid configuration or unavailable/untrustworthy device telemetry."""


class SchedulerTimeout(TimeoutError):
    """No eligible inference device became available before the deadline."""


def _integer(env: Mapping[str, str], key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(env.get(key, str(default)))
    except ValueError as exc:
        raise SchedulerError(f"{key} must be an integer between {low} and {high}.") from exc
    if not low <= value <= high:
        raise SchedulerError(f"{key} must be an integer between {low} and {high}.")
    return value


@dataclass(frozen=True)
class SchedulerConfig:
    lock_dir: Path = field(default_factory=lambda: Path.home() / ".cache/medsegagent/scheduler")
    allowed_gpus: tuple[str, ...] | None = None
    min_free_memory_mib: int = 8192
    max_utilization_percent: int = 20
    max_concurrent: int = 3
    poll_seconds: float = 0.25
    query_timeout_seconds: float = 5.0

    def __post_init__(self):
        object.__setattr__(self, "lock_dir", Path(self.lock_dir))
        if any(
            type(x) is not int
            for x in (self.max_concurrent, self.min_free_memory_mib, self.max_utilization_percent)
        ):
            raise SchedulerError("Scheduler capacity, memory and utilization must be integers.")
        if not 1 <= self.max_concurrent <= 64:
            raise SchedulerError("Maximum inference concurrency must be between 1 and 64.")
        if not 1 <= self.min_free_memory_mib <= 1048576:
            raise SchedulerError("Minimum free GPU memory must be a positive MiB value.")
        if not 0 <= self.max_utilization_percent <= 100:
            raise SchedulerError("GPU utilization threshold must be between 0 and 100.")
        if not all(
            math.isfinite(x) and x > 0 for x in (self.poll_seconds, self.query_timeout_seconds)
        ):
            raise SchedulerError("Scheduler polling and query timeouts must be positive.")
        if self.allowed_gpus is not None:
            if len(set(self.allowed_gpus)) != len(self.allowed_gpus):
                raise SchedulerError("GPU allowlist must not contain duplicates.")
            if any(
                not isinstance(x, str)
                or (not (x.isascii() and x.isdigit()) and not _UUID.fullmatch(x))
                for x in self.allowed_gpus
            ):
                raise SchedulerError(
                    "GPU allowlist must contain physical indices or full GPU UUIDs."
                )

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> SchedulerConfig:
        env = os.environ if env is None else env
        # Explicit operator policy overrides an inherited CUDA restriction. Otherwise
        # preserve that restriction, including an empty string/-1 disabling all GPUs.
        raw = env.get("MEDSEGAGENT_GPU_IDS", env.get("CUDA_VISIBLE_DEVICES"))
        allowed = None
        if raw is not None:
            allowed = () if raw.strip() in {"", "-1"} else tuple(x.strip() for x in raw.split(","))
        legacy = env.get("MEDSEGAGENT_LOCK_PATH")
        default_dir = (
            str(Path(legacy).expanduser().with_name(Path(legacy).name + ".scheduler"))
            if legacy
            else str(Path.home() / ".cache/medsegagent/scheduler")
        )
        return cls(
            lock_dir=Path(env.get("MEDSEGAGENT_SCHEDULER_LOCK_DIR", default_dir)).expanduser(),
            allowed_gpus=allowed,
            min_free_memory_mib=_integer(env, "MEDSEGAGENT_GPU_MIN_FREE_MIB", 8192, 1, 1048576),
            max_utilization_percent=_integer(env, "MEDSEGAGENT_GPU_MAX_UTILIZATION", 20, 0, 100),
            max_concurrent=_integer(env, "MEDSEGAGENT_MAX_CONCURRENT_INFERENCES", 3, 1, 64),
        )


@dataclass(frozen=True)
class GPUStatus:
    index: int
    uuid: str
    total_memory_mib: int
    free_memory_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...] = ()


def parse_nvidia_smi(gpu_csv: str, process_csv: str) -> tuple[GPUStatus, ...]:
    """Parse only explicit, numeric telemetry; unsupported/malformed output fails closed."""
    try:
        processes: dict[str, list[int]] = {}
        for row in csv.reader(io.StringIO(process_csv)):
            if not row:
                continue
            uuid, pid = [x.strip() for x in row]
            if not _UUID.fullmatch(uuid) or int(pid) <= 0:
                raise ValueError
            processes.setdefault(uuid, []).append(int(pid))
        result = []
        for row in csv.reader(io.StringIO(gpu_csv)):
            if not row:
                continue
            index, uuid, total, free, utilization = [x.strip() for x in row]
            index, total, free, utilization = map(int, (index, total, free, utilization))
            if (
                not _UUID.fullmatch(uuid)
                or index < 0
                or total <= 0
                or not 0 <= free <= total
                or not 0 <= utilization <= 100
            ):
                raise ValueError
            result.append(
                GPUStatus(index, uuid, total, free, utilization, tuple(processes.get(uuid, ())))
            )
        if (
            not result
            or len({x.index for x in result}) != len(result)
            or len({x.uuid for x in result}) != len(result)
            or set(processes) - {x.uuid for x in result}
        ):
            raise ValueError
        return tuple(result)
    except (ValueError, TypeError) as exc:
        raise SchedulerError("NVIDIA device telemetry is unavailable or invalid.") from exc


async def _query_nvidia_smi(arguments: list[str]) -> str:
    creation = asyncio.create_task(
        asyncio.create_subprocess_exec(
            "nvidia-smi",
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    )
    process = None
    try:
        process = await asyncio.shield(creation)
        stdout, _ = await process.communicate()
        if process.returncode:
            raise SchedulerError("NVIDIA device telemetry command failed.")
        if len(stdout) > 1024 * 1024:
            raise SchedulerError("NVIDIA device telemetry exceeds its size limit.")
        return stdout.decode("utf-8")
    except BaseException:
        if process is None:
            try:
                process = await creation
            except (OSError, asyncio.CancelledError):
                pass
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.communicate()
        raise


async def query_gpu_status() -> tuple[GPUStatus, ...]:
    try:
        gpu_csv = await _query_nvidia_smi(
            [
                "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        process_csv = await _query_nvidia_smi(
            [
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        )
        return parse_nvidia_smi(gpu_csv, process_csv)
    except (OSError, UnicodeError) as exc:
        raise SchedulerError("NVIDIA device telemetry is unavailable.") from exc


def _try_lock(path: Path) -> int | None:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise SchedulerError("Inference lock must be a private, owned regular file.")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@dataclass
class DeviceLease:
    device: str
    pass_fds: tuple[int, ...]
    gpu_index: int | None = None
    gpu_uuid: str | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def lock_fd(self) -> int:
        return self.pass_fds[0]

    def environment(self, base_env: Mapping[str, str]) -> dict[str, str]:
        if self._closed:
            raise SchedulerError("Inference device lease has already been released.")
        result = dict(base_env)
        if self.device == "gpu":
            result["CUDA_VISIBLE_DEVICES"] = self.gpu_uuid
        return result

    def audit_metadata(self) -> dict[str, str | int | None]:
        return {"device": self.device, "gpu_index": self.gpu_index, "gpu_uuid": self.gpu_uuid}

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            for descriptor in self.pass_fds:
                # No LOCK_UN: inherited descriptors keep the lease if a parent exits
                # while an inference descendant is still alive.
                os.close(descriptor)


class GPUScheduler:
    def __init__(
        self,
        config: SchedulerConfig | None = None,
        *,
        device: str | None = None,
        probe: Callable[[], Awaitable[tuple[GPUStatus, ...]]] = query_gpu_status,
    ):
        self.config = config or SchedulerConfig.from_environment()
        self.device = (
            device
            or os.environ.get(
                "MEDSEGAGENT_DEVICE", "mps" if platform.system() == "Darwin" else "gpu"
            ).lower()
        )
        if self.device not in {"cpu", "mps", "gpu"}:
            raise SchedulerError("Inference device must be cpu, mps or gpu.")
        self.probe = probe

    @property
    def capacity(self) -> int:
        if self.device != "gpu":
            return 1
        allowed = self.config.allowed_gpus
        if allowed == ():
            raise SchedulerError("The GPU allowlist disables all CUDA devices.")
        return (
            min(self.config.max_concurrent, len(allowed))
            if allowed is not None
            else self.config.max_concurrent
        )

    def _eligible(self, snapshot: tuple[GPUStatus, ...]) -> list[GPUStatus]:
        allowed = self.config.allowed_gpus
        if allowed is not None:
            tokens = {str(x.index) for x in snapshot} | {x.uuid for x in snapshot}
            if not set(allowed) <= tokens:
                raise SchedulerError("A configured GPU is not present in NVIDIA device telemetry.")
        candidates = [
            x
            for x in snapshot
            if (
                (allowed is None or str(x.index) in allowed or x.uuid in allowed)
                and x.free_memory_mib >= self.config.min_free_memory_mib
                and x.utilization_percent <= self.config.max_utilization_percent
            )
        ]
        # Available memory is the admission constraint; prefer idle utilization first,
        # then the most free memory. Physical index is only a stable tie-breaker.
        return sorted(
            candidates, key=lambda x: (x.utilization_percent, -x.free_memory_mib, x.index)
        )

    async def _snapshot(self, deadline: float) -> tuple[GPUStatus, ...]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SchedulerTimeout("Timed out waiting for an available inference device.")
        try:
            return await asyncio.wait_for(
                self.probe(), min(remaining, self.config.query_timeout_seconds)
            )
        except TimeoutError as exc:
            if time.monotonic() >= deadline:
                raise SchedulerTimeout(
                    "Timed out waiting for an available inference device."
                ) from exc
            raise SchedulerError("NVIDIA device telemetry timed out.") from exc

    async def _try_gpu(self, deadline: float) -> DeviceLease | None:
        directory = self.config.lock_dir
        guard = _try_lock(directory / "scheduler.lock")
        if guard is None:
            return None
        held: list[int] = []
        try:
            candidates = self._eligible(await self._snapshot(deadline))
            if not candidates:
                return None
            for index in range(self.config.max_concurrent):
                slot = _try_lock(directory / f"slot-{index}.lock")
                if slot is not None:
                    held.append(slot)
                    break
            else:
                return None
            for gpu in candidates:
                lock = _try_lock(directory / f"{gpu.uuid}.lock")
                if lock is None:
                    continue
                held.append(lock)
                # Recheck after acquiring the lock, before launching any child.
                fresh = self._eligible(await self._snapshot(deadline))
                if any(x.uuid == gpu.uuid for x in fresh):
                    if time.monotonic() >= deadline:
                        raise SchedulerTimeout(
                            "Timed out waiting for an available inference device."
                        )
                    lease = DeviceLease("gpu", (lock, slot), gpu.index, gpu.uuid)
                    held.clear()
                    return lease
                os.close(held.pop())
            return None
        finally:
            for descriptor in held:
                os.close(descriptor)
            os.close(guard)

    @asynccontextmanager
    async def acquire(self, *, deadline: float, device: str | None = None):
        selected = device or self.device
        if selected not in {"gpu", "cpu", "mps"}:
            raise SchedulerError("Inference device must be cpu, mps or gpu.")
        if not math.isfinite(deadline):
            raise SchedulerError("Inference deadline must be finite.")
        if selected == "gpu" and self.config.allowed_gpus == ():
            raise SchedulerError("The GPU allowlist disables all CUDA devices.")
        self.config.lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        lease = None
        try:
            while time.monotonic() < deadline:
                if selected == "gpu":
                    lease = await self._try_gpu(deadline)
                else:
                    fd = _try_lock(self.config.lock_dir / "cpu-mps.lock")
                    if fd is not None:
                        lease = DeviceLease(selected, (fd,))
                if lease is not None:
                    if time.monotonic() >= deadline:
                        break
                    yield lease
                    return
                await asyncio.sleep(
                    min(self.config.poll_seconds, max(0, deadline - time.monotonic()))
                )
            raise SchedulerTimeout("Timed out waiting for an available inference device.")
        finally:
            if lease is not None:
                lease.close()

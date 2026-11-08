"""GPU memory governance for astril.gpu: one process, many exam threads, never oversubscribe the device.

Why: on Windows the NVIDIA driver's default "CUDA sysmem fallback" lets allocations that exceed dedicated GPU memory
silently spill into system RAM (over PCIe) instead of failing, which slows everything by an order of magnitude. Without
an allocation failure PyTorch never frees its cache, so a busy process creeps past the device size.

What, per device (set up once per process by `prepare`):
  * a hard cap on this process's PyTorch allocations (torch.cuda.set_per_process_memory_fraction), sized from the
    memory free at first use minus headroom for the desktop and other programs. Exceeding it raises an out-of-memory
    error (after PyTorch empties its cache and retries) instead of spilling; callers then fall back to ITK.
  * a byte budget for the batched registration service (`service_bytes`).
  * a byte pool for heavy per-thread work (pyramid-level setup, B-spline resampling): `reserve(device, nbytes)` blocks
    until the estimated bytes fit, so 14 exam threads cannot all allocate at once. A request larger than the whole pool
    waits until it can run alone.

Environment overrides: ASTRIL_GPU_HEADROOM_MB (default 1024), ASTRIL_GPU_MAX_MEMORY_MB (cap ceiling).
"""
from __future__ import annotations

import contextlib
import os
import threading

import torch

SERVICE_SHARE = 0.50   # of the cap: batched service data (resident level data + per-iteration temporaries)
CLIENT_SHARE = 0.35    # of the cap: per-thread heavy operations; the rest is allocator fragmentation slack

_LOCK = threading.Lock()
_STATE: dict[str, "DeviceBudget"] = {}


def _key(device) -> str:
    d = torch.device(device)
    return f"cuda:{d.index if d.index is not None else 0}"


def _env_mb(name: str, default: float | None) -> float | None:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


class DeviceBudget:
    def __init__(self, device: str):
        self.device = torch.device(device)
        torch.zeros(1, device=self.device)                      # create the CUDA context first
        free, total = torch.cuda.mem_get_info(self.device)
        headroom = _env_mb("ASTRIL_GPU_HEADROOM_MB", 1024.0) * 2 ** 20
        cap = free - headroom
        ceiling = _env_mb("ASTRIL_GPU_MAX_MEMORY_MB", None)
        if ceiling is not None:
            cap = min(cap, ceiling * 2 ** 20)
        cap = max(cap, 512 * 2 ** 20)
        already = torch.cuda.memory_reserved(self.device)
        torch.cuda.set_per_process_memory_fraction(min(1.0, (cap + already) / total), self.device)
        self.cap_bytes = int(cap)
        self.total_bytes = int(total)
        self.service_bytes = int(cap * SERVICE_SHARE)
        self.client_pool = int(cap * CLIENT_SHARE)
        self._available = self.client_pool
        self._cond = threading.Condition()

    @contextlib.contextmanager
    def reserve(self, nbytes: int):
        need = int(min(max(nbytes, 1), self.client_pool))      # oversized requests run alone
        with self._cond:
            while self._available < need:
                self._cond.wait()
            self._available -= need
        try:
            yield
        finally:
            with self._cond:
                self._available += need
                self._cond.notify_all()


def prepare(device="cuda:0") -> DeviceBudget:
    """The process-wide budget for `device` (created, and the cap applied, on first use)."""
    k = _key(device)
    with _LOCK:
        if k not in _STATE:
            _STATE[k] = DeviceBudget(k)
        return _STATE[k]


@contextlib.contextmanager
def reserve(device, nbytes: int):
    with prepare(device).reserve(nbytes):
        yield


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def release_cache():
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# Measured peaks (tmp/measure_gpu_op_memory.py, float32 inputs): B-spline resample ~29 B per input voxel incl. its
# float64 coefficients and temporaries, plus the output; 2 mm pyramid-level setup ~230 B per virtual point at peak.
def resample_bytes(n_in_voxels: int, n_out_voxels: int) -> int:
    return int(32 * n_in_voxels + 8 * n_out_voxels)


def level_setup_bytes(n_points: int, n_moving_voxels: int) -> int:
    return int(260 * n_points + 48 * n_moving_voxels)

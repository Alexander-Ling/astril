"""Batched GPU registration service: one per GPU, shared by all exam threads of a process.

Clients (GpuImageRegistrationMethod.Execute in exam threads) prepare each pyramid level (ITK filters, virtual
points, LevelMetric upload) in their own thread, submit it, and block on a threading.Event (no CPU spin). A single
service thread owns the GPU: every iteration it evaluates ALL active levels in one batched pass (the same ~150
kernels whether 1 or 30 registrations are active), then advances each job's RegularStepGradientDescentOptimizerv4
state on the host. Per-job semantics are unchanged; batching only changes how the per-point work is launched.

Oversubscription controls:
  * admission budget: total virtual points across active levels (from torch.cuda.mem_get_info) and a job cap;
    levels beyond it wait FIFO in the queue;
  * optional spill-over (off by default; ASTRIL_GPU_SPILL_TO_CPU=1): when the queue is long, new Execute() calls run
    on ITK (CPU) instead (GpuImageRegistrationMethod);
  * one service per GPU per machine: a lock file per device; other processes do not get a service and use ITK.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .metrics import HIST_COPIES

try:  # fused kernels (pip: `triton` on Linux, `triton-windows` on Windows); pure-torch batch otherwise
    from .kernels import TritonBatch
    HAVE_TRITON = True
except Exception:  # noqa: BLE001 - any import/compile-environment problem -> pure-torch path
    TritonBatch = None
    HAVE_TRITON = False

RESIDENT_BYTES_PER_POINT = 160  # LevelMetric + batch buffers per virtual point (resident while a level is active)
TEMP_BYTES_PER_POINT = 720      # per-iteration temporaries per point inside one evaluated chunk (measured ~680-700)
CHUNK_POINTS = 3_000_000        # jobs are evaluated in chunks of whole jobs with at most this many points
SUM_COPIES = 256               # private copies for per-job segment sums (atomic-contention relief)
MATTES_OUTSIDE_MSG = ("All samples map outside moving image buffer. The images do not sufficiently overlap. They need "
                      "to be initialized to have more overlap before this metric will work. For instance, you can "
                      "align the image centers by translation.")


# ------------------------------------------------------------------------------------------ per-job state
@dataclass
class LevelState:
    """One pyramid level of one registration, as seen by the service."""
    lm: object                 # gpu_registration.LevelMetric (device data for this level)
    tfm: object                # gpu_metrics.TorchTransform (host params updated in place)
    scales: np.ndarray
    opt: dict
    on_iteration: object       # callable(value, nvalid, it, params) -> None (fires sitkIterationEvent)
    it: int = 0
    relax: float = 1.0
    grad: np.ndarray = None
    prev: np.ndarray = None
    stop: str = ""
    value: float = 0.0
    nvalid: int = 0
    error: BaseException = None
    done: threading.Event = field(default_factory=threading.Event)
    t_submit: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0

    def __post_init__(self):
        P = self.tfm.n_params
        self.grad, self.prev = np.zeros(P), np.zeros(P)
        self.inv_scales = 1.0 / self.scales

    def rsgd_step(self, value, deriv, nvalid) -> bool:
        """One RegularStepGradientDescentOptimizerv4 iteration given the metric at the current position.
        Returns True if the level continues."""
        o = self.opt
        self.prev, self.grad = self.grad, self.prev          # swap(m_PreviousGradient, m_Gradient)
        self.grad = np.asarray(deriv, dtype=np.float64) * self.inv_scales
        self.prev = self.prev * self.inv_scales
        self.value, self.nvalid = value, nvalid
        gmag = float(np.sqrt(np.sum(self.grad * self.grad)))
        if gmag < o["grad_tol"]:
            self.stop = (f"RegularStepGradientDescentOptimizerv4: Gradient magnitude tolerance met after {self.it} "
                         f"iterations. Gradient magnitude ({gmag}) is less than gradient magnitude tolerance ({o['grad_tol']}).")
            self.it += 1
            return False
        if float(np.dot(self.grad, self.prev)) < 0:
            self.relax *= o["relax"]
        step = self.relax * o["lr"]
        if step < o["min_step"]:
            self.stop = (f"RegularStepGradientDescentOptimizerv4: Step too small after {self.it} iterations. Current "
                         f"step ({step}) is less than minimum step ({o['min_step']}).")
            self.it += 1
            return False
        self.grad *= step
        self.tfm.set_params(self.tfm._params_np + self.grad * (1.0 / gmag))
        self.on_iteration(value, nvalid, self.it, self.tfm._params_np)
        self.it += 1
        return True

    def max_iterations_reached(self) -> bool:
        if self.it >= self.opt["iters"]:
            self.stop = f"RegularStepGradientDescentOptimizerv4: Maximum number of iterations ({self.opt['iters']}) exceeded."
            return True
        return False


# ------------------------------------------------------------------------------------- batched evaluation
class _Batch:
    """Flat, concatenated device buffers for a fixed set of active levels (rebuilt when the set changes)."""

    def __init__(self, states, device):
        self.states = list(states)
        self.device = device
        lms = [s.lm for s in self.states]
        J = len(lms)
        dev = device
        lengths = [lm.n_virtual for lm in lms]
        self.lengths = torch.as_tensor(lengths, device=dev)
        self.T = int(sum(lengths))
        self.job = torch.repeat_interleave(torch.arange(J, device=dev), self.lengths)
        self.local = torch.cat([torch.arange(n, device=dev) for n in lengths]) if self.T else torch.zeros(0, dtype=torch.int64, device=dev)
        self.ptsT = torch.cat([lm.vptsT for lm in lms], dim=1)
        self.vf = torch.cat([lm.vf for lm in lms])
        # Moving volumes, channel-last flat [V, 4] (value, gx, gy, gz), with per-job base offsets.
        vols = [lm.mov4.permute(1, 2, 3, 0).reshape(-1, 4) for lm in lms]
        vbase = np.cumsum([0] + [v.shape[0] for v in vols[:-1]])
        self.vol = torch.cat(vols)
        self.vbase = torch.as_tensor(vbase, device=dev)
        sizes = np.stack([lm._msize for lm in lms])                 # x,y,z
        self.size = torch.as_tensor(sizes, device=dev, dtype=torch.float64)
        self.size_i = self.size.to(torch.int64)
        self.is_mattes = torch.as_tensor([lm.metric == "mattes" for lm in lms], device=dev)
        # Moving masks (optional, per job): flat uint8 + geometry.
        self.any_mask = any(lm.mmask is not None for lm in lms)
        if self.any_mask:
            mvols, mbase, mgeo, has = [], [], [], []
            off = 0
            for lm in lms:
                if lm.mmask is None:
                    mvols.append(torch.zeros(1, device=dev)); mbase.append(off); off += 1
                    mgeo.append(np.zeros(12)); has.append(False)
                else:
                    g = lm.mmask
                    mvols.append(g.data[0].reshape(-1)); mbase.append(off); off += mvols[-1].numel()
                    mgeo.append(np.concatenate([g.phys_to_index.cpu().numpy().reshape(-1), g.origin.cpu().numpy()]))
                    has.append(True)
            self.mvol = torch.cat(mvols)
            self.mbase = torch.as_tensor(mbase, device=dev)
            self.mgeo = torch.as_tensor(np.stack(mgeo), device=dev)
            self.msize = torch.as_tensor(np.stack([lm.mmask.size_xyz.cpu().numpy() if lm.mmask is not None else np.ones(3)
                                                   for lm in lms]), device=dev)
            self.has_mask = torch.as_tensor(has, device=dev)
        # Mattes: per-job bins / ranges and histogram regions.
        bins = np.array([lm.bins if lm.metric == "mattes" else 0 for lm in lms])
        self.bins_np = bins
        hsize = HIST_COPIES * bins * bins
        hbase = np.cumsum([0] + list(hsize[:-1]))
        self.hist_len = int(hsize.sum())
        psize = bins * bins
        self.pbase_np = np.cumsum([0] + list(psize[:-1]))
        f64 = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64), device=dev)
        attr = lambda name, default=0.0: [getattr(lm, name, default) if lm.metric == "mattes" else default for lm in lms]
        self.m_min, self.m_max = f64(attr("m_min")), f64(attr("m_max", 1.0))
        self.inv_mbin, self.m_nmin = f64([1.0 / v if v else 0.0 for v in attr("m_bin", 1.0)]), f64(attr("m_nmin"))
        self.binsT = torch.as_tensor(bins, device=dev)
        if self.hist_len:
            fb = []
            for j, lm in enumerate(lms):
                if lm.metric != "mattes":
                    fb.append(torch.zeros(lm.n_virtual, dtype=torch.int64, device=dev)); continue
                B = lm.bins
                fterm = lm.vf / lm.f_bin - lm.f_nmin
                fbin = torch.clamp(fterm.to(torch.int64), 2, B - 3)
                copy = (torch.arange(lm.n_virtual, device=dev) % HIST_COPIES) * (B * B)
                fb.append(int(hbase[j]) + copy + fbin * B - 1)
            self.hist_off = torch.cat(fb)             # global histogram index of (copy, fbin, mbin=0) minus 1
            self.pr_off = torch.cat([int(self.pbase_np[jj]) + (torch.clamp((lm.vf / lm.f_bin - lm.f_nmin).to(torch.int64), 2, lm.bins - 3) * lm.bins - 1)
                                     if lm.metric == "mattes" else torch.zeros(lm.n_virtual, dtype=torch.int64, device=dev)
                                     for jj, lm in enumerate(lms)])
        self.sum_idx = self.job * SUM_COPIES + (self.local % SUM_COPIES)
        self.J = J
        self.pinned = torch.empty((J, 64), dtype=torch.float64, pin_memory=True)

    def evaluate(self):
        """Value / derivative / n_valid for every job at its current host parameters (one host sync)."""
        dev, J = self.device, self.J
        # Per-job affine maps: moving continuous index = A x + b; physical mapped point = M x + off.
        Ab, Moff = np.zeros((J, 12)), np.zeros((J, 12))
        for j, s in enumerate(self.states):
            M, off, _, _ = s.tfm.host_matrix_offset_jacobian()
            lm = s.lm
            Ab[j, :9] = (lm._p2i @ M).reshape(-1)
            Ab[j, 9:] = lm._p2i @ (off - lm._morigin)
            Moff[j, :9], Moff[j, 9:] = M.reshape(-1), off
        par = torch.as_tensor(Ab, device=dev)[self.job]                    # [T,12]
        x, y, z = self.ptsT[0], self.ptsT[1], self.ptsT[2]
        c = [x * par[:, 3 * i] + y * par[:, 3 * i + 1] + z * par[:, 3 * i + 2] + par[:, 9 + i] for i in range(3)]
        size = self.size[self.job]
        valid = (c[0] >= -0.5) & (c[0] < size[:, 0] - 0.5) & (c[1] >= -0.5) & (c[1] < size[:, 1] - 0.5) \
            & (c[2] >= -0.5) & (c[2] < size[:, 2] - 0.5)
        if self.any_mask:
            mo = torch.as_tensor(Moff, device=dev)[self.job]
            mp = [x * mo[:, 3 * i] + y * mo[:, 3 * i + 1] + z * mo[:, 3 * i + 2] + mo[:, 9 + i] for i in range(3)]
            g = self.mgeo[self.job]
            rel = [mp[i] - g[:, 9 + i] for i in range(3)]
            mi_ = [torch.floor(rel[0] * g[:, 3 * i] + rel[1] * g[:, 3 * i + 1] + rel[2] * g[:, 3 * i + 2] + 0.5) for i in range(3)]
            ms = self.msize[self.job]
            inside = (mi_[0] >= 0) & (mi_[0] < ms[:, 0]) & (mi_[1] >= 0) & (mi_[1] < ms[:, 1]) & (mi_[2] >= 0) & (mi_[2] < ms[:, 2])
            ii = [torch.clamp(m, min=0).to(torch.int64) for m in mi_]
            msi = ms.to(torch.int64)
            flat = self.mbase[self.job] + (ii[2] * msi[:, 1] + ii[1]) * msi[:, 0] + ii[0]
            flat = torch.where(inside, flat, self.mbase[self.job])
            mval = self.mvol[flat]
            valid &= ~self.has_mask[self.job] | (inside & (mval != 0))
        # Trilinear sample of [value, grad] with edge clamping (== grid_sample border / ITK linear interpolator).
        si = self.size_i[self.job]
        hi = si - 1
        base = [torch.minimum(torch.clamp(torch.floor(c[i]).to(torch.int64), min=0), hi[:, i]) for i in range(3)]
        d = [torch.clamp(c[i] - base[i].to(torch.float64), 0.0, 1.0).to(torch.float32) for i in range(3)]
        nx = [torch.minimum(base[i] + 1, hi[:, i]) for i in range(3)]
        vb = self.vbase[self.job]
        sx, sxy = si[:, 0], si[:, 0] * si[:, 1]
        acc = None
        for cz in (0, 1):
            iz = nx[2] if cz else base[2]; wz = d[2] if cz else 1.0 - d[2]
            for cy in (0, 1):
                iy = nx[1] if cy else base[1]; wy = d[1] if cy else 1.0 - d[1]
                row = vb + iz * sxy + iy * sx
                for cx in (0, 1):
                    ix = nx[0] if cx else base[0]; wx = d[0] if cx else 1.0 - d[0]
                    term = self.vol[row + ix] * (wx * wy * wz)[:, None]
                    acc = term if acc is None else acc + term
        mval, g = acc[:, 0], acc[:, 1:]                                       # float32
        out = torch.zeros((J, 64), dtype=torch.float64, device=dev)
        if self.hist_len:
            self._mattes(valid, mval, g, out)
        if not bool(self.is_mattes.all()):
            self._correlation(valid, mval, g, out)
        self.pinned.copy_(out, non_blocking=True)
        ev = torch.cuda.Event(blocking=True)
        ev.record()
        ev.synchronize()                                                       # yields the CPU while waiting
        return self.pinned.numpy()

    def _seg_sums(self, vals):
        """Per-job sums of per-point rows vals [T,K] (privatised index_add, then copy reduction) -> [J,K]."""
        K = vals.shape[1]
        acc = torch.zeros((self.J * SUM_COPIES, K), dtype=torch.float64, device=self.device)
        acc.index_add_(0, self.sum_idx, vals)
        return acc.view(self.J, SUM_COPIES, K).sum(dim=1)

    def _weighted_rows(self, wg):
        """[T,12] rows: wg (3) and wg_i * x_j (9), float64."""
        wg64 = wg.to(torch.float64)
        return torch.cat([wg64] + [wg64[:, i:i + 1] * self.ptsT.T for i in range(3)], dim=1)

    def _mattes(self, valid, mval, g, out):
        dev = self.device
        jb = self.job
        m64 = mval.to(torch.float64)
        valid = valid & self.is_mattes[jb] & (m64 >= self.m_min[jb]) & (m64 <= self.m_max[jb])
        w = valid.to(torch.float32)
        B = self.binsT[jb]
        mterm = torch.where(valid, (m64 * self.inv_mbin[jb] - self.m_nmin[jb]).to(torch.float32), torch.full_like(mval, 2.0))
        mbin = torch.minimum(torch.clamp(mterm.to(torch.int64), min=2), torch.clamp(B - 3, min=2))
        u = mterm - mbin.to(torch.float32)
        um, u2 = 1.0 - u, u * u
        u3 = u2 * u
        wts = torch.stack([um * um * um / 6.0, (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
                           (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0, u3 / 6.0]) * w
        idx = (self.hist_off + mbin)[None, :] + torch.arange(4, device=dev)[:, None]
        hist = torch.zeros(self.hist_len, dtype=torch.float64, device=dev)
        hist.index_add_(0, idx.reshape(-1), wts.reshape(-1).to(torch.float64))
        # Per-job histogram post-processing, batched over jobs that share a bin count.
        pr_all = torch.zeros(int(sum(b * b for b in self.bins_np)), dtype=torch.float32, device=dev)
        hb = 0
        groups = {}
        for j, b in enumerate(self.bins_np):
            if b:
                groups.setdefault(int(b), []).append((j, hb))
                hb += HIST_COPIES * b * b
        eps = np.finfo(np.float64).eps
        for Bv, members in groups.items():
            js = [m[0] for m in members]
            stack = torch.stack([hist[o:o + HIST_COPIES * Bv * Bv].view(HIST_COPIES, Bv * Bv).sum(dim=0) for _, o in members])
            jsum = stack.sum(dim=1, keepdim=True)
            jn = (stack / torch.where(jsum > 0, jsum, torch.ones_like(jsum))).view(-1, Bv, Bv)
            fmarg, mmarg = jn.sum(dim=2), jn.sum(dim=1)
            ok = (fmarg[:, :, None] > eps) & (mmarg[:, None, :] > eps) & (jn > eps)
            one = torch.ones_like(jn)
            pratio = torch.where(ok, torch.log(torch.where(ok, jn, one) / torch.where(ok, mmarg[:, None, :].expand_as(jn), one)),
                                 torch.zeros_like(jn))
            mi = torch.where(ok, jn * (pratio - torch.log(torch.where(ok, fmarg[:, :, None].expand_as(jn), one))),
                             torch.zeros_like(jn)).sum(dim=(1, 2))
            jt = torch.as_tensor(js, device=dev)
            out[jt, 0] = mi
            out[jt, 2] = jsum[:, 0]
            for k, j in enumerate(js):
                o = int(self.pbase_np[j])
                pr_all[o:o + Bv * Bv] = pratio[k].reshape(-1).to(torch.float32)
        pidx = (self.pr_off + mbin)[None, :] + torch.arange(4, device=dev)[:, None]
        dw = torch.stack([0.5 * um * um, 2.0 * u - 1.5 * u2, -2.0 * um + 1.5 * um * um, -0.5 * u2])
        wn = (pr_all[pidx] * dw).sum(dim=0) * w
        rows = torch.cat([w[:, None].to(torch.float64), self._weighted_rows(g * wn[:, None])], dim=1)   # [T,13]
        sums = self._seg_sums(rows)
        mattes_rows = self.is_mattes
        out[:, 1] = torch.where(mattes_rows, sums[:, 0], out[:, 1])
        out[:, 3:15] = torch.where(mattes_rows[:, None], sums[:, 1:13], out[:, 3:15])

    def _correlation(self, valid, mval, g, out):
        jb = self.job
        valid = valid & ~self.is_mattes[jb]
        w = valid.to(torch.float64)
        f, m = self.vf * w, mval.to(torch.float64) * w
        w32, f32, m32 = w.to(torch.float32), f.to(torch.float32), m.to(torch.float32)
        rows = torch.cat([torch.stack([w, f, m, f * f, m * m, f * m], dim=1),
                          self._weighted_rows(g * w32[:, None]), self._weighted_rows(g * f32[:, None]),
                          self._weighted_rows(g * m32[:, None])], dim=1)                     # [T,42]
        sums = self._seg_sums(rows)
        corr = ~self.is_mattes
        out[:, 16:58] = torch.where(corr[:, None], sums, out[:, 16:58])


def _host_result(state: LevelState, row: np.ndarray):
    """Turn a packed batch row into (value, ITK derivative, n_valid) for this job (host math, float64)."""
    lm, tfm = state.lm, state.tfm
    contract = lm._contract
    if lm.metric == "mattes":
        mi, n_valid, jsum = row[0], int(round(row[1])), row[2]
        if n_valid == 0:
            raise RuntimeError(MATTES_OUTSIDE_MSG)
        if jsum < np.finfo(np.float64).eps:
            raise RuntimeError("Joint PDF summed to zero")
        deriv = contract(tfm, row[3:15]) * (-1.0 / (lm.m_bin * n_valid))
        return -float(mi), deriv, n_valid
    n, sf, sm, sff, smm, sfm = row[16:22]
    n_valid, P = int(round(n)), tfm.n_params
    if n_valid <= 0:
        return float(np.finfo(np.float64).max), np.zeros(P), 0
    af, am = sf / n, sm / n
    fm, f2, m2 = sfm - n * af * am, sff - n * af * af, smm - n * am * am
    if m2 * f2 <= np.finfo(np.float64).eps:
        return float(np.finfo(np.float64).max), np.zeros(P), n_valid
    v1, vf_, vm = contract(tfm, row[22:34]), contract(tfm, row[34:46]), contract(tfm, row[46:58])
    fdm, mdm = vf_ - af * v1, vm - am * v1
    return -1.0 * fm * fm / (m2 * f2), 2.0 * fm / (f2 * m2) * (fdm - fm / m2 * mdm), n_valid


# ------------------------------------------------------------------------------------------- the service
class GpuRegistrationService:
    def __init__(self, device="cuda:0", max_jobs=48, max_points=None, chunk_points=None, use_triton=None):
        self.device = torch.device(device)
        from . import triton_usable
        ok = HAVE_TRITON and triton_usable(str(self.device))
        self.use_triton = ok if use_triton is None else (bool(use_triton) and ok)
        self._batch_cls = TritonBatch if self.use_triton else _Batch
        temp_bytes = 24 if self.use_triton else TEMP_BYTES_PER_POINT
        if chunk_points is None:
            chunk_points = 20_000_000 if self.use_triton else CHUNK_POINTS
        from . import memory
        # The service's share of this process's GPU cap (memory.py); per-thread setup/resampling has its own pool.
        # Budget resident data for all active levels plus the temporaries of one evaluated chunk.
        self.chunk_points = int(chunk_points)
        budget = memory.prepare(self.device).service_bytes - self.chunk_points * temp_bytes
        self.max_points = int(max_points or max(self.chunk_points, budget // RESIDENT_BYTES_PER_POINT))
        self.max_jobs = int(max_jobs)
        self._lock = threading.Condition()
        self._pending: list[LevelState] = []
        self._active: list[LevelState] = []
        self._batches: list[_Batch] = []
        self._dirty = True
        self.stats = dict(iterations=0, batched_evals=0, job_iterations=0, rebuilds=0, busy_s=0.0, max_active=0)
        self._thread = threading.Thread(target=self._loop, name="gpu-registration-service", daemon=True)
        self._thread.start()

    # -- client API --
    def queue_length(self) -> int:
        with self._lock:
            return len(self._pending)

    def load(self) -> tuple[int, int]:
        with self._lock:
            return len(self._active), len(self._pending)

    def run_level(self, state: LevelState):
        """Submit one level and block (without spinning) until it finishes. Re-raises metric errors."""
        state.t_submit = time.perf_counter()
        with self._lock:
            self._pending.append(state)
            self._lock.notify_all()
        state.done.wait()
        if state.error is not None:
            raise state.error

    # -- service loop --
    def _admit(self):
        pts = sum(s.lm.n_virtual for s in self._active)
        while self._pending:
            nxt = self._pending[0]
            if self._active and (len(self._active) >= self.max_jobs or pts + nxt.lm.n_virtual > self.max_points):
                break
            self._pending.pop(0)
            nxt.t_start = time.perf_counter()
            self._active.append(nxt)
            pts += nxt.lm.n_virtual
            self._dirty = True

    def _partition(self, states):
        """Whole-job chunks of at most chunk_points points (a single larger job gets its own chunk)."""
        groups, cur, n = [], [], 0
        for s in states:
            if cur and n + s.lm.n_virtual > self.chunk_points:
                groups.append(cur); cur, n = [], 0
            cur.append(s); n += s.lm.n_virtual
        if cur:
            groups.append(cur)
        return [self._batch_cls(g, self.device) for g in groups]

    def _finish(self, s: LevelState, error=None):
        s.error = error
        s.t_end = time.perf_counter()
        s.done.set()
        self._dirty = True

    def _loop(self):
        torch.cuda.set_device(self.device)
        while True:
            with self._lock:
                self._admit()
                while not self._active:
                    self._lock.wait()
                    self._admit()
                # Levels that hit their iteration cap end before evaluating (ITK checks this first).
                for s in list(self._active):
                    if s.max_iterations_reached():
                        self._active.remove(s); self._finish(s)
                if not self._active:
                    continue
                if self._dirty:
                    self._batches = self._partition(self._active)
                    self._dirty = False
                    self.stats["rebuilds"] += 1
                batches = self._batches
            t0 = time.perf_counter()
            try:
                states = [s for b in batches for s in b.states]
                try:
                    packed = np.concatenate([b.evaluate() for b in batches], axis=0)
                except Exception as e:  # noqa: BLE001
                    from .memory import is_oom, release_cache
                    if not is_oom(e):
                        raise
                    release_cache()   # transient pressure (e.g. many threads resampling): retry the iteration once
                    packed = np.concatenate([b.evaluate() for b in batches], axis=0)
            except Exception as e:  # unexpected GPU failure: fail every job in the batch
                with self._lock:
                    for s in [s for b in batches for s in b.states]:
                        if s in self._active:
                            self._active.remove(s); self._finish(s, e)
                    self._dirty = True
                continue
            self.stats["busy_s"] += time.perf_counter() - t0
            self.stats["batched_evals"] += 1
            self.stats["job_iterations"] += len(states)
            self.stats["max_active"] = max(self.stats["max_active"], len(states))
            ended = []
            for j, s in enumerate(states):
                try:
                    value, deriv, nvalid = _host_result(s, packed[j])
                    if not s.rsgd_step(value, deriv, nvalid):
                        ended.append((s, None))
                except Exception as e:
                    ended.append((s, e))
            if ended:
                with self._lock:
                    for s, e in ended:
                        self._active.remove(s); self._finish(s, e)


# ------------------------------------------------------------------------------- per-GPU service registry
_SERVICES: dict[str, GpuRegistrationService | None] = {}
_REG_LOCK = threading.Lock()
_LOCK_FILES = {}


def _try_machine_lock(device_index: int) -> bool:
    """Hold an exclusive lock file per GPU for the life of this process (one service per GPU per machine).

    The 1-byte lock file lives in astril.gpu.scratch_root() ($ASTRIL_SCRATCH_DIR, else the system temp dir); processes
    coordinate only if they resolve the same location.
    """
    from . import scratch_root
    path = os.path.join(scratch_root(), f"astril_gpu{device_index}.lock")
    fh = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    _LOCK_FILES[device_index] = fh
    return True


def get_service(device="cuda:0", **kw) -> GpuRegistrationService | None:
    """The process's service for `device`, or None if another process on this machine owns that GPU."""
    with _REG_LOCK:
        if device not in _SERVICES:
            idx = torch.device(device).index or 0
            _SERVICES[device] = GpuRegistrationService(device, **kw) if _try_machine_lock(idx) else None
        return _SERVICES[device]

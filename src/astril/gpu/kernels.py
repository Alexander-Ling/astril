"""Fused Triton kernels for the batched ITK-v4 metric evaluation (drop-in for gpu_service._Batch).

Same inputs and the same packed [J, 64] output layout as gpu_service._Batch.evaluate, so either can back the
service. Per iteration:
  Mattes:      pass 1 (map, bounds/mask, trilinear value+gradient, B-spline Parzen weights, privatised histogram
               atomics, store 24 B/point), small torch histogram post-processing (pRatio, MI), pass 2 (pRatio gather,
               derivative weights, in-block reduction of the 12 per-job sums, 12 atomics per block).
  Correlation: one pass (raw moments in float64 + weighted gradient sums).
Each Triton program handles one BLOCK of one job (program table), so per-job reductions happen inside the block.

Requires `triton` (Linux, ships with CUDA torch wheels) or `triton-windows` (Windows); no CUDA toolkit / compiler.
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .metrics import HIST_COPIES

BLOCK = 1024
NPF = 40   # float64 per-job parameters
NPI = 12   # int64 per-job parameters
# fpar: 0-8 A (moving cont. index = A x + b), 9-11 b, 12-14 moving size (x,y,z), 15 m_min, 16 m_max, 17 1/m_bin,
#       18 m_nmin, 19 bins, 20 has_mask, 21-29 Am, 30-32 bm (mask cont. index = Am x + bm), 33-35 mask size,
#       36-38 transform centre c (for sums relative to c)
# ipar: 0 vol_base, 1 sx, 2 sy, 3 sz, 4 hist_base, 5 pr_base, 6 mask_base, 7 msx, 8 msy, 9 msz


@triton.jit
def _map_and_sample(pts_ptr, T, offs, m, fp, ip, vol_ptr, mask_ptr, HAS_MASK: tl.constexpr):
    x = tl.load(pts_ptr + offs, mask=m, other=0.0)
    y = tl.load(pts_ptr + T + offs, mask=m, other=0.0)
    z = tl.load(pts_ptr + 2 * T + offs, mask=m, other=0.0)
    cx = x * tl.load(fp + 0) + y * tl.load(fp + 1) + z * tl.load(fp + 2) + tl.load(fp + 9)
    cy = x * tl.load(fp + 3) + y * tl.load(fp + 4) + z * tl.load(fp + 5) + tl.load(fp + 10)
    cz = x * tl.load(fp + 6) + y * tl.load(fp + 7) + z * tl.load(fp + 8) + tl.load(fp + 11)
    sxf = tl.load(fp + 12)
    syf = tl.load(fp + 13)
    szf = tl.load(fp + 14)
    valid = m & (cx >= -0.5) & (cx < sxf - 0.5) & (cy >= -0.5) & (cy < syf - 0.5) & (cz >= -0.5) & (cz < szf - 0.5)
    if HAS_MASK:
        has = tl.load(fp + 20)
        mx = tl.floor(x * tl.load(fp + 21) + y * tl.load(fp + 22) + z * tl.load(fp + 23) + tl.load(fp + 30) + 0.5)
        my = tl.floor(x * tl.load(fp + 24) + y * tl.load(fp + 25) + z * tl.load(fp + 26) + tl.load(fp + 31) + 0.5)
        mz = tl.floor(x * tl.load(fp + 27) + y * tl.load(fp + 28) + z * tl.load(fp + 29) + tl.load(fp + 32) + 0.5)
        msx = tl.load(fp + 33)
        msy = tl.load(fp + 34)
        msz = tl.load(fp + 35)
        inside = (mx >= 0) & (mx < msx) & (my >= 0) & (my < msy) & (mz >= 0) & (mz < msz)
        mflat = tl.load(ip + 6) + ((tl.maximum(mz, 0.0).to(tl.int64) * tl.load(ip + 8) + tl.maximum(my, 0.0).to(tl.int64))
                                   * tl.load(ip + 7) + tl.maximum(mx, 0.0).to(tl.int64))
        mv = tl.load(mask_ptr + mflat, mask=valid & inside, other=0.0)
        valid = valid & ((has == 0.0) | (inside & (mv != 0.0)))
    sx = tl.load(ip + 1)
    sy = tl.load(ip + 2)
    sz = tl.load(ip + 3)
    bx = tl.minimum(tl.maximum(tl.floor(cx), 0.0), sxf - 1.0)
    by = tl.minimum(tl.maximum(tl.floor(cy), 0.0), syf - 1.0)
    bz = tl.minimum(tl.maximum(tl.floor(cz), 0.0), szf - 1.0)
    dx = tl.minimum(tl.maximum(cx - bx, 0.0), 1.0).to(tl.float32)
    dy = tl.minimum(tl.maximum(cy - by, 0.0), 1.0).to(tl.float32)
    dz = tl.minimum(tl.maximum(cz - bz, 0.0), 1.0).to(tl.float32)
    ix0 = bx.to(tl.int64)
    iy0 = by.to(tl.int64)
    iz0 = bz.to(tl.int64)
    ix1 = tl.minimum(ix0 + 1, sx - 1)
    iy1 = tl.minimum(iy0 + 1, sy - 1)
    iz1 = tl.minimum(iz0 + 1, sz - 1)
    vb = tl.load(ip + 0)
    v0 = tl.zeros_like(dx)
    v1 = tl.zeros_like(dx)
    v2 = tl.zeros_like(dx)
    v3 = tl.zeros_like(dx)
    for corner in tl.static_range(8):
        ix = ix1 if (corner & 1) else ix0
        iy = iy1 if (corner & 2) else iy0
        iz = iz1 if (corner & 4) else iz0
        wx = dx if (corner & 1) else 1.0 - dx
        wy = dy if (corner & 2) else 1.0 - dy
        wz = dz if (corner & 4) else 1.0 - dz
        w = wx * wy * wz
        base = (vb + (iz * sy + iy) * sx + ix) * 4
        v0 += w * tl.load(vol_ptr + base + 0, mask=valid, other=0.0)
        v1 += w * tl.load(vol_ptr + base + 1, mask=valid, other=0.0)
        v2 += w * tl.load(vol_ptr + base + 2, mask=valid, other=0.0)
        v3 += w * tl.load(vol_ptr + base + 3, mask=valid, other=0.0)
    return x, y, z, valid, v0, v1, v2, v3


@triton.jit
def _mattes_pass1(pts_ptr, fbin_ptr, prog_job, prog_start, prog_end, fpar, ipar, vol_ptr, mask_ptr,
                  hist_ptr, cnt_ptr, st_mbin, st_u, st_g, T,
                  COPIES: tl.constexpr, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    j = tl.load(prog_job + pid)
    start = tl.load(prog_start + pid)
    end = tl.load(prog_end + pid)
    offs = start + tl.arange(0, BLOCK)
    m = offs < end
    fp = fpar + j * 40
    ip = ipar + j * 12
    x, y, z, valid, v0, g0, g1, g2 = _map_and_sample(pts_ptr, T, offs, m, fp, ip, vol_ptr, mask_ptr, HAS_MASK)
    mmin = tl.load(fp + 15).to(tl.float32)
    mmax = tl.load(fp + 16).to(tl.float32)
    valid = valid & (v0 >= mmin) & (v0 <= mmax)
    B = tl.load(fp + 19).to(tl.int32)
    mterm = v0 * tl.load(fp + 17).to(tl.float32) - tl.load(fp + 18).to(tl.float32)
    mterm = tl.where(valid, mterm, 2.0)
    mbin = tl.minimum(tl.maximum(mterm.to(tl.int32), 2), B - 3)
    u = mterm - mbin.to(tl.float32)
    um = 1.0 - u
    u2 = u * u
    u3 = u2 * u
    vf = valid.to(tl.float32)
    w0 = um * um * um / 6.0 * vf
    w1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0 * vf
    w2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0 * vf
    w3 = u3 / 6.0 * vf
    fbin = tl.load(fbin_ptr + offs, mask=m, other=0)
    hidx = tl.load(ip + 4) + (pid % COPIES) * (B * B) + fbin + mbin
    tl.atomic_add(hist_ptr + hidx, w0.to(tl.float64), mask=valid)
    tl.atomic_add(hist_ptr + hidx + 1, w1.to(tl.float64), mask=valid)
    tl.atomic_add(hist_ptr + hidx + 2, w2.to(tl.float64), mask=valid)
    tl.atomic_add(hist_ptr + hidx + 3, w3.to(tl.float64), mask=valid)
    tl.atomic_add(cnt_ptr + j, tl.sum(vf.to(tl.float64), axis=0))
    tl.store(st_mbin + offs, tl.where(valid, mbin, -1), mask=m)
    tl.store(st_u + offs, u, mask=m)
    tl.store(st_g + offs, g0, mask=m)
    tl.store(st_g + T + offs, g1, mask=m)
    tl.store(st_g + 2 * T + offs, g2, mask=m)


@triton.jit
def _mattes_pass2(pts_ptr, fbin_ptr, prog_job, prog_start, prog_end, fpar, ipar, pr_ptr,
                  st_mbin, st_u, st_g, sums_ptr, T, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    j = tl.load(prog_job + pid)
    start = tl.load(prog_start + pid)
    end = tl.load(prog_end + pid)
    offs = start + tl.arange(0, BLOCK)
    m = offs < end
    fp = fpar + j * 40
    ip = ipar + j * 12
    mbin = tl.load(st_mbin + offs, mask=m, other=-1)
    valid = m & (mbin >= 0)
    u = tl.load(st_u + offs, mask=m, other=0.0)
    um = 1.0 - u
    u2 = u * u
    d0 = 0.5 * um * um
    d1 = 2.0 * u - 1.5 * u2
    d2 = -2.0 * um + 1.5 * um * um
    d3 = -0.5 * u2
    pidx = tl.load(ip + 5) + tl.load(fbin_ptr + offs, mask=m, other=0) + tl.maximum(mbin, 2)
    wn = (tl.load(pr_ptr + pidx, mask=valid, other=0.0) * d0 + tl.load(pr_ptr + pidx + 1, mask=valid, other=0.0) * d1
          + tl.load(pr_ptr + pidx + 2, mask=valid, other=0.0) * d2 + tl.load(pr_ptr + pidx + 3, mask=valid, other=0.0) * d3)
    wn = tl.where(valid, wn, 0.0)
    g0 = tl.load(st_g + offs, mask=m, other=0.0) * wn
    g1 = tl.load(st_g + T + offs, mask=m, other=0.0) * wn
    g2 = tl.load(st_g + 2 * T + offs, mask=m, other=0.0) * wn
    rx = (tl.load(pts_ptr + offs, mask=m, other=0.0) - tl.load(fp + 36)).to(tl.float32)
    ry = (tl.load(pts_ptr + T + offs, mask=m, other=0.0) - tl.load(fp + 37)).to(tl.float32)
    rz = (tl.load(pts_ptr + 2 * T + offs, mask=m, other=0.0) - tl.load(fp + 38)).to(tl.float32)
    o = sums_ptr + j * 16
    tl.atomic_add(o + 0, tl.sum(g0, axis=0).to(tl.float64))
    tl.atomic_add(o + 1, tl.sum(g1, axis=0).to(tl.float64))
    tl.atomic_add(o + 2, tl.sum(g2, axis=0).to(tl.float64))
    tl.atomic_add(o + 3, tl.sum(g0 * rx, axis=0).to(tl.float64))
    tl.atomic_add(o + 4, tl.sum(g0 * ry, axis=0).to(tl.float64))
    tl.atomic_add(o + 5, tl.sum(g0 * rz, axis=0).to(tl.float64))
    tl.atomic_add(o + 6, tl.sum(g1 * rx, axis=0).to(tl.float64))
    tl.atomic_add(o + 7, tl.sum(g1 * ry, axis=0).to(tl.float64))
    tl.atomic_add(o + 8, tl.sum(g1 * rz, axis=0).to(tl.float64))
    tl.atomic_add(o + 9, tl.sum(g2 * rx, axis=0).to(tl.float64))
    tl.atomic_add(o + 10, tl.sum(g2 * ry, axis=0).to(tl.float64))
    tl.atomic_add(o + 11, tl.sum(g2 * rz, axis=0).to(tl.float64))


@triton.jit
def _wsums(o, g0, g1, g2, rx, ry, rz):
    tl.atomic_add(o + 0, tl.sum(g0, axis=0).to(tl.float64))
    tl.atomic_add(o + 1, tl.sum(g1, axis=0).to(tl.float64))
    tl.atomic_add(o + 2, tl.sum(g2, axis=0).to(tl.float64))
    tl.atomic_add(o + 3, tl.sum(g0 * rx, axis=0).to(tl.float64))
    tl.atomic_add(o + 4, tl.sum(g0 * ry, axis=0).to(tl.float64))
    tl.atomic_add(o + 5, tl.sum(g0 * rz, axis=0).to(tl.float64))
    tl.atomic_add(o + 6, tl.sum(g1 * rx, axis=0).to(tl.float64))
    tl.atomic_add(o + 7, tl.sum(g1 * ry, axis=0).to(tl.float64))
    tl.atomic_add(o + 8, tl.sum(g1 * rz, axis=0).to(tl.float64))
    tl.atomic_add(o + 9, tl.sum(g2 * rx, axis=0).to(tl.float64))
    tl.atomic_add(o + 10, tl.sum(g2 * ry, axis=0).to(tl.float64))
    tl.atomic_add(o + 11, tl.sum(g2 * rz, axis=0).to(tl.float64))


@triton.jit
def _correlation_pass(pts_ptr, vf_ptr, prog_job, prog_start, prog_end, fpar, ipar, vol_ptr, mask_ptr, sums_ptr, T,
                      HAS_MASK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    j = tl.load(prog_job + pid)
    start = tl.load(prog_start + pid)
    end = tl.load(prog_end + pid)
    offs = start + tl.arange(0, BLOCK)
    m = offs < end
    fp = fpar + j * 40
    ip = ipar + j * 12
    x, y, z, valid, v0, g0, g1, g2 = _map_and_sample(pts_ptr, T, offs, m, fp, ip, vol_ptr, mask_ptr, HAS_MASK)
    w = valid.to(tl.float64)
    f = tl.load(vf_ptr + offs, mask=m, other=0.0) * w
    mv = v0.to(tl.float64) * w
    o = sums_ptr + j * 48
    tl.atomic_add(o + 0, tl.sum(w, axis=0))
    tl.atomic_add(o + 1, tl.sum(f, axis=0))
    tl.atomic_add(o + 2, tl.sum(mv, axis=0))
    tl.atomic_add(o + 3, tl.sum(f * f, axis=0))
    tl.atomic_add(o + 4, tl.sum(mv * mv, axis=0))
    tl.atomic_add(o + 5, tl.sum(f * mv, axis=0))
    rx = (x - tl.load(fp + 36)).to(tl.float32)
    ry = (y - tl.load(fp + 37)).to(tl.float32)
    rz = (z - tl.load(fp + 38)).to(tl.float32)
    w32 = w.to(tl.float32)
    f32 = f.to(tl.float32)
    m32 = mv.to(tl.float32)
    _wsums(o + 6, g0 * w32, g1 * w32, g2 * w32, rx, ry, rz)
    _wsums(o + 18, g0 * f32, g1 * f32, g2 * f32, rx, ry, rz)
    _wsums(o + 30, g0 * m32, g1 * m32, g2 * m32, rx, ry, rz)


def _program_table(lengths, include, dev):
    """(job, start, end) per BLOCK for the included jobs; offsets always advance over every job's points."""
    job, start, end = [], [], []
    off = 0
    for j, n in enumerate(lengths):
        if include[j]:
            for b in range(-(-n // BLOCK)):
                job.append(j); start.append(off + b * BLOCK); end.append(off + n)
        off += n
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=dev)
    return t(job), t(start), t(end), len(job)


class TritonBatch:
    """Same interface as gpu_service._Batch: TritonBatch(states, device).evaluate() -> packed [J, 64] numpy."""

    def __init__(self, states, device):
        self.states, self.device = list(states), device
        dev = device
        lms = [s.lm for s in self.states]
        self.J = J = len(lms)
        self.lengths = [lm.n_virtual for lm in lms]
        self.T = int(sum(self.lengths))
        self.ptsT = torch.cat([lm.vptsT for lm in lms], dim=1).contiguous() if self.T else torch.zeros((3, 0), dtype=torch.float64, device=dev)
        self.vf = torch.cat([lm.vf for lm in lms]).contiguous()
        vols = [lm.mov4.permute(1, 2, 3, 0).reshape(-1) for lm in lms]
        self.vol = torch.cat(vols).contiguous()
        vbase = np.cumsum([0] + [v.numel() // 4 for v in vols[:-1]])
        self.is_mattes = np.array([lm.metric == "mattes" for lm in lms])
        bins = np.array([lm.bins if lm.metric == "mattes" else 0 for lm in lms])
        self.bins = bins
        hsize = HIST_COPIES * bins * bins
        self.hbase = np.cumsum([0] + list(hsize[:-1]))
        self.hist_len = int(hsize.sum())
        self.pbase = np.cumsum([0] + list((bins * bins)[:-1]))
        self.pr_len = int((bins * bins).sum())
        ipar = np.zeros((J, NPI), dtype=np.int64)
        self.fpar_static = np.zeros((J, NPF))
        mvols, moff = [], 0
        self.any_mask = any(lm.mmask is not None for lm in lms)
        for j, lm in enumerate(lms):
            ipar[j, 0] = vbase[j]
            ipar[j, 1:4] = lm._msize.astype(np.int64)
            ipar[j, 4], ipar[j, 5] = self.hbase[j], self.pbase[j]
            fs = self.fpar_static[j]
            fs[12:15] = lm._msize
            if lm.metric == "mattes":
                fs[15], fs[16], fs[17], fs[18], fs[19] = lm.m_min, lm.m_max, 1.0 / lm.m_bin, lm.m_nmin, lm.bins
            if lm.mmask is not None:
                g = lm.mmask
                fs[20] = 1.0
                fs[33:36] = g.size_xyz.cpu().numpy()
                ipar[j, 6] = moff
                ipar[j, 7:10] = g.size_xyz.cpu().numpy().astype(np.int64)
                mvols.append(g.data[0].reshape(-1)); moff += mvols[-1].numel()
        self.ipar = torch.as_tensor(ipar, device=dev)
        self.mask = torch.cat(mvols).contiguous() if mvols else torch.zeros(1, device=dev)
        # Fixed-bin offsets (fbin*B - 1, local to the job's histogram row block) for Mattes points.
        fb = []
        for lm in lms:
            if lm.metric == "mattes":
                fterm = lm.vf / lm.f_bin - lm.f_nmin
                fb.append((torch.clamp(fterm.to(torch.int64), 2, lm.bins - 3) * lm.bins - 1).to(torch.int32))
            else:
                fb.append(torch.zeros(lm.n_virtual, dtype=torch.int32, device=dev))
        self.fbin = torch.cat(fb).contiguous()
        self.prog_m = _program_table(self.lengths, self.is_mattes, dev)
        self.prog_c = _program_table(self.lengths, ~self.is_mattes, dev)
        self.st_mbin = torch.empty(self.T, dtype=torch.int32, device=dev)
        self.st_u = torch.empty(self.T, dtype=torch.float32, device=dev)
        self.st_g = torch.empty((3, self.T), dtype=torch.float32, device=dev)
        self.pinned = torch.empty((J, 64), dtype=torch.float64, pin_memory=True)
        groups = {}
        for j, b in enumerate(bins):
            if b:
                groups.setdefault(int(b), []).append(j)
        self.groups = groups

    def _fpar(self):
        fp = self.fpar_static.copy()
        for j, s in enumerate(self.states):
            M, off, _, _ = s.tfm.host_matrix_offset_jacobian()
            lm = s.lm
            fp[j, 0:9] = (lm._p2i @ M).reshape(-1)
            fp[j, 9:12] = lm._p2i @ (off - lm._morigin)
            if lm.mmask is not None:
                p2i = lm.mmask.phys_to_index.cpu().numpy() if not hasattr(lm, "_mp2i") else lm._mp2i
                mor = lm.mmask.origin.cpu().numpy() if not hasattr(lm, "_mor") else lm._mor
                lm._mp2i, lm._mor = p2i, mor
                fp[j, 21:30] = (p2i @ M).reshape(-1)
                fp[j, 30:33] = p2i @ (off - mor)
            fp[j, 36:39] = s.tfm._center_np
        return torch.as_tensor(fp, device=self.device)

    def evaluate(self):
        dev, J = self.device, self.J
        fpar = self._fpar()
        out = torch.zeros((J, 64), dtype=torch.float64, device=dev)
        pj, ps, pe, n_m = self.prog_m
        if n_m:
            hist = torch.zeros(self.hist_len, dtype=torch.float64, device=dev)
            cnt = torch.zeros(J, dtype=torch.float64, device=dev)
            _mattes_pass1[(n_m,)](self.ptsT, self.fbin, pj, ps, pe, fpar, self.ipar, self.vol, self.mask, hist, cnt,
                                  self.st_mbin, self.st_u, self.st_g, self.T,
                                  COPIES=HIST_COPIES, HAS_MASK=self.any_mask, BLOCK=BLOCK)
            pr_all = torch.zeros(max(1, self.pr_len), dtype=torch.float32, device=dev)
            eps = np.finfo(np.float64).eps
            for Bv, js in self.groups.items():
                stack = torch.stack([hist[int(self.hbase[j]):int(self.hbase[j]) + HIST_COPIES * Bv * Bv]
                                     .view(HIST_COPIES, Bv * Bv).sum(dim=0) for j in js])
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
                    o = int(self.pbase[j])
                    pr_all[o:o + Bv * Bv] = pratio[k].reshape(-1).to(torch.float32)
            sums = torch.zeros((J, 16), dtype=torch.float64, device=dev)
            _mattes_pass2[(n_m,)](self.ptsT, self.fbin, pj, ps, pe, fpar, self.ipar, pr_all,
                                  self.st_mbin, self.st_u, self.st_g, sums, self.T, BLOCK=BLOCK)
            out[:, 1] = cnt
            out[:, 3:15] = sums[:, :12]
        pj, ps, pe, n_c = self.prog_c
        if n_c:
            csums = torch.zeros((J, 48), dtype=torch.float64, device=dev)
            _correlation_pass[(n_c,)](self.ptsT, self.vf, pj, ps, pe, fpar, self.ipar, self.vol, self.mask, csums, self.T,
                                      HAS_MASK=self.any_mask, BLOCK=BLOCK)
            out[:, 16:58] = csums[:, :42]
        self.pinned.copy_(out, non_blocking=True)
        ev = torch.cuda.Event(blocking=True)
        ev.record()
        ev.synchronize()
        res = self.pinned.numpy().copy()
        # Kernels accumulate sums relative to the transform centre c; convert to S = sum wg_i x_j (host contract form).
        for j, s in enumerate(self.states):
            c = s.tfm._center_np
            blocks = [3] if self.is_mattes[j] else [22, 34, 46]
            for b0 in blocks:
                h = res[j, b0:b0 + 3]
                res[j, b0 + 3:b0 + 12] += np.outer(h, c).reshape(-1)
        return res


# ----------------------------------------------------------------------------------------------- resampling
@triton.jit
def _resample_kernel(src_ptr, out_ptr, n_out, osx, osy, A_ptr, default, isx, isy, isz,
                     MODE: tl.constexpr, BLOCK: tl.constexpr):
    """Output voxel (i,j,k) -> input continuous index c = A [i,j,k,1]; MODE 0 nearest, 1 linear, 3 cubic B-spline."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_out
    i = (offs % osx).to(tl.float64)
    j = ((offs // osx) % osy).to(tl.float64)
    k = (offs // (osx * osy)).to(tl.float64)
    cx = tl.load(A_ptr + 0) * i + tl.load(A_ptr + 1) * j + tl.load(A_ptr + 2) * k + tl.load(A_ptr + 3)
    cy = tl.load(A_ptr + 4) * i + tl.load(A_ptr + 5) * j + tl.load(A_ptr + 6) * k + tl.load(A_ptr + 7)
    cz = tl.load(A_ptr + 8) * i + tl.load(A_ptr + 9) * j + tl.load(A_ptr + 10) * k + tl.load(A_ptr + 11)
    fx = isx.to(tl.float64)
    fy = isy.to(tl.float64)
    fz = isz.to(tl.float64)
    inside = m & (cx >= -0.5) & (cx < fx - 0.5) & (cy >= -0.5) & (cy < fy - 0.5) & (cz >= -0.5) & (cz < fz - 0.5)
    if MODE == 0:
        nx = tl.floor(cx + 0.5).to(tl.int64)          # RoundHalfIntegerUp
        ny = tl.floor(cy + 0.5).to(tl.int64)
        nz = tl.floor(cz + 0.5).to(tl.int64)
        val = tl.load(src_ptr + (nz * isy + ny) * isx + nx, mask=inside, other=0.0)
    elif MODE == 1:
        bx = tl.minimum(tl.maximum(tl.floor(cx), 0.0), fx - 1.0)
        by = tl.minimum(tl.maximum(tl.floor(cy), 0.0), fy - 1.0)
        bz = tl.minimum(tl.maximum(tl.floor(cz), 0.0), fz - 1.0)
        dx = tl.minimum(tl.maximum(cx - bx, 0.0), 1.0)
        dy = tl.minimum(tl.maximum(cy - by, 0.0), 1.0)
        dz = tl.minimum(tl.maximum(cz - bz, 0.0), 1.0)
        x0 = bx.to(tl.int64); y0 = by.to(tl.int64); z0 = bz.to(tl.int64)
        x1 = tl.minimum(x0 + 1, isx - 1); y1 = tl.minimum(y0 + 1, isy - 1); z1 = tl.minimum(z0 + 1, isz - 1)
        val = tl.zeros_like(cx)
        for corner in tl.static_range(8):
            xx = x1 if (corner & 1) else x0
            yy = y1 if (corner & 2) else y0
            zz = z1 if (corner & 4) else z0
            w = (dx if (corner & 1) else 1.0 - dx) * (dy if (corner & 2) else 1.0 - dy) * (dz if (corner & 4) else 1.0 - dz)
            val += w * tl.load(src_ptr + (zz * isy + yy) * isx + xx, mask=inside, other=0.0).to(tl.float64)
    else:
        # BSplineInterpolateImageFunction, order 3: support start = floor(float(x)) - 1, weights from w = x - (start+1),
        # mirror boundary conditions about [0, size-1].
        sx0 = tl.floor(cx.to(tl.float32)).to(tl.int64) - 1
        sy0 = tl.floor(cy.to(tl.float32)).to(tl.int64) - 1
        sz0 = tl.floor(cz.to(tl.float32)).to(tl.int64) - 1
        wx = cx - (sx0 + 1).to(tl.float64)
        wy = cy - (sy0 + 1).to(tl.float64)
        wz = cz - (sz0 + 1).to(tl.float64)
        val = tl.zeros_like(cx)
        for kz in tl.static_range(4):
            if kz == 3:
                az = (1.0 / 6.0) * wz * wz * wz
            elif kz == 0:
                az = (1.0 / 6.0) + 0.5 * wz * (wz - 1.0) - (1.0 / 6.0) * wz * wz * wz
            elif kz == 2:
                t3 = (1.0 / 6.0) * wz * wz * wz
                az = wz + ((1.0 / 6.0) + 0.5 * wz * (wz - 1.0) - t3) - 2.0 * t3
            else:
                t3 = (1.0 / 6.0) * wz * wz * wz
                t0 = (1.0 / 6.0) + 0.5 * wz * (wz - 1.0) - t3
                az = 1.0 - t0 - (wz + t0 - 2.0 * t3) - t3
            iz = sz0 + kz
            iz = tl.where(iz < 0, -iz, iz)
            iz = tl.where(iz >= isz - 1, (isz - 1) - (iz - (isz - 1)), iz)
            iz = tl.where(isz == 1, 0, iz)
            for ky in tl.static_range(4):
                if ky == 3:
                    ay = (1.0 / 6.0) * wy * wy * wy
                elif ky == 0:
                    ay = (1.0 / 6.0) + 0.5 * wy * (wy - 1.0) - (1.0 / 6.0) * wy * wy * wy
                elif ky == 2:
                    s3 = (1.0 / 6.0) * wy * wy * wy
                    ay = wy + ((1.0 / 6.0) + 0.5 * wy * (wy - 1.0) - s3) - 2.0 * s3
                else:
                    s3 = (1.0 / 6.0) * wy * wy * wy
                    s0 = (1.0 / 6.0) + 0.5 * wy * (wy - 1.0) - s3
                    ay = 1.0 - s0 - (wy + s0 - 2.0 * s3) - s3
                iy = sy0 + ky
                iy = tl.where(iy < 0, -iy, iy)
                iy = tl.where(iy >= isy - 1, (isy - 1) - (iy - (isy - 1)), iy)
                iy = tl.where(isy == 1, 0, iy)
                row = (iz * isy + iy) * isx
                for kx in tl.static_range(4):
                    if kx == 3:
                        ax_ = (1.0 / 6.0) * wx * wx * wx
                    elif kx == 0:
                        ax_ = (1.0 / 6.0) + 0.5 * wx * (wx - 1.0) - (1.0 / 6.0) * wx * wx * wx
                    elif kx == 2:
                        r3 = (1.0 / 6.0) * wx * wx * wx
                        ax_ = wx + ((1.0 / 6.0) + 0.5 * wx * (wx - 1.0) - r3) - 2.0 * r3
                    else:
                        r3 = (1.0 / 6.0) * wx * wx * wx
                        r0 = (1.0 / 6.0) + 0.5 * wx * (wx - 1.0) - r3
                        ax_ = 1.0 - r0 - (wx + r0 - 2.0 * r3) - r3
                    ix = sx0 + kx
                    ix = tl.where(ix < 0, -ix, ix)
                    ix = tl.where(ix >= isx - 1, (isx - 1) - (ix - (isx - 1)), ix)
                    ix = tl.where(isx == 1, 0, ix)
                    val += (az * ay * ax_) * tl.load(src_ptr + row + ix, mask=inside, other=0.0)
    val = tl.where(inside, val, default)
    tl.store(out_ptr + offs, val.to(out_ptr.dtype.element_ty), mask=m)


_LAUNCH_LOCK = __import__("threading").Lock()


def launch_resample(data, out, osx, osy, osz, Ab, default, isx, isy, isz, mode, block=256):
    """Launch the resample kernel. Called from many exam threads; Triton compiles a new specialisation on the first
    launch of each (dtype, MODE, integer-argument) combination and does not guarantee that path is thread-safe, so
    launches are serialised (a few per exam; the kernel itself runs asynchronously on the GPU)."""
    n = osx * osy * osz
    with _LAUNCH_LOCK:
        _resample_kernel[(triton.cdiv(n, block),)](data, out, n, osx, osy, Ab, float(default), isx, isy, isz,
                                                   MODE=mode, BLOCK=block)


def self_test(device="cuda:0"):
    """Compile and run the resample kernel on a tiny volume (identity, linear); raises if Triton is unusable."""
    src = torch.arange(64, dtype=torch.float64, device=device).reshape(4, 4, 4)
    out = torch.empty((4, 4, 4), dtype=torch.float32, device=device)
    Ab = torch.tensor([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0], dtype=torch.float64, device=device)
    launch_resample(src, out, 4, 4, 4, Ab, 0.0, 4, 4, 4, 1)
    if not torch.allclose(out.double(), src):
        raise RuntimeError("Triton resample self-test produced wrong values")

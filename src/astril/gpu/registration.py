"""Drop-in GPU replacement for the subset of sitk.ImageRegistrationMethod that astril uses (milestone 2).

Everything around the optimizer stays ITK: pyramid smoothing (SmoothingRecursiveGaussian), virtual-domain shrinking
(Shrink), and the moving-image gradient (GradientRecursiveGaussian) are computed with SimpleITK filters, exactly as
ImageRegistrationMethodv4 does per level. The GPU replaces only the per-iteration metric value/derivative (ITK v4
Mattes MI / Correlation, ported in gpu_metrics.py and validated there), and the optimizer loop is a line-by-line port
of RegularStepGradientDescentOptimizerv4 with RegistrationParameterScalesFromPhysicalShift scales.

Settings the GPU path does not implement (non-linear interpolators, other optimizers, learning-rate estimation) are
replayed on a real sitk.ImageRegistrationMethod instead (``fallback``), so every option astril uses keeps working.

Known, intentional difference: RANDOM/REGULAR sampling uses the same counts, voxel selection rule and jitter as ITK,
but numpy's RNG instead of ITK's MersenneTwister stream, so sample sets differ from ITK run to run (as ITK's own
wall-clock-seeded runs already do). NONE sampling is deterministic and matches ITK.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import SimpleITK as sitk
import torch

from .metrics import GpuImage, TorchTransform, HIST_COPIES
from . import filters as gf


class Geom:
    """Geometry-only stand-in for sitk.Image (GetSize/GetOrigin/GetSpacing/GetDirection) for virtual domains."""

    def __init__(self, size, origin, spacing, direction):
        self._s, self._o, self._sp, self._d = tuple(int(v) for v in size), tuple(origin), tuple(spacing), tuple(direction)

    @classmethod
    def of(cls, img):
        return cls(img.GetSize(), img.GetOrigin(), img.GetSpacing(), img.GetDirection())

    def GetSize(self): return self._s
    def GetOrigin(self): return self._o
    def GetSpacing(self): return self._sp
    def GetDirection(self): return self._d

    def to_sitk(self):
        img = sitk.Image(list(self._s), sitk.sitkUInt8)
        img.SetOrigin(self._o); img.SetSpacing(self._sp); img.SetDirection(self._d)
        return img


def _gpu_like(g: GpuImage, data: torch.Tensor) -> GpuImage:
    """Same geometry as g, new [C, Z, Y, X] data."""
    return GpuImage(data=data, size_xyz=g.size_xyz, index_to_phys=g.index_to_phys, origin=g.origin,
                    phys_to_index=g.phys_to_index)


def _spacing_direction(g: GpuImage):
    m = g.index_to_phys.cpu().numpy()
    sp = np.linalg.norm(m, axis=0)
    return sp, m / sp

_IRM = sitk.ImageRegistrationMethod
NONE, REGULAR, RANDOM = _IRM.NONE, _IRM.REGULAR, _IRM.RANDOM
MATTES_OUTSIDE_MSG = ("All samples map outside moving image buffer. The images do not sufficiently overlap. They need "
                      "to be initialized to have more overlap before this metric will work. For instance, you can "
                      "align the image centers by translation.")


# ------------------------------------------------------------------------------------------- per-level metric
class LevelMetric:
    """ITK v4 metric for one pyramid level: virtual points (dense shrunk grid or sampled points) -> value, derivative.

    fixed / moving: the level's (smoothed) images. virtual_points: [N,3] physical points (float64 numpy).
    dense: True for NONE sampling (Mattes intensity range from the whole fixed buffer), False for sampled point sets
    (range from the fixed pixel nearest each sample point).
    """

    def __init__(self, fixed, moving, virtual_points, dense, metric, bins, fixed_mask, moving_mask, device,
                 range_mask="same", moving_grad=None):
        # fixed / moving / masks: sitk.Image or already-uploaded GpuImage. moving_grad: precomputed gradient image
        # (GPU GradientRecursiveGaussian); computed here when absent.
        # range_mask: fixed mask used for the Mattes fixed intensity range ("same" = fixed_mask). Lets the caller
        # pre-filter virtual points (fixed_mask=None) while keeping the masked range computation of ITK.
        self.device = torch.device(device)
        self.metric, self.bins = metric, int(bins)
        up = lambda im: im if (im is None or isinstance(im, GpuImage)) else GpuImage.from_sitk(sitk.Cast(im, sitk.sitkFloat32), self.device)
        self.fixed_g = up(fixed)
        self.moving_g = up(moving)
        if moving_grad is None:
            sp, d = _spacing_direction(self.moving_g)
            moving_grad = _gpu_like(self.moving_g, gf.gradient_recursive_gaussian(self.moving_g.data[0], sp, d, float(sp.max())))
        mov_grad = moving_grad
        self.fmask = up(fixed_mask)
        self.mmask = up(moving_mask)

        pts = torch.as_tensor(np.ascontiguousarray(virtual_points, dtype=np.float64), device=self.device)
        # TransformAndEvaluateFixedPoint: fixed mask, fixed IsInsideBuffer, linear fixed interpolation. These do not
        # depend on the moving transform, so invalid fixed points are dropped once here.
        keep = torch.ones(pts.shape[0], dtype=torch.bool, device=self.device)
        if self.fmask is not None:
            keep &= self.fmask.mask_inside(pts)
        fc = self.fixed_g.cont_index(pts)
        keep &= self.fixed_g.inside_buffer(fc)
        self.n_domain = int(pts.shape[0])
        pts, fc = pts[keep], fc[keep]
        self.vptsT = pts.T.contiguous()
        self.vf = _linear_f64(self.fixed_g, fc)   # fixed values are per-level constants: exact double interpolation
        self.n_virtual = int(self.vf.numel())
        self.mov4 = torch.cat([self.moving_g.data, mov_grad.data], dim=0)
        self._p2i = self.moving_g.phys_to_index.cpu().numpy()
        self._morigin = self.moving_g.origin.cpu().numpy()
        self._msize = self.moving_g.size_xyz.cpu().numpy()
        self._norm = 2.0 / (self._msize - 1.0)
        if metric == "mattes":
            rmask = fixed_mask if isinstance(range_mask, str) else range_mask
            self._range_fmask = up(rmask)
            self._init_mattes(virtual_points, dense)

    def _init_mattes(self, virtual_points, dense):
        fg, mg = self.fixed_g, self.moving_g
        if dense:
            fvals = fg.data[0].reshape(-1) if self._range_fmask is None else self._masked_buffer_values(fg, self._range_fmask)
        else:
            # Sparse virtual point set: nearest fixed pixel (TransformPhysicalPointToIndex) when inside the buffer.
            pts = torch.as_tensor(np.ascontiguousarray(virtual_points, dtype=np.float64), device=self.device)
            idx = torch.floor(fg.cont_index(pts) + 0.5)
            inside = ((idx >= 0) & (idx < fg.size_xyz)).all(dim=1)
            ii = idx[inside].to(torch.int64)
            fvals = fg.data[0][ii[:, 2], ii[:, 1], ii[:, 0]]
        mvals = mg.data[0].reshape(-1) if self.mmask is None else self._masked_buffer_values(mg, self.mmask)
        self.f_min, self.f_max = float(fvals.min()), float(fvals.max())
        self.m_min, self.m_max = float(mvals.min()), float(mvals.max())
        pad, B = 2, self.bins
        self.f_bin = (self.f_max - self.f_min) / (B - 2 * pad)
        self.f_nmin = self.f_min / self.f_bin - pad
        self.m_bin = (self.m_max - self.m_min) / (B - 2 * pad)
        self.m_nmin = self.m_min / self.m_bin - pad
        fterm = self.vf / self.f_bin - self.f_nmin
        copy = (torch.arange(self.n_virtual, device=self.device) % HIST_COPIES) * (B * B)
        self.vfbin_off = torch.clamp(fterm.to(torch.int64), 2, B - 3) * B - 1 + copy

    def _masked_buffer_values(self, g, gmask):
        n = int(g.data[0].numel())
        keep = torch.zeros(n, dtype=torch.bool, device=self.device)
        flat = torch.arange(n, device=self.device)
        for s in range(0, n, 2_000_000):
            keep[s:s + 2_000_000] = gmask.mask_inside(g.voxel_points(flat[s:s + 2_000_000]))
        return g.data[0].reshape(-1)[keep].to(torch.float64)

    # -- shared evaluation pieces (see gpu_metrics.GpuMetric fast path for the derivation) --
    @staticmethod
    def _affine3(pT, A, b):
        return torch.stack([pT[0] * float(A[i, 0]) + pT[1] * float(A[i, 1]) + pT[2] * float(A[i, 2]) + float(b[i])
                            for i in range(3)])

    def _sample_moving(self, M, off):
        cidxT = self._affine3(self.vptsT, self._p2i @ M, self._p2i @ (off - self._morigin))
        s = self._msize
        valid = (cidxT[0] >= -0.5) & (cidxT[0] < s[0] - 0.5) & (cidxT[1] >= -0.5) & (cidxT[1] < s[1] - 0.5) \
            & (cidxT[2] >= -0.5) & (cidxT[2] < s[2] - 0.5)
        if self.mmask is not None:
            valid &= self.mmask.mask_inside(self._affine3(self.vptsT, M, off).T)
        grid = torch.stack([cidxT[i] * float(self._norm[i]) - 1.0 for i in range(3)], dim=1).to(torch.float32)
        n = grid.shape[0]
        out = torch.nn.functional.grid_sample(self.mov4[None], grid.view(1, 1, 1, n, 3), mode="bilinear",
                                              padding_mode="border", align_corners=True).view(4, n)
        return valid, out[0], out[1:]

    def _weighted_sums(self, wgT):
        wg64 = wgT.to(torch.float64)
        return torch.cat([wg64.sum(dim=1), torch.stack([(self.vptsT * wg64[i]).sum(dim=1) for i in range(3)]).reshape(-1)])

    @staticmethod
    def _contract(tfm, hs):
        h, S = hs[:3], hs[3:12].reshape(3, 3)
        _, _, dm, dt = tfm.host_matrix_offset_jacobian()
        G = S - np.outer(h, tfm._center_np)
        return np.einsum("ijp,ij->p", dm, G) + np.einsum("ip,i->p", dt, h)

    def value_and_derivative(self, tfm: TorchTransform):
        """Returns (value, ITK-v4 derivative (negated gradient), number of valid points)."""
        if self.n_virtual == 0:
            if self.metric == "mattes":
                raise RuntimeError(MATTES_OUTSIDE_MSG)
            return float(np.finfo(np.float64).max), np.zeros(tfm.n_params), 0
        M, off, _, _ = tfm.host_matrix_offset_jacobian()
        valid, mval, gT = self._sample_moving(M, off)
        return self._mattes(tfm, valid, mval, gT) if self.metric == "mattes" else self._correlation(tfm, valid, mval, gT)

    def _mattes(self, tfm, valid, mval, gT):
        B, dev = self.bins, self.device
        valid = valid & (mval >= self.m_min) & (mval <= self.m_max)
        w = valid.to(torch.float32)
        mterm = torch.where(valid, mval * float(1.0 / self.m_bin) - float(self.m_nmin), torch.full_like(mval, 2.0))
        mbin = torch.clamp(mterm.to(torch.int64), 2, B - 3)
        u = mterm - mbin.to(torch.float32)
        um, u2 = 1.0 - u, u * u
        u3 = u2 * u
        wts = torch.stack([um * um * um / 6.0, (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
                           (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0, u3 / 6.0]) * w
        idx = (self.vfbin_off + mbin)[None, :] + torch.arange(4, device=dev)[:, None]
        joint = torch.zeros(HIST_COPIES * B * B, dtype=torch.float64, device=dev)
        joint.index_add_(0, idx.reshape(-1), wts.reshape(-1).to(torch.float64))
        joint = joint.view(HIST_COPIES, B * B).sum(dim=0)
        idx = idx % (B * B)
        jsum = joint.sum()
        jn = (joint / jsum).view(B, B)
        fmarg, mmarg = jn.sum(dim=1), jn.sum(dim=0)
        eps = np.finfo(np.float64).eps
        ok = (fmarg[:, None] > eps) & (mmarg[None, :] > eps) & (jn > eps)
        one = torch.ones_like(jn)
        pratio = torch.where(ok, torch.log(torch.where(ok, jn, one) / torch.where(ok, mmarg[None, :].expand_as(jn), one)),
                             torch.zeros_like(jn))
        mi = torch.where(ok, jn * (pratio - torch.log(torch.where(ok, fmarg[:, None].expand_as(jn), one))),
                         torch.zeros_like(jn)).sum()
        dw = torch.stack([0.5 * um * um, 2.0 * u - 1.5 * u2, -2.0 * um + 1.5 * um * um, -0.5 * u2])
        wn = (pratio.reshape(-1).to(torch.float32)[idx] * dw).sum(dim=0) * w
        packed = torch.cat([torch.stack([mi, w.sum(dtype=torch.float64), jsum]), self._weighted_sums(gT * wn)]).cpu().numpy()
        n_valid = int(round(packed[1]))
        if n_valid == 0:
            raise RuntimeError(MATTES_OUTSIDE_MSG)
        if packed[2] < eps:
            raise RuntimeError("Joint PDF summed to zero")
        deriv = self._contract(tfm, packed[3:]) * (-1.0 / (self.m_bin * n_valid))
        return -float(packed[0]), deriv, n_valid

    def _correlation(self, tfm, valid, mval, gT):
        w = valid.to(torch.float64)
        f, m = self.vf * w, mval.to(torch.float64) * w
        mom = torch.stack([w.sum(), f.sum(), m.sum(), (f * f).sum(), (m * m).sum(), (f * m).sum()])
        packed = torch.cat([mom, self._weighted_sums(gT * w.to(torch.float32)), self._weighted_sums(gT * f.to(torch.float32)),
                            self._weighted_sums(gT * m.to(torch.float32))]).cpu().numpy()
        n, sf, sm, sff, smm, sfm = packed[:6]
        n_valid, P = int(round(n)), tfm.n_params
        if n_valid <= 0:
            return float(np.finfo(np.float64).max), np.zeros(P), 0       # VerifyNumberOfValidPoints
        af, am = sf / n, sm / n
        fm, f2, m2 = sfm - n * af * am, sff - n * af * af, smm - n * am * am
        if m2 * f2 <= np.finfo(np.float64).eps:
            return float(np.finfo(np.float64).max), np.zeros(P), n_valid
        v1, vf, vm = self._contract(tfm, packed[6:18]), self._contract(tfm, packed[18:30]), self._contract(tfm, packed[30:42])
        fdm, mdm = vf - af * v1, vm - am * v1
        return -1.0 * fm * fm / (m2 * f2), 2.0 * fm / (f2 * m2) * (fdm - fm / m2 * mdm), n_valid


def _rows_times_mt(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    """pts @ m.T for [N,3] points without BLAS. numpy's OpenBLAS (MAX_THREADS=24) crashes/deadlocks when many Python
    threads make large BLAS calls at once on this 32-thread machine, so point transforms use elementwise math."""
    return np.stack([pts[:, 0] * m[i, 0] + pts[:, 1] * m[i, 1] + pts[:, 2] * m[i, 2] for i in range(3)], axis=1)


def _linear_f64(g: GpuImage, cidx: torch.Tensor) -> torch.Tensor:
    """LinearInterpolateImageFunction in float64 at continuous indices [N,3] (x,y,z), edge-clamped like ITK."""
    data = g.data[0].to(torch.float64)
    hi = (g.size_xyz - 1).to(torch.int64)
    base = torch.minimum(torch.clamp(torch.floor(cidx).to(torch.int64), min=0), hi)
    d = torch.clamp(cidx - base.to(torch.float64), min=0.0, max=1.0)
    nxt = torch.minimum(base + 1, hi)
    out = torch.zeros(cidx.shape[0], dtype=torch.float64, device=cidx.device)
    for cx in (0, 1):
        ix = nxt[:, 0] if cx else base[:, 0]
        wx = d[:, 0] if cx else 1.0 - d[:, 0]
        for cy in (0, 1):
            iy = nxt[:, 1] if cy else base[:, 1]
            wy = d[:, 1] if cy else 1.0 - d[:, 1]
            for cz in (0, 1):
                iz = nxt[:, 2] if cz else base[:, 2]
                wz = d[:, 2] if cz else 1.0 - d[:, 2]
                out += data[iz, iy, ix] * (wx * wy * wz)
    return out


def _mask_on_grid_itk(mask: sitk.Image, grid: sitk.Image) -> np.ndarray:
    """Fixed-mask membership of every voxel of `grid`, decided by ITK's own index arithmetic.

    Shrunk virtual-domain points fall exactly halfway between fixed voxels, so nearest-voxel mask lookups are ties
    that floating-point rounding decides. Resampling the mask (nearest neighbour) through a zero BSpline transform
    forces ResampleImageFilter's per-pixel path (TransformIndexToPhysicalPoint -> TransformPhysicalPointToContinuousIndex
    -> RoundHalfIntegerUp), i.e. the same arithmetic ImageMaskSpatialObject uses. Returns a raster-order bool array.
    """
    zero = sitk.BSplineTransformInitializer(grid, [1, 1, 1])
    res = sitk.Resample(sitk.Cast(mask, sitk.sitkUInt8), grid, zero, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayViewFromImage(res).reshape(-1) != 0


# ---------------------------------------------------------------------------------- virtual domain sampling
def _grid_points(img: sitk.Image) -> np.ndarray:
    """Physical points of every voxel of img, raster order (x fastest), [N,3] float64."""
    sx, sy, sz = img.GetSize()
    k, j, i = np.meshgrid(np.arange(sz), np.arange(sy), np.arange(sx), indexing="ij")
    ijk = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(np.float64)
    m = np.asarray(img.GetDirection()).reshape(3, 3) @ np.diag(img.GetSpacing())
    return np.asarray(img.GetOrigin()) + _rows_times_mt(ijk, m)


def _mask_inside_np(mask: sitk.Image, pts: np.ndarray) -> np.ndarray:
    arr = sitk.GetArrayViewFromImage(mask)
    m = np.asarray(mask.GetDirection()).reshape(3, 3) @ np.diag(mask.GetSpacing())
    idx = np.floor(_rows_times_mt(pts - np.asarray(mask.GetOrigin()), np.linalg.inv(m)) + 0.5)
    ok = np.all((idx >= 0) & (idx < np.asarray(mask.GetSize())), axis=1)
    out = np.zeros(len(pts), dtype=bool)
    ii = idx[ok].astype(np.int64)
    out[ok] = arr[ii[:, 2], ii[:, 1], ii[:, 0]] != 0
    return out


def sample_virtual_domain(virtual: sitk.Image, strategy: int, percentage: float, rng: np.random.Generator,
                          fixed_mask: sitk.Image | None) -> np.ndarray:
    """ImageRegistrationMethodv4 metric sampling (REGULAR / RANDOM) with N(0, spacing/3) jitter and fixed-mask filter."""
    pts = _grid_points(virtual)
    sigma = np.asarray(virtual.GetSpacing()) / 3.0
    if strategy == REGULAR:
        step = int(np.ceil(1.0 / percentage))
        sel = pts[::step]
    elif strategy == RANDOM:
        count = int(np.float32(len(pts)) * np.float32(percentage))
        sel = pts[rng.integers(0, len(pts), size=count)]
    else:
        raise ValueError("invalid sampling strategy")
    sel = sel + rng.standard_normal(sel.shape) * sigma
    if fixed_mask is not None:
        sel = sel[_mask_inside_np(fixed_mask, sel)]
    return sel


# -------------------------------------------------------------------------------------------- optimizer bits
def physical_shift_scales(tfm: TorchTransform, virtual: sitk.Image, small_variation: float) -> np.ndarray:
    """RegistrationParameterScalesFromPhysicalShift.EstimateScales (corner sampling for matrix-offset transforms)."""
    P = tfm.n_params
    if tfm.kind == "translation":
        return np.ones(P)     # every sample shifts by exactly the parameter change -> (d/d)^2 = 1
    size = np.asarray(virtual.GetSize())
    corners = np.array([[(c >> d) & 1 for d in range(3)] for c in range(8)], dtype=np.float64) * (size - 1)
    m = np.asarray(virtual.GetDirection()).reshape(3, 3) @ np.diag(virtual.GetSpacing())
    pts = np.asarray(virtual.GetOrigin()) + corners @ m.T
    p0 = tfm._params_np.copy()
    M0, off0, _, _ = tfm.host_matrix_offset_jacobian()
    base = pts @ M0.T + off0
    shifts = np.zeros(P)
    for i in range(P):
        d = np.zeros(P); d[i] = small_variation
        tfm.set_params(p0 + d)
        M1, off1, _, _ = tfm.host_matrix_offset_jacobian()
        shifts[i] = np.max(np.linalg.norm(pts @ M1.T + off1 - base, axis=1))
    tfm.set_params(p0)
    eps = np.finfo(np.float64).eps
    nz = shifts[shifts > eps]
    if nz.size == 0:
        warnings.warn("Variation in any parameter won't change a voxel position; using scales of 1.0")
        return np.ones(P)
    scales = np.where(shifts <= eps, nz.min() ** 2, shifts ** 2)
    return scales / small_variation ** 2


# ---------------------------------------------------------------------------------------- the drop-in class
class GpuImageRegistrationMethod:
    """sitk.ImageRegistrationMethod look-alike (astril's subset). See module docstring."""

    NONE, REGULAR, RANDOM = NONE, REGULAR, RANDOM
    Never, Once, EachIteration = _IRM.Never, _IRM.Once, _IRM.EachIteration

    def __init__(self, device: str = "cuda:0", fallback: bool = True, verbose_fallback: bool = False,
                 use_service: bool = True, spill_to_cpu: bool = False, spill_queue: int | None = None):
        """use_service: run levels through the shared batched GpuRegistrationService (one per GPU).
        spill_to_cpu: when the service queue holds >= spill_queue levels (default: the service's job cap), run this
        registration on ITK (CPU) instead of waiting. Off by default: a queued GPU registration waits ~seconds, while
        an ITK one occupies a core for 10-40 s that the CPU-bound rest of the pipeline needs (measured: no throughput
        gain). If another process owns this GPU, ITK is used regardless."""
        self.device, self.allow_fallback, self.verbose_fallback = device, fallback, verbose_fallback
        self.use_service, self.spill_to_cpu, self.spill_queue = use_service, spill_to_cpu, spill_queue
        self.backend_used = None
        self._calls = []                         # recorded setter calls, replayed on ITK if we fall back
        self._unsupported = []
        self._commands = []                      # (event, callable)
        self._delegate = None                    # real sitk IRM after a fallback Execute
        self.metric, self.bins = "mattes", 50
        self.fixed_mask = self.moving_mask = None
        self.shrink, self.sigmas, self.sigmas_physical = [1], [0.0], True
        self.sampling, self.percentages, self.seed = NONE, [1.0], 0
        self.interpolator = sitk.sitkLinear
        self.opt = None
        self.scales_mode, self.scales, self.small_variation = None, None, 0.01
        self.initial, self.in_place = None, True
        # state exposed through getters
        self._value, self._iteration, self._level, self._position = 0.0, 0, 0, ()
        self._stop, self._lr, self._nvalid, self._level_scales = "", 0.0, 0, ()
        self.timing = {}

    # ---- recording helpers ----
    def _rec(self, name, *a, **k):
        self._calls.append((name, a, k))

    def _need_fallback(self, why):
        self._unsupported.append(why)

    # ---- metric ----
    def SetMetricAsMattesMutualInformation(self, numberOfHistogramBins=50):
        self._rec("SetMetricAsMattesMutualInformation", numberOfHistogramBins=numberOfHistogramBins)
        self.metric, self.bins = "mattes", int(numberOfHistogramBins)

    def SetMetricAsCorrelation(self):
        self._rec("SetMetricAsCorrelation")
        self.metric = "correlation"

    def SetMetricFixedMask(self, mask):
        self._rec("SetMetricFixedMask", mask); self.fixed_mask = mask

    def SetMetricMovingMask(self, mask):
        self._rec("SetMetricMovingMask", mask); self.moving_mask = mask

    def SetMetricSamplingStrategy(self, strategy):
        self._rec("SetMetricSamplingStrategy", strategy); self.sampling = int(strategy)

    def SetMetricSamplingPercentage(self, percentage, seed=sitk.sitkWallClock):
        self._rec("SetMetricSamplingPercentage", percentage, seed)
        self.percentages, self.seed = [float(percentage)], int(seed)

    def SetMetricSamplingPercentagePerLevel(self, percentages, seed=sitk.sitkWallClock):
        self._rec("SetMetricSamplingPercentagePerLevel", list(percentages), seed)
        self.percentages, self.seed = [float(p) for p in percentages], int(seed)

    # ---- pyramid ----
    def SetShrinkFactorsPerLevel(self, f):
        self._rec("SetShrinkFactorsPerLevel", list(f)); self.shrink = [int(x) for x in f]

    def SetSmoothingSigmasPerLevel(self, s):
        self._rec("SetSmoothingSigmasPerLevel", list(s)); self.sigmas = [float(x) for x in s]

    def SmoothingSigmasAreSpecifiedInPhysicalUnitsOn(self):
        self._rec("SmoothingSigmasAreSpecifiedInPhysicalUnitsOn"); self.sigmas_physical = True

    def SmoothingSigmasAreSpecifiedInPhysicalUnitsOff(self):
        self._rec("SmoothingSigmasAreSpecifiedInPhysicalUnitsOff"); self.sigmas_physical = False

    def SetSmoothingSigmasAreSpecifiedInPhysicalUnits(self, v):
        self._rec("SetSmoothingSigmasAreSpecifiedInPhysicalUnits", v); self.sigmas_physical = bool(v)

    # ---- interpolator / optimizer / transform ----
    def SetInterpolator(self, interp):
        self._rec("SetInterpolator", interp); self.interpolator = interp

    def SetOptimizerAsRegularStepGradientDescent(self, learningRate, minStep, numberOfIterations, relaxationFactor=0.5,
                                                 gradientMagnitudeTolerance=1e-4, estimateLearningRate=_IRM.Never,
                                                 maximumStepSizeInPhysicalUnits=0.0):
        self._rec("SetOptimizerAsRegularStepGradientDescent", learningRate, minStep, numberOfIterations,
                  relaxationFactor, gradientMagnitudeTolerance, estimateLearningRate, maximumStepSizeInPhysicalUnits)
        self.opt = dict(lr=float(learningRate), min_step=float(minStep), iters=int(numberOfIterations),
                        relax=float(relaxationFactor), grad_tol=float(gradientMagnitudeTolerance),
                        estimate=int(estimateLearningRate))

    def SetOptimizerScalesFromPhysicalShift(self, centralRegionRadius=5, smallParameterVariation=0.01):
        self._rec("SetOptimizerScalesFromPhysicalShift", centralRegionRadius, smallParameterVariation)
        self.scales_mode, self.small_variation = "physical", float(smallParameterVariation)

    def SetOptimizerScales(self, scales):
        self._rec("SetOptimizerScales", list(scales))
        self.scales_mode, self.scales = "manual", np.asarray(scales, dtype=np.float64)

    def SetInitialTransform(self, transform, inPlace=True):
        self._rec("SetInitialTransform", transform, inPlace)
        self.initial, self.in_place = transform, bool(inPlace)

    def SetNumberOfThreads(self, n):
        self._rec("SetNumberOfThreads", n)

    def GetNumberOfThreads(self):
        return 1

    def AddCommand(self, event, cmd):
        self._commands.append((event, cmd))
        return len(self._commands) - 1

    def RemoveAllCommands(self):
        self._commands.clear()

    def __getattr__(self, name):
        # Any other sitk setter we do not implement: record it and fall back to ITK at Execute time.
        if name.startswith("Set") or name.endswith("On") or name.endswith("Off"):
            def _unsupported_setter(*a, **k):
                self._rec(name, *a, **k)
                self._need_fallback(name)
            return _unsupported_setter
        raise AttributeError(name)

    # ---- getters (delegate after a fallback run) ----
    def GetMetricValue(self):
        return self._delegate.GetMetricValue() if self._delegate else self._value

    def GetOptimizerIteration(self):
        return self._delegate.GetOptimizerIteration() if self._delegate else self._iteration

    def GetOptimizerPosition(self):
        return self._delegate.GetOptimizerPosition() if self._delegate else tuple(self._position)

    def GetCurrentLevel(self):
        return self._delegate.GetCurrentLevel() if self._delegate else self._level

    def GetOptimizerStopConditionDescription(self):
        return self._delegate.GetOptimizerStopConditionDescription() if self._delegate else self._stop

    def GetOptimizerLearningRate(self):
        return self._delegate.GetOptimizerLearningRate() if self._delegate else self._lr

    def GetOptimizerScales(self):
        return self._delegate.GetOptimizerScales() if self._delegate else tuple(self._level_scales)

    def GetMetricNumberOfValidPoints(self):
        return self._delegate.GetMetricNumberOfValidPoints() if self._delegate else self._nvalid

    # ---- execution ----
    def _fire(self, event):
        for ev, cmd in self._commands:
            if ev == event or ev == sitk.sitkAnyEvent:
                cmd()

    def _fallback_reasons(self):
        why = list(self._unsupported)
        if self.interpolator != sitk.sitkLinear:
            why.append(f"interpolator {self.interpolator}")
        if self.opt is None:
            why.append("optimizer not RegularStepGradientDescent")
        elif self.opt["estimate"] != _IRM.Never:
            why.append("learning-rate estimation")
        if not torch.cuda.is_available() and str(self.device).startswith("cuda"):
            why.append("no CUDA device")
        if self.initial is not None:
            try:
                TorchTransform.from_sitk(self.initial, "cpu")
            except NotImplementedError as e:
                why.append(str(e))
        if len(self.shrink) != len(self.sigmas):
            why.append("shrink/sigma level mismatch")
        return why

    def _execute_itk(self, fixed, moving):
        r = _IRM()
        for name, a, k in self._calls:
            getattr(r, name)(*a, **k)
        for ev, cmd in self._commands:
            r.AddCommand(ev, cmd)
        self._delegate = r
        return r.Execute(fixed, moving)

    def Execute(self, fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
        self._delegate = None
        why = self._fallback_reasons()
        self._service = None
        if not why and self.use_service:
            from .service import get_service
            svc = get_service(self.device)
            if svc is None:
                why.append("GPU owned by another process")
            elif self.spill_to_cpu and svc.queue_length() >= (self.spill_queue if self.spill_queue is not None else svc.max_jobs):
                why.append("GPU queue full (spill to CPU)")
            else:
                self._service = svc
        self.backend_used = "itk" if why else ("gpu-service" if self._service else "gpu")
        if why:
            if not self.allow_fallback:
                raise NotImplementedError("GPU registration does not support: " + "; ".join(why))
            if self.verbose_fallback:
                print(f"[gpu_registration] falling back to ITK: {'; '.join(why)}")
            return self._execute_itk(fixed, moving)
        try:
            return self._execute_gpu(fixed, moving)
        except Exception as e:  # noqa: BLE001
            from .memory import is_oom, release_cache
            if is_oom(e):
                release_cache()
            # Any GPU-side failure (e.g. out of GPU memory while HD-BET shares the device) re-runs this registration on
            # ITK. Genuine metric errors (e.g. "All samples map outside moving image buffer") are raised by ITK the
            # same way, so behaviour matches the ITK backend.
            if not self.allow_fallback or "All samples map outside moving image buffer" in str(e):
                raise   # same exception ITK raises (astril's pre-pass catches it by this message)
            from . import _warn_once
            _warn_once(f"gpu-exec-{type(e).__name__}", f"astril GPU registration failed ({type(e).__name__}: {e}); "
                                                       "re-running on ITK. Further failures of this type are silent.")
            self.backend_used = "itk"
            return self._execute_itk(fixed, moving)

    def _level_images(self, fixed, moving, level):
        s = self.sigmas[level]
        if s > 0:
            fs = [s * (1.0 if self.sigmas_physical else sp) for sp in fixed.GetSpacing()]
            ms = [s * (1.0 if self.sigmas_physical else sp) for sp in moving.GetSpacing()]
            fixed = sitk.SmoothingRecursiveGaussian(fixed, fs, False)
            moving = sitk.SmoothingRecursiveGaussian(moving, ms, False)
        return fixed, moving

    def _execute_gpu(self, fixed, moving):
        from . import memory
        memory.prepare(self.device)        # per-process GPU memory cap (set once, before any allocation)
        t_all = time.perf_counter()
        self.timing = dict(setup=0.0, iterate=0.0, levels=[])
        fixed = sitk.Cast(fixed, sitk.sitkFloat32)
        moving = sitk.Cast(moving, sitk.sitkFloat32)
        # Upload once; pyramid levels are computed on the GPU from these.
        fixed_g = GpuImage.from_sitk(fixed, self.device)
        moving_g = GpuImage.from_sitk(moving, self.device)
        fmask_g = GpuImage.from_sitk(sitk.Cast(self.fixed_mask, sitk.sitkFloat32), self.device) if self.fixed_mask is not None else None
        mmask_g = GpuImage.from_sitk(sitk.Cast(self.moving_mask, sitk.sitkFloat32), self.device) if self.moving_mask is not None else None
        fsp, msp = np.asarray(fixed.GetSpacing()), np.asarray(moving.GetSpacing())
        mdir = np.asarray(moving.GetDirection())
        init = self.initial if self.initial is not None else sitk.Transform(3, sitk.sitkIdentity)
        tfm = TorchTransform.from_sitk(init, self.device)
        tfm.set_params(tfm.params.cpu().numpy())
        tfm.host_matrix_offset_jacobian()  # caches center
        rng = np.random.default_rng(None if self.seed == sitk.sitkWallClock else self.seed)
        n_levels = len(self.shrink)
        pcts = self.percentages if len(self.percentages) == n_levels else [self.percentages[0]] * n_levels
        o = self.opt
        self._fire(sitk.sitkStartEvent)
        for level in range(n_levels):
            t0 = time.perf_counter()
            self._level, self._iteration, self._value = level, 0, 0.0
            self._fire(sitk.sitkMultiResolutionIterationEvent)
            virtual = Geom(*gf.shrink_geometry(fixed, self.shrink[level])) if self.shrink[level] > 1 else Geom.of(fixed)
            fmask_for_metric = self.fixed_mask
            if self.sampling == NONE:
                pts, dense = _grid_points(virtual), True
                if self.fixed_mask is not None:
                    pts = pts[_mask_on_grid_itk(self.fixed_mask, virtual.to_sitk())]
                    fmask_for_metric = None   # membership already decided exactly; skip the GPU re-test
            else:
                pts, dense = sample_virtual_domain(virtual, self.sampling, pcts[level], rng, self.fixed_mask), False
            # Level setup (smoothing, gradient image, point upload) draws on the shared per-thread GPU pool; the
            # reservation is released before this level waits on the service.
            with memory.reserve(self.device, memory.level_setup_bytes(len(pts), moving.GetNumberOfPixels())):
                s_l = self.sigmas[level]
                if s_l > 0:   # SmoothingRecursiveGaussian on both images (ImageRegistrationMethodv4 level smoothing)
                    fsig = s_l * (np.ones(3) if self.sigmas_physical else fsp)
                    msig = s_l * (np.ones(3) if self.sigmas_physical else msp)
                    f_l = _gpu_like(fixed_g, gf.smoothing_recursive_gaussian(fixed_g.data[0], fsp, fsig)[None])
                    m_l = _gpu_like(moving_g, gf.smoothing_recursive_gaussian(moving_g.data[0], msp, msig)[None])
                else:
                    f_l, m_l = fixed_g, moving_g
                g_l = _gpu_like(m_l, gf.gradient_recursive_gaussian(m_l.data[0], msp, mdir, float(msp.max())))
                lm = LevelMetric(f_l, m_l, pts, dense, self.metric, self.bins,
                                 fmask_g if fmask_for_metric is not None else None, mmask_g, self.device,
                                 range_mask=fmask_g, moving_grad=g_l)
                del f_l, m_l, g_l
            if self.scales_mode == "physical":
                scales = physical_shift_scales(tfm, virtual, self.small_variation)
            elif self.scales_mode == "manual":
                scales = self.scales
            else:
                scales = np.ones(tfm.n_params)
            self._level_scales = scales
            t1 = time.perf_counter()
            if self._service is not None:
                self._run_level_on_service(lm, tfm, scales, o)
            else:
                self._run_rsgd(lm, tfm, scales, o)
            t2 = time.perf_counter()
            self.timing["levels"].append(dict(setup_s=t1 - t0, iterate_s=t2 - t1, iters=self._iteration,
                                              n_points=lm.n_virtual))
            self.timing["setup"] += t1 - t0
            self.timing["iterate"] += t2 - t1
        self._fire(sitk.sitkEndEvent)
        self.timing["total"] = time.perf_counter() - t_all
        return self._output_transform(init, tfm)

    def _run_level_on_service(self, lm, tfm, scales, o):
        from .service import LevelState

        def on_iteration(value, nvalid, it, params):
            self._value, self._nvalid, self._iteration, self._position = value, nvalid, it, tuple(params)
            self._fire(sitk.sitkIterationEvent)

        state = LevelState(lm=lm, tfm=tfm, scales=np.asarray(scales, dtype=np.float64), opt=o, on_iteration=on_iteration)
        self._lr = o["lr"]
        try:
            self._service.run_level(state)
        finally:
            self._iteration, self._stop = state.it, state.stop
            self._value, self._nvalid = state.value, state.nvalid
            self.timing.setdefault("queue_wait", 0.0)
            self.timing["queue_wait"] += max(0.0, state.t_start - state.t_submit)

    def _run_rsgd(self, lm: LevelMetric, tfm: TorchTransform, scales, o):
        """RegularStepGradientDescentOptimizerv4 (StartOptimization + ResumeOptimization + AdvanceOneStep)."""
        P = tfm.n_params
        grad = np.zeros(P)
        prev = np.zeros(P)
        relax = 1.0
        it = 0
        self._lr = o["lr"]
        stop = "RegularStepGradientDescentOptimizerv4: "
        inv_scales = 1.0 / scales
        while True:
            if it >= o["iters"]:
                stop += f"Maximum number of iterations ({o['iters']}) exceeded."
                break
            prev, grad = grad, prev                       # swap(m_PreviousGradient, m_Gradient)
            value, deriv, nvalid = lm.value_and_derivative(tfm)
            grad = np.asarray(deriv, dtype=np.float64).copy()
            self._value, self._nvalid = value, nvalid
            # ModifyGradientByScales: both current and previous gradient are divided by the scales.
            grad *= inv_scales
            prev = prev * inv_scales
            gmag = float(np.sqrt(np.sum(grad * grad)))
            if gmag < o["grad_tol"]:
                stop += (f"Gradient magnitude tolerance met after {it} iterations. Gradient magnitude ({gmag}) is "
                         f"less than gradient magnitude tolerance ({o['grad_tol']}).")
                it += 1                                    # ITK increments m_CurrentIteration after a stop, too
                break
            if float(np.dot(grad, prev)) < 0:
                relax *= o["relax"]
            step = relax * o["lr"]
            if step < o["min_step"]:
                stop += (f"Step too small after {it} iterations. Current step ({step}) is less than minimum step "
                         f"({o['min_step']}).")
                it += 1
                break
            grad *= step                                   # ModifyGradientByLearningRate (in place, like ITK)
            tfm.set_params(tfm._params_np + grad * (1.0 / gmag))
            self._iteration, self._position = it, tuple(tfm._params_np)
            self._fire(sitk.sitkIterationEvent)
            it += 1
        self._iteration = it
        self._stop = stop

    def _output_transform(self, init: sitk.Transform, tfm: TorchTransform) -> sitk.Transform:
        if self.in_place and self.initial is not None:
            self.initial.SetParameters(tuple(tfm._params_np))
            return self.initial
        inner = init.Downcast()
        if inner.GetName() == "CompositeTransform":
            inner = inner.GetNthTransform(0).Downcast()
        out_inner = sitk.Transform(inner).Downcast()
        out_inner.SetParameters(tuple(tfm._params_np))
        comp = sitk.CompositeTransform(3)
        comp.AddTransform(out_inner)
        return comp

    def MetricEvaluate(self, fixed: sitk.Image, moving: sitk.Image) -> float:
        tfm = TorchTransform.from_sitk(self.initial, self.device)
        tfm.set_params(tfm.params.cpu().numpy())
        pts = _grid_points(fixed)
        lm = LevelMetric(fixed, moving, pts, True, self.metric, self.bins, self.fixed_mask, self.moving_mask, self.device)
        return lm.value_and_derivative(tfm)[0]

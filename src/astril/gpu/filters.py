"""GPU ports of the ITK filters around registration: recursive Gaussian smoothing / gradient (pyramid + metric
gradient image), B-spline (cubic) decomposition and resampling (registration-time downsampling, final resample).

Recursive filters (RecursiveGaussianImageFilter, BSplineDecompositionImageFilter) are linear along each axis with
fixed boundary rules, so each axis pass is an exact linear operator K (n x n). K is built once per (length,
parameters) by running ITK's own 1-D recursion (ported verbatim, float64) on the identity matrix, and applied on the
GPU as a matrix product. Pass order and intermediate precision follow ITK:
  SmoothingRecursiveGaussian: z first, then x, then y; float32 between passes (InternalRealType = float).
  GradientRecursiveGaussian:  per component d: first-order along d, zero-order along the other axes (ascending);
                              float64 intermediates; / spacing[d]; then direction * gradient (UseImageDirection).
  BSplineDecomposition:       x, y, z; float64 coefficients (BSplineInterpolateImageFunction's coefficient type).
Resampling (ResampleImageFilter + BSpline/Linear/NearestNeighbor interpolators) is a Triton kernel (kernels.py).
"""
from __future__ import annotations

import functools
import math

import numpy as np
import SimpleITK as sitk
import torch


# ------------------------------------------------------------------------- recursive Gaussian (Deriche, ITK)
def _gauss_coefficients(sigmad: float, order: int, normalize_across_scale: bool, sigma: float):
    A1 = [1.3530, -0.6724, -1.3563]; B1 = [1.8151, -3.4327, 5.2318]
    A2 = [-0.3531, 0.6724, 0.3446]; B2 = [0.0902, 0.6100, -2.2355]
    W1, L1, W2, L2 = 0.6681, -1.3932, 2.0787, -1.3732

    def ncoef(a1, b1, a2, b2):
        s1, s2, c1, c2 = math.sin(W1 / sigmad), math.sin(W2 / sigmad), math.cos(W1 / sigmad), math.cos(W2 / sigmad)
        e1, e2 = math.exp(L1 / sigmad), math.exp(L2 / sigmad)
        n0 = a1 + a2
        n1 = e2 * (b2 * s2 - (a2 + 2 * a1) * c2)
        n1 += e1 * (b1 * s1 - (a1 + 2 * a2) * c1)
        n2 = (a1 + a2) * c2 * c1
        n2 -= b1 * c2 * s1 + b2 * c1 * s2
        n2 *= 2 * e1 * e2
        n2 += a2 * e1 * e1 + a1 * e2 * e2
        n3 = e2 * e1 * e1 * (b2 * s2 - a2 * c2)
        n3 += e1 * e2 * e2 * (b1 * s1 - a1 * c1)
        return n0, n1, n2, n3, n0 + n1 + n2 + n3, n1 + 2 * n2 + 3 * n3, n1 + 4 * n2 + 9 * n3

    c1, c2 = math.cos(W1 / sigmad), math.cos(W2 / sigmad)
    e1, e2 = math.exp(L1 / sigmad), math.exp(L2 / sigmad)
    d4 = e1 * e1 * e2 * e2
    d3 = -2 * c1 * e1 * e2 * e2
    d3 += -2 * c2 * e2 * e1 * e1
    d2 = 4 * c2 * c1 * e1 * e2
    d2 += e1 * e1 + e2 * e2
    d1 = -2 * (e2 * c2 + e1 * c1)
    SD = 1.0 + d1 + d2 + d3 + d4
    DD = d1 + 2 * d2 + 3 * d3 + 4 * d4
    if order == 0:
        n0, n1, n2, n3, SN, DN, EN = ncoef(A1[0], B1[0], A2[0], B2[0])
        alpha = 2 * SN / SD - n0
        norm = 1.0 / alpha
        symmetric = True
    elif order == 1:
        n0, n1, n2, n3, SN, DN, EN = ncoef(A1[1], B1[1], A2[1], B2[1])
        alpha = 2 * (SN * DD - DN * SD) / (SD * SD)       # * direction (+1 for positive spacing)
        norm = (sigma if normalize_across_scale else 1.0) / alpha
        symmetric = False
    else:
        raise NotImplementedError("order 2 not needed")
    n0, n1, n2, n3 = n0 * norm, n1 * norm, n2 * norm, n3 * norm
    if symmetric:
        m1, m2, m3, m4 = n1 - d1 * n0, n2 - d2 * n0, n3 - d3 * n0, -d4 * n0
    else:
        m1, m2, m3, m4 = -(n1 - d1 * n0), -(n2 - d2 * n0), -(n3 - d3 * n0), d4 * n0
    SN, SM = n0 + n1 + n2 + n3, m1 + m2 + m3 + m4
    bn = [d1 * SN / SD, d2 * SN / SD, d3 * SN / SD, d4 * SN / SD]
    bm = [d1 * SM / SD, d2 * SM / SD, d3 * SM / SD, d4 * SM / SD]
    return (n0, n1, n2, n3), (d1, d2, d3, d4), (m1, m2, m3, m4), bn, bm


def _filter_data_array(data: np.ndarray, N, D, M, BN, BM) -> np.ndarray:
    """RecursiveSeparableImageFilter::FilterDataArray, vectorised over columns of data [ln, C] (float64)."""
    ln = data.shape[0]
    if ln < 4:
        raise ValueError("ITK recursive Gaussian needs at least 4 pixels along each filtered axis")
    n0, n1, n2, n3 = N; d1, d2, d3, d4 = D; m1, m2, m3, m4 = M
    s1 = np.empty_like(data); s2 = np.empty_like(data)
    v1 = data[0]
    s1[0] = v1 * n0 + v1 * n1 + v1 * n2 + v1 * n3
    s1[1] = data[1] * n0 + v1 * n1 + v1 * n2 + v1 * n3
    s1[2] = data[2] * n0 + data[1] * n1 + v1 * n2 + v1 * n3
    s1[3] = data[3] * n0 + data[2] * n1 + data[1] * n2 + v1 * n3
    s1[0] -= v1 * BN[0] + v1 * BN[1] + v1 * BN[2] + v1 * BN[3]
    s1[1] -= s1[0] * d1 + v1 * BN[1] + v1 * BN[2] + v1 * BN[3]
    s1[2] -= s1[1] * d1 + s1[0] * d2 + v1 * BN[2] + v1 * BN[3]
    s1[3] -= s1[2] * d1 + s1[1] * d2 + s1[0] * d3 + v1 * BN[3]
    for i in range(4, ln):
        s1[i] = data[i] * n0 + data[i - 1] * n1 + data[i - 2] * n2 + data[i - 3] * n3
        s1[i] -= s1[i - 1] * d1 + s1[i - 2] * d2 + s1[i - 3] * d3 + s1[i - 4] * d4
    v2 = data[ln - 1]
    s2[ln - 1] = v2 * m1 + v2 * m2 + v2 * m3 + v2 * m4
    s2[ln - 2] = data[ln - 1] * m1 + v2 * m2 + v2 * m3 + v2 * m4
    s2[ln - 3] = data[ln - 2] * m1 + data[ln - 1] * m2 + v2 * m3 + v2 * m4
    s2[ln - 4] = data[ln - 3] * m1 + data[ln - 2] * m2 + data[ln - 1] * m3 + v2 * m4
    s2[ln - 1] -= v2 * BM[0] + v2 * BM[1] + v2 * BM[2] + v2 * BM[3]
    s2[ln - 2] -= s2[ln - 1] * d1 + v2 * BM[1] + v2 * BM[2] + v2 * BM[3]
    s2[ln - 3] -= s2[ln - 2] * d1 + s2[ln - 1] * d2 + v2 * BM[2] + v2 * BM[3]
    s2[ln - 4] -= s2[ln - 3] * d1 + s2[ln - 2] * d2 + s2[ln - 1] * d3 + v2 * BM[3]
    for i in range(ln - 4, 0, -1):
        s2[i - 1] = data[i] * m1 + data[i + 1] * m2 + data[i + 2] * m3 + data[i + 3] * m4
        s2[i - 1] -= s2[i] * d1 + s2[i + 1] * d2 + s2[i + 2] * d3 + s2[i + 3] * d4
    return s1 + s2


@functools.lru_cache(maxsize=256)
def gaussian_operator(n: int, sigma: float, spacing: float, order: int, normalize_across_scale: bool) -> np.ndarray:
    """K [n, n] with out = K @ line for ITK's RecursiveGaussianImageFilter along one axis."""
    if spacing < 1e-8:
        raise ValueError("spacing suspiciously small")
    coefs = _gauss_coefficients(sigma / abs(spacing), order, normalize_across_scale, sigma)
    return _filter_data_array(np.eye(n), *coefs)


# ------------------------------------------------------------------------------- B-spline decomposition (ITK)
@functools.lru_cache(maxsize=256)
def bspline3_prefilter_operator(n: int, tolerance: float = 1e-10) -> np.ndarray:
    """K [n, n] for BSplineDecompositionImageFilter::DataToCoefficients1D (order 3, mirror boundaries)."""
    data = np.eye(n)
    if n == 1:
        return data
    z = math.sqrt(3.0) - 2.0
    c0 = (1.0 - z) * (1.0 - 1.0 / z)
    data = data * c0
    horizon = int(math.ceil(math.log(tolerance) / math.log(abs(z)))) if tolerance > 0 else n
    if horizon < n:
        zn = z
        s = data[0].copy()
        for k in range(1, horizon):
            s += zn * data[k]
            zn *= z
        data[0] = s
    else:
        zn = z
        iz = 1.0 / z
        z2n = z ** (n - 1)
        s = data[0] + z2n * data[n - 1]
        z2n *= z2n * iz
        for k in range(1, n - 1):
            s += (zn + z2n) * data[k]
            zn *= z
            z2n *= iz
        data[0] = s / (1.0 - zn * zn)
    for k in range(1, n):
        data[k] += z * data[k - 1]
    data[n - 1] = (z / (z * z - 1.0)) * (z * data[n - 2] + data[n - 1])
    for k in range(n - 2, -1, -1):
        data[k] = z * (data[k + 1] - data[k])
    return data


# ------------------------------------------------------------------------------------------- axis application
def _apply_axis(vol: torch.Tensor, K: np.ndarray, axis_xyz: int) -> torch.Tensor:
    """Apply K along an image axis (0=x, 1=y, 2=z) of vol [Z, Y, X] (float64 GEMM)."""
    Kt = torch.as_tensor(K, device=vol.device, dtype=torch.float64)
    if axis_xyz == 0:
        return torch.matmul(vol, Kt.T)
    if axis_xyz == 1:
        return torch.einsum("zyx,ky->zkx", vol, Kt)
    return torch.einsum("zyx,kz->kyx", vol, Kt)


def smoothing_recursive_gaussian(vol: torch.Tensor, spacing_xyz, sigma_xyz) -> torch.Tensor:
    """sitk.SmoothingRecursiveGaussian(img, sigma_xyz, normalizeAcrossScale=False) on vol [Z, Y, X] float32."""
    Z, Y, X = vol.shape
    n = (X, Y, Z)
    out = vol.to(torch.float64)
    for ax in (2, 0, 1):                         # ITK: first filter along ImageDimension-1, then 0, 1
        out = _apply_axis(out, gaussian_operator(n[ax], float(sigma_xyz[ax]), float(spacing_xyz[ax]), 0, False), ax)
        out = out.to(torch.float32).to(torch.float64)   # RealImageType = float between passes
    return out.to(torch.float32)


def gradient_recursive_gaussian(vol: torch.Tensor, spacing_xyz, direction: np.ndarray, sigma: float,
                                normalize_across_scale: bool = True) -> torch.Tensor:
    """sitk.GradientRecursiveGaussian(img, sigma, normalizeAcrossScale, useImageDirection=True) -> [3, Z, Y, X] f32."""
    Z, Y, X = vol.shape
    n = (X, Y, Z)
    v64 = vol.to(torch.float64)
    comps = []
    for d in range(3):
        g = _apply_axis(v64, gaussian_operator(n[d], float(sigma), float(spacing_xyz[d]), 1, normalize_across_scale), d)
        for ax in [a for a in range(3) if a != d]:
            g = _apply_axis(g, gaussian_operator(n[ax], float(sigma), float(spacing_xyz[ax]), 0, normalize_across_scale), ax)
        comps.append(g / float(spacing_xyz[d]))
    g = torch.stack(comps)                                        # index-axis gradient (per mm)
    D = torch.as_tensor(np.asarray(direction, dtype=np.float64).reshape(3, 3), device=vol.device)
    g = torch.einsum("ij,jzyx->izyx", D, g)                       # TransformLocalVectorToPhysicalVector
    return g.to(torch.float32)


def bspline3_coefficients(vol: torch.Tensor) -> torch.Tensor:
    """BSplineDecompositionImageFilter (order 3) -> float64 coefficients [Z, Y, X]."""
    Z, Y, X = vol.shape
    n = (X, Y, Z)
    c = vol.to(torch.float64)
    for ax in (0, 1, 2):
        c = _apply_axis(c, bspline3_prefilter_operator(n[ax]), ax)
    return c


# ----------------------------------------------------------------------------------------------- resampling
def _geom(img_like):
    """(size xyz, origin, index->physical matrix) from an sitk.Image or a (size, origin, spacing, direction) tuple."""
    if isinstance(img_like, sitk.Image):
        size, origin, spacing, direction = img_like.GetSize(), img_like.GetOrigin(), img_like.GetSpacing(), img_like.GetDirection()
    else:
        size, origin, spacing, direction = img_like
    m = np.asarray(direction, dtype=np.float64).reshape(3, 3) @ np.diag(np.asarray(spacing, dtype=np.float64))
    return tuple(int(s) for s in size), np.asarray(origin, dtype=np.float64), m


def resample(src: torch.Tensor, src_geom, ref_geom, M: np.ndarray, off: np.ndarray, mode: int = 3,
             default: float = 0.0, out_dtype=torch.float32, coefficients: torch.Tensor | None = None) -> torch.Tensor:
    """ResampleImageFilter: output on ref grid, value = interp(src at T(x)) where T(x) = M x + off (physical, LPS).

    src: [Z, Y, X] tensor. mode: 0 nearest, 1 linear, 3 cubic B-spline (coefficients computed if not given).
    Returns [Zo, Yo, Xo] tensor of out_dtype.
    """
    (isx, isy, isz), iorig, im = _geom(src_geom)
    (osx, osy, osz), oorig, om = _geom(ref_geom)
    # output index -> physical -> transform -> input continuous index: affine in the output index.
    p2i = np.linalg.inv(im)
    A = p2i @ M @ om
    b = p2i @ (M @ oorig + off - iorig)
    Ab = torch.as_tensor(np.concatenate([np.c_[A, b]]).reshape(-1), device=src.device, dtype=torch.float64)
    if mode == 3:
        data = coefficients if coefficients is not None else bspline3_coefficients(src)
    elif mode == 1:
        data = src.to(torch.float64)
    else:
        data = src
    data = data.contiguous()
    from .kernels import launch_resample   # Triton; ImportError -> caller falls back to ITK
    out = torch.empty((osz, osy, osx), dtype=out_dtype, device=src.device)
    launch_resample(data, out, osx, osy, osz, Ab, float(default), isx, isy, isz, mode)
    return out


def shrink_geometry(img: sitk.Image, factor: int):
    """Output geometry of ITK ShrinkImageFilter (same factor on every axis), without copying pixel data."""
    tiny = sitk.Image([max(1, s) for s in img.GetSize()], sitk.sitkUInt8)  # geometry only
    tiny.CopyInformation(img)
    sh = sitk.Shrink(tiny, [factor] * 3)
    return sh.GetSize(), sh.GetOrigin(), sh.GetSpacing(), sh.GetDirection()

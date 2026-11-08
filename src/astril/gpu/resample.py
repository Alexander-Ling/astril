"""GPU drop-ins for astril's resampling helpers (preprocessing_utils.sitk_resample_to_spacing / make_sitk_resampler).

`resample_to_spacing` reproduces sitk_resample_to_spacing's output geometry exactly and resamples on the GPU.
`GpuResampleImageFilter` mimics the ResampleImageFilter calls astril makes (SetReferenceImage, SetTransform,
SetInterpolator, SetDefaultPixelValue, SetOutputPixelType, SetNumberOfThreads, Execute). Linear transforms (including
composites of linear transforms) are reduced to one affine map; anything else, or an unsupported interpolator / pixel
type, is executed by ITK unchanged.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch

from . import filters as gf

_MODES = {sitk.sitkNearestNeighbor: 0, sitk.sitkLinear: 1, sitk.sitkBSpline: 3}
_TORCH_OUT = {sitk.sitkFloat32: torch.float32, sitk.sitkFloat64: torch.float64}
_INT_OUT = {sitk.sitkUInt8: (np.uint8, 0, 255), sitk.sitkInt8: (np.int8, -128, 127),
            sitk.sitkUInt16: (np.uint16, 0, 65535), sitk.sitkInt16: (np.int16, -32768, 32767),
            sitk.sitkUInt32: (np.uint32, 0, 2 ** 32 - 1), sitk.sitkInt32: (np.int32, -2 ** 31, 2 ** 31 - 1)}


def _affine_of(transform: sitk.Transform):
    """(M, off) with T(x) = M x + off for a linear transform, probed exactly with TransformPoint."""
    t = transform.Downcast() if hasattr(transform, "Downcast") else transform
    if not t.IsLinear():
        return None
    o = np.asarray(t.TransformPoint((0.0, 0.0, 0.0)))
    cols = [np.asarray(t.TransformPoint(tuple(e))) - o for e in np.eye(3)]
    return np.stack(cols, axis=1), o


def _to_sitk(arr: np.ndarray, ref, pixel_id):
    if pixel_id in _INT_OUT:
        dt, lo, hi = _INT_OUT[pixel_id]
        arr = np.trunc(np.clip(arr, lo, hi)).astype(dt)          # CastPixelWithBoundsChecking + static_cast
    out = sitk.GetImageFromArray(arr)
    if isinstance(ref, sitk.Image):
        out.CopyInformation(ref)
    else:
        size, origin, spacing, direction = ref
        out.SetOrigin(origin); out.SetSpacing(spacing); out.SetDirection(direction)
    if pixel_id not in _INT_OUT and out.GetPixelID() != pixel_id:
        out = sitk.Cast(out, pixel_id)
    return out


def gpu_resample_image(moving: sitk.Image, ref, transform: sitk.Transform, interpolator, default=0.0,
                       pixel_id=sitk.sitkFloat32, device="cuda"):
    """ResampleImageFilter on the GPU; returns None if this configuration must stay on ITK."""
    if moving.GetDimension() != 3 or moving.GetNumberOfComponentsPerPixel() != 1 or interpolator not in _MODES:
        return None
    if pixel_id not in _TORCH_OUT and pixel_id not in _INT_OUT:
        return None
    aff = _affine_of(transform)
    if aff is None:
        return None
    from . import triton_usable
    if not triton_usable(str(device)):
        return None
    from . import memory
    n_out = int(np.prod(gf._geom(ref)[0]))
    try:
        # Reserve the measured peak (B-spline coefficients + temporaries + output) so concurrent exam threads cannot
        # oversubscribe the device; held until the result is back on the host.
        with memory.reserve(device, memory.resample_bytes(moving.GetNumberOfPixels(), n_out)):
            src = torch.as_tensor(sitk.GetArrayFromImage(moving).astype(np.float32, copy=False), device=device)
            out = gf.resample(src, moving, ref, aff[0], aff[1], mode=_MODES[interpolator], default=float(default),
                              out_dtype=torch.float64 if pixel_id == sitk.sitkFloat64 else torch.float32)
            arr = out.cpu().numpy()
            del src, out
    except Exception as e:  # noqa: BLE001 - e.g. out of GPU memory: ITK does it instead
        if memory.is_oom(e):
            memory.release_cache()
        return None
    return _to_sitk(arr, ref, pixel_id)


class GpuResampleImageFilter:
    """The ResampleImageFilter surface astril's make_sitk_resampler uses; falls back to ITK when needed."""

    def __init__(self, device="cuda"):
        self.device = device
        self.ref = None
        self.transform = sitk.Transform()
        self.interp = sitk.sitkLinear
        self.default = 0.0
        self.pixel_id = None
        self.threads = None
        self.used_gpu = None

    def SetReferenceImage(self, ref): self.ref = ref
    def SetTransform(self, t): self.transform = t
    def SetInterpolator(self, i): self.interp = i
    def GetInterpolator(self): return self.interp
    def SetDefaultPixelValue(self, v): self.default = float(v)
    def SetOutputPixelType(self, p): self.pixel_id = p
    def SetNumberOfThreads(self, n): self.threads = n

    def Execute(self, moving: sitk.Image) -> sitk.Image:
        pid = self.pixel_id if self.pixel_id is not None else moving.GetPixelID()
        out = gpu_resample_image(moving, self.ref, self.transform, self.interp, self.default, pid, self.device)
        self.used_gpu = out is not None
        if out is None:
            f = sitk.ResampleImageFilter()
            if self.threads:
                f.SetNumberOfThreads(int(self.threads))
            f.SetReferenceImage(self.ref); f.SetTransform(self.transform); f.SetInterpolator(self.interp)
            f.SetDefaultPixelValue(self.default); f.SetOutputPixelType(pid)
            out = f.Execute(moving)
        return out


def resample_to_spacing(img, target_spacing_mm, *, interp=None, interpolator=None, default_value=0.0, pixel_id=None,
                        n_workers=None, device="cuda"):
    """GPU version of astril.preprocessing_utils.sitk_resample_to_spacing (same output geometry)."""
    from ..preprocessing_utils import _normalize_spacing_mm, _interp_to_sitk_interpolator
    if img.GetDimension() != 3:
        raise ValueError(f"sitk_resample_to_spacing supports only 3D images, got dim={img.GetDimension()}")
    target = _normalize_spacing_mm(target_spacing_mm)
    in_spacing = tuple(float(x) for x in img.GetSpacing())
    if all(abs(in_spacing[i] - target[i]) <= 1e-6 for i in range(3)):
        return img
    in_size = tuple(int(x) for x in img.GetSize())
    new_size = [max(1, int(round(in_size[i] * (in_spacing[i] / target[i])))) for i in range(3)]
    if interpolator is None:
        interpolator, _, _ = _interp_to_sitk_interpolator(interp)
    ref = (new_size, img.GetOrigin(), tuple(float(x) for x in target), img.GetDirection())
    pid = pixel_id if pixel_id is not None else img.GetPixelID()
    out = gpu_resample_image(img, ref, sitk.Transform(), interpolator, default_value, pid, device)
    if out is not None:
        return out
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(tuple(float(x) for x in target)); f.SetSize(new_size)
    f.SetOutputOrigin(img.GetOrigin()); f.SetOutputDirection(img.GetDirection())
    f.SetTransform(sitk.Transform()); f.SetInterpolator(interpolator)
    f.SetDefaultPixelValue(float(default_value)); f.SetOutputPixelType(pid)
    return f.Execute(img)

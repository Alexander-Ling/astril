"""Optional GPU backend for astril registration (`registration_backend` in preprocess.register_images).

ITK stays the reference implementation and the default. The GPU backend reproduces ITK's registration (Mattes MI /
Correlation, ImageRegistrationMethodv4 pyramid, RegularStepGradientDescentOptimizerv4, PhysicalShift scales) and the
resampling helpers astril uses, validated against ITK to ~0.02 mm in final pose.

Layers and fallbacks:
  * CUDA unavailable                      -> ITK for everything.
  * CUDA, Triton unusable                 -> GPU registration with pure-torch batched kernels; ITK resampling.
  * CUDA + Triton (`triton` on Linux, `triton-windows` on Windows) -> fused GPU kernels for both.
Registrations from all threads of a process share one batched service per GPU (see service.py); settings the GPU path
does not implement are executed by ITK unchanged.
"""
from __future__ import annotations

import functools
import os
import threading
import warnings

_WARNED: set = set()
_LOCK = threading.Lock()


def scratch_root() -> str:
    """Where astril.gpu writes its own small files: $ASTRIL_SCRATCH_DIR if set, else the system temp dir
    (the same rule preprocessing uses for its scratch workspace)."""
    import tempfile
    return os.environ.get("ASTRIL_SCRATCH_DIR") or tempfile.gettempdir()


# Triton caches compiled kernels (a few MB) in ~/.triton/cache by default. When the user points astril's scratch at a
# specific drive, keep Triton's cache there too (unless TRITON_CACHE_DIR is already set). Must run before any compile.
if os.environ.get("ASTRIL_SCRATCH_DIR") and not os.environ.get("TRITON_CACHE_DIR"):
    os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ["ASTRIL_SCRATCH_DIR"], "astril_triton_cache")


def _warn_once(key: str, msg: str) -> None:
    with _LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - no/broken torch -> no GPU backend
        return False


_TRITON_OK: dict = {}
_TRITON_LOCK = threading.Lock()


def triton_usable(device: str = "cuda:0") -> bool:
    """True if Triton imports, compiles and runs correctly on `device` (checked exactly once per process: exam threads
    calling this at the same moment must not compile the self-test kernel concurrently)."""
    if not cuda_available():
        return False
    with _TRITON_LOCK:
        if device not in _TRITON_OK:
            _TRITON_OK[device] = _triton_self_test(device)
        return _TRITON_OK[device]


def _triton_self_test(device: str) -> bool:
    try:
        from .kernels import self_test
        self_test(device)
        return True
    except Exception as e:  # noqa: BLE001
        _warn_once("triton", "astril GPU backend: Triton unavailable or incompatible with the installed torch "
                             f"({type(e).__name__}: {e}); using pure-torch GPU kernels and ITK resampling. "
                             "Install a Triton build matching your torch version (Windows: `pip install "
                             "\"triton-windows<3.(torch minor + 1)\"`) for full speed.")
        return False


def resolve_backend(requested: str | None) -> str:
    """'itk' | 'gpu' | 'auto' -> the backend actually used ('itk' or 'gpu')."""
    r = (requested or "auto").strip().lower()
    if r == "itk":
        return "itk"
    if r not in ("gpu", "auto"):
        raise ValueError(f"registration_backend must be 'itk', 'gpu' or 'auto'; got {requested!r}")
    if cuda_available():
        return "gpu"
    if r == "gpu":
        _warn_once("nocuda", "registration_backend='gpu' requested but CUDA is not available; using ITK.")
    return "itk"


def registration_method_class():
    """Class to use in place of sitk.ImageRegistrationMethod for the GPU backend."""
    from .registration import GpuImageRegistrationMethod

    class _Configured(GpuImageRegistrationMethod):
        def __init__(self):
            super().__init__(device=os.environ.get("ASTRIL_GPU_DEVICE", "cuda:0"),
                             spill_to_cpu=os.environ.get("ASTRIL_GPU_SPILL_TO_CPU", "0").strip() == "1")

    return _Configured


def resample_to_spacing(img, target_spacing_mm, *, interp=None, interpolator=None, default_value=0.0, pixel_id=None,
                        n_workers=None):
    """GPU drop-in for preprocessing_utils.sitk_resample_to_spacing (same signature and output geometry)."""
    from .resample import resample_to_spacing as _r
    return _r(img, target_spacing_mm, interp=interp, interpolator=interpolator, default_value=default_value,
              pixel_id=pixel_id, n_workers=n_workers, device=os.environ.get("ASTRIL_GPU_DEVICE", "cuda:0"))


def make_resampler(*, reference_img, transform, default_value, pixel_id, n_workers=None, interpolator=None, interp=None):
    """GPU drop-in for preprocessing_utils.make_sitk_resampler (returns an object with ResampleImageFilter's calls)."""
    from ..preprocessing_utils import _interp_to_sitk_interpolator
    from .resample import GpuResampleImageFilter
    f = GpuResampleImageFilter(device=os.environ.get("ASTRIL_GPU_DEVICE", "cuda:0"))
    if n_workers is not None:
        f.SetNumberOfThreads(n_workers)
    f.SetReferenceImage(reference_img)
    f.SetTransform(transform)
    if interpolator is None:
        interpolator, _, _ = _interp_to_sitk_interpolator(interp)
    f.SetInterpolator(interpolator)
    f.SetDefaultPixelValue(float(default_value))
    f.SetOutputPixelType(pixel_id)
    return f

# astril/preprocessing_utils.py

import numpy as np
import nibabel as nib
import shutil
import os
import warnings
try:
    import pydicom
except Exception:
    pydicom = None
import re
from collections import OrderedDict
import datetime as _dt
import pandas as pd
import xlsxwriter
import hashlib
import tempfile
import json
import subprocess
import glob
import contextlib
import threading
from scipy.ndimage import zoom
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version
from packaging.specifiers import SpecifierSet
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -----------------------------------------------------------------------------
# SimpleITK / ITK thread control helpers
# -----------------------------------------------------------------------------
#
# Rationale
# ---------
# SimpleITK wraps ITK ProcessObjects which may run multi-threaded. When running
# multiple registrations in parallel, setting the *global* ITK thread cap can
# cause collisions across workers. We therefore:
#   1) Prefer per-object thread limits via obj.SetNumberOfThreads(n) when available
#   2) Fall back to a temporary global cap ONLY when required, protected by a
#      module-level lock so concurrent calls do not race while the global cap is active.
#
# Notes
# -----
# - The lock only affects the fallback path (global cap). If per-object thread
#   controls are available, parallelism is preserved.
# - Even with the lock, the ITK thread cap is global while held, so this may
#   reduce concurrency for registrations that require this fallback.

_SITK_GLOBAL_THREADCAP_LOCK = threading.Lock()


def normalize_n_workers(n):
    """Normalize/validate a thread cap value.

    Parameters
    ----------
    n : int | None
        Desired number of worker threads.

    Returns
    -------
    int | None
        None means: do not set any caps (use library defaults).
    """
    if n is None:
        return None
    try:
        n_int = int(n)
    except Exception:
        raise ValueError(f"n_workers must be an int or None; got {n!r}")
    if n_int <= 0:
        raise ValueError(f"n_workers must be >= 1 or None; got {n_int}")
    return n_int


def set_sitk_object_threads(obj, n_workers):
    """Best-effort per-object thread cap for SimpleITK ProcessObjects.

    Returns
    -------
    bool
        True if the cap was applied (or no cap requested), False if the object
        does not expose SetNumberOfThreads.
    """
    if n_workers is None:
        return True
    setter = getattr(obj, "SetNumberOfThreads", None)
    if callable(setter):
        setter(int(n_workers))
        return True
    return False


@contextlib.contextmanager
def global_sitk_thread_cap(n_workers, enabled: bool, verbose: bool = False):
    """Temporarily set the *global* ITK thread cap (last resort).

    This is protected by a module-level lock to avoid races when multiple
    register_images() calls happen concurrently in the same Python process.
    """
    if not enabled:
        yield
        return

    if n_workers is None:
        # Nothing to do
        yield
        return

    # Lazy import to avoid pulling SimpleITK at module import time.
    import SimpleITK as sitk

    with _SITK_GLOBAL_THREADCAP_LOCK:
        old = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        try:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(n_workers))
            if verbose:
                print(
                    f"[register_images] WARNING: falling back to global ITK thread cap ({n_workers}). "
                    f"This is protected by a lock, but the setting is still global while active."
                )
            yield
        finally:
            # Best effort restore; do not mask the original exception if restore fails
            try:
                sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(old))
            except Exception:
                pass


def make_sitk_resampler(
    *,
    reference_img,
    transform,
    default_value: float,
    pixel_id,
    n_workers=None,
    interpolator=None,
):
    """Create a ResampleImageFilter with optional per-object thread cap.

    Parameters are keyword-only to keep callsites explicit.
    """
    import SimpleITK as sitk

    f = sitk.ResampleImageFilter()
    set_sitk_object_threads(f, n_workers)
    f.SetReferenceImage(reference_img)
    f.SetTransform(transform)
    if interpolator is None:
        interpolator = sitk.sitkLinear
    f.SetInterpolator(interpolator)
    f.SetDefaultPixelValue(float(default_value))
    f.SetOutputPixelType(pixel_id)
    return f

# -------- small helper for progress --------
def _progress(iterable, total=None, desc=None, unit=None, enable=True):
    if not enable or tqdm is None:
        return iterable
    kwargs = {}
    if total is not None: kwargs["total"] = total
    if desc: kwargs["desc"] = desc
    if unit: kwargs["unit"] = unit
    return tqdm(iterable, **kwargs)

def apply_padding_anydim(arr, pad):
    """Apply padding/cropping to the first 3 axes of an array (3D or 4D).

    Parameters
    ----------
    arr : np.ndarray
        Input array with ndim >= 3. If 4D, the last axis is treated as time/frames and is not padded/cropped.
    pad : array-like, shape (3, 2)
        Padding/cropping for each spatial axis: [[before, after], ...].
        Positive values pad with zeros; negative values crop.

    Returns
    -------
    np.ndarray
        Padded/cropped array.
    """
    pad = np.asarray(pad, dtype=int)
    if pad.shape != (3, 2):
        raise ValueError(f"pad must be shape (3,2), got {pad.shape}")
    if arr.ndim < 3:
        raise ValueError(f"arr must have ndim >= 3, got {arr.ndim}")

    # --- Crop first (negative padding) via slicing ---
    slices = []
    for ax in range(3):
        before, after = int(pad[ax, 0]), int(pad[ax, 1])
        start = max(-before, 0)
        end_crop = max(-after, 0)
        end = None if end_crop == 0 else -end_crop
        slices.append(slice(start, end))
    # Keep any remaining axes (e.g. time) intact
    for ax in range(3, arr.ndim):
        slices.append(slice(None))
    cropped = arr[tuple(slices)]

    # --- Pad (positive padding) ---
    pad_width = []
    for ax in range(3):
        before, after = int(pad[ax, 0]), int(pad[ax, 1])
        pad_width.append((max(before, 0), max(after, 0)))
    for ax in range(3, arr.ndim):
        pad_width.append((0, 0))

    if any(p[0] > 0 or p[1] > 0 for p in pad_width):
        cropped = np.pad(cropped, pad_width, mode='constant', constant_values=0)
    return cropped


def apply_padding(data, pad):
    """Backward-compatible wrapper: pads/crops the first 3 axes."""
    return apply_padding_anydim(data, pad)


def _interp_to_scipy_order(interp):
    # Accept ints directly
    if isinstance(interp, (int, np.integer)):
        order = int(interp)
    elif isinstance(interp, str):
        key = interp.strip().lower()
        mapping = {
            "nearest": 0,
            "linear": 1,
            "bilinear": 1,
            "quadratic": 2,
            "cubic": 3,
            "quartic": 4,
            "quintic": 5,
        }
        if key not in mapping:
            raise ValueError(f"Unknown interp='{interp}'. Use one of: {sorted(mapping)} or an int 0-5.")
        order = mapping[key]
    else:
        raise TypeError(f"interp must be an int (0-5) or a string like 'linear'. Got {type(interp)}")

    if order < 0 or order > 5:
        raise ValueError(f"scipy.ndimage.zoom order must be in [0,5]. Got {order}")
    return order

def prepare_zoom(original_voxel_dims, target_voxel_dims, interp):
    """Precompute scipy.ndimage.zoom factors + interpolation order."""
    zoom_factors = np.divide(original_voxel_dims, target_voxel_dims)
    order = _interp_to_scipy_order(interp)
    return zoom_factors, order


def interpolate_to_voxel_dims_precomputed(data, zoom_factors, order):
    """Resample a single 3D frame using precomputed zoom factors/order."""
    return zoom(data, zoom_factors, order=order)


def interpolate_to_voxel_dims(data, original_voxel_dims, target_voxel_dims, interp):
    """Backward-compatible helper: compute zoom factors/order and resample."""
    zoom_factors, order = prepare_zoom(original_voxel_dims, target_voxel_dims, interp)
    return interpolate_to_voxel_dims_precomputed(data, zoom_factors, order)


def update_origin_for_padding(affine_matrix, padding, voxel_dims):
    shifts = np.array([pad[0] * voxel_dim for pad, voxel_dim in zip(padding, voxel_dims)])
    affine_matrix[:3, 3] -= shifts
    return affine_matrix


def adjust_to_target_shape(data, target_shape, padding_record=None, shape_padding=None):
    """Pad/crop the first 3 axes of data to match target_shape.

    Supports 3D (X,Y,Z) and 4D (X,Y,Z,T). For 4D, the time axis is preserved and the same
    spatial padding/cropping is applied to every frame.
    """
    current_shape = np.array(data.shape[:3], dtype=int)
    target_shape = np.array(target_shape, dtype=int)

    if shape_padding is None:
        shape_padding = np.zeros((3, 2), dtype=int)
        for axis in range(3):
            diff = int(target_shape[axis] - current_shape[axis])
            pad_before = diff // 2
            pad_after = diff - pad_before
            shape_padding[axis] = [pad_before, pad_after]

    final_data = apply_padding_anydim(data, shape_padding)
    if padding_record is not None:
        padding_record['shape_padding'] = np.array(shape_padding, dtype=int).tolist()
    return final_data, padding_record


def read_padding_record(filepath):
    with open(filepath, 'r') as f:
        return eval(f.read(), {"array": np.array})


def load_roi_mask(filepath, shape):
    mask = nib.load(filepath).get_fdata()
    if mask.shape != shape:
        raise ValueError("ROI mask dimensions must match data dimensions.")
    return mask

def ensure_hd_bet_installed(
    version_spec=">=2.0.0,<3.0.0"
):
    # 1. Ensure CLI exists
    if shutil.which("hd-bet") is None:
        raise ImportError(
            "HD-BET CLI not found in PATH.\n\n"
            "Install it via pip (into the same environment you're using):\n"
            "    pip install hd-bet\n\n"
            "Or use the preprocessing extra:\n"
            "    pip install astril[preprocessing]"
        )

    # 2. Check installed package version
    try:
        installed_version = Version(version("hd-bet"))
    except PackageNotFoundError:
        raise ImportError(
            "hd-bet appears to be on PATH, but the Python package is not installed "
            "in this environment.\n"
            "Please install it with:\n"
            "    pip install hd-bet"
        )

    spec = SpecifierSet(version_spec)
    if installed_version not in spec:
        raise ImportError(
            f"hd-bet version {installed_version} is installed, but "
            f"version {version_spec} is required.\n\n"
            "Please upgrade/downgrade:\n"
            f"    pip install 'hd-bet{version_spec}'"
        )

    # 3. Ensure parameter directory exists
    param_dir = os.path.expanduser("~/hd-bet_params")
    os.makedirs(param_dir, exist_ok=True)

def load_nifti_data(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header

def save_nifti_data(data, affine, header, path):
    img = nib.Nifti1Image(data, affine, header=header)
    nib.save(img, path)

def validate_volume_shapes(volumes):
    shapes = [v[0].shape for v in volumes]
    if len(set(shapes)) > 1:
        raise ValueError("All input volumes must have the same shape.")
    affines = [v[1] for v in volumes]
    if not all(np.allclose(aff, affines[0]) for aff in affines):
        warnings.warn("Affine matrices differ across input volumes.")

def ensure_dicom2nifti_installed():
    try:
        import dicom2nifti
    except ImportError:
        raise ImportError(
            "dicom2nifti is not installed.\n\n"
            "Install it with:\n"
            "    pip install astril[preprocessing]\n"
            "or:\n"
            "    pip install dicom2nifti"
        )


# ------------------------------------------------------------------------
# Helper functions for DICOM file classification
# ------------------------------------------------------------------------

# Orientation tokens should NEVER imply "derived"
_ORIENT_TOKENS = {"ax","axial","sag","sagittal","cor","coronal","oblique","obl"}

def _detect_plane(tokens: str) -> str | None:
    t = tokens
    if re.search(r"\b(ax|axial)\b", t): return "AX"
    if re.search(r"\b(sag|sagittal)\b", t): return "SAG"
    if re.search(r"\b(cor|coronal)\b", t): return "COR"
    # 'tra' used in some MPR labels; avoid matching 'trace' (DWI)
    if re.search(r"\btra\b", t): return "AX"
    if "oblique" in t or "obl" in t: return "OBL"
    return None

def _plane_from_iop(ds, thresh: float = 0.8) -> str | None:
    """Infer plane (AX/SAG/COR) from ImageOrientationPatient.
    Returns None for oblique stacks when alignment is below threshold.
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    if not iop or len(iop) < 6:
        return None
    try:
        row = np.array([float(iop[0]), float(iop[1]), float(iop[2])], dtype=float)
        col = np.array([float(iop[3]), float(iop[4]), float(iop[5])], dtype=float)
        n = np.cross(row, col)
        if np.linalg.norm(n) == 0:
            return None
        n = n / np.linalg.norm(n)
        axes = np.abs(n)  # projection onto x,y,z
        idx = int(np.argmax(axes))
        if axes[idx] < thresh:
            return None  # oblique
        return ["SAG", "COR", "AX"][idx]  # ex,ey,ez order
    except Exception:
        return None

def _is_projection(tokens: str) -> bool:
    t = tokens
    # Only real projections should be caught name-only. Do NOT treat MPR/RFMT as derived by name.
    if re.search(r"\bmip\b", t) or re.search(r"\bminip\b", t):
        return True
    return any(k in t for k in ["project","projection","thick slab","slab"])

_TEXT_TRUE = {"1","true","t","y","yes"}
def _nz(x, default=None):
    return x if (x is not None and x != "") else default

def _to_list_upper(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i).upper() for i in x]
    s = str(x)
    return [i.strip().upper() for i in re.split(r"[\\^,; ]+", s) if i.strip()]

def _safe_float(x):
    try: return float(x)
    except: return None

def _parse_dt(ds):
    """Return (datetime or None, iso_str) using AcquisitionDate/Time or ContentDate/Time or SeriesDate/Time."""
    for dtag, ttag in [
        ("AcquisitionDate", "AcquisitionTime"),
        ("ContentDate", "ContentTime"),
        ("SeriesDate", "SeriesTime"),
        ("StudyDate", "StudyTime"),
    ]:
        d = _nz(getattr(ds, dtag, None))
        t = _nz(getattr(ds, ttag, None))
        if d:
            try:
                # DICOM time may be HHMMSS(.ffffff)
                hh = mm = ss = 0
                us = 0
                if t:
                    th = t.split(".")
                    base = th[0].rjust(6,"0")
                    hh, mm, ss = int(base[0:2]), int(base[2:4]), int(base[4:6])
                    if len(th) > 1:
                        us = int(th[1].ljust(6,"0")[:6])
                dt = _dt.datetime.strptime(d, "%Y%m%d").replace(hour=hh, minute=mm, second=ss, microsecond=us)
                return dt, dt.isoformat()
            except Exception:
                pass
    return None, None

def _first_dcm_in(folder):
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(".dcm"):
                return os.path.join(root, f)
        break
    return None

def _norm_text(*vals):
    parts = []
    for v in vals:
        if not v: continue
        parts.append(str(v))
    s = " ".join(parts)
    return re.sub(r"\s+", " ", s).strip()

def _name_tokens(text):
    s = (text or "").lower()
    s = re.sub(r"[_\-]+", " ", s)
    return s

def _is_after(a, b):
    return (a is not None and b is not None and a > b)

def _detect_fspgr(name_tokens, seq, prot):
    s = " ".join([name_tokens, str(seq or "").lower(), str(prot or "").lower()])
    flags = ["fspgr", "spgr", "bravo", "mprage", "mp rage", "vibe", "spoiled", "t1 cube", "t1cube", "tfl3d", "tfl"]
    return any(k in s for k in flags)

def _vendor_hints(ds):
    """
    Extract cross-vendor hints if present. Always return a dict with keys, some may be None.
    """
    def g(name, default=None): 
        return getattr(ds, name, default) if hasattr(ds, name) else default

    hints = {
        "manufacturer": g("Manufacturer"),
        "pulse_sequence_name": g("PulseSequenceName", None) or g("SequenceName", None),
        "scanning_sequence": g("ScanningSequence", None),
        "sequence_variant": g("SequenceVariant", None),
        "scan_options": g("ScanOptions", None),
        "mr_acq_type": g("MRAcquisitionType", None),  # 2D/3D
        "contrast_agent": g("ContrastBolusAgent", None),
        "contrast_volume": g("ContrastBolusVolume", None),
        "acquisition_contrast": g("AcquisitionContrast", None),  # sometimes "CONTRAST" on enhanced MR
    }

    # Try multiple places for B-value (Siemens/GE/Philips variants)
    b_candidates = [
        "DiffusionBValue", "DiffusionBFactor", "Philips_b_value",
        "Private_0019_100c", "Private_0043_1039", "Private_2001_1003",
    ]
    bval = None
    for cand in b_candidates:
        v = g(cand, None)
        if v is None: 
            continue
        try:
            # GE 0043,1039 may contain a vector-like string; take the last numeric
            if isinstance(v, str) and any(ch in v for ch in ["\\", " ", ",", ";"]):
                nums = [float(x) for x in re.split(r"[\\,;\s]+", v) if re.match(r"^[\d.]+$", x)]
                if nums:
                    bval = nums[-1]
                    break
            bval = float(v)
            break
        except Exception:
            continue

    hints["b_value"] = bval
    return hints

def _match_cat_from_spec(spec: "OrderedDict[str, dict]", tokens_lc: str, imgtypes_uc: set[str]) -> str | None:
    """Return the first matching category key from spec based on synonyms or ImageType/FrameType tokens."""
    for cat, meta in spec.items():
        syn = [s.lower() for s in meta.get("syn", [])]
        if any(s in tokens_lc for s in syn) or (cat.upper() in imgtypes_uc):
            return cat
    return None

# ---------- Single source of truth for derived families ----------
# Each entry: { category_name: {"gen": generator_key, "syn": [name synonyms ...], "policy": [convert_only or derive or ignore]"} }
DERIVED_CATEGORY_SPEC: dict[str, "OrderedDict[str, dict]"] = {
    "DWI": OrderedDict([
        ("TRACE",      {"gen": "dwi_trace",       "syn": ["trace", "tracew", "trace w", "isotropic", "iso"],        "policy": "derive"}),
        ("ADC",        {"gen": "dwi_adc",         "syn": ["adc"],                                                   "policy": "derive"}),
        ("FA",         {"gen": "dwi_fa",          "syn": ["fa"],                                                    "policy": "derive"}),
        ("MD",         {"gen": "dwi_md",          "syn": ["md", "mean diffusivity", "mean diff", "avdc"],           "policy": "derive"}), #Note that AvDC scans are average (mean) diffusivity while ADC scans are apparent diffusivity.
        ("EXP_ATTEN",  {"gen": "dwi_exp_atten",   "syn": ["exp atten", "expatten"],                                 "policy": "derive"}),
    ]),
    "SWI": OrderedDict([
        ("MIP",        {"gen": "swi_mip",         "syn": ["mip"],                                                   "policy": "derive"}),
        ("MINIP",      {"gen": "swi_minip",       "syn": ["minip", "min ip"],                                       "policy": "derive"}),
        ("QSM",        {"gen": None,              "syn": ["qsm"],                                                   "policy": "convert_only"}),
    ]),
    "SWI_GAD": OrderedDict([
        ("MIP",        {"gen": "swi_mip",         "syn": ["mip"],                                                   "policy": "derive"}),
        ("MINIP",      {"gen": "swi_minip",       "syn": ["minip", "min ip"],                                       "policy": "derive"}),
        ("QSM",        {"gen": None,              "syn": ["qsm"],                                                   "policy": "convert_only"}),
    ]),
    # PERFUSION family includes both classic param maps and simple time summaries
    "PERFUSION": OrderedDict([
        ("CBV",        {"gen": "perf_cbv",        "syn": ["cbv"],                                                   "policy": "derive"}),
        ("CBF",        {"gen": "perf_cbf",        "syn": ["cbf"],                                                   "policy": "derive"}),
        ("MTT",        {"gen": "perf_mtt",        "syn": ["mtt"],                                                   "policy": "derive"}),
        ("TTP",        {"gen": "perfusion_ttp_index", "syn": ["ttp"],                                               "policy": "derive"}),
        ("TMAX",       {"gen": "perf_tmax",       "syn": ["tmax"],                                                  "policy": "derive"}),
        ("KTRANS",     {"gen": "dce_ktrans",      "syn": ["ktrans", "k trans"],                                     "policy": "derive"}),
        ("KEP",        {"gen": "dce_kep",         "syn": ["kep"],                                                   "policy": "derive"}),
        ("VE",         {"gen": "dce_ve",          "syn": ["ve"],                                                    "policy": "derive"}),
        ("VP",         {"gen": "dce_vp",          "syn": ["vp"],                                                    "policy": "derive"}),
        ("LEAKAGE",    {"gen": "perf_leakage",    "syn": ["leakage"],                                               "policy": "derive"}),
        ("PARAM_MAP",  {"gen": "perf_param_map",  "syn": ["parametric", "param map", "parametric map"],             "policy": "derive"}),
        ("PBP",        {"gen": "perf_pbp",        "syn": ["pbp"],                                                   "policy": "derive"}),
        ("GBP",        {"gen": "perf_gbp",        "syn": ["gbp"],                                                   "policy": "derive"}),
        # Time-series summaries
        ("MEAN",       {"gen": "perfusion_mean_t","syn": ["mean"],                                                  "policy": "derive"}),
        ("MAX",        {"gen": "perfusion_max_t", "syn": ["max"],                                                   "policy": "derive"}),
        ("AUC",        {"gen": "perfusion_auc_t", "syn": ["auc"],                                                   "policy": "derive"}),
    ]),
}

def _dwi_derived_category(tokens: str, imgtypes: set[str]) -> str | None:
    return _match_cat_from_spec(DERIVED_CATEGORY_SPEC["DWI"], tokens.lower(), set(t.upper() for t in imgtypes))

def _swi_derived_category(tokens: str, imgtypes: set[str]) -> str | None:
    return _match_cat_from_spec(DERIVED_CATEGORY_SPEC["SWI"], tokens.lower(), set(t.upper() for t in imgtypes))

def _swi_gad_derived_category(tokens: str, imgtypes: set[str]) -> str | None:
    return _match_cat_from_spec(DERIVED_CATEGORY_SPEC["SWI_GAD"], tokens.lower(), set(t.upper() for t in imgtypes))

def _swi_primary_subtype(tokens: str, imgtypes: set[str]) -> str | None:
    """
    Detect primary SWI subtypes (MAG or PHASE) from series description or ImageType/FrameType.
    Returns "MAG" | "PHASE" | None.
    NOTE: Use word-boundary matching so that "Images" does not falsely match "mag".
    """
    t_raw = tokens if isinstance(tokens, str) else str(tokens)
    # normalize underscores/hyphens -> space; lowercase
    t = re.sub(r"[_\-]+", " ", t_raw).lower()
    iu = set(x.upper() for x in (imgtypes or set()))
    # magnitude: token "mag" or "magnitude", or ImageType says MAGNITUDE
    if re.search(r"(?<![a-z0-9])mag(?![a-z0-9])", t) or \
       re.search(r"(?<![a-z0-9])magnitude(?![a-z0-9])", t) or \
       ("MAGNITUDE" in iu):
        return "MAG"
    # phase: token "phase"/"pha"/"filt pha", or ImageType says PHASE
    if re.search(r"(?<![a-z0-9])phase(?![a-z0-9])", t) or \
       re.search(r"(?<![a-z0-9])pha(?![a-z0-9])", t) or \
       re.search(r"(?<![a-z0-9])filt\s*pha(?![a-z0-9])", t) or \
       ("PHASE" in iu):
        return "PHASE"
    return None

def _perfusion_derived_category(tokens: str, imgtypes: set[str]) -> str | None:
    return _match_cat_from_spec(DERIVED_CATEGORY_SPEC["PERFUSION"], tokens.lower(), set(t.upper() for t in imgtypes))

def _t1_derived_category(tokens: str, imgtypes: set[str]) -> str | None:
    t = tokens
    if "mip" in t: return "MIP"
    if "minip" in t: return "MINIP"
    return None

def _compute_is_derived(tokens_any, imgtypes: set[str], dcat: str | None) -> bool:
    """
    Decide whether a series should be flagged as 'derived'.
    Revised rule:
      1) If ANY ImageType/FrameType keywords are present:
           → return True if a derived keyword is present, OR if we matched a known derived sub-label (dcat).
           → otherwise return False.
      2) If NO ImageType/FrameType keywords are present:
           → fall back to sub-label (dcat), then name-only projection checks (projections → derived).
    """
    # Normalize tokens for name-only fallback checks
    t = " ".join(sorted(tokens_any)).lower() if isinstance(tokens_any, set) else str(tokens_any).lower()

    DERIVED_TOKENS = {
        "DERIVED", "SECONDARY", "REFORMATTED", "RESAMPLED", "MPR",
        "SUBTRACTED", "PROJECTION", "MIP", "MINIP", "AVERAGE", "MINIMUM", "MAXIMUM", "SUBTRACTION",
        # Many vendors include explicit map names in ImageType/FrameType:
        "ADC", "FA", "TRACE", "TRACEW", "DIFFUSION",  # DWI maps
        "CBV", "CBF", "MTT", "TTP", "TMAX", "KTRANS", "KEP", "VE", "VP"  # perfusion maps
    }

    # (1) Metadata present → trust metadata, but allow sub-label to assert "derived"
    if imgtypes:
        if imgtypes & DERIVED_TOKENS:
            return True
        # If classifier already identified a known derived sub-label, honor it.
        try:
            from collections import OrderedDict
            # Build the cross-family set once
            all_cats = set()
            for _fam, _spec in DERIVED_CATEGORY_SPEC.items():
                all_cats.update(str(k).upper() for k in _spec.keys())
        except Exception:
            all_cats = set()
        if dcat and str(dcat).upper() in all_cats:
            return True
        return False

    # (2) No metadata flags at all → allow fallbacks
    if dcat is not None:
        return True
    if _is_projection(t):
        return True
    return False

def _collect_imgtype_flags(ds) -> set[str]:
    """
    Collect 'derived-ness' indicators from all relevant locations:
      - (0008,0008) ImageType
      - (0008,9007) FrameType
      - SharedFunctionalGroupsSequence / MRImageFrameTypeSequence / FrameType
      - PerFrameFunctionalGroupsSequence / MRImageFrameTypeSequence / FrameType (sampled)
    Returns a UPPERCASED set of tokens.
    """
    flags: set[str] = set()

    def _ingest(val):
        if val is None:
            return
        try:
            # val is a pydicom MultiValue or list-like
            for v in list(val):
                if v is None:
                    continue
                s = str(v).strip().upper()
                if s:
                    flags.add(s)
        except Exception:
            # fallback: split a raw string conservatively
            s = str(val)
            for v in re.split(r"[\\,; ]+", s):
                v = v.strip().upper()
                if v:
                    flags.add(v)

    # Classic
    _ingest(getattr(ds, "ImageType", None))
    _ingest(getattr(ds, "FrameType", None))

    # Enhanced: Shared
    try:
        sfg = getattr(ds, "SharedFunctionalGroupsSequence", None)
        if sfg and len(sfg) > 0:
            mrfts = getattr(sfg[0], "MRImageFrameTypeSequence", None)
            if mrfts and len(mrfts) > 0:
                _ingest(getattr(mrfts[0], "FrameType", None))
    except Exception:
        pass

    # Enhanced: Per-Frame (sample some frames for speed)
    try:
        pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
        if pffg:
            for item in pffg[:100]:
                mrfts = getattr(item, "MRImageFrameTypeSequence", None)
                if mrfts and len(mrfts) > 0:
                    _ingest(getattr(mrfts[0], "FrameType", None))
    except Exception:
        pass

    return flags

def _image_type_tokens(ds) -> set[str]:
    """
    Collect ImageType/FrameType-like tokens from both classic and Enhanced MR.
    Ensures Enhanced MR (multi-frame) 'FrameType' / 'Derivation...' info is visible.
    """
    out = set(_to_list_upper(ds.get("ImageType")))

    # Enhanced MR: SharedFunctionalGroups may carry FrameType and derivation info
    try:
        sfg = getattr(ds, "SharedFunctionalGroupsSequence", None)
        if sfg:
            sfg0 = sfg[0]
            mrf = getattr(sfg0, "MRImageFrameTypeSequence", None)
            if mrf:
                out.update(_to_list_upper(mrf[0].get("FrameType")))
            # Presence of derivation sequences is also a derived hint
            if hasattr(sfg0, "DerivationImageSequence") or hasattr(sfg0, "DerivationCodeSequence"):
                out.add("DERIVED")
    except Exception:
        pass

    # Some vendors also put FrameType on the per-frame group; peek at frame 0
    try:
        pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
        if pffg:
            mrf = getattr(pffg[0], "MRImageFrameTypeSequence", None)
            if mrf:
                out.update(_to_list_upper(mrf[0].get("FrameType")))
    except Exception:
        pass

    return out

def _looks_localizer(t: str) -> bool:
    keys = ["localizer","scout","aahead","a ahead","3 plane","3-plane","3plane","loc"]
    return any(k in t for k in keys)

def _localizer_subtype(t: str) -> str | None:
    if re.search(r"\bmpr(_| )?sag\b", t): return "MPR_SAG"
    if re.search(r"\bmpr(_| )?cor\b", t): return "MPR_COR"
    if re.search(r"\bmpr(_| )?tra\b", t): return "MPR_TRA"
    if re.search(r"(?<![a-z])mpr(?!age)\b", t): return "MPR"  # MPR but not MPRAGE
    return None

def _looks_calibration(t: str) -> bool:
    """
    Parallel-imaging and prescan/reference calibrations:
    GE: ASSET/ARC; Philips: SENSE; Siemens: reference scans/prescans.
    We keep tokens conservative to avoid false positives.
    """
    # common forms seen in SeriesDescription / ProtocolName
    hard_tokens = [
        "assetcal", "asset cal", "arc calib", "arc calibration",
        "prescan", "pre-scan", "pre scan",
        "refscan", "ref scan", "reference scan",
        "coil survey", "coil_survey",
    ]
    # Allow “sense” only when paired with calib/ref wording to avoid noise
    if "sense" in t and any(k in t for k in ["cal", "calib", "reference", "ref", "refscan"]):
        return True
    return any(k in t for k in hard_tokens) or ("asset" in t and "cal" in t)

def _looks_fieldmap(t: str) -> bool:
    """
    Field mapping (B0/phase/gre field map) sequences should never fall through to T1/T2 physics.
    """
    keys = [
        "fieldmap", "field map", "fmap",
        "b0map", "b0 map", "b0 ",
        "phase map", "phasediff", "phase-diff",
        "topup", "gre field map", "dual echo field map",
    ]
    return any(k in t for k in keys)

def _safe_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        return None

def _count_dicoms(series_folder: str) -> int:
    cnt = 0
    for root, _, files in os.walk(series_folder):
        cnt += sum(f.lower().endswith(".dcm") for f in files)
        break  # don't recurse
    return cnt

def _get_pixel_spacing(ds):
    """Return (row_mm, col_mm) from classic or Enhanced MR. None if missing.
    Tries classic PixelSpacing, then SharedFunctionalGroups, then Per-Frame median,
    and finally ImagerPixelSpacing.
    """
    # Classic (non-enhanced)
    ps = getattr(ds, "PixelSpacing", None)
    if ps:
        try:
            return float(ps[0]), float(ps[1])
        except Exception:
            pass

    # Enhanced MR: Shared PixelMeasures
    try:
        sfg = ds.SharedFunctionalGroupsSequence[0]
        pms = sfg.PixelMeasuresSequence[0]
        ps = pms.PixelSpacing
        return float(ps[0]), float(ps[1])
    except Exception:
        pass

    # Enhanced MR: Per-Frame PixelMeasures (median across frames)
    try:
        pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
        if pffg:
            rows, cols = [], []
            for item in pffg[:100]:
                pms = getattr(item, "PixelMeasuresSequence", None)
                if not pms:
                    continue
                ps = getattr(pms[0], "PixelSpacing", None)
                if ps:
                    try:
                        rows.append(float(ps[0])); cols.append(float(ps[1]))
                    except Exception:
                        continue
            if rows and cols:
                import statistics as _stats
                return _stats.median(rows), _stats.median(cols)
    except Exception:
        pass

    # Fallback: ImagerPixelSpacing
    try:
        ips = getattr(ds, "ImagerPixelSpacing", None)
        if ips:
            return float(ips[0]), float(ips[1])
    except Exception:
        pass

    return None, None

def _get_slice_metrics(ds):
    """Return (slice_thickness_mm, spacing_between_slices_mm, number_of_frames)
    with best-effort fallbacks: classic → Shared → Per-Frame.
    """
    st = _safe_float(getattr(ds, "SliceThickness", None))
    sbs = _safe_float(getattr(ds, "SpacingBetweenSlices", None))
    nof = _safe_int(getattr(ds, "NumberOfFrames", None))

    # Enhanced MR: Shared PixelMeasures
    if st is None or sbs is None:
        try:
            pms = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            if st is None:
                st = _safe_float(getattr(pms, "SliceThickness", None))
            if sbs is None:
                sbs = _safe_float(getattr(pms, "SpacingBetweenSlices", None))
        except Exception:
            pass

    # Enhanced MR: Per-Frame PixelMeasures (median across frames)
    if st is None or sbs is None:
        try:
            pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
            if pffg:
                st_list, sbs_list = [], []
                for item in pffg[:200]:
                    pms = getattr(item, "PixelMeasuresSequence", None)
                    if not pms:
                        continue
                    st_i = _safe_float(getattr(pms[0], "SliceThickness", None))
                    sbs_i = _safe_float(getattr(pms[0], "SpacingBetweenSlices", None))
                    if st_i is not None: st_list.append(st_i)
                    if sbs_i is not None: sbs_list.append(sbs_i)
                import statistics as _stats
                if st is None and st_list:
                    st = _stats.median(st_list)
                if sbs is None and sbs_list:
                    sbs = _stats.median(sbs_list)
        except Exception:
            pass

    return st, sbs, nof


def _iter_dicom_paths(series_folder: str, max_files: Optional[int] = None):
    """Yield full paths to DICOM files in a series folder (non-recursive)."""
    count = 0
    for fname in os.listdir(series_folder):
        if fname.lower().endswith(".dcm"):
            yield os.path.join(series_folder, fname)
            count += 1
            if max_files is not None and count >= max_files:
                break

def _get_imager_pixel_spacing(ds) -> Tuple[Optional[float], Optional[float]]:
    """Optional alternative to PixelSpacing: (0018,1164) ImagerPixelSpacing."""
    ips = getattr(ds, "ImagerPixelSpacing", None)
    if ips:
        try:
            return float(ips[0]), float(ips[1])
        except Exception:
            return None, None
    return None, None

def _resolve_series_pixel_spacing(series_folder: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Scan multiple files in the series to find a reliable in-plane spacing.
    Priority:
      1) (0028,0030) PixelSpacing
      2) Enhanced MR SharedFunctionalGroups -> PixelMeasures -> PixelSpacing
      3) (0018,1164) ImagerPixelSpacing
    Returns (row_mm, col_mm, source).
    """
    # Fast path: try the first file (most sites are consistent)
    first = _first_dcm_in(series_folder)
    if first:
        try:
            ds0 = pydicom.dcmread(first, stop_before_pixels=True, force=True)
            r, c = _get_pixel_spacing(ds0)
            if r and c:
                return r, c, "PixelSpacing:first"
            r2, c2 = _get_imager_pixel_spacing(ds0)
            if r2 and c2:
                return r2, c2, "ImagerPixelSpacing:first"
        except Exception:
            pass

    # Otherwise scan a handful of files and take the median of hits
    row_vals, col_vals = [], []
    found_via = None
    for p in _iter_dicom_paths(series_folder, max_files=50):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception:
            continue
        r, c = _get_pixel_spacing(ds)
        via = "PixelSpacing"
        if not (r and c):
            r, c = _get_imager_pixel_spacing(ds)
            via = "ImagerPixelSpacing" if (r and c) else None
        if r and c:
            row_vals.append(r); col_vals.append(c)
            found_via = via

    if row_vals and col_vals:
        import statistics as _stats
        try:
            return _stats.median(row_vals), _stats.median(col_vals), f"{found_via}:median"
        except Exception:
            return row_vals[0], col_vals[0], f"{found_via}:first-hit"

    return None, None, None

def _estimate_z_spacing_from_positions(series_folder: str) -> Tuple[Optional[float], int]:
    """
    Estimate through-plane spacing from ImagePositionPatient + ImageOrientationPatient.
    We read multiple instances, project position differences onto the slice normal,
    and return the median absolute distance. Returns (z_mm, n_positions).
    """
    positions = []
    normals = []
    for p in _iter_dicom_paths(series_folder, max_files=200):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception:
            continue

        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if ipp is None or iop is None or len(iop) < 6:
            continue

        positions.append(np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=float))

        # Derive slice normal from row/col direction cosines
        row = np.array(iop[:3], dtype=float)
        col = np.array(iop[3:6], dtype=float)
        n = np.cross(row, col)
        norm = np.linalg.norm(n)
        if norm > 0:
            normals.append(n / norm)

    if len(positions) < 2 or not normals:
        return None, 0

    # Use average normal (robust enough for typical series)
    n_avg = np.mean(normals, axis=0)
    n_norm = np.linalg.norm(n_avg)
    if n_norm == 0:
        return None, len(positions)

    n = n_avg / n_norm
    # Sort by projection along normal; nearest-neighbor distances
    positions_sorted = sorted(positions, key=lambda v: np.dot(v, n))
    diffs = []
    for a, b in zip(positions_sorted[:-1], positions_sorted[1:]):
        d = abs(np.dot((b - a), n))
        if d > 0:
            diffs.append(d)

    if not diffs:
        return None, len(positions_sorted)

    import statistics as _stats
    return float(_stats.median(diffs)), len(positions_sorted)

# ------------------------------------------------------------------------
# Primary function for DICOM file classification
# ------------------------------------------------------------------------

def _classify_all_series_once(exam_dir, mr_subdir="MR", verbose=False):
    """
    Unified 'read-everything-and-classify' in one function.

    Returns: pandas.DataFrame with one row per series folder:
      ['folder','series_number','acq_dt','acq_dt_iso','manufacturer','modality',
       'series_description','protocol_name','sequence_name','image_type',
       'te','tr','ti','flip_angle','b_value','primary_secondary','is_derived','is_fspgr',
       'base_type','final_label','is_postcontrast','is_flair','reason','confidence']
    """
    mr_dir = os.path.join(exam_dir, mr_subdir)
    if not os.path.isdir(mr_dir):
        raise FileNotFoundError(f"MR folder not found: {mr_dir}")

    rows = []
    # 1) Read minimal metadata for EVERY series (1 file per series)
    for series_folder in sorted([os.path.join(mr_dir, d) for d in os.listdir(mr_dir) if os.path.isdir(os.path.join(mr_dir, d))]):
        dcm_path = _first_dcm_in(series_folder)
        if not dcm_path:
            if verbose: print(f"[skip] no DICOM in {series_folder}")
            continue
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
        except Exception as e:
            if verbose: print(f"[warn] failed to read {dcm_path}: {e}")
            continue

        series_number = getattr(ds, "SeriesNumber", None)
        series_desc   = _nz(getattr(ds, "SeriesDescription", None))
        protocol_name = _nz(getattr(ds, "ProtocolName", None))
        sequence_name = _nz(getattr(ds, "SequenceName", None))
        study_desc    = _nz(getattr(ds, "StudyDescription", None))
        procstep_desc = _nz(getattr(ds, "PerformedProcedureStepDescription", None)) or \
                        _nz(getattr(ds, "ProcedureStepDescription", None))
        manufacturer  = _nz(getattr(ds, "Manufacturer", None))
        modality      = _nz(getattr(ds, "Modality", None))
        image_type    = _to_list_upper(getattr(ds, "ImageType", []))
        imgtype_flags = _collect_imgtype_flags(ds)
        acq_dt, acq_iso = _parse_dt(ds)

        te = _safe_float(getattr(ds, "EchoTime", None))
        tr = _safe_float(getattr(ds, "RepetitionTime", None))
        ti = _safe_float(getattr(ds, "InversionTime", None))
        fa = _safe_float(getattr(ds, "FlipAngle", None))

        rows_px = _safe_int(getattr(ds, "Rows", None))
        cols_px = _safe_int(getattr(ds, "Columns", None))
        ps_row_mm, ps_col_mm = _get_pixel_spacing(ds)
        st_mm, sbs_mm, num_frames = _get_slice_metrics(ds)
        z_mm = sbs_mm or st_mm  # best guess for through-plane spacing

        # --- Fallbacks when first-instance metadata isn't sufficient ---
        # In-plane spacing: scan multiple instances if needed
        if (ps_row_mm is None) or (ps_col_mm is None):
            ps_r_f, ps_c_f, _src = _resolve_series_pixel_spacing(series_folder)
            ps_row_mm = ps_row_mm if ps_row_mm is not None else ps_r_f
            ps_col_mm = ps_col_mm if ps_col_mm is not None else ps_c_f

        # Through-plane spacing: derive from positions if missing
        if z_mm is None:
            z_est, _npos = _estimate_z_spacing_from_positions(series_folder)
            if z_est is not None:
                z_mm = z_est

        images_in_acq = _safe_int(getattr(ds, "ImagesInAcquisition", None))
        loc_in_acq = _safe_int(getattr(ds, "LocationsInAcquisition", None))
        num_dicoms = _count_dicoms(series_folder)

        # prefer explicit counts (Enhanced→NumberOfFrames, then ImagesInAcquisition, then Locations, then file count)
        n_slices_est = num_frames or images_in_acq or loc_in_acq or num_dicoms

        # For multiframe (4D) series, estimate slices PER 3D VOLUME by counting unique slice positions.
        # This prevents 4D series (e.g., DWI/Perfusion) from inflating n_slices via total frames.
        n_slices_per_vol_est = None
        try:
            if num_frames and hasattr(ds, "PerFrameFunctionalGroupsSequence") and ds.PerFrameFunctionalGroupsSequence:
                # Orientation (row/col) → slice normal
                iop = None
                try:
                    sfg = getattr(ds, "SharedFunctionalGroupsSequence", None)
                    if sfg:
                        pos = getattr(sfg[0], "PlaneOrientationSequence", None)
                        if pos:
                            iop = getattr(pos[0], "ImageOrientationPatient", None)
                except Exception:
                    pass
                if iop is None:
                    iop = getattr(ds, "ImageOrientationPatient", None)

                n_vec = None
                if iop and len(iop) >= 6:
                    _row = np.array(iop[:3], dtype=float)
                    _col = np.array(iop[3:6], dtype=float)
                    _n = np.cross(_row, _col)
                    _norm = float(np.linalg.norm(_n))
                    if _norm > 0:
                        n_vec = (_n / _norm)

                # Project frame positions onto the slice normal (or fallback to z)
                vals = []
                # Cap frames inspected to keep header-only iteration light
                _max = int(num_frames) if int(num_frames) < 1500 else 1500
                for item in ds.PerFrameFunctionalGroupsSequence[:_max]:
                    ipp = None
                    try:
                        pps = getattr(item, "PlanePositionSequence", None)
                        if pps:
                            ipp = getattr(pps[0], "ImagePositionPatient", None)
                    except Exception:
                        ipp = None
                    if ipp is None:
                        continue
                    try:
                        if n_vec is not None:
                            ip = np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=float)
                            vals.append(float(np.dot(ip, n_vec)))
                        else:
                            vals.append(float(ipp[2]))
                    except Exception:
                        continue
                if vals:
                    # Bin with a tolerance to merge nearly identical positions
                    bin_mm = float(z_mm) / 2.0 if (z_mm is not None and float(z_mm) > 0) else 0.2
                    if bin_mm <= 0:
                        bin_mm = 0.2
                    arr = np.asarray(vals, dtype=float)
                    n_slices_per_vol_est = int(np.unique(np.round(arr / bin_mm)).size)
        except Exception:
            n_slices_per_vol_est = None

        # Try common vendor B-value locations (not guaranteed)
        bval = None
        for tag in [("DiffusionBValue",), ("Private_0019_100c",), ("Private_0043_1039",)]:
            try:
                if hasattr(ds, tag[0]):
                    bval = _safe_float(getattr(ds, tag[0]))
                    break
            except Exception:
                pass

        name_combo = _norm_text(series_desc, protocol_name, sequence_name)
        tokens = _name_tokens(name_combo)

        # Vendor hints + robust B-value parsing
        vh = _vendor_hints(ds)
        if vh.get("b_value") is not None:
            bval = vh["b_value"]  # prefer vendor-parsed value

        primary_secondary = "PRIMARY" if "PRIMARY" in image_type else ("SECONDARY" if "SECONDARY" in image_type else None)
        # NOTE: do NOT compute is_derived here; we’ll do it during classification for consistency.
        is_fspgr = _detect_fspgr(tokens, sequence_name, protocol_name)
        plane = _detect_plane(tokens)
        if plane is None:
            plane = _plane_from_iop(ds)

        rows.append(dict(
            folder=series_folder,
            series_number=series_number,
            acq_dt=acq_dt, acq_dt_iso=acq_iso,
            manufacturer=manufacturer, modality=modality,
            series_description=series_desc, protocol_name=protocol_name, sequence_name=sequence_name,
            study_description=study_desc, procedure_step_description=procstep_desc,
            image_type=";".join(image_type),
            imgtype_flags=";".join(sorted(list(imgtype_flags))) if imgtype_flags else "",
            te=te, tr=tr, ti=ti, flip_angle=fa, b_value=bval,
            primary_secondary=primary_secondary,
            # is_derived -> computed later
            is_fspgr=is_fspgr,
            plane=plane,
            rows_px=rows_px, cols_px=cols_px,
            pixdim_row_mm=ps_row_mm, pixdim_col_mm=ps_col_mm,
            slice_thickness_mm=st_mm, spacing_between_slices_mm=sbs_mm,
            z_spacing_mm=z_mm,
            num_frames=num_frames, images_in_acq=images_in_acq, locations_in_acq=loc_in_acq,
            num_dicoms=num_dicoms, n_slices_est=n_slices_est, n_slices_per_vol_est=n_slices_per_vol_est,
            # vendor/context fields (debuggable and optional)
            pulse_sequence_name = vh.get("pulse_sequence_name"),
            scanning_sequence   = vh.get("scanning_sequence"),
            sequence_variant    = vh.get("sequence_variant"),
            scan_options        = vh.get("scan_options"),
            mr_acq_type         = vh.get("mr_acq_type"),
            contrast_agent      = vh.get("contrast_agent"),
            contrast_volume     = vh.get("contrast_volume"),
            acquisition_contrast= vh.get("acquisition_contrast"),
            _tokens=tokens
        ))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 2) Compute exam-level context ONCE (e.g., inferred contrast time & non-contrast flag)
    def _looks_noncontrast(tok: str) -> bool:
        # conservative but inclusive: capture common "no contrast" and "pre-contrast" phrasing
        return any(k in tok for k in [
            "w/o contrast","wo contrast","without contrast","no contrast",
            "non-contrast","noncontrast","precontrast","pre contrast","pre-contrast",
            "without gad","no gad","no gado"
        ])

    # Build an exam-level non-contrast hint from Study/Procedure descriptions
    exam_noncontrast = False
    try:
        import pandas as _pd
        pool_text = _norm_text(
            " ".join(df.get("study_description", _pd.Series(dtype=str)).dropna().astype(str).tolist()),
            " ".join(df.get("procedure_step_description", _pd.Series(dtype=str)).dropna().astype(str).tolist()),
        )
        exam_noncontrast = _looks_noncontrast(_name_tokens(pool_text))
    except Exception:
        exam_noncontrast = False

    # Perfusion or explicit post-contrast cues are our best proxy.
    def _looks_perfusion(tok):
        return any(k in tok for k in ["perfusion","pwi","dsc","dce","asl","pcasl"])
    # Ignore generic "with contrast" tokens when the exam is explicitly non-contrast
    _allow_generic_with = not exam_noncontrast
    def _looks_post(tok):
        base = any(k in tok for k in ["post","c+","gad","gadolinium","t1c"])
        if _allow_generic_with:
            base = base or any(k in tok for k in ["with contrast","w/ contrast"])
        return base
    def _looks_t1(tok):
        return ("t1" in tok) or ("mprage" in tok) or ("bravo" in tok) or ("spgr" in tok) or ("vibe" in tok)
    # earliest perfusion OR earliest explicit post-contrast cue OR earliest T1 with post-y tokens
    perf_times = [r["acq_dt"] for r in rows if r["acq_dt"] and _looks_perfusion(r["_tokens"])]
    explicit_post_times = [r["acq_dt"] for r in rows if r["acq_dt"] and _looks_post(r["_tokens"])]
    # (Optional) consider earliest 'T1c' guess—but we're single-pass; keep signal minimal here.
    inferred_contrast_time = None
    for pool in [perf_times, explicit_post_times]:
        if pool:
            inferred_contrast_time = min(pool)
            break

    # 3) Classify each series ONCE using local + global context
    base_types, final_labels = [], []
    post_flags, flair_flags = [], []
    reasons, confidences = [], []
    derived_flags = []

    for _, r in df.iterrows():
        t = r["_tokens"]
        te = r["te"] or 0.0
        tr = r["tr"] or 0.0
        imgtypes = set(r["image_type"].split(";")) if r["image_type"] else set()
        # Use the UNION of classic ImageType and collected FrameType/Enhanced flags
        imgtypes = set()
        if r.get("image_type"):
            imgtypes.update(x for x in r["image_type"].split(";") if x)
        if r.get("imgtype_flags"):
            imgtypes.update(x for x in r["imgtype_flags"].split(";") if x)
        acq_dt = r["acq_dt"]

        # vendor/context fields
        contrast_agent = (r.get("contrast_agent") or "") or ""
        acquisition_contrast = (r.get("acquisition_contrast") or "") or ""
        b_value = r.get("b_value", None)

        reason = []
        base = None
        label = None
        is_post = False
        is_flair = False
        conf = 0.5

        # record what we saw (great for debugging audits)
        if imgtypes:
            reason.append(f"ImageType={','.join(sorted(imgtypes))}")
        elif r.get("image_type"):  # rare: classic present but empty flags after union
            reason.append(f"ImageType={r['image_type']}")
        if r.get("plane"):
            reason.append(f"plane={r['plane']}")
        if r.get("pulse_sequence_name"):
            reason.append(f"PulseSequenceName={r['pulse_sequence_name']}")
        if b_value is not None:
            reason.append(f"b_value={b_value}")

        # --- Strong families first: Localizer, DWI, SWI, Perfusion ---
        dcat_hint = _dwi_derived_category(t, imgtypes)
        # Treat DWI as a strong family: name, b-value, vendor hints, or ImageType flags can light it up.
        imgtype_has_diff = any((x or "").upper() == "DIFFUSION" for x in imgtypes)
        dwi_hit = (
            (dcat_hint is not None)
            or any(k in t for k in ["dwi", "diff", "ep2d", "ep_b", "trace w", "trace"])
            or (b_value is not None and b_value > 0)
            or imgtype_has_diff
        )
        if not dwi_hit:
            vh = _vendor_hints(ds)
            psn = str(vh.get("pulse_sequence_name","") or "").lower()
            ss  = str(vh.get("scanning_sequence","") or "").lower()
            sv  = str(vh.get("sequence_variant","") or "").lower()
            so  = str(vh.get("scan_options","") or "").lower()
            if any(s in psn for s in ["ep2d", "diff", "dti"]) or \
               ("diff" in ss) or ("diff" in sv) or ("diff" in so):
                dwi_hit = True

        if _looks_localizer(t):
            base = "Localizer"
            lcat = _localizer_subtype(t)
            label = "Localizer" if lcat is None else f"Localizer({lcat})"
            is_derived = _compute_is_derived(t, imgtypes, lcat)  # MPR variants → derived True
            reason.append(f"Localizer family; subtype={lcat}")
            conf = 0.95 if lcat else 0.9

        elif dwi_hit:
            base = "DWI"
            dcat = dcat_hint or _dwi_derived_category(t, imgtypes)  # keep
            label = "DWI" if dcat is None else f"DWI({dcat})"
            is_derived = _compute_is_derived(t, imgtypes, dcat)
            reason.append(f"DWI family; derived={is_derived}; dcat={dcat}")
            conf = 0.9 if dcat is None else 0.95

        elif any(k in t for k in ["swi","swan","suscept","venogr"]):
            # Decide between SWI vs SWI_GAD using tokens or vendor/Enhanced hints
            is_gad = ("gad" in t) or (str(acquisition_contrast).upper() == "CONTRAST") or bool(contrast_agent)
            base = "SWI_GAD" if is_gad else "SWI"
            if "swan" in t: reason.append("SWI subtype=SWAN")
            # First: detect primary subtypes MAG/PHASE
            dsub_tokens = _swi_primary_subtype(t, set())
            dsub_imgtyp = _swi_primary_subtype("", imgtypes)
            primary_sub = dsub_tokens or dsub_imgtyp
            # Then: detect true derived categories (e.g., MIP/MINIP)
            if base == "SWI_GAD":
                dcat_tokens = _swi_gad_derived_category(t, set())
                dcat_imgtyp = _swi_gad_derived_category("", imgtypes)
                is_post = True  # SWI_GAD is, by definition, post-contrast
            else:
                dcat_tokens = _swi_derived_category(t, set())
                dcat_imgtyp = _swi_derived_category("", imgtypes)
            # If name is generic SWI/SWAN (no subtype or derived words), keep label at SWI
            generic_swi_name = (("swi" in t or "swan" in t) and not any(w in t for w in ["pha","phase","mag","magnitude","min ip","minip","mip"]))
            if primary_sub:
                label = f"{base}({primary_sub})"
                is_derived = False
                dcat = None
            else:
                dcat = None if generic_swi_name else (dcat_tokens or dcat_imgtyp)
                label = base if dcat is None else f"{base}({dcat})"
                is_derived = _compute_is_derived(t, imgtypes, dcat)
            reason.append(f"{base} family; derived={is_derived}; dcat={dcat}")
            conf = 0.9 if (primary_sub or dcat is None) else 0.95

        elif any(k in t for k in ["perfusion","pwi","dsc","dce","asl","pcasl"]):
            base = "Perfusion"
            dcat = _perfusion_derived_category(t, imgtypes)
            label = "Perfusion" if dcat is None else f"Perfusion({dcat})"
            is_derived = _compute_is_derived(t, imgtypes, dcat)
            reason.append(f"Perfusion family; derived={is_derived}; dcat={dcat}")
            conf = 0.85 if dcat is None else 0.9

        # --- FLAIR ---
        elif "flair" in t:
            base = "T2f"
            label = "T2f"
            is_flair = True
            is_derived = _compute_is_derived(t, imgtypes, None)  # usually False
            reason.append("FLAIR in name")
            conf = 0.9

        # --- Calibration / Fieldmap families (avoid physics fallthrough) ---
        elif _looks_fieldmap(t):
            base = "FMAP"
            label = "FieldMap"
            is_derived = _compute_is_derived(t, imgtypes, None)
            reason.append("Fieldmap tokens")
            conf = 0.9

        elif _looks_calibration(t):
            base = "Calibration"
            label = "Calibration"
            is_derived = _compute_is_derived(t, imgtypes, None)
            reason.append("Calibration tokens (ASSET/SENSE/ARC/Prescan/RefScan)")
            conf = 0.95

        # --- T2 vs T1 (physics + names) ---
        else:
            if "t2" in t:
                base = "T2w"
                label = "T2w"
                is_derived = _compute_is_derived(t, imgtypes, None)
                reason.append("T2 token")
                conf = 0.8

            elif ("t1" in t) or any(k in t for k in ["mprage","bravo","spgr","vibe"]):
                base = "T1"
                # decide pre/post from explicit vendor/contrast cues or timing
                post_hint = _looks_post(t)
                noncontrast_hint = _looks_noncontrast(t)
                vendor_post = bool(contrast_agent) or ("CONTRAST" in str(acquisition_contrast).upper())
                # explicit non-contrast on the series overrides generic post tokens
                if noncontrast_hint:
                    is_post = False
                    reason.append("explicit non-contrast tokens")
                elif post_hint:
                    is_post = True
                    reason.append("explicit post-contrast tokens")
                elif vendor_post:
                    is_post = True
                    reason.append(f"vendor contrast present (agent='{contrast_agent}' or AcquisitionContrast)")
                elif inferred_contrast_time and _is_after(acq_dt, inferred_contrast_time):
                    is_post = True
                    reason.append("acquired after inferred contrast time")

                dcat = _t1_derived_category(t, imgtypes)

                core = "T1c" if is_post else "T1n"
                label = core if dcat is None else f"{core}({dcat})"
                is_derived = _compute_is_derived(t, imgtypes, dcat)
                # FSPGR is a note, not derivation
                if r["is_fspgr"]:
                    reason.append("FSPGR-like sequence")
                    conf = max(conf, 0.85)
                conf = max(conf, 0.8 if is_post else 0.7)

            else:
                # physics thresholds as fallback
                if te and tr and te >= 80 and tr >= 2000:
                    base = "T2w"
                    label = "T2w"
                    is_derived = _compute_is_derived(t, imgtypes, None)
                    reason.append(f"TE/TR suggest T2 (TE={te}, TR={tr})")
                    conf = 0.7
                elif te and tr and te <= 20 and tr <= 1000:
                    # Guardrail: only allow T1 physics fallback if it *also* looks like T1
                    # by name/vendor tokens OR has enough slices to be plausible anatomy.
                    looks_t1_name = ("t1" in t) or any(k in t for k in ["mprage","bravo","spgr","vibe"])
                    enough_slices = False
                    try:
                        enough_slices = (int(r.get("n_slices_est") or 0) >= 8)
                    except Exception:
                        pass
                    if looks_t1_name or enough_slices:
                        base = "T1"
                        is_post = False
                        dcat = None
                        label = "T1n"
                        is_derived = _compute_is_derived(t, imgtypes, dcat)
                        reason.append(f"TE/TR suggest T1 (TE={te}, TR={tr}); gate ok (name={looks_t1_name}, slices≥8={enough_slices})")
                        conf = 0.65
                    else:
                        base = None
                        label = "Unknown"
                        is_derived = _compute_is_derived(t, imgtypes, None)
                        reason.append("TE/TR suggest T1 but rejected by gate (no T1 tokens, few slices) → Unknown")
                        conf = 0.3
                else:
                    base = None
                    label = "Unknown"
                    is_derived = _compute_is_derived(t, imgtypes, None)
                    reason.append("No strong name/physics cues")
                    conf = 0.3

        # Upgrade T1n -> T1c when timing strongly indicates post-contrast
        if base == "T1" and label.startswith("T1n") and inferred_contrast_time and _is_after(acq_dt, inferred_contrast_time):
            label = label.replace("T1n", "T1c")
            is_post = True
            reason.append("timing upgrade to post-contrast")

        base_types.append(base)
        final_labels.append(label)
        post_flags.append(bool(is_post))
        flair_flags.append(bool(is_flair))
        reasons.append("; ".join(reason))
        confidences.append(min(0.99, max(0.0, conf)))
        derived_flags.append(bool(is_derived))

    df["base_type"] = base_types
    df["final_label"] = final_labels
    df["is_postcontrast"] = post_flags
    df["is_flair"] = flair_flags
    df["reason"] = reasons
    df["confidence"] = confidences
    df["is_derived"] = derived_flags

    # Clean / order
    df = df.drop(columns=["_tokens"]).sort_values(
        by=["series_number","acq_dt"], ascending=[True, True], na_position="last"
    ).reset_index(drop=True)

    # Normalize Unknown + derived
    df.loc[df["final_label"].eq("Unknown") & df["is_derived"], "final_label"] = "Unknown-derived"

    # Pretty fields for quick sorting/reading
    df["matrix"] = df.apply(lambda r: f"{_safe_int(r.cols_px) or '?'}x{_safe_int(r.rows_px) or '?'}", axis=1)
    def _voxel_str(r):
        a, b, c = r.pixdim_row_mm, r.pixdim_col_mm, r.z_spacing_mm
        return None if (a is None or b is None) else (f"{a:.2f}x{b:.2f}" + (f"x{c:.2f}" if c else ""))
    df["voxel_mm"] = df.apply(_voxel_str, axis=1)
    # Use per-volume slice count for true 4D (multiframe) series when available; else fall back.
    df["n_slices"] = df["n_slices_est"]
    try:
        # Avoid FutureWarning by coercing to numeric before fillna/casting
        num_frames = pd.to_numeric(df.get("num_frames", pd.Series(index=df.index)), errors="coerce")
        is4d = num_frames.fillna(0).astype("int64") > 0
        pervol = pd.to_numeric(df.get("n_slices_per_vol_est", pd.Series(index=df.index)), errors="coerce")
        has_pervol = pervol.notna() & (pervol > 0)
        mask = is4d & has_pervol
        if mask.any():
            df.loc[mask, "n_slices"] = pervol.loc[mask]
    except Exception:
        pass

    # Post-process: inherit in-plane spacing for derived series missing spacing
    try:
        mask = (df['is_derived'] == True) & ((df['pixdim_row_mm'].isna()) | (df['pixdim_col_mm'].isna()))
        if mask.any():
            for idx, r in df[mask].iterrows():
                candidates = df[(df['base_type'] == r['base_type']) & (df['is_derived'] == False)]
                if not candidates.empty:
                    # nearest in time
                    candidates = candidates.copy()
                    candidates['tdiff'] = (candidates['acq_dt'] - r['acq_dt']).abs()
                    parent = candidates.sort_values('tdiff').iloc[0]
                    a, b = parent['pixdim_row_mm'], parent['pixdim_col_mm']
                    if _safe_float(a) and _safe_float(b):
                        if _safe_float(r['pixdim_row_mm']) is None:
                            df.at[idx, 'pixdim_row_mm'] = float(a)
                        if _safe_float(r['pixdim_col_mm']) is None:
                            df.at[idx, 'pixdim_col_mm'] = float(b)
    except Exception:
        pass

    # Consistency: avoid contradictory reason text
    if 'reason' in df.columns:
        def _fix_reason(r):
            rs = r.get('reason', [])
            if isinstance(rs, list):
                cleaned = []
                for x in rs:
                    s = str(x)
                    # Strip any lingering RFMT/MPR chatter for non-Localizer series
                    if re.search(r"\brfmt\b", s, re.I):
                        continue
                    if re.search(r"\bmpr(_?(sag|cor|tra))?\b", s, re.I):
                        # Keep Localizer(MPR_*) messages only
                        if "Localizer" not in r.get("final_label", ""):
                            continue
                    if "treat as primary" in s.lower():
                        continue
                    cleaned.append(s)
                rs = cleaned
            return rs
        try:
            df['reason'] = df.apply(_fix_reason, axis=1)
        except Exception:
            pass

    return df


def classify_exam_series(exam_dir, mr_subdir="MR", verbose=False):
    """
    PUBLIC API: one-call metadata extraction + classification.
    """
    return _classify_all_series_once(exam_dir, mr_subdir=mr_subdir, verbose=verbose)

# ------------------------------------------------------------------------
# Helper functions for creating patient Metadata tables
# ------------------------------------------------------------------------

def _first_dicom_in(folder: str) -> str | None:
    """Return path to the first readable DICOM file in a folder, else None."""
    # Prefer *.dcm; fallback: try first few files
    try:
        files = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    except Exception:
        return None
    # Try .dcm files first
    for fn in files:
        if fn.lower().endswith(".dcm"):
            p = os.path.join(folder, fn)
            if _safe_dcmread(p) is not None:
                return p
    # Fallback: probe up to 20 files
    for fn in files[:20]:
        p = os.path.join(folder, fn)
        if _safe_dcmread(p) is not None:
            return p
    return None

def _safe_dcmread(path: str):
    # Lazy import ensures utils can load even if pydicom isn't installed
    global pydicom
    if pydicom is None:
        try:
            import pydicom as _p
            pydicom = _p
        except Exception:
            return None
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None

def _get_attr(ds, key: str) -> str:
    try:
        val = getattr(ds, key, "")
        return str(val) if val is not None else ""
    except Exception:
        return ""

def _clean_lower(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def _read_table(path: str):
    """
    Robust reader that preserves string-ish identifiers (e.g., '00123') by
    forcing string dtype on input. This avoids Pandas' numeric inference
    from stripping leading zeros.
    """
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if ext in (".xlsx", ".xls"):
        # dtype=str keeps leading zeros (requires pandas>=1.5); fillna("") for parity with CSV branch
        df = pd.read_excel(path, dtype=str)
        return df.fillna("")
    raise ValueError(f"Unsupported previousMetadata extension: {ext}")

def _save_table(df, out_path: str):
    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".csv":
        df.to_csv(out_path, index=False)
    elif ext in (".tsv", ".txt"):
        df.to_csv(out_path, sep="\t", index=False)
    elif ext in (".xlsx", ".xls"):
        # lazy import of engine
        import xlsxwriter  # noqa: F401
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
            df.to_excel(xw, index=False, sheet_name="metadata")
    else:
        raise ValueError(f"Unsupported metadataOut extension: {ext}")

def _normalize_patient_name(name: str) -> str:
    """
    Replace punctuation characters (EXCEPT apostrophes) with a space, collapse repeated spaces,
    and lowercase. Keeps alphanumerics and whitespace and apostrophes only.
    """
    if not name:
        return ""
    s = str(name)
    # turn common DICOM caret separators into spaces up front
    s = s.replace("^", " ")
    # replace any char that is NOT [A-Za-z0-9], whitespace, or apostrophe with a space
    s = re.sub(r"[^A-Za-z0-9\s']", " ", s)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

# ------------------------------------------------------------------------
# Helper functions for demix dicoms
# ------------------------------------------------------------------------
def _safe_int_like(s: str | None):
    try:
        return int(str(s).strip())
    except Exception:
        return None

def _propose_series_dirname(series_no, series_desc: str | None, series_uid: str | None) -> str:
    """
    Build a clean folder name like '3_Ax_DWI_abc123' with punctuation normalized.
    """
    s_no = series_no if (series_no is not None) else 0
    desc = (series_desc or "").strip()
    desc = desc if desc else "Series"
    # normalize punctuation -> space (keep basic ASCII letters/digits/_-)
    safe = re.sub(r"[^\w\-]+", " ", desc)
    safe = re.sub(r"\s+", " ", safe).strip().replace(" ", "_")
    # Use a short, stable hash of the FULL UID (better uniqueness than last 6 chars)
    uid6 = None
    if series_uid:
        try:
            uid6 = hashlib.sha1(series_uid.encode("utf-8")).hexdigest()[:6]
        except Exception:
            uid6 = None
    return f"{s_no}_{safe}{('_' + uid6) if uid6 else ''}"

def _avoid_name_collision(dst_path: str) -> str:
    if not os.path.exists(dst_path):
        return dst_path
    base, ext = os.path.splitext(dst_path)
    k = 2
    while True:
        cand = f"{base}-{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

def _sha1_file(path: str, bufsize: int = 2 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _files_identical(a: str, b: str) -> bool:
    try:
        sa, sb = os.stat(a).st_size, os.stat(b).st_size
        if sa != sb:
            return False
    except Exception:
        return False
    try:
        return _sha1_file(a) == _sha1_file(b)
    except Exception:
        return False

# ============================================================
# Derived-scan planning and generation helpers
# ============================================================

def enumerate_supported_derivatives(base_type: str, only_with_registered_generator: bool = False) -> List[Tuple[str,str]]:
    """
    Return [(final_label, generator_key)] for this base_type using DERIVED_CATEGORY_SPEC.
    Labels match classifier style, e.g. "DWI(ADC)", "SWI(MIP)", "Perfusion(TTP)".
    If only_with_registered_generator=True, include only categories with a registered generator.
    """
    bt = (base_type or "").strip().upper()
    spec_key = "PERFUSION" if bt in ("PERFUSION","DSC","DCE") else bt
    spec = DERIVED_CATEGORY_SPEC.get(spec_key, OrderedDict())
    base_label = "Perfusion" if spec_key == "PERFUSION" else spec_key
    pairs = [(f"{base_label}({cat})", meta.get("gen")) for cat, meta in spec.items()]
    if only_with_registered_generator:
        try:
            from .generators import GENERATOR_REGISTRY
            pairs = [(lab, key) for (lab, key) in pairs if key in GENERATOR_REGISTRY]
        except Exception:
            pass
    return pairs


def _filter_derivatives_by_policy(base_type: str, plan_mode: str) -> List[Tuple[str,str]]:
    """
    plan_mode: "make" (make_derived_from_scratch), "add" (add_missing_derived), or "none".
    Returns [(final_label, generator_key)] filtered by each sublabel's 'policy':
       - "ignore"        → exclude always
       - "convert_only"  → exclude for planning (still allowed if present as vendor DICOM)
       - "derive"/None   → include when plan_mode in {"make","add"}
    """
    bt = (base_type or "").strip().upper()
    spec_key = "PERFUSION" if bt in ("PERFUSION","DSC","DCE") else bt
    spec = DERIVED_CATEGORY_SPEC.get(spec_key, OrderedDict())
    base_label = "Perfusion" if spec_key == "PERFUSION" else spec_key
    out: List[Tuple[str,str]] = []
    for cat, meta in spec.items():
        pol = str(meta.get("policy","derive")).lower()
        if pol == "ignore":
            continue
        if plan_mode in ("make","add"):
            if pol == "convert_only":
                continue
            out.append((f"{base_label}({cat})", meta.get("gen")))
    return out


def choose_primary_for_derivation(series_df):
    """
    Given a classify_exam_series() dataframe for an exam, return a list of dicts with:
      - series_dir
      - series_number
      - final_label
      - base_type
    for candidate primaries to derive from (skip derived=True rows).
    """
    prims = []
    try:
        cols = series_df.columns
        for _, r in series_df.iterrows():
            if bool(r.get("is_derived", False)):
                continue
            base_type = str(r.get("base_type",""))
            if not enumerate_supported_derivatives(base_type):
                continue
            prims.append({
                # Prefer the column produced by classify_exam_series():
                # 'folder' holds the on-disk series directory.
                "series_dir": (
                    r.get("series_dir")
                    or r.get("folder")
                    or r.get("SeriesPath")
                    or r.get("path") or ""
                ),
                "series_number": r.get("series_number",""),
                "final_label": r.get("final_label",""),
                "base_type": base_type,
            })
    except Exception:
        pass
    return [p for p in prims if p["series_dir"]]


def build_derived_output_name(exam_alias: str, out_root: str, primary_label: str, derived_label: str) -> str:
    """
    Simple deterministic destination:
      {out_root}/{exam_alias}/{derived_label}.nii.gz
    """
    exam_alias = str(exam_alias).strip().replace(os.sep, "_")
    # Within-field separators should be '-', not '_'. Use the global sanitizer.
    derived_label = _sanitize_label(str(derived_label).strip())
    return os.path.join(out_root, exam_alias, f"{derived_label}.nii.gz")


def _is_dicom_dir(path: str) -> bool:
    try:
        if not os.path.isdir(path):
            return False
        for name in os.listdir(path)[:8]:
            if name.lower().endswith(".dcm"):
                return True
        return False
    except Exception:
        return False

def _dbg(verbose, *a):
    #Disable logging for now
    #print("[nifti_from_any]", *a, flush=True)
    return None

def _log_nifti_shape(nifti_path: str, orig_path: str | None, verbose=None):
    try:
        img = nib.load(nifti_path)
        shape = tuple(img.header.get_data_shape())
        ndim = len(shape)
        is4d = (ndim == 4 and shape[3] >= 2)
        src = orig_path if orig_path else nifti_path
        _dbg(verbose, f"input = {src}; shape={shape} (ndim={ndim}) → {'4D OK' if is4d else 'NOT 4D'}")
    except Exception as e:
        _dbg(verbose, "failed to inspect nifti:", e)

def _which(cmd: str) -> str | None:
    return shutil.which(cmd)

def _transcode_dicom_dir_to_explicit_le(src_dir: str, *, verbose=None) -> str | None:
    """Return path to a temp dir with decoded DICOMs (Explicit VR Little Endian), or None if no tool is available."""
    src = Path(src_dir)
    if not src.is_dir():
        return None
    dst = Path(tempfile.mkdtemp(prefix="decoded_"))
    use_gdcm = _which("gdcmconv")
    use_dcmd = _which("dcmdjpeg")
    if not (use_gdcm or use_dcmd):
        _dbg(verbose, "No gdcmconv/dcmdjpeg found; cannot transcode.")
        return None

    dcm_files = sorted([p for p in src.iterdir() if p.suffix.lower()==".dcm"])
    if not dcm_files:
        _dbg(verbose, "No .dcm files to transcode.")
        return None

    _dbg(verbose, f"Transcoding {len(dcm_files)} DICOMs → {dst}")
    for f in dcm_files:
        out = dst / f.name
        if use_gdcm:
            # -w : write decompressed (Explicit VR Little Endian)
            cmd = [use_gdcm, "-w", str(f), str(out)]
        else:
            # +te : transcode to Explicit VR Little Endian
            cmd = [use_dcmd, "+te", str(f), str(out)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            _dbg(verbose, f"Transcode failed for {f.name}: {res.stdout.strip()[:200]}")
            # Best effort: skip this file; dcm2niix can still work with the rest
    return str(dst)

def _nifti_from_any(input_path_or_dir: str,
                    output_path: str | None = None,
                    verbose=None,
                    strict_deid: bool = True):
    """
    Accept a DICOM series directory or a NIfTI file path.
    If DICOM dir, convert to a temp NIfTI (keeping 4D if present).
    Uses dcm2niix (emits bval/bvec when applicable) with optional JPEG-lossless transcode+retry.
    Return (nifti_path, cleanup_temp_bool). If output_path is provided, the returned path is output_path and cleanup=False.
    """
    if _is_dicom_dir(input_path_or_dir):
        _dbg(verbose, "Input is DICOM dir:", input_path_or_dir)
        # Ensure dcm2niix exists
        if not _which("dcm2niix"):
            raise RuntimeError("dcm2niix not found on PATH. Please install dcm2niix and ensure it is discoverable.")
        # Try dcm2niix (with optional transcode+retry)
        try:
            tmpdir = tempfile.mkdtemp(prefix="dcm2niix_")
            cmd = ["dcm2niix", "-z", "y", "-f", "tmp", "-o", tmpdir, input_path_or_dir]
            _dbg(verbose, "running:", " ".join(cmd))
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            _dbg(verbose, f"dcm2niix returncode={proc.returncode}")
            if proc.returncode != 0:
                # NEW: try to transcode JPEG-lossless to Explicit LE, then retry dcm2niix
                _dbg(verbose, "dcm2niix failed; attempting JPEG-lossless transcode and retry.")
                decoded = _transcode_dicom_dir_to_explicit_le(input_path_or_dir, verbose=verbose)
                if decoded:
                    tmpdir2 = tempfile.mkdtemp(prefix="dcm2niix_")
                    cmd2 = ["dcm2niix", "-z", "y", "-f", "tmp", "-o", tmpdir2, decoded]
                    _dbg(verbose, "retry:", " ".join(cmd2))
                    proc2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    _dbg(verbose, f"dcm2niix retry returncode={proc2.returncode}")
                    if proc2.returncode == 0:
                        nii_candidates = sorted(glob.glob(os.path.join(tmpdir2, "tmp*.nii*"))) \
                                     or sorted(glob.glob(os.path.join(tmpdir2, "*.nii*")))
                        if not nii_candidates:
                            raise RuntimeError("dcm2niix retry produced no NIfTI")
                        nifti_path = nii_candidates[0]
                        _dbg(verbose, "dcm2niix retry output:", nifti_path)
                        _log_nifti_shape(nifti_path, input_path_or_dir, verbose)
                        # Write PHI-safe JSON sidecar using ORIGINAL DICOM dir (not the decoded copy)
                        try:
                            sidecar = build_dynamic_sidecar_from_dicoms(input_path_or_dir)
                            _merge_and_write_json_sidecar(nifti_path, sidecar, strict_deid=strict_deid)
                            _dbg(verbose, "wrote JSON sidecar next to NIfTI")
                        except Exception as e:
                            _dbg(verbose, "WARNING: failed to write JSON sidecar:", e)
                        # Move/copy to output_path if requested
                        if output_path:
                            dest = _finalize_output(nifti_path, output_path)
                            return dest, False
                        return nifti_path, True
                # If transcode+retry did not succeed, fall through to fallback
                err = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(f"dcm2niix failed:\n{err[:400]}")
            # Success path (original dcm2niix)
            nii_candidates = sorted(glob.glob(os.path.join(tmpdir, "tmp*.nii*"))) \
                          or sorted(glob.glob(os.path.join(tmpdir, "*.nii*")))
            if not nii_candidates:
                raise RuntimeError("dcm2niix produced no NIfTI")
            nifti_path = nii_candidates[0]
            _dbg(verbose, "dcm2niix output:", nifti_path)
            has_bval = os.path.exists(os.path.join(tmpdir, "tmp.bval"))
            has_bvec = os.path.exists(os.path.join(tmpdir, "tmp.bvec"))
            _dbg(verbose, f"sidecars: bval={has_bval} bvec={has_bvec}")
            _log_nifti_shape(nifti_path, input_path_or_dir, verbose)
            # write JSON sidecar
            try:
                sidecar = build_dynamic_sidecar_from_dicoms(input_path_or_dir)
                _merge_and_write_json_sidecar(nifti_path, sidecar, strict_deid=strict_deid)
                if verbose:
                    print("[nifti_from_any] wrote JSON sidecar next to NIfTI")
            except Exception as e:
                _dbg(verbose, "WARNING: failed to write JSON sidecar:", e)
            if output_path:
                dest = _finalize_output(nifti_path, output_path)
                return dest, False
            return nifti_path, True
        except Exception as e:
            # No fallback: surface the dcm2niix error clearly
            msg = f"dcm2niix could not convert '{input_path_or_dir}': {type(e).__name__}: {e}"
            _dbg(verbose, msg)
            raise RuntimeError(msg)
    else:
        _log_nifti_shape(input_path_or_dir, input_path_or_dir, verbose)
        # If caller asked for a specific output_path, copy to that path
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(input_path_or_dir), str(output_path))
            return output_path, False
        return input_path_or_dir, False

def _scrub_sidecar(info: Dict[str, Any], *, strict: bool = True) -> Dict[str, Any]:
    """Remove potentially identifying fields from a dcm2niix-style JSON.
    This is conservative and intended for sharing outside the institution.
    For internal use, set strict=False to keep everything.
    """
    if not strict or not isinstance(info, dict):
        return info
    drop_keys = {
        # Site/equipment revealing fields
        "InstitutionName", "InstitutionAddress", "StationName", "DeviceSerialNumber",
        # Date/time of acquisition
        "AcquisitionTime", "AcquisitionDate", "AcquisitionDateTime",
        "SeriesDate", "SeriesTime", "StudyDate", "StudyTime",
        "ContentDate", "ContentTime",
        # Free-text that may carry PHI (site-dependent)
        "StudyDescription", "SeriesDescription", "ProtocolName", "ProcedureStepDescription",
    }
    # Also drop any unexpected Patient* fields if present
    info = {k: v for k, v in info.items() if (k not in drop_keys and not k.startswith("Patient"))}
    return info

def _nii_stem(p: str) -> str:
    return p[:-7] if p.lower().endswith(".nii.gz") else os.path.splitext(p)[0]

def _merge_and_write_json_sidecar(nifti_path: str, dynamic_info: Dict[str, Any], *, strict_deid: bool = True) -> None:
    """If a dcm2niix JSON exists next to nifti_path, load it, scrub PHI,
    then merge in dynamic_info and write back.

    Notes
    -----
    We intentionally namespace Astril-specific fields under a single top-level key
    to avoid colliding with existing (or future) dcm2niix/BIDS-style keys.
    """
    base = _nii_stem(nifti_path)
    src_json = base + ".json"
    merged: Dict[str, Any] = {}
    if os.path.exists(src_json):
        try:
            with open(src_json, "r", encoding="utf-8") as fh:
                dcm2 = json.load(fh)
            dcm2 = _scrub_sidecar(dcm2, strict=strict_deid)
            if isinstance(dcm2, dict):
                merged.update(dcm2)
        except Exception:
            pass
    # Namespace Astril-specific metadata to avoid polluting the top-level JSON
    # (and to reduce the chance of schema/tool conflicts).
    if isinstance(dynamic_info, dict) and dynamic_info:
        astril = merged.get("Astril")
        if not isinstance(astril, dict):
            astril = {}
        # dynamic_info is already PHI-scrubbed; keep it under the namespace.
        astril.update(dynamic_info)
        merged["Astril"] = astril
    # record deid mode
    merged.setdefault("deidentification", {"strict": bool(strict_deid)})
    try:
        write_json_sidecar(nifti_path, merged)
    except Exception:
        # last resort: write a minimal sidecar containing only the Astril namespace
        write_json_sidecar(nifti_path, {"Astril": dynamic_info or {}, "deidentification": {"strict": bool(strict_deid)}})

def _finalize_output(src_nii: str, dest_nii: str) -> str:
    """
    Copy the NIfTI and any sidecars (.json/.bval/.bvec) from the temp location
    to the requested destination basename.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest_nii)), exist_ok=True)
    shutil.copy2(src_nii, dest_nii)
    # Map: src basename -> dest basename
    def _base(p):
        return p[:-7] if p.lower().endswith(".nii.gz") else os.path.splitext(p)[0]
    sbase, dbase = _base(src_nii), _base(dest_nii)
    
    def _is_base_dwi_output(path_nii: str) -> bool:
        """
        Return True if the destination NIfTI name corresponds to a *base* DWI series.
        Assumes naming like: {patientID}_{timepoint}_{modality}.nii[.gz]
        and base diffusion modality exactly "DWI".
        """
        stem = os.path.basename(_base(path_nii))
        # Parse outer underscore-delimited fields and inspect the *series label* field.
        # New convention: label-internal separators use '-', so a derived diffusion like
        # {pid}_{tp}_DWI-FA would have label "DWI-FA", while base diffusion is exactly "DWI".
        parts = stem.split("_")
        if len(parts) < 3:
            return False
        series_label = parts[2].strip().upper()
        # Base DWI must be exactly DWI (not DWI-FA / DWI(FA) / etc.)
        return series_label == "DWI"

    # Always carry JSON. Only carry diffusion vectors for base DWI outputs.
    exts = [".json"]
    if _is_base_dwi_output(dest_nii):
        exts.extend([".bval", ".bvec"])

    for ext in exts:
        # prefer the canonical sidecar (e.g., tmp.json), but gracefully handle legacy tmp.nii.json
        if ext == ".json":
            primary = sbase + ".json"
            legacy  = sbase + ".nii.json"
            if os.path.exists(primary):
                shutil.copy2(primary, dbase + ".json")
            elif os.path.exists(legacy):
                shutil.copy2(legacy, dbase + ".json")
        else:
            s = sbase + ext
            if os.path.exists(s):
                shutil.copy2(s, dbase + ext)
    return dest_nii

def export_dwi_nrrd_from_dicoms(
    dicom_series_dir: str,
    output_nrrd_path: str,
    *,
    verbose: Optional[bool] = None,
) -> str:
    """Export a diffusion NRRD (with embedded gradients) using dcm2niix.

    This is primarily intended for 3D Slicer / SlicerDMRI compatibility.
    We keep the standard NIfTI + .bval/.bvec + .json outputs as well.
    """
    if not _which("dcm2niix"):
        raise RuntimeError("dcm2niix not found on PATH. Please install dcm2niix and ensure it is discoverable.")

    out_dir = os.path.dirname(os.path.abspath(output_nrrd_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Use the desired basename so the produced file matches the NIfTI naming.
    base = os.path.basename(output_nrrd_path)
    # strip known extensions for dcm2niix -f
    for ext in (".nrrd", ".nhdr"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break

    # Write into a temp dir first (avoid clobbering / partial outputs on failure)
    tmpdir = tempfile.mkdtemp(prefix="dwi_nrrd_", dir=out_dir)
    try:
        cmd = ["dcm2niix", "-e", "y", "-f", base, "-o", tmpdir, dicom_series_dir]
        _dbg(verbose, "[export_dwi_nrrd] running:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"dcm2niix NRRD export failed:\n{err[:400]}")
        # dcm2niix may emit .nrrd or .nhdr (with separate .raw.gz/.raw)
        candidates = (
            glob.glob(os.path.join(tmpdir, base + ".nrrd")) +
            glob.glob(os.path.join(tmpdir, base + ".nhdr")) +
            glob.glob(os.path.join(tmpdir, "*.nrrd")) +
            glob.glob(os.path.join(tmpdir, "*.nhdr"))
        )
        if not candidates:
            raise RuntimeError("dcm2niix NRRD export produced no .nrrd/.nhdr output")
        produced = candidates[0]

        # Move the header/data pair if needed
        if produced.lower().endswith(".nhdr"):
            # Move header and associated data file(s)
            dest_hdr = output_nrrd_path if output_nrrd_path.lower().endswith(".nhdr") else (os.path.splitext(output_nrrd_path)[0] + ".nhdr")
            shutil.move(produced, dest_hdr)
            for p in glob.glob(os.path.join(tmpdir, "*")):
                if os.path.abspath(p) == os.path.abspath(produced):
                    continue
                shutil.move(p, os.path.join(os.path.dirname(dest_hdr), os.path.basename(p)))
            return dest_hdr
        else:
            dest = output_nrrd_path if output_nrrd_path.lower().endswith(".nrrd") else (os.path.splitext(output_nrrd_path)[0] + ".nrrd")
            shutil.move(produced, dest)
            for p in glob.glob(os.path.join(tmpdir, "*")):
                if os.path.abspath(p) == os.path.abspath(produced):
                    continue
                dstp = os.path.join(os.path.dirname(dest), os.path.basename(p))
                if not os.path.exists(dstp):
                    shutil.move(p, dstp)
            return dest
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _fsl_sidecar_paths(nifti_path: str) -> tuple[str, str]:
    """Return (.bval_path, .bvec_path) for a NIfTI path."""
    base = _nii_stem(nifti_path)
    return base + ".bval", base + ".bvec"


def _read_fsl_bvals(path: str):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip().split()
    return np.array([float(x) for x in txt], dtype=float)


def _read_fsl_bvecs(path: str):
    # FSL bvec is typically 3 rows x N cols, but sometimes N rows x 3.
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    arr = np.array(rows, dtype=float)
    if arr.shape[0] == 3:
        return arr
    if arr.shape[1] == 3:
        return arr.T
    raise ValueError(f"Unrecognized bvec shape {arr.shape} in {path}")


def _write_fsl_bvecs(path: str, bvecs_3xN):
    # Preserve canonical 3-row FSL format
    with open(path, "w", encoding="utf-8") as f:
        for r in range(3):
            f.write(" ".join(f"{float(x):.10g}" for x in bvecs_3xN[r, :]))
            f.write("\n")


def _rotation_from_sitk_transform(tfm):
    """Extract an orthonormal 3x3 rotation matrix from a SimpleITK transform.

    For affine-like transforms, SimpleITK exposes a 3x3 matrix via GetMatrix().
    We orthonormalize it (polar decomposition via SVD) to drop any numeric shear/scale.
    """
    if not hasattr(tfm, "GetMatrix"):
        return None
    mat = tfm.GetMatrix()
    if mat is None:
        return None
    mat = list(mat)
    if len(mat) != 9:
        return None
    A = np.array(mat, dtype=float).reshape(3, 3)
    # Orthonormalize: R = U V^T
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    # Ensure right-handed
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def update_fsl_vectors_after_transform(
    src_nifti_path: str,
    dst_nifti_path: str,
    transform,
    *,
    inverse: bool = False,
    verbose: Optional[bool] = None,
) -> dict:
    """Copy .bval and update (rotate) .bvec to match an applied spatial transform."""
    src_bval, src_bvec = _fsl_sidecar_paths(src_nifti_path)
    dst_bval, dst_bvec = _fsl_sidecar_paths(dst_nifti_path)
    out = {"copied_bval": False, "updated_bvec": False, "reason": ""}

    if not os.path.exists(src_bval) and not os.path.exists(src_bvec):
        out["reason"] = "no_bval_or_bvec"
        return out

    # Always copy bvals if present
    if os.path.exists(src_bval):
        try:
            shutil.copy2(src_bval, dst_bval)
            out["copied_bval"] = True
        except Exception as e:
            out["reason"] = f"failed_copy_bval:{type(e).__name__}"
            _dbg(verbose, "[update_fsl_vectors] WARNING copying bval:", e)

    if not os.path.exists(src_bvec):
        out["reason"] = out["reason"] or "no_bvec"
        return out

    R = _rotation_from_sitk_transform(transform)
    if R is None:
        out["reason"] = out["reason"] or "transform_no_matrix"
        return out

    if inverse:
        R = np.linalg.inv(R)

    try:
        bvecs = _read_fsl_bvecs(src_bvec)  # 3xN
        if bvecs.shape[0] != 3:
            raise ValueError("bvecs not 3xN after parsing")
        new_bvecs = R @ bvecs
        _write_fsl_bvecs(dst_bvec, new_bvecs)
        out["updated_bvec"] = True
        return out
    except Exception as e:
        out["reason"] = f"failed_update_bvec:{type(e).__name__}"
        _dbg(verbose, "[update_fsl_vectors] WARNING updating bvec:", e)
        # fallback: copy original bvec without modification
        try:
            shutil.copy2(src_bvec, dst_bvec)
        except Exception:
            pass
        return out

def _sanitize_label(lbl: str) -> str:
    """Sanitize a label for use inside filenames.

    **Naming policy:** outer filename fields are separated by underscores, but *within-field*
    separators must be hyphens ("-"). This function therefore converts underscores to
    hyphens to prevent ambiguous parsing later.
    """
    if lbl is None:
        return "Unknown"

    s = str(lbl).strip()

    # Never allow underscores inside the field; treat them as within-field separators.
    s = s.replace("_", "-")

    # Replace common punctuation with within-field separator.
    s = s.replace("(", "-").replace(")", "-")

    # Keep alphanumerics and a conservative set of safe characters; convert everything else to '-'.
    # NOTE: '\w' includes '_' but we've already converted '_' to '-'.
    s = re.sub(r"[^\w\-\+\.]+", "-", s)

    # Collapse repeated separators and trim.
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "Unknown"


# ----------------------------
# Minimal generator functions
# ----------------------------

def run_derived_generator(input_path_or_dicom_dir, output_path: str, generator_key: str, primary_label: str = "", derived_label: str = "") -> str:
    """
    Convert DICOM->NIfTI if needed, then run the requested generator and save output_path.
    Accepts either a single path/dir (str) or a dict of inputs (e.g., {"MAG": <path>, "PHASE": <path>}).
    """
    from .generators import GENERATOR_REGISTRY
    fn = GENERATOR_REGISTRY.get(generator_key)
    if fn is None:
        raise ValueError(f"Unsupported generator '{generator_key}'")
    # dict (multi-input) case
    if isinstance(input_path_or_dicom_dir, dict):
        nifti_inputs, cleanups = {}, {}
        try:
            for k, v in input_path_or_dicom_dir.items():
                p, c = _nifti_from_any(input_path_or_dir=v)
                ku = str(k).upper()
                nifti_inputs[ku] = p
                cleanups[ku] = c
            written = fn(nifti_inputs, output_path)
        finally:
            for ku, c in cleanups.items():
                try:
                    if c:
                        os.remove(nifti_inputs[ku])
                except Exception:
                    pass
        return written
    # single-input
    nifti_path, cleanup = _nifti_from_any(input_path_or_dir=input_path_or_dicom_dir)
    try:
        return fn(nifti_path, output_path)
    finally:
        if cleanup:
            try:
                os.remove(nifti_path)
            except Exception:
                pass

def _convert_one(rec: dict) -> dict:
    rec = dict(rec)
    out_path = rec.get("nii_out") or ""
    series_dir = rec.get("series_path") or rec.get("SeriesPath") or rec.get("SourceSeriesPath") or ""
    action = (rec.get("Action") or "CONVERT").upper()
    rec.update({
        "status": None,
        "message": "",
        "nii_path": "",
    })

# ----------------------------
# PHI-safe sidecar helpers
# ----------------------------

def _seconds_from_time_str(t: str) -> Optional[float]:
    """Parse 'HHMMSS' or 'HHMMSS.FFFFFF' -> seconds from midnight."""
    if not t:
        return None
    t = str(t)
    try:
        if "." in t:
            main, frac = t.split(".")
            frac = float("0." + frac)
        else:
            main, frac = t, 0.0
        main = main.zfill(6)
        hh = int(main[0:2]); mm = int(main[2:4]); ss = int(main[4:6])
        return hh * 3600 + mm * 60 + ss + frac
    except Exception:
        return None

def build_dynamic_sidecar_from_dicoms(dicom_dir: str, *, max_files: int = 5000) -> Dict[str, Any]:
    """PHI-safe per-frame timing (relative) & diffusion summary from a DICOM series directory."""
    if pydicom is None:
        return {"warning": "pydicom_not_available", "source": "dicom"}
    try:
        import glob
        files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    except Exception:
        files = []
    if not files:
        return {"warning": "no_dicom_files_found", "source": "dicom"}
    files = files[:max_files]

    recs = []
    nframes_per_file = []
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, specific_tags=[
                (0x0008,0x0032),  # AcquisitionTime
                (0x0018,0x1060),  # TriggerTime (ms)
                (0x0020,0x0013),  # InstanceNumber
                (0x0020,0x0032),  # ImagePositionPatient
                (0x0020,0x1041),  # SliceLocation
                (0x0028,0x0008),  # NumberOfFrames
                (0x0018,0x9087),  # Diffusion b-value
                (0x0018,0x9089),  # Diffusion Gradient Orientation
                (0x0018,0x0080),  # RepetitionTime (ms)
            ])
            acq_time = _seconds_from_time_str(getattr(ds, "AcquisitionTime", None))
            trig = getattr(ds, "TriggerTime", None)
            if trig is not None:
                try:
                    trig = float(trig) / 1000.0
                except Exception:
                    trig = None
            inst = getattr(ds, "InstanceNumber", None)
            ipp  = getattr(ds, "ImagePositionPatient", None)
            sl   = getattr(ds, "SliceLocation", None)
            bval = getattr(ds, "DiffusionBValue", None)
            grad = getattr(ds, "DiffusionGradientOrientation", None)
            recs.append({
                "acq_time_s": acq_time,
                "trigger_time_s": trig,
                "instance": inst,
                "z": float(ipp[2]) if (ipp is not None and len(ipp)>=3) else (float(sl) if sl is not None else None),
                "bval": float(bval) if bval is not None else None,
                "bvec": [float(x) for x in grad] if grad is not None else None,
            })

            # Track multiframe-per-file layouts (common in some diffusion exports)
            try:
                nf = getattr(ds, "NumberOfFrames", None)
                nf = int(nf) if nf is not None else None
            except Exception:
                nf = None
            nframes_per_file.append(nf)
        except Exception:
            continue

    if not recs:
        return {"warning": "dicom_unreadable", "source": "dicom"}

    zs = [r["z"] for r in recs if r.get("z") is not None]
    if zs:
        slices_per_vol = len(sorted({round(z, 3) for z in zs}))
    else:
        slices_per_vol = None

    # Robustness: handle "one multiframe DICOM per volume" layouts.
    multiframe_per_file = False
    inferred_slices_per_vol_from_frames = None
    try:
        nfs = [nf for nf in nframes_per_file if nf is not None]
        if len(files) > 1 and nfs:
            max_nf = max(nfs)
            if max_nf and max_nf > 1 and (slices_per_vol is None or int(slices_per_vol) <= 1):
                multiframe_per_file = True
                inferred_slices_per_vol_from_frames = int(max_nf)
                slices_per_vol = inferred_slices_per_vol_from_frames
    except Exception:
        pass

    use_trigger = any(r.get("trigger_time_s") is not None for r in recs)
    key = "trigger_time_s" if use_trigger else "acq_time_s"

    times = []
    if multiframe_per_file:
        # One DICOM file per volume: times are already per-volume, so do NOT chunk.
        times = [r.get(key) for r in recs]
    elif slices_per_vol:
        for i in range(0, len(recs), slices_per_vol):
            chunk = recs[i:i+slices_per_vol]
            tvals = [r.get(key) for r in chunk if r.get(key) is not None]
            times.append(min(tvals) if tvals else None)
    else:
        times = [r.get(key) for r in recs]

    first = next((t for t in times if t is not None), None)
    if first is not None:
        times = [None if t is None else max(0.0, float(t) - first) for t in times]

    bvals = [r["bval"] for r in recs if r.get("bval") is not None]
    unique_bvals = sorted({round(float(b), 2) for b in bvals}) if bvals else None

    bvecs = [r["bvec"] for r in recs if r.get("bvec") is not None]
    if multiframe_per_file:
        # One file per volume: bvecs are already per-volume (when present).
        if len(bvecs) != len(recs):
            bvecs = None
    elif bvecs and slices_per_vol:
        B = []
        for i in range(0, len(bvecs), slices_per_vol):
            chunk = [v for v in bvecs[i:i+slices_per_vol] if v is not None]
            if chunk:
                B.append(np.mean(np.asarray(chunk), axis=0).tolist())
            else:
                B.append(None)
        bvecs = B
    else:
        bvecs = None

    diffs = [j - i for i, j in zip(times[:-1], times[1:]) if (i is not None and j is not None)]
    est_TR = float(np.median(diffs)) if diffs else None

    return {
        "source": "dicom",
        "phi_scrubbed": True,
        "frame_times_sec": times,
        "n_frames": len(times),
        "slices_per_volume": int(slices_per_vol) if slices_per_vol else None,
        "estimated_TR_sec": est_TR,
        "unique_bvals": unique_bvals,
        "bvecs_per_frame": bvecs,
        "multiframe_per_file": bool(multiframe_per_file),
        "notes": "Times normalized to first frame; no absolute dates/times stored."
    }

def write_json_sidecar(nifti_path: str, info: Dict[str, Any], *, suffix: str = ".json") -> str:
    out = _nii_stem(nifti_path) + suffix
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    return out


# ------------------------------------------------------------------------
# General NIfTI helpers for 3D/4D pipelines
# ------------------------------------------------------------------------
def get_nifti_ndim(nifti_path):
    """
    Return (ndim, shape) for a NIfTI file.
    ndim is 3 or 4 for typical MRI volumes.
    """
    import nibabel as nib
    img = nib.load(nifti_path)
    shape = img.shape
    ndim = len(shape)
    return ndim, shape

def extract_nifti_frame(nifti_4d_path, frame_index, out_path):
    """
    Extract a single 3D frame from a 4D NIfTI and write it to out_path.
    Preserves affine and header (as appropriate for nibabel).
    """
    import nibabel as nib
    import numpy as np
    img = nib.load(nifti_4d_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"extract_nifti_frame expects 4D input. Got shape={data.shape} from {nifti_4d_path}")
    if frame_index < 0 or frame_index >= data.shape[3]:
        raise IndexError(f"frame_index out of range: {frame_index} for shape={data.shape}")
    frame = data[..., frame_index]
    out_img = nib.Nifti1Image(frame, img.affine, img.header)
    nib.save(out_img, out_path)
    return out_path

def stack_nifti_frames(frame_paths, out_path):
    """
    Stack multiple 3D NIfTI frames into a 4D NIfTI, using the affine/header from the first frame.
    """
    import nibabel as nib
    import numpy as np
    if not frame_paths:
        raise ValueError("stack_nifti_frames requires at least one frame path.")
    first = nib.load(frame_paths[0])
    frames = []
    for p in frame_paths:
        img = nib.load(p)
        dat = img.get_fdata(dtype=np.float32)
        if dat.ndim != 3:
            raise ValueError(f"Expected 3D frame at {p}, got shape={dat.shape}")
        frames.append(dat)
    data4d = np.stack(frames, axis=3)
    out_img = nib.Nifti1Image(data4d, first.affine, first.header)
    nib.save(out_img, out_path)
    return out_path


# ------------------------------------------------------------------------
# SimpleITK helpers for fast in-memory 4D workflows
# ------------------------------------------------------------------------
def sitk_extract_3d_from_4d(img4d, frame_index: int):
    """Extract a 3D volume (frame) from a 4D SimpleITK image.

    SimpleITK represents 4D NIfTI as a 4D image; many registration routines operate on 3D.
    This helper extracts the requested frame while preserving spatial metadata.
    """
    import SimpleITK as sitk

    if img4d.GetDimension() != 4:
        raise ValueError(f"sitk_extract_3d_from_4d expects a 4D image; got dim={img4d.GetDimension()}")

    size = list(img4d.GetSize())
    if frame_index < 0 or frame_index >= size[3]:
        raise IndexError(f"frame_index out of range: {frame_index} for size={size}")

    idx = [0, 0, 0, int(frame_index)]
    ext = size[:]
    ext[3] = 0  # Extract removes the 4th dimension when the size is 0 in that dimension
    return sitk.Extract(img4d, ext, idx)

def sitk_join_3d_frames_to_4d(frames3d, *, spatial_reference, time_reference):
    """Join a list of 3D SimpleITK images into a 4D image, preserving useful metadata.

    - spatial_reference: 3D image whose spatial grid (origin/spacing/direction) we want to preserve.
      (Typically the fixed image used for resampling.)
    - time_reference: original 4D moving image, used to preserve time spacing/origin when available.
    """
    import SimpleITK as sitk

    if not frames3d:
        raise ValueError("sitk_join_3d_frames_to_4d requires at least one 3D frame.")

    out = sitk.JoinSeries(list(frames3d))

    # Preserve time spacing/origin if available; otherwise fall back to 1.0 / 0.0.
    t_spacing = 1.0
    t_origin = 0.0
    if time_reference is not None and time_reference.GetDimension() == 4:
        try:
            t_spacing = float(time_reference.GetSpacing()[3])
            t_origin = float(time_reference.GetOrigin()[3])
        except Exception:
            pass

    # Spatial metadata comes from the reference (which is in the target space).
    sp = spatial_reference.GetSpacing()
    org = spatial_reference.GetOrigin()
    dir3 = spatial_reference.GetDirection()  # length 9

    out.SetSpacing(tuple(list(sp) + [t_spacing]))
    out.SetOrigin(tuple(list(org) + [t_origin]))

    # 4D direction is 16-length flattened 4x4 matrix; embed 3D direction in the top-left.
    d = [0.0] * 16
    d[0], d[1], d[2] = dir3[0], dir3[1], dir3[2]
    d[4], d[5], d[6] = dir3[3], dir3[4], dir3[5]
    d[8], d[9], d[10] = dir3[6], dir3[7], dir3[8]
    d[15] = 1.0
    out.SetDirection(tuple(d))

    return out

def apply_mask_anydim_sitk(input_image_path, mask_path, output_path):
    """
    Apply a 3D brain mask to a 3D or 4D image using SimpleITK.

    - Works for NIfTI (.nii/.nii.gz) and NRRD (.nrrd) inputs/outputs.
    - For 4D images, the mask is applied to every frame (broadcast over frame axis).
    - For diffusion NRRD (common in Slicer), the data is often stored as a 3D *vector* image
      (dim=3, components-per-pixel > 1). In that case we preserve the vector structure and
      copy metadata keys so the result remains diffusion-aware.
    - Writes output in the format implied by output_path extension.
    """
    import SimpleITK as sitk
    import numpy as np

    # Preserve original pixel type to avoid accidentally changing vector layout on write.
    img = sitk.ReadImage(str(input_image_path))
    msk = sitk.ReadImage(str(mask_path))

    if msk.GetDimension() != 3:
        raise ValueError(f"Mask must be 3D. Got dim={msk.GetDimension()} from {mask_path}")

    img_dim = img.GetDimension()
    ncomp = int(getattr(img, "GetNumberOfComponentsPerPixel", lambda: 1)())
    is_vector_3d = (img_dim == 3 and ncomp > 1)
    orig_pixel_id = img.GetPixelID()

    if img_dim not in (3, 4):
        raise ValueError(f"Unsupported image dim={img_dim} for {input_image_path}")

    # Compare spatial sizes (SimpleITK uses x,y,z ordering for GetSize()).
    img_size = img.GetSize()
    msk_size = msk.GetSize()
    if img_dim == 3:
        if tuple(img_size) != tuple(msk_size):
            raise ValueError(f"Mask size {msk_size} does not match image size {img_size} for {input_image_path}")
    else:
        # 4D: first 3 dims must match (x,y,z)
        if tuple(img_size[:3]) != tuple(msk_size):
            raise ValueError(f"Mask size {msk_size} does not match image spatial size {img_size[:3]} for {input_image_path}")

    # Build a binary mask image (float32)
    msk_bin = sitk.Cast(msk > 0, sitk.sitkFloat32)

    def _geom_tuple(im):
        return (tuple(im.GetSize()), tuple(im.GetSpacing()), tuple(im.GetOrigin()), tuple(im.GetDirection()))

    def _resample_mask_to_ref(mask_img, ref_img):
        """
        Resample a 3D scalar mask onto ref_img geometry using nearest neighbor.
        Ensures exact physical-space match for strict ITK filters.
        """
        if mask_img.GetDimension() != 3 or ref_img.GetDimension() != 3:
            raise ValueError("Mask and reference must be 3D for resampling")
        # If geometry matches exactly, return as-is
        if _geom_tuple(mask_img) == _geom_tuple(ref_img):
            return mask_img
        res = sitk.Resample(
            mask_img,
            ref_img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0.0,
            sitk.sitkFloat32,
        )
        return res

    # Masking strategy:
    # - For 3D VECTOR images (common diffusion NRRD): use ITK Multiply(img, scalar_mask)
    #   to preserve the vector length/components.
    # - For 3D scalar images: Multiply works fine too.
    # - For 4D scalar images: SimpleITK arithmetic on 4D can be inconsistent across IO backends,
    #   so we do a controlled numpy broadcast for 4D only.
    if img_dim == 3:
        if is_vector_3d:
            # Diffusion NRRD in Slicer is often a 3D *vector* image (dim=3, components>1).
            # SimpleITK MultiplyImageFilter does NOT support vector pixel types in 3D, so we mask
            # each component as a scalar volume and then Compose back into a vector image.

            # Important: ensure the mask occupies EXACTLY the same physical space as the diffusion NRRD.
            # Even tiny floating-point differences in spacing/origin/direction can cause ITK to throw.
            ref_scalar = sitk.VectorIndexSelectionCast(img, 0)  # scalar 3D with the diffusion grid
            msk_on_ref = _resample_mask_to_ref(msk_bin, ref_scalar)

            masked_components = []
            for c in range(ncomp):
                comp = sitk.VectorIndexSelectionCast(img, c)  # scalar 3D
                comp_f = sitk.Cast(comp, sitk.sitkFloat32)
                masked_f = sitk.Multiply(comp_f, msk_on_ref)  # scalar×scalar supported (same geometry)
                masked_components.append(masked_f)

            out_vf = sitk.Compose(masked_components)  # vector float32 (ncomp components)
            out_img = sitk.Cast(out_vf, orig_pixel_id)
        else:
            # Scalar 3D: scalar×scalar
            # Ensure mask matches image geometry exactly (strict ITK)
            if _geom_tuple(msk_bin) != _geom_tuple(img):
                msk_bin = _resample_mask_to_ref(msk_bin, img)
            img_f = sitk.Cast(img, sitk.sitkFloat32) if img.GetPixelID() != sitk.sitkFloat32 else img
            out_f = sitk.Multiply(img_f, msk_bin)
            out_img = sitk.Cast(out_f, orig_pixel_id)
    else:
        # 4D scalar: do numpy masking (t,z,y,x) with broadcast
        img_arr = sitk.GetArrayFromImage(img)
        # For 4D scalar, SimpleITK doesn't enforce physical space in the numpy route,
        # but we still want the mask sampled in the 4D image's spatial grid.
        # Use the first timepoint as reference to resample the mask.
        try:
            # Extract a 3D reference by taking the first frame via numpy and reconstructing,
            # then CopyInformation from the 4D image's spatial metadata (best-effort).
            # If your 4D images are true 4D in ITK, this is usually fine as-is.
            pass
        except Exception:
            pass
        msk_arr = sitk.GetArrayFromImage(msk_bin) > 0  # (z,y,x)
        if img_arr.ndim != 4:
            raise ValueError(f"Expected 4D array (t,z,y,x) for {input_image_path}, got shape {img_arr.shape}")
        out_arr = np.where(msk_arr[None, ...], img_arr, 0)
        out_img = sitk.GetImageFromArray(out_arr, isVector=False)
        out_img.CopyInformation(img)

    # Copy all metadata keys so diffusion NRRD remains diffusion-aware (gradients, modality, etc.).
    # Multiply should preserve metadata in most cases, but be explicit.
    try:
        for k in img.GetMetaDataKeys():
            out_img.SetMetaData(k, img.GetMetaData(k))
    except Exception:
        pass

    sitk.WriteImage(out_img, str(output_path))
    return output_path

def apply_mask_anydim(input_image_path, mask_path, output_path):
    import nibabel as nib
    import numpy as np

    img = nib.load(input_image_path)
    data = img.get_fdata(dtype=np.float32)

    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata(dtype=np.float32) > 0
    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D. Got shape={mask.shape} from {mask_path}")

    if data.ndim == 3:
        if data.shape != mask.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {data.shape}")
        out = np.where(mask, data, 0.0).astype(np.float32, copy=False)

    elif data.ndim == 4:
        if data.shape[:3] != mask.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image spatial shape {data.shape[:3]}")
        out = np.where(mask[..., None], data, 0.0).astype(np.float32, copy=False)

    else:
        raise ValueError(f"Unsupported ndim={data.ndim} for {input_image_path}")

    nib.save(nib.Nifti1Image(out, img.affine, img.header), output_path)
    return output_path

def normalize_masked_anydim(input_image_path, mask_path, output_path):
    import nibabel as nib
    import numpy as np

    img = nib.load(input_image_path)
    data = img.get_fdata(dtype=np.float32)

    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata(dtype=np.float32) > 0
    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D. Got shape={mask.shape} from {mask_path}")

    if data.ndim == 3:
        if data.shape != mask.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {data.shape}")

        out = np.zeros_like(data)
        vals = data[mask]
        if vals.size:
            mu = vals.mean()
            sigma = vals.std()
            if sigma <= 0:
                sigma = 1.0
            out[mask] = (vals - mu) / sigma

    elif data.ndim == 4:
        if data.shape[:3] != mask.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image spatial shape {data.shape[:3]}")

        out = np.zeros_like(data)
        for t in range(data.shape[3]):
            frame = data[..., t]
            vals = frame[mask]
            if not vals.size:
                continue
            mu = vals.mean()
            sigma = vals.std()
            if sigma <= 0:
                sigma = 1.0
            out_frame = out[..., t]          # view
            out_frame[mask] = (vals - mu) / sigma

    else:
        raise ValueError(f"Unsupported ndim={data.ndim} for {input_image_path}")

    nib.save(nib.Nifti1Image(out, img.affine, img.header), output_path)
    return output_path

# ------------------------------------------------------------------------
# Sidecar and scan discovery helpers (generalized preprocessing)
# ------------------------------------------------------------------------
def _strip_nii_ext(path_or_name: str) -> str:
    """Return filename without .nii or .nii.gz suffix."""
    name = os.path.basename(str(path_or_name))
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def discover_scans_in_dir(scan_dir: str, prefer_gz: bool = True, modalities: list[str] | None = None) -> dict[str, str]:
    """
    Discover NIfTI scans in a directory (non-recursive) using astril naming convention:
        {patientID}_{timepoint}_{modality}.nii[.gz]
    where {modality} may contain underscores.

    Returns
    -------
    dict[str, str]
        Mapping {modality_label -> filepath}.
        If both .nii and .nii.gz exist for the same modality, prefers .nii.gz when prefer_gz=True.
        If `modalities` is provided, only those modalities are returned.
    """
    import os
    scan_dir = os.fspath(scan_dir)
    if not os.path.isdir(scan_dir):
        raise FileNotFoundError(f"scan_dir not found or not a directory: {scan_dir}")

    entries = []
    for fn in os.listdir(scan_dir):
        p = os.path.join(scan_dir, fn)
        if not os.path.isfile(p):
            continue
        if fn.endswith(".nii") or fn.endswith(".nii.gz"):
            entries.append(p)

    found: dict[str, str] = {}
    for p in entries:
        stem = _strip_nii_ext(p)
        parts = stem.split("_")
        if len(parts) < 3:
            # Can't infer modality reliably
            continue
        modality = "_".join(parts[2:])
        # Resolve conflicts: prefer .nii.gz if requested
        if modality in found:
            prev = found[modality]
            if prefer_gz:
                if prev.endswith(".nii") and p.endswith(".nii.gz"):
                    found[modality] = p
            else:
                # keep first seen
                pass
        else:
            found[modality] = p

    if modalities is not None:
        want = {str(m) for m in modalities}
        found = {k: v for k, v in found.items() if k in want}
    return found


def find_sidecars_for_nifti(nifti_path: str) -> list[str]:
    """
    Find sidecar files that accompany a NIfTI (same stem, different extension),
    e.g. .bval/.bvec/.json. Returns absolute paths.
    """
    import os
    nifti_path = os.fspath(nifti_path)
    d = os.path.dirname(nifti_path)
    stem = _strip_nii_ext(nifti_path)
    sidecars = []
    if not os.path.isdir(d):
        return sidecars
    for fn in os.listdir(d):
        if fn.startswith(stem + "."):
            # exclude the nifti itself
            if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                continue
            sidecars.append(os.path.join(d, fn))
    return sorted(sidecars)


def copy_sidecars_for_output(sidecar_paths: list[str], source_nifti: str, output_nifti: str, dry_run: bool = False) -> list[str]:
    """
    Copy sidecars associated with `source_nifti` to match the stem of `output_nifti`.

    Example:
        source:  P001_d0_DWI.nii.gz  has sidecar P001_d0_DWI.bvec
        output:  P001_d0_DWI_brain.nii.gz  -> copies to P001_d0_DWI_brain.bvec

    Parameters
    ----------
    sidecar_paths : list[str]
        Sidecar absolute paths (typically from find_sidecars_for_nifti()).
    source_nifti : str
        The original nifti path used to derive the stem prefix that sidecars match.
    output_nifti : str
        Destination nifti path whose stem determines copied sidecar filenames.
    dry_run : bool
        If True, do not copy; only return the would-be destination paths.

    Returns
    -------
    list[str]
        Destination sidecar paths.
    """
    import os
    import shutil

    src_stem = _strip_nii_ext(source_nifti)
    out_dir = os.path.dirname(os.fspath(output_nifti))
    out_stem = _strip_nii_ext(output_nifti)
    os.makedirs(out_dir, exist_ok=True)

    dests: list[str] = []
    for sp in sidecar_paths or []:
        fn = os.path.basename(sp)
        if not fn.startswith(src_stem + "."):
            # only copy sidecars that match this source stem
            continue
        suffix = fn[len(src_stem):]  # includes leading '.'
        dest = os.path.join(out_dir, out_stem + suffix)
        dests.append(dest)
        if not dry_run:
            shutil.copy2(sp, dest)
    return dests


# -----------------------------------------------------------------------------
# QC PDF helpers for preprocessed MRI libraries
# -----------------------------------------------------------------------------
#
# These helpers support generate_preprocessing_qc_pdfs() in preprocess.py.
# They are imported lazily where possible.

# Optional plotting deps (only used by QC PDF helpers)
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:
    plt = None
    PdfPages = None


def parse_preprocessed_series_filename(fname: str):
    """Parse filenames like:

        {patient}_{timepoint}_{series_type}_brain.nii.gz
        {patient}_{timepoint}_{series_type}_brain-norm.nii.gz

    Returns
    -------
    (patient, timepoint, series_type, kind) or None
        kind is 'brain' or 'brain_norm' (file tag 'brain' or 'brain-norm')
    """
    base = os.path.basename(fname)
    if not base.endswith('.nii.gz'):
        return None
    if base.endswith('_unregistered.nii.gz'):
        return None

    kind = None
    stem = base[:-7]  # strip .nii.gz

    # New convention: kind tag uses within-field '-' (brain-norm).
    # Backwards-compatible with legacy *_brain_norm filenames.
    if stem.endswith('_brain-norm'):
        kind = 'brain_norm'
        core = stem[:-11]
    elif stem.endswith('_brain_norm'):
        kind = 'brain_norm'
        core = stem[:-11]
    elif stem.endswith('_brain'):
        kind = 'brain'
        core = stem[:-6]
    else:
        return None

    parts = core.split('_')
    if len(parts) < 3:
        return None
    patient = parts[0]
    timepoint = parts[1]
    series_type = '_'.join(parts[2:])
    if not series_type:
        return None
    return patient, timepoint, series_type, kind


def iter_patient_exam_dirs(root_dir: str):
    root_dir = os.path.abspath(os.fspath(root_dir))
    for pd in sorted(os.listdir(root_dir)):
        pdir = os.path.join(root_dir, pd)
        if not os.path.isdir(pdir):
            continue
        exam_dirs = []
        try:
            for ed in sorted(os.listdir(pdir)):
                edir = os.path.join(pdir, ed)
                if os.path.isdir(edir):
                    exam_dirs.append(edir)
        except Exception:
            exam_dirs = []
        yield pdir, exam_dirs


def sort_series_types(series_types: list[str]) -> list[str]:
    """Stable, deterministic series ordering.

    Groups derived under parent prefix.

    **New convention:** derived labels are separated from parent with '-' (e.g., DWI-FA).
    This function remains backwards-compatible with legacy '_' (e.g., DWI_FA).

    Parent series appears before derived series.
    """
    def key(s: str):
        if '-' in s:
            parent, rest = s.split('-', 1)
            derived_flag = 1
        elif '_' in s:
            parent, rest = s.split('_', 1)
            derived_flag = 1
        else:
            parent, rest = s, ''
            derived_flag = 0
        return (parent, derived_flag, rest)

    return sorted(series_types, key=key)


def collect_preprocessed_series_types(root_dir: str) -> list[str]:
    """Scan entire library and return global ordered list of series types."""
    series = set()
    for pdir, exam_dirs in iter_patient_exam_dirs(root_dir):
        for edir in exam_dirs:
            try:
                for fn in os.listdir(edir):
                    parsed = parse_preprocessed_series_filename(fn)
                    if not parsed:
                        continue
                    _patient, _tp, series_type, _kind = parsed
                    series.add(series_type)
            except Exception:
                continue
    return sort_series_types(list(series))


def _robust_scale_to_uint8(slice2d: np.ndarray) -> np.ndarray:
    """Percentile-based scaling to 0..255 for display."""
    x = np.asarray(slice2d, dtype=float)
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=np.uint8)
    lo, hi = np.nanpercentile(x, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(x)
        hi = np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def load_center_axial_slice(nifti_path: str) -> np.ndarray:
    """Load NIfTI, take frame 0 if 4D, return center axial slice as uint8."""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D NIfTI for QC slice, got ndim={data.ndim}: {nifti_path}")
    z = data.shape[2] // 2
    sl = data[:, :, z]
    # Display convention: transpose to show radiological-ish orientation consistently
    sl = np.asarray(sl).T
    return _robust_scale_to_uint8(sl)


def _collect_exam_images(exam_dir: str):
    """Return mapping kind->series_type->path for one exam dir."""
    out = {'brain': {}, 'brain_norm': {}}
    try:
        for fn in os.listdir(exam_dir):
            parsed = parse_preprocessed_series_filename(fn)
            if not parsed:
                continue
            _patient, _tp, series_type, kind = parsed
            out[kind][series_type] = os.path.join(exam_dir, fn)
    except Exception:
        pass
    return out


def generate_patient_qc_pdfs(
    patient_dir: str,
    series_order: list[str],
    out_dir: str | None = None,
    *,
    max_exams_per_page: int = 4,
    left_margin_scale: float = 2.25,
) -> tuple[str | None, str | None]:
    """Generate two PDFs for a patient: brain and brain_norm.

    The page height is always 8.5" and row heights are fixed by always allocating
    max_exams_per_page exam slots per page (unused slots are left blank).

    Width is 2.25" per series column.
    """
    if plt is None or PdfPages is None:
        raise ImportError("matplotlib is required to generate QC PDFs (install matplotlib).")

    patient_dir = os.path.abspath(os.fspath(patient_dir))
    if not os.path.isdir(patient_dir):
        return None, None

    exam_names = [d for d in sorted(os.listdir(patient_dir)) if os.path.isdir(os.path.join(patient_dir, d))]
    exam_dirs = [os.path.join(patient_dir, d) for d in exam_names]
    if not exam_dirs:
        return None, None

    # output location
    patient_name = os.path.basename(patient_dir.rstrip(os.sep))
    out_base_dir = os.path.abspath(os.fspath(out_dir)) if out_dir else patient_dir
    os.makedirs(out_base_dir, exist_ok=True)
    out_brain = os.path.join(out_base_dir, f"{patient_name}_qc_brain.pdf")
    out_norm = os.path.join(out_base_dir, f"{patient_name}_qc_brain-norm.pdf")

    def _write(kind: str, out_path: str):
        n_series = len(series_order)
        # page geometry
        page_h = 8.5
        page_w = 2.15 * max(1, n_series)

        # allocate an extra label column (inches) via gridspec width ratios
        label_w_units = 0.4 * float(left_margin_scale)  # relative units vs series cols
        width_ratios = [label_w_units] + [1.0] * n_series

        # rows: max_exams_per_page exams, 2 rows each
        n_rows = max_exams_per_page
        n_rows_total = n_rows + 1  # header row
        height_ratios = [0.25] + [1.0] * n_rows

        with PdfPages(out_path) as pdf:
            # paginate exams
            for start in range(0, len(exam_dirs), max_exams_per_page):
                batch = list(zip(exam_names[start:start+max_exams_per_page], exam_dirs[start:start+max_exams_per_page]))
                # pad to full page rows for consistent row height
                while len(batch) < max_exams_per_page:
                    batch.append((None, None))

                fig = plt.figure(figsize=(page_w, page_h))
                gs = fig.add_gridspec(nrows=n_rows_total, ncols=n_series + 1, width_ratios=width_ratios, height_ratios=height_ratios)

                # column headers
                for c, st in enumerate(series_order):
                    axh = fig.add_subplot(gs[0, c+1])
                    axh.axis('off')
                    axh.text(0.5, 0.5, st, ha='center', va='center', fontsize=8, rotation=0)

                # For each exam slot
                for i, (ename, edir) in enumerate(batch):
                    # each exam uses two rows: raw+norm in original design; here we only render one kind
                    r0 = 1 + i

                    # row label spanning two rows (place on first row)
                    axlbl = fig.add_subplot(gs[r0, 0])
                    axlbl.axis('off')
                    if ename is not None:
                        axlbl.text(0.0, 0.5, ename, ha='left', va='center', fontsize=8)

                    if edir is None:
                        # fill blank axes
                        rr = r0
                        for c in range(n_series):
                            ax = fig.add_subplot(gs[rr, c+1])
                            ax.axis('off')
                        continue

                    exam_map = _collect_exam_images(edir)
                    # occupy both rows with the same kind (top row used, bottom row blank to keep 2-row structure)
                    # top row: images
                    rr = r0
                    for c, st in enumerate(series_order):
                        ax = fig.add_subplot(gs[rr, c+1])
                        ax.axis('off')
                        pth = exam_map.get(kind, {}).get(st)
                        if pth:
                            try:
                                sl = load_center_axial_slice(pth)
                                ax.imshow(sl, cmap='gray', aspect='auto')
                            except Exception:
                                # leave blank on error
                                pass

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    _write('brain', out_brain)
    _write('brain_norm', out_norm)
    return out_brain, out_norm

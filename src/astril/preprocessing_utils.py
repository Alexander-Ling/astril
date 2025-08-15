# astril/preprocessing_utils.py

import numpy as np
import nibabel as nib
import shutil
import os
import warnings
import pydicom
import re
import datetime as _dt
import pandas as pd
import xlsxwriter
import hashlib
from scipy.ndimage import zoom
from pathlib import Path
from typing import Optional, Tuple, List
try:
    from tqdm import tqdm
except Exception:
    tqdm = None # Optional progress bar (graceful fallback if not installed)

# -------- small helper for progress --------
def _progress(iterable, total=None, desc=None, unit=None, enable=True):
    if not enable or tqdm is None:
        return iterable
    kwargs = {}
    if total is not None: kwargs["total"] = total
    if desc: kwargs["desc"] = desc
    if unit: kwargs["unit"] = unit
    return tqdm(iterable, **kwargs)

def apply_padding(data, pad):
    result = data.copy()
    for axis in range(3):
        pad_before, pad_after = pad[axis]

        if pad_before < 0:
            result = np.delete(result, np.s_[:abs(pad_before)], axis=axis)
        if pad_after < 0:
            result = np.delete(result, np.s_[-abs(pad_after):], axis=axis)

        pad_before = max(pad_before, 0)
        pad_after = max(pad_after, 0)
        padding = [(0, 0)] * 3
        padding[axis] = (pad_before, pad_after)
        result = np.pad(result, padding, mode='constant', constant_values=0)
    return result


def interpolate_to_voxel_dims(data, original_voxel_dims, target_voxel_dims, interp):
    zoom_factors = np.divide(original_voxel_dims, target_voxel_dims)
    return zoom(data, zoom_factors, order=interp)


def update_origin_for_padding(affine_matrix, padding, voxel_dims):
    shifts = np.array([pad[0] * voxel_dim for pad, voxel_dim in zip(padding, voxel_dims)])
    affine_matrix[:3, 3] -= shifts
    return affine_matrix


def adjust_to_target_shape(data, target_shape, padding_record=None, shape_padding=None):
    current_shape = np.array(data.shape)
    target_shape = np.array(target_shape)

    if shape_padding is None:
        shape_padding = np.zeros((3, 2), dtype=int)
        for axis in range(3):
            diff = target_shape[axis] - current_shape[axis]
            pad_before = diff // 2
            pad_after = diff - pad_before
            shape_padding[axis] = [pad_before, pad_after] if diff > 0 else [pad_before, pad_after]

    final_data = apply_padding(data, shape_padding)
    if padding_record is not None:
        padding_record['shape_padding'] = shape_padding
    return final_data, padding_record


def read_padding_record(filepath):
    with open(filepath, 'r') as f:
        return eval(f.read(), {"array": np.array})


def load_roi_mask(filepath, shape):
    mask = nib.load(filepath).get_fdata()
    if mask.shape != shape:
        raise ValueError("ROI mask dimensions must match data dimensions.")
    return mask

def ensure_hd_bet_installed():
    if shutil.which("hd-bet") is None:
        raise ImportError(
            "HD-BET CLI not found in PATH.\n\n"
            "Install it via pip (into the same environment you're using):\n"
            "    pip install hd-bet\n\n"
            "Or use the preprocessing extra:\n"
            "    pip install astril[preprocessing]"
        )
    # Ensure the parameter directory exists so downloads don't fail
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

def _dwi_derived_category(tokens: str, imgtypes: set[str]) -> str | None:
    # Order matters: most specific first
    if "adc" in tokens or "ADC" in imgtypes: return "ADC"
    if re.search(r"\bfa\b", tokens) or "FA" in imgtypes: return "FA"
    if "trace" in tokens or "isotropic" in tokens or "ISO" in imgtypes: return "TRACE"
    if "avdc" in tokens: return "AvDC"
    if "exp atten" in tokens or "expatten" in tokens: return "EXP_ATTEN"
    return None

def _swi_derived_category(tokens: str, imgtypes: set[str]) -> str | None:
    t = tokens
    if "min ip" in t or "minip" in t or "MINIP" in imgtypes: return "MINIP"
    if "mip" in t or "MIP" in imgtypes: return "MIP"
    if any(k in t for k in ["pha", "filt pha", "filt_pha", "phase"]) or "PHASE" in imgtypes: return "PHASE"
    if any(k in t for k in ["mag", "magnitude"]) or "MAGNITUDE" in imgtypes: return "MAG"
    if "qsm" in t: return "QSM"
    return None

def _perfusion_derived_category(tokens: str, imgtypes: set[str]) -> str | None:
    for key, lab in [
        ("cbv","CBV"), ("cbf","CBF"), ("mtt","MTT"), ("ttp","TTP"),
        ("tmax","TMAX"), ("ktrans","KTRANS"), ("k trans","KTRANS"),
        ("kep","KEP"), ("ve","VE"), ("vp","VP"),
        ("leakage","LEAKAGE"), ("parametric","PARAM_MAP"),
        ("pbp","PBP"), ("gbp","GBP"),
    ]:
        if key in tokens: return lab
    if "MIP" in imgtypes or "MINIP" in imgtypes: return "PROJECTION"
    return None

def _t1_derived_category(tokens: str, imgtypes: set[str]) -> str | None:
    t = tokens
    if "mip" in t: return "MIP"
    if "minip" in t: return "MINIP"
    return None

def _compute_is_derived(tokens_any, imgtypes: set[str], dcat: str | None) -> bool:
    """
    Decide whether a series should be flagged as 'derived'.

    STRICT ORDER (requested):
      1) If ANY ImageType/FrameType keywords are present in metadata:
           → return True IFF a derived keyword is present (ignore PRIMARY/ORIGINAL).
           → otherwise return False (do NOT fall back to sublabel/name).
      2) If NO ImageType/FrameType keywords were present at all:
           → fall back to sublabel (dcat), then name-only projection safety net.
    """
    # Normalize tokens for name-only fallback checks
    t = " ".join(sorted(tokens_any)).lower() if isinstance(tokens_any, set) else str(tokens_any).lower()

    DERIVED_TOKENS = {
        "DERIVED", "SECONDARY", "REFORMATTED", "RESAMPLED", "MPR",
        "MIP", "MINIP", "T2_STAR", "TRACEW", "ADC", "FA", "EXP_ATTEN", "AVDC",
        "AVERAGE", "MINIMUM", "MAXIMUM", "SUBTRACTION"
    }

    # (1) Metadata present → trust only metadata
    if imgtypes:
        return bool(imgtypes & DERIVED_TOKENS)

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
            num_dicoms=num_dicoms, n_slices_est=n_slices_est,
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

    # 2) Compute exam-level context ONCE (e.g., inferred contrast time)
    # Perfusion or explicit post-contrast cues are our best proxy.
    def _looks_perfusion(tok):
        return any(k in tok for k in ["perfusion","pwi","dsc","dce","asl","pcasl"])
    def _looks_post(tok):
        return any(k in tok for k in ["post","c+","gad","gadolinium","with contrast","w/ contrast","t1c"])
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
        dwi_hit = (dcat_hint is not None) or any(k in t for k in ["dwi","diff","ep2d","ep b","trace w","trace"]) or \
                  (b_value is not None and b_value > 0)

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
            base = "SWI"
            if "swan" in t: reason.append("SWI subtype=SWAN")
            dcat_tokens = _swi_derived_category(t, set())
            dcat_imgtyp = _swi_derived_category("", imgtypes)
            # If name is generic SWI/SWAN (no mag/phase/mip/minip words), hold label at SWI
            generic_swi_name = (("swi" in t or "swan" in t) and not any(w in t for w in ["pha","phase","mag","magnitude","min ip","minip","mip"]))
            dcat = None if generic_swi_name else (dcat_tokens or dcat_imgtyp)
            label = "SWI" if dcat is None else f"SWI({dcat})"
            is_derived = _compute_is_derived(t, imgtypes, dcat)
            reason.append(f"SWI family; derived={is_derived}; dcat={dcat}")
            conf = 0.9 if dcat is None else 0.95

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
            label = "T2-FLAIR"
            is_flair = True
            is_derived = _compute_is_derived(t, imgtypes, None)  # usually False
            reason.append("FLAIR in name")
            conf = 0.9

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
                post_hint = any(k in t for k in ["post","c+","gad","with contrast","w/ contrast","t1c"])
                vendor_post = bool(contrast_agent) or ("CONTRAST" in str(acquisition_contrast).upper())
                if post_hint:
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
                    base = "T1"
                    is_post = False
                    dcat = None
                    label = "T1n"
                    is_derived = _compute_is_derived(t, imgtypes, dcat)
                    reason.append(f"TE/TR suggest T1 (TE={te}, TR={tr})")
                    conf = 0.65
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
    df["n_slices"] = df["n_slices_est"]

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
    if pydicom is None:
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

def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported previousMetadata extension: {ext}")

def _save_table(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".csv":
        df.to_csv(out_path, index=False)
    elif ext in (".tsv", ".txt"):
        df.to_csv(out_path, sep="\t", index=False)
    elif ext in (".xlsx", ".xls"):
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
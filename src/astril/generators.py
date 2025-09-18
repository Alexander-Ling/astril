from __future__ import annotations
import os, math, tempfile
import numpy as np
import nibabel as nib

# Write-unit for DWI scalar maps (ADC/MD/EXP_ATTEN): mm^2/s * DWI_SCALE.
# 1e6 → outputs in "×10^-6 mm^2/s" (aka µm^2/s)
DWI_SCALE = 1e6

# Relaxed brain-mask threshold: keep voxels above 5% of S0 99th percentile
BRAIN_MASK_FRAC = 0.07

# ----------------------------
# Helpers
# ----------------------------
def _save_like(img: nib.Nifti1Image, data: np.ndarray, out_path: str, *, dtype=np.float32) -> str:
    """
    Save 'data' with the input image's affine but force a floating dtype for derived maps.
    Also reset slope/intercept to neutral to avoid unintended scaling on read.
    """
    hdr = img.header.copy()
    try:
        hdr.set_data_dtype(np.dtype(dtype))
    except Exception:
        pass
    # Neutralize scaling fields if present
    try:
        hdr["scl_slope"] = 1.0
        hdr["scl_inter"] = 0.0
    except Exception:
        pass
    out = nib.Nifti1Image(data.astype(dtype, copy=False), img.affine, header=hdr)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    nib.save(out, out_path)
    return out_path

def _ensure_3d(data: np.ndarray) -> np.ndarray:
    """For SWI-style slab projections: accept 3D, or take first frame of 4D."""
    if data.ndim == 4:
        return data[..., 0]
    return data

def _maybe_load_bvals_bvecs(nifti_path: str):
    """
    Try the standard siblings: *.bval / *.bvec (dcm2niix convention).
    Returns (bvals: (V,), bvecs: (3,V)|None) or (None, None) if not found.
    """
    root = nifti_path
    for ext in [".nii.gz", ".nii"]:
        if root.endswith(ext):
            root = root[: -len(ext)]
            break
    bval_path, bvec_path = root + ".bval", root + ".bvec"
    if not os.path.exists(bval_path):
        return None, None   # no bvals → nothing we can do
    bvals = None; bvecs = None
    try:
        bvals = np.loadtxt(bval_path).astype(float).ravel()
        if os.path.exists(bvec_path):
            _b = np.loadtxt(bvec_path).astype(float)
            bvecs = _b.reshape(3, -1) if _b.ndim == 1 else _b
            # Normalize bvecs; guard divide by zero
            n = np.linalg.norm(bvecs, axis=0); n[n == 0] = 1.0
            bvecs = bvecs / n
        return bvals, bvecs
    except Exception:
        return None, None

# ----------------------------
# DWI family
# ----------------------------
def gen_dwi_trace(nifti_path: str, out_path: str) -> str:
    """Isotropic TRACE (vendor-like): geometric mean across b>0 volumes (no b0)."""
    img = nib.load(nifti_path); data = img.get_fdata()
    if data.ndim != 4 or data.shape[3] < 2:
        return _save_like(img, data if data.ndim == 3 else data.mean(axis=3), out_path)
    bvals, _ = _maybe_load_bvals_bvecs(nifti_path)
    if bvals is None or bvals.shape[0] != data.shape[3]:
        # Fallback: mean over all non-first frames (heuristic)
        proj = np.exp(np.mean(np.log(np.clip(data[..., 1:], 1e-6, None)), axis=3))
        return _save_like(img, proj, out_path)
    pos = bvals > 10
    if not np.any(pos):
        # No non-b0 → plain mean
        return _save_like(img, data.mean(axis=3), out_path)
    # Geometric mean over diffusion-weighted frames (b>0)
    v = np.clip(data[..., pos], 1e-6, None)
    trace_iso = np.exp(np.mean(np.log(v), axis=3))
    return _save_like(img, trace_iso, out_path)

def _estimate_adc_from_logfit(S: np.ndarray, bvals: np.ndarray) -> np.ndarray:
    """
    Fit log(S) ~ log(S0) - b * ADC using b≈0 and b>0 volumes (include b0 rows for stability).
    S: (..., V)
    """
    # Use all volumes; require at least two distinct b-values
    b = bvals.astype(np.float64)
    if np.unique(b).size < 2:
        # Fallback: single b-value -> TRACE equivalent
        return S.mean(axis=-1)
    Y = np.log(np.clip(S, 1e-6, None))
    # Linear least squares: Y = A * [logS0, -ADC]^T with A=[1, b]
    A = np.stack([np.ones_like(b), -b], axis=-1)  # (V, 2)
    # Solve per voxel using normal equations
    AtA = A.T @ A
    AtY = np.tensordot(A.T, Y, axes=(1, -1))  # (2, ...spatial...)
    try:
        # Compute [logS0, ADC] for each voxel
        sol = np.linalg.solve(AtA, AtY.reshape(2, -1))  # (2, Nvox)
        ADC = sol[1].reshape(Y.shape[:-1])
        # ADC should be non-negative; clamp tiny negatives to 0
        ADC = np.clip(ADC, 0, None)
        # Mask outside brain-like S0 if available
        try:
            is_b0 = bvals < 10
            if np.any(is_b0):
                S0 = np.maximum(S[..., is_b0].mean(axis=-1), 1e-6)
                p99 = float(np.nanpercentile(S0, 99.0))
                thr = BRAIN_MASK_FRAC * max(p99, 1e-6)
                mask = S0 > thr
                ADC = np.where(mask, ADC, 0.0)
        except Exception:
            pass
        return ADC
    except np.linalg.LinAlgError:
        # Degenerate design (e.g., all nonzero b the same shell) → two-point fallback using S0
        b0 = bvals <= 10
        if not np.any(b0):
            raise
        S0 = np.mean(S[..., b0], axis=-1)  # (...)
        Sb = S[..., pos]                   # (..., K)
        # Avoid division by zero and log(0)
        ratio = np.clip(Sb / np.maximum(S0[..., None], 1e-6), 1e-6, 1.0)
        adc = np.mean(-np.log(ratio) / b[None, None, None, :], axis=-1)
        adc = np.clip(adc, 0, None)
        try:
            S0 = np.maximum(S[..., b0].mean(axis=-1), 1e-6)
            p99 = float(np.nanpercentile(S0, 99.0))
            thr = BRAIN_MASK_FRAC * max(p99, 1e-6)
            mask = S0 > thr
            adc = np.where(mask, adc, 0.0)
        except Exception:
            pass
        return adc

def gen_dwi_adc(nifti_path: str, out_path: str) -> str:
    """Estimate ADC via voxelwise log-linear fit across all b>0 frames.
    Debug logging prints input path, shape, voxel sizes, and bval/bvec summaries.
    """
    # ---- debug logging ----
    img = nib.load(nifti_path); data = img.get_fdata()
    try:
        hdr = img.header
        zooms = tuple(float(z) for z in hdr.get_zooms())
    except Exception:
        pass
    bvals, bvecs = _maybe_load_bvals_bvecs(nifti_path)
    try:
        if bvals is not None:
            uv, uc = np.unique(bvals, return_counts=True)
            # show up to first 12 unique b-values to keep logs compact
            preview = list(map(float, uv[:12]))
            counts  = list(map(int, uc[:12]))
    except Exception:
        pass

    # ---- existing checks / fit ----
    if data.ndim != 4 or data.shape[3] < 2:
        raise ValueError("DWI ADC requires a 4D diffusion series (>=2 volumes).")
    if bvals is None or bvals.shape[0] != data.shape[3]:
        raise ValueError("Missing or mismatched .bval/.bvec for DWI ADC.")
    adc = _estimate_adc_from_logfit(data, bvals)
    # Scale to µm^2/s for output
    return _save_like(img, adc * DWI_SCALE, out_path, dtype=np.float32)

def _fit_tensor(S: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray):
    """
    Basic linear DTI fit (Basser 1994): log(S/S0) = -b * g^T D g.
    Returns MD and FA maps. Assumes >=6 unique (b>0, non-collinear) directions.
    """
    # Identify a b0 (S0); if many, take mean
    is_b0 = bvals < 10
    if not np.any(is_b0):
        raise ValueError("DTI fit needs at least one b≈0 volume.")
    S0 = np.maximum(S[..., is_b0].mean(axis=-1), 1e-6)
    pos = bvals > 10
    if np.count_nonzero(pos) < 6:
        raise ValueError("DTI fit needs at least 6 non-collinear diffusion directions.")
    b = bvals[pos]
    g = bvecs[:, pos].T  # (Vpos, 3)
    # Design matrix for unique tensor elems [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    G = np.stack([
        g[:,0]*g[:,0],
        g[:,1]*g[:,1],
        g[:,2]*g[:,2],
        2*g[:,0]*g[:,1],
        2*g[:,0]*g[:,2],
        2*g[:,1]*g[:,2],
    ], axis=1)  # (Vpos, 6)
    A = (b[:, None] * G)  # (Vpos, 6)
    # RHS
    Y = -np.log(np.clip(S[..., pos] / S0[..., None], 1e-6, 1.0))  # (..., Vpos)
    # Solve normal equations per voxel: (A^T A) d = A^T y
    AtA = A.T @ A  # (6,6)
    AtA_inv = np.linalg.pinv(AtA)
    At = A.T
    Y2 = Y.reshape(-1, Y.shape[-1]).T  # (Vpos, Nvox)
    D6 = (AtA_inv @ (At @ Y2)).T  # (Nvox, 6)
    D6 = D6.reshape(S.shape[:-1] + (6,))
    # Build full 3x3 tensor and eigen-decompose
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = [D6[..., i] for i in range(6)]
    # (… , 3, 3)
    D = np.stack([
        np.stack([Dxx, Dxy, Dxz], axis=-1),
        np.stack([Dxy, Dyy, Dyz], axis=-1),
        np.stack([Dxz, Dyz, Dzz], axis=-1),
    ], axis=-2)
    # Eigenvalues
    w = np.linalg.eigvalsh(D)  # (..., 3)
    w = np.clip(w, 0, None)
    l1, l2, l3 = w[..., 2], w[..., 1], w[..., 0]
    md = (l1 + l2 + l3) / 3.0
    # FA
    num = 1.5 * ((l1 - md)**2 + (l2 - md)**2 + (l3 - md)**2)
    den = (l1**2 + l2**2 + l3**2) + 1e-12
    fa = np.sqrt(np.clip(num / den, 0, 1))
    # Mask outside brain-like S0
    try:
        p99 = float(np.nanpercentile(S0, 99.0))
        thr = BRAIN_MASK_FRAC * max(p99, 1e-6)
        mask = S0 > thr
        md = np.where(mask, md, 0.0)
        fa = np.where(mask, fa, 0.0)
    except Exception:
        pass
    return md, fa

def gen_dwi_fa(nifti_path: str, out_path: str) -> str:
    img = nib.load(nifti_path); data = img.get_fdata()
    if data.ndim != 4 or data.shape[3] < 7:
        raise ValueError("DWI FA requires a 4D diffusion series with >=1 b0 and >=6 directions.")
    bvals, bvecs = _maybe_load_bvals_bvecs(nifti_path)
    if bvals is None or bvecs is None or bvals.shape[0] != data.shape[3]:
        raise ValueError("Missing or mismatched .bval/.bvec for DWI FA.")
    md, fa = _fit_tensor(data, bvals, bvecs)
    return _save_like(img, fa, out_path, dtype=np.float32)

def gen_dwi_md(nifti_path: str, out_path: str) -> str:
    img = nib.load(nifti_path); data = img.get_fdata()
    if data.ndim != 4 or data.shape[3] < 7:
        raise ValueError("DWI MD requires a 4D diffusion series with >=1 b0 and >=6 directions.")
    bvals, bvecs = _maybe_load_bvals_bvecs(nifti_path)
    if bvals is None or bvecs is None or bvals.shape[0] != data.shape[3]:
        raise ValueError("Missing or mismatched .bval/.bvec for DWI MD.")
    md, fa = _fit_tensor(data, bvals, bvecs)
    # Scale to µm^2/s for output
    return _save_like(img, md * DWI_SCALE, out_path, dtype=np.float32)

def gen_dwi_exp_atten(nifti_path: str, out_path: str) -> str:
    """
    Exponential attenuation (vendor-like, dimensionless):
      EA = mean_{b>0} [ S(b) / S0 ], clamped to [ε, 1].
    """
    img = nib.load(nifti_path); S = img.get_fdata()
    if S.ndim != 4 or S.shape[3] < 2:
        raise ValueError("DWI EXP_ATTEN requires a 4D diffusion series (>=2 volumes).")
    bvals, _ = _maybe_load_bvals_bvecs(nifti_path)
    if bvals is None or bvals.shape[0] != S.shape[3]:
        raise ValueError("Missing or mismatched .bval for DWI EXP_ATTEN.")
    is_b0 = bvals < 10
    if not np.any(is_b0):
        raise ValueError("EXP_ATTEN needs at least one b≈0 frame.")
    S0 = np.maximum(S[..., is_b0].mean(axis=-1), 1e-6)
    pos = bvals > 10
    ratio = np.clip(S[..., pos] / S0[..., None], 1e-6, 1.0)
    # Dimensionless attenuation (bright WM, dark CSF), average over b>0 frames
    EA = ratio.mean(axis=-1)
    # Mask outside brain-like S0
    try:
        p99 = float(np.nanpercentile(S0, 99.0))
        thr = BRAIN_MASK_FRAC * max(p99, 1e-6)
        mask = S0 > thr
        EA = np.where(mask, EA, 0.0)
    except Exception:
        pass
    # Write dimensionless (no DWI_SCALE)
    return _save_like(img, EA, out_path)

# ----------------------------
# SWI family
# ----------------------------
def gen_swi_mip(nifti_path: str, out_path: str, slab_mm: int = 8) -> str:
    """Sliding-slab MIP along slice axis; outputs a 3D volume (per-slice local MIP)."""
    img = nib.load(nifti_path); vol = _ensure_3d(img.get_fdata())
    z = vol.shape[2]
    dz = float(img.header.get_zooms()[2] or 1.0)
    k = max(1, int(round(slab_mm / dz)))
    half = k // 2
    out = np.empty_like(vol)
    for i in range(z):
        lo = max(0, i - half); hi = min(z, i + half + 1)
        out[..., i] = vol[..., lo:hi].max(axis=2)
    return _save_like(img, out, out_path)

def gen_swi_minip(nifti_path: str, out_path: str, slab_mm: int = 8) -> str:
    """Sliding-slab MINIP along slice axis; outputs a 3D volume (per-slice local MINIP)."""
    img = nib.load(nifti_path); vol = _ensure_3d(img.get_fdata())
    z = vol.shape[2]
    dz = float(img.header.get_zooms()[2] or 1.0)
    k = max(1, int(round(slab_mm / dz)))
    half = k // 2
    out = np.empty_like(vol)
    for i in range(z):
        lo = max(0, i - half); hi = min(z, i + half + 1)
        out[..., i] = vol[..., lo:hi].min(axis=2)
    return _save_like(img, out, out_path)

def gen_swi_composite(inputs_or_path, out_path: str) -> str:
    """
    Build a composite SWI volume from MAG + PHASE using a simple negative-phase mask.
    inputs_or_path:
      - dict with keys {"MAG": <path/dir>, "PHASE": <path/dir>} (case-insensitive), or
      - str path to an existing SWI (pass-through).
    """
    # Pass-through if a single path was provided
    if isinstance(inputs_or_path, str):
        img = nib.load(inputs_or_path)
        data = _ensure_3d(img.get_fdata())
        return _save_like(img, data, out_path)

    mag_key = next((k for k in inputs_or_path if str(k).upper() in ("MAG","SWI_MAG","SWIMAG")), None)
    pha_key = next((k for k in inputs_or_path if str(k).upper() in ("PHASE","PHA","SWI_PHASE","SWIPHASE","PHI")), None)
    if mag_key is None or pha_key is None:
        raise ValueError("gen_swi_composite expected a dict with MAG and PHASE inputs")

    img_mag = nib.load(inputs_or_path[mag_key]); mag = _ensure_3d(img_mag.get_fdata()).astype(np.float32)
    img_pha = nib.load(inputs_or_path[pha_key]); pha = _ensure_3d(img_pha.get_fdata()).astype(np.float32)
    if mag.shape != pha.shape:
        raise ValueError(f"MAG and PHASE shapes differ: {mag.shape} vs {pha.shape}")

    # Robustly map phase to [-pi, pi]
    p99 = float(np.percentile(mag, 99.0))
    thr = max(1e-6, 0.05 * p99)
    mask = mag > thr
    if not np.any(mask):
        mask = mag > 0
    pha_in = pha[mask]
    med = float(np.median(pha_in))
    p1, p99p = float(np.percentile(pha_in, 1.0)), float(np.percentile(pha_in, 99.0))
    scale = max(1e-6, max(abs(p1 - med), abs(p99p - med)))
    phi = np.clip((pha - med) / scale, -1.0, 1.0) * np.pi

    # Haacke-style negative-phase mask (paramagnetic veins)
    P = 4.0
    m = np.ones_like(phi, dtype=np.float32)
    neg = phi < 0
    m[neg] = ((phi[neg] / np.pi) + 1.0) ** P
    comp = (mag * m).astype(np.float32)
    return _save_like(img_mag, comp, out_path)

# NOTE: QSM would require a full pipeline (unwrap, background removal, dipole inversion).
# Intentionally NOT registering QSM until a validated implementation is available.

# ----------------------------
# Perfusion family (time summaries)
# ----------------------------
def _ensure_4d_time(data: np.ndarray) -> np.ndarray:
    if data.ndim != 4 or data.shape[3] < 2:
        raise ValueError("Expected a 4D time series (>=2 frames).")
    return data

def gen_perfusion_mean_t(nifti_path: str, out_path: str) -> str:
    img = nib.load(nifti_path); data = _ensure_4d_time(img.get_fdata())
    return _save_like(img, data.mean(axis=3), out_path)

def gen_perfusion_max_t(nifti_path: str, out_path: str) -> str:
    img = nib.load(nifti_path); data = _ensure_4d_time(img.get_fdata())
    return _save_like(img, data.max(axis=3), out_path)

def gen_perfusion_auc_t(nifti_path: str, out_path: str) -> str:
    img = nib.load(nifti_path); data = _ensure_4d_time(img.get_fdata())
    return _save_like(img, data.sum(axis=3), out_path)

def gen_perfusion_ttp_index(nifti_path: str, out_path: str) -> str:
    img = nib.load(nifti_path); data = _ensure_4d_time(img.get_fdata())
    idx = np.argmax(data, axis=3).astype(np.float32)
    return _save_like(img, idx, out_path)

# ----------------------------
# (Optional) generic dynamic summaries for true dynamics (kept out of T1/T2 by default)
# ----------------------------
def gen_dyn_mip_t(nifti_path: str, out_path: str) -> str:
    img = nib.load(nifti_path); data = _ensure_4d_time(img.get_fdata())
    return _save_like(img, data.max(axis=3), out_path)

def gen_dyn_minip_t(nifti_path: str, out_path: str) -> str:
    img = nib.load(nifti_path); data = _ensure_4d_time(img.get_fdata())
    return _save_like(img, data.min(axis=3), out_path)

# ----------------------------
# Public registry
# ----------------------------
GENERATOR_REGISTRY = {
    # DWI
    "dwi_trace":       gen_dwi_trace,
    "dwi_adc":         gen_dwi_adc,
    "dwi_fa":          gen_dwi_fa,
    "dwi_md":          gen_dwi_md,
    "dwi_exp_atten":   gen_dwi_exp_atten,
    # SWI composite or slab projections)
    "swi_mip":         gen_swi_mip,
    "swi_minip":       gen_swi_minip,
    "swi_composite":   gen_swi_composite,
    # Perfusion summaries
    "perfusion_mean_t":   gen_perfusion_mean_t,
    "perfusion_max_t":    gen_perfusion_max_t,
    "perfusion_auc_t":    gen_perfusion_auc_t,
    "perfusion_ttp_index":gen_perfusion_ttp_index,
    # Generic dynamic projections (used for Perfusion/DCE only)
    "dyn_mip_t":       gen_dyn_mip_t,
    "dyn_minip_t":     gen_dyn_minip_t,
}

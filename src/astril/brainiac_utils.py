"""
Utilities for downloading BrainIAC model weights and extracting frozen
BrainIAC patch-token encoder features for astril encoder-fusion models.

BrainIAC reference:
  "A generalizable foundation model for analysis of human brain MRI"
  Nature Neuroscience, Feb 2026. https://doi.org/10.1038/s41593-026-02202-6
  GitHub: https://github.com/AIM-KannLab/BrainIAC

Weights are downloaded automatically from Dropbox on first use and cached
locally. A manual path override is available via --BrainIAC_Weights_Path.
Astril does not redistribute BrainIAC weights.

License note: BrainIAC weights are distributed under a custom non-commercial
research-only license (see GitHub for full terms).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from .paths import preferred_models_dir

# Direct download URL for the BrainIAC foundation model checkpoint.
# dl=1 forces Dropbox to serve the raw file rather than the preview page.
BRAINIAC_CKPT_URL = (
    "https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/"
    "AMblt6reQVvlSrORTB3_2lE/BrainIAC.ckpt"
    "?rlkey=9w55le6tslwxlfz6c0viylmjb&dl=1"
)
BRAINIAC_CKPT_FILENAME = "BrainIAC.ckpt"

_DOWNLOAD_FAILED_MSG = (
    "Automatic download of BrainIAC weights failed.\n\n"
    "You can download the checkpoint manually from the Dropbox folder linked\n"
    "in the BrainIAC GitHub repository:\n\n"
    "  https://github.com/AIM-KannLab/BrainIAC\n\n"
    "Once downloaded, supply the local path via:\n\n"
    "  --BrainIAC_Weights_Path <path/to/BrainIAC.ckpt>\n\n"
    "The weights will be cached automatically so you will not need to supply\n"
    "the path again."
)

_MISSING_DEPS_MSG = (
    "BrainIAC feature extraction requires optional dependencies. "
    "Install them with:\n\n"
    "  pip install astril[brainiac]\n\n"
    "or manually:\n\n"
    "  pip install torch>=2.0 monai>=1.3.2 einops>=0.7"
)


class BrainIACWeightsNotFoundError(RuntimeError):
    pass


def _brainiac_cache_dir() -> Path:
    """Returns {preferred_models_dir}/brainiac/ — local cache for BrainIAC weights."""
    d = preferred_models_dir(create=True) / "brainiac"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_cached_weights_path() -> Optional[Path]:
    """Returns the path to a locally cached BrainIAC checkpoint, or None if absent."""
    cache = _brainiac_cache_dir()
    ckpts = list(cache.glob("*.ckpt"))
    if ckpts:
        return ckpts[0]
    return None


def copy_user_weights(src_path: str, dest_dir: Optional[Path] = None) -> Path:
    """
    Copies a user-supplied BrainIAC checkpoint file into the astril cache.
    Returns the destination path.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"BrainIAC weights file not found: {src}")
    dest = (dest_dir or _brainiac_cache_dir()) / src.name
    if dest.resolve() != src.resolve():
        shutil.copy2(str(src), str(dest))
    return dest


def download_brainiac_weights(dest_dir: Optional[Path] = None) -> Path:
    """
    Downloads BrainIAC.ckpt from Dropbox into the astril cache directory.
    Uses requests (a core astril dependency) with a tqdm progress bar.
    Returns the path to the downloaded file.
    Raises BrainIACWeightsNotFoundError if the download fails.
    """
    import requests
    from tqdm import tqdm

    dest = (dest_dir or _brainiac_cache_dir()) / BRAINIAC_CKPT_FILENAME
    tmp = dest.with_suffix(".part")

    print(f"[brainiac] Downloading BrainIAC weights from Dropbox...")
    print(f"[brainiac] Destination: {dest}")

    try:
        with requests.get(BRAINIAC_CKPT_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) or None
            with tqdm(total=total, unit="B", unit_scale=True, desc="BrainIAC.ckpt") as bar:
                with tmp.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise BrainIACWeightsNotFoundError(
            f"{_DOWNLOAD_FAILED_MSG}\n\nOriginal error: {e}"
        )

    tmp.replace(dest)
    print(f"[brainiac] Weights saved to: {dest}")
    return dest


def ensure_brainiac_weights(weights_path: Optional[str]) -> Path:
    """
    Resolves BrainIAC weights using the following priority:
      1. User-supplied weights_path (copied into cache on first use)
      2. Previously cached weights
      3. Automatic download from Dropbox
      4. Raise BrainIACWeightsNotFoundError with manual download instructions

    After the first successful resolution the weights are cached locally,
    so neither weights_path nor a download is needed on subsequent runs.
    """
    if weights_path is not None:
        return copy_user_weights(weights_path)

    cached = get_cached_weights_path()
    if cached is not None:
        print(f"[brainiac] Using cached weights: {cached}")
        return cached

    return download_brainiac_weights()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_brainiac_model(weights_path: Path, device: str):
    """Loads the BrainIAC ViT backbone from a .ckpt checkpoint.

    MONAI 1.5 renamed pos_embed -> proj_type (patch projection) and added
    pos_embed_type (position embedding style).
    """
    import torch
    from monai.networks.nets import ViT

    model = ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        proj_type="conv",           # was pos_embed="conv" in MONAI < 1.4
        pos_embed_type="learnable", # standard ViT learnable positional embeddings
        classification=False,
        dropout_rate=0.0,
        save_attn=False,
    ).to(device)

    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    # BrainIAC checkpoints may be pytorch-lightning state_dicts or plain dicts
    state = ckpt.get("state_dict", ckpt)
    # Strip lightning "model." or "backbone." prefix if present
    cleaned = {}
    for k, v in state.items():
        for prefix in ("model.backbone.", "backbone.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    cross_attn_keys = [k for k in missing if "cross_attn" in k or "norm_cross_attn" in k]
    other_missing = [k for k in missing if k not in cross_attn_keys]
    if cross_attn_keys:
        # MONAI added optional cross-attention sub-layers to SABlock in newer versions.
        # These are never invoked in a standard single-input ViT forward pass, so
        # randomly-initialised weights here have no effect on standard ViT patch-token extraction.
        print(f"[brainiac] Note: {len(cross_attn_keys)} cross-attention keys not in checkpoint "
              f"(MONAI API addition, unused in standard forward pass — safe to ignore).")
    if other_missing:
        print(f"[brainiac] Warning: {len(other_missing)} unexpected missing keys from checkpoint: "
              f"{other_missing[:3]}{'...' if len(other_missing) > 3 else ''}")

    return model


# ---------------------------------------------------------------------------
# BrainIAC patch-token encoder feature pipeline
# ---------------------------------------------------------------------------

# Increment when the patch-grid embedding algorithm changes in a way that makes
# old cached embeddings incompatible with newly generated PCA maps.
BRAINIAC_EMBEDDING_CACHE_VERSION = 2

def _extract_patch_embeddings_for_window(model, tensor, device: str):
    """Run one BrainIAC forward pass and return the final patch token embeddings.

    Args:
        tensor: torch.Tensor of shape (1, 1, 96, 96, 96) already on device.

    Returns:
        np.ndarray of shape (6, 6, 6, 768), float32.
        Ordering matches MONAI ViT patch flattening: depth-major (D, H, W).
    """
    import torch
    with torch.no_grad():
        x, _ = model(tensor)   # x: (1, 216, 768)
    return x[0].reshape(6, 6, 6, 768).cpu().numpy().astype("float32")


def _compute_sliding_window_embeddings(
    sitk_1mm,
    model,
    device: str,
    window_size: int = 96,
    stride: int = 48,
):
    """Tile a 1 mm isotropic volume with overlapping 96^3 windows, extract
    BrainIAC patch token embeddings for each window, and blend overlapping
    contributions with a cosine taper.

    Returns:
        np.ndarray of shape (D_p, H_p, W_p, 768), float32 at patch resolution
        where D_p = ceil(sz_z / 16), H_p = ceil(sz_y / 16), W_p = ceil(sz_x / 16).

    Global patch mapping: window-local patch (pk, pj, pi) at window origin (sx, sy, sz)
    maps to global patch grid position gk/gj/gi = round((s_dim + p * 16 + 8) / 16),
    clipped to [0, dim_p - 1].
    """
    import math
    import numpy as np
    import SimpleITK as sitk
    import torch

    sz_x, sz_y, sz_z = sitk_1mm.GetSize()   # SimpleITK: (x, y, z)
    D_p = math.ceil(sz_z / 16)
    H_p = math.ceil(sz_y / 16)
    W_p = math.ceil(sz_x / 16)

    def window_starts(size: int) -> list:
        if size <= window_size:
            return [0]
        starts = list(range(0, size - window_size, stride))
        if starts[-1] + window_size < size:
            starts.append(size - window_size)
        return starts

    xs = window_starts(sz_x)
    ys = window_starts(sz_y)
    zs = window_starts(sz_z)
    total = len(xs) * len(ys) * len(zs)
    print(f"[brainiac]   sliding-window embeddings: {len(xs)}×{len(ys)}×{len(zs)} = {total} windows")

    embed_accum  = np.zeros((D_p, H_p, W_p, 768), dtype=np.float64)
    weight_accum = np.zeros((D_p, H_p, W_p),       dtype=np.float64)

    # Cosine blend reduced from voxel-space (96,96,96) to patch-space (6,6,6)
    blend_full  = _make_cosine_window(window_size)                      # (96,96,96)
    blend_patch = blend_full.reshape(6, 16, 6, 16, 6, 16).mean(axis=(1, 3, 5))  # (6,6,6)

    roi = sitk.RegionOfInterestImageFilter()

    for sx in xs:
        for sy in ys:
            for sz in zs:
                roi.SetIndex([sx, sy, sz])
                roi.SetSize([window_size, window_size, window_size])
                crop = roi.Execute(sitk_1mm)

                arr = sitk.GetArrayFromImage(crop).astype(np.float32)  # (z,y,x)
                arr = _normalize_volume(arr)
                tensor = torch.from_numpy(arr[np.newaxis, np.newaxis]).to(device)

                emb = _extract_patch_embeddings_for_window(model, tensor, device)  # (6,6,6,768)

                for pk in range(6):
                    gk = min((sz + pk * 16 + 8) // 16, D_p - 1)
                    for pj in range(6):
                        gj = min((sy + pj * 16 + 8) // 16, H_p - 1)
                        for pi in range(6):
                            gi = min((sx + pi * 16 + 8) // 16, W_p - 1)
                            w = blend_patch[pk, pj, pi]
                            embed_accum[gk, gj, gi]  += emb[pk, pj, pi] * w
                            weight_accum[gk, gj, gi] += w

    result = embed_accum / np.maximum(weight_accum[..., np.newaxis], 1e-9)
    return result.astype(np.float32)


def _sitk_patch_grid_affine(sitk_img, patch_size: float = 16.0):
    """Return an RAS affine for a patch grid derived from a 1 mm SITK image."""
    import numpy as np

    spacing = np.array(sitk_img.GetSpacing(), dtype=float) * float(patch_size)
    origin = np.array(sitk_img.GetOrigin(), dtype=float)
    direction = np.array(sitk_img.GetDirection(), dtype=float).reshape(3, 3)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    affine = np.eye(4)
    affine[:3, :3] = (lps_to_ras @ direction) * spacing
    affine[:3, 3] = lps_to_ras @ origin
    return affine


def _embedding_grid_to_canonical(emb, sitk_1mm):
    """Convert embedding grid from SITK (z,y,x,c) order to canonical RAS (x,y,z,c)."""
    import nibabel as nib
    import numpy as np
    from nibabel.funcs import as_closest_canonical

    xyzc = np.transpose(emb, (2, 1, 0, 3))
    img = nib.Nifti1Image(xyzc, _sitk_patch_grid_affine(sitk_1mm))
    canonical = as_closest_canonical(img)
    return np.asarray(canonical.dataobj, dtype=np.float32)


def compute_brainiac_encoder_features(
    nifti_paths: List[str],
    weights_path: Path,
    output_dir: Path,
    sequence_label: str,
    device: Optional[str] = None,
    stride: int = 48,
    overwrite: bool = False,
) -> List[str]:
    """Extract frozen BrainIAC patch-token features for encoder-fusion models.

    Saves one canonical RAS .npy feature grid per input volume. Each array is
    shaped (X_p, Y_p, Z_p, 768), where spatial axes are patch-resolution analogs
    of astril's canonical image axes.
    """
    try:
        import torch
        import numpy as np
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError(_MISSING_DEPS_MSG) from e
    try:
        from monai.networks.nets import ViT  # noqa: F401
    except ImportError as e:
        raise ImportError(_MISSING_DEPS_MSG) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_brainiac_model(weights_path, device)
    model.eval()

    output_paths: List[str] = []
    for nifti_path in nifti_paths:
        stem = Path(nifti_path).name.replace(".nii.gz", "").replace(".nii", "")
        out_path = output_dir / f"{stem}_{sequence_label}_encoder_embeddings.npy"
        version_path = output_dir / f"{stem}_{sequence_label}_encoder_cache_version.txt"

        cache_version = None
        if version_path.exists():
            try:
                cache_version = int(version_path.read_text().strip())
            except ValueError:
                cache_version = None

        if out_path.exists() and cache_version == BRAINIAC_EMBEDDING_CACHE_VERSION and not overwrite:
            output_paths.append(str(out_path))
            continue

        if out_path.exists() and not overwrite:
            print(
                f"[brainiac] Ignoring stale {sequence_label} encoder cache "
                f"for {stem}; regenerating."
            )
        print(f"[brainiac] Extracting {sequence_label} encoder embeddings: {stem}")
        sitk_img = sitk.ReadImage(str(nifti_path), sitk.sitkFloat32)
        sitk_1mm = _ensure_1mm_isotropic(sitk_img)
        emb = _compute_sliding_window_embeddings(sitk_1mm, model, device, stride=stride)
        emb_canonical = _embedding_grid_to_canonical(emb, sitk_1mm)
        np.save(str(out_path), emb_canonical.astype(np.float32))
        version_path.write_text(f"{BRAINIAC_EMBEDDING_CACHE_VERSION}\n")
        output_paths.append(str(out_path))

    del model
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("[brainiac] Released GPU memory after encoder embedding extraction.")

    return output_paths


def _ensure_1mm_isotropic(sitk_img, tol: float = 0.01):
    """Returns the image unchanged if already 1 mm isotropic (within tol),
    otherwise resamples to 1 mm isotropic preserving the physical FOV."""
    import SimpleITK as sitk
    import numpy as np

    spacing = np.array(sitk_img.GetSpacing())
    if np.all(np.abs(spacing - 1.0) < tol):
        return sitk_img

    orig_size = np.array(sitk_img.GetSize(), dtype=float)
    new_size = np.round(orig_size * spacing).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([1.0, 1.0, 1.0])
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(sitk_img)


def _make_cosine_window(size: int):
    """3-D sine taper: weight=0 at every edge face, 1 at the centre.
    Uses sin(t) for t in [0, π] — symmetric 0 → 1 → 0 profile — so that
    overlapping windows blend without directional bias at seam boundaries."""
    import numpy as np
    t = np.linspace(0.0, np.pi, size, dtype=np.float32)
    w1d = np.sin(t)                          # symmetric: 0 → 1 → 0
    return w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]


def _normalize_volume(arr):
    """Z-score normalization; returns float32."""
    import numpy as np
    std = arr.std()
    if std > 0:
        arr = (arr - arr.mean()) / std
    return arr.astype("float32")

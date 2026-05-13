"""
Utilities for downloading BrainIAC model weights and extracting saliency maps
for use as additional input channels during training and inference.

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


def compute_brainiac_saliency_maps(
    nifti_paths: List[str],
    weights_path: Path,
    output_dir: Path,
    device: Optional[str] = None,
    overwrite: bool = False,
) -> List[str]:
    """
    Runs BrainIAC attention rollout on each NIfTI volume and saves the resulting
    saliency map as a NIfTI file. Returns the list of output paths in the same
    order as the inputs.

    Saliency maps are saved as:
      {output_dir}/{subject_stem}_brainiac_saliency.nii.gz

    Input NIfTIs should be skull-stripped and co-registered (as produced by
    astril's preprocessing pipeline). Any MRI modality is accepted; only the
    spatial content matters for attention computation.

    BrainIAC operates at 96x96x96 / 1mm isotropic. Inputs are resampled to
    that space before processing, then saliency maps are resampled back to the
    original voxel space before saving.
    """
    # Lazy imports — only needed when this function is called
    try:
        import torch
        import numpy as np
        import nibabel as nib
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError(_MISSING_DEPS_MSG) from e

    try:
        from monai.networks.nets import ViT
    except ImportError as e:
        raise ImportError(_MISSING_DEPS_MSG) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load BrainIAC model
    model = _load_brainiac_model(weights_path, device)
    model.eval()

    output_paths = []
    for nifti_path in nifti_paths:
        stem = Path(nifti_path).name.replace(".nii.gz", "").replace(".nii", "")
        out_path = output_dir / f"{stem}_brainiac_saliency.nii.gz"

        if out_path.exists() and not overwrite:
            output_paths.append(str(out_path))
            continue

        print(f"[brainiac] Computing saliency map for: {stem}")

        # Resample input to 96^3 at 1mm isotropic
        sitk_img = sitk.ReadImage(str(nifti_path), sitk.sitkFloat32)
        resampled, original_info = _resample_to_brainiac_space(sitk_img)

        # Convert to tensor: (1, 1, 96, 96, 96)
        arr = sitk.GetArrayFromImage(resampled).astype(np.float32)
        arr = _normalize_volume(arr)
        tensor = torch.from_numpy(arr[np.newaxis, np.newaxis]).to(device)

        # Compute attention rollout saliency
        with torch.no_grad():
            saliency_arr = _compute_attention_rollout(model, tensor, device)

        # Resample saliency back to original space
        saliency_sitk = sitk.GetImageFromArray(saliency_arr.astype(np.float32))
        saliency_sitk.CopyInformation(resampled)
        saliency_original = _resample_to_original_space(saliency_sitk, original_info)

        # Save as NIfTI
        nib_img = _sitk_to_nibabel(saliency_original)
        nib.save(nib_img, str(out_path))
        output_paths.append(str(out_path))

    return output_paths


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_brainiac_model(weights_path: Path, device: str):
    """Loads the BrainIAC ViT backbone from a .ckpt checkpoint.

    MONAI 1.5 renamed pos_embed -> proj_type (patch projection) and added
    pos_embed_type (position embedding style). save_attn=True stores attention
    weights on each block's .attn.att_mat after every forward pass, which is
    used directly by _compute_attention_rollout without needing hooks.
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
        save_attn=True,             # stores att_mat on each SABlock after forward
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
    if missing:
        print(f"[brainiac] Warning: {len(missing)} keys not loaded from checkpoint "
              f"(may be MONAI API rename): {missing[:3]}{'...' if len(missing) > 3 else ''}")

    return model


def _resample_to_brainiac_space(sitk_img):
    """Resamples a SimpleITK image to 96^3 at 1mm isotropic. Returns (resampled, original_info)."""
    import SimpleITK as sitk

    original_info = {
        "size": sitk_img.GetSize(),
        "spacing": sitk_img.GetSpacing(),
        "origin": sitk_img.GetOrigin(),
        "direction": sitk_img.GetDirection(),
    }

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([1.0, 1.0, 1.0])
    resampler.SetSize([96, 96, 96])
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)

    # Adjust origin to center the resampled volume within the original FOV
    orig_spacing = sitk_img.GetSpacing()
    orig_size = sitk_img.GetSize()
    center_orig = [
        sitk_img.GetOrigin()[i] + (orig_size[i] * orig_spacing[i]) / 2.0
        for i in range(3)
    ]
    new_origin = [center_orig[i] - 48.0 for i in range(3)]
    resampler.SetOutputOrigin(new_origin)

    return resampler.Execute(sitk_img), original_info


def _resample_to_original_space(saliency_sitk, original_info):
    """Resamples a saliency map back to the original scan space."""
    import SimpleITK as sitk

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(original_info["spacing"])
    resampler.SetSize(original_info["size"])
    resampler.SetOutputOrigin(original_info["origin"])
    resampler.SetOutputDirection(original_info["direction"])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(saliency_sitk)


def _normalize_volume(arr):
    """Z-score normalization; returns float32."""
    import numpy as np
    std = arr.std()
    if std > 0:
        arr = (arr - arr.mean()) / std
    return arr.astype("float32")


def _compute_attention_rollout(model, tensor, device):
    """
    Computes a 3D saliency map via attention rollout from BrainIAC's ViT backbone.
    Returns a float32 numpy array of shape (96, 96, 96).

    MONAI 1.5 ViT (with save_attn=True) stores attention weights directly on
    each block as block.attn.att_mat with shape (batch, heads, patches, patches).
    There is no CLS token in MONAI's ViT when classification=False, so rollout
    produces a (patches, patches) matrix; per-patch importance is the mean
    attention received across all query positions (column mean of the rollout).
    """
    import torch
    import numpy as np

    num_patches = 6 * 6 * 6  # 216 patches for 96^3 input / 16^3 patch size

    with torch.no_grad():
        _ = model(tensor)

    # Collect per-layer attention matrices stored by save_attn=True
    # att_mat shape: (batch, heads, num_patches, num_patches)
    attentions = [block.attn.att_mat.detach().cpu() for block in model.blocks]

    if not attentions or attentions[0].shape[-1] != num_patches:
        return np.ones((96, 96, 96), dtype=np.float32)

    # Attention rollout across layers:
    # For each layer: avg over heads, add residual identity, row-normalise.
    # Multiply layer matrices together to propagate attention through depth.
    rollout = torch.eye(num_patches)
    for attn in attentions:
        attn_avg = attn[0].mean(0)                         # (P, P) avg over heads
        attn_avg = attn_avg + torch.eye(num_patches)       # residual connection
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)  # row-normalise
        rollout = torch.mm(attn_avg, rollout)

    # Per-patch importance = mean attention received across all query positions
    # (column mean); equivalent to: how much does each patch get attended to?
    patch_importance = rollout.mean(dim=0)  # (num_patches,) = (216,)

    # Reshape to spatial grid (6, 6, 6) and upsample to (96, 96, 96)
    patch_grid = patch_importance.reshape(1, 1, 6, 6, 6).float()
    saliency_3d = torch.nn.functional.interpolate(
        patch_grid, size=(96, 96, 96), mode='trilinear', align_corners=False
    )
    saliency_arr = saliency_3d[0, 0].numpy()

    # Normalise to [0, 1]
    sal_min, sal_max = saliency_arr.min(), saliency_arr.max()
    if sal_max > sal_min:
        saliency_arr = (saliency_arr - sal_min) / (sal_max - sal_min)

    return saliency_arr.astype(np.float32)


def _sitk_to_nibabel(sitk_img):
    """Converts a SimpleITK image to a nibabel Nifti1Image."""
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(sitk_img)  # (z, y, x) in SimpleITK
    arr = np.transpose(arr, (2, 1, 0))      # -> (x, y, z) for nibabel

    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)

    affine = np.eye(4)
    affine[:3, :3] = direction * np.array(spacing)
    affine[:3, 3] = origin

    return nib.Nifti1Image(arr, affine)

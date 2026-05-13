"""
Utilities for downloading BrainIAC model weights and extracting saliency maps
for use as additional input channels during training and inference.

BrainIAC reference:
  "A generalizable foundation model for analysis of human brain MRI"
  Nature Neuroscience, Feb 2026. https://doi.org/10.1038/s41593-026-02202-6
  GitHub: https://github.com/AIM-KannLab/BrainIAC
  HuggingFace: https://huggingface.co/Divytak/brainiac

License note: BrainIAC weights are distributed under a custom non-commercial
research-only license. Users must agree to the BrainIAC license on HuggingFace
before downloading. Astril does not redistribute BrainIAC weights.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from .paths import preferred_models_dir

BRAINIAC_HF_REPO = "Divytak/brainiac"
# Exact filename resolved at download time via huggingface_hub.list_repo_files().
# This fallback name is used only when the repo has exactly one .ckpt file.
_BRAINIAC_CKPT_FALLBACK = "BrainIAC.ckpt"

_MISSING_WEIGHTS_MSG = (
    "BrainIAC model weights not found locally.\n\n"
    "To use BrainIAC embeddings you must do one of the following:\n\n"
    "  (a) Provide --HF_Token <your_token> so astril can download the weights\n"
    "      automatically from HuggingFace. Obtain a free token at:\n"
    "      https://huggingface.co/settings/tokens\n"
    "      You may also need to accept the BrainIAC license agreement at:\n"
    "      https://huggingface.co/Divytak/brainiac\n\n"
    "  (b) Manually download the BrainIAC checkpoint from the link above,\n"
    "      then pass the local file path via --BrainIAC_Weights_Path <path>.\n\n"
    "A free HuggingFace account is required."
)

_MISSING_DEPS_MSG = (
    "BrainIAC feature extraction requires optional dependencies. "
    "Install them with:\n\n"
    "  pip install astril[brainiac]\n\n"
    "or manually:\n\n"
    "  pip install torch>=2.0 monai>=1.3.2 huggingface_hub>=0.20 einops>=0.7"
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


def download_brainiac_weights(hf_token: str, dest_dir: Optional[Path] = None) -> Path:
    """
    Downloads the BrainIAC checkpoint from HuggingFace using huggingface_hub.
    Returns the local path to the downloaded file.
    Raises BrainIACWeightsNotFoundError with a clear message on auth failure.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
        from requests.exceptions import HTTPError
    except ImportError:
        raise ImportError(_MISSING_DEPS_MSG)

    dest = dest_dir or _brainiac_cache_dir()

    # Resolve the exact checkpoint filename from the repo
    try:
        repo_files = list(list_repo_files(BRAINIAC_HF_REPO, token=hf_token))
        ckpt_files = [f for f in repo_files if f.endswith(".ckpt")]
        if not ckpt_files:
            raise BrainIACWeightsNotFoundError(
                f"No .ckpt files found in HuggingFace repo '{BRAINIAC_HF_REPO}'. "
                "The repository structure may have changed. Check: "
                f"https://huggingface.co/{BRAINIAC_HF_REPO}"
            )
        filename = ckpt_files[0]
        if len(ckpt_files) > 1:
            print(
                f"[brainiac] Multiple checkpoints found in repo: {ckpt_files}. "
                f"Using '{filename}'."
            )
    except (RepositoryNotFoundError, HTTPError) as e:
        raise BrainIACWeightsNotFoundError(
            f"Could not access HuggingFace repo '{BRAINIAC_HF_REPO}'. "
            "Verify your HF_Token is valid and you have accepted the BrainIAC "
            f"license at https://huggingface.co/{BRAINIAC_HF_REPO}\n"
            f"Original error: {e}"
        )

    print(f"[brainiac] Downloading '{filename}' from {BRAINIAC_HF_REPO}...")
    try:
        local_path = hf_hub_download(
            repo_id=BRAINIAC_HF_REPO,
            filename=filename,
            token=hf_token,
            local_dir=str(dest),
        )
    except (EntryNotFoundError, HTTPError) as e:
        raise BrainIACWeightsNotFoundError(
            f"Failed to download BrainIAC weights. "
            f"Check your token and license acceptance.\nError: {e}"
        )

    final_path = dest / filename
    if Path(local_path).resolve() != final_path.resolve():
        shutil.copy2(local_path, str(final_path))

    print(f"[brainiac] Weights saved to: {final_path}")
    return final_path


def ensure_brainiac_weights(
    hf_token: Optional[str],
    weights_path: Optional[str],
) -> Path:
    """
    Resolves BrainIAC weights using the following priority:
      1. User-supplied weights_path (copied into cache)
      2. Previously cached weights
      3. Download from HuggingFace using hf_token
      4. Raise BrainIACWeightsNotFoundError with instructions
    """
    if weights_path is not None:
        return copy_user_weights(weights_path)

    cached = get_cached_weights_path()
    if cached is not None:
        print(f"[brainiac] Using cached weights: {cached}")
        return cached

    if hf_token is not None:
        return download_brainiac_weights(hf_token)

    raise BrainIACWeightsNotFoundError(_MISSING_WEIGHTS_MSG)


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
    """Loads the BrainIAC ViT backbone from a .ckpt checkpoint."""
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
        pos_embed="conv",
        classification=False,
        dropout_rate=0.0,
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
        print(f"[brainiac] Warning: missing keys in checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")

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
    """
    import torch
    import numpy as np

    # Register hooks to capture attention weights from each transformer block
    attentions = []

    def _hook(module, input, output):
        # MONAI ViT SABlock returns (x, attn_weights) or just x depending on version
        if isinstance(output, tuple) and len(output) == 2:
            attentions.append(output[1].detach().cpu())

    hooks = []
    for block in model.blocks:
        hooks.append(block.attn.register_forward_hook(_hook))

    with torch.no_grad():
        _ = model(tensor)

    for h in hooks:
        h.remove()

    if not attentions:
        # Fallback: return uniform saliency map if hooks captured nothing
        return np.ones((96, 96, 96), dtype=np.float32)

    # Attention rollout: multiply attention maps across layers
    # Each attn: (batch, heads, num_patches+1, num_patches+1)
    num_patches = 6 * 6 * 6  # 216 patches for 96^3 / 16^3
    rollout = torch.eye(num_patches + 1)
    for attn in attentions:
        # Average over heads, add identity (residual connection)
        attn_avg = attn[0].mean(0)  # (num_patches+1, num_patches+1)
        attn_avg = attn_avg + torch.eye(num_patches + 1)
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        rollout = torch.mm(attn_avg, rollout)

    # CLS token row: attention from CLS to all patches
    cls_attn = rollout[0, 1:]  # (num_patches,) = (216,)
    # Reshape to 3D spatial grid (6, 6, 6) and upsample to (96, 96, 96)
    patch_grid = cls_attn.reshape(1, 1, 6, 6, 6).float()
    saliency_3d = torch.nn.functional.interpolate(
        patch_grid, size=(96, 96, 96), mode='trilinear', align_corners=False
    )
    saliency_arr = saliency_3d[0, 0].numpy()

    # Normalize to [0, 1]
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

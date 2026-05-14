import os
import random
import shutil
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from concurrent.futures import ThreadPoolExecutor

# Nibabel orientation imports:
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
from nibabel.funcs import as_closest_canonical

from .config import n_cores, minimum_height_width


class Sequence:
    """Minimal sequence protocol used by astril's NumPy batch generators."""

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

# -------------------------------------------------
# Basic I/O
# -------------------------------------------------
def load_nifti_image(file_path):
    """
    Load a NIfTI image, returning a numpy array.
    """
    img = nib.load(file_path, mmap=False)
    data = np.asarray(img.dataobj, dtype=np.float32)
    del img  # optional
    return data

def read_paths_from_file(file_path):
    """
    Read lines (paths) from a text file.
    """
    with open(file_path, 'r') as file:
        paths = [line.strip() for line in file.readlines()]
    return paths

# -----------------------------------------------------------------
# Functions for canonical orientation + reversing it
# -----------------------------------------------------------------
def load_nifti_canonical_with_transform(file_path):
    """
    Load a NIfTI image, reorient it to canonical (RAS) space,
    and return:
      - data: the canonical-oriented volume as a NumPy array
      - transform_to_canonical: a nibabel orientation transform
        that was used to transform to canonical orientation.
      - transform_from_canonical: a nibabel orientation transform
        that can be used to invert back to the original orientation.
      - original_affine: the original 4x4 affine.
      - canonical_affine: the 4x4 affine after canonical transform.
    """
    original_img = nib.load(file_path)
    original_affine = original_img.affine.copy()
    # Reorient to canonical orientation
    canonical_img = as_closest_canonical(original_img)
    canonical_affine = canonical_img.affine.copy()

    # Compute orientation transforms for later reversal
    ornt_original = io_orientation(original_affine)
    ornt_canonical = io_orientation(canonical_affine)
    
    transform_to_canonical = ornt_transform(ornt_original, ornt_canonical)
    transform_from_canonical = ornt_transform(ornt_canonical, ornt_original)

    data = np.asarray(canonical_img.dataobj, dtype=np.float32)
    return data, transform_to_canonical, transform_from_canonical, original_affine, canonical_affine

def reorder_axes_for_plane(volume, plane='axial'):
    """
    Reorder axes so that the dimension we slice over is the last axis.
    For canonical data shape [X, Y, Z], we get [H, W, D].
    plane='axial'    -> (0,1,2)   (Z is last)
    plane='sagittal' -> (1,2,0)   (X is last)
    plane='coronal'  -> (0,2,1)   (Y is last)
    """
    reorder_dict = {
        'axial':    (0, 1, 2),
        'sagittal': (1, 2, 0),
        'coronal':  (0, 2, 1),
    }
    if plane not in reorder_dict:
        raise ValueError(f"Invalid slicing plane: {plane}")

    axes_order = reorder_dict[plane]
    return np.transpose(volume, axes_order), axes_order

def undo_reorder_axes(volume, axes_order):
    """
    Invert the axes reordering for a 3D or 4D volume.
    Handles 4D volumes (with a class/channel dimension) by ignoring the last dimension.
    
    Args:
        volume: np.ndarray, the array to reorder back.
        axes_order: tuple, the original reorder axes (for the first 3 dimensions).

    Returns:
        np.ndarray: Volume reordered back to its original axes.
    """
    ndim = volume.ndim
    if ndim < 3 or ndim > 4:
        raise ValueError(f"Invalid volume dimensions {ndim}. Expected 3D or 4D array.")

    if len(axes_order) != 3:
        raise ValueError(f"axes_order must have 3 elements, but got {axes_order}.")

    # Compute inverse permutation for the first 3 dimensions
    inverse = [0] * 3
    for i, a in enumerate(axes_order):
        inverse[a] = i

    # For 4D volumes, preserve the last dimension (classes/channels)
    if ndim == 4:
        inverse.append(3)  # Add the last dimension as-is

    # Transpose the volume back
    return np.transpose(volume, tuple(inverse))

# -------------------------------------------------
# Volume Adjustments (Padding)
# -------------------------------------------------
def adjust_volume_dimensions(volume, target_height=minimum_height_width, target_width=minimum_height_width):
    """
    Pad the [H, W] dimensions so they're at least target_height/width.
    volume shape is assumed [H, W, D].
    Returns (padded_volume, pad_info).
    """
    current_height, current_width, depth = volume.shape
    padding_height = max(target_height - current_height, 0)
    padding_width = max(target_width - current_width, 0)

    pad_top = padding_height // 2
    pad_bottom = padding_height - pad_top
    pad_left = padding_width // 2
    pad_right = padding_width - pad_left

    padded_volume = np.pad(
        volume,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0
    )
    pad_info = {
        'pad_top': pad_top, 'pad_bottom': pad_bottom,
        'pad_left': pad_left, 'pad_right': pad_right
    }
    return padded_volume, pad_info

def undo_adjust_dimensions(volume, pad_info):
    """
    Undo the zero-padding from adjust_volume_dimensions using stored pad_info.
    """

    pt, pb, pl, pr = pad_info['pad_top'], pad_info['pad_bottom'], pad_info['pad_left'], pad_info['pad_right']
    H = volume.shape[0] - pb
    W = volume.shape[1] - pr

    if H <= 0 or W <= 0:
        raise ValueError("Padding removal resulted in invalid dimensions!")

    return volume[pt:H, pl:W, :]

def pad_volume_edges(volume, pad_slices):
    """
    Zero-pad along the last axis (D dimension).
    """
    return np.pad(
        volume,
        pad_width=((0, 0), (0, 0), (pad_slices, pad_slices)),
        mode='constant',
        constant_values=0
    )

def undo_pad_volume_edges(volume, pad_slices):
    """
    Undo zero-padding on the slice (third) axis.
    Handles 3D volumes (H, W, D) and 4D volumes (H, W, D, C).
    """

    # Prevent removing all slices
    if pad_slices == 0:
        return volume  # No padding, so return unchanged

    if volume.shape[2] <= 2 * pad_slices:
        print("WARNING: Padding removal would result in empty depth! Adjusting pad_slices.")
        pad_slices = max(volume.shape[2] // 2 - 1, 0)

    # Apply padding removal only if it won't collapse depth
    if volume.shape[2] > 2 * pad_slices:
        if volume.ndim == 3:  # 3D volume
            return volume[:, :, pad_slices:-pad_slices]
        elif volume.ndim == 4:  # 4D volume
            return volume[:, :, pad_slices:-pad_slices, :]
    else:
        print("ERROR: Depth collapse detected! Returning original volume.")
        return volume  # Return unchanged if depth would collapse

# -------------------------------------------------
# Debug Helpers
# -------------------------------------------------
def debug_plot_slices(volume, num_slices=5, plane='axial'):
    """
    Quick slices visualization for debugging orientation.
    """

    d = volume.shape[-1]
    step = max(d // num_slices, 1)
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
    for i, ax in enumerate(axes):
        z = i * step
        ax.imshow(volume[..., z], cmap='gray', origin='lower')
        ax.set_title(f"{plane} slice {z}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def debug_plot_25d_slices(
    slices_list,
    num_to_show=10,
    channel_or_slice_idx=0,
    title_prefix="Debug 2.5D Slice"
):
    """
    Display up to `num_to_show` slices in grayscale from slices_list,
    where each element of slices_list has shape [H, W, C] 
    (C = number of channels or combined input-slices * channels).
    
    We only display a single 2D channel/plane from each 2.5D slice:
      - `channel_or_slice_idx` indicates which channel (or slice)
        in the last dimension to visualize.

    Example usage:
      debug_plot_25d_slices(my_X_scan_data, num_to_show=5, channel_or_slice_idx=0)

    Arguments:
      slices_list: list of arrays, each shape (H, W, Channels)
      num_to_show: how many slices to display
      channel_or_slice_idx: which channel or sub-slice to display in grayscale
      title_prefix: prefix for the subplot titles
    """
    if not slices_list:
        print("No slices available to plot.")
        return

    # Limit how many slices we actually show
    num_to_show = min(num_to_show, len(slices_list))

    fig, axes = plt.subplots(1, num_to_show, figsize=(4*num_to_show, 4))

    # If we only have 1 to show, axes will not be a list
    if num_to_show == 1:
        axes = [axes]

    for i in range(num_to_show):
        # Each item is a 2.5D slice => shape (H, W, [C]).
        slice_array = slices_list[i]
        if slice_array.ndim < 3:
            # If it is 2D, just show it as is
            img_2d = slice_array
        else:
            # Otherwise select the requested channel or sub-slice
            if channel_or_slice_idx >= slice_array.shape[-1]:
                print(f"Warning: channel_or_slice_idx={channel_or_slice_idx} "
                      f"out of range for slice shape {slice_array.shape}. Clamping to last channel.")
                channel_or_slice_idx = slice_array.shape[-1] - 1
            img_2d = slice_array[..., channel_or_slice_idx]

        ax = axes[i]
        ax.imshow(img_2d, cmap="gray", origin="lower")
        ax.set_title(f"{title_prefix} #{i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# The main alignment function (per-file)
# -------------------------------------------------
def robust_align_volume(file_path, plane, pad_amt, enforce_canonical=True, target_height=minimum_height_width, target_width=minimum_height_width):
    """
    Load 1 NIfTI file, optionally canonical-ize, reorder axes to plane,
    and pad [H,W] + pad the slice dimension by pad_amt.

    Returns (data_3d, transform_info).
      data_3d has shape [H, W, D].
    """
    transform_info = {
        'plane': plane,
        'pad_slices': pad_amt,
        'axes_order': None,
        'pad_info': None,
        'transform_to_canonical': None,
        'transform_from_canonical':None,
        'original_affine': None,
        'canonical_affine': None
    }

    # 1) Load + canonical
    if enforce_canonical:
        data, xf_to_canonical, xf_from_canonical, aff_orig, aff_can = load_nifti_canonical_with_transform(file_path)
        transform_info['transform_to_canonical'] = xf_to_canonical
        transform_info['transform_from_canonical'] = xf_from_canonical
        transform_info['original_affine'] = aff_orig
        transform_info['canonical_affine'] = aff_can
    else:
        data = load_nifti_image(file_path)

    # 2) Reorder so that plane is last axis
    data_reordered, axes_order = reorder_axes_for_plane(data, plane)
    transform_info['axes_order'] = axes_order

    # 3) Pad to minimum height/width
    data_adj, pad_info = adjust_volume_dimensions(data_reordered, target_height=target_height, target_width=target_width)
    transform_info['pad_info'] = pad_info

    # 4) Pad slice dimension
    data_final = pad_volume_edges(data_adj, pad_amt)

    return data_final, transform_info


def _reorder_brainiac_axes(feature_volume, plane):
    """Reorder canonical BrainIAC grid axes so the slice axis is third."""
    axes_order = {
        'axial': (0, 1, 2),
        'sagittal': (1, 2, 0),
        'coronal': (0, 2, 1),
    }.get(plane)
    if axes_order is None:
        raise ValueError(f"Invalid slicing plane: {plane}")
    return np.transpose(feature_volume, axes_order + (3,))


def _pad_brainiac_hw(feature_volume, target_h, target_w):
    """Pad/crop patch-grid H/W axes to a stable per-sample size."""
    h, w, d, c = feature_volume.shape
    out = np.zeros((target_h, target_w, d, c), dtype=np.float32)
    copy_h = min(h, target_h)
    copy_w = min(w, target_w)
    src_h0 = max((h - target_h) // 2, 0)
    src_w0 = max((w - target_w) // 2, 0)
    dst_h0 = max((target_h - h) // 2, 0)
    dst_w0 = max((target_w - w) // 2, 0)
    out[dst_h0:dst_h0 + copy_h, dst_w0:dst_w0 + copy_w] = feature_volume[
        src_h0:src_h0 + copy_h,
        src_w0:src_w0 + copy_w,
    ]
    return out


def load_brainiac_feature_volumes(
    brainiac_paths,
    plane,
    image_shape,
    pad_amt,
    target_height=minimum_height_width,
    target_width=minimum_height_width,
):
    """Load canonical patch-token grids and align them to astril's slicing plane."""
    if not brainiac_paths:
        return None

    target_h = int(np.ceil(target_height / 16.0))
    target_w = int(np.ceil(target_width / 16.0))
    volumes = []
    for path in brainiac_paths:
        feat = np.load(path).astype(np.float32)  # (X_p,Y_p,Z_p,C)
        feat = _reorder_brainiac_axes(feat, plane)
        feat = _pad_brainiac_hw(feat, target_h, target_w)
        volumes.append(feat)
    return np.concatenate(volumes, axis=-1)


def brainiac_slice_for_center(feature_volume, z_center, pad_amt):
    """Return the nearest patch-plane features for a padded image-slice center."""
    z_unpadded = max(int(z_center) - int(pad_amt), 0)
    patch_z = min(z_unpadded // 16, feature_volume.shape[2] - 1)
    return feature_volume[:, :, patch_z, :]

def undo_all_transforms(volume_3D, transform_info):
    """
    Inverts robust_align_volume steps:
      - Undo slice-padding
      - Undo [H,W] padding
      - Undo reorder axes
    """


    # (1) Undo slice padding
    ps = transform_info['pad_slices']
    vol_unpad_slices = undo_pad_volume_edges(volume_3D, ps)

    # (2) Undo H/W dimension padding
    vol_unpad_hw = undo_adjust_dimensions(vol_unpad_slices, transform_info['pad_info'])

    # (3) Undo reorder
    axes_order = transform_info['axes_order']
    if axes_order is None:
        raise ValueError("axes_order is None. Ensure reorder_axes_for_plane was called.")

    vol_reoriented = undo_reorder_axes(vol_unpad_hw, axes_order)

    return vol_reoriented

def apply_inverse_canonical_4d(prob_4d, transform_from_canonical):
    """
    Given a 4D array (X, Y, Z, C) in canonical orientation,
    apply the inverse orientation 'transform_from_canonical'
    so that it matches the volume's original orientation.

    We use nibabel.orientations.apply_orientation(prob_4d, transform_from_canonical).
    If transform_from_canonical is None or identity, we return prob_4d unchanged.
    """
    if transform_from_canonical is None:
        return prob_4d  # no reorientation needed
    # shape => (X, Y, Z, C)
    reoriented = apply_orientation(prob_4d, transform_from_canonical)
    return reoriented

# -------------------------------------------------
# Helper functions (class presence checks, etc.)
# -------------------------------------------------
def slice_has_all_classes(slice_array, required_classes):
    for cls_ in required_classes:
        if not np.any(slice_array == cls_):
            return False
    return True

def slice_has_any_classes(slice_array, candidate_classes):
    for cls_ in candidate_classes:
        if np.any(slice_array == cls_):
            return True
    return False

def slice_has_none_classes(slice_array, forbidden_classes):
    for cls_ in forbidden_classes:
        if np.any(slice_array == cls_):
            return False
    return True

def passes_require_classes(combined_gt_slice, require_classes):
    for classes_tuple, mode in require_classes.items():
        if mode == "all":
            if not slice_has_all_classes(combined_gt_slice, classes_tuple):
                return False
        elif mode == "any":
            if not slice_has_any_classes(combined_gt_slice, classes_tuple):
                return False
        elif mode == "none":
            if not slice_has_none_classes(combined_gt_slice, classes_tuple):
                return False
        else:
            raise ValueError(f"Invalid require_classes mode: {mode}")
    return True

# -------------------------------------------------
# Validation Data Loading (2.5D, no augmentation) — kept for inference path
# -------------------------------------------------
def load_val_slices(
    idx,
    volume_paths_list,
    mask_paths,
    gt_paths,
    slicing_plane,
    num_input_slices,
    num_output_slices,
    return_transform_info=False,
    target_height=minimum_height_width,
    target_width=minimum_height_width,
    brainiac_paths_list=None,
):
    """
    Load all 2.5D slices from the scan at index `idx` (without augmentation),
    exactly as done in training/validation. 
    Returns:
      X_scan_data, y_scan_data, mask_scan_data, z_indices_used,
      [optional] mask_transform_info (for orientation inversion, etc.)
    """

    half_in = num_input_slices // 2
    half_out = num_output_slices // 2
    pad_amt = max(half_in, half_out)

    # 1) Robustly load + orient mask, capturing transform info
    mask_data, mask_info = robust_align_volume(
        mask_paths[idx],
        plane=slicing_plane,
        pad_amt=pad_amt,
        enforce_canonical=True,
        target_height=target_height,
        target_width=target_width
    )
    mask_info['post_alignment_shape'] = mask_data.shape

    # 2) Robustly load + orient all channels
    channel_volumes = []
    for ch_path in volume_paths_list[idx]:
        ch_data, _ = robust_align_volume(
            ch_path,
            plane=slicing_plane,
            pad_amt=pad_amt,
            enforce_canonical=True,
            target_height=target_height,
            target_width=target_width
        )
        channel_volumes.append(ch_data)
    brainiac_volume = None
    if brainiac_paths_list is not None:
        brainiac_volume = load_brainiac_feature_volumes(
            brainiac_paths_list[idx],
            slicing_plane,
            channel_volumes[0].shape,
            pad_amt,
            target_height=target_height,
            target_width=target_width,
        )

    # 3) Robustly load + orient ground-truth
    gt_data, _ = robust_align_volume(
        gt_paths[idx],
        plane=slicing_plane,
        pad_amt=pad_amt,
        enforce_canonical=True,
        target_height=target_height,
        target_width=target_width
    )

    # Identify relevant slices (mask > 0)
    relevant_z_indices = np.where(np.any(mask_data > 0, axis=(0,1)))[0]

    X_scan_data = []
    B_scan_data = []
    y_scan_data = []
    mask_scan_data = []
    z_indices_used = []  # store which z_center each slice corresponds to

    start_in = -half_in
    end_in   = start_in + num_input_slices
    start_out = -half_out
    end_out   = start_out + num_output_slices

    for z_center in relevant_z_indices:
        # (A) Build input window
        input_slices = []
        for offset in range(start_in, end_in):
            z_in = z_center + offset
            # each channel => 2D slice from channel_volumes[ch]
            ch_slices = [ch_vol[:, :, z_in] for ch_vol in channel_volumes]
            stacked_ch = np.stack(ch_slices, axis=-1)
            input_slices.append(stacked_ch)
        X_window = np.concatenate(input_slices, axis=-1)
        if brainiac_volume is not None:
            B_scan_data.append(brainiac_slice_for_center(brainiac_volume, z_center, pad_amt))

        # (B) Build label + mask window
        gt_slices = []
        mask_slices = []
        for offset in range(start_out, end_out):
            z_out = z_center + offset
            gt_slices.append(gt_data[:, :, z_out].astype(np.int32))
            mask_slices.append((mask_data[:, :, z_out] > 0.5).astype(np.int32))

        Y_window = np.stack(gt_slices, axis=-1)[..., np.newaxis]
        M_window = np.stack(mask_slices, axis=-1)[..., np.newaxis]

        X_scan_data.append(X_window)
        y_scan_data.append(Y_window)
        mask_scan_data.append(M_window)
        z_indices_used.append(z_center)

    if brainiac_paths_list is not None and not return_transform_info:
        return X_scan_data, B_scan_data, y_scan_data, mask_scan_data, z_indices_used
    if brainiac_paths_list is not None:
        return X_scan_data, B_scan_data, y_scan_data, mask_scan_data, z_indices_used, mask_info
    if not return_transform_info:
        return X_scan_data, y_scan_data, mask_scan_data, z_indices_used
    else:
        return X_scan_data, y_scan_data, mask_scan_data, z_indices_used, mask_info


# -------------------------------------------------
# Streaming infrastructure (replaces load_epoch_data / load_val_data)
# -------------------------------------------------

def _save_scan_volumes_to_temp(
    idx,
    volume_paths_list,
    mask_paths,
    gt_paths,
    slicing_plane,
    num_input_slices,
    num_output_slices,
    brainiac_paths_list,
    target_height,
    target_width,
    prefix="astril_vol_",
    temp_base_dir=None,
):
    """
    Load and orient all NIfTI volumes for one scan, then write them to a temp
    directory as memory-mappable .npy files.  Returns (tmp_dir, has_data, has_brainiac).
    Only paths/small objects are ever sent back through the IPC pipe, avoiding
    Windows WinError 1450.
    """
    import os as _os
    half_in  = num_input_slices  // 2
    half_out = num_output_slices // 2
    pad_amt  = max(half_in, half_out)
    has_brainiac = brainiac_paths_list is not None

    sample_name = _os.path.basename(mask_paths[idx]).replace('-brainmask.nii.gz', '')

    # Load mask first to find relevant z-indices
    mask_vol, _ = robust_align_volume(
        mask_paths[idx], plane=slicing_plane, pad_amt=pad_amt,
        enforce_canonical=True, target_height=target_height, target_width=target_width,
    )
    relevant_z = np.where(np.any(mask_vol > 0, axis=(0, 1)))[0].astype(np.int32)

    tmp_dir = tempfile.mkdtemp(prefix=prefix, dir=temp_base_dir)

    if len(relevant_z) == 0:
        return tmp_dir, False, has_brainiac

    # Load all channels and stack to (num_channels, H, W, D)
    ch_vols = []
    for ch_path in volume_paths_list[idx]:
        ch, _ = robust_align_volume(
            ch_path, plane=slicing_plane, pad_amt=pad_amt,
            enforce_canonical=True, target_height=target_height, target_width=target_width,
        )
        ch_vols.append(ch)
    vols = np.stack(ch_vols, axis=0).astype(np.float32)  # (C, H, W, D)

    gt_vol, _ = robust_align_volume(
        gt_paths[idx], plane=slicing_plane, pad_amt=pad_amt,
        enforce_canonical=True, target_height=target_height, target_width=target_width,
    )

    np.save(_os.path.join(tmp_dir, 'vols.npy'), vols)
    np.save(_os.path.join(tmp_dir, 'gt.npy'),   gt_vol.astype(np.int32))
    np.save(_os.path.join(tmp_dir, 'mask.npy'), (mask_vol > 0.5).astype(np.uint8))
    np.savez(
        _os.path.join(tmp_dir, 'meta.npz'),
        pad_amt      = np.int32(pad_amt),
        num_channels = np.int32(len(ch_vols)),
        sample_name  = np.array([sample_name], dtype=object),
        relevant_z   = relevant_z,
    )

    if has_brainiac:
        brainiac_vol = load_brainiac_feature_volumes(
            brainiac_paths_list[idx], slicing_plane, vols.shape[1:],
            pad_amt, target_height=target_height, target_width=target_width,
        )
        if brainiac_vol is not None:
            np.save(_os.path.join(tmp_dir, 'brainiac.npy'), brainiac_vol.astype(np.float32))

    return tmp_dir, True, has_brainiac


def _dataloader_worker_init(worker_id):
    """Seed numpy and random in each DataLoader worker for independent augmentation."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class AstrilSliceDataset(torch.utils.data.Dataset):
    """
    Streams 2.5D slice windows from pre-loaded, memory-mapped NIfTI volumes.

    scan_temp_dirs: list of paths created by _save_scan_volumes_to_temp.
    Only file paths and small scalars/dicts are stored, so the Dataset pickles
    cleanly into DataLoader workers on Windows (spawn mode).
    """

    def __init__(
        self,
        scan_temp_dirs,
        num_input_slices,
        num_output_slices,
        class_multiplication_factors,
        require_classes,
        is_training,
        use_flip_augmentation,
        use_intensity_augmentation,
        intensity_augmentation_strength,
        has_brainiac,
    ):
        self.scan_temp_dirs               = list(scan_temp_dirs)
        self.num_input_slices             = num_input_slices
        self.num_output_slices            = num_output_slices
        self.class_multiplication_factors = dict(class_multiplication_factors or {})
        self.require_classes              = dict(require_classes or {})
        self.is_training                  = is_training
        self.use_flip_augmentation        = use_flip_augmentation
        self.use_intensity_augmentation   = use_intensity_augmentation
        self.intensity_augmentation_strength = intensity_augmentation_strength
        self.has_brainiac                 = has_brainiac
        self._index = []          # list of (scan_idx, z_center)
        self.class_weights = None
        self._build_index()

    def _build_index(self):
        import os as _os
        half_out    = self.num_output_slices // 2
        class_counts = {}

        for scan_idx, tmp_dir in enumerate(self.scan_temp_dirs):
            meta       = np.load(_os.path.join(tmp_dir, 'meta.npz'), allow_pickle=True)
            relevant_z = meta['relevant_z']
            gt_vol     = np.load(_os.path.join(tmp_dir, 'gt.npy'),   mmap_mode='r')
            mask_vol   = np.load(_os.path.join(tmp_dir, 'mask.npy'), mmap_mode='r')

            for z_center in relevant_z:
                z = int(z_center)
                gt_center = gt_vol[:, :, z].astype(np.int32)

                if self.require_classes:
                    if not passes_require_classes(gt_center, self.require_classes):
                        continue

                # Oversample based on class_multiplication_factors (replaces rotation augmentation)
                n_copies = 1
                for key_tuple, factor_val in self.class_multiplication_factors.items():
                    if slice_has_all_classes(gt_center, key_tuple):
                        n_copies = 1 + int(factor_val)
                        break

                for _ in range(n_copies):
                    self._index.append((scan_idx, z))

                # Accumulate class pixel counts for class weight computation
                mask_center = mask_vol[:, :, z] > 0
                if mask_center.any():
                    unique, cnts = np.unique(gt_center[mask_center], return_counts=True)
                    for cls, cnt in zip(unique.tolist(), cnts.tolist()):
                        class_counts[cls] = class_counts.get(cls, 0) + cnt

        if len(self._index) == 0:
            print("WARNING: AstrilSliceDataset has zero valid slice entries after filtering.")

        # Inverse-frequency class weights
        if class_counts:
            n_cls  = max(class_counts.keys()) + 1
            counts = np.array([class_counts.get(c, 0) for c in range(n_cls)], dtype=np.float64)
            total  = counts.sum() + 1e-8
            freqs  = counts / total
            weights = np.where(freqs > 0, 1.0 / freqs, 1.0)
            self.class_weights = np.clip(weights, 0, 1000.0).astype(np.float32)
        else:
            self.class_weights = np.ones(1, dtype=np.float32)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        import os as _os
        scan_idx, z_center = self._index[i]
        tmp_dir = self.scan_temp_dirs[scan_idx]

        meta         = np.load(_os.path.join(tmp_dir, 'meta.npz'), allow_pickle=True)
        pad_amt      = int(meta['pad_amt'])
        num_channels = int(meta['num_channels'])
        sample_name  = str(meta['sample_name'][0])

        vols     = np.load(_os.path.join(tmp_dir, 'vols.npy'),  mmap_mode='r')  # (C, H, W, D)
        gt_vol   = np.load(_os.path.join(tmp_dir, 'gt.npy'),    mmap_mode='r')
        mask_vol = np.load(_os.path.join(tmp_dir, 'mask.npy'),  mmap_mode='r')

        half_in  = self.num_input_slices  // 2
        half_out = self.num_output_slices // 2

        # Build X window: (H, W, C * num_input_slices)
        input_planes = []
        for offset in range(-half_in, -half_in + self.num_input_slices):
            z_in = z_center + offset
            ch_planes = [vols[c, :, :, z_in] for c in range(num_channels)]
            input_planes.append(np.stack(ch_planes, axis=-1))  # (H, W, C)
        X_window = np.concatenate(input_planes, axis=-1).copy().astype(np.float32)

        # Build Y and M windows: (H, W, num_output_slices, 1)
        gt_planes, mask_planes = [], []
        for offset in range(-half_out, -half_out + self.num_output_slices):
            z_out = z_center + offset
            gt_planes.append(gt_vol[:, :, z_out].astype(np.int32))
            mask_planes.append((mask_vol[:, :, z_out] > 0).astype(np.int32))
        Y_window = np.stack(gt_planes,   axis=-1)[..., np.newaxis].copy()
        M_window = np.stack(mask_planes, axis=-1)[..., np.newaxis].copy()

        # BrainIAC: sentinel zeros when absent so the tuple arity is always 5
        B_window = np.zeros((1,), dtype=np.float32)
        if self.has_brainiac:
            b_path = _os.path.join(tmp_dir, 'brainiac.npy')
            if _os.path.exists(b_path):
                brainiac_vol = np.load(b_path, mmap_mode='r')
                B_window = brainiac_slice_for_center(brainiac_vol, z_center, pad_amt).copy()

        # Augmentation (training only)
        if self.is_training:
            if self.use_flip_augmentation:
                for flip_axis in (0, 1):
                    if random.random() > 0.5:
                        X_window = np.flip(X_window, axis=flip_axis).copy()
                        Y_window = np.flip(Y_window, axis=flip_axis).copy()
                        M_window = np.flip(M_window, axis=flip_axis).copy()
                        if self.has_brainiac and B_window.ndim > 1:
                            B_window = np.flip(B_window, axis=flip_axis).copy()

            if self.use_intensity_augmentation:
                s   = self.intensity_augmentation_strength
                std = float(X_window.std())
                if std > 0:
                    X_window = (X_window + np.random.normal(0, s * std, X_window.shape)).astype(np.float32)
                contrast = 1.0 + float(np.random.uniform(-s, s))
                X_window = (X_window * contrast).astype(np.float32)

        return X_window, B_window, Y_window, M_window, sample_name


def compute_class_weights_from_dataset(dataset, num_classes):
    """Return per-class inverse-frequency weights computed during Dataset index build."""
    w = dataset.class_weights
    if len(w) < num_classes:
        padded = np.ones(num_classes, dtype=np.float32)
        padded[:len(w)] = w
        return padded
    return w[:num_classes].copy()


def load_epoch_dataset(
    scan_indexes,
    volume_paths_list,
    mask_paths,
    gt_paths,
    slicing_plane,
    num_input_slices,
    num_output_slices,
    class_multiplication_factors,
    require_classes,
    use_flip_augmentation,
    use_intensity_augmentation,
    intensity_augmentation_strength,
    brainiac_paths_list,
    target_height,
    target_width,
    executor,
    temp_base_dir=None,
):
    """
    Phase 1: load scan volumes in parallel via ProcessPoolExecutor into temp dirs.
    Phase 2: construct AstrilSliceDataset wrapping those dirs.
    Returns (dataset, temp_dirs).  Caller is responsible for shutil.rmtree on temp_dirs.
    temp_base_dir: parent directory for temp scan dirs (defaults to system temp).
    """
    has_brainiac = brainiac_paths_list is not None
    futures = [
        executor.submit(
            _save_scan_volumes_to_temp,
            idx, volume_paths_list, mask_paths, gt_paths,
            slicing_plane, num_input_slices, num_output_slices,
            brainiac_paths_list, target_height, target_width,
            "astril_vol_train_", temp_base_dir,
        )
        for idx in scan_indexes
    ]
    temp_dirs = []
    for future in futures:
        tmp_dir, has_data, _ = future.result()
        if has_data:
            temp_dirs.append(tmp_dir)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    dataset = AstrilSliceDataset(
        scan_temp_dirs=temp_dirs,
        num_input_slices=num_input_slices,
        num_output_slices=num_output_slices,
        class_multiplication_factors=class_multiplication_factors,
        require_classes=require_classes,
        is_training=True,
        use_flip_augmentation=use_flip_augmentation,
        use_intensity_augmentation=use_intensity_augmentation,
        intensity_augmentation_strength=intensity_augmentation_strength,
        has_brainiac=has_brainiac,
    )
    return dataset, temp_dirs


def load_val_dataset(
    scan_indexes,
    volume_paths_list,
    mask_paths,
    gt_paths,
    slicing_plane,
    num_input_slices,
    num_output_slices,
    brainiac_paths_list,
    target_height,
    target_width,
    executor,
    temp_base_dir=None,
):
    """Parallel volume loading for validation; returns (AstrilSliceDataset, temp_dirs)."""
    has_brainiac = brainiac_paths_list is not None
    futures = [
        executor.submit(
            _save_scan_volumes_to_temp,
            idx, volume_paths_list, mask_paths, gt_paths,
            slicing_plane, num_input_slices, num_output_slices,
            brainiac_paths_list, target_height, target_width,
            "astril_vol_val_", temp_base_dir,
        )
        for idx in scan_indexes
    ]
    temp_dirs = []
    for future in futures:
        tmp_dir, has_data, _ = future.result()
        if has_data:
            temp_dirs.append(tmp_dir)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    dataset = AstrilSliceDataset(
        scan_temp_dirs=temp_dirs,
        num_input_slices=num_input_slices,
        num_output_slices=num_output_slices,
        class_multiplication_factors={},
        require_classes={},
        is_training=False,
        use_flip_augmentation=False,
        use_intensity_augmentation=False,
        intensity_augmentation_strength=0.0,
        has_brainiac=has_brainiac,
    )
    return dataset, temp_dirs


def detect_input_shape(sample_file_path, slicing_plane, num_channels):
    """
    Detect a representative input shape (height, width, num_channels) 
    consistent with our robust alignment approach:
      - Reorient to canonical (RAS) if desired
      - Reorder axes so slicing_plane is last dimension
      - Ensure height/width >= minimum_height_width
      - Return the shape after these adjustments (except slice-padding).
    
    Returns (padded_input_shape, original_shape_for_logging).
      original_shape_for_logging is the shape after reorder but before padding,
      so you can see how much padding was added.
    """
    # 1) Load + canonical
    data, _, _, _, _ = load_nifti_canonical_with_transform(sample_file_path)

    # 2) Reorder so 'slicing_plane' is last axis
    data_reordered, _ = reorder_axes_for_plane(data, slicing_plane)

    # Shape after reorder (but before pad)
    oh, ow, od = data_reordered.shape
    original_shape_for_logging = (oh, ow, num_channels)  # we only show H/W in logs

    # 3) Pad to ensure min height/width
    data_adj, _ = adjust_volume_dimensions(data_reordered)
    h, w, d = data_adj.shape

    # Our final input shape for a single slice batch is (H, W, [channels]).
    # We do NOT know how many slices will be concatenated; we only know
    # the final model sees (H, W, num_input_slices*N_channels).
    # For logging, we return (h, w, num_channels).
    padded_input_shape = (h, w, num_channels)

    return padded_input_shape, original_shape_for_logging

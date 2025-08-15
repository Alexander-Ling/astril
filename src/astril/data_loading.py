import os
import gc
import random
import numpy as np
import nibabel as nib
import psutil
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import rotate

# Nibabel orientation imports:
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
from nibabel.funcs import as_closest_canonical

from .config import n_cores, minimum_height_width

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
# Data Generators
# -------------------------------------------------
class EpochDataGenerator(Sequence):
    def __init__(self, x_data, y_data, mask_data, sample_names, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.mask_data = mask_data
        self.batch_size = batch_size
        self.sample_names = sample_names

    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.x_data))
        if start >= len(self.x_data):
            raise StopIteration
        return (self.x_data[start:end],
                self.y_data[start:end],
                self.mask_data[start:end],
                self.sample_names[start:end])

class ValDataGenerator(Sequence):
    def __init__(self, x_data, y_data, mask_data, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.mask_data = mask_data
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.x_data))
        x_batch = self.x_data[start:end]
        mask_batch = self.mask_data[start:end]
        
        if self.y_data is not None:
            y_batch = self.y_data[start:end]
            return(x_batch, y_batch, mask_batch)
        else:
            y_batch = None
            return (x_batch, mask_batch)

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

def get_ram_usage_percent():
    return psutil.virtual_memory().percent

def append_or_replace(
    X_list, Y_list, M_list, sample_names,
    X_item, Y_item, M_item, sample_name,
    replace_index=None
):
    if replace_index is None:
        X_list.append(X_item)
        Y_list.append(Y_item)
        M_list.append(M_item)
        sample_names.append(sample_name)
    else:
        X_list[replace_index] = X_item
        Y_list[replace_index] = Y_item
        M_list[replace_index] = M_item
        sample_names[replace_index] = sample_name

# ------------------------------------------------- 
# Training Data Loading (2.5D) 
# ------------------------------------------------- 
def load_train_slices(
    idx,
    volume_paths_list,
    mask_paths,
    gt_paths,
    slicing_plane,
    num_input_slices,
    num_output_slices,
    class_multiplication_factors=None,
    require_classes=None
):
    """
    Load all 2.5D slices from the scan at index `idx` for training,
    with optional filtering and augmentation.
    """
    if class_multiplication_factors is None:
        class_multiplication_factors = {}
    if require_classes is None:
        require_classes = {}

    import os
    sample_name = os.path.basename(mask_paths[idx]).replace('-brainmask.nii.gz', '')

    # Decide how many slices to pad
    half_in = num_input_slices // 2
    half_out = num_output_slices // 2
    pad_amt = max(half_in, half_out)

    # 1) Robustly load + orient each channel
    channel_volumes = []
    for ch_path in volume_paths_list[idx]:
        ch_data, _ = robust_align_volume(
            ch_path,
            plane=slicing_plane,
            pad_amt=pad_amt,
            enforce_canonical=True
        )
        #debug_plot_slices(ch_data, plane = slicing_plane)
        channel_volumes.append(ch_data)

    # 2) Robustly load + orient mask
    mask_data, _ = robust_align_volume(
        mask_paths[idx],
        plane=slicing_plane,
        pad_amt=pad_amt,
        enforce_canonical=True
    )

    # 3) Robustly load + orient ground-truth
    gt_data, _ = robust_align_volume(
        gt_paths[idx],
        plane=slicing_plane,
        pad_amt=pad_amt,
        enforce_canonical=True
    )

    # Identify relevant slices (mask>0)
    relevant_z_indices = np.where(np.any(mask_data > 0, axis=(0,1)))[0]

    X_scan_data = []
    y_scan_data = []
    mask_scan_data = []
    scan_sample_names = []

    # Slice offsets
    start_in = -half_in
    end_in   = start_in + num_input_slices
    start_out = -half_out
    end_out   = start_out + num_output_slices

    for z_center in relevant_z_indices:
        # (A) Build input window
        input_slices = []
        for offset in range(start_in, end_in):
            z_in = z_center + offset
            ch_slices = [ch_vol[:, :, z_in] for ch_vol in channel_volumes]
            stacked_ch = np.stack(ch_slices, axis=-1)
            input_slices.append(stacked_ch)
        X_window = np.concatenate(input_slices, axis=-1)  

        # (B) Build label window
        gt_slices = []
        mask_slices = []
        for offset in range(start_out, end_out):
            z_out = z_center + offset
            gt_slices.append(gt_data[:, :, z_out].astype(np.int32))
            mask_slices.append((mask_data[:, :, z_out] > 0.5).astype(np.int32))
        Y_window = np.stack(gt_slices, axis=-1)[..., np.newaxis]  
        M_window = np.stack(mask_slices, axis=-1)[..., np.newaxis]

        # (C) Filter out if not containing required classes
        combined_gt = np.squeeze(Y_window, axis=-1)
        if not passes_require_classes(combined_gt, require_classes):
            continue

        # (D) Determine augmentation factor
        augmentation_factor = 0
        for key_tuple, factor_val in class_multiplication_factors.items():
            if slice_has_all_classes(combined_gt, key_tuple):
                augmentation_factor = factor_val
                break

        # (E) Check memory usage, append or replace
        ram_usage = get_ram_usage_percent()
        if ram_usage < 90.0:
            replace_index = None
        else:
            if random.random() < 0.5:
                # skip
                continue
            else:
                replace_index = (
                    random.randrange(len(X_scan_data))
                    if len(X_scan_data) > 0
                    else None
                )

        # Insert original
        append_or_replace(
            X_scan_data, y_scan_data, mask_scan_data, scan_sample_names,
            X_window, Y_window, M_window, sample_name,
            replace_index=replace_index
        )

        # (F) Augment if needed
        for _ in range(augmentation_factor):
            angle = np.random.uniform(0, 360)
            # Rotate input
            X_rot = rotate(X_window, angle, reshape=False, mode='constant', cval=0)
            # Rotate output
            Y_rot = np.zeros_like(Y_window)
            M_rot = np.zeros_like(M_window)
            for s in range(num_output_slices):
                y2d = Y_window[:, :, s, 0]
                m2d = M_window[:, :, s, 0]
                y2d_rot = rotate(y2d, angle, reshape=False, mode='constant', cval=0, order=0)
                m2d_rot = rotate(m2d, angle, reshape=False, mode='constant', cval=0, order=0)
                Y_rot[:, :, s, 0] = y2d_rot
                M_rot[:, :, s, 0] = m2d_rot

            ram_usage_aug = get_ram_usage_percent()
            if ram_usage_aug < 90.0:
                replace_index_aug = None
            else:
                if random.random() < 0.5:
                    continue
                else:
                    replace_index_aug = (
                        random.randrange(len(X_scan_data))
                        if len(X_scan_data) > 0
                        else None
                    )

            append_or_replace(
                X_scan_data, y_scan_data, mask_scan_data, scan_sample_names,
                X_rot, Y_rot, M_rot, sample_name,
                replace_index=replace_index_aug
            )

    return X_scan_data, y_scan_data, mask_scan_data, scan_sample_names

# -------------------------------------------------
# Validation Data Loading (2.5D, no augmentation)
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
    target_width=minimum_height_width
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

    if not return_transform_info:
        return X_scan_data, y_scan_data, mask_scan_data, z_indices_used
    else:
        return X_scan_data, y_scan_data, mask_scan_data, z_indices_used, mask_info

# -------------------------------------------------
# Multi-threaded loaders for an epoch
# -------------------------------------------------
def load_epoch_data(
    scan_indexes,
    volume_paths_list,
    mask_paths,
    gt_paths,
    slicing_plane,
    num_input_slices,
    num_output_slices,
    num_classes,
    class_multiplication_factors=None,
    require_classes=None,
    do_shuffle=True
):
    X_epoch_list = []
    y_epoch_list = []
    mask_epoch_list = []
    sample_names_list = []

    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        futures = [
            executor.submit(
                load_train_slices,
                idx,
                volume_paths_list,
                mask_paths,
                gt_paths,
                slicing_plane,
                num_input_slices,
                num_output_slices,
                class_multiplication_factors,
                require_classes
            )
            for idx in scan_indexes
        ]
        for future in futures:
            X_scan, y_scan, m_scan, names_scan = future.result()
            X_epoch_list.extend(X_scan)
            y_epoch_list.extend(y_scan)
            mask_epoch_list.extend(m_scan)
            sample_names_list.extend(names_scan)

    if do_shuffle and len(X_epoch_list) > 0:
        combined = list(zip(X_epoch_list, y_epoch_list, mask_epoch_list, sample_names_list))
        random.shuffle(combined)
        X_epoch_list, y_epoch_list, mask_epoch_list, sample_names_list = zip(*combined)
        del combined
        X_epoch_list = list(X_epoch_list)
        y_epoch_list = list(y_epoch_list)
        mask_epoch_list = list(mask_epoch_list)
        sample_names_list = list(sample_names_list)

    X_epoch_data = np.array(X_epoch_list, dtype=np.float32)
    y_epoch_data = np.array(y_epoch_list, dtype=np.uint8)
    mask_epoch_data = np.array(mask_epoch_list, dtype=bool)
    epoch_sample_names = np.array(sample_names_list, dtype=object)

    del X_epoch_list, y_epoch_list, mask_epoch_list, sample_names_list
    gc.collect()

    # Compute inverse-frequency class weights
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for y_slice, m_slice in zip(y_epoch_data, mask_epoch_data):
        valid_pixels = (m_slice > 0)
        y_valid_slice = y_slice[valid_pixels]
        for c in range(num_classes):
            class_counts[c] += np.count_nonzero(y_valid_slice == c)

    total_valid = np.sum(class_counts) + 1e-8
    freqs = class_counts / total_valid
    class_weights = np.zeros_like(freqs, dtype=np.float32)
    for c in range(num_classes):
        if freqs[c] > 0.0:
            class_weights[c] = 1.0 / freqs[c]
        else:
            class_weights[c] = 1.0
    class_weights = np.clip(class_weights, 0, 1000.0)

    return X_epoch_data, y_epoch_data, mask_epoch_data, epoch_sample_names, class_weights


def load_val_data(
    scan_indexes,
    volume_paths_list,
    mask_paths,
    gt_paths,
    slicing_plane,
    num_input_slices,
    num_output_slices,
    return_transform_info=False,
    target_height=minimum_height_width,
    target_width=minimum_height_width
):
    """
    Multi-threaded loader for validation (or test) 2.5D slices.

    If return_transform_info=True, then for each scan index we
    also return the mask_info from the first subject. Usually
    you'd only pass 1 subject in `scan_indexes` at a time
    during inference, but the code still supports multiple.
    """
    from concurrent.futures import ThreadPoolExecutor

    X_epoch_list = []
    y_epoch_list = []
    mask_epoch_list = []
    z_indices_all = []
    transform_infos = []

    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for idx in scan_indexes:
            futures.append(executor.submit(
                load_val_slices,
                idx,
                volume_paths_list,
                mask_paths,
                gt_paths,
                slicing_plane,
                num_input_slices,
                num_output_slices,
                return_transform_info,
                target_height=target_height,
                target_width=target_width
            ))

        for future in futures:
            result = future.result()
            if return_transform_info:
                (X_scan, y_scan, m_scan, z_inds, t_info) = result
                transform_infos.append(t_info)
            else:
                (X_scan, y_scan, m_scan, z_inds) = result

            X_epoch_list.extend(X_scan)
            y_epoch_list.extend(y_scan)
            mask_epoch_list.extend(m_scan)
            z_indices_all.extend(z_inds)

    X_epoch_data = np.array(X_epoch_list, dtype=np.float32)
    y_epoch_data = np.array(y_epoch_list, dtype=np.uint8)
    mask_epoch_data = np.array(mask_epoch_list, dtype=bool)
    z_indices_all = np.array(z_indices_all, dtype=np.int32)

    if return_transform_info:
        return (X_epoch_data, y_epoch_data, mask_epoch_data, z_indices_all, transform_infos)
    else:
        return (X_epoch_data, y_epoch_data, mask_epoch_data, z_indices_all)

        
# -------------------------------------------------
# Updated detect_input_shape to match robust approach
# -------------------------------------------------
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

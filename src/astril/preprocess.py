# preprocess.py
# Author: Alex Ling
# E. Antonio Chiocca Group, BWH
# Description: Preprocessing utilities for MRI normalization, resampling, etc.

# Description: Preprocessing utilities for MRI normalization, resampling, etc.
from __future__ import annotations

import os
import sys
import argparse
import shutil
import subprocess
import contextlib
import re
import ast
import json
import tempfile
import warnings
import uuid
import string
import csv as _csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# NOTE: Heavy dependencies (numpy, nibabel, SimpleITK, pandas, nilearn, scipy)
# and astril.preprocessing_utils are now imported *inside* the functions that need them.

# -----------------------------------------------------------------
# Function to normalize an MRI image using only the masked region
# -----------------------------------------------------------------

def normalize_masked_image(input_image_path, mask_path, output_path=None, zero_outside_mask=False):
    # Lazy imports
    import nibabel as nib
    import numpy as np
    """
    Normalize an MRI volume using the provided mask.
    Voxels inside the mask are zero-mean, unit-variance normalized.
    Voxels outside the mask are set to 0.

    Args:
        input_image_path (str): Path to the input NIfTI file
        mask_path (str): Path to the binary brain mask NIfTI file
        output_path (str, optional): If provided, saves the output image

    Returns:
        nib.Nifti1Image: The normalized image
    """
    img = nib.load(input_image_path)
    data = img.get_fdata()

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    if data.shape != mask_data.shape:
        raise ValueError(f"Shape mismatch: image {data.shape} vs mask {mask_data.shape}")

    brain_values = data[mask_data > 0]
    if brain_values.size == 0:
        raise ValueError("Mask contains no non-zero voxels!")

    mean = np.mean(brain_values)
    std = np.std(brain_values)
    if std == 0:
        raise ValueError("Standard deviation within mask is zero.")

    if zero_outside_mask:
        normalized_data = np.where(mask_data > 0, (data - mean) / std, 0)
    else:
        normalized_data = (data - mean) / std

    normalized_img = nib.Nifti1Image(normalized_data, affine=img.affine, header=img.header)

    if output_path:
        nib.save(normalized_img, output_path)

    return normalized_img


# -------------------------------------------------------------------------
# Function to reshape an MRI volume to specified data and voxel dimensions
# -------------------------------------------------------------------------

def resize_mri(
    input_filepath,
    output_filepath,
    target_shape,
    target_voxel_dims,
    interp,
    save_padding_record=False,
    padding_record_path=None,
    roi_mask_path=None,
    translation_only=False,
    reg_frame_index=0,   # <-- for 4D, which frame defines padding/shape decisions
):
    """
    Resize a 3D or 4D NIfTI to (target_shape, target_voxel_dims).

    - For 3D: identical behavior to your current implementation.
    - For 4D: loads entire 4D volume once, computes padding decisions once (using reg_frame_index),
      then processes each frame in memory and stacks into a 4D output.
    """
    # Lazy imports
    import nibabel as nib
    import numpy as np

    from .preprocessing_utils import (
        apply_padding_anydim,
        prepare_zoom,
        interpolate_to_voxel_dims,
        interpolate_to_voxel_dims_precomputed,
        update_origin_for_padding,
        adjust_to_target_shape,
        read_padding_record,
        load_roi_mask,
    )

    if not os.path.exists(input_filepath):
        raise ValueError(f"[Error] Attempting to resize {input_filepath}, but file does not exist.")

    img = nib.load(input_filepath)
    data = img.get_fdata()
    ndim = data.ndim

    if ndim not in (3, 4):
        raise ValueError(f"Unsupported ndim={ndim} for resize: {input_filepath}")

    header_zooms = img.header.get_zooms()
    original_voxel_dims = header_zooms[:3]  # spatial only

    # ---------------------------------------------------------------------
    # padding_record init (works for both 3D and 4D; stores spatial grid)
    # ---------------------------------------------------------------------
    padding_record = {
        "target_voxel_dims": tuple(target_voxel_dims),
        "target_shape": tuple(target_shape),
        "original_voxel_dims": tuple(original_voxel_dims),
        "original_shape": tuple(data.shape),
        "original_grid": {
            "size": list(data.shape[:3]),
            "spacing": list(original_voxel_dims),
            "origin": list(img.affine[:3, 3]),
            "direction": list(np.ravel(img.affine[:3, :3] / np.array(original_voxel_dims))),
        },
    }

    loaded_padding_record = None
    if padding_record_path and os.path.exists(padding_record_path):
        loaded_padding_record = read_padding_record(padding_record_path)

    # ROI mask should be 3D; for 4D we apply it based on spatial dims
    roi_mask = load_roi_mask(roi_mask_path, data.shape[:3]) if roi_mask_path else None

    # Select the 3D reference volume used to decide padding/shape for 4D
    if ndim == 3:
        ref_vol = data
    else:
        if not (0 <= reg_frame_index < data.shape[3]):
            raise ValueError(
                f"reg_frame_index={reg_frame_index} out of range for T={data.shape[3]}: {input_filepath}"
            )
        ref_vol = data[..., reg_frame_index]

    # ---------------------------------------------------------------------
    # Determine (or load) center_padding
    # ---------------------------------------------------------------------
    if roi_mask is not None or loaded_padding_record:
        if not loaded_padding_record:
            roi_indices = np.where(roi_mask > 0)
            if len(roi_indices[0]) == 0:
                raise ValueError(f"ROI mask appears empty: {roi_mask_path}")
            roi_center = (np.min(roi_indices, axis=1) + np.max(roi_indices, axis=1)) // 2
            data_center = np.array(ref_vol.shape) // 2
            translation = data_center - roi_center
            center_padding = np.zeros((3, 2), dtype=int)
            for dim, shift in enumerate(translation):
                center_padding[dim] = [shift, -shift]
        else:
            center_padding = np.array(loaded_padding_record["center_padding"], dtype=int)

        # Apply center padding to 3D or 4D in one shot (pads/crops only spatial axes)
        data = apply_padding_anydim(data, center_padding)
    else:
        center_padding = np.zeros((3, 2), dtype=int)

    padding_record["center_padding"] = center_padding.tolist()

    # Update affine origin to account for the voxel-index shift introduced by center padding/cropping.
    # IMPORTANT: This must happen even when translation_only=True, otherwise physical coordinates
    # will be inconsistent after we shift the data array.
    affine_after_center = update_origin_for_padding(img.affine.copy(), center_padding, original_voxel_dims)

    # ---------------------------------------------------------------------
    # If translation_only: skip interpolation + target shape adjustment
    # ---------------------------------------------------------------------
    if translation_only:
        padding_record["shape_padding"] = np.zeros((3, 2), dtype=int).tolist()
        final_data = data
        new_affine = affine_after_center

    else:
        # For 4D, compute shape_padding once using the reference frame, then reuse for all frames
        if ndim == 3:
            interpolated = interpolate_to_voxel_dims(data, original_voxel_dims, target_voxel_dims, interp)

            if loaded_padding_record:
                final_data, padding_record = adjust_to_target_shape(
                    interpolated,
                    target_shape,
                    padding_record,
                    np.array(loaded_padding_record["shape_padding"], dtype=int),
                )
            else:
                final_data, padding_record = adjust_to_target_shape(interpolated, target_shape, padding_record)

        else:
            # Precompute zoom factors/order once (avoid overhead per frame)
            zoom_factors, order = prepare_zoom(original_voxel_dims, target_voxel_dims, interp)

            # Interpolate reference frame first to determine (or validate) shape padding
            interp_ref = interpolate_to_voxel_dims_precomputed(ref_vol, zoom_factors, order)

            if loaded_padding_record:
                shape_padding = np.array(loaded_padding_record["shape_padding"], dtype=int)
                _ref_resized, padding_record = adjust_to_target_shape(
                    interp_ref,
                    target_shape,
                    padding_record,
                    shape_padding,
                )
            else:
                _ref_resized, padding_record = adjust_to_target_shape(interp_ref, target_shape, padding_record)
                shape_padding = np.array(padding_record["shape_padding"], dtype=int)

            # Interpolate all frames into a preallocated array, then apply shape padding ONCE to the 4D stack
            T = int(data.shape[3])
            interp_shape3 = tuple(interp_ref.shape)
            interped = np.empty((*interp_shape3, T), dtype=np.float32)
            interped[..., 0] = interp_ref.astype(np.float32, copy=False)

            for t in range(1, T):
                interped[..., t] = interpolate_to_voxel_dims_precomputed(
                    data[..., t], zoom_factors, order
                ).astype(np.float32, copy=False)

            final_data = apply_padding_anydim(interped, shape_padding)

        # Update affine (robust for non-diagonal / permuted affines)
        # NOTE: Do NOT rely on np.diag(...) here; many valid affines have zeros on the diagonal
        # (e.g., axis permutations), and using diag-signs can create a singular affine that nibabel
        # cannot decompose into qform/sform.
        # Start from the affine that already includes the center-padding origin update.
        new_affine = affine_after_center.copy()
        R = new_affine[:3, :3].astype(np.float64, copy=True)
        eps = 1e-8

        # Direction cosines from affine columns (handles rotations / axis permutations)
        D = np.zeros((3, 3), dtype=np.float64)
        for ax in range(3):
            col = R[:, ax]
            n = float(np.linalg.norm(col))
            if (not np.isfinite(n)) or (n < eps):
                # Fallback to canonical axis if the column is degenerate
                D[:, ax] = 0.0
                D[ax, ax] = 1.0
            else:
                D[:, ax] = col / n

        # IMPORTANT:
        # Do NOT orthonormalize (QR/SVD) and do NOT force det(Q)>0 here.
        # Many valid NIfTI affines are left-handed in sform (det<0), especially with
        # a negative X column (common RAS/LPS conventions). Forcing a "proper rotation"
        # can introduce axis flips/rotations that make axial no longer axial.
        #
        # We only need to preserve the *direction* of each axis while changing the
        # voxel sizes. Using normalized columns preserves orientation and obliqueness.
        Q = D

        tvd = np.asarray(target_voxel_dims, dtype=np.float64)
        if tvd.shape != (3,):
            raise ValueError(f"target_voxel_dims must be length-3, got {tvd!r}")
        if (not np.all(np.isfinite(tvd))) or np.any(tvd <= 0):
            raise ValueError(f"target_voxel_dims must be finite positive values, got {tvd!r}")

        new_affine[:3, :3] = Q @ np.diag(tvd)
        new_affine = update_origin_for_padding(
            new_affine, np.array(padding_record["shape_padding"], dtype=int), tvd
        )

    # ---------------------------------------------------------------------
    # Save output
    # ---------------------------------------------------------------------
    out_hdr = img.header.copy()
    out_img = nib.Nifti1Image(final_data.astype(np.float32), new_affine, header=out_hdr)

    # Always set sform to the new affine (this is what most tools should trust).
    try:
        scode = int(out_hdr.get("sform_code", 0))
    except Exception:
        scode = 0
    try:
        out_img.set_sform(new_affine, code=(scode if scode else 1))
    except Exception:
        # If set_sform fails for any reason, we still at least have the base affine.
        pass

    # Only set qform if the affine is representable as a proper quaternion rotation:
    # - top-left 3x3 must be (approximately) orthonormal
    # - determinant must be +1 (proper rotation)
    #
    # Otherwise, set qform_code=0 to avoid viewers/tools applying an inconsistent qform.
    try:
        import numpy as np
        R = np.asarray(new_affine[:3, :3], dtype=float)
        # normalize columns to remove scaling (should already be scaled by tvd)
        # Check orthogonality up to tolerance
        RtR = (R / np.linalg.norm(R, axis=0, keepdims=True)).T @ (R / np.linalg.norm(R, axis=0, keepdims=True))
        detR = float(np.linalg.det(R / np.linalg.norm(R, axis=0, keepdims=True)))
        is_ortho = np.allclose(RtR, np.eye(3), atol=1e-3, rtol=1e-3)
        is_proper = abs(detR - 1.0) < 1e-3
    except Exception:
        is_ortho = False
        is_proper = False

    if is_ortho and is_proper:
        try:
            qcode = int(out_hdr.get("qform_code", 0))
        except Exception:
            qcode = 0
        try:
            out_img.set_qform(new_affine, code=(qcode if qcode else 1))
        except Exception:
            try:
                out_img.header["qform_code"] = 0
            except Exception:
                pass
    else:
        try:
            out_img.header["qform_code"] = 0
        except Exception:
            pass

    # Preserve 4th zoom (time spacing) if present
    if ndim == 4:
        out_hdr = out_img.header
        tzoom = header_zooms[3] if len(header_zooms) > 3 else 1.0
        out_hdr.set_zooms(tuple(list(target_voxel_dims) + [tzoom]))

    nib.save(out_img, output_filepath)

    # ---------------------------------------------------------------------
    # Save padding record if requested
    # ---------------------------------------------------------------------
    if save_padding_record:
        path_to_save = padding_record_path or f"{output_filepath}_padding.txt"
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        with open(path_to_save, "w") as f:
            f.write(str(padding_record))


# -----------------------------------------------------------------------------------
# Function to undo reshape of MRI volume using saved padding record from resize_mri()
# -----------------------------------------------------------------------------------

def reverse_resize_mri(input_filepath, output_filepath, padding_record_path, interp=1):
    """
    Reverse a resizing operation performed by resize_mri(), using the original
    spacing and padding information stored in a padding record file.

    Supports 3D and 4D inputs. For 4D, resampling is done framewise to avoid
    any interpolation across the time axis.
    """
    import os
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom
    from .preprocessing_utils import read_padding_record

    if not os.path.exists(padding_record_path):
        raise FileNotFoundError(f"Padding record not found: {padding_record_path}")

    padding_record = read_padding_record(padding_record_path)

    img = nib.load(input_filepath)
    data = img.get_fdata()
    ndim = data.ndim
    if ndim not in (3, 4):
        raise ValueError(f"Unsupported ndim={ndim} for reverse_resize_mri: {input_filepath}")

    # ---- Helper: undo padding on first 3 axes only (works for 3D or 4D) ----
    def _undo_padding_anydim(arr, pad_3x2):
        """
        Inverse of apply_padding_anydim logic:
        - If pad value was positive (we padded), undo by cropping.
        - If pad value was negative (we cropped), undo by padding zeros.
        Operates on first 3 axes only; leaves extra axes untouched.
        """
        pad_3x2 = np.asarray(pad_3x2, dtype=int)
        if pad_3x2.shape != (3, 2):
            raise ValueError(f"pad must be shape (3,2), got {pad_3x2.shape}")

        out = arr

        # First: undo positive padding by cropping
        slices = [slice(None)] * out.ndim
        for ax in range(3):
            before, after = pad_3x2[ax]
            if before > 0 or after > 0:
                start = before
                end = out.shape[ax] - after if after > 0 else out.shape[ax]
                if end < start:
                    raise ValueError(
                        f"Invalid undo crop on axis {ax}: start={start}, end={end}, shape={out.shape}"
                    )
                slices[ax] = slice(start, end)
        out = out[tuple(slices)]

        # Second: undo negative padding (i.e., original crop) by adding zeros
        pad_width = [(0, 0)] * out.ndim
        for ax in range(3):
            before, after = pad_3x2[ax]
            if before < 0 or after < 0:
                pad_before = -before if before < 0 else 0
                pad_after = -after if after < 0 else 0
                pad_width[ax] = (pad_before, pad_after)

        if any(p[0] > 0 or p[1] > 0 for p in pad_width[:3]):
            out = np.pad(out, pad_width, mode="constant", constant_values=0)

        return out

    # ---- Step 1: Resize back to original voxel spacing (spatial only) ----
    current_zooms = img.header.get_zooms()
    current_voxel_dims = np.array(current_zooms[:3], dtype=float)
    original_voxel_dims = np.array(padding_record["original_voxel_dims"][:3], dtype=float)

    zoom_factors = current_voxel_dims / original_voxel_dims  # spatial zoom only

    if ndim == 3:
        resampled = zoom(data, zoom_factors, order=interp)
    else:
        # Framewise zoom to avoid interpolation across time axis
        T = data.shape[3]
        first = zoom(data[..., 0], zoom_factors, order=interp)
        resampled = np.empty((*first.shape, T), dtype=first.dtype)
        resampled[..., 0] = first
        for t in range(1, T):
            resampled[..., t] = zoom(data[..., t], zoom_factors, order=interp)

    # ---- Step 2: Undo shape padding ----
    shape_padding = np.asarray(padding_record["shape_padding"], dtype=int)
    adjusted = _undo_padding_anydim(resampled, shape_padding)

    # ---- Step 3: Undo center padding ----
    center_padding = np.asarray(padding_record["center_padding"], dtype=int)
    final = _undo_padding_anydim(adjusted, center_padding)

    # ---- Step 4: Restore original affine ----
    original_affine = np.eye(4)
    original_affine[:3, 3] = padding_record["original_grid"]["origin"]

    direction_matrix = np.reshape(padding_record["original_grid"]["direction"], (3, 3))
    voxel_dims = np.array(padding_record["original_voxel_dims"][:3], dtype=float)

    scaled_direction = direction_matrix * voxel_dims[np.newaxis, :]
    original_affine[:3, :3] = scaled_direction

    out_img = nib.Nifti1Image(final.astype(np.float32), original_affine)

    # Preserve time zoom if 4D
    if ndim == 4:
        tzoom = current_zooms[3] if len(current_zooms) > 3 else 1.0
        out_img.header.set_zooms(tuple(list(original_voxel_dims) + [tzoom]))

    nib.save(out_img, output_filepath)
    print(f"[Done] Reversed resize saved to: {output_filepath}")

# -------------------------------------------------------------------------
# Function to match affine matrices between two nifti files
# -------------------------------------------------------------------------

def match_direction_matrices(input_path, donor_path, output_path, *, debug: bool = False):
    """Resample `input_path` onto `donor_path` grid (shape/origin/spacing/direction).

    Primary implementation uses nibabel+nilearn (keeps headers consistent with the rest of the pipeline),
    but some vendor/converted NIfTI files can have headers/affines that nibabel cannot safely decompose
    (e.g., NaNs in scl_slope/scl_inter or rank-deficient qform). In those cases, we fall back to a
    SimpleITK resample, which is typically more tolerant, then (best-effort) sanitize the output header.

    Parameters
    ----------
    input_path : str
        Path to a NIfTI image to be resampled.
    donor_path : str
        Path to a NIfTI image providing the target grid.
    output_path : str
        Path to write the resampled output.
    debug : bool
        Print verbose geometry/header diagnostics.

    Notes
    -----
    * Setting qform can fail for rank-deficient affines (cannot be decomposed into a quaternion).
      In that case we still set sform and clear qform.
    """
    import numpy as np

    def _sanitize_nifti_header_inplace_local(path: str) -> None:
        """Best-effort header cleanup to avoid nibabel warnings / decompositions later."""
        try:
            import nibabel as nib

            img = nib.load(path)
            aff = np.array(img.affine, dtype=float, copy=True)
            hdr = img.header.copy()

            # Fix invalid/zero/NaN zooms
            try:
                zooms = tuple(float(z) for z in hdr.get_zooms()[:3])
            except Exception:
                zooms = (None, None, None)

            def _valid_zooms(zs):
                return (zs is not None) and all((z is not None and np.isfinite(z) and float(z) > 0) for z in zs)

            if not _valid_zooms(zooms):
                col_norms = tuple(float(np.linalg.norm(aff[:3, i])) for i in range(3))
                fixed_zooms = tuple((n if (np.isfinite(n) and n > 0) else 1.0) for n in col_norms)
                rest = list(hdr.get_zooms()[3:]) if len(hdr.get_zooms()) > 3 else []
                try:
                    hdr.set_zooms(tuple(list(fixed_zooms) + rest))
                except Exception:
                    pass
                if debug:
                    print(f"[match_direction_matrices][debug] Sanitized zooms for {path}: {zooms} -> {fixed_zooms}")

            # Clear any scaling fields that can become NaN
            try:
                slope = float(hdr.get("scl_slope", 1.0))
                inter = float(hdr.get("scl_inter", 0.0))
                if (not np.isfinite(slope)) or slope == 0.0:
                    hdr["scl_slope"] = 1.0
                if not np.isfinite(inter):
                    hdr["scl_inter"] = 0.0
            except Exception:
                pass

            # Ensure qform/sform are set to the affine so downstream tools don't fall back to base affine
            try:
                hdr.set_qform(aff, code=1)
                hdr.set_sform(aff, code=1)
            except Exception:
                # If qform fails due to quaternion decomposition, keep sform only.
                try:
                    hdr["qform_code"] = 0
                    hdr.set_sform(aff, code=1)
                except Exception:
                    pass

            data = img.get_fdata(dtype=np.float32)
            out = nib.Nifti1Image(data, aff, header=hdr)
            nib.save(out, path)
        except Exception:
            return

    # -------- Attempt nibabel + nilearn path first --------
    try:
        import nibabel as nib
        from nilearn.image import resample_to_img

        donor_img = nib.load(donor_path)
        input_img = nib.load(input_path)

        if debug:
            try:
                in_aff = np.asarray(input_img.affine)
                dn_aff = np.asarray(donor_img.affine)
                print(f"[match_direction_matrices][debug] input_path={input_path}")
                print(f"[match_direction_matrices][debug] donor_path={donor_path}")
                print(f"[match_direction_matrices][debug] input shape={input_img.shape} dtype={input_img.get_data_dtype()}")
                print(f"[match_direction_matrices][debug] donor shape={donor_img.shape} dtype={donor_img.get_data_dtype()}")
                print(f"[match_direction_matrices][debug] input affine=\n{in_aff}")
                print(f"[match_direction_matrices][debug] donor affine=\n{dn_aff}")
                try:
                    print(f"[match_direction_matrices][debug] rank(input_aff[:3,:3])={np.linalg.matrix_rank(in_aff[:3,:3])}")
                except Exception:
                    pass
                try:
                    print(f"[match_direction_matrices][debug] rank(donor_aff[:3,:3])={np.linalg.matrix_rank(dn_aff[:3,:3])}")
                except Exception:
                    pass

                ih = input_img.header
                dh = donor_img.header
                print(f"[match_direction_matrices][debug] input zooms={ih.get_zooms()} units={ih.get_xyzt_units()}")
                print(f"[match_direction_matrices][debug] donor zooms={dh.get_zooms()} units={dh.get_xyzt_units()}")
                print(f"[match_direction_matrices][debug] input qform={ih.get_qform(coded=True)}")
                print(f"[match_direction_matrices][debug] input sform={ih.get_sform(coded=True)}")
                print(f"[match_direction_matrices][debug] input base_affine=\n{ih.get_base_affine()}")
                print(f"[match_direction_matrices][debug] donor qform={dh.get_qform(coded=True)}")
                print(f"[match_direction_matrices][debug] donor sform={dh.get_sform(coded=True)}")
                print(f"[match_direction_matrices][debug] donor base_affine=\n{dh.get_base_affine()}")
            except Exception as _e:
                print(f"[match_direction_matrices][debug] Failed to print debug header/affine info: {_e}")

        resampled_img = resample_to_img(
            input_img,
            donor_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

        resampled_data = resampled_img.get_fdata().astype(input_img.get_data_dtype())
        header = input_img.header.copy()

        # qform may fail for rank-deficient affines (cannot be decomposed into quaternion).
        # sform does not require quaternion decomposition, so we still set sform as best-effort.
        try:
            header.set_qform(donor_img.affine, code=1)
        except Exception as e:
            try:
                header["qform_code"] = 0
            except Exception:
                pass
            if debug:
                print(f"[match_direction_matrices][debug] set_qform FAILED: {e}")

        try:
            header.set_sform(donor_img.affine, code=1)
        except Exception as e:
            if debug:
                print(f"[match_direction_matrices][debug] set_sform FAILED: {e}")

        output_img = nib.Nifti1Image(resampled_data, affine=donor_img.affine, header=header)

        if debug:
            try:
                oh = output_img.header
                print(f"[match_direction_matrices][debug] output_path={output_path}")
                print(f"[match_direction_matrices][debug] output qform={oh.get_qform(coded=True)}")
                print(f"[match_direction_matrices][debug] output sform={oh.get_sform(coded=True)}")
                print(f"[match_direction_matrices][debug] output base_affine=\n{oh.get_base_affine()}")
            except Exception as _e:
                print(f"[match_direction_matrices][debug] Failed to print output header info: {_e}")

        nib.save(output_img, output_path)
        #_sanitize_nifti_header_inplace_local(output_path) #skip for now
        return

    except Exception as e:
        if debug:
            print(f"[match_direction_matrices][debug] nibabel/nilearn path FAILED; falling back to SimpleITK. error={e!r}")

    # -------- Fallback: SimpleITK resample (tolerates many header issues) --------
    try:
        import SimpleITK as sitk

        donor = sitk.ReadImage(str(donor_path))
        inp = sitk.ReadImage(str(input_path))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(donor)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)

        # Identity transform (resampler maps from input physical space into donor grid)
        tx = sitk.Transform(3, sitk.sitkIdentity)
        resampler.SetTransform(tx)

        out = resampler.Execute(inp)
        sitk.WriteImage(out, str(output_path))

        #_sanitize_nifti_header_inplace_local(output_path) #skip for now

        if debug:
            try:
                print(f"[match_direction_matrices][debug] SimpleITK fallback wrote: {output_path}")
                print(f"[match_direction_matrices][debug] donor direction={donor.GetDirection()}")
                print(f"[match_direction_matrices][debug] input direction={inp.GetDirection()}")
                print(f"[match_direction_matrices][debug] out direction={out.GetDirection()}")
            except Exception:
                pass
        return
    except Exception as e2:
        if debug:
            print(f"[match_direction_matrices][debug] SimpleITK fallback FAILED: {e2!r}")
        raise

# -------------------------------------------------------------------------
# Function to merge mask files into a single mask
# -------------------------------------------------------------------------

def merge_binary_masks(mask_paths, output_path, fill_holes=False, strict_affine=False):
    """
    Merge multiple binary masks (NIfTI format) into one.
    Voxels are 1 if any input mask has a 1 at that position.

    Args:
        mask_paths (list of str): Paths to input NIfTI mask files
        output_path (str): Path to save the merged mask
        fill_holes (bool): Whether to apply hole filling
        strict_affine (bool): If True, check that affines match exactly
    """
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import binary_fill_holes

    if len(mask_paths) < 2:
        raise ValueError("At least two mask files are required.")

    # Load first mask
    first_img = nib.load(mask_paths[0])
    merged_data = (first_img.get_fdata() > 0).astype(np.uint8)
    reference_affine = first_img.affine
    reference_shape = merged_data.shape

    # Iterate and combine
    for path in mask_paths[1:]:
        img = nib.load(path)
        data = (img.get_fdata() > 0).astype(np.uint8)

        if data.shape != reference_shape:
            raise ValueError(f"Shape mismatch: {path} has shape {data.shape}, expected {reference_shape}")
        if strict_affine and not np.allclose(img.affine, reference_affine):
            raise ValueError(f"Affine mismatch: {path} does not match reference affine")

        merged_data = np.logical_or(merged_data, data)

    if fill_holes:
        merged_data = binary_fill_holes(merged_data).astype(np.uint8)

    merged_img = nib.Nifti1Image(merged_data, affine=reference_affine)
    nib.save(merged_img, output_path)

# -------------------------------------------------------------------------
# Function to register scans to each other
# -------------------------------------------------------------------------

def register_images(
    fixed_path,
    moving_path,
    output_path,
    transform_path=None,
    apply_only=False,
    registration_type="rigid",
    similarity_metric="mi",
    registration_strategy="medium",
    metric_sampling_seed=None,
    metric_focus: str = "none",
    metric_focus_percentile: float = 95.0,
    metric_focus_sigma_mm: float = 1.0,
    metric_focus_dilate_vox: int = 1,
    verbose=False,
    save_dummy_ref=False,
    interpolation="auto",
    integer: bool = False,
    registration_voxel_mm="1,1,1",
    n_workers: int = None,
    *,
    fixed_frame_index: int = 0,
    moving_frame_index: int = 0,
    use_first_frame_only=False,
    keep_moving_grid: bool = False,
    debug = False,
):
    """
    Register (estimate a transform) or apply an existing transform to align a moving MRI volume
    into the space of a fixed MRI volume, using SimpleITK.

    **3D and 4D support**
    - If both inputs are 3D, behavior is unchanged.
    - If either input is 4D, the transform is **estimated** (when apply_only=False) from a single 3D frame
      (selected by fixed_frame_index / moving_frame_index), then applied in-memory to all frames and written
      as a 4D NIfTI at output_path.

    Notes
    -----
    - SimpleITK's registration methods operate on 3D volumes; for 4D we therefore register a representative
      frame, then resample each frame with the resulting transform.
    - In apply-only mode, the transform is applied to the full moving image (3D or 4D).
    - Dummy reference images (save_dummy_ref=True) reflect the *3D images used to estimate* the transform.
      For 4D inputs, these are the selected frames.

    Parameters
    ----------
    fixed_path : str or os.PathLike
        Path to the fixed/reference image (3D or 4D). The output is resampled onto the fixed grid.
    moving_path : str or os.PathLike
        Path to the moving image (3D or 4D).
    output_path : str or os.PathLike
        Path where the registered/resampled moving image will be written.
    transform_path : str or os.PathLike, optional
        Transform file path.
        - If apply_only=False and provided, the estimated transform is written here.
        - If apply_only=True, this must exist and is loaded and applied.
    apply_only : bool, default=False
        If True, skip registration and only apply an existing transform.
    registration_type : {"rigid", "affine", "translation"}, default="rigid"
        Transform family to estimate (registration mode only).
    similarity_metric : {"mi", "correlation"}, default="mi"
        Similarity metric to optimize during registration (registration mode only).
    registration_strategy : {"accurate", "medium", "fast"}, default="accurate"
        Preset controlling pyramid levels, sampling fraction, and iteration budget.
    metric_sampling_seed : int or None, default=None
        Seed for stochastic metric sampling used in "medium"/"fast" presets.
    metric_focus : {"none", "background_subtracted", "foreground", "edges", "lowhigh", "highlow"}, default="none"
        Controls whether the registration metric is restricted to a subset of voxels (via SimpleITK metric masks).
        This is mainly intended to improve robustness when registering partial-FOV/low-coverage scans (e.g. perfusion)
        to full-head structural images.

        - "none": do not set metric mask
        - "background_subtracted": do not set a metric mask; instead, compute a foreground mask for each image
          and set voxels outside the foreground to 0.0 *before* registration. The metric still uses all sampled voxels,
          but the background contributes less noise/spurious structure. This can help when moving scans have a lot of
          background/air (partial head, perfusion) and the optimizer otherwise chases background patterns.s (default SimpleITK behavior). The metric uses all sampled voxels, which may
          include background/air in partial-FOV scans and can lead to unstable optimization.
        - "foreground": restrict the metric to a simple foreground mask (computed from an Otsu threshold on a robustly
          normalized image, cleaned and reduced to the largest connected component). This removes background/air but does
          not otherwise emphasize edges or intensities.
        - "edges": restrict the metric to high-contrast voxels (large intensity gradients) within the foreground.
          This emphasizes anatomical boundaries such as ventricles and tissue interfaces.
        - "lowhigh": use different intensity tails for fixed vs moving.
          The fixed mask selects **low-intensity** voxels, while the moving mask selects **high-intensity** voxels.
          This can help when a structure of interest appears dark in the fixed image but bright in the moving image.
        - "highlow": the inverse of "lowhigh": fixed uses **high-intensity** voxels and moving uses **low-intensity** voxels.

    metric_focus_percentile : float, default=95.0
        Percentile (0–100) controlling how selective the focus masks are.

        - For metric_focus="edges": this is the percentile of the **gradient-magnitude** distribution (within the
          foreground). Voxels with gradient magnitude >= this percentile are included.
        - For metric_focus in {"lowhigh", "highlow"}: this controls the **intensity tails** (within foreground) taken from
          each image:
            * "high" tail uses >= Pth percentile (e.g., P=95 keeps the brightest 5%).
            * "low" tail uses <= (100-P)th percentile (e.g., P=95 keeps the darkest 5%).
          Higher values focus on more extreme tails; lower values include more voxels.

    metric_focus_sigma_mm : float, default=1.0
        Physical-space Gaussian smoothing (in millimeters) applied **before** computing gradient magnitude for
        metric_focus="edges". Increasing this value suppresses fine-scale noise and thin skull edges while emphasizing
        broader anatomical boundaries (e.g., ventricular walls). Typical values are in the range 0.5–2.0 mm.

    metric_focus_dilate_vox : int, default=1
        Number of voxels by which the focus mask is dilated after thresholding/selection. Dilation increases mask support
        and can improve optimizer stability when the selected set is sparse. Set to 0 to disable dilation.

    verbose : bool, default=False
        Print additional details.
    save_dummy_ref : bool, default=False
        If True (and transform_path is provided), write de-identified 0-filled dummy references alongside
        the transform, preserving geometry for later apply/reverse steps.
    interpolation : int | str, default="auto"
        Interpolation used for (1) the registration optimizer interpolator and (2) final resampling into fixed space.
        Accepts int 0–5 or strings: nearest, linear/bilinear, quadratic, cubic, quartic, quintic, bspline.
        "auto": if apply_only=True, try to reuse the interpolation saved next to transform_path (.meta.json); if unavailable, defaults to linear.
    registration_voxel_mm : (float, float, float) | str | None, default="1,1,1"
        Optional spacing (mm, mm, mm) to use *during transform estimation* (apply_only=False).
        This speeds up registration by downsampling both fixed and moving frames to a common voxel size
        before optimization.
        - Examples: (1, 1, 1) or "1,1,1"
        - If you request a spacing finer than both input scans along any axis, that axis is clamped to the
          finest available input spacing and a warning is printed (to avoid unintended upsampling).
        The final resampling to output_path is still done onto the fixed grid at full resolution.
    
    n_workers : int, default = None
        Limits how many CPU threads are used for registration and resampling. Default behavior is to use all available threads.
    fixed_frame_index : int, default=0
        If fixed_path is 4D and apply_only=False, the 3D frame index used to estimate the transform.
    moving_frame_index : int, default=0
        If moving_path is 4D and apply_only=False, the 3D frame index used to estimate the transform.
    use_first_frame_only : bool, default=False
        If True, moving 4d volumes are returned as a 3d volume registered via the first frame of the input volume.
    keep_moving_grid : bool, default=False
        If True, the moving image is resampled onto its **own** original 3D grid (same size/spacing/origin/direction),
        rather than onto the fixed image grid. This can be useful when you want to preserve the moving image's
        native voxel lattice while still applying the estimated transform.

        Notes:
        - The estimated transform is still written (when transform_path is provided) and maps moving->fixed.
        - When keep_moving_grid=True, the output image will not share the fixed grid; downstream consumers should
          use the saved transform (or its inverse) to relate the output back to fixed space.

    debug : bool, default=False
        Print additional information for 4d image registration to debug affine matrice issues.

    Returns
    -------
    None
    """
    import os
    import json
    import SimpleITK as sitk
    import numpy as np
    from datetime import datetime
    from pathlib import Path

    # NOTE:
    # For NRRD outputs (especially 3D Slicer segmentations / labelmaps), important
    # per-segment fields (e.g., Segment0_Name, Segment1_Name, ...) live in NRRD metadata.
    # SimpleITK resampling/writing can drop these unless we explicitly copy them over.
    #
    # We copy metadata keys using a strict allowlist to avoid copying any Slicer
    # segmentation container fields (e.g. Segmentation_ConversionParameters) that can
    # cause display shifts if they become inconsistent after resampling.
    def _copy_nrrd_metadata_nongeom(src_img, dst_img):
        try:
            src_keys = list(src_img.GetMetaDataKeys())
        except Exception:
            return

        # Allowlist only: copy per-segment identity fields (safe) and skip everything else.
        # This intentionally does NOT copy:
        #   - any Segmentation_* keys (may embed stale reference geometry)
        #   - Segment*_Extent (derived/cached bounds; invalid after resampling)
        #   - any geometry/space header fields
        allowed_segment_fields = {
            "name",
            "nameautogenerated",
            "color",
            "colorautogenerated",
            "labelvalue",
            "id",
            "layer",
            "tags",
        }

        for k in src_keys:
            try:
                key = str(k).strip()
            except Exception:
                continue
            if not key:
                continue

            lk = key.lower()
            if not lk.startswith("segment"):
                continue

            # Expected form: Segment{n}_{Field}
            # e.g., Segment0_Name, Segment12_LabelValue
            m = re.match(r"^segment(\d+)_([a-z0-9]+)$", lk)
            if not m:
                continue

            field = m.group(2)
            if field not in allowed_segment_fields:
                continue

            try:
                dst_img.SetMetaData(key, src_img.GetMetaData(key))
            except Exception:
                # Best-effort: skip keys that error (e.g., malformed/unsupported)
                pass

    # ------------------------------------------------------------------
    # Debug output directory (for masks, intermediate images, etc.)
    # ------------------------------------------------------------------
    _debug_mask_dir: Path | None = None

    def _ensure_debug_mask_dir() -> Path:
        """Create (once) a per-call directory to dump debug registration masks."""
        nonlocal _debug_mask_dir
        if _debug_mask_dir is not None:
            return _debug_mask_dir

        # Keep debug artifacts next to the final output by default.
        out_dir = Path(str(output_path)).expanduser().resolve().parent
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        def _stem(p):
            try:
                return Path(str(p)).stem
            except Exception:
                return "img"

        tag = f"register_debug_{_stem(fixed_path)}_to_{_stem(moving_path)}_{stamp}"
        # Avoid pathological characters in filenames on Windows.
        tag = "".join((c if (c.isalnum() or c in "._-") else "_") for c in tag)
        _debug_mask_dir = out_dir / tag
        _debug_mask_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[register_images][debug] Saving metric masks to: {_debug_mask_dir}")
        return _debug_mask_dir

    def _debug_write_mask(mask: sitk.Image, *, stage: str, role: str, kind: str) -> None:
        """Best-effort: write a (binary) mask for inspection."""
        if not debug:
            return
        try:
            d = _ensure_debug_mask_dir()
            path = d / f"{stage}_{role}_{kind}.nii.gz"
            sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), str(path))
        except Exception:
            return

    # ------------------------------------------------------------------
    # Interpolation handling
    # ------------------------------------------------------------------
    # We accept a scipy.ndimage.zoom-like interpolation spec (int 0-5 or strings like
    # 'linear'/'cubic'). For SimpleITK resampling, orders >=2 are mapped to cubic B-spline.
    from .preprocessing_utils import _interp_to_sitk_interpolator

    def _transform_meta_path(tfm_path: str) -> str:
        # Store sidecar JSON next to the .tfm so apply/reverse steps can stay consistent.
        return os.path.splitext(str(tfm_path))[0] + ".meta.json"

    def _read_transform_meta(tfm_path: str):
        meta_path = _transform_meta_path(tfm_path)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_transform_meta(tfm_path: str, meta: dict):
        meta_path = _transform_meta_path(tfm_path)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
        except Exception:
            # Best-effort only; the transform itself is the critical artifact.
            return

    # Resolve interpolation: allow 'auto' to reuse prior settings when applying an existing transform.
    interp_spec = interpolation
    if isinstance(interp_spec, str) and interp_spec.strip().lower() == "auto":
        interp_spec = None

    if apply_only and interp_spec is None and transform_path:
        meta = _read_transform_meta(transform_path)
        if isinstance(meta, dict) and "interp" in meta:
            interp_spec = meta["interp"]
            if verbose:
                print(f"[register_images] Using recorded interpolation from sidecar: {interp_spec!r}")
    
    # If saving integer labelmaps (e.g., segmentations), enforce nearest-neighbor
    if integer and not interp_spec == "nearest":
        print(f"[WARNING] You are registering images with {interp_spec!r} interpolation but saving integer volumes. This is probably wrong. For integer volumes, you should use 'nearest' interpolation.")

    sitk_interp, sitk_interp_name, scipy_order = _interp_to_sitk_interpolator(interp_spec)

    def _transform_translation_vector(tfm) -> tuple[float, float, float]:
        """Best-effort extraction of a translation-like vector from an ITK transform.

        SimpleITK can return CompositeTransform even when you expect a TranslationTransform.
        """
        import numpy as np

        if tfm is None:
            return (0.0, 0.0, 0.0)
        if hasattr(tfm, "GetOffset"):
            o = tfm.GetOffset()
            return (float(o[0]), float(o[1]), float(o[2]))
        if hasattr(tfm, "GetTranslation"):
            o = tfm.GetTranslation()
            return (float(o[0]), float(o[1]), float(o[2]))
        # CompositeTransform: sum any translation terms (best effort)
        if hasattr(tfm, "GetNumberOfTransforms") and hasattr(tfm, "GetNthTransform"):
            v = np.zeros(3, dtype=float)
            try:
                for i in range(int(tfm.GetNumberOfTransforms())):
                    sub = tfm.GetNthTransform(i)
                    if hasattr(sub, "GetOffset"):
                        o = sub.GetOffset()
                        v += np.array([float(o[0]), float(o[1]), float(o[2])], dtype=float)
                    elif hasattr(sub, "GetTranslation"):
                        o = sub.GetTranslation()
                        v += np.array([float(o[0]), float(o[1]), float(o[2])], dtype=float)
            except Exception:
                pass
            return (float(v[0]), float(v[1]), float(v[2]))
        return (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Helper functions for foreground masking and metric focus
    # ------------------------------------------------------------------
    def _robust_normalize_to_0_1(img: sitk.Image) -> sitk.Image:
        """Robustly scale intensities to ~[0,1] using percentiles (registration-frame only)."""
        arr = sitk.GetArrayViewFromImage(img).astype(np.float32, copy=False)
        # Ignore exact zeros for percentile estimation (background-heavy volumes)
        nz = arr[arr != 0]
        if nz.size < 1000:
            nz = arr.reshape(-1)
        lo = float(np.percentile(nz, 1.0))
        hi = float(np.percentile(nz, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return sitk.Normalize(img)  # fallback
        x = sitk.Clamp(img, lowerBound=lo, upperBound=hi)
        x = (x - lo) / (hi - lo)
        return sitk.Cast(x, sitk.sitkFloat32)

    def _make_foreground_mask(norm01: sitk.Image) -> sitk.Image:
        """Fast-ish foreground mask to remove background and air."""
        # Otsu on normalized image works surprisingly well across modalities
        m = sitk.OtsuThreshold(norm01, 0, 1, 128)  # uint8 0/1
        m = sitk.BinaryFillhole(m)
        m = sitk.BinaryMorphologicalClosing(m, [1, 1, 1])
        # Keep largest component to avoid stray junk in partial FOV scans
        cc = sitk.ConnectedComponent(m)
        rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
        m = sitk.BinaryThreshold(rel, 1, 1, 1, 0)
        return sitk.Cast(m, sitk.sitkUInt8)

    def _make_focus_mask_edges(
        img: sitk.Image,
        *,
        percentile: float,
        sigma_mm: float,
        dilate_vox: int,
        stage: str,
        role: str,
    ) -> sitk.Image:
        """
        Make a binary mask selecting only strong edges (high gradient magnitude),
        restricted to foreground.
        """
        norm01 = _robust_normalize_to_0_1(img)
        fg = _make_foreground_mask(norm01)
        _debug_write_mask(fg, stage=stage, role=role, kind="foreground")

        # Gradient magnitude emphasizes boundaries like ventricles/skull/GM-WM
        gm = sitk.GradientMagnitudeRecursiveGaussian(norm01, sigma=float(sigma_mm))
        gm = sitk.Mask(gm, fg)  # kill background edges

        garr = sitk.GetArrayViewFromImage(gm).astype(np.float32, copy=False)
        vals = garr[garr > 0]
        if vals.size < 1000:
            # Too little signal -> fall back to just foreground
            _debug_write_mask(fg, stage=stage, role=role, kind="edges_fallback_foreground")
            return fg

        thr = float(np.percentile(vals, float(percentile)))
        mask = sitk.BinaryThreshold(gm, lowerThreshold=thr, upperThreshold=1e9, insideValue=1, outsideValue=0)

        if dilate_vox and int(dilate_vox) > 0:
            dx, dy, dz = img.GetSpacing()
            rad = [int(dilate_vox), int(dilate_vox), int(dilate_vox)]
            if dz > 2.5 * max(dx, dy):   # heuristic
                rad[2] = 0
            mask = sitk.BinaryDilate(mask, rad)

        _debug_write_mask(mask, stage=stage, role=role, kind="edges")

        # Ensure uint8 mask
        return sitk.Cast(mask, sitk.sitkUInt8)

    def _maybe_dilate_mask(mask: sitk.Image, *, img: sitk.Image, dilate_vox: int) -> sitk.Image:
        """Dilate mask with a small heuristic to avoid dilating in very-thick-slice Z."""
        if not dilate_vox or int(dilate_vox) <= 0:
            return sitk.Cast(mask, sitk.sitkUInt8)
        try:
            dx, dy, dz = img.GetSpacing()
            rad = [int(dilate_vox), int(dilate_vox), int(dilate_vox)]
            # If Z spacing is much larger than in-plane, avoid Z dilation (thin stacks / perfusion).
            if float(dz) > 2.5 * max(float(dx), float(dy)):
                rad[2] = 0
            return sitk.Cast(sitk.BinaryDilate(mask, rad), sitk.sitkUInt8)
        except Exception:
            return sitk.Cast(mask, sitk.sitkUInt8)

    def _make_focus_mask_foreground(
        img: sitk.Image,
        *,
        dilate_vox: int,
        stage: str,
        role: str,
    ) -> sitk.Image:
        """Foreground-only metric mask (removes background/air)."""
        norm01 = _robust_normalize_to_0_1(img)
        fg = _make_foreground_mask(norm01)
        _debug_write_mask(fg, stage=stage, role=role, kind="foreground")
        fg = _maybe_dilate_mask(fg, img=img, dilate_vox=dilate_vox)
        _debug_write_mask(fg, stage=stage, role=role, kind="foreground_dilated" if dilate_vox else "foreground")
        return sitk.Cast(fg, sitk.sitkUInt8)

    def _make_focus_mask_intensity_tail(
        img: sitk.Image,
        *,
        tail: str,
        percentile: float,
        dilate_vox: int,
        stage: str,
        role: str,
        keep_largest_cc: bool = False,
    ) -> sitk.Image:
        """Intensity-tail mask within foreground: select low or high intensities in robust [0,1] space."""
        tail = str(tail).strip().lower()
        if tail not in ("low", "high"):
            raise ValueError(f"tail must be 'low' or 'high' (got {tail!r})")

        # Robust normalize + foreground restrict
        norm01 = _robust_normalize_to_0_1(img)
        fg = _make_foreground_mask(norm01)
        _debug_write_mask(fg, stage=stage, role=role, kind="foreground")

        arr = sitk.GetArrayViewFromImage(norm01).astype(np.float32, copy=False)
        fgm = sitk.GetArrayViewFromImage(fg) > 0
        vals = arr[fgm]
        if vals.size < 1000:
            _debug_write_mask(fg, stage=stage, role=role, kind=f"intensity_{tail}_fallback_foreground")
            return sitk.Cast(fg, sitk.sitkUInt8)

        P = float(percentile)
        P = 50.0 if (not np.isfinite(P)) else max(0.0, min(100.0, P))

        if tail == "high":
            thr = float(np.percentile(vals, P))
            mask = sitk.BinaryThreshold(norm01, lowerThreshold=thr, upperThreshold=1e9, insideValue=1, outsideValue=0)
        else:
            # Keep darkest (100-P)%  (e.g., P=95 -> <= 5th percentile)
            thr = float(np.percentile(vals, 100.0 - P))
            mask = sitk.BinaryThreshold(norm01, lowerThreshold=-1e9, upperThreshold=thr, insideValue=1, outsideValue=0)

        mask = sitk.Mask(mask, fg)  # ensure we stay in foreground

        # For intensity-tail modes, we optionally keep only contiguous components that are
        # significant in physical size. This avoids the "single-largest-only" failure mode
        # where thin structures (e.g., ventricles) fragment into multiple pieces.
        #
        # Heuristic default: keep components with volume >= 2.0 cc (2000 mm^3). If none meet
        # the threshold, fall back to keeping the largest component (to avoid an empty mask).
        if keep_largest_cc:
            _debug_write_mask(mask, stage=stage, role=role, kind=f"intensity_{tail}_raw")
            try:
                min_cc = 2.0  # cubic centimeters
                # Label connected components
                cc_img = sitk.ConnectedComponent(sitk.Cast(mask, sitk.sitkUInt8))
                rel = sitk.RelabelComponent(cc_img, sortByObjectSize=True)

                # Compute physical volume threshold in mm^3
                sx, sy, sz = (float(x) for x in img.GetSpacing())
                voxel_mm3 = sx * sy * sz
                thr_mm3 = float(min_cc) * 1000.0

                # Use LabelShapeStatistics to get voxel counts per label
                ls = sitk.LabelShapeStatisticsImageFilter()
                ls.Execute(rel)

                keep_labels = []
                for lab in ls.GetLabels():
                    nvox = float(ls.GetNumberOfPixels(int(lab)))
                    if (nvox * voxel_mm3) >= thr_mm3:
                        keep_labels.append(int(lab))

                if not keep_labels:
                    keep_labels = [1]  # fall back to largest

                # Build mask from kept labels (numpy is simplest here)
                lab_arr = sitk.GetArrayFromImage(rel)  # z,y,x
                keep_arr = np.isin(lab_arr, keep_labels).astype(np.uint8, copy=False)
                mask = sitk.GetImageFromArray(keep_arr)
                mask.CopyInformation(rel)
            except Exception:
                # If CC filtering fails, fall back to the original mask.
                pass
            _debug_write_mask(mask, stage=stage, role=role, kind=f"intensity_{tail}_significant")
        
        mask = _maybe_dilate_mask(mask, img=img, dilate_vox=dilate_vox)

        _debug_write_mask(mask, stage=stage, role=role, kind=f"intensity_{tail}")
        return sitk.Cast(mask, sitk.sitkUInt8)


    def _normalize_metric_focus_mode(value: str) -> str:
        """Normalize metric_focus values (incl. aliases) to a canonical mode string."""
        mf_raw = str(value).strip().lower()

        # Accept a few convenience synonyms.
        aliases = {
            "": "none",
            "off": "none",
            "false": "none",
            "0": "none",
            "none": "none",
            "fg": "foreground",
            "fore": "foreground",
            "foreground": "foreground",
            "edges": "edges",
            "edge": "edges",
            "low-high": "lowhigh",
            "low_high": "lowhigh",
            "lowhigh": "lowhigh",
            "high-low": "highlow",
            "high_low": "highlow",
            "highlow": "highlow",
            # Background subtraction (use all voxels, but set background intensities to 0)
            "background_subtracted": "background_subtracted",
            "background-subtracted": "background_subtracted",
            "backgroundsubtracted": "background_subtracted",
            "bgsub": "background_subtracted",
            "bg_sub": "background_subtracted",
            "bgs": "background_subtracted",
        }
        return aliases.get(mf_raw, mf_raw)

    def _background_subtract(img: sitk.Image, *, stage: str, role: str) -> sitk.Image:
        """Zero background voxels (registration-frame only) while preserving geometry.

        This is *not* a metric mask: the metric still sees all sampled voxels, but
        background is set to 0 so it contributes less noise / fewer spurious edges.
        """
        # Foreground mask computed in robust [0,1] space.
        norm01 = _robust_normalize_to_0_1(img)
        fg = _make_foreground_mask(norm01)
        _debug_write_mask(fg, stage=stage, role=role, kind="bgsub_foreground")

        # IMPORTANT: apply to the *original* intensity image (not normalized),
        # so the metric still sees the native modality contrast in the foreground.
        out = sitk.Mask(img, fg, outsideValue=0.0)
        return sitk.Cast(out, sitk.sitkFloat32)

    def _maybe_background_subtract_images(
        fixed_img: sitk.Image,
        moving_img: sitk.Image,
        *,
        stage: str,
    ) -> tuple[sitk.Image, sitk.Image]:
        """Apply background subtraction if metric_focus requests it."""
        mf = _normalize_metric_focus_mode(metric_focus)
        if mf != "background_subtracted":
            return fixed_img, moving_img

        try:
            f2 = _background_subtract(fixed_img, stage=stage, role="fixed")
            m2 = _background_subtract(moving_img, stage=stage, role="moving")
            return f2, m2
        except Exception as e:
            if verbose:
                print(f"[register_images] WARNING: background_subtracted preprocessing failed at stage={stage}: {type(e).__name__}: {e}")
            return fixed_img, moving_img
    def _maybe_set_metric_focus_masks(
        reg_method: sitk.ImageRegistrationMethod,
        fixed_img: sitk.Image,
        moving_img: sitk.Image,
        stage: str,
    ):
        mf = _normalize_metric_focus_mode(metric_focus)

        # background_subtracted is handled by preprocessing the images; it does not
        # set metric masks (metric still uses all voxels).
        if mf in ("none", "background_subtracted"):
            return

        supported = {"foreground", "edges", "lowhigh", "highlow"}
        if mf not in supported:
            raise ValueError(
                f"Unknown metric_focus={metric_focus!r}. Supported: 'none', 'background_subtracted', 'foreground', 'edges', 'lowhigh', 'highlow'."
            )

        try:
            if mf == "foreground":
                fxm = _make_focus_mask_foreground(
                    fixed_img,
                    dilate_vox=int(metric_focus_dilate_vox),
                    stage=stage,
                    role="fixed",
                )
                mvm = _make_focus_mask_foreground(
                    moving_img,
                    dilate_vox=int(metric_focus_dilate_vox),
                    stage=stage,
                    role="moving",
                )
                extra = f"(dilate={int(metric_focus_dilate_vox)})"

            elif mf == "edges":
                fxm = _make_focus_mask_edges(
                    fixed_img,
                    percentile=float(metric_focus_percentile),
                    sigma_mm=float(metric_focus_sigma_mm),
                    dilate_vox=int(metric_focus_dilate_vox),
                    stage=stage,
                    role="fixed",
                )
                mvm = _make_focus_mask_edges(
                    moving_img,
                    percentile=float(metric_focus_percentile),
                    sigma_mm=float(metric_focus_sigma_mm),
                    dilate_vox=int(metric_focus_dilate_vox),
                    stage=stage,
                    role="moving",
                )
                extra = f"(pct={float(metric_focus_percentile)}, sigma_mm={float(metric_focus_sigma_mm)}, dilate={int(metric_focus_dilate_vox)})"

            else:
                # Intensity-tail modes:
                # - lowhigh: fixed uses low tail, moving uses high tail
                # - highlow: fixed uses high tail, moving uses low tail
                fixed_tail = "low" if mf == "lowhigh" else "high"
                moving_tail = "high" if mf == "lowhigh" else "low"

                fxm = _make_focus_mask_intensity_tail(
                    fixed_img,
                    tail=fixed_tail,
                    percentile=float(metric_focus_percentile),
                    dilate_vox=int(metric_focus_dilate_vox),
                    stage=stage,
                    role="fixed",
                    keep_largest_cc=True,
                )
                mvm = _make_focus_mask_intensity_tail(
                    moving_img,
                    tail=moving_tail,
                    percentile=float(metric_focus_percentile),
                    dilate_vox=int(metric_focus_dilate_vox),
                    stage=stage,
                    role="moving",
                    keep_largest_cc=True,
                )
                extra = f"(pct={float(metric_focus_percentile)}, dilate={int(metric_focus_dilate_vox)})"

            reg_method.SetMetricFixedMask(fxm)
            reg_method.SetMetricMovingMask(mvm)

            if verbose:
                fcnt = int(np.count_nonzero(sitk.GetArrayViewFromImage(fxm)))
                mcnt = int(np.count_nonzero(sitk.GetArrayViewFromImage(mvm)))
                print(f"[register_images] metric_focus='{mf}' ({stage}): fixed_mask_vox={fcnt} moving_mask_vox={mcnt} {extra}")
        except Exception as e:
            if verbose:
                print(f"[register_images] WARNING: metric_focus masks failed at stage={stage}: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Helper: sanitize NIfTI headers written by ITK/SimpleITK (can sometimes
    # emit pixdim/zooms that trigger nibabel warnings or downstream failures).
    # ------------------------------------------------------------------
    def _sanitize_nifti_header_inplace(path: str):
        try:
            import numpy as np
            import nibabel as nib

            img = nib.load(path)
            aff = np.array(img.affine, dtype=float, copy=True)
            hdr = img.header.copy()

            # Fix invalid/zero/NaN zooms
            try:
                zooms = tuple(float(z) for z in hdr.get_zooms()[:3])
            except Exception:
                zooms = (None, None, None)

            def _valid_zooms(zs):
                return (zs is not None) and all((z is not None and np.isfinite(z) and float(z) > 0) for z in zs)

            if not _valid_zooms(zooms):
                col_norms = tuple(float(np.linalg.norm(aff[:3, i])) for i in range(3))
                fixed_zooms = tuple((n if (np.isfinite(n) and n > 0) else 1.0) for n in col_norms)
                rest = list(hdr.get_zooms()[3:]) if len(hdr.get_zooms()) > 3 else []
                try:
                    hdr.set_zooms(tuple(list(fixed_zooms) + rest))
                except Exception:
                    pass
                if debug:
                    print(f"[register_images][debug] Sanitized zooms for {path}: {zooms} -> {fixed_zooms}")


            # Clear any scaling fields that can become NaN (can break nibabel affine decomposition).
            try:
                slope = float(hdr.get("scl_slope", 1.0))
                inter = float(hdr.get("scl_inter", 0.0))
                if (not np.isfinite(slope)) or slope == 0.0:
                    hdr["scl_slope"] = 1.0
                if not np.isfinite(inter):
                    hdr["scl_inter"] = 0.0
            except Exception:
                pass

            # Ensure qform/sform are set to the affine so downstream tools don't fall back to base affine
            try:
                hdr.set_qform(aff, code=1)
                hdr.set_sform(aff, code=1)
            except Exception:
                pass

            # Rewrite using a fresh NIfTI container (keeps geometry consistent)
            data = img.get_fdata(dtype=np.float32)
            out = nib.Nifti1Image(data, aff, header=hdr)
            nib.save(out, path)
        except Exception:
            # Best-effort only
            return


    # ------------------------------------------------------------------
    # Thread control
    # ------------------------------------------------------------------
    # Prefer per-object thread limits to avoid global collisions when running
    # register_images in parallel. Fall back to a global ITK cap only if the
    # current SimpleITK build does not expose per-object controls.
    from .preprocessing_utils import (
        normalize_n_workers,
        set_sitk_object_threads,
        global_sitk_thread_cap,
        make_sitk_resampler,
        sitk_resample_to_spacing,
        _normalize_spacing_mm,
    )
    n_workers = normalize_n_workers(n_workers)

    # ------------------------------------------------------------------
    # Image I/O: support NIfTI (.nii/.nii.gz) and NRRD (.nrrd)
    # ------------------------------------------------------------------
    # We use SimpleITK for I/O and dimensionality checks so we can support NRRD.
    # NRRD 4D time-series are commonly stored as either:
    #   (a) a true 4D image (dim=4), or
    #   (b) a 3D VectorImage where components-per-pixel == n_frames.

    def _is_nifti_path(p: str) -> bool:
        s = str(p).lower()
        return s.endswith(".nii") or s.endswith(".nii.gz")

    def _is_nrrd_path(p: str) -> bool:
        return str(p).lower().endswith(".nrrd")

    def _read_image_any(path_like) -> sitk.Image:
        p = str(path_like)

        r = sitk.ImageFileReader()
        r.SetFileName(p)
        r.ReadImageInformation()

        dim = int(r.GetDimension())
        ncomp = int(r.GetNumberOfComponents())

        # Preserve vector images (common for NRRD time-series: kinds[..., 'list'])
        if dim == 3 and ncomp > 1:
            r.SetOutputPixelType(sitk.sitkVectorFloat32)
        else:
            r.SetOutputPixelType(sitk.sitkFloat32)

        return r.Execute()

    def _infer_3d4d(img: sitk.Image):
        dim = int(img.GetDimension())
        comps = int(img.GetNumberOfComponentsPerPixel())
        if dim == 4:
            sx, sy, sz, st = img.GetSize()
            return 4, (int(sx), int(sy), int(sz), int(st)), False
        if dim == 3 and comps > 1:
            sx, sy, sz = img.GetSize()
            return 4, (int(sx), int(sy), int(sz), int(comps)), True
        if dim == 3:
            sx, sy, sz = img.GetSize()
            return 3, (int(sx), int(sy), int(sz), 1), False
        raise ValueError(f"Unsupported image dimensionality: dim={dim}, components={comps}")

    def _extract_3d_frame(img: sitk.Image, t: int, *, is_vector_4d: bool) -> sitk.Image:
        if is_vector_4d:
            return sitk.VectorIndexSelectionCast(img, int(t), sitk.sitkFloat32)
        size = list(img.GetSize())
        if len(size) != 4:
            raise ValueError(f"Expected 4D image for frame extraction; got dim={img.GetDimension()}")
        size[3] = 0
        idx = [0, 0, 0, int(t)]
        return sitk.Extract(img, size, idx)

    def _compose_vector_image(frames_3d: list[sitk.Image], spatial_reference: sitk.Image) -> sitk.Image:
        v = sitk.Compose(frames_3d)
        try:
            v.CopyInformation(spatial_reference)
        except Exception:
            pass
        return v

    fixed_img = _read_image_any(fixed_path)
    moving_img = _read_image_any(moving_path)

    fixed_ndim, fixed_shape, fixed_is_vector4d = _infer_3d4d(fixed_img)
    moving_ndim, moving_shape, moving_is_vector4d = _infer_3d4d(moving_img)
    if verbose:
        print(f"[register_images] fixed: dim={fixed_img.GetDimension()} comps={fixed_img.GetNumberOfComponentsPerPixel()} inferred={fixed_ndim} shape={fixed_shape} vector4d={fixed_is_vector4d}")
        print(f"[register_images] moving: dim={moving_img.GetDimension()} comps={moving_img.GetNumberOfComponentsPerPixel()} inferred={moving_ndim} shape={moving_shape} vector4d={moving_is_vector4d}")

    if fixed_ndim not in (3, 4) or moving_ndim not in (3, 4):
        raise ValueError(
            f"register_images supports only 3D/4D inputs. "
            f"Got fixed_ndim={fixed_ndim} moving_ndim={moving_ndim}."
        )

    # For registration, pick 3D frames if needed.
    fixed_for_reg = fixed_img if fixed_ndim == 3 else _extract_3d_frame(fixed_img, int(fixed_frame_index), is_vector_4d=fixed_is_vector4d)
    moving_for_reg = moving_img if moving_ndim == 3 else _extract_3d_frame(moving_img, int(moving_frame_index), is_vector_4d=moving_is_vector4d)

    # ------------------------------------------------------------------
    # Optionally treat 4D as 3D by taking a single frame
    # ------------------------------------------------------------------
    if use_first_frame_only and moving_ndim == 4:
        if verbose:
            print(f"[register_images] treating moving 4D as 3D (frame {moving_frame_index})")
        moving_img = moving_for_reg
        moving_ndim = 3
        moving_shape = None
        # IMPORTANT: now that moving is effectively 3D, keep these aligned:
        moving_for_reg = moving_img


    # ------------------------------------------------------------------
    # Optional downsampling for transform estimation (registration-time only)
    # ------------------------------------------------------------------
    # User can request a target spacing (mm, mm, mm). This affects only the transform
    # estimation step (apply_only=False). The final resampling to output_path is still
    # performed on the full-resolution fixed grid.
    fixed_reg_img = fixed_for_reg
    moving_reg_img = moving_for_reg

    req_spacing = _normalize_spacing_mm(registration_voxel_mm)
    if req_spacing is not None and not apply_only:
        if verbose:
            print(f"Using {req_spacing} mm voxel spacing for registration optimization.")
        fixed_sp = tuple(float(x) for x in fixed_for_reg.GetSpacing())
        moving_sp = tuple(float(x) for x in moving_for_reg.GetSpacing())

        # Clamp any axis that would be finer than BOTH inputs (avoid unintended upsampling).
        adj = list(req_spacing)
        clamped_axes = []
        for i in range(3):
            finest_in = min(fixed_sp[i], moving_sp[i])
            if adj[i] < finest_in:
                adj[i] = finest_in
                clamped_axes.append(i)
        adj = tuple(float(x) for x in adj)
        if clamped_axes and verbose:
            ax = ["x", "y", "z"]
            which = ",".join(ax[i] for i in clamped_axes)
            print(
                f"[register_images] WARNING: requested registration_voxel_mm={req_spacing} is finer than both inputs "
                f"on axis(es) {which}; clamping to finest available input spacing -> {adj}."
            )

        # Downsample both frames to the requested spacing for faster optimization.
        fixed_reg_img = sitk_resample_to_spacing(
            fixed_for_reg,
            adj,
            interp=interpolation,
            default_value=0.0,
            pixel_id=sitk.sitkFloat32,
            n_workers=n_workers,
        )
        moving_reg_img = sitk_resample_to_spacing(
            moving_for_reg,
            adj,
            interp=interpolation,
            default_value=0.0,
            pixel_id=sitk.sitkFloat32,
            n_workers=n_workers,
        )

    if apply_only:
        if not transform_path or not os.path.isfile(transform_path):
            raise ValueError("Transform file is required and must exist when apply_only=True.")
        transform = sitk.ReadTransform(transform_path)
        if verbose:
            print(f"Applying transform from: {transform_path}")

    else:
        strategy = str(registration_strategy).lower().strip()
        presets = {
            "accurate": dict(
                sampling_fraction=1.0,
                iters=300,
                shrink=[8, 4, 2, 1],
                smooth=[3, 2, 1, 0],
                mi_bins=50,
            ),
            "medium": dict(
                sampling_fraction=0.25,
                iters=200,
                shrink=[4, 2, 1],
                smooth=[2, 1, 0],
                mi_bins=50,
            ),
            "fast": dict(
                sampling_fraction=0.10,
                iters=120,
                shrink=[4, 2, 1],
                smooth=[2, 1, 0],
                mi_bins=32,
            ),
        }
        if strategy not in presets:
            raise ValueError("registration_strategy must be one of: 'accurate', 'medium', 'fast'.")
        p = presets[strategy]

        # Transform family
        if registration_type == "rigid":
            tx = sitk.Euler3DTransform()
        elif registration_type == "affine":
            tx = sitk.AffineTransform(3)
        elif registration_type == "translation":
            tx = sitk.TranslationTransform(3)
        else:
            raise ValueError("Invalid registration_type. Choose 'rigid', 'affine', or 'translation'.")


        # ------------------------------------------------------------------
        # Initializer: translation pre-pass ("translation_then_main")
        # ------------------------------------------------------------------
        # Default behavior: do a coarse translation-only registration on heavily
        # downsampled images, then use that translation to initialize the main
        # rigid/affine/translation optimization. This often reduces iterations to
        # convergence and helps avoid poor local minima for high-res volumes.
        prepass_offset = (0.0, 0.0, 0.0)
        try:
            fixed_sp = tuple(float(x) for x in fixed_for_reg.GetSpacing())
            moving_sp = tuple(float(x) for x in moving_for_reg.GetSpacing())
            req_sp = _normalize_spacing_mm(registration_voxel_mm)
            # Aim for ~2mm isotropic (or coarser), but never upsample beyond the finest available axis.
            # If the user already requested a coarser spacing for the main pass, reuse that.
            prepass_spacing = []
            for i in range(3):
                finest_in = min(fixed_sp[i], moving_sp[i])
                user_req = float(req_sp[i]) if req_sp is not None else 0.0
                prepass_spacing.append(max(2.0, user_req, finest_in))
            prepass_spacing = tuple(prepass_spacing)

            fixed_pre = sitk_resample_to_spacing(
                fixed_for_reg,
                prepass_spacing,
                interp="linear",
                default_value=0.0,
                pixel_id=sitk.sitkFloat32,
                n_workers=n_workers,
            )
            moving_pre = sitk_resample_to_spacing(
                moving_for_reg,
                prepass_spacing,
                interp="linear",
                default_value=0.0,
                pixel_id=sitk.sitkFloat32,
                n_workers=n_workers,
            )

            tx_pre = sitk.TranslationTransform(3)
            reg_pre = sitk.ImageRegistrationMethod()
            
            # If requested, zero background voxels for the *metric images* (no hard masks).
            fixed_pre_use, moving_pre_use = _maybe_background_subtract_images(fixed_pre, moving_pre, stage="prepass")
            _maybe_set_metric_focus_masks(reg_pre, fixed_pre_use, moving_pre_use, stage="prepass")

            # CenteredTransformInitializer returns a transform with a good initial offset.
            # Using GEOMETRY (not MOMENTS) is typically more robust across modalities.
            try:
                tx_pre_init = sitk.CenteredTransformInitializer(
                    fixed_pre_use,
                    moving_pre_use,
                    sitk.TranslationTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )
            except Exception:
                tx_pre_init = sitk.TranslationTransform(3)
            reg_pre.SetInitialTransform(tx_pre_init, inPlace=False)

            set_sitk_object_threads(reg_pre, n_workers)

            # Metric (keep user's choice, but keep it light)
            if similarity_metric == "correlation":
                reg_pre.SetMetricAsCorrelation()
            else:
                reg_pre.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)

            reg_pre.SetShrinkFactorsPerLevel([4, 2, 1])
            reg_pre.SetSmoothingSigmasPerLevel([2, 1, 0])
            reg_pre.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            # Always sample stochastically for speed
            reg_pre.SetMetricSamplingStrategy(reg_pre.RANDOM)
            if metric_sampling_seed is None:
                reg_pre.SetMetricSamplingPercentage(0.10)
            else:
                reg_pre.SetMetricSamplingPercentage(0.10, int(metric_sampling_seed))

            reg_pre.SetInterpolator(sitk.sitkLinear)
            reg_pre.SetOptimizerAsRegularStepGradientDescent(
                learningRate=2.0,
                minStep=1e-4,
                numberOfIterations=60,
                gradientMagnitudeTolerance=1e-6,
            )
            reg_pre.SetOptimizerScalesFromPhysicalShift()

            use_global_cap_pre = (n_workers is not None and not set_sitk_object_threads(reg_pre, n_workers))
            with global_sitk_thread_cap(n_workers, enabled=use_global_cap_pre, verbose=False):
                try:
                    tfm_pre = reg_pre.Execute(fixed_pre_use, moving_pre_use)
                except Exception as e:
                    # Common failure for partial-FOV scans: MI can't evaluate because samples are out of bounds.
                    msg = str(e)
                    if "All samples map outside moving image buffer" in msg:
                        if verbose:
                            print(
                                "[register_images] WARNING: translation pre-pass MI failed due to insufficient overlap; "
                                "falling back to center-alignment translation."
                            )
                        tfm_pre = tx_pre_init
                    else:
                        raise

            # SimpleITK may return a CompositeTransform even for a pure translation stage
            # (e.g., when an initial transform is provided with inPlace=False).
            # CompositeTransform has no GetOffset, so unwrap carefully.
            try:
                tfm_use = tfm_pre
                if hasattr(sitk, "CompositeTransform") and isinstance(tfm_use, sitk.CompositeTransform):
                    n_t = int(tfm_use.GetNumberOfTransforms())
                    if n_t > 0:
                        tfm_use = tfm_use.GetNthTransform(n_t - 1)

                prepass_offset = _transform_translation_vector(tfm_use)
            except Exception as _e:
                if verbose:
                    print(f"[register_images] WARNING: could not read translation from pre-pass transform ({type(tfm_pre)}): {_e}")

            if verbose:
                print(f"[register_images] translation pre-pass spacing={prepass_spacing} offset={prepass_offset}")

        except Exception as e:
            # Pre-pass is a performance/robustness optimization; failure should not abort registration.
            if verbose:
                print(f"[register_images] WARNING: translation pre-pass failed: {type(e).__name__}: {e}")

        fixed_main_use, moving_main_use = _maybe_background_subtract_images(fixed_reg_img, moving_reg_img, stage="main")

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_main_use, moving_main_use, tx, sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        # Add pre-pass translation to the main initializer if supported.
        try:
            base = np.array(initial_transform.GetTranslation(), dtype=float)
            off = np.array(prepass_offset, dtype=float)
            initial_transform.SetTranslation(tuple((base + off).tolist()))
        except Exception:
            pass

        registration = sitk.ImageRegistrationMethod()
        _maybe_set_metric_focus_masks(registration, fixed_main_use, moving_main_use, stage="main")
        registration.SetInitialTransform(initial_transform, inPlace=False)

        # Try to cap threads per-object for registration if supported.
        reg_threads_ok = set_sitk_object_threads(registration, n_workers)

        # Metric
        if similarity_metric == "correlation":
            registration.SetMetricAsCorrelation()
        elif similarity_metric == "mi":
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(p["mi_bins"]))
        else:
            raise ValueError("Invalid similarity_metric. Choose 'correlation' or 'mi'.")

        # Multi-resolution pyramid
        registration.SetShrinkFactorsPerLevel([int(x) for x in p["shrink"]])
        registration.SetSmoothingSigmasPerLevel([float(x) for x in p["smooth"]])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Sampling
        f = float(p["sampling_fraction"])
        if f >= 1.0:
            registration.SetMetricSamplingStrategy(registration.NONE)
            if verbose:
                print(f"[{strategy}] metric sampling: NONE (all voxels)")
        else:
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            if metric_sampling_seed is None:
                registration.SetMetricSamplingPercentage(f)
            else:
                registration.SetMetricSamplingPercentage(f, int(metric_sampling_seed))
            if verbose:
                print(f"[{strategy}] metric sampling: RANDOM ({f:.3f} of voxels), seed={metric_sampling_seed}")

        # Optimizer
        registration.SetInterpolator(sitk_interp)
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0,
            minStep=1e-4,
            numberOfIterations=int(p["iters"]),
            gradientMagnitudeTolerance=1e-6,
        )
        registration.SetOptimizerScalesFromPhysicalShift()

        # If per-object thread control isn't available for registration, we can
        # optionally fall back to a global cap (LAST RESORT).
        use_global_cap = (n_workers is not None and not reg_threads_ok)
        with global_sitk_thread_cap(n_workers, enabled=use_global_cap, verbose=verbose):
            transform = registration.Execute(fixed_main_use, moving_main_use)

        if verbose:
            print(f"Final {similarity_metric} = {registration.GetMetricValue():.4f}")

        if transform_path:
            sitk.WriteTransform(transform, transform_path)
            _write_transform_meta(
                transform_path,
                {
                    "interp": interp_spec if interp_spec is not None else "linear",
                    "sitk_interpolator": sitk_interp_name,
                    "scipy_order": int(scipy_order),
                    "output_grid": "moving" if keep_moving_grid else "fixed",
                    "keep_moving_grid": bool(keep_moving_grid),
                    "fixed_frame_index": int(fixed_frame_index),
                    "moving_frame_index": int(moving_frame_index),
                    "use_first_frame_only": bool(use_first_frame_only),
                    "registration_type": str(registration_type),
                    "similarity_metric": str(similarity_metric),
                    "registration_strategy": str(registration_strategy),
                },
            )

    # ------------------------------------------------------------------
    # Resample moving -> fixed space
    # ------------------------------------------------------------------
    if verbose:
        print(f"Resampling moving image...")
    # Spatial reference for resampling: always a 3D grid.
    # Default behavior: resample onto the fixed grid.
    # Optional behavior (keep_moving_grid=True): resample onto the moving grid instead.
    if keep_moving_grid:
        spatial_ref = moving_img if moving_ndim == 3 else moving_for_reg
    else:
        spatial_ref = (moving_img if moving_ndim == 3 else moving_for_reg) if keep_moving_grid else (fixed_img if fixed_ndim == 3 else fixed_for_reg)

    if moving_ndim == 3:
        # Use ResampleImageFilter so we can cap threads per-object.
        resampler = make_sitk_resampler(
            reference_img=spatial_ref,
            transform=transform,
            default_value=0.0,
            pixel_id=moving_img.GetPixelID(),
            n_workers=n_workers,
            interp=interp_spec,
        )
        registered = resampler.Execute(moving_img)

        # If integer output requested (e.g., labelmap), cast appropriately
        if integer:
            orig_pixel_id = moving_img.GetPixelID()
            if "UInt" in sitk.GetPixelIDValueAsString(orig_pixel_id):
                registered = sitk.Cast(registered, orig_pixel_id)
            else:
                registered = sitk.Cast(registered, sitk.sitkUInt16)
 
        # Preserve NRRD metadata (e.g., 3D Slicer segment names) without copying geometry-ish keys.
        if _is_nrrd_path(output_path):
            _copy_nrrd_metadata_nongeom(moving_img, registered)

        if verbose:
            print(f"Saving resampled image...")
        sitk.WriteImage(registered, str(output_path))
        # Sanitize header to avoid nibabel warnings / decompositions later.
        #_sanitize_nifti_header_inplace(str(output_path)) #skip for now
    else:
        # 4D wrapper
        # - If output is NIfTI, write a 4D NIfTI (xyzt) via nibabel (preserves fixed affine).
        # - If output is NRRD, write a 3D VectorImage (n_frames components) via SimpleITK.

        if _is_nrrd_path(output_path) and moving_ndim == 4 and verbose:
            print(
                "Saving 4D NRRD output is experimental. "
                "Axis structure and metadata (e.g. time-series semantics) "
                "may not be preserved, and some viewers may not recognize "
                "the result as a time series. If using with 3D Slicer,"
                "consider saving output as .seq.nrrd."
            )
        
        n_frames = int(moving_shape[3])
        spatial_ref = (moving_img if moving_ndim == 3 else moving_for_reg) if keep_moving_grid else (fixed_img if fixed_ndim == 3 else fixed_for_reg)

        resampler = make_sitk_resampler(
            reference_img=spatial_ref,
            transform=transform,
            default_value=0.0,
            pixel_id=moving_for_reg.GetPixelID(),
            n_workers=n_workers,
            interp=interp_spec,
        )

        if _is_nrrd_path(output_path):
            # Write as VectorImage so viewers (e.g., 3D Slicer MultiVolume) can treat it as a volume sequence.
            if verbose:
                print(f"Resampling {n_frames} frames for NRRD VectorImage output...")
            out_frames = []
            for t in range(n_frames):
                frame3d = _extract_3d_frame(moving_img, t, is_vector_4d=moving_is_vector4d)
                reg3d = resampler.Execute(frame3d)
                if integer:
                    orig_pixel_id = frame3d.GetPixelID()
                    if "UInt" in sitk.GetPixelIDValueAsString(orig_pixel_id):
                        reg3d = sitk.Cast(reg3d, orig_pixel_id)
                    else:
                        reg3d = sitk.Cast(reg3d, sitk.sitkUInt16)
                out_frames.append(reg3d)

            vimg = _compose_vector_image(out_frames, spatial_reference=spatial_ref)
 
            # Preserve NRRD metadata from the original moving image (non-geometry keys).
            # Note: if moving_img is a Slicer segmentation/labelmap NRRD, this retains Segment*_Name, etc.
            _copy_nrrd_metadata_nongeom(moving_img, vimg)
 
            if verbose:
                print("Saving 4D-as-VectorImage output via SimpleITK...")
            sitk.WriteImage(vimg, str(output_path))

        else:
            # NIfTI output path: preserve affine via nibabel.
            if not (_is_nifti_path(fixed_path) and _is_nifti_path(moving_path) and _is_nifti_path(output_path)):
                raise ValueError(
                    "Writing 4D NIfTI output currently requires fixed/moving/output to all be NIfTI "
                    "(.nii or .nii.gz). For NRRD inputs/outputs, write .nrrd instead."
                )

            import numpy as np
            import nibabel as nib
            from datetime import datetime

            ref_nii = nib.load(str(moving_path if keep_moving_grid else fixed_path))
            ref_affine = ref_nii.affine

            # Target grid size from the 3D reference (SimpleITK size is (x,y,z))
            sx, sy, sz = spatial_ref.GetSize()
            out_xyzt = np.zeros((sx, sy, sz, n_frames), dtype=np.float32)

            if verbose:
                print(f"Resampling {n_frames} frames (4D wrapper) starting at {datetime.now()}")

            for t in range(n_frames):
                frame3d = _extract_3d_frame(moving_img, t, is_vector_4d=moving_is_vector4d)
                reg3d = resampler.Execute(frame3d)
                if integer:
                    orig_pixel_id = frame3d.GetPixelID()
                    if "UInt" in sitk.GetPixelIDValueAsString(orig_pixel_id):
                        reg3d = sitk.Cast(reg3d, orig_pixel_id)
                    else:
                        reg3d = sitk.Cast(reg3d, sitk.sitkUInt16)

                # SITK -> numpy gives (z,y,x)
                reg_zyx = sitk.GetArrayFromImage(reg3d).astype(np.float32, copy=False)
                # Convert to (x,y,z)
                reg_xyz = np.transpose(reg_zyx, (2, 1, 0))

                if reg_xyz.shape != (sx, sy, sz):
                    raise RuntimeError(
                        f"Frame {t} resampled shape {reg_xyz.shape} != expected {(sx, sy, sz)}. "
                        f"spatial_ref size={spatial_ref.GetSize()}"
                    )

                out_xyzt[..., t] = reg_xyz

            if verbose:
                print(f"4D stacking complete at {datetime.now()}")

            out_nii = nib.Nifti1Image(out_xyzt, ref_affine)
            hdr = out_nii.header
            hdr.set_data_dtype(np.float32)

            # Copy zooms: spatial from fixed; time from moving if present
            fixed_zooms = ref_nii.header.get_zooms()[:3]
            moving_zooms = nib.load(str(moving_path)).header.get_zooms()
            tzoom = float(moving_zooms[3]) if len(moving_zooms) > 3 else 1.0
            hdr.set_zooms(tuple(list(fixed_zooms) + [tzoom]))

            try:
                qcode = int(ref_nii.header["qform_code"]) or 1
                scode = int(ref_nii.header["sform_code"]) or 1
            except Exception:
                qcode, scode = 1, 1
            out_nii.set_qform(ref_affine, code=qcode)
            out_nii.set_sform(ref_affine, code=scode)

            try:
                hdr["scl_slope"] = 1.0
                hdr["scl_inter"] = 0.0
            except Exception:
                pass

            if verbose:
                print(f"Saving 4D output via nibabel (xyzt) starting at {datetime.now()}")
            nib.save(out_nii, str(output_path))
            if verbose:
                print(f"Save completed at {datetime.now()}")
    # Optional dummy references
    if save_dummy_ref and transform_path:
        base = os.path.splitext(str(transform_path))[0]
        fixed_dummy_path = base + "-fixed-ref.nii.gz"
        moving_dummy_path = base + "-moving-ref.nii.gz"

        for ref_img, path in [(fixed_for_reg, fixed_dummy_path), (moving_for_reg, moving_dummy_path)]:
            zero_array = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.float32)
            dummy = sitk.GetImageFromArray(zero_array)
            dummy.CopyInformation(ref_img)
            sitk.WriteImage(dummy, path)
            if verbose:
                print(f"Dummy reference saved to: {path}")

    # If this looks like diffusion data (FSL .bval/.bvec next to moving_path),
    # copy/update sidecars so downstream tools see the correct gradient table after resampling.
    try:
        from .preprocessing_utils import update_fsl_vectors_after_transform
        update_fsl_vectors_after_transform(
            str(moving_path),
            str(output_path),
            transform,
            inverse=False,
            verbose=verbose,
        )
    except Exception:
        # Non-fatal: image resampling succeeded, but sidecar update may not apply.
        pass

    if verbose:
        print(f"Output saved to: {output_path}")
        if transform_path and not apply_only:
            print(f"Transform saved to: {transform_path}")


# ---------------------------------------------------------------------------------------
# Function to apply inverse of a transform previously created during a registration step
# ---------------------------------------------------------------------------------------

def inverse_transform_image(
    original_image_path,
    transformed_image_path,
    transform_path,
    output_path,
    interpolation="linear",
    verbose=True,
    *,
    original_frame_index: int = 0,
):
    """
    Apply the inverse of a saved transform to return an image to its original space.

    This supports both 3D and 4D NIfTI:
      - If transformed_image_path is 3D, writes a 3D output.
      - If transformed_image_path is 4D, applies the inverse transform to each 3D frame in-memory and writes a 4D output.

    Parameters
    ----------
    original_image_path : str | PathLike
        Path to the original (pre-registered) image defining the **reference grid** to recover into.
        If 4D, the spatial grid is taken from `original_frame_index`.
    transformed_image_path : str | PathLike
        Path to the transformed image (3D or 4D) to be mapped back into the original grid.
    transform_path : str | PathLike
        Path to the forward transform (.tfm) that produced the transformed image.
    output_path : str | PathLike
        Output path for the recovered image.
    interpolation : {'linear','nearest'}
        Interpolation for resampling.
    verbose : bool
        Print summary information.
    original_frame_index : int
        When original_image_path is 4D, which frame to use as the spatial reference grid.

    Notes
    -----
    - This function assumes the forward transform maps: original -> transformed_reference_space.
      It applies the inverse to map: transformed -> original_reference_grid.
    """
    import os
    import json
    import SimpleITK as sitk
    from .preprocessing_utils import get_nifti_ndim, sitk_extract_3d_from_4d, sitk_join_3d_frames_to_4d

    orig_ndim, orig_shape = get_nifti_ndim(original_image_path)
    tr_ndim, tr_shape = get_nifti_ndim(transformed_image_path)

    if orig_ndim not in (3, 4) or tr_ndim not in (3, 4):
        raise ValueError(
            f"inverse_transform_image supports only 3D/4D NIfTI. "
            f"Got original_ndim={orig_ndim} transformed_ndim={tr_ndim}."
        )

    original_img = sitk.ReadImage(str(original_image_path), sitk.sitkFloat32)
    transformed_img = sitk.ReadImage(str(transformed_image_path), sitk.sitkFloat32)

    # Choose a 3D spatial reference grid to recover into
    original_ref_3d = original_img if orig_ndim == 3 else sitk_extract_3d_from_4d(original_img, int(original_frame_index))

    transform = sitk.ReadTransform(str(transform_path))
    try:
        inverse_transform = transform.GetInverse()
    except Exception as e:
        raise ValueError(f"Failed to compute inverse transform for '{transform_path}': {e}")
    # Interpolation: by default, reuse whatever was used for the forward registration if available.
    from .preprocessing_utils import _interp_to_sitk_interpolator

    interp_spec = interpolation
    if isinstance(interp_spec, str) and interp_spec.strip().lower() == "auto":
        interp_spec = None
        # Try to read from register_images() sidecar JSON.
        try:
            meta_path = os.path.splitext(str(transform_path))[0] + ".meta.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict) and "interp" in meta:
                interp_spec = meta["interp"]
                if verbose:
                    print(f"[inverse_transform_image] Using recorded interpolation from sidecar: {interp_spec!r}")
        except Exception:
            interp_spec = None

    interp_method, interp_name, _ = _interp_to_sitk_interpolator(interp_spec)

    if tr_ndim == 3:
        recovered = sitk.Resample(
            transformed_img,
            original_ref_3d,
            inverse_transform,
            interp_method,
            0.0,
            transformed_img.GetPixelID(),
        )
        sitk.WriteImage(recovered, str(output_path))
    else:
        # 4D: invert each 3D frame in memory, then re-stack to 4D.
        n_frames = int(tr_shape[3])
        frames_out = []
        for t in range(n_frames):
            frame3d = sitk_extract_3d_from_4d(transformed_img, t)
            recovered3d = sitk.Resample(
                frame3d,
                original_ref_3d,
                inverse_transform,
                interp_method,
                0.0,
                frame3d.GetPixelID(),
            )
            frames_out.append(recovered3d)

        out4d = sitk_join_3d_frames_to_4d(frames_out, spatial_reference=original_ref_3d, time_reference=transformed_img)
        sitk.WriteImage(out4d, str(output_path))

    # If diffusion sidecars exist next to the transformed image, rotate/copy them
    # to remain consistent with the inverse-resampled output.
    try:
        from .preprocessing_utils import update_fsl_vectors_after_transform
        update_fsl_vectors_after_transform(
            str(transformed_image_path),
            str(output_path),
            inverse_transform,
            inverse=True,
            verbose=verbose,
        )
    except Exception:
        pass

    if verbose:
        print(f"Inverse-transformed image saved to: {output_path}")

# ---------------------------------------------------------------------------------------
# Function to run hd-bet for brainmask creation
# ---------------------------------------------------------------------------------------

def run_hd_bet(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike | None = None,
    mask_path: str | os.PathLike | None = None,
    device: str = "cpu",
    disable_tta: bool = True,
    verbose: bool = False,
    n_workers: int | None = None,
):
    """
    HD-BET v2.x CLI wrapper.

    - If output_path is provided: writes brain-extracted image there (default HD-BET output).
    - If mask_path is provided: also writes a brain mask (HD-BET decides the on-disk naming; we discover it and move it).
    - If only mask_path is provided: we force no bet image via --no_bet_image
    - Suggested to disable TTA if running on CPU for speed
    - Set device to cuda to run on GPU
    - If n_workers is provided and device is CPU-like, best-effort limit of CPU threads is applied
      PER SUBPROCESS via environment variables (OMP/MKL/OpenBLAS/NumExpr/Torch). This avoids
      global collisions when running multiple preprocessing pipelines in parallel.
    """
    from .preprocessing_utils import ensure_hd_bet_installed
    ensure_hd_bet_installed()

    input_path = str(input_path)

    if output_path is None and mask_path is None:
        raise ValueError("Must provide at least one of output_path or mask_path.")

    # ---------------------------
    # Per-subprocess thread control (CPU mode)
    # ---------------------------
    def _normalize_n_workers(n):
        if n is None:
            return None
        try:
            n = int(n)
        except Exception:
            raise ValueError(f"n_workers must be int or None; got {n!r}")
        if n <= 0:
            raise ValueError(f"n_workers must be >= 1 or None; got {n}")
        return n

    n_workers = _normalize_n_workers(n_workers)

    # Decide how output paths should be handled
    if output_path is not None and mask_path is not None:
        out_for_cli = str(output_path)
        out_is = "skullstripped_scan"
    elif output_path is not None:
        out_for_cli = str(output_path)
        out_is = "skullstripped_scan"
    else:
        out_for_cli = str(mask_path)
        out_is = "mask"

    cmd = ["hd-bet", "-i", input_path, "-o", out_for_cli, "-device", str(device)]

    # TTA behavior: defaults to TTA enabled; disabling is a speed optimization (esp. on CPU).
    if disable_tta is True:
        cmd.append("--disable_tta")

    # Ask for mask if requested
    if mask_path is not None:
        cmd.append("--save_bet_mask")

    # If user doesn't want a bet image, explicitly request that.
    if output_path is None:
        cmd.append("--no_bet_image")

    if verbose:
        cmd.append("--verbose")

    if verbose:
        print("Running command:", " ".join(cmd))

    with contextlib.ExitStack():
        stdout = subprocess.DEVNULL if not verbose else None
        stderr = subprocess.DEVNULL if not verbose else None

        # Build a per-call environment so parallel runs don't collide.
        env = os.environ.copy()
        dev = str(device).lower().strip()
        is_cpu = (dev == "cpu") or (dev.startswith("cpu"))
        if is_cpu and n_workers is not None:
            # These cover the common BLAS/OpenMP backends and PyTorch intraop threading.
            # Note: hd-bet is a separate process, so this is the safest per-call control mechanism.
            env.update(
                {
                    "OMP_NUM_THREADS": str(n_workers),
                    "MKL_NUM_THREADS": str(n_workers),
                    "OPENBLAS_NUM_THREADS": str(n_workers),
                    "NUMEXPR_NUM_THREADS": str(n_workers),
                    "TORCH_NUM_THREADS": str(n_workers),
                }
            )
            if verbose:
                print(f"[run_hd_bet] CPU thread cap for this subprocess: {n_workers}")

        subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr, env=env)

    # If a mask was requested in addition to skullstripped scan, find what HD-BET produced and move it to mask_path.
    if mask_path is not None:
        out_p = Path(out_for_cli)

        # Heuristic: HD-BET historically used "<output>_bet.nii.gz".
        candidates = []

        # Same folder, same stem prefix, containing "bet"
        parent = out_p.parent
        stem = out_p.name
        for p in parent.glob("*"):
            name = p.name.lower()
            if "_bet" in name and (stem.lower().split(".nii")[0] in name):
                if p.suffix in [".nii"] or name.endswith(".nii.gz"):
                    candidates.append(p)

        # Fallback: common legacy naming patterns
        if not candidates:
            if out_for_cli.endswith(".nii.gz"):
                candidates.append(Path(out_for_cli[:-7] + "_bet.nii.gz"))
            elif out_for_cli.endswith(".nii"):
                candidates.append(Path(out_for_cli[:-4] + "_bet.nii"))

        candidates = [p for p in candidates if p.exists()]
        if not candidates:
            raise FileNotFoundError(
                f"HD-BET finished but no mask file was found near {out_for_cli}. "
                f"Try running with verbose=True to see what it writes."
            )

        # Prefer the most recently modified candidate
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if verbose:
            print(f"Moving {str(candidates[0])} to {str(mask_path)}.")
        os.replace(str(candidates[0]), str(mask_path))

# ---------------------------------------------------------------------------------------
# Function to do basic math between MRI volumes
# ---------------------------------------------------------------------------------------

def perform_mri_math(args):
    import numpy as np
    from .preprocessing_utils import load_nifti_data, save_nifti_data, validate_volume_shapes

    if args.applymask:
        img_data, affine, header = load_nifti_data(args.input)
        mask_data, mask_affine, _ = load_nifti_data(args.mask)

        if img_data.shape != mask_data.shape:
            raise ValueError("Input and mask must have the same shape.")
        if not np.allclose(affine, mask_affine):
            warnings.warn("Affine matrices do not match between input and mask.")

        result = img_data * (mask_data > 0)
        save_nifti_data(result, affine, header, args.output)

    elif args.average:
        volumes = [load_nifti_data(p) for p in args.average]
        validate_volume_shapes(volumes)
        data = np.mean([v[0] for v in volumes], axis=0)
        save_nifti_data(data, volumes[0][1], volumes[0][2], args.output)

    elif args.operation:
        if not args.inputs or len(args.inputs) > 26:
            raise ValueError("You must provide between 1 and 26 input files via --inputs.")

        # Map A–Z to the input files
        var_names = [chr(i) for i in range(ord('A'), ord('A') + len(args.inputs))]
        variable_map = dict(zip(var_names, args.inputs))

        # Load and validate
        volume_data = {var: load_nifti_data(fname) for var, fname in variable_map.items()}
        validate_volume_shapes(list(volume_data.values()))

        # Build AST and evaluate
        expr = args.operation
        allowed_vars = set(variable_map.keys())
        allowed_funcs = {"where", "log", "log10", "exp"}

        parsed = ast.parse(expr, mode='eval')

        class SafeTransformer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id not in allowed_vars and node.id not in allowed_funcs:
                    raise ValueError(f"Disallowed variable or function in expression: {node.id}")
                return node

        SafeTransformer().visit(parsed)
        data_arrays = {k: v[0] for k, v in volume_data.items()}
        safe_namespace = {
            **data_arrays,
            "where": np.where,
            "log": np.log,
            "log10": np.log10,
            "exp": np.exp
        }

        try:
            result = eval(compile(parsed, "<string>", "eval"), {}, safe_namespace)
    
            if not np.isfinite(result).all():
                raise ValueError("Result contains non-finite values (NaN or inf) possibly due to invalid operation (e.g. log of negative values).")

        except Exception as e:
            print(f"[ERROR] Failed to evaluate expression: {e}")
            print("[ABORTED] No output was saved.")
            return  # Exit function without saving

        affine, header = list(volume_data.values())[0][1:]
        save_nifti_data(result, affine, header, args.output)

# -------------------------------------------------------------------------------------------
# Function to re-apply or reverse all transformations recorded in a pipeline transform json
# -------------------------------------------------------------------------------------------

def apply_or_reverse_transforms(
    input_path,
    transform_record_path,
    output_path,
    mode="apply",
    interp=None,  # <-- None means "use stored interpolation unless user overrides"
):
    """
    Apply or reverse a transform pipeline defined in a transform_record.json.

    Args:
        input_path (str): Path to the scan to transform.
        transform_record_path (str): Path to the transform_record.json file.
        output_path (str): Where to write the transformed result.
        mode (str): "apply" or "reverse".
        interp (int|str|None): If provided, overrides stored interpolation.
            - For resize steps: 0=nearest, 1=linear
            - For inverse .tfm steps: "nearest" or "linear"
            If None, interpolation is inferred from each transform_record entry (fallback=linear).
    """
    assert mode in ["apply", "reverse"], "mode must be 'apply' or 'reverse'"

    import os
    import json
    import shutil
    import tempfile

    from .preprocessing_utils import read_padding_record
    # assumes these are in-scope/importable in your module
    # from .preprocessing_functions import register_images, inverse_transform_image, resize_mri, reverse_resize_mri

    def _normalize_interp_override(user_interp):
        """
        Returns:
          (resize_order_int, sitk_interp_str) or (None, None) if user_interp is None
        """
        if user_interp is None:
            return None, None
        if isinstance(user_interp, str):
            s = user_interp.strip().lower()
            if s in ("linear", "lin", "1"):
                return 1, "linear"
            if s in ("nearest", "nn", "0"):
                return 0, "nearest"
            raise ValueError(f"interp override string must be 'linear' or 'nearest' (got {user_interp!r})")
        try:
            i = int(user_interp)
        except Exception:
            raise ValueError(f"interp override must be int/str/None (got {user_interp!r})")
        if i not in (0, 1):
            raise ValueError(f"interp override int must be 0 or 1 (got {i})")
        return i, ("nearest" if i == 0 else "linear")

    def _infer_interp_from_record(record_entry):
        """
        Try a few likely keys. Supports:
          - int 0/1
          - str "nearest"/"linear"
        Falls back to linear.
        """
        if not isinstance(record_entry, dict):
            return 1, "linear"

        # Common key candidates (add/remove as needed to match your record)
        for k in ("interpolation", "interp", "resample_interp", "resample_interpolation"):
            if k in record_entry and record_entry[k] is not None:
                v = record_entry[k]
                # reuse override parser for consistent behavior
                return _normalize_interp_override(v)

        return 1, "linear"

    base_dir = os.path.dirname(os.path.abspath(transform_record_path))

    # Load the record
    with open(transform_record_path, "r") as f:
        record = json.load(f)

    steps = list(record.items())
    if mode == "reverse":
        steps = list(reversed(steps))

    # If user overrides, apply to ALL steps. If None, infer per-step.
    user_resize_order, user_sitk_interp = _normalize_interp_override(interp)

    temp_file = input_path
    temp_files = []

    for step_name, record_entry in steps:
        if isinstance(record_entry, dict):
            tfm_path = os.path.normpath(os.path.join(base_dir, record_entry["transform"]))
            if mode == "apply":
                ref_path = os.path.normpath(os.path.join(base_dir, record_entry.get("fixed_reference", "")))
            else:
                ref_path = os.path.normpath(os.path.join(base_dir, record_entry.get("moving_reference", "")))

            # Determine interpolation for this step
            if user_resize_order is None:
                step_resize_order, step_sitk_interp = _infer_interp_from_record(record_entry)
            else:
                step_resize_order, step_sitk_interp = user_resize_order, user_sitk_interp
        else:
            tfm_path = os.path.normpath(os.path.join(base_dir, record_entry))
            ref_path = None

            # Non-dict entries: no metadata available
            step_resize_order, step_sitk_interp = (user_resize_order, user_sitk_interp) if user_resize_order is not None else (1, "linear")

        if tfm_path.endswith(".tfm"):
            intermediate = tempfile.mktemp(suffix=".nii.gz")
            if not ref_path or not os.path.exists(ref_path):
                raise RuntimeError(f"[Error] Reference image not found for transform: {tfm_path}")

            if mode == "apply":
                # NOTE: your current register_images() apply-only path uses its own interpolator internally.
                # If you want stored interpolation to affect APPLY direction too, we should add an
                # interpolation param to register_images() and pass it through to the resampler.
                register_images(
                    fixed_path=ref_path,
                    moving_path=temp_file,
                    output_path=intermediate,
                    transform_path=tfm_path,
                    apply_only=True,
                    verbose=False,
                )
            else:
                inverse_transform_image(
                    original_image_path=ref_path,
                    transformed_image_path=temp_file,
                    transform_path=tfm_path,
                    output_path=intermediate,
                    interpolation=step_sitk_interp,  # <-- was hardcoded "linear"
                    verbose=False,
                )

            temp_files.append(intermediate)
            temp_file = intermediate

        elif tfm_path.endswith("_padding.txt"):
            padding_record = read_padding_record(tfm_path)
            intermediate = tempfile.mktemp(suffix=".nii.gz")

            if mode == "apply":
                resize_mri(
                    input_filepath=temp_file,
                    output_filepath=intermediate,
                    target_shape=padding_record["target_shape"],
                    target_voxel_dims=padding_record["target_voxel_dims"],
                    interp=step_resize_order,  # <-- inferred unless user overrides
                    save_padding_record=False,
                    padding_record_path=tfm_path,
                    translation_only=False,
                )
            else:
                reverse_resize_mri(
                    input_filepath=temp_file,
                    output_filepath=intermediate,
                    padding_record_path=tfm_path,
                    interp=step_resize_order,  # <-- inferred unless user overrides
                )

            temp_files.append(intermediate)
            temp_file = intermediate

        else:
            raise ValueError(f"Unsupported transform step: {tfm_path}")

    shutil.copy(temp_file, output_path)
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)

    print(f"[Info] {'Applied' if mode == 'apply' else 'Reversed'} transforms to: {output_path}")

# -------------------------------------------------------------------------------------------
# Function to report inferred scan types of DICOM files
# -------------------------------------------------------------------------------------------

def summarize_exam_series(dicom_exam_dir, mr_subdir="MR", to_csv=None, verbose=False):
    """
    Run one-pass metadata extraction and scan-type classification on a single exam directory.

    Parameters
    ----------
    dicom_exam_dir : str
        Path to the exam directory containing the MR/ subfolder of series.
    mr_subdir : str
        Name of the MR subdirectory (default 'MR').
    to_csv : str or None
        If provided, write the resulting table to this CSV path.
    verbose : bool
        Print a preview to stdout.

    Returns
    -------
    pandas.DataFrame
    """
    from .preprocessing_utils import classify_exam_series

    df = classify_exam_series(dicom_exam_dir, mr_subdir=mr_subdir, verbose=verbose)
    if to_csv:
        # create parent dir if needed
        import os
        os.makedirs(os.path.dirname(os.path.abspath(to_csv)), exist_ok=True)
        df.to_csv(to_csv, index=False)
    if verbose:
        cols = ["series_number","acq_dt_iso","final_label","base_type",
                "is_postcontrast","is_flair","is_derived","plane",
                "matrix","voxel_mm","n_slices","mr_acq_type",
                "b_value","pulse_sequence_name","is_fspgr","series_description","confidence"]
        print(df[cols].to_string(index=False))
        for _, row in df.iterrows():
            print(f"  - reasons[{row.series_number}]: {row.reason}")
    return df

# ------------------------------------------------------------------------
# create_patient_metadata
# ------------------------------------------------------------------------
def create_patient_metadata(root_dir, out_path, previous_paths=None, omit_previous=False, subdirs=None, exclude_empty=False, show_progress=True, n_workers=None):
    """
    Build a per-patient metadata table by scanning {root_dir}/{Patient_folder}/.../MR/{series}.
    Columns:
      - Directory (relative to root_dir)
      - patientID (user-assigned; blank unless prefilled from previous tables)
      - patientName (lowercased unique names from DICOM)
      - dicomPatientID (lowercased unique IDs from DICOM)
      - day0Date (blank unless prefilled from previous tables)
    Args:
        n_workers (int | None): If >1, use a thread pool with this many workers to scan
            patient folders concurrently. If None, choose a sensible default for I/O.
            Use 1 to disable threading.
    """
    import pandas as pd
    from .preprocessing_utils import (
        _first_dicom_in, _read_table, _save_table, _progress,
        _safe_dcmread, _get_attr, _normalize_patient_name, _clean_lower,
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import math

    if previous_paths is None:
        previous_paths = []
    # which subfolders under each patient folder to search
    subdirs = subdirs or ["MR"]
    subdir_set = {s.strip(os.sep) for s in subdirs if s}
    root_dir = os.path.abspath(root_dir)
    rows = []

    # Gather top-level patient folders
    patient_folders = sorted([d.path for d in os.scandir(root_dir) if d.is_dir()])

    def _scan_one_patient(pf: str):
        """Return a row dict for one patient folder, or None if skipped."""
        rel_dir = os.path.relpath(pf, root_dir)
        names_norm: set[str] = set()
        names_raw: set[str] = set()
        ids: set[str] = set()
        found_any_dicom = False

        try:
            # Walk to find any target subdir(s) (default 'MR'), then take immediate subfolders as series
            for walk_root, dirnames, _ in os.walk(pf):
                base = os.path.basename(walk_root)
                if base not in subdir_set:
                    continue
                series_dirs = sorted(
                    os.path.join(walk_root, d)
                    for d in dirnames
                    if os.path.isdir(os.path.join(walk_root, d))
                )
                for sdir in series_dirs:
                    dcm_path = _first_dicom_in(sdir)
                    if not dcm_path:
                        continue
                    ds = _safe_dcmread(dcm_path)
                    if ds is None:
                        continue
                    found_any_dicom = True
                    # normalize name: keep apostrophes, replace punctuation, collapse spaces, lowercase
                    pname_raw = _get_attr(ds, "PatientName")
                    pname = _normalize_patient_name(pname_raw)
                    pid = _clean_lower(_get_attr(ds, "PatientID"))
                    if pname:
                        names_norm.add(pname)
                    if pname_raw:
                        names_raw.add(pname_raw)
                    if pid:
                        ids.add(pid)
        except Exception:
            # Be robust to odd directories; skip with no crash
            return None

        if exclude_empty and not found_any_dicom:
            return None

        return dict(
            Directory=rel_dir,
            patientID="",
            patientName="; ".join(sorted(names_norm)) if names_norm else "",
            dicomPatientID="; ".join(sorted(ids)) if ids else "",
            day0Date="",
        )

    use_threads = (n_workers is None) or (isinstance(n_workers, int) and n_workers != 1)
    # Choose a sensible default if not provided; I/O-bound so more threads can help.
    if n_workers in (None, 0):
        try:
            import os as _os
            cpu = max(1, (_os.cpu_count() or 1))
            n_workers = min(32, cpu * 4)
        except Exception:
            n_workers = 8

    if use_threads and n_workers > 1:
        # Threaded path: submit each patient folder
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as _pool:
            futures = [_pool.submit(_scan_one_patient, pf) for pf in patient_folders]
            for fut in _progress(
                as_completed(futures),
                total=len(futures),
                desc="Scanning patients",
                unit="patient",
                enable=show_progress,
            ):
                try:
                    row = fut.result()
                    if row:
                        results.append(row)
                except Exception:
                    # Ignore a failed folder; continue
                    pass
        rows.extend(results)
    else:
        # Sequential path (original behavior)
        for pf in _progress(
            patient_folders,
            total=len(patient_folders),
            desc="Scanning patients",
            unit="patient",
            enable=show_progress,
        ):
            row = _scan_one_patient(pf)
            if row:
                rows.append(row)

    df = pd.DataFrame(rows, columns=["Directory", "patientID", "patientName", "dicomPatientID", "day0Date"])
    df["Directory"] = df["Directory"].astype(str)  # ensure merge key is string

    # Apply previous metadata rules
    prev_tables = [_read_table(p) for p in (previous_paths or []) if p]
    if prev_tables:
        # Union of directories present in any previous table
        prev_dirs_union = set()
        for prev in prev_tables:
            if "Directory" in prev.columns:
                prev_dirs_union.update(prev["Directory"].astype(str).tolist())

        if omit_previous:
            df = df[~df["Directory"].isin(prev_dirs_union)].reset_index(drop=True)
        else:
            # Prefill patientID & day0Date from the *first* previous table that matches
            lookup_list = []
            for prev in prev_tables:
                if "Directory" not in prev.columns:
                    continue
                sub = prev.copy()
                for col in ("patientID", "day0Date"):
                    if col not in sub.columns:
                        sub[col] = ""
                sub = sub[["Directory", "patientID", "day0Date"]]
                sub = sub[["Directory", "patientID", "day0Date"]]
                sub["Directory"] = sub["Directory"].astype(str)  # normalize dtype before indexing
                lookup_list.append(sub.set_index("Directory"))
                # Combined (ordered) lookup — first non-missing wins
                combined = pd.concat(lookup_list, axis=1, join="outer", keys=range(len(lookup_list)))
                # Flatten to first non-empty per column
                for col in ("patientID", "day0Date"):
                    cols = [c for c in combined.columns if c[1] == col]
                    combined[(0, f"__first_nonempty_{col}")] = combined[cols].bfill(axis=1).iloc[:, 0]
                flat = combined[[ (0, "__first_nonempty_patientID"), (0, "__first_nonempty_day0Date") ]]
                flat.columns = ["_prefill_patientID", "_prefill_day0Date"]
                flat = flat.reset_index()  # Directory becomes a column
                flat["Directory"] = flat["Directory"].astype(str)  # match df key dtype
                df = df.merge(flat, on="Directory", how="left")
                df["patientID"] = df["patientID"].mask(df["patientID"].eq("") & df["_prefill_patientID"].notna(), df["_prefill_patientID"].fillna(""))
                df["day0Date"] = df["day0Date"].mask(df["day0Date"].eq("") & df["_prefill_day0Date"].notna(), df["_prefill_day0Date"].fillna(""))
                df = df.drop(columns=["_prefill_patientID", "_prefill_day0Date"])

    df = df.sort_values(by='patientName')
    _save_table(df, out_path)
    return df

# ------------------------------------------------------------------------
# demix_dicoms
# ------------------------------------------------------------------------
def demix_dicoms(
    root_dir: str,
    show_progress: bool = True,
    log_out: str | None = None,
    out_dir: str | None = None,
    dry_run: bool = False,
    n_workers: int | None = None,
    in_place: bool = False,
) -> None:
    """
    Ensure each leaf DICOM series folder contains files from only ONE scan.

    Strategy
    --------
    1) Walk every subdirectory of {root_dir} and identify *leaf series* folders
       (folders that directly contain .dcm files).
    2) Group found leaves by their immediate parent (typically the MR folder):
         { .../PatientX/TimepointY/MR } -> [leaf1, leaf2, ...]
    3) For each MR parent, read metadata for *all* DICOM files across *all* its
       series subfolders, and group files by a robust scan key:
         - Prefer (SeriesInstanceUID)
         - Fallback to (SeriesNumber, SeriesDescription, ProtocolName)
    4) Move files so that each resulting subfolder under the MR parent contains
       exactly one scan. New subfolder names are synthesized as:
         "{SeriesNumber}_{sanitized-SeriesDescription}_{uid6}" (uid6 short hash)
       Collisions are avoided by appending a numeric suffix.
    5) Show a progress bar and print a short summary.

    Options
    -------
    out_dir : if provided, create a fully de-mixed COPY of {root_dir} under {out_dir}.
              (Original tree left unchanged; files are COPIED, not moved.)
              The demix log will still ONLY list files that were “misplaced”
              (i.e., whose target subfolder name differs from their original leaf name).
    dry_run : if True, do NOT move/copy anything; still compute and write the demix log
              of what WOULD change (same “misplaced only” rule).
    in_place : if True, allow moving files within {root_dir}. If False, you MUST
               provide out_dir. This guard prevents accidental in-place reshuffles.
    """
    # --- safety guard: require either out_dir or explicit in_place=True
    if out_dir is None and not in_place:
        raise ValueError(
            "[demix_dicoms] Refusing to run without an explicit destination.\n"
            "You must either specify an output directory (out_dir=...)\n"
            "OR run explicitly in place (in_place=True)."
        )
    root_dir = os.path.abspath(root_dir)

    # All utilities imported on demand
    from .preprocessing_utils import (
        _first_dicom_in,
        _safe_dcmread,
        _get_attr,
        _read_table,
        _save_table,
        _progress,
        _normalize_patient_name,
        _safe_int_like,
        _propose_series_dirname,
        _avoid_name_collision,
        _files_identical,
    )

    # Prepare move log (we record only “misplaced” files; in dry_run these are planned moves)
    moved_rows: list[dict] = []
    if (out_dir and not dry_run) or (out_dir and not log_out):
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    if not dry_run:
        default_log = (os.path.join(out_dir or root_dir, f"demix_log_{datetime.now():%Y%m%d_%H%M%S}.csv")
                       if log_out is None else log_out)
    else:
        default_log = (os.path.join(out_dir or root_dir, f"dryrun_demix_log_{datetime.now():%Y%m%d_%H%M%S}.csv")
                           if log_out is None else log_out)

    _log_path = default_log
    _ext = os.path.splitext(_log_path)[1].lower()
    if _ext not in (".csv", ".tsv"):
        # Fallback to csv if user passed an unknown extension
        _log_path = os.path.splitext(_log_path)[0] + ".csv"
        print(f"[demix_dicoms][note] {_ext} not supported for demixing logs; writing .csv instead: {_log_path}")
        _ext = ".csv"
    _delim = "\t" if _ext == ".tsv" else ","
    os.makedirs(os.path.dirname(_log_path) or ".", exist_ok=True)
    _log_fh = open(_log_path, "a", newline="", encoding="utf-8")
    _log_writer = _csv.writer(_log_fh, delimiter=_delim)
    # Write header if file is empty
    try:
        if _log_fh.tell() == 0:
            _log_writer.writerow(["mr_parent_rel","src","dst","series_uid","series_number","series_description","protocol_name"])
            _log_fh.flush()
    except Exception:
        pass

    # Thread-safe logging
    _logged_rows = 0
    import threading as _threading
    _log_lock = _threading.Lock()
    def _stream_log_row(mr_rel: str, src: str, dst: str, s_uid, s_no, s_desc, s_proto):
        nonlocal _logged_rows
        with _log_lock:
            _log_writer.writerow([mr_rel, src, dst, s_uid or "", (s_no if s_no is not None else ""), s_desc or "", s_proto or ""])
            _log_fh.flush()
            _logged_rows += 1

    # ------------- pass A: find leaf series folders and their MR parent -------------
    mr_parent_to_leaves: dict[str, list[str]] = {}

    for curr, _dirnames, filenames in os.walk(root_dir):
        # Identify "leaf" series by presence of .dcm directly in this folder
        if any(f.lower().endswith(".dcm") for f in filenames):
            parent = os.path.dirname(curr)
            mr_parent_to_leaves.setdefault(parent, []).append(curr)

    if not mr_parent_to_leaves:
        print("[demix_dicoms] No DICOM series folders found under:", root_dir)
        return

    # ------------- pass B: for each MR parent, index all DICOM files & demix -------------
    parents = sorted(mr_parent_to_leaves.keys())
    total_moves = 0
    # Choose a default worker count tuned for I/O bound work (DICOM header reads + file copies)
    if n_workers is None:
        try:
            import os as _os
            _cpu = (_os.cpu_count() or 8)
            n_workers = min(32, _cpu * 4)
        except Exception:
            n_workers = 8
    else:
        # Coerce to int if callers passed a string (e.g., via CLI)
        try:
            n_workers = int(n_workers)
        except Exception:
            n_workers = 8

    # progress over MR parents
    for mr_parent in _progress(parents, total=len(parents), desc="Demixing MR folders", unit="MR", enable=show_progress):
        leaves = sorted(mr_parent_to_leaves[mr_parent])
        if not leaves:
            continue
        mr_rel = os.path.relpath(mr_parent, root_dir)
        # Collect all DICOM files across all leaf series under this MR parent
        all_dicoms: list[str] = []
        for leaf in leaves:
            try:
                for fn in os.listdir(leaf):
                    if fn.lower().endswith(".dcm"):
                        all_dicoms.append(os.path.join(leaf, fn))
            except Exception:
                continue

        if not all_dicoms:
            continue

        # Build grouping key for each DICOM (THREADED)
        # each entry: (path, group_key, series_number, series_desc, proto_name, series_uid)
        def _read_one(_p: str):
            ds = _safe_dcmread(_p)
            if ds is None:
                return (_p, ("__UNKNOWN__", None, None, None), None, None, None, None)
            series_uid  = _get_attr(ds, "SeriesInstanceUID") or None
            series_no   = _safe_int_like(_get_attr(ds, "SeriesNumber"))
            series_desc = _get_attr(ds, "SeriesDescription") or ""
            proto_name  = _get_attr(ds, "ProtocolName") or ""
            if series_uid:
                key = ("UID", series_uid)
            else:
                key = ("FALLBACK",
                       series_no if series_no is not None else -1,
                       series_desc.strip().lower(),
                       proto_name.strip().lower())
            return (_p, key, series_no, series_desc, proto_name, series_uid)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        entries = []
        with ThreadPoolExecutor(max_workers=n_workers) as _pool:
            futures = [_pool.submit(_read_one, p) for p in all_dicoms]
            for fut in as_completed(futures):
                try:
                    entries.append(fut.result())
                except Exception as e:
                    # fall back: keep the file in a coarse group to avoid dropping it
                    # (won't happen often, but keeps logic robust)
                    pass

        # Nothing to do if everything already lives in one group per leaf
        # (We still compute target paths below to be thorough across all leaves)
        # Build grouping -> files
        from collections import defaultdict
        groups: dict[tuple, list[tuple]] = defaultdict(list)
        for rec in entries:
            groups[rec[1]].append(rec)

        # ---- Build per-leaf group counts to detect purity/idempotence
        from collections import defaultdict, Counter
        leaf_group_counts = defaultdict(Counter)  # leaf -> Counter(group_key -> count)
        for p, key, *_rest in entries:
            leaf = os.path.dirname(p)
            leaf_group_counts[leaf][key] += 1

        # Idempotence early-exit:
        # If every leaf contains exactly one group, and each group appears in exactly one leaf,
        # then this MR parent is already fully demixed → skip entirely.
        all_pure = all(len(cnts) == 1 for cnts in leaf_group_counts.values())
        if all_pure:
            group_to_unique_leaf = {}
            unique = True
            for leaf, cnts in leaf_group_counts.items():
                g = next(iter(cnts.keys()))
                if g in group_to_unique_leaf:
                    unique = False; break
                group_to_unique_leaf[g] = leaf
            if unique and len(group_to_unique_leaf) == len(groups):
                # Already demixed:
                # - If writing to out_dir: we still need to copy to an identical structure under out_dir.
                # - If in-place: skip this MR parent entirely.
                if out_dir is None:
                    continue
                # Fall through to build group_to_target under out_dir with same leaf basenames.

        # ---- Unique assignment of groups -> existing leaves (one group per leaf, one leaf per group)
        # Totals per group to detect "pure & complete" leaves
        group_totals = {g: len(recs) for g, recs in groups.items()}
        assigned_group_to_leaf: dict[tuple, str] = {}
        used_leaves: set[str] = set()

        # (A) Assign pure leaves that fully contain a single group (exact coverage)
        for leaf, cnts in leaf_group_counts.items():
            if len(cnts) == 1:
                g, c = next(iter(cnts.items()))
                if c == group_totals.get(g, 0):
                    assigned_group_to_leaf[g] = leaf
                    used_leaves.add(leaf)

        # (B) Greedy assign remaining leaves to their best (max count) remaining group
        triples = []  # (count, leaf, group)
        for leaf, cnts in leaf_group_counts.items():
            if leaf in used_leaves:
                continue
            for g, c in cnts.items():
                if g in assigned_group_to_leaf:
                    continue
                if c > 0:
                    triples.append((c, leaf, g))
        # Sort: highest count first; tie-breaker: deterministic by leaf then group
        triples.sort(key=lambda x: (-x[0], x[1], repr(x[2])))
        for c, leaf, g in triples:
            if leaf in used_leaves or g in assigned_group_to_leaf:
                continue
            assigned_group_to_leaf[g] = leaf
            used_leaves.add(leaf)

        # ---- Compute final targets: reuse assigned existing leaves; create new folders otherwise
        group_to_target: dict[tuple, str] = {}
        used_basenames: set[str] = set(os.listdir(mr_parent)) if os.path.isdir(mr_parent) else set()
        # Determine the parent under which targets will be created
        target_parent = mr_parent if (out_dir is None) else os.path.join(out_dir, mr_rel)
        if not dry_run:
            os.makedirs(target_parent, exist_ok=True)
        used_basenames: set[str] = set(os.listdir(target_parent)) if os.path.isdir(target_parent) else set()

        for key, recs in groups.items():
            # representative metadata for (potential) new-folder naming
            _, _, s_no, s_desc, _s_proto, s_uid = recs[0]

            if key in assigned_group_to_leaf:
                # Reuse existing leaf name; for out_dir we mirror its basename under target_parent
                existing_leaf = assigned_group_to_leaf[key]
                base = os.path.basename(existing_leaf)
                target = os.path.join(target_parent, base)
                used_basenames.add(base)
            else:
                # No suitable existing leaf -> make a new, stable folder
                base = _propose_series_dirname(s_no, s_desc, s_uid)
                target = os.path.join(target_parent, base)
                # Collision-avoidance only for NEW folders
                if base in used_basenames or os.path.exists(target):
                    k = 2
                    while True:
                        alt = f"{base}-{k}"
                        # IMPORTANT: create under target_parent (respects --outDir)
                        target2 = os.path.join(target_parent, alt)
                        if alt not in used_basenames and not os.path.exists(target2):
                            target = target2
                            base = alt
                            break
                        k += 1
                used_basenames.add(base)
            group_to_target[key] = target

        # Create targets (idempotent)
        if not dry_run:
            for target in group_to_target.values():
                os.makedirs(target, exist_ok=True)

        # Transfer files to their target (move in-place; copy when out_dir is set) — THREADED
        _moves_lock = _threading.Lock()
        def _xfer_one(rec):
            nonlocal total_moves
            p, key, s_no, s_desc, s_proto, s_uid = rec
            tgt_dir = group_to_target[key]
            curr_dir = os.path.dirname(p)
            changed_subdir = (os.path.basename(curr_dir) != os.path.basename(tgt_dir))
            # Already in correct dir (in-place mode): skip
            if out_dir is None and os.path.normpath(curr_dir) == os.path.normpath(tgt_dir):
                return
            dst = os.path.join(tgt_dir, os.path.basename(p))
            # If destination exists, verify identical or note conflict
            if os.path.exists(dst):
                if _files_identical(p, dst):
                    print(f"[demix_dicoms][info] identical exists, skipping: {p} == {dst}")
                    if changed_subdir and dry_run:
                        _stream_log_row(mr_rel, p, dst, s_uid, s_no, s_desc, s_proto)
                    return
                else:
                    print(f"[demix_dicoms][WARN] conflict: destination exists with different content, skipping move: {p} -> {dst}")
                    if changed_subdir and dry_run:
                        _stream_log_row(mr_rel, p, dst, s_uid, s_no, s_desc, s_proto)
                    return
            # Log “misplaced” files (different subfolder name than target)
            if changed_subdir:
                _stream_log_row(mr_rel, p, dst, s_uid, s_no, s_desc, s_proto)
            if dry_run:
                return
            try:
                if out_dir is None:
                    os.replace(p, dst)
                else:
                    import shutil
                    shutil.copy2(p, dst)
                with _moves_lock:
                    total_moves += 1
            except Exception as e:
                print(f"[demix_dicoms][warn] failed to transfer {p} -> {dst}: {e}")

        with ThreadPoolExecutor(max_workers=n_workers) as _pool2:
            futures = [_pool2.submit(_xfer_one, rec) for rec in entries]
            # wait for completion
            for _ in as_completed(futures):
                pass

        # Clean up any empty series folders left behind
        if out_dir is None and not dry_run:
            for leaf in leaves:
                try:
                    if os.path.isdir(leaf) and not os.listdir(leaf):
                        os.rmdir(leaf)
                except Exception:
                    pass

    # Close streaming log and summarize
    try:
        _log_fh.close()
    except Exception:
        pass
    action = "Planned" if dry_run else ("Copied" if out_dir else "Moved")
    if _logged_rows > 0:
        print(f"[demix_dicoms] Completed. Files {action.lower()}: {total_moves}. Log: {_log_path}")
    else:
        print(f"[demix_dicoms] Completed. No files to {'copy' if out_dir else 'move'}{', dry run only' if dry_run else ''}.")


# -------------------------------------------------------------
# Function to plan conversion of DICOM library to NIFTI library
# -------------------------------------------------------------

def plan_dicom_to_nifti_conversion(
    patient_metadata: str,
    root_dir: str,
    out_dir: str,
    n_workers: int | None = None,
    plan_out: str | None = None,
    show_progress: bool = True,
    previous_plans: list[str] | None = None,
    ignore_previous: bool = False,
    include_mr_subdirs: list[str] | None = None,
    min_slices: int = 10,
    use_actual_exam_ids: bool = False,
    add_missing_derived: bool = False,
    make_derived_from_scratch: bool = False,
    unexpected_multiframe_policy: str = "keep_first",
):
    """
    Plan DICOM->NIfTI conversion using existing series classification.

    Inputs:
      - patient_metadata: table from create_patient_metadata() (manually filled).
        Must include Directory, patientID, day0Date. Rows missing any are skipped.
      - root_dir: the DICOM library root; patient subfolders match 'Directory'.
      - out_dir: the target NIfTI root.
      - n_workers: thread count for I/O-bound work (default: min(32, cpu*4)).
      - plan_out: path to write the plan (.csv/.tsv streamed; .xlsx at end).
      - show_progress: show progress bars if tqdm is available.
      - previous_plans: 0+ prior plan files; if provided we will either reuse
        matching ExamDirectory rows from them (default) or skip them when
        ignore_previous=True.
      - ignore_previous: when True, discovered exams present in previous_plans
        are skipped entirely; when False, rows for those exams are copied from
        the previous plan(s) into the new plan without reprocessing.
      - include_mr_subdirs: list[str] | None
        If provided, only exams whose 'mr_subdir_name' (case-insensitive) is in this list
        will be planned.
      - min_slices: int
        Minimum number of slices a sequence must have to be considered for selection.
        (Rows remain in the plan; the threshold only affects 'selected_for_conversion'.)
      - use_actual_exam_ids: boolean to indicate if actual exam IDs should be used instead of unique random ones.
      - add_missing_derived: boolean to indicate if function should dentify derived scan types missing for each primary in an exam and add DERIVE jobs to the plan.
      - make_derived_from_scratch: boolean to indicate if function should ignore existing derived scans and plan DERIVE jobs for all supported derived types from primaries.
      - unexpected_multiframe_policy: str
        Policy for series that are expected to be single-frame/3D but convert to multi-frame/4D.
        One of: 'keep_first' (default; keep frame 0 with a warning) or 'skip' (skip conversion with a warning).

    Returns:
      pandas.DataFrame with one row per discovered series, including:
        Directory (patient folder), ExamDirectory (study folder), ExamAlias (8-char A–Z/0–9, unique),
        patientID, day0Date, folder, series_number, acq_dt/acq_dt_iso, final_label/base_type,
        plane, n_slices, confidence, timepoint_days, selected_for_conversion (bool),
        proposed_nifti_path. Rows are grouped by ExamDirectory (blank line "-" between exams).
    """
    import os, re, csv as _csv
    from datetime import datetime, date
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .preprocessing_utils import (
        classify_exam_series,           # returns labeled per-series rows
        _read_table, _save_table,       # robust I/O helpers
        _progress,                      # optional tqdm wrapper
        enumerate_supported_derivatives,
        choose_primary_for_derivation,
        build_derived_output_name,
        _filter_derivatives_by_policy,
        _sanitize_label
    )

    # ---------- helpers ----------
    def _parse_day0(d) -> date | None:
        if pd.isna(d): return None
        s = str(d).strip()
        if not s: return None
        try: return pd.to_datetime(s, errors="coerce").date()
        except Exception: return None

    def _safe_int(x, default=None):
        try: return int(x)
        except Exception: return default

    def _discover_exams(patient_abs: str) -> list[tuple[str, str | None]]:
        """Return de-duplicated (exam_dir_abs, mr_subdir_name) for supported layouts.

        Supported layouts under ``patient_abs``:
          1) {patient_abs}/{Exam}/{RadiologyType}/{Series}/.../*.dcm
          2) {patient_abs}/{Exam}/{Series}/.../*.dcm

        For layout (1), ``mr_subdir_name`` is the radiology-type folder (for example
        ``MR``). For layout (2), ``mr_subdir_name`` is ``None`` so all series directly
       beneath the exam are grouped into a single ExamAlias.

        Notes
        -----
        We intentionally detect layout from the *immediate* children of each exam
        directory rather than inferring it from every DICOM leaf path. This prevents
        a direct-series layout like ``Exam/T1n/DICOM/*.dcm`` from being misread as
        ``Exam/{RadiologyType}/{Series}``, which would incorrectly split each series
        into its own ExamAlias.
        """
        patient_abs = os.path.normpath(os.path.abspath(patient_abs))

        def _subtree_has_dicom(folder: str, max_depth: int = 6) -> bool:
            """Return True if ``folder`` or a shallow descendant contains any .dcm file."""
            try:
                folder = os.path.normpath(folder)
                base_depth = folder.count(os.sep)
                for curr, dirs, files in os.walk(folder):
                    if any(f.lower().endswith('.dcm') for f in files):
                        return True
                    depth = curr.count(os.sep) - base_depth
                    if depth >= max_depth:
                        dirs[:] = []
                return False
            except Exception:
                return False

        def _count_series_like_children(folder: str) -> int:
            """Count immediate child directories whose subtree contains DICOM files."""
            try:
                names = sorted(os.listdir(folder))
            except Exception:
                return 0
            n = 0
            for name in names:
                child = os.path.join(folder, name)
                if os.path.isdir(child) and _subtree_has_dicom(child):
                    n += 1
            return n

        # Common radiology/modality container names seen in DICOM exports.
        known_radiology_names = {
            'mr', 'mri', 'ct', 'pt', 'pet', 'nm', 'us', 'xa', 'mg', 'cr', 'dx'
        }

        seen: set[tuple[str, str | None]] = set()
        try:
            exam_names = sorted(os.listdir(patient_abs))
        except Exception:
            return []

        for exam_name in exam_names:
            exam_dir = os.path.normpath(os.path.join(patient_abs, exam_name))
            if not os.path.isdir(exam_dir):
                continue
            try:
                child_names = sorted(os.listdir(exam_dir))
            except Exception:
                continue

            radiology_children: list[str] = []
            has_direct_series = False

            for child_name in child_names:
                child_dir = os.path.join(exam_dir, child_name)
                if not os.path.isdir(child_dir):
                    continue
                if not _subtree_has_dicom(child_dir):
                    continue

                n_series_children = _count_series_like_children(child_dir)
                child_key = str(child_name).strip().lower()

                # Treat folders like MR/CT/PET as explicit radiology containers even if
                # they currently hold only one series. Also treat folders with multiple
                # series-like children as radiology containers.
                if child_key in known_radiology_names or n_series_children >= 2:
                    radiology_children.append(child_name)
                else:
                    # Use the direct-series path when the exam contains series folders
                    # directly (for example Exam/T1n/*.dcm or Exam/T1n/DICOM/*.dcm).
                    has_direct_series = True

            if radiology_children:
                for mr_name in sorted(set(radiology_children)):
                    seen.add((exam_dir, mr_name))
            elif has_direct_series:
                seen.add((exam_dir, None))

        return sorted(seen)

    # Classifier columns (stable) + our metadata columns.
    CLASSIFY_CORE = [
        "folder","series_number","acq_dt","acq_dt_iso","manufacturer","modality",
        "series_description","protocol_name","sequence_name","image_type",
        "te","tr","ti","flip_angle","b_value","primary_secondary","is_fspgr",
        "base_type","final_label","is_postcontrast","is_flair","reason","confidence",
        "plane","matrix","voxel_mm","n_slices","mr_acq_type","pulse_sequence_name"
    ]

    # Put these side-by-side in the plan for easy viewing:
    _VIEW_CLUSTER = [
        "ExamDirectory","ExamAlias","patientID","timepoint_days","series_identifier",
        "final_label","plane","is_derived","matrix","voxel_mm","n_slices",
        "selected_for_conversion","proposed_nifti_path",
        "unexpected_multiframe_policy",
        # show derivation info when present
        "Action","GeneratorKey","PrimaryLabel","DerivedLabel",
        "PrimarySeriesIdentifier","PrimarySeriesPath",
    ]

    # Build the header: Directory first, then the cluster, then the rest (no dups)
    _PREFIX = ["Directory"]
    _SUFFIX = ["day0Date","mr_subdir_name"]  # keep but not inside the cluster

    _all = _PREFIX + _VIEW_CLUSTER + _SUFFIX + CLASSIFY_CORE
    seen = set()
    HEADER = [c for c in _all if not (c in seen or seen.add(c))]

    # ---------- setup ----------
    root_dir = os.path.abspath(root_dir)
    out_dir  = os.path.abspath(out_dir)

    # threads tuned for I/O
    if n_workers is None:
        try:
            n_workers = min(32, (os.cpu_count() or 8) * 4)
        except Exception:
            n_workers = 8
    else:
        n_workers = _safe_int(n_workers, default=8)

    meta = _read_table(patient_metadata)
    need_cols = {"Directory","patientID","day0Date"}
    miss = need_cols - set(map(str, meta.columns))
    if miss:
        raise ValueError(f"patient_metadata is missing required columns: {sorted(miss)}")

    # normalize and filter meta rows
    meta = meta.copy()
    meta["Directory"] = meta["Directory"].astype(str)
    meta["patientID"] = meta["patientID"].astype(str)
    meta["_day0"]     = meta["day0Date"].map(_parse_day0)
    meta = meta[(meta["Directory"].str.strip()!="") & (meta["patientID"].str.strip()!="") & meta["_day0"].notna()].reset_index(drop=True)

    # --- sanity checks vs disk and metadata (warnings) ---
    try:
        disk_dirs = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
    except Exception:
        disk_dirs = []
    meta_dirs = sorted({s for s in meta["Directory"].astype(str).str.strip().tolist() if s})

    # (a) Patients on disk that are NOT in the provided metadata sheet
    missing_in_meta = [d for d in disk_dirs if d not in meta_dirs]
    if missing_in_meta:
        examples = ", ".join(missing_in_meta[:10])
        print(
            f"[plan_dicom_to_nifti_conversion][WARN] {len(missing_in_meta)} patient "
            f"directories under root_dir have no row in patient_metadata 'Directory' "
            f"(e.g., {examples}). These patients will be ignored."
        )

    # (b) Metadata rows whose Directory does NOT exist under root_dir
    missing_on_disk = [d for d in meta_dirs if not os.path.isdir(os.path.join(root_dir, d))]
    if missing_on_disk:
        examples = ", ".join(missing_on_disk[:10])
        print(
            f"[plan_dicom_to_nifti_conversion][WARN] {len(missing_on_disk)} 'Directory' "
            f"values in patient_metadata do not exist under root_dir (e.g., {examples}). "
            f"These rows will be skipped."
        )

    # Decide how to treat derivatives in this planning run
    # "make" = plan all supported derivatives de-novo (ignore existing derived);
    # "add"  = only add truly missing derived for which a primary exists;
    # "none" = do not plan any new derived rows.
    plan_mode = "make" if make_derived_from_scratch else ("add" if add_missing_derived else "none")

    # ---------- phase 1: discover all exam directories (progress over patients) ----------
    patient_rows = meta.to_dict(orient="records")
    discovered: list[dict] = []

    def _log(prefix: str, msg: str):
        """Cheap, contextual logger. Prefix is a short tag like 'DISCOVER', 'CLASSIFY', etc."""
        #Disabling print logging for now
        #print(f"[plan:{prefix}] {msg}")
        return None

    def _discover_one(row):
        patient_rel = row["Directory"]
        patient_abs = os.path.join(root_dir, patient_rel)
        if not os.path.isdir(patient_abs):
            return []
        out = []
        for exam_abs, mr_name in _discover_exams(patient_abs):
            exam_rel = os.path.relpath(exam_abs, root_dir)
            out.append({
                "Directory": patient_rel,
                "ExamDirectory": exam_rel,
                "patientID": row["patientID"],
                "day0Date": row["day0Date"],
                "_day0": row["_day0"],
                "mr_subdir_name": mr_name,
                "exam_abs": exam_abs,
            })
        return out

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_discover_one, r) for r in patient_rows]
            for fut in _progress(as_completed(futures), total=len(futures), desc="Discovering exams", unit="patient", enable=show_progress):
                try:
                    discovered.extend(fut.result())
                except Exception:
                    pass
    else:
        for r in _progress(patient_rows, total=len(patient_rows), desc="Discovering exams", unit="patient", enable=show_progress):
            discovered.extend(_discover_one(r))

    # Deduplicate exams
    key = lambda d: (d["ExamDirectory"], d["mr_subdir_name"])
    # keep first occurrence
    dedup = {}
    for rec in discovered:
        dedup.setdefault(key(rec), rec)
    exams = list(dedup.values())
    _log("DISCOVER", f"Found {len(exams)} exam(s) after de-duplication")


    # Optional MR-subdir filter (case-insensitive)
    if include_mr_subdirs:
        want = {s.lower().strip() for s in include_mr_subdirs if s}
        exams = [rec for rec in exams if str(rec.get("mr_subdir_name","")).lower().strip() in want]

    # ---------- previous plan handling ----------
    prev_tables = []
    prev_exam_keys = set()
    used_aliases = set()
    if previous_plans:
        for p in previous_plans:
            try:
                t = _read_table(p)
                if "ExamDirectory" in t.columns:
                    prev_tables.append(t)
                    prev_exam_keys.update((str(ed), str(ms) if "mr_subdir_name" in t.columns else None)
                                          for ed, ms in zip(t["ExamDirectory"].astype(str),
                                                            t.get("mr_subdir_name", pd.Series([None]*len(t))).astype(str)))
                if "ExamAlias" in t.columns:
                    used_aliases.update(
                        s for s in t["ExamAlias"].astype(str).fillna("")
                        if s.strip() not in ("", "-")
                    )
            except Exception:
                pass
        # convert to tuple set: (ExamDirectory, mr_subdir_name) — mr_subdir_name may be "None"
        norm_prev = set()
        for ed, ms in prev_exam_keys:
            norm_prev.add((ed, None if ms == "None" else ms))
        prev_exam_keys = norm_prev

    # Partition: to_skip / to_process
    exams_to_process = []
    exams_to_reuse   = []
    for rec in exams:
        k = (rec["ExamDirectory"], rec["mr_subdir_name"])
        if previous_plans and k in prev_exam_keys:
            if ignore_previous:
                continue
            else:
                exams_to_reuse.append(k)
                continue
        exams_to_process.append(rec)

    # Generate unique 8-char alphanumeric ExamAlias for exams we're processing now
    import secrets, string
    ALPHABET = string.ascii_uppercase + string.digits

    def _new_alias() -> str:
        while True:
            s = "".join(secrets.choice(ALPHABET) for _ in range(8))
            if s not in used_aliases:
                used_aliases.add(s)
                return s

    alias_map: dict[tuple[str, str | None], str] = {}
    for rec in exams_to_process:
        k = (rec["ExamDirectory"], rec["mr_subdir_name"])
        if use_actual_exam_ids:
            base = os.path.basename(rec["ExamDirectory"])  # terminal dir name
            alias = base
            i = 2
            while alias in used_aliases:
                alias = f"{base}_{i}"
                i += 1
            used_aliases.add(alias)
            alias_map[k] = alias
        else:
            alias_map[k] = _new_alias()

    # ---------- streaming writer (CSV/TSV only) ----------
    wrote_header = False
    plan_fh = None
    delim = ","
    ext = None
    def _open_stream(path):
        nonlocal plan_fh, delim, ext, wrote_header
        if path is None:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv",".tsv"):
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            plan_fh = open(path, "w", newline="", encoding="utf-8")
            delim = "\t" if ext == ".tsv" else ","
            writer = _csv.writer(plan_fh, delimiter=delim)
            writer.writerow(HEADER)
            plan_fh.flush()
            wrote_header = True

    def _stream_block(df: pd.DataFrame):
        if plan_fh is None:
            return
        # enforce header order; fill missing cols with ""
        df2 = df.reindex(columns=HEADER)
        df2.to_csv(plan_fh, index=False, header=False, sep=delim, lineterminator="\n")
        # one blank row (per exam)
        plan_fh.write(delim.join(["-"]*len(HEADER)) + "\n")
        plan_fh.flush()

    _open_stream(plan_out)

    # If reusing previous plans, stream those rows first (without recomputing)
    all_results = []
    if exams_to_reuse and prev_tables:
        prev_tables = [t for t in prev_tables if t is not None and not t.empty]
        prev_all = pd.concat(prev_tables, axis=0, ignore_index=True) if prev_tables else pd.DataFrame(columns=HEADER)
        # normalize string types to avoid dtype drift
        for col in ("Directory","ExamDirectory","patientID","day0Date","mr_subdir_name"):
            if col in prev_all.columns:
                prev_all[col] = prev_all[col].astype(str)
        # write per exam group (keep blank line per exam)
        for k in _progress(sorted(set(exams_to_reuse)), total=len(set(exams_to_reuse)),
                           desc="Copying from previous plans", unit="exam", enable=show_progress):
            ed, ms = k
            sub = prev_all[(prev_all.get("ExamDirectory","") == ed) &
                           ((prev_all.get("mr_subdir_name") == ms) | (~prev_all.columns.isin(["mr_subdir_name"]).any()))]
            if sub.empty:
                continue
            # enforce header & stream
            _stream_block(sub)
            all_results.append(sub)

    # ---------- phase 2: process exams (progress over exams) ----------
    def _process_exam(rec) -> pd.DataFrame:
        import pandas as _pd
        exam_abs = rec["exam_abs"]
        pid      = rec["patientID"]
        day0     = rec["_day0"]
        mr_name  = rec["mr_subdir_name"]
        exam_rel = rec["ExamDirectory"]
        def _elog(stage: str, msg: str):
            _log(stage, f"patient={pid} exam='{exam_rel}' mr='{mr_name}': {msg}")

        try:
            df = classify_exam_series(exam_abs, mr_subdir=mr_name, verbose=False)
        except Exception as e:
            _elog("CLASSIFY", f"ERROR classify_exam_series: {type(e).__name__}: {e}")
            return _pd.DataFrame([{
                "Directory": rec["Directory"], "ExamDirectory": exam_rel, "patientID": pid,
                "day0Date": rec["day0Date"], "mr_subdir_name": mr_name,
                "error": f"classify_exam_series failed: {e}",
            }])

        if df is None or df.empty:
            _elog("CLASSIFY", "No series returned (empty dataframe)")
            return _pd.DataFrame()

        df = df.copy()
        df["Directory"]       = rec["Directory"]
        df["ExamDirectory"]   = exam_rel
        df["ExamAlias"]       = alias_map.get((exam_rel, mr_name), None)
        df["unexpected_multiframe_policy"] = str(unexpected_multiframe_policy)
        # Locals used by the DERIVE block
        exam_alias = alias_map.get((exam_rel, mr_name), None)
        patient_rel = rec["Directory"]
        df["patientID"]       = pid
        df["day0Date"]        = rec["day0Date"]
        df["mr_subdir_name"]  = mr_name
        df["series_identifier"] = df["folder"].map(lambda p: os.path.basename(str(p)) if pd.notna(p) else "")
        # Quick classification summary
        try:
            n_total = len(df)
            n_derived = int(df.get("is_derived", _pd.Series([False]*n_total)).astype(bool).sum())
            top_labels = (
                df.get("final_label", _pd.Series([""]*n_total))
                  .astype(str).str.strip().replace("", "_NA_")
                  .value_counts().head(8).to_dict()
            )
            _elog("CLASSIFY", f"rows={n_total}, derived={n_derived}, top labels={top_labels}")
        except Exception:
            pass

        # timepoint (int days from acq_dt to day0)
        def _tp(acq):
            if pd.isna(acq): return None
            try:
                acqd = acq.date() if hasattr(acq, "date") else pd.to_datetime(acq).date()
                return int((acqd - day0).days)
            except Exception:
                return None
        df["timepoint_days"] = df["acq_dt"].map(_tp)

        # pick best per type within this exam
        def _select_best(group: _pd.DataFrame) -> int | None:
            g = group.copy()
            if "primary_secondary" in g.columns:
                pri = g["primary_secondary"].fillna("").str.upper().eq("PRIMARY")
                if pri.any(): g = g[pri]
            if "is_derived" in g.columns and (~g["is_derived"].astype(bool)).any():
                g = g[~g["is_derived"].astype(bool)]
            if "plane" in g.columns:
                ax = g["plane"].fillna("").str.upper().str.startswith("AX")
                if ax.any(): g = g[ax]
            g = g.assign(
                _ns=g.get("n_slices", _pd.Series(index=g.index, dtype="float")).fillna(-1).astype(float),
                _conf=g.get("confidence", _pd.Series(index=g.index, dtype="float")).fillna(0.0).astype(float),
                _acq=g.get("acq_dt", _pd.Series(index=g.index)).map(lambda x: _pd.to_datetime(x) if pd.notna(x) else pd.NaT),
            ).sort_values(by=["_conf","_ns","_acq"], ascending=[False, False, True])
            return None if g.empty else int(g.index[0])

        label_col = "final_label" if "final_label" in df.columns else ("base_type" if "base_type" in df.columns else None)

        # Eligibility masks affect selection ONLY (rows still appear in plan)
        # Never select Calibration / FieldMap (FMAP) for conversion
        excluded_labels = {"unknown", "unknown-derived", "localizer",
                           "calibration", "fieldmap", "fmap"}
        eligible = pd.Series(True, index=df.index)

        # Exclude by label
        if "final_label" in df.columns:
            eligible &= ~df["final_label"].fillna("").str.strip().str.lower().isin(excluded_labels)
        elif label_col:
            eligible &= ~df[label_col].fillna("").str.strip().str.lower().isin(excluded_labels)

        # Exclude by min slices (default 10)
        if "n_slices" in df.columns:
            eligible &= df["n_slices"].fillna(-1).astype(float) >= float(min_slices)

        df["selected_for_conversion"] = False
        try:
            _elog(
                "ELIGIBILITY",
                f"eligible={int(eligible.sum())}/{len(eligible)} "
                f"(min_slices={min_slices}, excluded_labels={sorted(excluded_labels)})"
            )
        except Exception:
            pass

        def _is_primary_4d_label(lbl: str) -> bool:
            base = str(lbl or "").split("(", 1)[0].strip().upper()
            return base in {"DWI", "PERFUSION"}

        def _primary_4d_source_score(r: _pd.Series, lbl: str) -> float:
            """Heuristic score to prefer the *acquisition container* for primary 4D families.

            This is intentionally label-aware and vendor-tolerant:
            it prefers multi-volume evidence / Siemens MOSAIC / ORIGINAL over derived maps.
            """
            base = str(lbl or "").split("(", 1)[0].strip().upper()

            # Pull a few text fields (robust to NaN / lists serialized as strings)
            def _s(x):
                return str(x or "").strip().upper()

            img_type = _s(r.get("image_type", ""))
            seq_name = _s(r.get("sequence_name", ""))
            ser_desc = _s(r.get("series_description", ""))
            prot     = _s(r.get("protocol_name", ""))
            tokens   = " ".join([img_type, seq_name, ser_desc, prot])

            score = 0.0

            # Strong positives
            if "MOSAIC" in img_type:
                score += 8.0
            if "ORIGINAL" in img_type:
                score += 4.0
            if _s(r.get("primary_secondary", "")) == "PRIMARY":
                score += 2.0
            # num_frames is populated for Enhanced multiframe (true 4D) when available
            try:
                nf = float(r.get("num_frames") or 0)
            except Exception:
                nf = 0.0
            if nf and nf > 1:
                score += 6.0

            # Strong negatives (derived maps should not win as the family "primary")
            if "DERIVED" in img_type:
                score -= 6.0
            if bool(r.get("is_derived", False)):
                score -= 2.0

            if base == "DWI":
                bad = ["ADC", "TRACE", "TRACEW", "FA", "MD", "RD", "AD", "EXP", "COLFA", "TENSOR", "MAP"]
            else:  # PERFUSION
                bad = [
                    "CBV", "CBF", "MTT", "TTP", "TMAX", "KTRANS", "KEP", "VP", "VE",
                    "AUC", "MEAN", "MAX", "PARAM", "MAP", "LEAK", "LEAKAGE"
                ]
            if any(b in tokens for b in bad):
                score -= 10.0

            # Mild positive: looks like a diffusion/perfusion acquisition string
            if base == "DWI" and any(k in tokens for k in ["DTI", "DWI", "DIFF", "EP_B", "EPI"]):
                score += 1.5
            if base == "PERFUSION" and any(k in tokens for k in ["PERF", "PWI", "DSC", "DCE", "ASL", "PCASL"]):
                score += 1.5

            return score

        def _select_best(group: _pd.DataFrame) -> int | None:
            g = group.copy()
            # keep only eligible rows within this label group
            g = g[eligible.loc[g.index]]
            if g.empty:
                return None

            # Label-aware selection for primary 4D families (DWI, Perfusion): prefer the acquisition container.
            try:
                grp_label = str(g.iloc[0].get(label_col, "") if label_col else "")
            except Exception:
                grp_label = ""

            if label_col and _is_primary_4d_label(grp_label):
                base = grp_label.split("(", 1)[0].strip().upper()

                # Prefer the base label (no parentheses) when present: e.g., DWI over DWI(ADC)
                no_paren = ~g[label_col].astype(str).str.contains(r"\(")
                if no_paren.any():
                    g = g[no_paren]

                # Add a source-likeness score to drive selection.
                g = g.assign(
                    _score=g.apply(lambda r: _primary_4d_source_score(r, r.get(label_col, "")), axis=1),
                    _nf=_pd.to_numeric(g.get("num_frames", _pd.Series(index=g.index)), errors="coerce").fillna(0).astype(float),
                    _ns=_pd.to_numeric(g.get("n_slices", _pd.Series(index=g.index)), errors="coerce").fillna(-1).astype(float),
                    _conf=_pd.to_numeric(g.get("confidence", _pd.Series(index=g.index)), errors="coerce").fillna(0.0).astype(float),
                    _acq=g.get("acq_dt", _pd.Series(index=g.index)).map(lambda x: _pd.to_datetime(x) if pd.notna(x) else pd.NaT),
                ).sort_values(by=["_score", "_nf", "_conf", "_ns", "_acq"], ascending=[False, False, False, False, True])

                return None if g.empty else int(g.index[0])

            # Generic selection (other families): prefer PRIMARY & not derived → AX plane → max confidence → max slices → earliest acq_dt
            if "primary_secondary" in g.columns:
                pri = g["primary_secondary"].fillna("").str.upper().eq("PRIMARY")
                if pri.any():
                    g = g[pri]
            if "is_derived" in g.columns and (~g["is_derived"].astype(bool)).any():
                g = g[~g["is_derived"].astype(bool)]
            if "plane" in g.columns:
                ax = g["plane"].fillna("").str.upper().str.startswith("AX")
                if ax.any():
                    g = g[ax]

            g = g.assign(
                _ns=g.get("n_slices", _pd.Series(index=g.index, dtype="float")).fillna(-1).astype(float),
                _conf=g.get("confidence", _pd.Series(index=g.index, dtype="float")).fillna(0.0).astype(float),
                _acq=g.get("acq_dt", _pd.Series(index=g.index)).map(lambda x: _pd.to_datetime(x) if pd.notna(x) else pd.NaT),
            ).sort_values(by=["_conf","_ns","_acq"], ascending=[False, False, True])

            return None if g.empty else int(g.index[0])


        if label_col:
            chosen = []
            for lbl, g in df.groupby(df[label_col]):
                idx = _select_best(g)
                if idx is not None:
                    chosen.append(idx)
            if chosen:
                df.loc[chosen, "selected_for_conversion"] = True
        try:
            n_sel = int(df["selected_for_conversion"].astype(bool).sum())
            _elog("SELECT", f"selected_for_conversion={n_sel}/{len(df)} "
                            f"unique_labels={len(df[label_col].astype(str).unique()) if label_col else 'NA'}")
            if n_sel == 0:
                # Show a hint of why nothing was picked
                _elog("SELECT", f"Sample of labels with max slices/conf: " +
                      str(df.groupby(df[label_col]).agg(
                          n_slices=("n_slices","max"),
                          confidence=("confidence","max")
                      ).head(6).to_dict()) )
        except Exception:
            pass

        # proposed nifti paths (only for selected)
        def _proposed_path(r):
            if not bool(r.get("selected_for_conversion", False)):
                return ""
            tp = r.get("timepoint_days", None)
            tp_str = f"d{tp}" if tp is not None else "dNA"
            lbl = _sanitize_label(r.get(label_col, "Unknown"))
            ea = r.get("ExamAlias", None)
            subdir = f"{pid}_{tp_str}_{ea}" if ea else f"{pid}_{tp_str}"
            fname  = f"{pid}_{tp_str}_{lbl}.nii.gz"
            return os.path.join(out_dir, pid, subdir, fname)

        df["proposed_nifti_path"] = df.apply(_proposed_path, axis=1)
        # Tag original rows explicitly as CONVERT for clarity downstream
        if "Action" not in df.columns:
            df["Action"] = "CONVERT"
        # If deriving from scratch, do not convert vendor-derived series
        if make_derived_from_scratch and "is_derived" in df.columns:
            df.loc[df["is_derived"].fillna(False).astype(bool), "selected_for_conversion"] = False

        # --- If requested, append DERIVE rows for derived outputs ---
        if add_missing_derived or make_derived_from_scratch:
            # Which labels already exist in this exam (case-insensitive)
            have_labels = set(str(x).strip().upper() for x in df.get("final_label", []))

            # Consider only PRIMARIES that were selected for conversion
            sel_mask = df.get("selected_for_conversion", False).astype(bool) & ~df.get("is_derived", False).astype(bool)
            sel_prim = df.loc[sel_mask].copy()
            try:
                _elog("DERIVE", f"primaries selected={len(sel_prim)}; have_labels={sorted(list(have_labels))[:8]}...")
                if not len(sel_prim):
                    _elog("DERIVE", "No primaries selected — skipping derivative planning for this exam")
            except Exception:
                pass

            def _norm_primary_label(lbl: str) -> str:
                s = str(lbl or "")
                return s.split("(", 1)[0].strip().upper()

            # Build primary map preferring true primaries (labels without parentheses)
            prim_by_label = {}
            if not sel_prim.empty:
                sel_prim["_lbl_norm"] = sel_prim[label_col].map(_norm_primary_label)
                for lbl, g in sel_prim.groupby("_lbl_norm"):
                    # Prefer rows whose display label has no parentheses (e.g., 'DWI', not 'DWI(TRACE)')
                    g_no_paren = g[~g[label_col].astype(str).str.contains(r"\(")]
                    idx = _select_best(g_no_paren if not g_no_paren.empty else g)
                    if idx is not None:
                        prim_by_label[lbl] = df.loc[idx]

            planned_labels: set[str] = set()   # ensure one per DerivedLabel per exam
            derived_rows: list[dict] = []

            for norm_lbl, prim_row in prim_by_label.items():
                try:
                    _elog("DERIVE", f"primary base='{norm_lbl}' chosen='{prim_row.get('final_label','')}'")
                except Exception:
                    pass
                # DWI family: force derivations to use the true DWI primary, never TRACE/AvDC
                if str(norm_lbl).upper() == "DWI":
                    raw_lbl = str(prim_row.get(label_col, "")).upper()
                    if "(" in raw_lbl or "TRACE" in raw_lbl or "AVDC" in raw_lbl or "MD" in raw_lbl:
                        _cands = sel_prim[
                            (sel_prim[label_col].map(_norm_primary_label) == "DWI") &
                            (~sel_prim[label_col].astype(str).str.contains(r"\("))
                        ]
                        idx_fix = _select_best(_cands) if not _cands.empty else None
                        if idx_fix is not None:
                            prim_row = df.loc[idx_fix]
                # SWI family: prefer vendor-composited SWI (no parentheses) when available
                if str(norm_lbl).upper() == "SWI":
                    raw_lbl = str(prim_row.get(label_col, "")).upper()
                    if "(" in raw_lbl:
                        _cands = sel_prim[
                            (sel_prim[label_col].map(_norm_primary_label) == "SWI") &
                            (~sel_prim[label_col].astype(str).str.contains(r"\("))
                        ]
                        idx_fix = _select_best(_cands) if not _cands.empty else None
                        if idx_fix is not None:
                            prim_row = df.loc[idx_fix]
                # SWI_GAD family: prefer vendor-composited SWI_GAD (no parentheses) when available
                if str(norm_lbl).upper() == "SWI_GAD":
                    raw_lbl = str(prim_row.get(label_col, "")).upper()
                    if "(" in raw_lbl:
                        _cands = sel_prim[
                            (sel_prim[label_col].map(_norm_primary_label) == "SWI_GAD") &
                            (~sel_prim[label_col].astype(str).str.contains(r"\("))
                        ]
                        idx_fix = _select_best(_cands) if not _cands.empty else None
                        if idx_fix is not None:
                            prim_row = df.loc[idx_fix]
                base_type = str(prim_row.get("base_type", norm_lbl)).strip()
                prim_lbl  = str(prim_row.get("final_label", "")).strip()
                if bool(prim_row.get("is_derived", False)):
                    continue

                # Use the actual base_type variable and honor DERIVED_CATEGORY_SPEC['policy'].
                # plan_mode is computed earlier in this function ("make" | "add" | "none").
                from .preprocessing_utils import _filter_derivatives_by_policy
                # First, filter by policy (respect "ignore" / "convert_only" / "derive")
                _candidates = _filter_derivatives_by_policy(base_type, plan_mode)
                # Then, require a registered generator (same behavior as before)
                try:
                    from .generators import GENERATOR_REGISTRY as _GENS
                    _candidates = [(lab, key) for (lab, key) in _candidates if key in _GENS]
                except Exception:
                    # If registry can't be imported, keep whatever the policy allowed
                    pass
                for derived_lbl, gen_key in _candidates:
                    d_up = str(derived_lbl).strip().upper()
                    if d_up == prim_lbl.strip().upper():
                        continue
                    if (not make_derived_from_scratch) and (d_up in have_labels):
                        continue
                    if d_up in planned_labels:
                        continue

                    # Mirror convert naming/placement for this exam/timepoint
                    tp     = prim_row.get("timepoint_days", None)
                    tp_str = f"d{tp}" if tp is not None else "dNA"
                    ea     = prim_row.get("ExamAlias", None)
                    subdir = f"{pid}_{tp_str}_{ea}" if ea else f"{pid}_{tp_str}"
                    out_path = os.path.join(out_dir, pid, subdir, f"{pid}_{tp_str}_{_sanitize_label(derived_lbl)}.nii.gz")

                    sid = f"DERIVE:{prim_row.get('series_number','')}:{derived_lbl}"
                    rec = {
                        "Directory":             patient_rel,
                        # Keep this consistent with other plan rows (use the same ExamDirectory value)
                        "ExamDirectory":         df.loc[prim_row.name, "ExamDirectory"],
                        "ExamAlias":             df.loc[prim_row.name, "ExamAlias"],
                        "patientID":             pid,
                        "day0Date":              df.loc[prim_row.name, "day0Date"],
                        "mr_subdir_name":        df.loc[prim_row.name, "mr_subdir_name"],
                        "acq_dt":                prim_row.get("acq_dt"),
                        "timepoint_days":        tp,
                        "series_identifier":     sid,
                        "final_label":           derived_lbl,
                        "DerivedLabel":          derived_lbl,
                        "is_derived":            True,
                        "folder":                prim_row.get("folder") or prim_row.get("series_dir") or "",
                        "selected_for_conversion": True,
                        "Action":                "DERIVE",
                        "GeneratorKey":          gen_key,
                        "PrimaryLabel":          prim_lbl,
                        "PrimarySeriesIdentifier": prim_row.get("series_identifier") or str(prim_row.get("series_number","")),
                        "PrimarySeriesPath":     prim_row.get("folder") or prim_row.get("series_dir") or "",
                        "proposed_nifti_path":   out_path,
                    }
                    # --- Special handling: composite SWI planning ---
                    # If a vendor SWI already exists, DO NOT schedule any composite SWI derivation.
                    if base_type.upper() == "SWI" and d_up == "SWI":
                        # 1) If vendor SWI already present (non-derived "SWI"), skip scheduling a composite SWI.
                        try:
                            _vendor = df[(df.get("final_label","").astype(str).str.upper() == "SWI") & (~df.get("is_derived", False))]
                            if not _vendor.empty:
                                continue
                        except Exception:
                            # if any issue reading df, fall through to try MAG+PHASE
                            pass
                        # 2) If no vendor SWI: require MAG + PHASE primaries to synthesize one
                        try:
                            _mag = df[(df.get("final_label","").astype(str).str.upper() == "SWI(MAG)") & (~df.get("is_derived", False))]
                            _pha = df[(df.get("final_label","").astype(str).str.upper() == "SWI(PHASE)") & (~df.get("is_derived", False))]
                            mag_dir = (_mag.iloc[0].get("folder") or _mag.iloc[0].get("series_dir") or "")
                            pha_dir = (_pha.iloc[0].get("folder") or _pha.iloc[0].get("series_dir") or "")
                        except Exception:
                            mag_dir = ""; pha_dir = ""
                        if not (mag_dir and pha_dir):
                            # Can't synthesize a composite → skip planning a derived SWI
                            continue
                        # 3) Transform the row to represent a *single* derived SWI using the composite generator
                        rec["DerivedLabel"] = "SWI"                 # outward-facing label is plain SWI
                        rec["final_label"]  = "SWI"
                        rec["GeneratorKey"] = "swi_composite"       # still use the composite generator
                        rec["DeriveInputs"] = {"MAG": mag_dir, "PHASE": pha_dir}
                        # Make sure the proposed output filename uses "SWI.nii.gz"
                        try:
                            out_path = str(rec.get("proposed_nifti_path",""))
                            if out_path:
                                base = os.path.basename(out_path)
                                # Replace whatever derived label was there with SWI
                                base_swi = base.replace(_sanitize_label(derived_lbl), _sanitize_label("SWI"))
                                rec["proposed_nifti_path"] = os.path.join(os.path.dirname(out_path), base_swi)
                        except Exception:
                            pass
                        # record and move on
                        derived_rows.append(rec)
                        planned_labels.add("SWI")
                        continue  # don't run the generic path below for this composite case

                    # --- default derived planning path ---
                    derived_rows.append(rec)
                    planned_labels.add(d_up)

            if derived_rows:
                df = pd.concat([df, pd.DataFrame(derived_rows)], ignore_index=True, sort=False)

        # sort within exam and return
        sort_cols = [c for c in ["series_number","acq_dt"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort", na_position="last")
        return df

    # Process all new exams
    results = []
    if exams_to_process:
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_process_exam, r) for r in exams_to_process]
                for fut in _progress(as_completed(futures), total=len(futures), desc="Processing exams", unit="exam", enable=show_progress):
                    try:
                        df = fut.result()
                        if df is not None and not df.empty:
                            results.append(df)
                            _stream_block(df)  # stream per-exam block + blank line
                    except Exception as e:
                        print(f"[plan:ERROR] {e.__class__.__name__}: {e}")
                        raise
        else:
            for r in _progress(exams_to_process, total=len(exams_to_process), desc="Processing exams", unit="exam", enable=show_progress):
                df = _process_exam(r)
                if df is not None and not df.empty:
                    results.append(df)
                    _stream_block(df)

    # Combine everything for return value — normalize to HEADER first and
    # exclude empty AND all-NA frames to avoid pandas concat FutureWarning.
    normalized = []
    for df in (all_results + results):
        if df is None or df.empty:
            continue
        df2 = df.reindex(columns=HEADER)
        # if *every* cell is NA, skip (prevents the deprecation warning)
        if not df2.notna().to_numpy().any():
            continue
        normalized.append(df2)

    out_df = pd.concat(normalized, ignore_index=True) if normalized else pd.DataFrame(columns=HEADER)

    # If we couldn’t stream (e.g., .xlsx), write once at end with exam separators
    if plan_out and (plan_fh is None):
        # order: per exam by series_number then acq_dt
        sort_cols = [c for c in ["ExamDirectory","series_number","acq_dt"] if c in out_df.columns]
        out_df = out_df.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)

        # blank line between ExamDirectory groups
        def _with_sep(df):
            if "ExamDirectory" not in df.columns:
                return df
            blocks = []
            for _, g in df.groupby("ExamDirectory", sort=False, dropna=False):
                blocks.append(g)
                blocks.append(pd.DataFrame([{c: "" for c in df.columns}]))
            return pd.concat(blocks, ignore_index=True)
        _save_table(_with_sep(out_df), plan_out)

    # close stream if open
    try:
        if plan_fh is not None:
            plan_fh.close()
    except Exception:
        pass

    return out_df

# ------------------------------------------------------------
# Convert a single DICOM series to NIFTI
# ------------------------------------------------------------

def convert_dicom_to_nifti(
    dicom_series_dir: str,
    output_path: str,
    verbose: bool | None = None,
    debug: bool = False,
) -> str:
    import os
    from pathlib import Path
    from .preprocessing_utils import _nifti_from_any

    def _dbg(*a):
        if (verbose is True) or (verbose is None and os.environ.get("ASTRIL_DEBUG_CONVERT","") not in ("","0","false","False","FALSE")):
            print("[convert_dicom_to_nifti]", *a, flush=True)

    series = Path(dicom_series_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not series.exists():
        raise FileNotFoundError(f"Series directory not found: {series}")

    _dbg("Converting via dcm2niix:", series)
    # Delegate to dcm2niix-first converter which also writes PHI-safe JSON sidecar
    # and honors the requested output_path (including copying .bval/.bvec/.json if present).
    final_path, _ = _nifti_from_any(
        input_path_or_dir=str(series),
        verbose=verbose,
        output_path=str(out),
    )
    _dbg("final NIfTI:", Path(final_path).name)
    return final_path


# ------------------------------------------------------------
# Execute DICOM -> NIfTI conversions from a saved plan
# ------------------------------------------------------------
def convert_dicom_plan(
    plan_path: str,
    n_workers: int | None = None,
    overwrite: bool = False,
    log_out: str | None = None,
    show_progress: bool = True,
    unexpected_multiframe_policy: str = "keep_first",
    debug=False,
):
    """Read a plan produced by `plan_dicom_to_nifti_conversion` and run conversions
    for rows with a non-empty, non-"-" `proposed_nifti_path`.

    Streams a per-row CSV/TSV log to disk (thread-safe), similar to `demix_dicoms()`.
    unexpected_multiframe_policy controls what to do if a series that is expected
    to be a single 3D volume converts to a multi-frame/4D NIfTI.
      - "keep_first" (default): keep frame 0, overwrite the output, and warn
      - "skip": delete the output and warn

    Returns a DataFrame log with per-row status.
    """
    import os
    import csv as _csv
    from datetime import datetime
    import threading as _threading
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .preprocessing_utils import (
        _read_table, _save_table, _progress,
        _sanitize_label,
    )

    # Normalize/validate policy
    unexpected_multiframe_policy = str(unexpected_multiframe_policy or "keep_first").strip().lower()
    if unexpected_multiframe_policy not in {"keep_first", "skip"}:
        raise ValueError(
            "unexpected_multiframe_policy must be one of {'keep_first','skip'}; "
            f"got: {unexpected_multiframe_policy!r}"
        )

    # ---------- read and validate plan ----------
    plan = _read_table(plan_path)
    required = {"Directory", "ExamDirectory", "series_identifier", "final_label", "selected_for_conversion", "Action", "folder", "proposed_nifti_path"}
    missing = required - set(map(str, plan.columns))
    if missing:
        raise ValueError(f"Plan is missing required columns: {sorted(missing)}")

    def _valid_path(val) -> bool:
        if val is None:
            return False
        s = str(val).strip()
        return bool(s) and s != "-"

    # Keep only rows explicitly selected AND that have a valid target path
    def _to_bool(x):
        s = str(x).strip().lower()
        # truthy tokens
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        # falsy tokens (incl. empty string from CSV/TSV)
        if s in {"", "0", "false", "f", "no", "n", "off", "-"}:
            return False
        # anything else → False by default
        return False

    if "selected_for_conversion" in plan.columns:
        _sel = plan["selected_for_conversion"].map(_to_bool)
    else:
        _sel = pd.Series(True, index=plan.index)

    _has_path = plan["proposed_nifti_path"].map(_valid_path)
    todo = plan.loc[_sel & _has_path].copy()
    if todo.empty:
        cols = ["Directory","ExamDirectory","series_identifier","final_label",
                "folder","proposed_nifti_path","status","message","nii_path"]
        return pd.DataFrame(columns=cols)

    # Ensure primaries are converted before derived sequences
    if "Action" in todo.columns:
        _order = todo["Action"].astype(str).str.upper().map({"CONVERT": 0, "DERIVE": 1}).fillna(2)
        if "ExamDirectory" in todo.columns:
            todo = (
                todo.assign(_order=_order)
                    .sort_values(by=["ExamDirectory", "_order"], kind="mergesort")
                    .drop(columns="_order")
            )
        else:
            todo = todo.assign(_order=_order).sort_values(by=["_order"], kind="mergesort").drop(columns="_order")

    # threads tuned for I/O
    if n_workers is None:
        try:
            n_workers = min(8, max(1, os.cpu_count() or 2))
        except Exception:
            n_workers = 4

    # ---------- thread-safe streaming log (CSV/TSV only) ----------
    # Choose/default path
    if log_out is None:
        default_dir = os.path.dirname(os.path.abspath(plan_path)) or "."
        log_out = os.path.join(default_dir, f"convert_log_{datetime.now():%Y%m%d_%H%M%S}.csv")
    _log_path = log_out
    _ext = os.path.splitext(_log_path)[1].lower()
    if _ext not in (".csv", ".tsv"):
        # Fallback to csv if user passed an unknown extension (incl. .xlsx)
        _log_path = os.path.splitext(_log_path)[0] + ".csv"
        print(f"[convert_dicom_plan][note] {_ext} not supported for logs; writing .csv instead: {_log_path}")
        _ext = ".csv"
    _delim = "\t" if _ext == ".tsv" else ","
    os.makedirs(os.path.dirname(_log_path) or ".", exist_ok=True)

    LOG_HEADER = [
        "Directory","ExamDirectory","series_identifier","final_label",
        "Action","GeneratorKey","PrimaryLabel","DerivedLabel",
        "PrimarySeriesIdentifier","PrimarySeriesPath",
        "folder","proposed_nifti_path","status","message","nii_path",
        "nrrd_path","nrrd_error",
    ]
    _log_fh = open(_log_path, "a", newline="", encoding="utf-8")
    _log_writer = _csv.writer(_log_fh, delimiter=_delim)
    try:
        if _log_fh.tell() == 0:
            _log_writer.writerow(LOG_HEADER)
            _log_fh.flush()
    except Exception:
        pass

    _log_lock = _threading.Lock()
    def _stream_log_row(rec: dict):
        # Write one row; thread-safe; tolerate missing keys.
        with _log_lock:
            _log_writer.writerow([
                rec.get("Directory", ""),
                rec.get("ExamDirectory", ""),
                rec.get("series_identifier", ""),
                rec.get("final_label", ""),
                rec.get("Action", ""),
                rec.get("GeneratorKey", ""),
                rec.get("PrimaryLabel", ""),
                rec.get("DerivedLabel", ""),
                rec.get("PrimarySeriesIdentifier", ""),
                rec.get("PrimarySeriesPath", ""),
                rec.get("folder", ""),
                rec.get("proposed_nifti_path", ""),
                rec.get("status", ""),
                rec.get("message", ""),
                rec.get("nii_path", ""),
                rec.get("nrrd_path", ""),
                rec.get("nrrd_error", ""),
            ])
            _log_fh.flush()

    # ---------- expected-4D validation helpers ----------
    def _base_label(lbl: str) -> str:
         # Base label is the parent "family" name for a series label.
         #
         # New convention: derived labels use '-' within the label (e.g., DWI-FA).
         # Classifier labels may use parentheses (e.g., DWI(FA)).
         # Legacy data may still include '_' (e.g., DWI_FA).
         s = str(lbl or "").strip()
         # Parentheses form: DWI(FA) -> DWI
         if '(' in s:
             s = s.split('(', 1)[0]
         # Hyphen form: DWI-FA -> DWI
         if '-' in s:
             s = s.split('-', 1)[0]
         # Legacy underscore form: DWI_FA -> DWI
         if '_' in s:
             s = s.split('_', 1)[0]
         return s.strip().upper()

    _EXPECTED_4D_BASE_LABELS = {"DWI", "PERFUSION"}

    def _is_expected_4d_primary_row(row: pd.Series, rec: dict | None = None) -> bool:
        """
        Return True iff this conversion-plan row corresponds to a modality
        that is explicitly defined as a primary 4D acquisition.

        IMPORTANT:
        The conversion plan is the single source of truth here.
        No metadata- or heuristic-based inference is performed.
        """
        lbl = str(row.get("final_label", "") or "").strip().upper()
        return lbl in _EXPECTED_4D_BASE_LABELS

    def _delete_nifti_and_sidecars(nii_path: str):
        """Best-effort cleanup of a NIfTI output and its common sidecars."""
        import os
        if not nii_path:
            return
        candidates = [nii_path]
        base = nii_path
        if base.lower().endswith('.nii.gz'):
            stem = base[:-7]
        else:
            stem = os.path.splitext(base)[0]
        candidates += [stem + ext for ext in ('.json', '.bval', '.bvec')]
        for p in candidates:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    def _validate_expected_4d(nii_path: str) -> tuple[bool, str]:
        """Return (ok, message). ok=True iff the image is actually 4D."""
        try:
            from .preprocessing_utils import get_nifti_ndim
            ndim, shape = get_nifti_ndim(nii_path)
            if int(ndim) == 4 and len(shape) == 4 and int(shape[3]) > 1:
                return True, f"ndim={ndim}, shape={shape}"
            return False, f"expected 4D but got ndim={ndim}, shape={shape}"
        except Exception as e:
            return False, f"unable to validate ndim: {type(e).__name__}: {e}"

    def _resolve_unexpected_multiframe_policy(row: pd.Series) -> str:
        """Per-row override (when present in plan); otherwise use function default."""
        try:
            if "unexpected_multiframe_policy" in row.index:
                v = str(row.get("unexpected_multiframe_policy") or "").strip().lower()
                if v in {"keep_first", "skip"}:
                    return v
        except Exception:
            pass
        return unexpected_multiframe_policy

    def _validate_3d_or_singleton_4d(nii_path: str) -> tuple[bool, int | None, tuple[int, ...] | None, str]:
        """Return (ok, ndim, shape, message). ok=True iff the image is effectively 3D.

        - ndim==3 -> ok
        - ndim==4 and shape[3]==1 -> ok (can be safely squeezed to 3D)
        - ndim==4 and shape[3]>1 -> not ok (unexpected multiframe)
        """
        try:
            from .preprocessing_utils import get_nifti_ndim
            ndim, shape = get_nifti_ndim(nii_path)
            ndim_i = int(ndim)
            shp = tuple(int(x) for x in (shape or ()))
            if ndim_i == 3:
                return True, ndim_i, shp, f"ndim={ndim_i}, shape={shp}"
            if ndim_i == 4 and len(shp) == 4:
                if shp[3] == 1:
                    return True, ndim_i, shp, f"ndim={ndim_i}, shape={shp} (singleton frame)"
                return False, ndim_i, shp, f"ndim={ndim_i}, shape={shp}"
            return False, ndim_i, shp, f"unexpected ndim={ndim_i}, shape={shp}"
        except Exception as e:
            return False, None, None, f"unable to validate ndim: {type(e).__name__}: {e}"

    def _enforce_expected_single_frame(
        written_nii_path: str,
        policy: str,
    ) -> tuple[str | None, str, str]:
        """Ensure output is a single-frame 3D NIfTI.

        Returns (final_path_or_none, status, message).
        - If policy=="keep_first" and the output is multiframe 4D, we overwrite with frame 0.
        - If policy=="skip" and the output is multiframe 4D, we delete outputs and return None.
        - If output is 3D or singleton-4D, we return the (possibly overwritten) path.
        """
        ok, ndim, shape, msg = _validate_3d_or_singleton_4d(written_nii_path)
        if ok and ndim == 3:
            return written_nii_path, "ok", ""

        # For singleton 4D, squeeze to 3D quietly (but note in message for transparency)
        if ok and ndim == 4 and shape and len(shape) == 4 and shape[3] == 1:
            try:
                from .preprocessing_utils import extract_nifti_frame
                extract_nifti_frame(written_nii_path, 0, written_nii_path)
                return written_nii_path, "ok_singleton_frame_squeezed", f"Converted 4D singleton to 3D: {msg}"
            except Exception as e:
                # If we can't squeeze, treat as failure and follow policy
                ok = False
                msg = f"singleton-4D squeeze failed: {type(e).__name__}: {e}; original: {msg}"

        # Unexpected multiframe 4D (or unknown) -> apply policy
        if policy == "skip":
            _delete_nifti_and_sidecars(written_nii_path)
            return None, "skipped_expected_3d_got_4d", msg

        # keep_first
        try:
            from .preprocessing_utils import extract_nifti_frame
            extract_nifti_frame(written_nii_path, 0, written_nii_path)
            return written_nii_path, "ok_first_frame_only", f"[WARN] Expected 3D but got multi-frame; kept frame 0. {msg}"
        except Exception as e:
            _delete_nifti_and_sidecars(written_nii_path)
            return None, "failed", f"Failed to extract first frame: {type(e).__name__}: {e}; original: {msg}"

    # ---------- worker ----------
    def _convert_row(row: pd.Series) -> dict:
        series_dir = str(row.get("folder", ""))
        out_path = str(row.get("proposed_nifti_path", "")).strip()
        rec = {
            "Directory": row.get("Directory", ""),
            "ExamDirectory": row.get("ExamDirectory", ""),
            "series_identifier": row.get("series_identifier", ""),
            "final_label": row.get("final_label", ""),
            "folder": series_dir,
            "proposed_nifti_path": out_path,
            # include derivation context for logging (even for CONVERT rows)
            "Action": str(row.get("Action", "CONVERT")).upper(),
            "GeneratorKey": str(row.get("GeneratorKey", "")).strip(),
            "PrimaryLabel": str(row.get("PrimaryLabel", "")).strip(),
            "DerivedLabel": str(row.get("DerivedLabel", "")).strip(),
            "PrimarySeriesIdentifier": row.get("PrimarySeriesIdentifier", ""),
            "PrimarySeriesPath": row.get("PrimarySeriesPath", ""),
            "status": None,
            "message": "",
            "nii_path": "",
        }
        if not series_dir or not os.path.isdir(series_dir):
            rec["status"] = "missing_series_folder"
            rec["message"] = f"Series folder not found: {series_dir}"
            return rec

        # Skip if exists and not overwriting; ensure parent exists
        if os.path.isabs(out_path):
            out_parent = os.path.dirname(out_path)
        else:
            out_parent = os.path.dirname(os.path.abspath(out_path))
        try:
            if os.path.isfile(out_path) and not overwrite:
                rec["status"] = "exists"
                rec["nii_path"] = out_path
                return rec
            os.makedirs(out_parent or ".", exist_ok=True)
        except Exception as e:
            rec["status"] = "error"
            rec["message"] = f"Unable to create parent dir: {type(e).__name__}: {e}"
            return rec

        try:
            from .preprocess import convert_dicom_to_nifti  # local import to avoid cycles
            action = rec.get("Action", "CONVERT")
            if action == "CONVERT":
                written = convert_dicom_to_nifti(
                    dicom_series_dir=series_dir,
                    output_path=out_path,
                    debug=debug,
                )

                # For certain *primary* sequences (e.g., DWI/Perfusion), the converted file must be 4D.
                # If it is not, treat this conversion as skipped (remove outputs) so we don't
                # accidentally derive from an incomplete primary.
                if _is_expected_4d_primary_row(row, rec):
                    ok4d, msg = _validate_expected_4d(written)
                    if not ok4d:
                        _delete_nifti_and_sidecars(written)
                        rec["status"] = "skipped_expected_4d_not_4d"
                        rec["message"] = msg
                        rec["nii_path"] = ""
                        return rec

                # For *non-primary* sequences, we expect a single-frame/3D output.
                # Some DICOM series that "should" be 3D (e.g., vendor-derived maps like DWI_TRACE)
                # can be stored as multi-frame in the source directory and convert to 4D.
                # Apply policy: keep frame 0 (default) or skip.
                if not _is_expected_4d_primary_row(row, rec):
                    pol = _resolve_unexpected_multiframe_policy(row)
                    final_path, st, msg = _enforce_expected_single_frame(written, pol)
                    if final_path is None:
                        rec["status"] = st
                        rec["message"] = msg
                        rec["nii_path"] = ""
                        return rec
                    written = final_path
                    # Preserve warnings/messages in the log
                    if msg:
                        rec["message"] = msg
                    # Only override status if it indicates a non-standard success
                    if st != "ok":
                        rec["status"] = st
                        rec["nii_path"] = written
                        # Continue to optional NRRD export
                    else:
                        rec["status"] = "ok"
                        rec["nii_path"] = written

                if rec.get("status") is None:
                    rec["status"] = "ok"
                    rec["nii_path"] = written

                # Also export diffusion NRRD for Slicer compatibility (parent diffusion only).
                # We keep NIfTI + sidecars as primary outputs.
                try:
                    lbl_raw = (rec.get("final_label") or rec.get("PrimaryLabel") or "").strip()
                    raw_up = lbl_raw.upper()
                    # Only export NRRD for the *base primary* acquisitions explicitly supported:
                    #   - DWI
                    #   - PERFUSION
                    #
                    # Important: vendor-derived maps can still be Action=CONVERT and look like
                    # "DWI(FA)" / "DWI-FA" / legacy "DWI_FA". We must NOT export NRRD for those.
                    is_base_primary = raw_up in {"DWI", "PERFUSION"}
                    is_derived = bool((rec.get("DerivedLabel") or "").strip()) or (str(rec.get("Action","")).upper() == "DERIVE")
 
                    if is_base_primary and not is_derived:
                        from .preprocessing_utils import export_dwi_nrrd_from_dicoms
                        if out_path.lower().endswith(".nii.gz"):
                            stem = out_path[:-7]
                        else:
                            stem = os.path.splitext(out_path)[0]
                        nrrd_path = stem + ".nrrd"
                        rec["nrrd_path"] = export_dwi_nrrd_from_dicoms(series_dir, nrrd_path, verbose=False, debug=debug)
                except Exception as e:
                    rec["nrrd_path"] = ""
                    rec["nrrd_error"] = f"{type(e).__name__}: {e}"
            elif action == "DERIVE":
                from .preprocessing_utils import run_derived_generator
                generator_key = rec.get("GeneratorKey", "")
                primary_label = rec.get("PrimaryLabel", "")
                derived_label = rec.get("DerivedLabel", "")
                src_input = rec.get("DeriveInputs") or rec.get("PrimarySeriesPath") or series_dir  # dict for multi-input, else DICOM dir
                if "is_derived" in plan.columns:
                    _not_derived = ~plan["is_derived"].map(_to_bool).fillna(False)
                else:
                    _not_derived = pd.Series(True, index=plan.index)
                # Enforce: SWI/SWI_GAD MIP/MINIP must use a composite of the same family (vendor or synthesized).
                if str(rec.get("GeneratorKey","")) in ("swi_mip","swi_minip"):
                    # Decide which family this derived label belongs to
                    _derived_up = str(rec.get("DerivedLabel","")).upper()
                    comp_base = "SWI_GAD" if _derived_up.startswith("SWI_GAD") else "SWI"
                    base_name = os.path.basename(out_path)
                    comp_name = base_name.replace(_sanitize_label(rec.get("DerivedLabel","")),
                                                  _sanitize_label(comp_base))
                    comp_path = os.path.join(os.path.dirname(out_path), comp_name)
                    if os.path.isfile(comp_path):
                        # 1) already-derived composite exists → use it
                        src_input = comp_path
                    else:
                        # 2) try vendor composite primary for the same family (label == comp_base)
                        vendor_swi_dir = ""
                        vendor_swi_nii = ""
                        try:
                            _vendor = plan[
                                (plan.get("final_label","").astype(str).str.strip().str.upper() == comp_base) & _not_derived
                            ]
                            if not _vendor.empty:
                                vendor_swi_dir = (_vendor.iloc[0].get("folder") or
                                                  _vendor.iloc[0].get("series_dir") or "")
                                vendor_swi_nii = _vendor.iloc[0].get("proposed_nifti_path") or ""
                        except Exception:
                            pass
                        if vendor_swi_dir or vendor_swi_nii:
                            # Use the vendor composite directly (DICOM dir or existing NIfTI)
                            src_input = vendor_swi_nii if (vendor_swi_nii and os.path.isfile(vendor_swi_nii)) \
                                                       else vendor_swi_dir
                        else:
                            # 3) synthesize composite now from family-matched MAG + PHASE, then use it
                            mag_dir = ""
                            pha_dir = ""
                            try:
                                _mag = plan[
                                    (plan.get("final_label","").astype(str).str.strip().str.upper() == f"{comp_base}(MAG)") & _not_derived
                                ]
                                _pha = plan[
                                    (plan.get("final_label","").astype(str).str.strip().str.upper() == f"{comp_base}(PHASE)") & _not_derived
                                ]
                                if not _mag.empty:
                                    mag_dir = (_mag.iloc[0].get("folder") or _mag.iloc[0].get("series_dir") or "")
                                if not _pha.empty:
                                    pha_dir = (_pha.iloc[0].get("folder") or _pha.iloc[0].get("series_dir") or "")
                            except Exception:
                                pass
                            if mag_dir and pha_dir:
                                _ = run_derived_generator(
                                    input_path_or_dicom_dir={"MAG": mag_dir, "PHASE": pha_dir},
                                    output_path=comp_path,
                                    generator_key="swi_composite",
                                    primary_label=comp_base,
                                    derived_label=comp_base,
                                    debug=debug,
                                )
                                src_input = comp_path
                            else:
                                # Could not find vendor SWI or both MAG+PHASE → fail this row cleanly
                                rec["status"] = "error"
                                rec["message"] = (
                                    f"{comp_base}(MIP/MINIP) requires a composite {comp_base}, but neither a vendor {comp_base} "
                                    f"nor both {comp_base}(MAG) and {comp_base}(PHASE) were found."
                                )
                                return rec
                written = run_derived_generator(
                    input_path_or_dicom_dir=src_input,
                    output_path=out_path,
                    generator_key=generator_key,
                    primary_label=primary_label,
                    derived_label=derived_label,
                    debug=debug,
                )

                # Derived outputs are expected to be single-frame/3D.
                pol = _resolve_unexpected_multiframe_policy(row)
                final_path, st, msg = _enforce_expected_single_frame(written, pol)
                if final_path is None:
                    rec["status"] = st
                    rec["message"] = msg
                    rec["nii_path"] = ""
                    return rec
                if msg:
                    rec["message"] = msg
                rec["status"] = "ok" if st == "ok" else st
                rec["nii_path"] = final_path

                # Optional: also export a diffusion NRRD for DWI parent series to improve
                # 3D Slicer / SlicerDMRI compatibility (embedded gradients).
                # We intentionally keep the standard NIfTI + sidecars as the primary output.
                try:
                    lbl = str(rec.get("final_label") or rec.get("Label") or rec.get("PrimaryLabel") or "").strip().upper()
                    if lbl == "DWI":
                        from .preprocessing_utils import export_dwi_nrrd_from_dicoms
                        stem = out_path[:-7] if out_path.lower().endswith(".nii.gz") else os.path.splitext(out_path)[0]
                        nrrd_path = stem + ".nrrd"
                        rec["nrrd_path"] = export_dwi_nrrd_from_dicoms(series_dir, nrrd_path, verbose=False, debug=debug)
                except Exception as e:
                    rec["nrrd_path"] = ""
                    rec["nrrd_error"] = f"{type(e).__name__}: {e}"
            else:
                rec["status"] = "skipped"
                rec["message"] = f"Unknown Action '{action}'"
        except Exception as e:
            rec["status"] = "failed"
            rec["message"] = f"{type(e).__name__}: {e}"
        return rec

    # ---------- execute ----------
    # We run conversions in two phases so that DERIVE jobs never run off a primary
    # 4D sequence that converted incorrectly (e.g., DWI/Perfusion that ended up 3D).
    # Vendor-provided derived series are still converted normally (they are Action=CONVERT).

    def _action_of(r: "pd.Series") -> str:
        return str(r.get("Action", "CONVERT")).upper().strip() or "CONVERT"

    convert_jobs = [r for _, r in todo.iterrows() if _action_of(r) == "CONVERT"]
    derive_jobs  = [r for _, r in todo.iterrows() if _action_of(r) == "DERIVE"]

    # Track which primary 4D labels were successfully converted per exam.
    # Keyed by (ExamDirectory, BASE_LABEL) where BASE_LABEL is e.g. 'DWI' or 'PERFUSION'.
    primary4d_ok: dict[tuple[str, str], bool] = {}

    records = []
    try:
        # ---- Phase 1: CONVERT (includes vendor-derived series) ----
        if convert_jobs:
            def _update_primary4d_ok(r: "pd.Series", rec: dict):
                # Only track *primary* members of expected-4D families (DWI/Perfusion).
                #
                # IMPORTANT:
                # Derivation may run in a later invocation where the primary NIfTI already exists.
                # In that case _convert_row() returns status="exists" and we must re-validate the
                # dimensionality from disk; otherwise we can incorrectly block DERIVE jobs.
                try:
                    if _is_expected_4d_primary_row(r, rec):
                        from .preprocessing_utils import get_nifti_ndim

                        key = (str(rec.get("ExamDirectory", "")), _base_label(rec.get("final_label", "")))
                        status = str(rec.get("status", "") or "").strip().lower()
                        nii_path = str(rec.get("nii_path", "") or "").strip()

                        # For both freshly converted and pre-existing primaries, confirm on-disk ndim.
                        if status in {"ok", "exists"} and nii_path and os.path.isfile(nii_path):
                            try:
                                ndim, _shape = get_nifti_ndim(nii_path)
                                if int(ndim) == 4:
                                    primary4d_ok[key] = True
                                else:
                                    primary4d_ok.setdefault(key, False)
                            except Exception:
                                # If we can't read ndim, be conservative and mark as not-ok.
                                primary4d_ok.setdefault(key, False)
                        else:
                            # Any non-ok status (or missing file) is treated as not-ok.
                            primary4d_ok.setdefault(key, False)
                except Exception:
                    pass

            if n_workers == 1:
                for r in _progress(convert_jobs, total=len(convert_jobs), desc="Converting", unit="series", enable=show_progress):
                    rec = _convert_row(r)
                    records.append(rec)
                    _stream_log_row(rec)
                    _update_primary4d_ok(r, rec)
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    ft_to_row = {pool.submit(_convert_row, r): r for r in convert_jobs}
                    for ft in _progress(as_completed(ft_to_row), total=len(ft_to_row), desc="Converting", unit="series", enable=show_progress):
                        r = ft_to_row.get(ft)
                        try:
                            rec = ft.result()
                        except Exception as e:
                            rec = {"status": "exception", "message": f"Future failed: {e}"}
                        records.append(rec)
                        _stream_log_row(rec)
                        if r is not None:
                            _update_primary4d_ok(r, rec)

        # ---- Phase 2: DERIVE (skip if primary 4D conversion was invalid) ----
        if derive_jobs:
            def _derive_wrapper(r: "pd.Series") -> dict:
                prim_lbl = str(r.get("PrimaryLabel", "")).strip() or str(r.get("final_label", "")).strip()
                exam_dir = str(r.get("ExamDirectory", ""))
                base = _base_label(prim_lbl)
                if base in {"DWI", "PERFUSION"}:
                    ok = primary4d_ok.get((exam_dir, base), None)
                    if ok is False:
                        # Explicitly block derivation from an invalid 4D primary conversion.
                        return {
                            "Directory": r.get("Directory", ""),
                            "ExamDirectory": exam_dir,
                            "series_identifier": r.get("series_identifier", ""),
                            "final_label": r.get("final_label", ""),
                            "Action": "DERIVE",
                            "GeneratorKey": r.get("GeneratorKey", ""),
                            "PrimaryLabel": prim_lbl,
                            "DerivedLabel": r.get("DerivedLabel", ""),
                            "PrimarySeriesIdentifier": r.get("PrimarySeriesIdentifier", ""),
                            "PrimarySeriesPath": r.get("PrimarySeriesPath", ""),
                            "folder": str(r.get("folder", "")),
                            "proposed_nifti_path": str(r.get("proposed_nifti_path", "")).strip(),
                            "status": "skipped_invalid_primary",
                            "message": f"Skipped DERIVE because primary '{base}' converted as non-4D (see prior warning/log for this exam).",
                            "nii_path": "",
                            "nrrd_path": "",
                            "nrrd_error": "",
                        }
                return _convert_row(r)

            if n_workers == 1:
                for r in _progress(derive_jobs, total=len(derive_jobs), desc="Deriving", unit="series", enable=show_progress):
                    rec = _derive_wrapper(r)
                    records.append(rec)
                    _stream_log_row(rec)
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futs = [pool.submit(_derive_wrapper, r) for r in derive_jobs]
                    for ft in _progress(as_completed(futs), total=len(futs), desc="Deriving", unit="series", enable=show_progress):
                        try:
                            rec = ft.result()
                        except Exception as e:
                            rec = {"status": "exception", "message": f"Future failed: {e}"}
                        records.append(rec)
                        _stream_log_row(rec)

    finally:
        try:
            _log_fh.close()
        except Exception:
            pass
    # Build pandas result for return value (stream already written to disk)
    log_df = pd.DataFrame.from_records(records)

    # No _save_table() here since we've streamed the log.
    # _save_table(log_df, log_out)  # (intentionally disabled for streamed log)

    try:
        counts = log_df["status"].value_counts(dropna=False).to_dict()
        print("[convert_dicom_plan] status counts:", counts)
    except Exception:
        pass

    return log_df

# ------------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# QC PDF generation for preprocessed MRI libraries
# -----------------------------------------------------------------------------

def _qc_generate_one_patient_qc_pdfs_worker(args):
    """Top-level worker for Windows ProcessPoolExecutor (must be picklable)."""
    patient_dir, series_order, out_dir, max_exams_per_page, left_margin_scale = args
    from .preprocessing_utils import generate_patient_qc_pdfs
    brain_pdf, brain_norm_pdf = generate_patient_qc_pdfs(
        patient_dir=patient_dir,
        series_order=series_order,
        out_dir=out_dir,
        max_exams_per_page=max_exams_per_page,
        left_margin_scale=left_margin_scale,
    )
    return patient_dir, brain_pdf, brain_norm_pdf


def generate_preprocessing_qc_pdfs(
    root_dir: str,
    n_workers: int | None = None,
    out_dir: str | None = None,
    show_progress: bool = True,
    *,
    max_exams_per_page: int = 4,
    left_margin_scale: float = 2.25,
):
    """Generate per-patient QC PDFs for a preprocessed MRI library.

    Directory layout:
        root_dir/{patient_dirs}/{exam_dirs}/*.nii.gz

    Filenames expected:
        {patient}_{timepoint}_{seriesType}_brain.nii.gz
        {patient}_{timepoint}_{seriesType}_brain-norm.nii.gz

    Skips:
        - *_unregistered.nii.gz
        - non-.nii.gz (e.g., .nrrd)

    Produces two PDFs per patient:
        - *_qc_brain.pdf
        - *_qc_brain-norm.pdf
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        _tqdm = None

    from .preprocessing_utils import collect_preprocessed_series_types, _progress

    root_dir = os.path.abspath(os.fspath(root_dir))
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"--dir not found or not a directory: {root_dir}")

    patient_dirs = [
        os.path.join(root_dir, d)
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    series_order = collect_preprocessed_series_types(root_dir)
    if not series_order:
        raise RuntimeError(
            "No eligible preprocessed NIfTI files found under --dir. "
            "Expected *_brain.nii.gz or *_brain-norm.nii.gz under {patient}/{exam}/ (legacy *_brain_norm also supported)."
        )

    # count exams for progress reporting
    patient_exam_counts = {}
    total_exams = 0
    for pdir in patient_dirs:
        try:
            n_exams = sum(1 for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d)))
        except Exception:
            n_exams = 0
        patient_exam_counts[pdir] = n_exams
        total_exams += n_exams

    if n_workers is None:
        try:
            n_workers = max(1, min(8, os.cpu_count() or 1))
        except Exception:
            n_workers = 1
    else:
        n_workers = int(n_workers)
        if n_workers < 1:
            n_workers = 1

    patients_total = len(patient_dirs)

    if show_progress and _tqdm is not None:
        pbar_pat = _tqdm(total=patients_total, desc="QC PDFs (patients)", unit="patient")
        pbar_ex = _tqdm(total=total_exams, desc="QC PDFs (exams)", unit="exam")
    else:
        pbar_pat = None
        pbar_ex = None

    results = []
    try:
        if n_workers == 1:
            it = patient_dirs
            if pbar_pat is None:
                it = _progress(it, total=patients_total, desc="QC PDFs", unit="patient", enable=show_progress)
            from .preprocessing_utils import generate_patient_qc_pdfs
            for pdir in it:
                brain_pdf, brain_norm_pdf = generate_patient_qc_pdfs(
                    patient_dir=pdir,
                    series_order=series_order,
                    out_dir=out_dir,
                    max_exams_per_page=max_exams_per_page,
                    left_margin_scale=left_margin_scale,
                )
                results.append({
                    "patient_dir": pdir,
                    "qc_brain_pdf": brain_pdf,
                    "qc_brain_norm_pdf": brain_norm_pdf,
                })
                if pbar_pat is not None:
                    pbar_pat.update(1)
                if pbar_ex is not None:
                    pbar_ex.update(patient_exam_counts.get(pdir, 0))
        else:
            job_args = [
                (pdir, series_order, out_dir, max_exams_per_page, left_margin_scale)
                for pdir in patient_dirs
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futs = {pool.submit(_qc_generate_one_patient_qc_pdfs_worker, a): a[0] for a in job_args}
                it = as_completed(futs)
                if pbar_pat is None:
                    it = _progress(it, total=patients_total, desc="QC PDFs", unit="patient", enable=show_progress)
                for fut in it:
                    pdir = futs[fut]
                    try:
                        _pdir, brain_pdf, brain_norm_pdf = fut.result()
                    except Exception as e:
                        brain_pdf, brain_norm_pdf = None, None
                        print(f"[generate_preprocessing_qc_pdfs][WARN] Failed for {pdir}: {e}")
                    results.append({
                        "patient_dir": pdir,
                        "qc_brain_pdf": brain_pdf,
                        "qc_brain_norm_pdf": brain_norm_pdf,
                    })
                    if pbar_pat is not None:
                        pbar_pat.update(1)
                    if pbar_ex is not None:
                        pbar_ex.update(patient_exam_counts.get(pdir, 0))
    finally:
        if pbar_pat is not None:
            pbar_pat.close()
        if pbar_ex is not None:
            pbar_ex.close()

    return results
def _build_cli_parser() -> "argparse.ArgumentParser":
    # Combine RawText (preserve newlines) + show defaults
    class _SmartFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        prog="python -m astril.preprocess",
        description=(
            "MRI Preprocessing Tools\n"
            "\n"
            "Usage:\n"
            "  python -m astril.preprocess <command> [options]\n"
            "\n"
            "Commands cover normalization, resizing (and reversal), registration, HD-BET,\n"
            "DICOM demixing, planning and converting to NIfTI, and metadata utilities."
        ),
        formatter_class=_SmartFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    # Print top-level help when no subcommand is provided (optional polish)
    parser.set_defaults(func=lambda *_a, **_k: parser.print_help())

    # ---------- normalize ----------
    p = sub.add_parser(
        "normalize",
        help="Normalize an MRI volume using a binary mask (zero-mean/unit-variance in-mask).",
        formatter_class=_SmartFormatter,
    )
    p.add_argument("--input", required=True, help="Input NIfTI image (.nii|.nii.gz).")
    p.add_argument("--mask", required=True, help="Binary mask NIfTI (same shape; >0=in brain).")
    p.add_argument("--output", required=True, help="Output NIfTI path for normalized image.")
    p.add_argument("--zero_outside_mask", action="store_true", help="Set voxels outside masked region to 0.")
    def _run_normalize(a):
        normalize_masked_image(a.input, a.mask, a.output, a.zero_outside_mask)
    p.set_defaults(func=_run_normalize)

    # ---------- resize ----------
    p = sub.add_parser(
        "resize",
        help="Resample to target shape/voxel size; optionally recenter ROI and save padding record.",
        formatter_class=_SmartFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input NIfTI to be resampled.")
    p.add_argument("--output", required=True, help="Output path for resized NIfTI.")
    p.add_argument("--data_dims", default="240,240,155", help="Target voxel grid as comma-separated integers X,Y,Z (e.g., 240,240,155).")
    p.add_argument("--voxel_dims", default="1.0,1.0,1.0", help="Target voxel spacing (mm) as comma-separated floats X,Y,Z (e.g., 1.0,1.0,1.0).")
    p.add_argument("--interp", type=int, default=1, help="Interpolation order for resampling (0=nearest, 1=linear, 2=quadratic, ...).")
    p.add_argument("--save_padding_record", action="store_true", help="Write a padding/resize record alongside the output to enable exact reversal later.")
    p.add_argument("--padding_record", help="If provided, read this existing padding record to reproduce previous centering/shape steps.")
    p.add_argument("--roimask", help="Optional ROI mask (NIfTI). If set (and no --padding_record), the ROI centroid is translated to the volume center before resampling.")
    p.add_argument("--translation_only", action="store_true", help="Only apply translation from ROI centering; do not change shape or voxel spacing.")
    def _run_resize(a):
        resize_mri(
            input_filepath=a.input,
            output_filepath=a.output,
            target_shape=tuple(map(int, a.data_dims.split(","))),
            target_voxel_dims=tuple(map(float, a.voxel_dims.split(","))),
            interp=a.interp,
            save_padding_record=a.save_padding_record,
            padding_record_path=a.padding_record,
            roi_mask_path=a.roimask,
            translation_only=a.translation_only,
        )
    p.set_defaults(func=_run_resize)

    # ---------- reverse_resize ----------
    p = sub.add_parser(
        "reverse_resize",
        help="Reverse a previous resize using a saved padding record.", formatter_class=_SmartFormatter)
    p.add_argument("--input", required=True, help="Resized NIfTI to reverse back to the original grid.")
    p.add_argument("--output", required=True, help="Path for reversed (original space) output NIfTI.")
    p.add_argument("--padding_record", required=True, help="Padding record produced by the prior `resize` operation.")
    p.add_argument("--interp", type=int, default=1, help="Interpolation order used during resampling back (0=nearest, 1=linear, ...).")
    def _run_reverse(a):
        reverse_resize_mri(a.input, a.output, a.padding_record, interp=a.interp)
    p.set_defaults(func=_run_reverse)

    # ---------- match_affine ----------
    p = sub.add_parser(
        "match_affine",
        help="Match affine of INPUT to DONOR image.", formatter_class=_SmartFormatter)
    p.add_argument("--input", required=True, help="Source NIfTI to be resampled to the donor grid.")
    p.add_argument("--donor", required=True, help="Donor NIfTI whose shape/affine to match.")
    p.add_argument("--output", required=True, help="Output path for resampled image (matches donor grid).")
    def _run_match(a):
        match_direction_matrices(a.input, a.donor, a.output)
    p.set_defaults(func=_run_match)

    # ---------- merge_masks ----------
    p = sub.add_parser(
        "merge_masks",
        help="Merge multiple binary masks (logical OR); optional hole filling and affine checks.", formatter_class=_SmartFormatter)
    p.add_argument("--masks", nargs="+", required=True, help="Two or more binary mask NIfTIs to merge (same shape; >0=in mask).")
    p.add_argument("--output", required=True, help="Output path for merged mask NIfTI.")
    p.add_argument("--fill_holes", action="store_true", help="Fill interior holes within the merged mask.")
    p.add_argument("--strict_affine", action="store_true", help="Require all input affines to match exactly; otherwise raise an error.")
    def _run_merge(a):
        merge_binary_masks(a.masks, a.output, fill_holes=a.fill_holes, strict_affine=a.strict_affine)
    p.set_defaults(func=_run_merge)

    # ---------- register ----------
    p = sub.add_parser(
        "register",
        help="Rigid/affine/translation registration via SimpleITK; or apply an existing transform.", formatter_class=_SmartFormatter)
    p.add_argument("--fixed", required=True, help="Fixed/reference image (NIfTI).")
    p.add_argument("--moving", required=True, help="Moving image to align (NIfTI).")
    p.add_argument("--output", required=True, help="Registered output path (NIfTI).")
    basic = p.add_argument_group("Registration basics")
    basic.add_argument("--registration_type", choices=["rigid", "affine", "translation"], default="rigid",
                       help="Transform family to optimize.")
    basic.add_argument("--similarity_metric", choices=["correlation", "mi"], default="mi",
                       help="Similarity metric: Pearson correlation or Mattes mutual information.")
    basic.add_argument(
        "--registration_strategy",
        choices=["accurate", "medium", "fast"],
        default="medium",
        help=(
            "Speed/accuracy preset. 'accurate' uses all voxels; 'medium'/'fast' use random metric sampling."
        ),
    )
    basic.add_argument(
        "--metric_sampling_seed",
        type=int,
        default=None,
        help=(
            "Optional integer seed for deterministic random metric sampling (only relevant for 'medium'/'fast')."
        ),
    )
    basic.add_argument("--metric_focus", default="none",
                       help="{'none', 'background_subtracted', 'foreground', 'edges', 'lowhigh', 'highlow'} Restrict the registration metric to a subset of voxels. 'foreground' masks background/air; 'edges' keeps strong edges; 'lowhigh' uses fixed low-intensity + moving high-intensity tails; 'highlow' uses fixed high-intensity + moving low-intensity tails.")
    basic.add_argument("--metric_focus_percentile", default = 95, help="Percentile (0-100) of the high-contrast voxels to use for edge detection.")
    basic.add_argument("--metric_focus_sigma_mm", default = 1, help="Physical-space Gaussian smoothing (in millimeters) applied before computing gradient magnitude for edge selection.")
    basic.add_argument("--metric_focus_dilate_vox", default=1, help="Number of voxels by which the edge-based metric mask is dilated isotropically after thresholding.")
    basic.add_argument("--interp", default="cubic", help="Interpolation for registration+resampling. Accepts int 0-5 or strings like linear/cubic; use 'auto' to reuse recorded interpolation when --apply_only.")
    basic.add_argument("--as_integer", action="store_true", help="Save volume with integer data rather than as a float. Recommended if registering segmentation masks.")
    basic.add_argument("--registration_voxel_mm", default="1,1,1", help="Optional spacing (mm, mm, mm) to use *during transform estimation* (apply_only=False). This speeds up registration by downsampling both fixed and moving frames to a common voxel size before optimization. Does not affect spacing of output images.")
    basic.add_argument("--use_first_frame_only", action="store_true", help="For 4d moving volumes, return registered volume as a 3d volume registered only using the first frame of the input volume.")
    basic.add_argument("--keep_moving_grid", action="store_true", help="Retain voxel dimensions/spacing from the original moving image instead of resampling to match dimensions/spacing of the fixed image.")

    io = p.add_argument_group("Transforms I/O")
    io.add_argument("--transform",
                    help="Where to save the fitted transform (.tfm), or load from when --apply_only is set.")
    io.add_argument("--apply_only", action="store_true",
                    help="Skip optimization and only apply the transform given by --transform.")
    perf = p.add_argument_group("Performance & logging")
    perf.add_argument("--verbose", action="store_true", help="Print metric values and detailed status messages.")
    perf.add_argument("--debug", action="store_true", help="Print/save debug outputs.")
    perf.add_argument("--save_dummy_ref", action="store_true",
                      help="Save zeroed fixed/moving reference images next to the transform for later reversal.")
    def _run_register(a):
        register_images(
            fixed_path=a.fixed, moving_path=a.moving, output_path=a.output,
            transform_path=a.transform, apply_only=a.apply_only,
            registration_type=a.registration_type, similarity_metric=a.similarity_metric,
            registration_strategy=a.registration_strategy,
            metric_sampling_seed=a.metric_sampling_seed,
            metric_focus=a.metric_focus,
            metric_focus_percentile=a.metric_focus_percentile,
            metric_focus_sigma_mm=a.metric_focus_sigma_mm,
            metric_focus_dilate_vox=a.metric_focus_dilate_vox,
            save_dummy_ref=a.save_dummy_ref,
            interpolation=a.interp,
            integer=a.as_integer,
            registration_voxel_mm=a.registration_voxel_mm,
            use_first_frame_only=a.use_first_frame_only,
            keep_moving_grid=a.keep_moving_grid,
            verbose=a.verbose,
            debug=a.debug,
        )
    p.set_defaults(func=_run_register)

    # ---------- invert_transform ----------
    p = sub.add_parser(
        "invert_transform",
        help="Apply the inverse of a saved rigid/affine transform to return an image to original space.",
        formatter_class=_SmartFormatter,
    )
    p.add_argument("--original", required=True, help="Original pre-registered reference image.")
    p.add_argument("--transformed", required=True, help="Image currently in transformed space.")
    p.add_argument("--transform", required=True, help="Transform file (.tfm) to invert.")
    p.add_argument("--output", required=True, help="Output NIfTI path for inverse-transformed image.")
    p.add_argument("--interp", default="auto", help="Interpolation for inverse resampling. Accepts int 0-5 or strings like linear/cubic; default \"auto\" reuses recorded interpolation from the forward registration if available.")
    p.add_argument("--verbose", action="store_true", help="Print actions and summary.")
    def _run_inverse(a):
        inverse_transform_image(
            original_image_path=a.original,
            transformed_image_path=a.transformed,
            transform_path=a.transform,
            output_path=a.output,
            interpolation=a.interp,
            verbose=a.verbose,
        )
    p.set_defaults(func=_run_inverse)

    # ---- skullstrip (hd-bet)
    p = sub.add_parser("skullstrip", help="Run HD-BET to produce a brain mask and/or betted output volume.", formatter_class=_SmartFormatter)
    p.add_argument("--input", required=True, help="Input NIfTI to skull-strip.")
    p.add_argument("--output", help="Optional betted output path; omit to only save a mask.")
    p.add_argument("--mask", help="Optional mask output path.")
    p.add_argument("--device", default="cpu", help="Target device (e.g., cpu, cuda).")
    p.add_argument("--enable_tta", action="store_true", help="Enable HD-BET test-time augmentation (TTA). Slower; may improve masks.")
    p.add_argument("--verbose", action="store_true", help="Print HD-BET CLI output.")
    def _run_hd_bet_cli(a):
        run_hd_bet(
            input_path=a.input,
            output_path=a.output,
            mask_path=a.mask,
            device=a.device,
            disable_tta=not a.enable_tta,
            verbose=a.verbose,
        )
    p.set_defaults(func=_run_hd_bet_cli)

    # ---- math
    p = sub.add_parser(
        "math",
        help="Arithmetic/masking on MRI volumes (apply a mask, average volumes, or evaluate an expression).",
        formatter_class=_SmartFormatter,
        description=(
            "Use one of the three modes below. Each mode has its own required/optional arguments.\n"
            "\n"
            "Modes:\n"
            "  (A) Apply a mask: elementwise multiply an image by a binary mask.\n"
            "  (B) Average volumes: compute the mean of several volumes.\n"
            "  (C) Custom expression: evaluate a NumPy-like expression over inputs A..Z.\n"
        ),
        epilog=(
            "Examples:\n"
            "  # (A) Apply a mask\n"
            "  python -m astril.preprocess math --applymask --input img.nii.gz --mask brainmask.nii.gz --output img_masked.nii.gz\n"
            "\n"
            "  # (B) Average volumes\n"
            "  python -m astril.preprocess math --average subj1.nii.gz subj2.nii.gz subj3.nii.gz --output group_mean.nii.gz\n"
            "\n"
            "  # (C) Custom expression (keep A inside B, else 0)\n"
            "  python -m astril.preprocess math --operation \"where(B>0, A, 0)\" --inputs A.nii.gz B.nii.gz --output masked.nii.gz\n"
            "\n"
            "Allowed functions in --operation: where, log, log10, exp\n"
        ),
    )
    # Mode A — Apply a mask
    modeA = p.add_argument_group("Mode A — Apply a mask")
    modeA.add_argument("--applymask", action="store_true",
                       help="Enable mask application mode (A).")
    modeA.add_argument("--input",
                       help="Input NIfTI for --applymask mode (A).")
    modeA.add_argument("--mask",
                       help="Binary mask NIfTI for --applymask mode (A). Same shape as --input; >0=in mask.")

    # Mode B — Average volumes
    modeB = p.add_argument_group("Mode B — Average volumes")
    modeB.add_argument("--average", nargs="*",
                       help="One or more NIfTI files to average together (mode B).")

    # Mode C — Custom expression
    modeC = p.add_argument_group("Mode C — Custom expression")
    modeC.add_argument("--operation",
                       help="Expression over variables A..Z (mode C), e.g., 'where(B>0, A, 0)'.")
    modeC.add_argument("--inputs", nargs="*",
                       help="Input NIfTIs bound to A..Z for --operation (order matters).")

    # Common output
    p.add_argument("--output", required=True, help="Output NIfTI path (required for all modes).")
    def _run_math(a):
        # Reuse the existing, flexible argument contract
        perform_mri_math(a)
    p.set_defaults(func=_run_math)

    # ---- transform_pipeline
    p = sub.add_parser(
        "transform_pipeline",
        help="Apply or reverse a transform pipeline (json).", formatter_class=_SmartFormatter)
    p.add_argument("--input", required=True, help="Input NIfTI to transform.")
    p.add_argument("--record", required=True, help="Path to transform_record.json describing the pipeline.")
    p.add_argument("--output", required=True, help="Where to write the transformed output.")
    p.add_argument("--mode", default="apply", choices=["apply", "reverse"], help="Apply pipeline in forward or reverse order.")
    p.add_argument("--interp", type=int, default=1, help="Interpolation order for any resampling steps (0=nearest, 1=linear).")
    def _run_pipeline(a):
        apply_or_reverse_transforms(
            input_path=a.input,
            transform_record_path=a.record,
            output_path=a.output,
            mode=a.mode,
            interp=a.interp,
        )
    p.set_defaults(func=_run_pipeline)

    # ---- summarize_exam_series
    p = sub.add_parser(
        "summarize_exam_series",
        help="Infer scan types from an exam's MR/ series.", formatter_class=_SmartFormatter)
    p.add_argument("--dir", required=True, help="Path to exam directory containing the MR/ subfolder.")
    p.add_argument("--mrSubdir", default="MR", help="Name of the MR subdirectory under the exam directory (default: MR).")
    p.add_argument("--to_csv", help="Optional CSV path to write the per-series summary table.")
    p.add_argument("--quiet", action="store_true", help="Suppress printed preview; still returns DataFrame if used as a function.")
    def _run_summarize(a):
        summarize_exam_series(a.dir, mr_subdir=a.mrSubdir, to_csv=a.to_csv, verbose=not a.quiet)
    p.set_defaults(func=_run_summarize)

    # ---- create_patient_metadata
    p = sub.add_parser(
        "create_patient_metadata",
        help="Scan multi-patient DICOM dir to build a metadata table.", formatter_class=_SmartFormatter)
    p.add_argument("--dir", required=True, help="Root directory with {Patient}/.../MR/{series}")
    p.add_argument("--metadataOut", required=True, help="Output table (.csv|.tsv|.xlsx)")
    p.add_argument("--previousMetadata", nargs="*", default=[], help="Zero or more prior tables to prefill from")
    p.add_argument("--omitPrevious", action="store_true", help="Omit rows already present in previous tables")
    p.add_argument("--subdirs", nargs="+", default=["MR"], help="Subfolder names to search under each patient folder")
    p.add_argument("--excludeEmpty", action="store_true", help="Drop patient folders with no DICOM series found")
    p.add_argument("--n_workers", type=int, default=None, help="Threads for scanning (I/O bound). Set 1 to disable.")
    def _run_cpm(a):
        create_patient_metadata(
            root_dir=a.dir,
            out_path=a.metadataOut,
            previous_paths=a.previousMetadata,
            omit_previous=a.omitPrevious,
            subdirs=a.subdirs,
            exclude_empty=a.excludeEmpty,
            n_workers=a.n_workers,
        )
    p.set_defaults(func=_run_cpm)

    # ---- demix_dicoms
    p = sub.add_parser(
        "demix_dicoms",
        help="Ensure each series folder contains files from only one scan; move/copy into clean folders.", formatter_class=_SmartFormatter)
    p.add_argument("--dir", required=True, help="Root directory containing patient/exam/MR folders with DICOM (.dcm) files.")
    p.add_argument("--outDir", default=None, help="Write a fully de-mixed COPY of --dir under this path")
    p.add_argument("--logOut", default=None, help="Optional path for move log (.csv|.tsv)")
    p.add_argument("--n_workers", type=int, default=12, help="Threads for header reads/transfers (I/O bound)")
    p.add_argument("--dryRun", action="store_true", help="Plan demix and write log, but do not move/copy files")
    p.add_argument("--in_place", action="store_true", help="Allow in-place moves inside --dir (no --outDir)")
    p.add_argument("--noProgress", action="store_true", help="Disable progress bar")
    def _run_demix(a):
        demix_dicoms(
            root_dir=a.dir,
            out_dir=a.outDir,
            log_out=a.logOut,
            n_workers=a.n_workers,
            dry_run=a.dryRun,
            in_place=a.in_place,
            show_progress=not a.noProgress,
        )
    p.set_defaults(func=_run_demix)

    # ---- plan_dicom_to_nifti_conversion
    p = sub.add_parser(
        "plan_dicom_to_nifti_conversion",
        help="Discover/select DICOM series to convert and (optionally) derived products; stream a plan file.", formatter_class=_SmartFormatter)
    p.add_argument("--patientMetadata", required=True, help="Table from create_patient_metadata() (filled in)")
    p.add_argument("--dir", required=True, help="Root DICOM directory; must contain subfolders in 'Directory' column")
    p.add_argument("--outDir", required=True, help="Planned destination root for converted files")
    p.add_argument("--planOut", required=True, help="Where to write the plan (.csv|.tsv|.xlsx). .csv|.tsv files will be streamed; .xlsx files will only write after function is complete.")
    p.add_argument("--n_workers", type=int, default=None, help="Threads per exam (I/O bound)")
    p.add_argument("--noProgress", action="store_true", help="Disable progress bar")
    p.add_argument("--previousPlan", nargs="*", default=None,
                    help="0+ previous plan files to reuse/skip exam directories from.")
    p.add_argument("--ignorePrevious", action="store_true",
                    help="Skip exams already present in previous plan files (instead of reusing their rows).")
    p.add_argument("--mrSubdirs", nargs="*", default=None,
                    help="Only include these MR subfolder names (case-insensitive).")
    p.add_argument("--minSlices", type=int, default=10,
                    help="Minimum slices required to consider a sequence for selection.")
    p.add_argument(
        "--use_actual_exam_ids",
        action="store_true",
        help="Use the terminal ExamDirectory folder name as ExamAlias (may contain PHI) instead of a random 8-char alias."
    )
    p.add_argument("--add_missing_derived", action="store_true",
                   help="Identify derived scan types missing for each primary in an exam and add DERIVE jobs to the plan.")
    p.add_argument("--make_derived_from_scratch", action="store_true",
                   help="Ignore existing derived scans and plan DERIVE jobs for all supported derived types from primaries.")
    p.add_argument(
        "--unexpectedMultiframePolicy",
        choices=["keep_first", "skip"],
        default="keep_first",
        help=(
            "What to do if a sequence expected to be single-frame/3D converts to multi-frame/4D. "
            "keep_first = keep frame 0 (warn); skip = skip conversion (warn)."
        ),
    )
    def _run_plan(a):
        plan_dicom_to_nifti_conversion(
            patient_metadata=a.patientMetadata,
            root_dir=a.dir,
            out_dir=a.outDir,
            plan_out=a.planOut,
            n_workers=a.n_workers,
            show_progress=not a.noProgress,
            previous_plans=getattr(a, "previousPlan", None),
            ignore_previous=getattr(a, "ignorePrevious", False),
            include_mr_subdirs=getattr(a, "mrSubdirs", None),
            min_slices=getattr(a, "minSlices", 10),
            use_actual_exam_ids=getattr(a, "use_actual_exam_ids", False),
            add_missing_derived=getattr(a, "add_missing_derived", False),
            make_derived_from_scratch=getattr(a, "make_derived_from_scratch", False),
            unexpected_multiframe_policy=getattr(a, "unexpectedMultiframePolicy", "keep_first"),
        )
    p.set_defaults(func=_run_plan)

    # ---- convert_dicom_to_nifti (single series)
    p = sub.add_parser(
        "convert_dicom_to_nifti",
        help="Convert one DICOM series directory to NIfTI (dicom2nifti).", formatter_class=_SmartFormatter)
    p.add_argument("--dicom_dir", required=True, help="Directory containing one DICOM series")
    p.add_argument("--output_path", required=True, help="Output NIfTI path (.nii or .nii.gz)")
    p.add_argument("--debug", action="store_true", help="Print debug statements.")
    def _run_c2n(a):
        convert_dicom_to_nifti(
            dicom_series_dir=a.dicom_dir,
            output_path=a.output_path,
            debug=a.debug,
        )
        print(f"Saved: {a.output_path}")
    p.set_defaults(func=_run_c2n)

    # ---- convert_dicom_plan (batch from plan file)
    p = sub.add_parser(
        "convert_dicom_plan",
        help="Execute DICOM→NIfTI conversions from a saved plan file.", formatter_class=_SmartFormatter)
    p.add_argument("--plan", required=True, help="Path to plan CSV/TSV/XLSX from plan_dicom_to_nifti_conversion.")
    p.add_argument("--n_workers", type=int, default=None, help="Parallel workers (I/O-bound).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output NIfTI if present.")
    p.add_argument("--logOut", default=None, help="Optional CSV/TSV log path for results.")
    p.add_argument(
        "--unexpectedMultiframePolicy",
        choices=["keep_first", "skip"],
        default="keep_first",
        help=(
            "What to do if a sequence expected to be single-frame/3D converts to multi-frame/4D. "
            "keep_first = keep frame 0 (warn); skip = skip conversion (warn)."
        ),
    )
    p.add_argument("--debug", action="store_true", help="Print debug statements.")
    def _run_convert(a):
        convert_dicom_plan(
            plan_path=a.plan,
            n_workers=a.n_workers,
            overwrite=a.overwrite,
            log_out=a.logOut,
            show_progress=True,
            unexpected_multiframe_policy=getattr(a, "unexpectedMultiframePolicy", "keep_first"),
            debug=a.debug,
        )
    p.set_defaults(func=_run_convert)

    # ---- generate_preprocessing_qc_pdfs
    p = sub.add_parser(
        "generate_preprocessing_qc_pdfs",
        help="Generate per-patient QC PDF(s) from preprocessed NIfTI volumes.",
        formatter_class=_SmartFormatter,
    )
    p.add_argument("--dir", required=True, help="Root directory of preprocessed MRIs: {dir}/{patient_dirs}/{exam_dirs}.")
    p.add_argument("--n_workers", type=int, default=None, help="Parallel workers (per-patient).")
    p.add_argument("--outDir", default=None, help="Optional output directory for PDFs (default: patient_dir).")
    p.add_argument("--maxExamsPerPage", type=int, default=4, help="Max exams per PDF page (landscape).")
    p.add_argument("--leftMarginScale", type=float, default=2.25, help="Scale factor for left margin used by row labels.")
    p.add_argument("--noProgress", action="store_true", help="Disable progress bars.")

    def _run_qc(a):
        generate_preprocessing_qc_pdfs(
            root_dir=a.dir,
            n_workers=a.n_workers,
            out_dir=a.outDir,
            show_progress=not a.noProgress,
            max_exams_per_page=a.maxExamsPerPage,
            left_margin_scale=a.leftMarginScale,
        )

    p.set_defaults(func=_run_qc)



    return parser


def main():
    parser = _build_cli_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    # Dispatch
    return args.func(args)

if __name__ == "__main__":
    main()
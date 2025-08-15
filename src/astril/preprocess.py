# preprocessing_functions.py
# Author: Alex Ling
# E. Antonio Chiocca Group, BWH
# Description: Preprocessing utilities for MRI normalization, resampling, etc.

import nibabel as nib
import numpy as np
import pandas as pd
import os
import sys
import argparse
import SimpleITK as sitk
import shutil
import subprocess
import re
import ast
import json
import tempfile
import warnings
import uuid
import string
import csv as _csv
from datetime import datetime
from nilearn.image import resample_to_img
from pathlib import Path
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import zoom
from typing import List, Dict
from .preprocessing_utils import (
    apply_padding,
    interpolate_to_voxel_dims,
    update_origin_for_padding,
    adjust_to_target_shape,
    read_padding_record,
    load_roi_mask,
    ensure_hd_bet_installed,
    classify_exam_series,
    _first_dicom_in,
    _safe_dcmread,
    _get_attr,
    _clean_lower,
    _read_table,
    _save_table,
    _progress,
    _normalize_patient_name,
    _safe_int_like,
    _propose_series_dirname,
    _avoid_name_collision,
    _files_identical
)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None # Optional progress bar (graceful fallback if not installed)

# -----------------------------------------------------------------
# Function to normalize an MRI image using only the masked region
# -----------------------------------------------------------------

def normalize_masked_image(input_image_path, mask_path, output_path=None):
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

    normalized_data = np.where(mask_data > 0, (data - mean) / std, 0)

    normalized_img = nib.Nifti1Image(normalized_data, affine=img.affine, header=img.header)

    if output_path:
        nib.save(normalized_img, output_path)

    return normalized_img


# -------------------------------------------------------------------------
# Function to reshape an MRI volume to specified data and voxel dimensions
# -------------------------------------------------------------------------

def resize_mri(input_filepath, output_filepath, target_shape, target_voxel_dims, interp,
               save_padding_record=False, padding_record_path=None,
               roi_mask_path=None, translation_only=False):

    if not os.path.exists(input_filepath):
        raise ValueError(f"[Error] Attempting to resize {input_filepath}, but file does not exist.")

    mri = nib.load(input_filepath)
    data = mri.get_fdata()
    original_voxel_dims = mri.header.get_zooms()

    padding_record = {
        'target_voxel_dims': target_voxel_dims,
        'target_shape': target_shape,
        'original_voxel_dims': original_voxel_dims,
        'original_shape': data.shape,
        'original_grid': {
            'size': list(data.shape),
            'spacing': list(original_voxel_dims),
            'origin': list(mri.affine[:3, 3]),
            'direction': list(np.ravel(mri.affine[:3, :3] / np.array(original_voxel_dims)))
        }
    }

    loaded_padding_record = None
    if padding_record_path and os.path.exists(padding_record_path):
        loaded_padding_record = read_padding_record(padding_record_path)

    roi_mask = load_roi_mask(roi_mask_path, data.shape) if roi_mask_path else None

    if roi_mask is not None or loaded_padding_record:
        if not loaded_padding_record:
            roi_indices = np.where(roi_mask > 0)
            roi_center = (np.min(roi_indices, axis=1) + np.max(roi_indices, axis=1)) // 2
            data_center = np.array(data.shape) // 2
            translation = data_center - roi_center
            center_padding = np.zeros((3, 2), dtype=int)
            for dim, shift in enumerate(translation):
                center_padding[dim] = [shift, -shift]
        else:
            center_padding = loaded_padding_record['center_padding']
        data = apply_padding(data, center_padding)
    else:
        center_padding = np.zeros((3, 2), dtype=int)

    padding_record['center_padding'] = center_padding

    if translation_only:
        padding_record['shape_padding'] = np.zeros((3, 2), dtype=int)
        final_data = data
        new_affine = mri.affine
    else:
        interpolated = interpolate_to_voxel_dims(data, original_voxel_dims, target_voxel_dims, interp)
        if loaded_padding_record:
            final_data, padding_record = adjust_to_target_shape(
                interpolated, target_shape, padding_record, loaded_padding_record['shape_padding']
            )
        else:
            final_data, padding_record = adjust_to_target_shape(interpolated, target_shape, padding_record)
        new_affine = mri.affine.copy()
        new_affine[:3, :3] = np.diag(np.sign(np.diag(new_affine[:3, :3])) * np.array(target_voxel_dims))
        new_affine = update_origin_for_padding(new_affine, padding_record['shape_padding'], target_voxel_dims)

    nib.save(nib.Nifti1Image(final_data.astype(np.float32), new_affine), output_filepath)

    if save_padding_record:
        path_to_save = padding_record_path or f"{output_filepath}_padding.txt"
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        with open(path_to_save, 'w') as f:
            f.write(str(padding_record))

# -----------------------------------------------------------------------------------
# Function to undo reshape of MRI volume using saved padding record from resize_mri()
# -----------------------------------------------------------------------------------

def reverse_resize_mri(input_filepath, output_filepath, padding_record_path, interp=1):
    """
    Reverse a resizing operation performed by resize_mri(), using the original
    spacing and padding information stored in a padding record file.

    Args:
        input_filepath (str): Path to the resized image (.nii.gz)
        output_filepath (str): Path where the reversed (original space) image should be saved
        padding_record_path (str): Path to the .txt file storing the resize/padding metadata
        interp (int): Interpolation order for resampling (0 = nearest, 1 = linear, etc.)
    """
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom

    if not os.path.exists(padding_record_path):
        raise FileNotFoundError(f"Padding record not found: {padding_record_path}")

    padding_record = read_padding_record(padding_record_path)

    img = nib.load(input_filepath)
    data = img.get_fdata()
    current_voxel_dims = img.header.get_zooms()
    original_voxel_dims = np.array(padding_record['original_voxel_dims'])

    # Step 1: Resize back to original voxel spacing
    zoom_factors = np.array(current_voxel_dims) / original_voxel_dims
    resampled = zoom(data, zoom_factors, order=interp)

    # Step 2: Undo shape padding (crop or pad)
    shape_padding = np.array(padding_record['shape_padding'])
    adjusted = resampled
    for axis in range(3):
        before, after = shape_padding[axis]
        if before > 0 or after > 0:
            adjusted = np.take(adjusted, indices=range(before, adjusted.shape[axis] - after), axis=axis)
        elif before < 0 or after < 0:
            pad_width = [(0, 0)] * 3
            pad_width[axis] = (-before if before < 0 else 0, -after if after < 0 else 0)
            adjusted = np.pad(adjusted, pad_width, mode='constant', constant_values=0)

    # Step 3: Undo center padding (crop or pad)
    center_padding = np.array(padding_record['center_padding'])
    final = adjusted
    for axis in range(3):
        before, after = center_padding[axis]
        if before > 0 or after > 0:
            final = np.take(final, indices=range(before, final.shape[axis] - after), axis=axis)
        elif before < 0 or after < 0:
            pad_width = [(0, 0)] * 3
            pad_width[axis] = (-before if before < 0 else 0, -after if after < 0 else 0)
            final = np.pad(final, pad_width, mode='constant', constant_values=0)

    # Step 4: Restore original affine
    original_affine = np.eye(4)
    original_affine[:3, 3] = padding_record['original_grid']['origin']

    direction_matrix = np.reshape(padding_record['original_grid']['direction'], (3, 3))
    voxel_dims = np.array(padding_record['original_voxel_dims'])

    # Apply voxel size along the correct axis (columns)
    scaled_direction = direction_matrix * voxel_dims[np.newaxis, :]  # shape (3, 3)
    original_affine[:3, :3] = scaled_direction

    nib.save(nib.Nifti1Image(final.astype(np.float32), original_affine), output_filepath)
    print(f"[Done] Reversed resize saved to: {output_filepath}")


# -------------------------------------------------------------------------
# Function to match affine matrices between two nifti files
# -------------------------------------------------------------------------

def match_direction_matrices(input_path, donor_path, output_path):
    """
    Resample an input NIfTI image to match the affine direction matrix and shape of a donor image.

    Args:
        input_path (str): Path to input NIfTI image
        donor_path (str): Path to donor NIfTI image
        output_path (str): Path to save the matched image
    """
    donor_img = nib.load(donor_path)
    input_img = nib.load(input_path)

    # Resample to match donor using nearest neighbor (default for labels, safe fallback for others)
    resampled_img = resample_to_img(input_img, donor_img, interpolation='nearest')

    # Preserve data type from original image
    resampled_data = resampled_img.get_fdata().astype(input_img.get_data_dtype())

    # Create a new image with the donor's affine, preserving header information
    header = input_img.header.copy()
    header.set_qform(donor_img.affine, code=1)
    header.set_sform(donor_img.affine, code=1)

    output_img = nib.Nifti1Image(resampled_data, affine=donor_img.affine, header=header)
    nib.save(output_img, output_path)

# -------------------------------------------------------------------------
# Function to merge mask files into a single mask
# -------------------------------------------------------------------------

def merge_binary_masks(mask_paths, output_path, fill_holes=True, strict_affine=False):
    """
    Merge multiple binary masks (NIfTI format) into one.
    Voxels are 1 if any input mask has a 1 at that position.

    Args:
        mask_paths (list of str): Paths to input NIfTI mask files
        output_path (str): Path to save the merged mask
        fill_holes (bool): Whether to apply hole filling
        strict_affine (bool): If True, check that affines match exactly
    """
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
    similarity_metric="correlation",
    use_gpu=False,
    verbose=True,
    save_dummy_ref=False
):
    """
    Register or apply transform to align moving image to fixed image using SimpleITK.

    Args:
        fixed_path (str): Path to the fixed image (reference).
        moving_path (str): Path to the moving image (to be registered or transformed).
        output_path (str): Path to save the output image.
        transform_path (str): Path to save or load transform.
        apply_only (bool): If True, apply existing transform instead of performing registration.
        registration_type (str): One of "rigid", "affine", or "translation".
        similarity_metric (str): "correlation" or "mi" (mutual information).
        use_gpu (bool): Use GPU acceleration if supported.
        save_dummy_ref (bool): Whether to save a zeroed copy of the moving image as a deidentified, space-efficient way to keep a reference for later re-application or reversal of the transformation.
        verbose (bool): Whether to print metric score and status.
    """
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    if apply_only:
        if not transform_path or not os.path.isfile(transform_path):
            raise ValueError("Transform file is required and must exist when apply_only=True.")
        transform = sitk.ReadTransform(transform_path)
        if verbose:
            print(f"Applying transform from: {transform_path}")
    else:
        # Select transform type
        if registration_type == "rigid":
            tx = sitk.Euler3DTransform()
        elif registration_type == "affine":
            tx = sitk.AffineTransform(3)
        elif registration_type == "translation":
            tx = sitk.TranslationTransform(3)
        else:
            raise ValueError("Invalid registration_type. Choose 'rigid', 'affine', or 'translation'.")

        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, tx, sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        registration = sitk.ImageRegistrationMethod()
        registration.SetInitialTransform(initial_transform, inPlace=False)

        # Metric
        if similarity_metric == "correlation":
            registration.SetMetricAsCorrelation()
        elif similarity_metric == "mi":
            registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        else:
            raise ValueError("Invalid similarity_metric. Choose 'correlation' or 'mi'.")

        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0, minStep=1e-4, numberOfIterations=200,
            gradientMagnitudeTolerance=1e-6
        )
        registration.SetOptimizerScalesFromPhysicalShift()

        if use_gpu:
            try:
                registration.SetMetricSamplingStrategy(registration.RANDOM)
                registration.SetMetricSamplingPercentage(0.2)
                if verbose:
                    print("Using GPU-style fast approximation (random sampling).")
            except Exception as e:
                if verbose:
                    print(f"GPU acceleration setup failed: {e}")

        transform = registration.Execute(fixed, moving)

        if verbose:
            final_metric = registration.GetMetricValue()
            print(f"Final {similarity_metric} = {final_metric:.4f}")

        if transform_path:
            sitk.WriteTransform(transform, transform_path)

    # Resample the moving image using the transform
    registered = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    sitk.WriteImage(registered, output_path)

    if save_dummy_ref and transform_path:
        base = os.path.splitext(transform_path)[0]
        fixed_dummy_path = base + "_fixed_ref.nii.gz"
        moving_dummy_path = base + "_moving_ref.nii.gz"

        for ref_img, path in [(fixed, fixed_dummy_path), (moving, moving_dummy_path)]:
            zero_array = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.float32)
            dummy = sitk.GetImageFromArray(zero_array)
            dummy.CopyInformation(ref_img)
            sitk.WriteImage(dummy, path)
            if verbose:
                print(f"Dummy reference saved to: {path}")

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
):
    """
    Apply the inverse of a saved transform to return an image to its original space.

    Args:
        original_image_path (str): Path to the original (pre-registered) image (reference grid).
        transformed_image_path (str): Path to the image that has been transformed.
        transform_path (str): Path to the saved transform (.tfm).
        output_path (str): Path to save the inverse-transformed image.
        interpolation (str): One of 'linear' or 'nearest'.
        verbose (bool): Print actions and summary.
    """
    original_img = sitk.ReadImage(original_image_path, sitk.sitkFloat32)
    transformed_img = sitk.ReadImage(transformed_image_path, sitk.sitkFloat32)
    transform = sitk.ReadTransform(transform_path)

    if not transform.IsLinear():
        raise ValueError("Transform is not linear (rigid/affine). Inverse may not be supported.")

    inverse_transform = transform.GetInverse()

    if interpolation == "linear":
        interp_method = sitk.sitkLinear
    elif interpolation == "nearest":
        interp_method = sitk.sitkNearestNeighbor
    else:
        raise ValueError("Unsupported interpolation type.")

    recovered = sitk.Resample(
        transformed_img,
        original_img,
        inverse_transform,
        interp_method,
        0.0,
        transformed_img.GetPixelID()
    )
    sitk.WriteImage(recovered, output_path)

    if verbose:
        print(f"Inverse-transformed image saved to: {output_path}")


# ---------------------------------------------------------------------------------------
# Function to run hd-bet for brainmask creation
# ---------------------------------------------------------------------------------------

def run_hd_bet(input_path, output_path=None, mask_path=None, mode="accurate", device="cpu", tta=0, pp=1, overwrite_existing=0):
    ensure_hd_bet_installed()

    if not output_path and not mask_path:
        raise ValueError("Must provide at least --output or --mask path.")

    bet_flag = 1 if output_path else 0
    save_mask_flag = 1 if mask_path else 0

    # Generate safe dummy output if only mask is being saved
    use_dummy_output = False
    if output_path:
        hd_bet_output = output_path
    elif mask_path:
        dummy_base = os.path.splitext(os.path.basename(input_path))[0]
        dummy_output = f"{dummy_base}_dummy_{uuid.uuid4().hex[:8]}.nii.gz"
        hd_bet_output = os.path.join(os.path.dirname(mask_path), dummy_output)
        use_dummy_output = True
    else:
        raise RuntimeError("Unexpected logic error in determining output path.")

    cmd = [
        "hd-bet",
        "-i", input_path,
        "-o", hd_bet_output,
        "-mode", mode,
        "-device", str(device),
        "-tta", str(tta),
        "-pp", str(pp),
        "--overwrite_existing", str(overwrite_existing),
        "--bet", str(bet_flag),
        "--save_mask", str(save_mask_flag),
    ]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if mask_path:
        expected_mask = hd_bet_output.replace(".nii.gz", "_mask.nii.gz")
        if not os.path.exists(expected_mask):
            raise FileNotFoundError(f"Expected mask file not found: {expected_mask}")
        os.replace(expected_mask, mask_path)

    if use_dummy_output and os.path.exists(hd_bet_output):
        os.remove(hd_bet_output)


# ---------------------------------------------------------------------------------------
# Function to do basic math between MRI volumes
# ---------------------------------------------------------------------------------------

def perform_mri_math(args):
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
    interp=1
):
    """
    Apply or reverse a transform pipeline defined in a transform_record.json.

    Args:
        input_path (str): Path to the scan to transform.
        transform_record_path (str): Path to the transform_record.json file.
        output_path (str): Where to write the transformed result.
        mode (str): "apply" or "reverse".
        interp (int): Interpolation order (0=nearest, 1=linear).
    """
    assert mode in ["apply", "reverse"], "mode must be 'apply' or 'reverse'"

    base_dir = os.path.dirname(os.path.abspath(transform_record_path))

    # Load the record
    with open(transform_record_path, 'r') as f:
        record = json.load(f)

    steps = list(record.items())
    if mode == "reverse":
        steps = list(reversed(steps))

    temp_file = input_path
    temp_files = []

    for step_name, record_entry in steps:
        if isinstance(record_entry, dict):
            tfm_path = os.path.normpath(os.path.join(base_dir, record_entry["transform"]))
            if mode == "apply":
                ref_path = os.path.normpath(os.path.join(base_dir, record_entry.get("fixed_reference", "")))
            else:
                ref_path = os.path.normpath(os.path.join(base_dir, record_entry.get("moving_reference", "")))
        else:
            tfm_path = os.path.normpath(os.path.join(base_dir, record_entry))
            ref_path = None

        if tfm_path.endswith(".tfm"):
            intermediate = tempfile.mktemp(suffix=".nii.gz")
            if not ref_path or not os.path.exists(ref_path):
                raise RuntimeError(f"[Error] Reference image not found for transform: {tfm_path}")

            if mode == "apply":
                register_images(
                    fixed_path=ref_path,
                    moving_path=temp_file,
                    output_path=intermediate,
                    transform_path=tfm_path,
                    apply_only=True,
                    verbose=False
                )
            else:
                inverse_transform_image(
                    original_image_path=ref_path,
                    transformed_image_path=temp_file,
                    transform_path=tfm_path,
                    output_path=intermediate,
                    interpolation="linear",
                    verbose=False
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
                    interp=interp,
                    save_padding_record=False,
                    padding_record_path=tfm_path,
                    translation_only=False
                )
            else:
                reverse_resize_mri(
                    input_filepath=temp_file,
                    output_filepath=intermediate,
                    padding_record_path=tfm_path,
                    interp=interp
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
def create_patient_metadata(root_dir: str, out_path: str, previous_paths=None,                              #Should consider adding multithread support to this in the future.
                            omit_previous: bool = False, show_progress: bool = True,
                            subdirs: list[str] | None = None, exclude_empty: bool = False) -> pd.DataFrame:
    """
    Build a per-patient metadata table by scanning {root_dir}/{Patient_folder}/.../MR/{series}.
    Columns:
      - Directory (relative to root_dir)
      - patientID (user-assigned; blank unless prefilled from previous tables)
      - patientName (lowercased unique names from DICOM)
      - dicomPatientID (lowercased unique IDs from DICOM)
      - day0Date (blank unless prefilled from previous tables)
    """
    if previous_paths is None:
        previous_paths = []
    # which subfolders under each patient folder to search
    subdirs = subdirs or ["MR"]
    subdir_set = {s.strip(os.sep) for s in subdirs if s}
    root_dir = os.path.abspath(root_dir)
    rows = []

    # Gather top-level patient folders
    patient_folders = sorted([d.path for d in os.scandir(root_dir) if d.is_dir()])

    # Inform user if tqdm is unavailable but progress was requested
    if show_progress and tqdm is None:
        print("[Info] tqdm not installed; proceeding without a progress bar. Run `pip install tqdm` to enable it.")

    # Top-level progress (per patient folder)
    for pf in _progress(patient_folders, total=len(patient_folders), desc="Scanning patients", unit="patient", enable=show_progress):
        rel_dir = os.path.relpath(pf, root_dir)
        names = set()
        names_norm = set()
        names_raw  = set()  # (kept only to improve normalization if ever needed)
        ids = set()
        found_any_dicom = False

        # Walk to find any target subdir(s) (default 'MR'), then take immediate subfolders as series
        for walk_root, dirnames, _ in os.walk(pf):
            base = os.path.basename(walk_root)
            if base not in subdir_set:
                continue
            series_dirs = sorted([os.path.join(walk_root, d) for d in dirnames if os.path.isdir(os.path.join(walk_root, d))])
            for sdir in series_dirs:
                dcm_path = _first_dicom_in(sdir)
                if not dcm_path:
                    continue
                found_any_dicom = True
                ds = _safe_dcmread(dcm_path)
                if ds is None:
                    continue
                # normalize name: keep apostrophes, replace all other punctuation with spaces, collapse spaces, lowercase
                pname_raw = _get_attr(ds, "PatientName")
                pname = _normalize_patient_name(pname_raw)
                pid = _clean_lower(_get_attr(ds, "PatientID"))
                if pname:      names_norm.add(pname)
                if pname_raw:  names_raw.add(pname_raw)
                if pid:
                    ids.add(pid)
        # optionally exclude patients with no DICOMs discovered under the chosen subdirs
        if exclude_empty and not found_any_dicom:
            continue


        rows.append(dict(
            Directory=rel_dir,
            patientID="",  # user-assigned; prefilled from previous if provided
            patientName="; ".join(sorted(names_norm)) if names_norm else "",
            dicomPatientID="; ".join(sorted(ids)) if ids else "",
            day0Date="",
        ))

    df = pd.DataFrame(rows, columns=["Directory", "patientID", "patientName", "dicomPatientID", "day0Date"])

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
                lookup_list.append(sub.set_index("Directory"))
            if lookup_list:
                # Combined (ordered) lookup — first non-missing wins
                combined = pd.concat(lookup_list, axis=1, join="outer", keys=range(len(lookup_list)))
                # Flatten to first non-empty per column
                for col in ("patientID", "day0Date"):
                    cols = [c for c in combined.columns if c[1] == col]
                    combined[(0, f"__first_nonempty_{col}")] = combined[cols].bfill(axis=1).iloc[:, 0]
                flat = combined[[ (0, "__first_nonempty_patientID"), (0, "__first_nonempty_day0Date") ]]
                flat.columns = ["_prefill_patientID", "_prefill_day0Date"]
                flat = flat.reset_index()  # Directory becomes a column
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
def demix_dicoms(root_dir: str, show_progress: bool = True, log_out: str | None = None, out_dir: str | None = None,  dry_run: bool = False) -> None:
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
    """
    root_dir = os.path.abspath(root_dir)
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

    _logged_rows = 0
    def _stream_log_row(mr_rel: str, src: str, dst: str, s_uid, s_no, s_desc, s_proto):
        nonlocal _logged_rows
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
    total_parents = len(parents)

    for mr_parent in _progress(parents, total=total_parents, desc="Demixing MR folders", unit="MR", enable=show_progress):
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

        # Build grouping key for each DICOM
        entries = []  # (path, group_key, series_number, series_desc, proto_name, series_uid)
        for p in all_dicoms:
            ds = _safe_dcmread(p)
            if ds is None:
                # keep file, use a coarse fallback key to avoid dropping it
                entries.append((p, ("__UNKNOWN__", None, None, None), None, None, None, None))
                continue
            series_uid  = _get_attr(ds, "SeriesInstanceUID") or None
            series_no   = _safe_int_like(_get_attr(ds, "SeriesNumber"))
            series_desc = _get_attr(ds, "SeriesDescription") or ""
            proto_name  = _get_attr(ds, "ProtocolName") or ""

            if series_uid:
                key = ("UID", series_uid)  # strongest key
            else:
                key = ("FALLBACK",
                       series_no if series_no is not None else -1,
                       series_desc.strip().lower(),
                       proto_name.strip().lower())
            entries.append((p, key, series_no, series_desc, proto_name, series_uid))

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
                        target2 = os.path.join(mr_parent, alt)
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

        # Transfer files to their target (move in-place; copy when out_dir is set)
        for rec in _progress(entries, total=len(entries), desc=mr_rel, unit="file", enable=show_progress):
            p, key, s_no, s_desc, s_proto, s_uid = rec
            tgt_dir = group_to_target[key]
            curr_dir = os.path.dirname(p)
            # Determine if this file is "misplaced" (different subfolder name than target)
            changed_subdir = (os.path.basename(curr_dir) != os.path.basename(tgt_dir))

            # If in-place mode AND already in correct dir, skip I/O; still no log entry (not misplaced)
            if out_dir is None and os.path.normpath(curr_dir) == os.path.normpath(tgt_dir):
                continue
            # Build destination path
            dst = os.path.join(tgt_dir, os.path.basename(p))
            # SAFETY: if destination exists, verify identical before skipping
            if os.path.exists(dst):
                if _files_identical(p, dst):
                    print(f"[demix_dicoms][info] identical exists, skipping: {p} == {dst}")
                    # Log only if it would have been “misplaced”; still no transfer.
                    if changed_subdir and dry_run:
                        _stream_log_row(mr_rel, p, dst, s_uid, s_no, s_desc, s_proto)
                    continue
                else:
                    print(f"[demix_dicoms][WARN] conflict: destination exists with different content, skipping move: {p} -> {dst}")
                    # Also reflect intended move in dry_run logs if it was misplaced
                    if changed_subdir and dry_run:
                        _stream_log_row(mr_rel, p, dst, s_uid, s_no, s_desc, s_proto)
                    continue
            # Log if “misplaced” (i.e. different subfolder) — both dry_run and real run
            if changed_subdir:
                _stream_log_row(mr_rel, p, dst, s_uid, s_no, s_desc, s_proto)

            # Execute transfer unless dry_run
            if dry_run:
                continue
            try:
                if out_dir is None:
                    os.replace(p, dst)        # in-place demix → move
                else:
                    import shutil
                    shutil.copy2(p, dst)      # to out_dir → copy (preserve original)
                total_moves += 1
            except Exception as e:
                print(f"[demix_dicoms][warn] failed to transfer {p} -> {dst}: {e}")

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


# ------------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MRI Preprocessing Tools",
        usage=(
            "python -m astril.preprocessing_functions <command> [<args>]\n\n"
            "Available commands:\n"
            "  normalize                Normalize an MRI volume using a mask\n"
            "  resize                   Resize an MRI volume to target dimensions and voxel spacing\n"
            "  reverse_resize           Reverse a resize operation using padding record\n"
            "  match_affine             Match affine direction matrix of input to donor image\n"
            "  merge_masks              Merge 2+ binary masks (logical OR, optional fill holes)\n"
            "  register                 Register an MRI volume to match the position/spacing of a reference volume\n"
            "  inverse_transform        Apply inverse of a transform to return image to original space\n"
            "  skullstrip               Perform skullstripping on a T1c, T1n, T2f, or T2w volume using hd-bet\n"
            "  math                     Perform arithmetic or masking operations on MRI volumes\n"
            "  transform_pipeline       Apply or reverse full transform pipeline using saved record\n"
            "  summarize_exam_series    Infer scan types from DICOM series metadata in MR/ folders\n"
            "  create_patient_metadata  Create patient metadata table from multi-patient DICOM directory\n"
            "  demix_dicoms             Looks through a directory to ensure no subdirectories contain .dcm files from more than one scan\n"
        )
    )
    parser.add_argument("command", help="Subcommand to run (e.g., normalize)")

    args = parser.parse_args(sys.argv[1:2])
    if not hasattr(sys.modules[__name__], f"cli_{args.command}"):
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    # Call the appropriate CLI handler
    getattr(sys.modules[__name__], f"cli_{args.command}")()


def cli_normalize():
    parser = argparse.ArgumentParser(description="Normalize an MRI volume using a binary mask.")
    parser.add_argument("--input", required=True, help="Path to input NIfTI image")
    parser.add_argument("--mask", required=True, help="Path to binary brain mask")
    parser.add_argument("--output", required=True, help="Path to save the normalized image")

    args = parser.parse_args(sys.argv[2:])
    normalize_masked_image(args.input, args.mask, args.output)
    #print(f"Normalized image saved to: {args.output}")

def cli_resize():
    parser = argparse.ArgumentParser(description="Resize MRI scan to target shape and voxel dims.")
    parser.add_argument("--input", required=True, help="Input NIfTI image")
    parser.add_argument("--output", required=True, help="Output NIfTI path")
    parser.add_argument("--data_dims", default="240,240,155", help="Target data dimensions (e.g., 240,240,155)")
    parser.add_argument("--voxel_dims", default="1.0,1.0,1.0", help="Target voxel dims (e.g., 1.0,1.0,1.0)")
    parser.add_argument("--interp", type=int, default=1, help="Interpolation order (0=nearest, 1=linear, etc.)")
    parser.add_argument("--save_padding_record", action="store_true", help="Save padding record")
    parser.add_argument("--padding_record", help="Use saved padding record instead of recalculating")
    parser.add_argument("--roimask", help="ROI mask for centering (ignored if padding_record is used)")
    parser.add_argument("--translation_only", action="store_true", help="Only center without resampling")

    args = parser.parse_args(sys.argv[2:])
    resize_mri(
        input_filepath=args.input,
        output_filepath=args.output,
        target_shape=tuple(map(int, args.data_dims.split(','))),
        target_voxel_dims=tuple(map(float, args.voxel_dims.split(','))),
        interp=args.interp,
        save_padding_record=args.save_padding_record,
        padding_record_path=args.padding_record,
        roi_mask_path=args.roimask,
        translation_only=args.translation_only
    )
    #print(f"Resized image saved to: {args.output}")

def cli_reverse_resize():
    parser = argparse.ArgumentParser(description="Reverse resize using a saved padding record.")
    parser.add_argument("--input", required=True, help="Path to the resized input image (.nii.gz)")
    parser.add_argument("--output", required=True, help="Path to save the reversed (original) image")
    parser.add_argument("--padding_record", required=True, help="Path to the saved padding record file (.txt)")
    parser.add_argument("--interp", type=int, default=1, help="Interpolation order (0=nearest, 1=linear, etc.)")

    args = parser.parse_args(sys.argv[2:])
    reverse_resize_mri(
        input_filepath=args.input,
        output_filepath=args.output,
        padding_record_path=args.padding_record,
        interp=args.interp
    )

def cli_match_affine():
    parser = argparse.ArgumentParser(description="Match direction matrix (affine) of input image to donor image.")
    parser.add_argument("--input", required=True, help="Path to input NIfTI image to be resampled")
    parser.add_argument("--donor", required=True, help="Path to donor NIfTI image (defines target affine/orientation)")
    parser.add_argument("--output", required=True, help="Path to save output image")

    args = parser.parse_args(sys.argv[2:])
    match_direction_matrices(args.input, args.donor, args.output)
    #print(f"Affine-matched image saved to: {args.output}")

def cli_merge_masks():
    parser = argparse.ArgumentParser(description="Merge 2+ binary brain masks into a single mask (logical OR).")
    parser.add_argument("--masks", nargs='+', required=True, help="Paths to input NIfTI mask files")
    parser.add_argument("--output", required=True, help="Output file path (.nii.gz)")
    parser.add_argument("--no_fill", action="store_true", help="Disable hole filling")
    parser.add_argument("--strict_affine", action="store_true", help="Require affines to match exactly")

    args = parser.parse_args(sys.argv[2:])
    merge_binary_masks(
        mask_paths=args.masks,
        output_path=args.output,
        fill_holes=not args.no_fill,
        strict_affine=args.strict_affine
    )
    #print(f"Merged mask saved to: {args.output}")

def cli_register():
    parser = argparse.ArgumentParser(description="Register or transform MRI using SimpleITK.")
    parser.add_argument("--fixed", required=True, help="Path to fixed (reference) image")
    parser.add_argument("--moving", required=True, help="Path to moving image")
    parser.add_argument("--output", required=True, help="Path to save output registered image")
    parser.add_argument("--transform", help="Path to save or load transform (.tfm)")
    parser.add_argument("--apply_only", action="store_true", help="Apply existing transform only")
    parser.add_argument("--type", default="rigid", choices=["rigid", "affine", "translation"], help="Registration type (rigid, affine, translation)")
    parser.add_argument("--metric", default="correlation", choices=["correlation", "mi"], help="Similarity metric: correlation or mutual information")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU-style random sampling if available")
    parser.add_argument("--save_dummy_ref", action="store_true", help="Save a blanked copy of the moving image to establish original spacing, etc. if you ever want to reverse the registration.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args(sys.argv[2:])
    register_images(
        fixed_path=args.fixed,
        moving_path=args.moving,
        output_path=args.output,
        transform_path=args.transform,
        apply_only=args.apply_only,
        registration_type=args.type,
        similarity_metric=args.metric,
        use_gpu=args.use_gpu,
        save_dummy_ref=args.save_dummy_ref,
        verbose=not args.quiet
    )

def cli_inverse_transform():
    parser = argparse.ArgumentParser(description="Apply inverse of saved transform to restore image to original space.")
    parser.add_argument("--original", required=True, help="Original (pre-registered) image to restore spacing/grid")
    parser.add_argument("--transformed", required=True, help="Registered/transformed image to be inverse-transformed")
    parser.add_argument("--transform", required=True, help="Path to transform file (.tfm) to invert")
    parser.add_argument("--output", required=True, help="Output path for inverse-transformed image")
    parser.add_argument("--interp", default="linear", choices=["linear", "nearest"], help="Interpolation method")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args(sys.argv[2:])
    inverse_transform_image(
        original_image_path=args.original,
        transformed_image_path=args.transformed,
        transform_path=args.transform,
        output_path=args.output,
        interpolation=args.interp,
        verbose=not args.quiet
    )

def cli_skullstrip():
    parser = argparse.ArgumentParser(description="Run HD-BET skull stripping.")
    parser.add_argument("--input", required=True, help="Path to the input image (NIfTI)")
    parser.add_argument("--output", required=False, help="Path to save the stripped brain image")
    parser.add_argument("--mask", required=False, help="Path to save the brain mask image")
    parser.add_argument("--mode", default="accurate", choices=["accurate", "fast"], help="HD-BET mode")
    parser.add_argument("--device", default="cpu", help="Device: integer (GPU ID), 'cpu', or 'mps'")
    parser.add_argument("--tta", default=0, type=int, help="Test-time augmentation (1=True, 0=False)")
    parser.add_argument("--pp", default=1, type=int, help="Postprocessing (1=True, 0=False)")
    parser.add_argument("--overwrite_existing", default=0, type=int, help="Allow overwrite of output files")
    
    args = parser.parse_args(sys.argv[2:])
    run_hd_bet(
        input_path=args.input,
        output_path=args.output,
        mask_path=args.mask,
        mode=args.mode,
        device=args.device,
        tta=args.tta,
        pp=args.pp,
        overwrite_existing=args.overwrite_existing,
    )

def cli_math():
    parser = argparse.ArgumentParser(description="MRI volume math: expressions, masks, averages")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--operation",
        help=(
            "Math expression using variables A, B, C, etc. for inputs. "
            "Supports +, -, *, /, **, (), where(), log(), log10(), exp()."
        )
    )
    group.add_argument("--applymask", action="store_true", help="Apply a binary mask to an image")
    group.add_argument("--average", nargs='+', help="Average multiple volumes")

    parser.add_argument("--inputs", nargs='+', help="Input NIfTI volumes in A, B, C, etc. order (used with --operation)")
    parser.add_argument("--input", help="Input image (used with --applymask)")
    parser.add_argument("--mask", help="Mask image (used with --applymask)")
    parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args(sys.argv[2:])
    perform_mri_math(args)

def cli_transform_pipeline():
    parser = argparse.ArgumentParser(description="Apply or reverse transform pipeline using transform record JSON.")
    parser.add_argument("--input", required=True, help="Input image (e.g. transformed or original scan)")
    parser.add_argument("--record", required=True, help="Path to transform record JSON file")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--mode", choices=["apply", "reverse"], default="apply", help="Apply or reverse transform pipeline")
    parser.add_argument("--interp", type=int, default=1, help="Interpolation order for resizing (default=1)")

    args = parser.parse_args(sys.argv[2:])
    apply_or_reverse_transforms(
        input_path=args.input,
        transform_record_path=args.record,
        output_path=args.output,
        mode=args.mode,
        interp=args.interp
    )

def cli_summarize_exam_series():
    parser = argparse.ArgumentParser(description="Identify scan types in DICOM directory using metadata and naming.")
    parser.add_argument("--dicom_dir", required=True, help="Path to top-level DICOM directory (should contain 'MR/' subdirectory)")
    parser.add_argument("--to_csv", help="Path to save CSV of results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose reasoning output")
    args = parser.parse_args(sys.argv[2:])
    df = summarize_exam_series(dicom_exam_dir=args.dicom_dir, to_csv=args.to_csv, verbose=not args.quiet)

def cli_create_patient_metadata():
    parser = argparse.ArgumentParser(description="Create a patient metadata table for use in converting DICOM directories into nifti directories.")
    parser.add_argument("--dir", required=True, help="Root directory with {Patient_folder}/.../MR/{series}")
    parser.add_argument("--metadataOut", required=True, help="Output path (.csv | .tsv | .xlsx)")
    parser.add_argument("--previousMetadata", nargs="*", default=[], help="Zero or more previous metadata tables (.csv|.tsv|.xlsx)")
    parser.add_argument("--omitPrevious", action="store_true", help="Omit rows whose Directory appears in previous metadata")
    parser.add_argument("--subdirs", nargs="+", default=["MR"],
                    help="One or more subfolder names to search under each patient folder (default: MR). Example: --subdirs MR MR2")
    parser.add_argument("--excludeEmpty", action="store_true",
                    help="If set, exclude patient folders where no DICOM files were found under the chosen subfolders")
    args = parser.parse_args(sys.argv[2:])
    create_patient_metadata(root_dir=args.dir, out_path=args.metadataOut, previous_paths=args.previousMetadata, omit_previous=args.omitPrevious, subdirs=args.subdirs, exclude_empty=args.excludeEmpty,)

def cli_demix_dicoms():
    parser = argparse.ArgumentParser(description="Ensure each series folder contains only one scan; demix if needed.")
    parser.add_argument("--dir", required=True, help="Root directory containing patient/exam/MR folders")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", help="Disable progress bar")
    parser.add_argument("--logOut", default=None,
                    help="Optional path for the move log (.csv | .tsv). "
                         "If omitted, a default demix_log_{date}_{time}.csv is written under --dir")
    parser.add_argument("--outDir", default=None,
                        help="If provided, write a fully de-mixed COPY of --dir under this path "
                         "(original tree left unchanged).")
    parser.add_argument("--dryRun", action="store_true",
                        help="Compute assignments and write the demix log, but do NOT move/copy any files.")
    args = parser.parse_args(sys.argv[2:])
    demix_dicoms(root_dir=args.dir, show_progress=args.show_progress,log_out=args.logOut,out_dir=args.outDir,dry_run=args.dryRun,)

if __name__ == "__main__":
    main()
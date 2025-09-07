# preprocessing_functions.py
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

def normalize_masked_image(input_image_path, mask_path, output_path=None):
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
    # Lazy imports
    import nibabel as nib
    import numpy as np
    # Utilities are imported lazily as well
    from .preprocessing_utils import (
        apply_padding,
        interpolate_to_voxel_dims,
        update_origin_for_padding,
        adjust_to_target_shape,
        read_padding_record,
        load_roi_mask,
    )

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
    from .preprocessing_utils import read_padding_record

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
    import nibabel as nib
    from nilearn.image import resample_to_img

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
    import SimpleITK as sitk
    import numpy as np

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
    import SimpleITK as sitk
    import numpy as np

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
    from .preprocessing_utils import ensure_hd_bet_installed
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
    from .preprocessing_utils import read_padding_record

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
        _filter_derivatives_by_policy
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

    def _sanitize_label(lbl: str) -> str:
        if lbl is None: return "Unknown"
        s = str(lbl).strip()
        s = s.replace("(", "_").replace(")", "_")
        s = re.sub(r"[^\w\-\+\.]+", "_", s)
        s = re.sub(r"__+", "_", s).strip("_")
        return s or "Unknown"

    def _discover_exams(patient_abs: str) -> list[tuple[str, str]]:
        """Return de-duplicated (exam_dir_abs, mr_subdir_name) based on .dcm leaves."""
        seen = set()
        for curr, _dirs, files in os.walk(patient_abs):
            if any(f.lower().endswith(".dcm") for f in files):
                mr_dir = os.path.dirname(curr)
                exam_dir = os.path.dirname(mr_dir)
                mr_name = os.path.basename(mr_dir)
                seen.add((os.path.normpath(exam_dir), mr_name))
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
        print(f"[plan:{prefix}] {msg}")

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
            ).sort_values(by=["_ns","_conf","_acq"], ascending=[False, False, True])
            return None if g.empty else int(g.index[0])

        label_col = "final_label" if "final_label" in df.columns else ("base_type" if "base_type" in df.columns else None)

        # Eligibility masks affect selection ONLY (rows still appear in plan)
        excluded_labels = {"unknown", "unknown-derived","localizer"}
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

        def _select_best(group: _pd.DataFrame) -> int | None:
            g = group.copy()
            # keep only eligible rows within this label group
            g = g[eligible.loc[g.index]]

            if g.empty:
                return None

            # prefer PRIMARY & not derived → AX plane → max slices → max confidence → earliest acq_dt
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
            ).sort_values(by=["_ns","_conf","_acq"], ascending=[False, False, True])

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
    reorient: bool = True,
    compress: bool = True,
) -> str:
    """
    Convert a single DICOM series directory to NIfTI using dicom2nifti.

    Parameters
    ----------
    dicom_series_dir : str
        Path to a directory containing a single series of DICOM files.
    output_path : str
        Destination NIfTI path (.nii or .nii.gz). Parent directories will be created.
    reorient : bool
        If True, reorient to standard (dicom2nifti's default).
    compress : bool
        If True, create .nii.gz; else create .nii.

    Returns
    -------
    str
        The path written to (same as output_path).

    Raises
    ------
    FileNotFoundError
        If the series directory is missing.
    RuntimeError
        If dicom2nifti did not produce a NIfTI.
    """
    from .preprocessing_utils import ensure_dicom2nifti_installed
    ensure_dicom2nifti_installed()
    from pathlib import Path
    import tempfile
    import dicom2nifti

    series = Path(dicom_series_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not series.exists():
        raise FileNotFoundError(f"Series directory not found: {series}")

    # Run conversion into a temp folder, then move the first produced file to desired name.
    with tempfile.TemporaryDirectory() as tmpdir:
        dicom2nifti.convert_directory(
            str(series),
            tmpdir,
            reorient=reorient,
            compression=compress,
        )
        candidates = sorted(Path(tmpdir).glob("*.nii*"))  # .nii or .nii.gz
        if not candidates:
            raise RuntimeError(f"No NIfTI produced from {series}")
        candidates[0].replace(out)
    return str(out)


# ------------------------------------------------------------
# Execute DICOM -> NIfTI conversions from a saved plan
# ------------------------------------------------------------
def convert_dicom_plan(
    plan_path: str,
    n_workers: int | None = None,
    overwrite: bool = False,
    reorient: bool = True,
    compress: bool = True,
    log_out: str | None = None,
    show_progress: bool = True,
):
    """Read a plan produced by `plan_dicom_to_nifti_conversion` and run conversions
    for rows with a non-empty, non-"-" `proposed_nifti_path`.

    Streams a per-row CSV/TSV log to disk (thread-safe), similar to `demix_dicoms()`.
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
            ])
            _log_fh.flush()

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
                    reorient=reorient,
                    compress=compress,
                )
                rec["status"] = "ok"
                rec["nii_path"] = written
            elif action == "DERIVE":
                from .preprocessing_utils import run_derived_generator
                generator_key = rec.get("GeneratorKey", "")
                primary_label = rec.get("PrimaryLabel", "")
                derived_label = rec.get("DerivedLabel", "")
                src_input = rec.get("DeriveInputs") or rec.get("PrimarySeriesPath") or series_dir  # dict for multi-input, else DICOM dir
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
                            _vendor = plan[(plan.get("final_label","").astype(str).str.upper()==comp_base) &
                                           (~plan.get("is_derived", False))]
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
                                _mag = plan[(plan.get("final_label","").astype(str).str.upper()==f"{comp_base}(MAG)") &
                                            (~plan.get("is_derived", False))]
                                _pha = plan[(plan.get("final_label","").astype(str).str.upper()==f"{comp_base}(PHASE)") &
                                            (~plan.get("is_derived", False))]
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
                )
                rec["status"] = "ok"
                rec["nii_path"] = written
            else:
                rec["status"] = "skipped"
                rec["message"] = f"Unknown Action '{action}'"
        except Exception as e:
            rec["status"] = "failed"
            rec["message"] = f"{type(e).__name__}: {e}"
        return rec

    # ---------- execute ----------
    jobs = [r for _, r in todo.iterrows()]
    records = []
    try:
        if n_workers == 1:
            for r in _progress(jobs, total=len(jobs), desc="Converting", unit="series", enable=show_progress):
                rec = _convert_row(r)
                records.append(rec)
                _stream_log_row(rec)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = [pool.submit(_convert_row, r) for r in jobs]
                for ft in _progress(as_completed(futs), total=len(futs), desc="Converting", unit="series", enable=show_progress):
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

def _build_cli_parser() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(
        prog="python -m astril.preprocess",
        description="MRI Preprocessing Tools (subcommand-style CLI)",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ---- normalize
    p = sub.add_parser("normalize", help="Normalize an MRI volume using a binary mask.")
    p.add_argument("--input", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--output", required=True)
    def _run_normalize(a):
        normalize_masked_image(a.input, a.mask, a.output)
    p.set_defaults(func=_run_normalize)

    # ---- resize
    p = sub.add_parser("resize", help="Resize MRI scan to target shape and voxel dims.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data_dims", default="240,240,155", help="e.g., 240,240,155")
    p.add_argument("--voxel_dims", default="1.0,1.0,1.0", help="e.g., 1.0,1.0,1.0")
    p.add_argument("--interp", type=int, default=1)
    p.add_argument("--save_padding_record", action="store_true")
    p.add_argument("--padding_record")
    p.add_argument("--roimask")
    p.add_argument("--translation_only", action="store_true")
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

    # ---- reverse_resize
    p = sub.add_parser("reverse_resize", help="Reverse a previous resize using a saved padding record.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--padding_record", required=True)
    p.add_argument("--interp", type=int, default=1)
    def _run_reverse(a):
        reverse_resize_mri(a.input, a.output, a.padding_record, interp=a.interp)
    p.set_defaults(func=_run_reverse)

    # ---- match_affine
    p = sub.add_parser("match_affine", help="Match affine of INPUT to DONOR image.")
    p.add_argument("--input", required=True)
    p.add_argument("--donor", required=True)
    p.add_argument("--output", required=True)
    def _run_match(a):
        match_direction_matrices(a.input, a.donor, a.output)
    p.set_defaults(func=_run_match)

    # ---- merge_masks
    p = sub.add_parser("merge_masks", help="Merge 2 binary masks into one.")
    p.add_argument("--masks", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no_fill", action="store_true")
    p.add_argument("--strict_affine", action="store_true")
    def _run_merge(a):
        merge_binary_masks(a.masks, a.output, fill_holes=not a.no_fill, strict_affine=a.strict_affine)
    p.set_defaults(func=_run_merge)

    # ---- register
    p = sub.add_parser("register", help="Register or apply transform to align moving->fixed.")
    p.add_argument("--fixed", required=True)
    p.add_argument("--moving", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--transform")
    p.add_argument("--apply_only", action="store_true")
    p.add_argument("--type", default="rigid", choices=["rigid", "affine", "translation"])
    p.add_argument("--metric", default="correlation", choices=["correlation", "mi"])
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--save_dummy_ref", action="store_true")
    p.add_argument("--quiet", action="store_true")
    def _run_register(a):
        register_images(
            fixed_path=a.fixed, moving_path=a.moving, output_path=a.output,
            transform_path=a.transform, apply_only=a.apply_only,
            registration_type=a.type, similarity_metric=a.metric,
            use_gpu=a.use_gpu, save_dummy_ref=a.save_dummy_ref, verbose=not a.quiet,
        )
    p.set_defaults(func=_run_register)

    # ---- inverse_transform
    p = sub.add_parser("inverse_transform", help="Apply inverse of a saved transform.")
    p.add_argument("--original", required=True, help="Original (pre-registered) image")
    p.add_argument("--transformed", required=True, help="Already transformed image")
    p.add_argument("--transform", required=True, help="Transform .tfm")
    p.add_argument("--output", required=True)
    p.add_argument("--interp", default="linear", choices=["linear", "nearest"])
    p.add_argument("--quiet", action="store_true")
    def _run_inverse(a):
        inverse_transform_image(
            original_image_path=a.original,
            transformed_image_path=a.transformed,
            transform_path=a.transform,
            output_path=a.output,
            interpolation=a.interp,
            verbose=not a.quiet,
        )
    p.set_defaults(func=_run_inverse)

    # ---- skullstrip (hd-bet)
    p = sub.add_parser("skullstrip", help="Run HD-BET skullstripping.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", help="Optional output (bet) image")
    p.add_argument("--mask", help="Optional output mask")
    p.add_argument("--mode", default="accurate")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tta", type=int, default=0)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--overwrite_existing", type=int, default=0)
    def _run_hd_bet_cli(a):
        run_hd_bet(
            input_path=a.input,
            output_path=a.output,
            mask_path=a.mask,
            mode=a.mode,
            device=a.device,
            tta=a.tta,
            pp=a.pp,
            overwrite_existing=a.overwrite_existing,
        )
    p.set_defaults(func=_run_hd_bet_cli)

    # ---- math
    p = sub.add_parser("math", help="Arithmetic / masking on MRI volumes.")
    # Keep the flexible interface used by perform_mri_math()
    p.add_argument("--applymask", action="store_true")
    p.add_argument("--input")
    p.add_argument("--mask")
    p.add_argument("--average", nargs="*")
    p.add_argument("--operation")
    p.add_argument("--inputs", nargs="*")
    p.add_argument("--output")
    def _run_math(a):
        # Reuse the existing, flexible argument contract
        perform_mri_math(a)
    p.set_defaults(func=_run_math)

    # ---- transform_pipeline
    p = sub.add_parser("transform_pipeline", help="Apply or reverse a transform pipeline (json).")
    p.add_argument("--input", required=True)
    p.add_argument("--record", required=True, help="transform_record.json")
    p.add_argument("--output", required=True)
    p.add_argument("--mode", default="apply", choices=["apply", "reverse"])
    p.add_argument("--interp", type=int, default=1)
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
    p = sub.add_parser("summarize_exam_series", help="Infer scan types from an exam's MR/ series.")
    p.add_argument("--dir", required=True, help="Exam directory containing MR/ subfolder")
    p.add_argument("--mrSubdir", default="MR")
    p.add_argument("--to_csv")
    p.add_argument("--quiet", action="store_true")
    def _run_summarize(a):
        summarize_exam_series(a.dir, mr_subdir=a.mrSubdir, to_csv=a.to_csv, verbose=not a.quiet)
    p.set_defaults(func=_run_summarize)

    # ---- create_patient_metadata
    p = sub.add_parser("create_patient_metadata", help="Scan multi-patient DICOM dir to build a metadata table.")
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
    p = sub.add_parser("demix_dicoms", help="Ensure each series folder contains files from only one scan.")
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
    p = sub.add_parser("plan_dicom_to_nifti_conversion", help="Plan which DICOM series to convert and propose NIfTI names.")
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
        )
    p.set_defaults(func=_run_plan)

    # ---- convert_dicom_to_nifti (single series)
    p = sub.add_parser("convert_dicom_to_nifti", help="Convert one DICOM series directory to NIfTI (dicom2nifti).")
    p.add_argument("--dicom_dir", required=True, help="Directory containing one DICOM series")
    p.add_argument("--output_path", required=True, help="Output NIfTI path (.nii or .nii.gz)")
    p.add_argument("--no_reorient", action="store_true", help="Disable reorientation to standard space")
    p.add_argument("--no_compress", action="store_true", help="Write .nii instead of .nii.gz")
    def _run_c2n(a):
        convert_dicom_to_nifti(
            dicom_series_dir=a.dicom_dir,
            output_path=a.output_path,
            reorient=not a.no_reorient,
            compress=not a.no_compress,
        )
        print(f"Saved: {a.output_path}")
    p.set_defaults(func=_run_c2n)

    # ---- convert_dicom_plan (batch from plan file)
    p = sub.add_parser("convert_dicom_plan", help="Run DICOM->NIfTI conversions from a saved plan (dicom2nifti).")
    p.add_argument("--plan", required=True, help="Path to plan CSV/TSV/XLSX from plan_dicom_to_nifti_conversion.")
    p.add_argument("--n_workers", type=int, default=None, help="Parallel workers (I/O-bound).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output NIfTI if present.")
    p.add_argument("--no_reorient", action="store_true", help="Disable reorientation to standard space.")
    p.add_argument("--no_compress", action="store_true", help="Write .nii instead of .nii.gz.")
    p.add_argument("--logOut", default=None, help="Optional CSV/TSV log path for results.")
    def _run_convert(a):
        convert_dicom_plan(
            plan_path=a.plan,
            n_workers=a.n_workers,
            overwrite=a.overwrite,
            reorient=(not a.no_reorient),
            compress=(not a.no_compress),
            log_out=a.logOut,
            show_progress=True,
        )
    p.set_defaults(func=_run_convert)


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
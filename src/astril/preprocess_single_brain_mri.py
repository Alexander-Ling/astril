"""
preprocess_single_brain_mri.py
Author: Alex Ling
E. Antonio Chiocca Group, BWH
Description: Full MRI preprocessing pipeline for brain scans (T1c, T1n, T2f, T2w), replicating the original preprocessing pipeline.
"""

import os
import sys
import argparse
import copy
import tempfile
import shutil
import json

def run_preprocessing_pipeline(
    scans,
    output_dir,
    temp_dir,
    anchor_label="T1c",
    registration_metric="mi",
    co_register_path=None,
    registration_strategy="medium",
    save_scans_with_skulls=False,
    final_dims=(240, 240, 155),
    final_voxels=(1.0, 1.0, 1.0),
    patientID=None,
    timepoint=None,
    scanID=None,
    use_gpu=False,
    enable_tta=False,
    brainmask_path=None,
    family_parent_map=None,
    debug=False,
    verbose=True,
):
    """
    Generalized preprocessing pipeline.

    Parameters
    ----------
    scans : dict[str, str | os.PathLike]
        Mapping {label -> nifti_path}. Must include `anchor_label`.
        Supports any number of additional labels. 3D and 4D NIfTI are supported.
        For 4D scans, registration estimates a transform from frame 0, then applies it to all frames.
    output_dir : str | os.PathLike
        Output directory for final preprocessed files.
    temp_dir : str | os.PathLike
        Temporary working directory (created by caller, or TemporaryDirectory()).
    anchor_label : str, default="T1c"
        Label to use as the anchor for initial registration and (if needed) skull stripping.
        Anchor must be a 3D NIfTI.
    registration_metric : str, default="mi"
        Similarity metric used for registration (passed to `register_images`), e.g. "mi" or "correlation".
    co_register_path : str | os.PathLike | None, default=None
        Optional patient-level reference volume. If provided, the anchor-space outputs are co-registered
        to this reference and the same transform is applied to all scans.
    registration_strategy : str, default="medium"
        Registration preset controlling speed/accuracy tradeoffs (passed to `register_images`).
    save_scans_with_skulls : bool, default=False
        If True, also save the registered/coregistered full-head scans (before masking) under
        output_dir/with_skulls/.
    final_dims : tuple[int,int,int], default=(240,240,155)
        Final target data dimensions (X,Y,Z) for outputs (and brainmask).
    final_voxels : tuple[float,float,float], default=(1.0,1.0,1.0)
        Final target voxel sizes (mm) for outputs (and brainmask).
    patientID, timepoint, scanID : str | None
        Optional identifiers. If any are None, values are derived from the anchor filename by splitting
        on '_' (patientID=timepoint=scanID = parts[0:3] when available). Output basenames use only
        "{patientID}_{timepoint}" (scanID intentionally omitted).
    use_gpu : bool, default=False
        If generating a brainmask with HD-BET, run HD-BET with device="cuda" when True.
    enable_tta : bool, default=False
        If generating a brainmask with HD-BET, enable test-time augmentation (TTA) when True.
        (Passed to hd-bet as disable_tta=not enable_tta.)
    brainmask_path : str | os.PathLike | None, default=None
        Optional pre-existing brainmask (3D NIfTI). If provided, HD-BET is skipped. If the mask grid
        does not match the (co-registered) anchor volume, it is resampled to the anchor grid using
        nearest-neighbor interpolation.
    family_parent_map : dict[str, str] | None, default=None
        Optional mapping {family -> parent_label}. JSON string to specify which sequence to use for registration
        when processing modality families (i.e. parent + derived sequences) for co-registration. Transformation
        matrices are calculcated from a single sequence per family and then applied to all family members, defaulting
        to use the parent sequence for calculation of the transformation matrix.
    debug : bool, default=False
        If True, intermediate files are kept in the caller-provided temp_dir.
    verbose : bool, default=True
        If True, print progress messages.
    """
    import os
    import json
    import shutil
    import nibabel as nib
    import numpy as np

    from astril.preprocessing_utils import (
        get_nifti_ndim,
        apply_mask_anydim,
        apply_mask_anydim_sitk,
        normalize_masked_anydim,
        ensure_hd_bet_installed,
        find_sidecars_for_nifti,
        copy_sidecars_for_output,
    )

    from astril.preprocess import (
        register_images,
        perform_mri_math,
        run_hd_bet,
        normalize_masked_image,
        resize_mri,
        match_direction_matrices,
        inverse_transform_image,
    )

    # ---- Helper function to verify that derived sequences retain dimensions/affine of parent sequences ---
    	
    def nifti_geometry_compatible(a_path: str, b_path: str, affine_atol: float = 1e-4, affine_rtol: float = 1e-4) -> bool:
 
        """Return True if two NIfTI files share the same voxel grid (shape + affine within tolerance)."""
 
        a_img = nib.load(a_path)
 
        b_img = nib.load(b_path)
 
        if a_img.shape[:3] != b_img.shape[:3]:
 
            return False
        return np.allclose(a_img.affine, b_img.affine, atol=affine_atol, rtol=affine_rtol)

    # ---- check if hd-bet is installed (only needed when generating a mask) ---
    if brainmask_path is None:
        ensure_hd_bet_installed()

    # ---- normalize inputs ----
    scans = {str(k): os.fspath(v) for k, v in (scans or {}).items()}
    if anchor_label not in scans:
        raise ValueError(f"scans must include anchor_label='{anchor_label}'. Got keys: {sorted(scans.keys())}")

    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.abspath(temp_dir)
    output_dir = os.path.abspath(output_dir)

    # Derive IDs from anchor filename if not provided
    anchor_path = scans[anchor_label]
    if patientID is None or timepoint is None or scanID is None:
        base = os.path.basename(anchor_path)
        stem = base
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.endswith(".nii"):
            stem = stem[:-4]
        parts = stem.split("_")
        # Expect {patientID}_{timepoint}_{scanID}_{label} (common in this repo)
        # Fall back gracefully if structure differs.
        if patientID is None and len(parts) >= 1:
            patientID = parts[0]
        if timepoint is None and len(parts) >= 2:
            timepoint = parts[1]
        if scanID is None and len(parts) >= 3:
            scanID = parts[2]

    patientID = patientID or "UNKNOWN"
    timepoint = timepoint or "UNKNOWN"
    scanID = scanID or "UNKNOWN"

    # Preprocessed basenames should not include source scanID; use PatientID_Timepoint only
    basename_prefix = f"{patientID}_{timepoint}"

    # Output transform directory
    transform_basedir = "transforms"
    transform_dir = os.path.join(output_dir, transform_basedir)
    os.makedirs(transform_dir, exist_ok=True)

    # Track transform provenance for each label
    transform_records: dict[str, dict] = {lbl: {"label": lbl} for lbl in scans.keys()}

    # Keep working paths for each stage
    reg_paths: dict[str, str] = {}

    # Track which inputs are 4D so we can treat them differently
    ndim_by_label: dict[str, int] = {}
    for lbl, pth in scans.items():
        ndim, _ = get_nifti_ndim(pth)
        ndim_by_label[lbl] = int(ndim)

    fourd_labels = [lbl for lbl, nd in ndim_by_label.items() if nd == 4]
    threed_labels = [lbl for lbl, nd in ndim_by_label.items() if nd == 3]

    if verbose and fourd_labels:
        print(f"[Info] Detected 4D scans (will be kept UNREGISTERED; skull-strip only): {', '.join(fourd_labels)}")

    # ---- Step 1: register everything to anchor space ----
    if verbose:
        print(f"[Step 1] Register all scans to anchor '{anchor_label}'")

    anchor_ndim, _ = get_nifti_ndim(anchor_path)
    if anchor_ndim != 3:
        raise ValueError(f"Anchor '{anchor_label}' must be 3D. Got ndim={anchor_ndim} at: {anchor_path}")

    reg_paths[anchor_label] = anchor_path



    def _label_family(label: str) -> str:
        # Default family grouping: prefix before first underscore (e.g., DWI_TRACE -> DWI)
        return label.split("_")[0] if "_" in label else label

    def _estimate_and_record_registration(lbl: str, src: str):
        # Estimate a new transform lbl->anchor, save reg volume, move tfm + dummy refs, record provenance.
        ndim, shape = get_nifti_ndim(src)
        tfm = os.path.join(temp_dir, f"{lbl}_to_{anchor_label}.tfm")

        # Note: 4D scans are allowed as family parents so we can estimate a transform from the 4D series
        # to the anchor (typically from a representative frame). However, we do not carry forward a registered
        # 4D volume in this pipeline; 4D outputs remain in native space and are skull-stripped only later.

        if verbose:
            print(f"Registering {lbl} to {anchor_label}...")

        if ndim == 3:
            out_reg = os.path.join(temp_dir, f"{basename_prefix}_{lbl}_reg.nii.gz")
            register_images(
                fixed_path=anchor_path,
                moving_path=src,
                output_path=out_reg,
                transform_path=tfm,
                apply_only=False,
                similarity_metric=registration_metric,
                registration_strategy=registration_strategy,
                save_dummy_ref=True,
                verbose=False,
                debug=debug,
            )
            reg_paths[lbl] = out_reg

        elif ndim == 4:
            # 4D: estimate transform from a single frame (default frame 0). We do NOT keep a registered 4D output.
            out_reg = os.path.join(temp_dir, f"{basename_prefix}_{lbl}_reg.nii.gz")
            register_images(
                fixed_path=anchor_path,
                moving_path=src,
                output_path=out_reg,
                transform_path=tfm,
                apply_only=False,
                similarity_metric=registration_metric,
                registration_strategy=registration_strategy,
                save_dummy_ref=True,
                verbose=False,
                moving_frame_index=0,
                debug=debug,
            )
            # Discard the registered 4D output (we only keep the transform + dummy refs).
            if not debug:
                try:
                    if os.path.exists(out_reg):
                        os.remove(out_reg)
                    # Also remove any incidental sidecars next to the temp registered file
                    out_base = out_reg[:-7] if out_reg.lower().endswith(".nii.gz") else os.path.splitext(out_reg)[0]
                    for _ext in (".json", ".bval", ".bvec", ".nii.json"):
                        _p = out_base + _ext
                        if os.path.exists(_p):
                            os.remove(_p)
                except Exception:
                    pass

        else:
            raise ValueError(f"Unsupported NIfTI dimensionality for label '{lbl}': ndim={ndim} path={src}")

        # Move transform + dummy refs into output/transforms and record relative paths
        tfm_dest = os.path.join(transform_dir, os.path.basename(tfm))
        shutil.move(tfm, tfm_dest)

        moving_ref_dummy = tfm.replace(".tfm", "_moving_ref.nii.gz")
        fixed_ref_dummy = tfm.replace(".tfm", "_fixed_ref.nii.gz")

        moving_ref_dest = os.path.join(transform_dir, os.path.basename(moving_ref_dummy))
        fixed_ref_dest = os.path.join(transform_dir, os.path.basename(fixed_ref_dummy))
        shutil.move(moving_ref_dummy, moving_ref_dest)
        shutil.move(fixed_ref_dummy, fixed_ref_dest)

        info = {
            "transform": f"./{transform_basedir}/{os.path.basename(tfm_dest)}",
            "fixed_reference": f"./{transform_basedir}/{os.path.basename(fixed_ref_dest)}",
            "moving_reference": f"./{transform_basedir}/{os.path.basename(moving_ref_dest)}",
            "estimated_from": "frame0" if ndim == 4 else "volume",
        }
        transform_records[lbl]["initial_registration"] = info
        return info, ndim

    def _apply_parent_transform(lbl: str, src: str, parent_info: dict, parent_ndim: int, parent_label: str, parent_src: str):
        # Apply a previously estimated parent transform to this scan.
        # Only valid when the scan geometry matches the parent scan geometry.
        if not nifti_geometry_compatible(src, parent_src):
            raise ValueError("apply_parent_transform called for incompatible geometry")

        if verbose:
            print(f"Applying {parent_label}->... transform to {lbl} (same family; compatible geometry)")

        tfm_path = os.path.join(output_dir, parent_info["transform"].lstrip("./"))
        ndim, shape = get_nifti_ndim(src)

        if ndim == 3:
            out_reg = os.path.join(temp_dir, f"{basename_prefix}_{lbl}_reg.nii.gz")
            register_images(
                fixed_path=anchor_path,
                moving_path=src,
                output_path=out_reg,
                transform_path=tfm_path,
                apply_only=True,
                save_dummy_ref=False,
                verbose=False,
                debug=debug,
            )
            reg_paths[lbl] = out_reg

        elif ndim == 4:
            # Desired behavior: 4D scans are NEVER registered/resampled in this pipeline.
            # They remain in native space and are skull-stripped only in Step 4b.
            # We still record provenance that this scan would share the parent's transform conceptually.
            if verbose:
                print(f"  [skip] {lbl} is 4D; keeping native (will output *_unregistered) and not applying parent transform.")

        else:
            raise ValueError(f"Unsupported NIfTI dimensionality for label '{lbl}': ndim={ndim} path={src}")

        # Record provenance: reuse parent transform/refs, and note the parent label used.
        transform_records[lbl]["initial_registration"] = {
            **parent_info,
            "applied_from_parent": parent_label,
            "estimated_from": "frame0" if ndim == 4 else "volume",
            "skipped_output": True if ndim == 4 else False,
        }

    # ---- Step 1: register everything to anchor space ----
    reg_paths[anchor_label] = anchor_path

    # NEW DEFAULT BEHAVIOR:
    # - Always do family-based registration when possible:
    #   * If the family has a "main" parent scan present (label exactly equals family, e.g. "DWI"),
    #     estimate only that transform and apply it to derived scans (e.g. "DWI_TRACE") when geometry matches.
    #   * If user provides family_parent_map, it overrides the chosen parent ONLY for families provided.
    #   * For families without a main parent present, we fall back to choosing a deterministic parent
    #     (first label in sorted order).
    #
    # This replaces the previous default of registering every label independently when family_parent_map is None.
    user_family_parent_map = family_parent_map if isinstance(family_parent_map, dict) else {}
 
    # Group non-anchor labels by inferred family
    by_family: dict[str, list[str]] = {}
    for lbl, src in scans.items():
        if lbl == anchor_label:
            continue
        fam = _label_family(lbl)
        by_family.setdefault(fam, []).append(lbl)
 
    def _default_parent_for_family(fam: str, labels: list[str]) -> str:
        """
        Default parent selection:
        1) Prefer a label exactly equal to the family name (e.g., family="DWI" -> label "DWI"),
            which corresponds to the 'main' sequence in your naming convention.
        2) Otherwise, choose the first label in deterministic (sorted) order.
        """
        return fam if fam in labels else sorted(labels)[0]
 
    # Preserve deterministic processing order across families and labels
    for fam in sorted(by_family.keys()):
        labels = sorted(by_family[fam])
 
        # Choose parent:
        # - If user specified this family, try to honor it.
        # - Otherwise follow default selection (main parent when present).
        parent = None
        if fam in user_family_parent_map:
            cand = user_family_parent_map.get(fam)
            if cand in labels:
                parent = cand
            else:
                # User tried to override, but the requested label isn't present in this run.
                # Fall back gracefully to default behavior for this family.
                if verbose:
                    print(
                        f"[family_parent_map] Requested parent '{cand}' for family '{fam}' "
                        f"but available labels are {labels}. Falling back to default parent selection."
                    )
 
        if parent is None:
            parent = _default_parent_for_family(fam, labels)
 
        parent_src = scans[parent]
        parent_info, parent_ndim = _estimate_and_record_registration(parent, parent_src)
 
        for lbl in labels:
            if lbl == parent:
                continue
 
            src = scans[lbl]
            # If geometry matches parent, reuse transform; otherwise fall back to individual registration.
            if nifti_geometry_compatible(src, parent_src):
                _apply_parent_transform(lbl, src, parent_info, parent_ndim, parent, parent_src)
            else:
                if verbose:
                    print(
                        f"{lbl} is not geometry-compatible with parent {parent} (family {fam}); "
                        f"falling back to individual registration."
                    )
                _estimate_and_record_registration(lbl, src)

    # ---- Step 2: optionally co-register anchor space to provided reference ----
    # Apply the same T1c->ref transform to anchor and all other registered scans.
    coreg_anchor_path = reg_paths[anchor_label]
    if co_register_path:
        if verbose:
            print("[Step 2] Co-register anchor space to provided reference and apply to all scans")

        # Apply co-registration ONLY to 3D scans
        # 4D scans are kept in their native space and handled later (skull-strip only).

        coreg_tfm = os.path.join(temp_dir, f"{anchor_label}_to_coreg.tfm")
        anchor_coreg = os.path.join(temp_dir, f"{basename_prefix}_{anchor_label}_coreg.nii.gz")

        if verbose:
                print(f"Registering {anchor_label} to {co_register_path}...")

        register_images(
            co_register_path, coreg_anchor_path, anchor_coreg,
            transform_path=coreg_tfm,
            apply_only=False,
            similarity_metric=registration_metric,
            registration_strategy=registration_strategy,
            save_dummy_ref=True,
            verbose=False,
            debug=debug,
        )

        # Move coreg transform + dummy refs
        coreg_tfm_dest = os.path.join(transform_dir, os.path.basename(coreg_tfm))
        shutil.move(coreg_tfm, coreg_tfm_dest)

        coreg_moving_ref_dummy = coreg_tfm.replace(".tfm", "_moving_ref.nii.gz")
        coreg_fixed_ref_dummy = coreg_tfm.replace(".tfm", "_fixed_ref.nii.gz")
        coreg_moving_ref_dest = os.path.join(transform_dir, os.path.basename(coreg_moving_ref_dummy))
        coreg_fixed_ref_dest = os.path.join(transform_dir, os.path.basename(coreg_fixed_ref_dummy))
        shutil.move(coreg_moving_ref_dummy, coreg_moving_ref_dest)
        shutil.move(coreg_fixed_ref_dummy, coreg_fixed_ref_dest)

        # Apply to all registered scans (including anchor)
        new_reg_paths: dict[str, str] = {}
        for lbl, src_reg in reg_paths.items():

            # Hard guard: never co-register 4D scans.
            # (They should not be in reg_paths, but this prevents accidental resampling if they are.)
            if ndim_by_label.get(lbl, 3) == 4:
                if verbose:
                    print(f"  [skip] {lbl} is 4D; not applying co-registration transform.")
                new_reg_paths[lbl] = src_reg
                continue

            ndim, shape = get_nifti_ndim(src_reg)

            if ndim == 3:
                out_coreg = os.path.join(temp_dir, f"{basename_prefix}_{lbl}_coreg.nii.gz")
                register_images(
                    co_register_path, src_reg, out_coreg,
                    transform_path=coreg_tfm_dest,
                    apply_only=True,
                    save_dummy_ref=False,
                    verbose=False,
                    debug=debug,
                )
                new_reg_paths[lbl] = out_coreg

            else:
                raise ValueError(f"Unexpected ndim={ndim} for registered scan: {src_reg}")

            transform_records[lbl]["coregistration"] = {
                "transform": f"./{transform_basedir}/{os.path.basename(coreg_tfm_dest)}",
                "fixed_reference": f"./{transform_basedir}/{os.path.basename(coreg_fixed_ref_dest)}",
                "moving_reference": f"./{transform_basedir}/{os.path.basename(coreg_moving_ref_dest)}",
                "reference": os.fspath(co_register_path),
                "estimated_from": "anchor_volume",
            }

        reg_paths = new_reg_paths
        coreg_anchor_path = reg_paths[anchor_label]
    else:
        if verbose:
            print("[Step 2] No co-registration reference provided; staying in anchor space")

    # ---- Step 3: skull strip anchor and generate brain mask ----
    brainmask_temp = os.path.join(temp_dir, f"{basename_prefix}_brainmask_temp.nii.gz")
    brainmask_source = {"type": "hd-bet"}
    brainmask_resampled = False

    
    if brainmask_path:
        if verbose:
            print("[Step 3] Using provided brainmask (skipping HD-BET)...")
        brainmask_path = os.fspath(brainmask_path)
        if not os.path.exists(brainmask_path):
            raise FileNotFoundError(f"Provided brainmask not found: {brainmask_path}")
        bm_ndim, _ = get_nifti_ndim(brainmask_path)
        if bm_ndim != 3:
            raise ValueError(f"Provided brainmask must be 3D. Got ndim={bm_ndim} at: {brainmask_path}")
        
        # Ensure brainmask matches the grid of the (co-registered) anchor volume
        if nifti_geometry_compatible(brainmask_path, coreg_anchor_path):
            shutil.copy(brainmask_path, brainmask_temp)
            brainmask_source = {"type": "provided", "path": brainmask_path, "resampled": False}
        else:
            if verbose:
                print("Provided brainmask grid does not match anchor; resampling mask to anchor grid...")
            match_direction_matrices(brainmask_path, coreg_anchor_path, brainmask_temp)
            brainmask_source = {"type": "provided", "path": brainmask_path, "resampled": True}
        
    else:
        if verbose:
            print("[Step 3] HD-BET skull strip on anchor and generate brainmask...")
        print(f"Creating temporary brainmask at {brainmask_temp}...")
        run_hd_bet(
            input_path=coreg_anchor_path,
            output_path=None,
            mask_path=brainmask_temp,
            device="cuda" if use_gpu else "cpu",
            disable_tta=not enable_tta,
            verbose=verbose,
        )

    # ---- Step 4: apply mask to all scans ----
    if verbose:
        print("[Step 4] Apply brainmask to all scans...")

    brain_paths: dict[str, str] = {}
    # 3D scans: apply anchor-space mask directly (registered/coregistered already)
    for lbl, src_reg in reg_paths.items():
        if ndim_by_label.get(lbl, 3) != 3:
            continue
        out_brain = os.path.join(temp_dir, f"{basename_prefix}_{lbl}_brain_temp.nii.gz")
        apply_mask_anydim(src_reg, brainmask_temp, out_brain)
        brain_paths[lbl] = out_brain

    # 4D scans: keep native-space; skull-strip only.
    # We map the anchor brainmask back into each 4D scan's native space using the inverse of a frame-based transform.
    unregistered_4d_outputs: dict[str, str] = {}
    for lbl in fourd_labels:
        src_4d = scans[lbl]
        if verbose:
            print(f"[Step 4b] Skull-strip 4D scan (native space, unregistered): {lbl}")

        # Estimate a forward transform (4D->anchor) using register_images, but we do not keep the resampled 4D output.
        tfm = os.path.join(temp_dir, f"{lbl}_to_{anchor_label}.tfm")
        tmp_reg = os.path.join(temp_dir, f"{basename_prefix}_{lbl}_TEMP_REG_SHOULD_NOT_USE.nii.gz")
        register_images(
            fixed_path=coreg_anchor_path,
            moving_path=src_4d,
            output_path=tmp_reg,
            transform_path=tfm,
            apply_only=False,
            similarity_metric=registration_metric,
            registration_strategy=registration_strategy,
            save_dummy_ref=True,
            verbose=False,
            debug=debug,
        )
        # Map the anchor brainmask back into the original 4D grid
        mask_in_native = os.path.join(temp_dir, f"{basename_prefix}_{lbl}_brainmask_native_temp.nii.gz")
        inverse_transform_image(
            original_image_path=src_4d,
            transformed_image_path=brainmask_temp,
            transform_path=tfm,
            output_path=mask_in_native,
            interpolation="nearest",
            verbose=False,
        )

        # Apply the native-space mask to the original 4D scan (no registration/resampling of the 4D data)
        out_unreg = os.path.join(output_dir, f"{basename_prefix}_{lbl}_unregistered.nii.gz")
        apply_mask_anydim(src_4d, mask_in_native, out_unreg)

        # If a sibling 4D NRRD exists (e.g., diffusion NRRD for Slicer), skull-strip it as well.
        # We do NOT estimate any additional transforms here: we reuse the same native-space mask.
        # Convention: <stem>.nrrd next to the NIfTI (same basename without .nii/.nii.gz).
        try:
            def _strip_nii_ext(p: str) -> str:
                pl = p.lower()
                if pl.endswith(".nii.gz"):
                    return p[:-7]
                if pl.endswith(".nii"):
                    return p[:-4]
                return os.path.splitext(p)[0]

            nrrd_in = _strip_nii_ext(src_4d) + ".nrrd"
            if os.path.exists(nrrd_in):
                out_unreg_nrrd = os.path.join(output_dir, f"{basename_prefix}_{lbl}_unregistered.nrrd")
                apply_mask_anydim_sitk(nrrd_in, mask_in_native, out_unreg_nrrd)

                # Carry forward JSON sidecar (if present) for NRRD as well
                # (NRRD diffusion should already store gradients; JSON is for your Astril provenance/metadata.)
                sidecars_nrrd = []
                # Prefer canonical JSON (stem.json) but tolerate legacy stem.nii.json
                stem = _strip_nii_ext(src_4d)
                cand1 = stem + ".json"
                cand2 = stem + ".nii.json"
                if os.path.exists(cand1):
                    sidecars_nrrd.append(cand1)
                elif os.path.exists(cand2):
                    sidecars_nrrd.append(cand2)
                if sidecars_nrrd:
                    copy_sidecars_for_output(sidecars_nrrd, src_4d, out_unreg_nrrd, dry_run=False)

                unregistered_4d_outputs[f"{lbl}.nrrd"] = out_unreg_nrrd
        except Exception as e:
            if verbose:
                print(f"[Warning] Failed to skull-strip NRRD sibling for {lbl}: {e}")

        # Carry forward sidecars:
        # - Always copy JSON
        # - Only copy bval/bvec for *base* DWI
        sidecars = find_sidecars_for_nifti(src_4d)
        keep = []
        for sp in sidecars:
            sp_l = sp.lower()
            if sp_l.endswith(".json"):
                keep.append(sp)
            elif (lbl.upper() == "DWI") and (sp_l.endswith(".bval") or sp_l.endswith(".bvec")):
                keep.append(sp)
        copy_sidecars_for_output(keep, src_4d, out_unreg, dry_run=False)

        unregistered_4d_outputs[lbl] = out_unreg

        # Cleanup the temporary registered 4D output unless debugging
        if (not debug) and os.path.exists(tmp_reg):
            try:
                os.remove(tmp_reg)
            except Exception:
                pass

    # Optionally save skull-containing (registered/coregistered) scans
    if save_scans_with_skulls:
        skull_dir = os.path.join(output_dir, "with_skulls")
        os.makedirs(skull_dir, exist_ok=True)
        for lbl, src_reg in reg_paths.items():
            # Avoid writing 4D skull volumes (they should not be registered outputs in this pipeline).
            if ndim_by_label.get(lbl, 3) == 4:
                continue
            shutil.copy(src_reg, os.path.join(skull_dir, f"{basename_prefix}_{lbl}_with_skull.nii.gz"))

    # ---- Step 5: normalize masked scans ----
    if verbose:
        print("[Step 5] Normalize masked scans...")

    norm_paths: dict[str, str] = {}
    # 3D only: 4D scans are intentionally not normalized
    for lbl, src_brain in brain_paths.items():
        out_norm = os.path.join(temp_dir, f"{basename_prefix}_{lbl}_brain_norm_temp.nii.gz")
        normalize_masked_anydim(src_brain, brainmask_temp, out_norm)
        norm_paths[lbl] = out_norm

    # ---- Step 6: resize to final shape/voxel dims ----
    if verbose:
        print("[Step 6] Resize scans to final dimensions/voxels...")

    # 3D only: 4D scans are intentionally not resized

    final_brain_paths: dict[str, str] = {}
    final_norm_paths: dict[str, str] = {}
    # Save final brainmask (resized consistently with outputs)
    brainmask_out = os.path.join(output_dir, f"{basename_prefix}_brainmask.nii.gz")
    resize_mri(brainmask_temp, brainmask_out, final_dims, final_voxels, interp="nearest")

    for lbl in threed_labels:
        out_brain = os.path.join(output_dir, f"{basename_prefix}_{lbl}_brain.nii.gz")
        out_norm = os.path.join(output_dir, f"{basename_prefix}_{lbl}_brain_norm.nii.gz")
        resize_mri(brain_paths[lbl], out_brain, final_dims, final_voxels, interp="linear")
        resize_mri(norm_paths[lbl], out_norm, final_dims, final_voxels, interp="linear")
        final_brain_paths[lbl] = out_brain
        final_norm_paths[lbl] = out_norm

        # Carry forward JSON sidecar from the original acquisition and annotate that it is original-acquisition metadata
        try:
            src_orig = scans[lbl]
            sidecars = find_sidecars_for_nifti(src_orig)
            jsons = [sp for sp in sidecars if sp.lower().endswith(".json")]
            copied = copy_sidecars_for_output(jsons, src_orig, out_brain, dry_run=False)
            copied_norm = copy_sidecars_for_output(jsons, src_orig, out_norm, dry_run=False)
            # Stamp Astril.originalAcquisition so users know SliceThickness/etc refer to original acquisition.
            for cp in (copied + copied_norm):
                if not cp.lower().endswith(".json"):
                    continue
                try:
                    with open(cp, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    astril = meta.get("Astril")
                    if not isinstance(astril, dict):
                        astril = {}
                    astril["originalAcquisition"] = {
                        "source_nifti": os.path.basename(src_orig),
                        "note": "This sidecar describes the original acquisition; image may have been registered/resampled/normalized."
                    }
                    meta["Astril"] = astril
                    with open(cp, "w", encoding="utf-8") as fh:
                        json.dump(meta, fh, indent=2)
                except Exception:
                    pass
        except Exception:
            pass

        # Save transform record for each label
        record_path = os.path.join(output_dir, f"{basename_prefix}_{lbl}_transform_record.json")
        transform_records[lbl]["inputs"] = {"original": os.fspath(scans[lbl])}
        transform_records[lbl]["outputs"] = {
            "brain": os.path.basename(out_brain),
            "brain_norm": os.path.basename(out_norm),
        }
        transform_records[lbl]["brainmask"] = os.path.basename(os.path.join(output_dir, f"{basename_prefix}_brainmask.nii.gz"))
        transform_records[lbl]["brainmask_source"] = brainmask_source
        with open(record_path, "w") as f:
            json.dump(transform_records[lbl], f, indent=2)


    if verbose:
        print("[Done] Outputs written to:", output_dir)



def preprocess_single_brain_mri(
    *,
    output_dir,
    scans=None,
    scan_dir=None,
    modalities=None,
    anchor_label="T1c",
    registration_metric="mi",
    co_register_path=None,
    registration_strategy="medium",
    save_scans_with_skulls=False,
    final_dims=(240, 240, 155),
    final_voxels=(1.0, 1.0, 1.0),
    debug=False,
    patientID=None,
    timepoint=None,
    scanID=None,
    brainmask_path=None,
    family_parent_map=None,
    use_gpu=False,
    enable_tta=False,
    verbose=True,
):
    """
    Preprocess a single exam's MRI scans using an anchor volume (default: T1c).

    Parameters
    ----------
    scans : dict[str, str | Path] | None
        Optional mapping from modality label to NIfTI path. If provided, must include `anchor_label`.
        Values may point to 3D or 4D NIfTI files.
    scan_dir : str | Path | None
        If `scans` is not provided, scans are auto-detected from this directory (non-recursively) using
        the astril naming convention: {patientID}_{timepoint}_{modality}.nii[.gz] (modality may include underscores).
    modalities : list[str] | None
        If provided, only these modality labels will be searched for / retained during auto-detection.
        Must include `anchor_label`.
    output_dir : str | Path
        Output directory where final brain-extracted, normalized, resized volumes and
        per-modality transform records will be written.
    anchor_label : str, default="T1c"
        Label in `scans` to use as the reference space for registration and skull stripping.
    registration_metric : str, default="mi"
        Similarity metric used for registration (passed to `register_images`).
    co_register_path : str | Path | None, default=None
        Optional patient-level reference volume. If provided, the anchor-space outputs are co-registered
        to this reference and the same transform is applied to all scans.
    registration_strategy : str, default="medium"
        Registration preset controlling speed/accuracy tradeoffs (passed to `register_images`).
    save_scans_with_skulls : bool, default=False
        If True, also save the registered/coregistered full-head scans (before masking) under
        output_dir/with_skulls/.
    final_dims : tuple[int,int,int], default=(240,240,155)
        Final target data dimensions (X,Y,Z) for outputs (and brainmask).
    final_voxels : tuple[float,float,float], default=(1.0,1.0,1.0)
        Final target voxel sizes (mm) for outputs (and brainmask).
    patientID, timepoint, scanID : str | None
        Optional identifiers. If any are None, values are derived from the anchor filename by splitting
        on '_' (patientID=timepoint=scanID = parts[0:3] when available). Output basenames use only
        "{patientID}_{timepoint}" (scanID intentionally omitted).
    use_gpu : bool, default=False
        If generating a brainmask with HD-BET, run HD-BET with device="cuda" when True.
    enable_tta : bool, default=False
        If generating a brainmask with HD-BET, enable test-time augmentation (TTA) when True.
    brainmask_path : str | Path | None, default=None
        Optional pre-existing brainmask (3D NIfTI). If provided, HD-BET is skipped. If the mask grid
        does not match the (co-registered) anchor volume, it is resampled to the anchor grid using
        nearest-neighbor interpolation.
    family_parent_map : dict[str, str] | None, default=None
        Optional mapping {family -> parent_label}. When provided, registration to the anchor is estimated
        only for the parent scan in each family, and the parent transform is applied to compatible
        sub-modalities in that family. If a sub-modality is not geometry-compatible (shape/affine) with
        the parent scan, it falls back to individual transform estimation.
        Family is inferred as the label prefix before the first underscore (e.g. "DWI_TRACE" -> "DWI").
    debug : bool, default=False
        If True, keep intermediate files in the temp workspace.
    verbose : bool, default=True
        If True, print progress messages.
    """
    import os
    import tempfile

    if output_dir is None:
        raise ValueError("output_dir is required.")
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    from astril.preprocessing_utils import discover_scans_in_dir

    if modalities is not None:
        if not isinstance(modalities, (list, tuple, set)):
            raise ValueError("modalities must be a list/tuple/set of modality labels or None.")
        modalities = [str(m) for m in modalities]
        if anchor_label not in modalities:
            raise ValueError(
                f"modalities must include anchor_label='{anchor_label}'. Got: {modalities}"
            )

    # If scans not explicitly provided, auto-detect from scan_dir
    if scans is None:
        scans = {}
    if not isinstance(scans, dict):
        raise ValueError("scans must be a dict mapping label -> nifti path (or None to auto-detect).")

    if scan_dir is not None:
        scan_dir = os.fspath(scan_dir)

    if not scans:
        if scan_dir is None:
            raise ValueError("You must provide either `scans` or `scan_dir` for auto-detection.")
        scans = discover_scans_in_dir(scan_dir, modalities=modalities)

    else:
        # If both are provided, merge auto-detected scans but let explicit scans override.
        if scan_dir is not None:
            auto = discover_scans_in_dir(scan_dir, modalities=modalities)
            auto.update({str(k): os.fspath(v) for k, v in scans.items()})
            scans = auto
        else:
            scans = {str(k): os.fspath(v) for k, v in scans.items()}

    if anchor_label not in scans:
        raise ValueError(
            f"Anchor label '{anchor_label}' not found in scans. Got: {sorted(scans.keys())}"
        )

    # Normalize to strings/paths and validate existence
    for lbl, p in list(scans.items()):

        if p is None:
            scans.pop(lbl, None)
            continue
        scans[lbl] = os.fspath(p)
        if not os.path.isfile(scans[lbl]):
            raise FileNotFoundError(f"Scan not found for label '{lbl}': {scans[lbl]}")

    if debug:
        # Debug mode keeps temp dir for inspection
        temp_dir = os.path.join(output_dir, "temp_preprocessing")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[Debug] Retaining temporary directory: {temp_dir}")
        run_preprocessing_pipeline(
            scans=scans,
            output_dir=output_dir,
            temp_dir=temp_dir,
            anchor_label=anchor_label,
            registration_metric=registration_metric,
            co_register_path=co_register_path,
            registration_strategy=registration_strategy,
            save_scans_with_skulls=save_scans_with_skulls,
            final_dims=final_dims,
            final_voxels=final_voxels,
            patientID=patientID,
            timepoint=timepoint,
            scanID=scanID,
            use_gpu=use_gpu,
            enable_tta=enable_tta,
            brainmask_path=brainmask_path,
            family_parent_map=family_parent_map,
            debug=True,
            verbose=verbose,
        )
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_preprocessing_pipeline(
                scans=scans,
                output_dir=output_dir,
                temp_dir=temp_dir,
                anchor_label=anchor_label,
                registration_metric=registration_metric,
                registration_strategy=registration_strategy,
                co_register_path=co_register_path,
                save_scans_with_skulls=save_scans_with_skulls,
                final_dims=final_dims,
                final_voxels=final_voxels,
                patientID=patientID,
                timepoint=timepoint,
                scanID=scanID,
                use_gpu=use_gpu,
                enable_tta=enable_tta,
                brainmask_path=brainmask_path,
                family_parent_map=family_parent_map,
                debug=False,
                verbose=verbose,
            )


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=(
            "Full MRI preprocessing pipeline using T1c as the anchor by default. "
            "Processes any provided modalities (3D or 4D NIfTI). "
            "For 4D scans, registers the first frame and applies the same transform to all frames."
        )
    )

    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--scan_dir",
        default=None,
        help=(
            "Directory containing input NIfTI scans for auto-detection (non-recursive). "
            "Used only if no --scan/--scans_json are provided. Expected filenames: "
            "{patientID}_{timepoint}_{modality}.nii[.gz] (modality may include underscores)."
        ),
    )

    parser.add_argument(
        "--modalities",
        default=None,
        help=(
            "Comma-separated list of modality labels to auto-detect from --scan_dir (non-recursive). "
            "Example: --modalities T1c,T2w,DWI_MD,DWI_TRACE. Must include the anchor label. "
            "Ignored when explicit --scan/--scans_json are provided."
        ),
    )

    # New, flexible modality inputs
    parser.add_argument(
        "--scan",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Add an input scan mapping. Repeatable. Example: --scan T1c=/path/t1c.nii.gz --scan FLAIR=/path/flair.nii.gz. "
            "At minimum, the anchor label (default T1c) must be provided."
        ),
    )
    parser.add_argument(
        "--scans_json",
        default=None,
        help=(
            "JSON string mapping labels to paths. Example: "
            """'{"T1c":"/path/t1c.nii.gz","T2w":"/path/t2w.nii.gz","DWI":"/path/dwi.nii.gz"}'"""
        ),
    )

    parser.add_argument(
        "--anchor_label",
        default="T1c",
        help="Label to use as the anchor / reference volume for registration and skull stripping (default: T1c).",
    )

    parser.add_argument(
        "--patientID",
        help=(
            "Patient identifier. If omitted, defaults to the first '_' separated field of the anchor filename."
        ),
    )
    parser.add_argument(
        "--timepoint",
        help=(
            "Timepoint identifier. If omitted, defaults to the second '_' separated field of the anchor filename."
        ),
    )
    parser.add_argument(
        "--scanID",
        help=(
            "Unique identifier for this sequence of scans. If omitted, defaults to the third '_' separated field of the anchor filename."
        ),
    )

    parser.add_argument(
        "--registration_metric",
        default="mi",
        help="Metric for registration steps: 'correlation' or 'mi' (mutual information) (default: mi).",
    )
    parser.add_argument(
        "--registration_strategy",
        default="medium",
        help="registration_strategy : {accurate, medium, or fast}, convenience preset controlling registration speed/accuracy tradeoffs",
    )
    parser.add_argument("--co_register", help="Optional reference image to co-register all scans to")
    parser.add_argument(
        "--save_scans_with_skulls",
        action="store_true",
        help=(
            "Save intermediate full-head scans (with skulls) in addition to brain-extracted outputs. "
            "Use caution if scans contain PHI."
        ),
    )
    parser.add_argument("--final_dims", default="240,240,155", help="Final data dimensions (default: 240,240,155)")
    parser.add_argument("--final_voxels", default="1.0,1.0,1.0", help="Final voxel sizes (default: 1.0,1.0,1.0)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration for hd-bet skull stripping.")
    parser.add_argument("--enable_tta", action="store_true", help="Enable test-time augmentation (TTA) for hd-bet skull stripping (slower, may improve accuracy; recommended when using GPU).")
    parser.add_argument(
        "--brainmask",
        default=None,
        help=(
            "Optional pre-existing brainmask (3D NIfTI) to apply instead of generating one with HD-BET. "
            "If its grid does not match the (co-registered) anchor volume, it will be resampled with nearest-neighbor interpolation."
        ),
    )
    parser.add_argument(
        "--family_parent_map",
        default=None,
        help=(
            "JSON string to specify which sequence to use for registration when processing modality families"
            "(i.e. parent + derived sequences) for co-registration. Transformation matrices are calculcated"
            "from a single sequence per family and then applied to all family members, defaulting to use the"
            "parent sequence for calculation of the transformation matrix."
            "Example: {\"DWI\":\"DWI\", \"SWI\":\"SWI\"}. "
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Keep intermediate files and temp directory after execution")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose print logging.")

    args = parser.parse_args()

    # Parse modalities
    modalities = None
    if args.modalities:
        modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
        if args.anchor_label not in modalities:
            raise SystemExit(
                f"--modalities must include anchor label '{args.anchor_label}'. Got: {modalities}"
            )

    # Parse dims/voxels
    dims = tuple(map(int, args.final_dims.split(",")))
    voxels = tuple(map(float, args.final_voxels.split(",")))

    # Build scans dict from scans_json and --scan entries
    scans = {}

    if args.scans_json:
        try:
            parsed = json.loads(args.scans_json)
            if not isinstance(parsed, dict):
                raise ValueError("--scans_json must decode to a JSON object/dict.")
            scans.update({str(k): str(v) for k, v in parsed.items()})
        except Exception as e:
            raise SystemExit(f"Failed to parse --scans_json: {e}")

    for item in args.scan or []:
        if "=" not in item:
            raise SystemExit(f"Invalid --scan value '{item}'. Expected format LABEL=PATH.")
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip().strip('"').strip("'")
        if not label:
            raise SystemExit(f"Invalid --scan value '{item}': empty LABEL.")
        if not path:
            raise SystemExit(f"Invalid --scan value '{item}': empty PATH.")
        scans[label] = path  # explicit --scan overrides JSON

    # If user did not provide explicit scans, allow auto-detection from --scan_dir
    if scans:
        scans = {k: str(Path(v)) for k, v in scans.items()}
        if args.anchor_label not in scans:
            raise SystemExit(
                f"Anchor label '{args.anchor_label}' not found in provided scans. "
                f"Provided labels: {sorted(scans.keys())}"
            )
    else:
        if not args.scan_dir:
            raise SystemExit(
                "You must provide either --scan/--scans_json (including the anchor) OR --scan_dir for auto-detection."
            )

    # Parse optional family_parent_map (JSON string or path)
    family_parent_map = None
    if args.family_parent_map:
        import os
        try:
            if os.path.isfile(args.family_parent_map):
                with open(args.family_parent_map, "r", encoding="utf-8") as fh:
                    family_parent_map = json.load(fh)
            else:
                family_parent_map = json.loads(args.family_parent_map)
            if not isinstance(family_parent_map, dict):
                raise ValueError("--family_parent_map must decode to a JSON object/dict.")
            family_parent_map = {str(k): str(v) for k, v in family_parent_map.items()}
        except Exception as e:
            raise SystemExit(f"Failed to parse --family_parent_map: {e}")

    preprocess_single_brain_mri(
        output_dir=args.output,
        scans=scans if scans else None,
        modalities=modalities,
        scan_dir=args.scan_dir,
        anchor_label=args.anchor_label,
        registration_metric=args.registration_metric,
        registration_strategy=args.registration_strategy,
        co_register_path=args.co_register,
        save_scans_with_skulls=args.save_scans_with_skulls,
        final_dims=dims,
        final_voxels=voxels,
        debug=args.debug,
        patientID=args.patientID,
        timepoint=args.timepoint,
        scanID=args.scanID,
        use_gpu=args.use_gpu,
        enable_tta=args.enable_tta,
        brainmask_path=args.brainmask,
        family_parent_map=family_parent_map,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

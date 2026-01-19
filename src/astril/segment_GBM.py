#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module: segment_GBM
This module runs a pre-specified segmentation pipeline using pre-trained models
on a directory of input scans (GBM segmentation). The pipeline now processes each subject
fully (Model 1 segmentation, remapping, then Model 2 segmentation) before moving on.
Model 2 configuration files are generated on a per‐subject basis (in a subject‐specific
subfolder) after necessary inputs have been generated. The pre‐trained models for each
stage are loaded only once.
"""
import os
import sys
import argparse
import shutil
from pathlib import Path
import configparser
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Optional: keeps editors/type-checkers happy without importing at runtime
    import nibabel as nib  # noqa: F401
    import numpy as np     # noqa: F401

# ------------------------------------------------------------
# Model set specification (single source of truth)
# ------------------------------------------------------------
GBM_V1_SPEC = {
    # name: {plane, model_dir (SavedModel folder name), train_cfg filename}
    "Axial_1":     {"plane": "Axial",    "dir": "Axial_1",    "cfg": "Axial_1_train_parameters.cfg"},
    "Coronal_1":   {"plane": "Coronal",  "dir": "Coronal_1",  "cfg": "Coronal_1_train_parameters.cfg"},
    "Sagittal_1":  {"plane": "Sagittal", "dir": "Sagittal_1", "cfg": "Sagittal_1_train_parameters.cfg"},
    "Axial_2":     {"plane": "Axial",    "dir": "Axial_2",    "cfg": "Axial_2_train_parameters.cfg"},
    "Coronal_2":   {"plane": "Coronal",  "dir": "Coronal_2",  "cfg": "Coronal_2_train_parameters.cfg"},
    "Sagittal_2":  {"plane": "Sagittal", "dir": "Sagittal_2", "cfg": "Sagittal_2_train_parameters.cfg"},
}

def _resolve_gbm_family_root(family: str = "GBM_seg_v1") -> Path:
    """Base directory where the GBM model family lives inside package models/."""
    # Lazy import so CLI help is instant (avoid importing anything heavy at module import time)
    from .models_download import locate_models_dir
    return Path(locate_models_dir()) / family

def _resolve_model_artifacts(names: list[str], family: str = "GBM_seg_v1"):
    """
    For the given list of logical model names (keys in GBM_V1_SPEC),
    return parallel lists: model_paths (dirs/.keras/.h5), train_cfg_paths, planes.
    """
    root = _resolve_gbm_family_root(family)
    model_paths, train_cfgs, planes = [], [], []
    for name in names:
        spec = GBM_V1_SPEC[name]
        model_dir = root / spec["dir"]
        keras_file = (root / f"{spec['dir']}.keras")
        h5_file = (root / f"{spec['dir']}.h5")
        # Prefer directory SavedModel (what your downloader now produces)
        if model_dir.is_dir():
            model_paths.append(str(model_dir))
        elif keras_file.is_file():
            model_paths.append(str(keras_file))
        elif h5_file.is_file():
            model_paths.append(str(h5_file))
        else:
            # Leave an unresolved path to fail fast later
            model_paths.append(str(model_dir))  # expected SavedModel dir
        train_cfgs.append(str(root / spec["cfg"]))
        planes.append(spec["plane"])
    return model_paths, train_cfgs, planes

def _required_gbm_paths(family: str = "GBM_seg_v1") -> tuple[list[Path], list[Path]]:
    """Return (required_model_dirs_or_files, required_cfg_files) for presence checks."""
    root = _resolve_gbm_family_root(family)
    dirs_or_files = [root / spec["dir"] for spec in GBM_V1_SPEC.values()]
    cfgs = [root / spec["cfg"] for spec in GBM_V1_SPEC.values()]
    return dirs_or_files, cfgs

def _ensure_models_available() -> None:
    need_dirs, need_cfgs = _required_gbm_paths("GBM_seg_v1")
    missing = [p for p in (need_dirs + need_cfgs) if not p.exists()]
    if missing:
        target = _resolve_gbm_family_root("GBM_seg_v1")
        items = "\n".join(f"  - {m}" for m in missing)
        print(
            "Required Astril GBM v1 model artifacts are missing:\n"
            f"{items}\n\n"
            "To fetch them, run:\n"
            "  astril-download-models\n\n"
            f"Artifacts are expected under:\n  {target}",
            file=sys.stderr,
        )
        sys.exit(2)

def cleanup_intermediate_files(root_dir):
    """
    Recursively remove all intermediate files from root_dir.
    Files containing any of the specified substrings in their name are removed.
    (Note: segmentation configuration files are removed only when cleaning up the
    overall config working directory.)
    """
    patterns_to_remove = [
        "_Model_1_seg.nii.gz",
        "_Model_1_DB.nii.gz",
        "_Model_2_mask.nii.gz",
        "_Model_2_seg.nii.gz"
    ]
    cfg_extensions = ("parameters.cfg", "segmentation_parameters.cfg")
    
    for file_path in Path(root_dir).rglob("*"):
        if file_path.is_file():
            filename = file_path.name
            # For individual subject cleanup, remove intermediate segmentation files.
            if any(pat in filename for pat in patterns_to_remove):
                try:
                    file_path.unlink()
                    print(f"Removed intermediate file: {file_path}")
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {e}")
            # Also remove config files when cleaning up the overall config directory.
            elif filename.endswith(cfg_extensions):
                try:
                    file_path.unlink()
                    print(f"Removed config file: {file_path}")
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {e}")


###############################################################################
# Helper: Compute the final segmentation file path.
###############################################################################
def compute_final_segmentation_path(mask_path, original_mask_pattern, final_mask_pattern, segment_suffix):
    """
    Given a subject's mask file (mask_path), compute the expected final segmentation
    file path by replacing the original mask pattern or the final mask pattern with
    the segment_suffix.

    Parameters:
      mask_path (str): Full path to the subject's mask file.
      original_mask_pattern (str): The pattern used to create the Model 1 mask file 
          (typically provided via --brainmask_pattern).
      final_mask_pattern (str): The pattern used for Model 2 mask files (e.g. "_Model_2_mask.nii.gz").
      segment_suffix (str): The suffix to use in the final segmentation filename (e.g. "_GBM_seg.nii.gz").

    Returns:
      Path: The full path (as a Path object) to the expected final segmentation file.
      
    Example:
      If mask_path is "074_d_62_E39089072_brainmask.nii.gz", original_mask_pattern is
      "_brainmask.nii.gz", final_mask_pattern is "_Model_2_mask.nii.gz", and segment_suffix is
      "_GBM_seg.nii.gz", then the function returns a Path corresponding to
      "074_d_62_E39089072_GBM_seg.nii.gz".
    """
    base_name = os.path.basename(mask_path)
    if final_mask_pattern in base_name:
        seg_name = base_name.replace(final_mask_pattern, segment_suffix)
    elif original_mask_pattern in base_name:
        seg_name = base_name.replace(original_mask_pattern, segment_suffix)
    else:
        seg_name = base_name.replace(".nii.gz", segment_suffix)
    out_dir = Path(os.path.dirname(mask_path))
    return out_dir / seg_name


###############################################################################
# New helper functions for per-subject processing.
###############################################################################
def process_subject_with_models(seg_config_file, subject_index, loaded_models,
                                slice_batch_size, overwrite, segment_suffix, tiebreaker_model, debug_models):
    """
    Process one subject (identified by its mask file in the segmentation config)
    using the provided pre-loaded models (for one segmentation stage).
    """
    # Lazy imports: avoid TF/nibabel/numpy at module import time
    import nibabel as nib
    import numpy as np
    from .run_segmentation import (
        read_paths_from_file,
        load_val_data,
        ValDataGenerator,
        undo_all_transforms,
        apply_inverse_canonical_4d,
        majority_vote,
    )

    # Parse segmentation config file.
    cp = configparser.ConfigParser()
    cp.read(seg_config_file)
    cfg = cp["DEFAULT"]
    
    # Get file lists for channels and mask.
    channel_cfg_files = cfg["channel_paths_files"].split(",")
    mask_cfg_file = cfg["mask_paths_file"]
    channel_file_lists = [read_paths_from_file(f) for f in channel_cfg_files]
    mask_paths = read_paths_from_file(mask_cfg_file)
    if subject_index >= len(mask_paths):
        raise ValueError("Exam index out of range!")
    volume_paths_list = list(zip(*channel_file_lists))
    volume_paths_list = [list(vp) for vp in volume_paths_list]

    # Get subject's mask file.
    mask_path = mask_paths[subject_index]
    
    # Determine output file name using the provided segment_suffix.
    maskPattern = cfg["maskPattern"]
    base_name = os.path.basename(mask_path)
    if ".nii.gz" in maskPattern:
        seg_name = base_name.replace(maskPattern, segment_suffix)
    else:
        seg_name = base_name.replace(".nii.gz", segment_suffix)
        if seg_name == base_name:
            seg_name += "_seg.nii.gz"
    output_directory = cfg["output_directory"]
    if output_directory == "in_place":
        out_dir = Path(os.path.dirname(mask_path))
    else:
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / seg_name
    # Check if the final output file exists.
    if out_path.exists() and not overwrite:
        print(f"[INFO] Skipping exam {subject_index+1}: final segmentation file {out_path} already exists.")
        return

    # Otherwise, proceed with processing.
    # Load the mask image.
    mask_nib = nib.load(mask_path)
    affine = mask_nib.affine
    (oh, ow, od) = mask_nib.shape

    # Get model parameters from config.
    model_slicing_planes = cfg["model_train_slicing_planes"].split(",")
    model_num_input_slices = list(map(int, cfg["model_train_num_input_slices"].split(",")))
    model_num_output_slices = list(map(int, cfg["model_train_num_output_slices"].split(",")))
    model_min_hw = list(map(int, cfg["model_train_minimum_hw"].split(",")))
    model_num_classes = list(map(int, cfg["model_train_num_classes"].split(",")))
    merging_method = cfg.get("merging_method", "majority_vote")
    
    plane_outputs = []
    for m_idx, model in enumerate(loaded_models):
        plane = model_slicing_planes[m_idx]
        in_sl = model_num_input_slices[m_idx]
        out_sl = model_num_output_slices[m_idx]
        n_cls = model_num_classes[m_idx]
        min_HW = model_min_hw[m_idx]
        print(f"[INFO] Exam {subject_index+1} | Using model {m_idx+1}: plane={plane}, in_slices={in_sl}, out_slices={out_sl}, n_cls={n_cls}, minHW={min_HW}")
        (X_data, _, M_data, z_indices, transform_infos) = load_val_data(
            scan_indexes=[subject_index],
            volume_paths_list=volume_paths_list,
            mask_paths=mask_paths,
            gt_paths=mask_paths,
            slicing_plane=plane,
            num_input_slices=in_sl,
            num_output_slices=out_sl,
            return_transform_info=True,
            target_height=min_HW,
            target_width=min_HW
        )
        mask_info = transform_infos[0]
        val_gen = ValDataGenerator(X_data, None, M_data, slice_batch_size)
        all_preds = []
        for x_batch, _ in val_gen:
            batch_pred = model.predict(x_batch)
            all_preds.append(batch_pred)
        if not all_preds:
            print(f"[WARNING] No predictions for exam {subject_index+1} (model {m_idx+1}).")
            reassembled_4d = np.zeros((oh, ow, od, n_cls), dtype=np.float32)
            reoriented_4d = reassembled_4d
        else:
            all_preds = np.concatenate(all_preds, axis=0)
            (Hpw, Wpw, Dpw) = mask_info['post_alignment_shape']
            reassembled_4d = np.zeros((Hpw, Wpw, Dpw, n_cls), dtype=np.float32)
            half_out = out_sl // 2
            start_out = -half_out
            end_out = start_out + out_sl
            for i_slice in range(len(X_data)):
                z_center = z_indices[i_slice]
                slice_pred = all_preds[i_slice]
                for oi, offset in enumerate(range(start_out, end_out)):
                    z_out = z_center + offset
                    if 0 <= z_out < Dpw:
                        reassembled_4d[..., z_out, :] = slice_pred[..., oi, :]
            unaligned_4d = undo_all_transforms(reassembled_4d, mask_info)
            if 'transform_from_canonical' in mask_info and mask_info['transform_from_canonical'] is not None:
                reoriented_4d = apply_inverse_canonical_4d(unaligned_4d, mask_info['transform_from_canonical'])
            else:
                reoriented_4d = unaligned_4d

        (Xf, Yf, Zf, _) = reoriented_4d.shape
        if (Xf, Yf, Zf) != (oh, ow, od):
            raise ValueError(f"[ERROR] Mismatch after transforms for exam {subject_index+1}! Expected {(oh, ow, od)}, got {(Xf, Yf, Zf)}.")
        mask_original = (mask_nib.get_fdata() > 0.5)
        for c in range(n_cls):
            reoriented_4d[..., c] *= mask_original
        if debug_models:
            dbg_lbl = np.argmax(reoriented_4d, axis=-1).astype(np.uint8)
            dbg_path = out_dir / f"Mod{m_idx+1}_debug.nii.gz"
            nib.save(nib.Nifti1Image(dbg_lbl, affine), str(dbg_path))
            print(f"[DEBUG] Wrote per-model debug label to {dbg_path}")
        plane_outputs.append(reoriented_4d)
    print(f"[INFO] Merging predictions for exam {subject_index+1} via {merging_method}...")
    merged_label = majority_vote(plane_outputs, tiebreaker=tiebreaker_model).astype(np.uint8)
    nib.save(nib.Nifti1Image(merged_label, affine), str(out_path))
    print(f"[INFO] Final segmentation saved: {out_path}")


def segment_GBM_per_subject(input_dir, slice_batch_size=1, n_threads=1,
                             overwrite_existing_outputs=False,
                             channel_patterns=None, brainmask_pattern="_brainmask.nii.gz",
                             segment_suffix="_GBM_seg.nii.gz", debug_models=False):
    """
    Implements the GBM segmentation pipeline per subject:
      1. Create and use a Model 1 segmentation config for all subjects.
      2. Process Model 1 segmentation.
      3. Remap Model 1 outputs.
      4. For each subject, generate a Model 2 config file (using --silent)
         with inputVolumeDirectory set to the subject directory.
      5. Process Model 2 segmentation.
      6. Clean up intermediate files in the subject's directory immediately.
    """
    # Lazy imports: avoid importing these (and their transitive deps) unless we actually run segmentation
    from .create_segmentation_config import (
        create_segmentation_config,
        parse_train_config_for_model_parameters,
    )
    from .remap_gt_classes import remap_gt_classes
    from .run_segmentation import (
        read_paths_from_file,
        load_models_for_config,
    )

    if channel_patterns is None:
        channel_patterns = ["_T1c_brain_norm.nii.gz",
                            "_T1n_brain_norm.nii.gz",
                            "_T2f_brain_norm.nii.gz",
                            "_T2w_brain_norm.nii.gz"]
    channels = ["t1c", "t1n", "t2f", "t2w"]
    
    working_dir = os.path.join(input_dir, "Segmentation_Configs")
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    
    #########################################
    # STEP 1: Prepare segmentation config for Model 1.
    #########################################
    model1_names = ["Axial_1", "Coronal_1", "Sagittal_1"]
    model1_paths, model1_train_configs, model1_planes = _resolve_model_artifacts(model1_names, "GBM_seg_v1")
    seg_config_model1 = create_segmentation_config(
        workingDirectory=working_dir,
        inputChannels=channels,
        channelPatterns=channel_patterns,
        maskPattern=brainmask_pattern,
        model_paths=model1_paths,
        modelTrainConfigFiles=model1_train_configs,
        merging_method="majority_vote",
        inputVolumeDirectory=input_dir,
        outputVolumeDirectory="in_place",
        segmentSuffix="_Model_1_seg.nii.gz",
        output_config_filename="model_1_parameters.cfg",
        silent=False
    )
    print("-------------------------")
    print("[INFO] Prepared segmentation config for Model 1.")
    
    #########################################
    # STEP 2: Load pre-trained models (Model 1 and Model 2).
    #########################################
    print("[INFO] Loading Model 1 weights...")
    loaded_models_model1 = []
    # derive lists needed by the unified loader from the model_1 config we just created
    cp_tmp = configparser.ConfigParser()
    cp_tmp.read(seg_config_model1)
    cfg1 = cp_tmp["DEFAULT"]
    m1_num_in = list(map(int, cfg1["model_train_num_input_slices"].split(",")))
    m1_min_hw = list(map(int, cfg1["model_train_minimum_hw"].split(",")))
    num_modal_channels = len(channels)  # for Model 1
    loaded_models_model1 = load_models_for_config(
        model_paths=model1_paths,
        model_train_config_files=model1_train_configs,
        model_num_input_slices=m1_num_in,
        model_min_hw=m1_min_hw,
        num_modal_channels=num_modal_channels,
        model_call_endpoints=None,
    )
    
    print("[INFO] Loading Model 2 weights...")
    model2_names = ["Axial_2", "Coronal_2", "Sagittal_2"]
    model2_paths, model2_train_configs, model2_planes = _resolve_model_artifacts(model2_names, "GBM_seg_v1")
    # Build a single-subject Model 2 config template (values used for loader dims)
    # We'll still generate per-subject configs later for file lists.
    dummy_cp2 = configparser.ConfigParser()
    # mimic create_segmentation_config scalar arrays for dims; we only need dims here
    # by reading train cfgs (safer) to avoid drifting from training settings
    m2_num_in, m2_min_hw = [], []
    for cfg_path in model2_train_configs:
        params = parse_train_config_for_model_parameters(cfg_path)
        m2_num_in.append(params.get("num_input_slices", 3))
        m2_min_hw.append(params.get("minimum_height_width", 256))
    # Model 2 uses 3 channels: t1c, t2f, mod1DB
    loaded_models_model2 = load_models_for_config(
        model_paths=model2_paths,
        model_train_config_files=model2_train_configs,
        model_num_input_slices=m2_num_in,
        model_min_hw=m2_min_hw,
        num_modal_channels=3,
        model_call_endpoints=None,
    )
    
    #########################################
    # STEP 3: Process each subject sequentially.
    #########################################
    cp_model1 = configparser.ConfigParser()
    cp_model1.read(seg_config_model1)
    mask_cfg_file = cp_model1["DEFAULT"]["mask_paths_file"]
    mask_paths = read_paths_from_file(mask_cfg_file)
    num_subjects = len(mask_paths)
    print(f"[INFO] Found {num_subjects} exam(s) to process in {input_dir}.")
    
    for subj_idx in range(num_subjects):
        # BEFORE ANY PROCESSING: Check if the final segmentation file already exists.
        # Use the original mask pattern (from --brainmask_pattern) and the Model 2 mask pattern.
        subject_mask = mask_paths[subj_idx]
        final_seg_path = compute_final_segmentation_path(subject_mask, brainmask_pattern, "_Model_2_mask.nii.gz", segment_suffix)
        if final_seg_path.exists() and not overwrite_existing_outputs:
            print(f"[INFO] Skipping subject {subj_idx+1}: final segmentation file {final_seg_path} already exists.")
            continue

        print("\n==============================")
        print(f"[INFO] Processing exam {subj_idx+1} of {num_subjects} with Model 1...")
        process_subject_with_models(seg_config_model1, subj_idx, loaded_models_model1,
                                    slice_batch_size, overwrite_existing_outputs, "_Model_1_seg.nii.gz",
                                    tiebreaker_model=0, debug_models=debug_models)
        # Remap Model 1 segmentation outputs in the subject's directory.
        subject_dir = os.path.dirname(mask_paths[subj_idx])
        print("[INFO] Remapping Model 1 segmentation for Model 2 inputs...")
        remap_gt_classes(trainDataDirectory=subject_dir,
                         gtPattern="_Model_1_seg.nii.gz",
                         outputPattern="_Model_2_mask.nii.gz",
                         classRemapDict='{(0,):0,(1,2):1}')
        remap_gt_classes(trainDataDirectory=subject_dir,
                         gtPattern="_Model_1_seg.nii.gz",
                         outputPattern="_Model_1_DB.nii.gz",
                         classRemapDict='{(0,2):0,(1,):1}')
        
        # For Model 2, generate a config file for this subject (using --silent)
        # Use a subject-specific working directory.
        subject_mod2_working_dir = os.path.join(working_dir, "Mod2", f"Exam_{subj_idx+1}")
        Path(subject_mod2_working_dir).mkdir(parents=True, exist_ok=True)
        seg_config_model2 = create_segmentation_config(
            workingDirectory=subject_mod2_working_dir,
            inputChannels=["t1c", "t2f", "mod1DB"],
            channelPatterns=[channel_patterns[0], channel_patterns[2], "_Model_1_DB.nii.gz"],
            maskPattern="_Model_2_mask.nii.gz",
            model_paths=model2_paths,
            modelTrainConfigFiles=model2_train_configs,
            merging_method="majority_vote",
            inputVolumeDirectory=subject_dir,
            outputVolumeDirectory="in_place",
            segmentSuffix=segment_suffix,
            output_config_filename="model_2_parameters.cfg",
            silent=True
        )
        print(f"[INFO] Processing exam {subj_idx+1} with Model 2...")
        # Since the generated Model 2 config corresponds to a single subject, use index 0.
        process_subject_with_models(seg_config_model2, 0, loaded_models_model2,
                                    slice_batch_size, overwrite_existing_outputs, segment_suffix,
                                    tiebreaker_model=0, debug_models=debug_models)
        
        # Clean up intermediate files in the subject's directory immediately.
        print(f"[INFO] Cleaning up intermediate files in exam directory: {subject_dir}")
        cleanup_intermediate_files(subject_dir)
    
    #########################################
    # FINAL STEP: Clean up segmentation configuration files.
    #########################################
    print("[INFO] Cleaning up segmentation configuration files...")
    cleanup_intermediate_files(working_dir)
    print("[INFO] GBM segmentation pipeline complete.")


def segment_GBM(input_dir, slice_batch_size=1, n_threads=1, overwrite_existing_outputs=False,
                channel_patterns=None, brainmask_pattern="_brainmask.nii.gz", segment_suffix="_GBM_seg.nii.gz"):
    """
    Runs the full GBM segmentation pipeline per subject.
    """
    segment_GBM_per_subject(input_dir, slice_batch_size, n_threads,
                            overwrite_existing_outputs, channel_patterns,
                            brainmask_pattern, segment_suffix, debug_models=False)


def main():
    import __main__
    module = getattr(__main__, "__spec__", None)
    prog = f"python -m {module.name}" if module and module.name else None
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run the full GBM segmentation pipeline using pre-trained models. Output segmentation volumes will have 4 levels: 0 = normal brain, 1 = tumor, 2 = surgical artefact, 3 = edema"
    )
    parser.add_argument("input_directory",
                        help="Directory containing input scans for segmentation.")
    parser.add_argument("--slice_batch_size", type=int, default=1,
                        help="Slice batch size for segmentation (default 1).")
    parser.add_argument("--n_threads", type=int, default=1,
                        help="Number of threads to use when quantifying volumes after segmentation (default 1).")
    parser.add_argument("--overwrite_existing_outputs", action="store_true",
                        help="Overwrite existing segmentation outputs if they exist.")
    parser.add_argument("--channel_patterns", nargs="+",
                        help=("List of filename patterns for the input scans (in order: T1-post, T1-pre, T2-FLAIR, and T2). "
                              "Default: _T1c_brain_norm.nii.gz _T1n_brain_norm.nii.gz _T2f_brain_norm.nii.gz _T2w_brain_norm.nii.gz"))
    parser.add_argument("--brainmask_pattern", type=str, default="_brainmask.nii.gz",
                        help="Brainmask pattern for Model 1 (default: _brainmask.nii.gz)")
    parser.add_argument("--segment_suffix", type=str, default="_GBM_seg.nii.gz",
                        help="Suffix to use in the final segmentation file names (default: _GBM_seg.nii.gz)")
    args = parser.parse_args()

    # Fail early with a clear instruction if user hasn't fetched models yet
    _ensure_models_available()
    
    segment_GBM(
        input_dir=args.input_directory,
        slice_batch_size=args.slice_batch_size,
        n_threads=args.n_threads,
        overwrite_existing_outputs=args.overwrite_existing_outputs,
        channel_patterns=args.channel_patterns,
        brainmask_pattern=args.brainmask_pattern,
        segment_suffix=args.segment_suffix
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module: segment_GBM
Runs the GBM_seg_v2 segmentation pipeline: a Model A three-plane gating pass (average-probability
consensus) whose output is thresholded, dilated, and intersected with the brainmask into a spatial
gate, followed by a Model B three-plane five-class consensus (average-logit) constrained to that
gate. Each subject is processed fully (Model A, gate derivation, Model B) before moving on. Model
B's per-subject configuration files are generated in a subject-specific subfolder after the gate is
available. Each stage's pre-trained models are loaded once, not per subject.
"""
import os
import argparse
import shutil
from pathlib import Path
import configparser
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Optional: keeps editors/type-checkers happy without importing at runtime
    import nibabel as nib  # noqa: F401
    import numpy as np     # noqa: F401

MISSING_CHANNEL_SENTINEL = "__MISSING__"

# ------------------------------------------------------------
# Model set specification (single source of truth)
# ------------------------------------------------------------
 # Model A is the binary abnormality detector used to create the spatial gate.
 # Model B is the five-class, normalized-MRI consensus selected for the final
 # segmentation. Both are deployed together under GBM_seg_v2 so the gate and
 # final model use the same normalized-input experiment family.
GBM_MODEL_A_SPEC = {
    "Axial":    {"plane": "Axial",    "dir": "Model_A_Axial",    "cfg": "Model_A_Axial_train_parameters.cfg"},
    "Coronal":  {"plane": "Coronal",  "dir": "Model_A_Coronal",  "cfg": "Model_A_Coronal_train_parameters.cfg"},
    "Sagittal": {"plane": "Sagittal", "dir": "Model_A_Sagittal", "cfg": "Model_A_Sagittal_train_parameters.cfg"},
}
GBM_MODEL_B_FIVE_CLASS_SPEC = {
    "Axial":    {"plane": "Axial",    "dir": "Model_B_Axial",    "cfg": "Model_B_Axial_train_parameters.cfg"},
    "Coronal":  {"plane": "Coronal",  "dir": "Model_B_Coronal",  "cfg": "Model_B_Coronal_train_parameters.cfg"},
    "Sagittal": {"plane": "Sagittal", "dir": "Model_B_Sagittal", "cfg": "Model_B_Sagittal_train_parameters.cfg"},
}

def _resolve_gbm_family_root(family: str) -> Path:
    """Base directory where the GBM model family lives inside package models/."""
    # Lazy import so CLI help is instant (avoid importing anything heavy at module import time)
    from .models_download import locate_models_dir
    return Path(locate_models_dir()) / family

def _resolve_model_artifacts(names: list[str], family: str, specs: dict):
    """
    For the given list of logical model names (keys in `specs`), return parallel lists:
    model_paths (.pt), train_cfg_paths, planes.
    """
    root = _resolve_gbm_family_root(family)
    model_paths, train_cfgs, planes = [], [], []
    for name in names:
        spec = specs[name]
        pt_file = root / f"{spec['dir']}.pt"
        nested_pt_file = root / spec["dir"] / f"{spec['dir']}.pt"
        if pt_file.is_file():
            model_paths.append(str(pt_file))
        elif nested_pt_file.is_file():
            model_paths.append(str(nested_pt_file))
        else:
            # Leave an unresolved path to fail fast later
            model_paths.append(str(pt_file))
        train_cfgs.append(str(root / spec["cfg"]))
        planes.append(spec["plane"])
    return model_paths, train_cfgs, planes

def _ensure_model_b_five_class_available() -> None:
    root = _resolve_gbm_family_root("GBM_seg_v2")
    missing = []
    for family_name, specs in (("Model A", GBM_MODEL_A_SPEC), ("Model B", GBM_MODEL_B_FIVE_CLASS_SPEC)):
        for spec in specs.values():
            if not ((root / f"{spec['dir']}.pt").exists() or (root / spec["dir"] / f"{spec['dir']}.pt").exists()):
                missing.append(root / f"{spec['dir']}.pt")
            cfg = root / spec["cfg"]
            if not cfg.exists():
                missing.append(cfg)
    if missing:
        items = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required Model A and five-class Model B artifacts are missing from Astril's GBM_seg_v2 family:\n"
            f"{items}\n\nRun `astril-download-models --family GBM_seg_v2` after the GBM_seg_v2 archive is registered in the model manifest."
        )


def _write_segmentation_provenance(subject_dir, model_family: str) -> None:
    """
    Write a `segmentation_provenance.json` sidecar into `subject_dir` recording which model
    family produced the segmentation output(s) there, and that family's level->label mapping
    (if that family's pipeline.json declares any -- see `model_labels.py`). `quantify_volumes.py`
    picks this up automatically to label volume columns meaningfully and to know which levels
    a model can produce even if this particular exam happens to have zero voxels at one of them.
    """
    import json
    from .model_labels import load_model_labels

    labels = load_model_labels(model_family)
    payload = {
        "model_family": model_family,
        "labels": {str(level): name for level, name in labels.items()} if labels else {},
    }
    out_path = Path(subject_dir) / "segmentation_provenance.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not write segmentation provenance to {out_path}: {e}")


def _print_model_b_citation_notice() -> None:
    """Surface the model/data citation requirements at segmentation time."""
    citation_path = _resolve_gbm_family_root("GBM_seg_v2") / "CITATIONS.md"
    print("[NOTICE] GBM_seg_v2 uses models trained on the BraTS 2024 BraTS-GLI TrainingData release.")
    print(f"[NOTICE] Required model/data citations and use conditions: {citation_path}")

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
        "_Model_2_seg.nii.gz",
        "_Model_A_seg.nii.gz",
        "_Model_A_gate.nii.gz"
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
      segment_suffix (str): The suffix to use in the final segmentation filename (e.g. "_GBM-seg.nii.gz").

    Returns:
      Path: The full path (as a Path object) to the expected final segmentation file.
      
    Example:
      If mask_path is "074_d_62_E12345_brainmask.nii.gz", original_mask_pattern is
      "_brainmask.nii.gz", final_mask_pattern is "_Model_2_mask.nii.gz", and segment_suffix is
      "_GBM-seg.nii.gz", then the function returns a Path corresponding to
      "074_d_62_E12345_GBM-seg.nii.gz".
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
                                slice_batch_size, overwrite, segment_suffix, tiebreaker_model, debug_models,
                                extra_channel_paths=None, brainiac_paths_list=None,
                                return_merged=False):
    """
    Process one subject (identified by its mask file in the segmentation config)
    using the provided pre-loaded models (for one segmentation stage).
    extra_channel_paths: optional list of additional channel NIfTI paths (e.g. BrainIAC PCA maps)
    appended to the subject's channel list at inference time.
    """
    # Lazy imports: avoid nibabel/numpy/torch at module import time
    import torch
    import torch.nn.functional as F
    import nibabel as nib
    import numpy as np
    from .data_loading import (
        read_paths_from_file,
        load_val_data,
        ValDataGenerator,
        undo_all_transforms,
        apply_inverse_canonical_4d,
    )
    from .run_segmentation import majority_vote, average_prob, average_logit, max_prob

    # Parse segmentation config file.
    cp = configparser.ConfigParser()
    cp.read(seg_config_file)
    cfg = cp["DEFAULT"]
    
    # Get file lists for channels and mask.
    channel_cfg_files = cfg["channel_paths_files"].split(",")
    mask_cfg_file = cfg["mask_paths_file"]
    channel_file_lists = [read_paths_from_file(f) for f in channel_cfg_files]
    channel_names = [
        x.strip() for x in cfg.get("channel_names", "").split(",") if x.strip()
    ] or [f"ch{i}" for i in range(len(channel_cfg_files))]
    optional_channels = {
        x.strip() for x in cfg.get("optional_channels", "").split(",") if x.strip()
    }
    mask_paths = read_paths_from_file(mask_cfg_file)
    if subject_index >= len(mask_paths):
        raise ValueError("Exam index out of range!")
    volume_paths_list = list(zip(*channel_file_lists))
    volume_paths_list = [list(vp) for vp in volume_paths_list]

    # Append extra channels (e.g. BrainIAC PCA maps) if provided
    if extra_channel_paths:
        volume_paths_list[subject_index] = volume_paths_list[subject_index] + list(extra_channel_paths)
    missing_channels = [
        channel_names[i]
        for i, path in enumerate(volume_paths_list[subject_index][:len(channel_names)])
        if str(path).strip() == MISSING_CHANNEL_SENTINEL
    ]
    missing_required = [ch for ch in missing_channels if ch not in optional_channels]
    if missing_required:
        raise ValueError(f"Exam {subject_index+1} has missing required channel(s): {missing_required}")
    if missing_channels:
        print(f"[INFO] Missing optional channel(s) zero-filled: {missing_channels}")

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
    merging_method = cfg.get("merging_method", "average_logit")
    merging_weights_raw = cfg.get("merging_weights", "").strip()
    merging_weights = (
        [float(value.strip()) for value in merging_weights_raw.split(",") if value.strip()]
        if merging_weights_raw
        else None
    )
    
    plane_outputs = []
    for m_idx, model in enumerate(loaded_models):
        plane = model_slicing_planes[m_idx]
        in_sl = model_num_input_slices[m_idx]
        out_sl = model_num_output_slices[m_idx]
        n_cls = model_num_classes[m_idx]
        min_HW = model_min_hw[m_idx]
        print(f"[INFO] Exam {subject_index+1} | Using model {m_idx+1}: plane={plane}, in_slices={in_sl}, out_slices={out_sl}, n_cls={n_cls}, minHW={min_HW}")
        model_uses_brainiac_fusion = getattr(model, "brainiac_input_channels", 0) > 0
        model_uses_presence = bool(getattr(model, "uses_modality_presence", False))
        val_data = load_val_data(
            idx=subject_index,
            volume_paths_list=volume_paths_list,
            mask_paths=mask_paths,
            gt_paths=mask_paths,
            slicing_plane=plane,
            num_input_slices=in_sl,
            num_output_slices=out_sl,
            return_transform_info=True,
            target_height=min_HW,
            target_width=min_HW,
            brainiac_paths_list=brainiac_paths_list if model_uses_brainiac_fusion else None,
            append_modality_presence=model_uses_presence,
        )
        if model_uses_brainiac_fusion:
            (X_data, B_data, _, M_data, z_indices, transform_infos) = val_data
        else:
            (X_data, _, M_data, z_indices, transform_infos) = val_data
        # ``load_val_data`` returns one transform-info mapping for the volume;
        # older versions returned a one-element list. Support both contracts.
        mask_info = transform_infos[0] if isinstance(transform_infos, (list, tuple)) else transform_infos
        val_gen = ValDataGenerator(
            X_data, None, M_data, slice_batch_size,
            brainiac_data=B_data if model_uses_brainiac_fusion else None,
        )
        all_preds = []
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for batch in val_gen:
                if model_uses_brainiac_fusion:
                    x_batch, b_batch, _ = batch
                else:
                    x_batch, _ = batch
                x_tensor = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
                x_tensor = x_tensor.permute(0, 3, 1, 2).contiguous()
                if model_uses_brainiac_fusion:
                    b_tensor = torch.as_tensor(b_batch, dtype=torch.float32, device=device)
                    b_tensor = b_tensor.permute(0, 3, 1, 2).contiguous()
                    batch_logits = model(x_tensor, b_tensor)
                else:
                    batch_logits = model(x_tensor)
                if isinstance(batch_logits, tuple):
                    batch_logits = batch_logits[0]
                all_preds.append(
                    batch_logits.float().cpu().numpy()
                    if merging_method == "average_logit"
                    else F.softmax(batch_logits.float(), dim=-1).cpu().numpy()
                )
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
    merged_probabilities = None
    if merging_method == "majority_vote":
        merged_label = majority_vote(plane_outputs, tiebreaker=tiebreaker_model).astype(np.uint8)
    elif merging_method == "average_prob":
        merged_probabilities = np.mean(np.stack(plane_outputs, axis=0), axis=0)
        merged_label = average_prob(plane_outputs, tiebreaker=tiebreaker_model).astype(np.uint8)
    elif merging_method == "max_prob":
        merged_label = max_prob(plane_outputs, tiebreaker=tiebreaker_model).astype(np.uint8)
    elif merging_method == "average_logit":
        merged_label = average_logit(
            plane_outputs,
            weights=merging_weights,
            tiebreaker=tiebreaker_model,
        ).astype(np.uint8)
    else:
        raise ValueError(f"Unknown merging method '{merging_method}'")
    nib.save(nib.Nifti1Image(merged_label, affine), str(out_path))
    print(f"[INFO] Final segmentation saved: {out_path}")
    if return_merged:
        return merged_label, merged_probabilities, affine
    return None


def _write_model_a_gate(probabilities, mask_path, output_path, threshold=0.30, dilation_voxels=5):
    """Write the Model A consensus foreground gate used by five-class Model B."""
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import binary_dilation

    mask_image = nib.load(str(mask_path))
    mask = mask_image.get_fdata() > 0.5
    if probabilities is None or probabilities.ndim != 4:
        raise ValueError("Model A probability consensus must be a 4-D array")
    if probabilities.shape[:3] != mask.shape:
        raise ValueError(f"Model A probability/mask shape mismatch: {probabilities.shape[:3]} vs {mask.shape}")
    foreground_probability = 1.0 - probabilities[..., 0]
    gate = foreground_probability >= float(threshold)
    if dilation_voxels > 0:
        gate = binary_dilation(gate, structure=np.ones((3, 3, 3), dtype=bool), iterations=dilation_voxels)
    gate &= mask
    nib.save(nib.Nifti1Image(gate.astype(np.uint8), mask_image.affine, mask_image.header), str(output_path))
    return output_path


def _segment_GBM_model_b_five_class(input_dir, slice_batch_size=1, overwrite_existing_outputs=False,
                                    channel_patterns=None, brainmask_pattern="_brainmask.nii.gz",
                                    segment_suffix="_GBM-seg.nii.gz", optional_channels=None,
                                    model_a_threshold=0.30, model_a_dilation=5):
    """Run Model A gating followed by the selected five-class Model B consensus."""
    from .create_segmentation_config import create_segmentation_config
    from .data_loading import read_paths_from_file
    from .run_segmentation import load_models_for_config

    if channel_patterns is None:
        channel_patterns = ["_T1c_brain-norm.nii.gz", "_T1n_brain-norm.nii.gz",
                            "_T2f_brain-norm.nii.gz", "_T2w_brain-norm.nii.gz"]
    channels = ["t1c", "t1n", "t2f", "t2w"]
    # Preserve None (not [] ) when the caller doesn't specify anything, so
    # create_segmentation_config's own per-model auto-inference (reading each checkpoint's
    # `optional_channels` from its train_parameters.cfg -- GBM_seg_v2 was trained with channel
    # dropout and only requires t1c) actually runs instead of being short-circuited.
    optional_channels = list(optional_channels) if optional_channels is not None else None
    working_dir = os.path.join(input_dir, "Segmentation_Configs")
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    model_a_names = ["Axial", "Coronal", "Sagittal"]
    model_a_paths, model_a_cfgs, _ = _resolve_model_artifacts(model_a_names, "GBM_seg_v2", GBM_MODEL_A_SPEC)
    model_a_config = create_segmentation_config(
        workingDirectory=working_dir, inputChannels=channels,
        channelPatterns=channel_patterns, maskPattern=brainmask_pattern,
        model_paths=model_a_paths, modelTrainConfigFiles=model_a_cfgs,
        merging_method="average_prob", inputVolumeDirectory=input_dir,
        outputVolumeDirectory="in_place", segmentSuffix="_Model_A_seg.nii.gz",
        output_config_filename="model_a_parameters.cfg", silent=False,
        optional_channels=optional_channels,
        allow_missing_optional_channels=(bool(optional_channels) if optional_channels is not None else None),
    )
    model_a_cp = configparser.ConfigParser()
    model_a_cp.read(model_a_config)
    model_a_cfg = model_a_cp["DEFAULT"]
    model_a_num_in = list(map(int, model_a_cfg["model_train_num_input_slices"].split(",")))
    model_a_min_hw = list(map(int, model_a_cfg["model_train_minimum_hw"].split(",")))
    loaded_model_a = load_models_for_config(
        model_paths=model_a_paths, model_train_config_files=model_a_cfgs,
        model_num_input_slices=model_a_num_in, model_min_hw=model_a_min_hw,
        num_modal_channels=len(channels),
    )

    _ensure_model_b_five_class_available()
    _print_model_b_citation_notice()
    model_b_names = ["Axial", "Coronal", "Sagittal"]
    model_b_paths, model_b_cfgs, _ = _resolve_model_artifacts(model_b_names, "GBM_seg_v2", GBM_MODEL_B_FIVE_CLASS_SPEC)
    model_b_cfg_values = []
    for cfg_path in model_b_cfgs:
        cp = configparser.ConfigParser()
        cp.read(cfg_path)
        values = cp["DEFAULT"]
        model_b_cfg_values.append((int(values.get("num_input_slices", 7)), int(values.get("minimum_height_width", 256))))
    loaded_model_b = load_models_for_config(
        model_paths=model_b_paths, model_train_config_files=model_b_cfgs,
        model_num_input_slices=[v[0] for v in model_b_cfg_values],
        model_min_hw=[v[1] for v in model_b_cfg_values], num_modal_channels=len(channels),
    )

    mask_paths = read_paths_from_file(model_a_cfg["mask_paths_file"])
    print(f"[INFO] Found {len(mask_paths)} exam(s) in {input_dir}.")
    for subject_index, subject_mask in enumerate(mask_paths):
        final_seg_path = compute_final_segmentation_path(subject_mask, brainmask_pattern, "_Model_A_gate.nii.gz", segment_suffix)
        if final_seg_path.exists() and not overwrite_existing_outputs:
            print(f"[INFO] Skipping subject {subject_index + 1}: final segmentation file {final_seg_path} already exists.")
            continue
        print(f"\n==============================\n[INFO] Processing exam {subject_index + 1} of {len(mask_paths)} with Model A consensus...")
        model_a_result = process_subject_with_models(
            model_a_config, subject_index, loaded_model_a, slice_batch_size,
            True, "_Model_A_seg.nii.gz", tiebreaker_model=0,
            debug_models=False, return_merged=True,
        )
        _, model_a_probabilities, _ = model_a_result
        subject_dir = os.path.dirname(subject_mask)
        gate_path = Path(subject_dir) / (Path(subject_mask).name.replace(brainmask_pattern, "_Model_A_gate.nii.gz"))
        _write_model_a_gate(model_a_probabilities, subject_mask, gate_path, model_a_threshold, model_a_dilation)
        print(f"[INFO] Model A gate threshold={model_a_threshold:.2f}, dilation={model_a_dilation}: {gate_path}")

        model_b_working_dir = os.path.join(working_dir, "Model_B", f"Exam_{subject_index + 1}")
        model_b_config = create_segmentation_config(
            workingDirectory=model_b_working_dir, inputChannels=channels,
            channelPatterns=channel_patterns, maskPattern="_Model_A_gate.nii.gz",
            model_paths=model_b_paths, modelTrainConfigFiles=model_b_cfgs,
            merging_method="average_logit", inputVolumeDirectory=subject_dir,
            outputVolumeDirectory="in_place", segmentSuffix=segment_suffix,
            output_config_filename="model_b_parameters.cfg", silent=True,
            optional_channels=optional_channels,
            allow_missing_optional_channels=(bool(optional_channels) if optional_channels is not None else None),
        )
        print(f"[INFO] Processing exam {subject_index + 1} with five-class Model B average-logit consensus...")
        process_subject_with_models(
            model_b_config, 0, loaded_model_b, slice_batch_size,
            overwrite_existing_outputs, segment_suffix, tiebreaker_model=0,
            debug_models=False,
        )
        _write_segmentation_provenance(subject_dir, "GBM_seg_v2")
        cleanup_intermediate_files(subject_dir)
    cleanup_intermediate_files(working_dir)
    print("[INFO] Model A-gated five-class Model B segmentation complete.")


def segment_GBM(input_dir, slice_batch_size=1, n_threads=1, overwrite_existing_outputs=False,
                channel_patterns=None, brainmask_pattern="_brainmask.nii.gz", segment_suffix="_GBM-seg.nii.gz",
                optional_channels=None):
    """
    Runs Model A three-plane gating followed by five-class Model B consensus.

    Model A uses average probabilities to form a foreground gate at 0.30,
    dilated by five voxels and intersected with the brain mask. Model B uses
    the selected Axial/Coronal/Sagittal checkpoints with average-logit fusion.
    """
    _segment_GBM_model_b_five_class(
        input_dir, slice_batch_size=slice_batch_size,
        overwrite_existing_outputs=overwrite_existing_outputs,
        channel_patterns=channel_patterns, brainmask_pattern=brainmask_pattern,
        segment_suffix=segment_suffix, optional_channels=optional_channels,
    )


def main():
    import __main__
    module = getattr(__main__, "__spec__", None)
    prog = f"python -m {module.name}" if module and module.name else None
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run the full GBM segmentation pipeline using pre-trained models (GBM_seg_v2: "
                    "Model A three-plane gating + five-class Model B consensus). Output segmentation "
                    "labels are model-specific -- see the README bundled with the model weights, or "
                    "astril.model_labels.load_model_labels('GBM_seg_v2'), for what each non-zero "
                    "level means. A segmentation_provenance.json sidecar recording the model family "
                    "and its labels is written alongside each exam's output."
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
                              "Default: _T1c_brain-norm.nii.gz _T1n_brain-norm.nii.gz _T2f_brain-norm.nii.gz _T2w_brain-norm.nii.gz"))
    parser.add_argument("--brainmask_pattern", type=str, default="_brainmask.nii.gz",
                        help="Brainmask pattern for Model 1 (default: _brainmask.nii.gz)")
    parser.add_argument("--segment_suffix", type=str, default="_GBM-seg.nii.gz",
                        help="Suffix to use in the final segmentation file names (default: _GBM-seg.nii.gz)")
    parser.add_argument("--optional_channels", nargs="*", default=None,
                        help="Model 1 channel names that may be absent and zero-filled, e.g. t1n t2f t2w.")
    args = parser.parse_args()

    # Fail early with a clear instruction if the v2 gate/segmentation models
    # have not been fetched yet. The legacy GBM_seg_v1 family is not required
    # by this entry point.
    _ensure_model_b_five_class_available()
    
    segment_GBM(
        input_dir=args.input_directory,
        slice_batch_size=args.slice_batch_size,
        n_threads=args.n_threads,
        overwrite_existing_outputs=args.overwrite_existing_outputs,
        channel_patterns=args.channel_patterns,
        brainmask_pattern=args.brainmask_pattern,
        segment_suffix=args.segment_suffix,
        optional_channels=args.optional_channels,
    )


if __name__ == "__main__":
    main()

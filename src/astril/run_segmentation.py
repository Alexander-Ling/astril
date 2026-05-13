#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module: run_segmentation
This module runs the segmentation process using a 2.5D pipeline and PyTorch
.pt checkpoints.
"""

import os
import sys
import argparse
import configparser
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path

# Import data loading functions from your package.
from .data_loading import (
    read_paths_from_file,
    load_val_data,
    ValDataGenerator,
    undo_all_transforms,
    apply_inverse_canonical_4d
)

# Import model architecture and custom layers.
from .model_architecture import (
    DynamicAttentionResUNet,
    create_dynamic_unet_from_metadata,
)

########################################################################
# Shared helper: load a list of models given config-derived arrays
########################################################################
def load_models_for_config(
    model_paths: list[str],
    model_train_config_files: list[str],
    model_num_input_slices: list[int],
    model_min_hw: list[int],
    num_modal_channels: int,
    device: torch.device | None = None,
):
    """
    Unified model loader used by run_segmentation.py and segment_GBM.py.
    Supports only migrated PyTorch .pt checkpoints.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = []
    for i, mp in enumerate(model_paths):
        mp = mp.strip()
        print(f"[INFO] Loading model {i+1}/{len(model_paths)} => {mp}")
        if not mp.lower().endswith(".pt"):
            raise ValueError(
                f"Unsupported model path '{mp}'. Migrated astril inference expects PyTorch .pt checkpoints."
            )
        if not os.path.isfile(mp):
            raise FileNotFoundError(f"PyTorch model checkpoint not found: {mp}")
        try:
            checkpoint = torch.load(mp, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(mp, map_location=device)
        if "architecture" in checkpoint:
            model = create_dynamic_unet_from_metadata(checkpoint["architecture"])
            state_dict = checkpoint.get("model_state_dict")
        else:
            train_cfg = model_train_config_files[i].strip()
            model = build_model_from_train_config(train_cfg, num_modal_channels=num_modal_channels)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        if state_dict is None:
            raise ValueError(f"Checkpoint '{mp}' does not contain model_state_dict.")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        loaded.append(model)

    return loaded

########################################################################
# Helper: Build model from a model training config file.
########################################################################
def build_model_from_train_config(config_path, num_modal_channels=None):
    """
    Reads a model training config file (INI format) and builds a U-Net model
    with the parameters specified.
    """
    cp = configparser.ConfigParser()
    cp.read(config_path)
    cfg = cp["DEFAULT"]

    num_input_slices = cfg.getint("num_input_slices", fallback=3)
    num_output_slices = cfg.getint("num_output_slices", fallback=1)
    num_classes = cfg.getint("num_classes", fallback=2)
    base_num_filters = cfg.getint("base_num_filters", fallback=32)
    encoder_level_factors = [int(x.strip()) for x in cfg.get("encoder_level_factors", fallback="1,2,4,8").split(",") if x.strip()]
    center_depth = cfg.getint("center_depth", fallback=1)
    
    if num_modal_channels is not None:
        num_channels = int(num_modal_channels)
    else:
        # Try to get num_channels from the config; otherwise, infer from image_paths_files.
        num_channels_val = cfg.get("num_channels", None)
        if num_channels_val is None or num_channels_val.strip() == "":
            ips = cfg.get("image_paths_files", "")
            if ips.strip():
                num_channels = len(ips.split(","))
            else:
                num_channels = 1  # fallback default
        else:
            num_channels = int(num_channels_val)

    if num_channels < 1:
        raise ValueError(f"Invalid num_channels={num_channels} while building model from {config_path}")

    input_channels = num_channels * num_input_slices

    use_se_blocks = cfg.getboolean("use_se_blocks", fallback=False)
    use_deep_supervision = cfg.getboolean("use_deep_supervision", fallback=False)

    model = DynamicAttentionResUNet(
        input_channels=input_channels,
        base_num_filters=base_num_filters,
        encoder_level_factors=encoder_level_factors,
        num_output_slices=num_output_slices,
        out_channels=num_classes,
        center_depth=center_depth,
        use_se_blocks=use_se_blocks,
        use_deep_supervision=use_deep_supervision,
    )
    return model


def _train_config_requires_brainiac(config_path: str) -> bool:
    """Returns True if the given training config declares use_brainiac_embeddings = true."""
    cp = configparser.ConfigParser()
    cp.read(config_path)
    return cp["DEFAULT"].getboolean("use_brainiac_embeddings", fallback=False)


########################################################################
# Merging helper functions.
########################################################################
def majority_vote(softmax_list, tiebreaker=0):
    label_vols = [np.argmax(s, axis=-1) for s in softmax_list]
    stacked = np.stack(label_vols, axis=0)
    num_models, H, W, D = stacked.shape
    max_label = stacked.max()
    n_cls = max_label + 1

    counts = np.zeros((n_cls, H, W, D), dtype=np.int32)
    for i in range(num_models):
        vol_1d = stacked[i].ravel()
        idx = np.arange(vol_1d.size, dtype=np.int32)
        np.add.at(counts.reshape(n_cls, -1), (vol_1d, idx), 1)

    best_label = counts.argmax(axis=0)
    best_count = counts.max(axis=0)
    tie_mask = (counts == best_count).sum(axis=0) > 1
    if tiebreaker > 0:
        tiebreaker_index = tiebreaker - 1
        best_label[tie_mask] = stacked[tiebreaker_index][tie_mask]
    else:
        best_label[tie_mask] = 0
    return best_label

def average_prob(softmax_list, tiebreaker=0):
    stacked = np.stack(softmax_list, axis=0)
    sum_probs = np.sum(stacked, axis=0)
    best_label = np.argmax(sum_probs, axis=-1)
    best_val = np.max(sum_probs, axis=-1, keepdims=True)
    tie_mask = np.isclose(sum_probs, best_val).sum(axis=-1) > 1
    if tiebreaker > 0:
        tiebreaker_index = tiebreaker - 1
        tiebreaker_probs = stacked[tiebreaker_index]
        best_label[tie_mask] = np.argmax(tiebreaker_probs, axis=-1)[tie_mask]
    else:
        best_label[tie_mask] = 0
    return best_label

def max_prob(softmax_list, tiebreaker=0):
    stacked = np.stack(softmax_list, axis=0)
    max_probs = np.max(stacked, axis=0)
    best_label = np.argmax(max_probs, axis=-1)
    best_val = np.max(max_probs, axis=-1, keepdims=True)
    tie_mask = np.isclose(max_probs, best_val).sum(axis=-1) > 1
    if tiebreaker > 0:
        tiebreaker_index = tiebreaker - 1
        tiebreaker_probs = stacked[tiebreaker_index]
        best_label[tie_mask] = np.argmax(tiebreaker_probs, axis=-1)[tie_mask]
    else:
        best_label[tie_mask] = 0
    return best_label


########################################################################
# Main segmentation function.
########################################################################
def run_segmentation(
    segmentation_config_file,
    slice_batch_size=1,
    overwrite=False,
    tiebreaker_model=0,
    debug_models=False
):
    force_cpu = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    print(f"[INFO] PyTorch device: {device}")

    # --- A) Parse segmentation config and load paths ---
    if not os.path.isfile(segmentation_config_file):
        raise FileNotFoundError(f"Config file not found: {segmentation_config_file}")

    config_parser = configparser.ConfigParser()
    if not config_parser.read(segmentation_config_file):
        raise ValueError(f"Failed to read config file '{segmentation_config_file}'.")

    channel_cfg_files_str = config_parser["DEFAULT"]["channel_paths_files"]
    channel_cfg_files = channel_cfg_files_str.split(",")
    mask_cfg_file = config_parser["DEFAULT"]["mask_paths_file"]

    model_paths_str = config_parser["DEFAULT"]["model_paths"]
    merging_method = config_parser["DEFAULT"].get("merging_method", "majority_vote")
    output_directory = config_parser["DEFAULT"]["output_directory"]

    model_slicing_planes = config_parser["DEFAULT"]["model_train_slicing_planes"].split(",")
    model_num_input_slices = list(map(int, config_parser["DEFAULT"]["model_train_num_input_slices"].split(",")))
    model_num_output_slices = list(map(int, config_parser["DEFAULT"]["model_train_num_output_slices"].split(",")))
    model_min_hw = list(map(int, config_parser["DEFAULT"]["model_train_minimum_hw"].split(",")))
    model_num_classes = list(map(int, config_parser["DEFAULT"]["model_train_num_classes"].split(",")))
    model_train_config_files = config_parser["DEFAULT"]["model_train_config_files"].split(",")

    model_paths = model_paths_str.split(",")
    if len(model_paths) != len(model_slicing_planes):
        raise ValueError("Mismatch between the number of model paths and model_train_slicing_planes.")

    maskPattern = config_parser["DEFAULT"]["maskpattern"]
    segmentSuffix = config_parser["DEFAULT"]["segmentsuffix"]

    if tiebreaker_model < 0 or tiebreaker_model > len(model_paths):
        raise ValueError(f"Invalid tiebreaker_model={tiebreaker_model}; must be in [0, {len(model_paths)}].")

    # Read channel and mask paths.
    channel_file_lists = []
    for chf in channel_cfg_files:
        cpaths = read_paths_from_file(chf)
        channel_file_lists.append(cpaths)
    mask_paths = read_paths_from_file(mask_cfg_file)

    num_subjects = len(mask_paths)
    if num_subjects == 0:
        raise ValueError(f"No exams found in {mask_cfg_file}. Check that it's not empty!")
    for cfiles in channel_file_lists:
        if len(cfiles) != num_subjects:
            raise ValueError("Mismatch in number of lines across channel cfg and mask cfg.")

    volume_paths_list = list(zip(*channel_file_lists))
    volume_paths_list = [list(vp) for vp in volume_paths_list]

    # --- B) Load models ---
    # How many MRI channels per sample (e.g., T1n, T1c, T2f, ...)
    num_modal_channels = len(channel_file_lists)
    loaded_models = load_models_for_config(
        model_paths=model_paths,
        model_train_config_files=model_train_config_files,
        model_num_input_slices=model_num_input_slices,
        model_min_hw=model_min_hw,
        num_modal_channels=num_modal_channels,
        device=device,
    )

    # --- C) Detect BrainIAC dependency and prepare weights/temp dir ---
    brainiac_needed = any(
        _train_config_requires_brainiac(tcfg) for tcfg in model_train_config_files
    )
    brainiac_weights_path = None
    brainiac_tmp_dir = None
    if brainiac_needed:
        from .brainiac_utils import (
            ensure_brainiac_weights,
            compute_brainiac_saliency_maps,
            BrainIACWeightsNotFoundError,
        )
        import tempfile
        try:
            brainiac_weights_path = ensure_brainiac_weights(weights_path=None)
        except BrainIACWeightsNotFoundError as e:
            raise RuntimeError(str(e))
        brainiac_tmp_dir = Path(tempfile.mkdtemp(prefix="astril_brainiac_"))
        print(f"[brainiac] BrainIAC embeddings required. Saliency maps will be cached at: {brainiac_tmp_dir}")

    # --- D) Define merging function ---
    def merge_predictions(volumes_4d, method=merging_method):
        if method == "majority_vote":
            return majority_vote(volumes_4d, tiebreaker=tiebreaker_model)
        elif method == "average_prob":
            return average_prob(volumes_4d, tiebreaker=tiebreaker_model)
        elif method == "max_prob":
            return max_prob(volumes_4d, tiebreaker=tiebreaker_model)
        else:
            raise ValueError(f"Unknown merging method '{method}'")

    # --- E) Process each subject ---
    for subj_idx in range(num_subjects):
        mask_path = mask_paths[subj_idx]
        if output_directory == "in_place":
            out_dir = Path(os.path.dirname(mask_path))
        else:
            out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = os.path.basename(mask_path)

        # Compute BrainIAC saliency map for this subject if required
        if brainiac_needed:
            from .brainiac_utils import compute_brainiac_saliency_maps
            ref_channel_path = volume_paths_list[subj_idx][0]
            sal_paths = compute_brainiac_saliency_maps(
                [ref_channel_path],
                brainiac_weights_path,
                brainiac_tmp_dir,
                overwrite=False,
            )
            volume_paths_list[subj_idx] = list(volume_paths_list[subj_idx]) + [sal_paths[0]]
        if ".nii.gz" in maskPattern:
            seg_name = base_name.replace(maskPattern, segmentSuffix)
        else:
            seg_name = base_name.replace(".nii.gz", segmentSuffix)
            if seg_name == base_name:
                seg_name += "_seg.nii.gz"
        out_path = out_dir / seg_name

        print(f"\n[INFO] Exam {subj_idx+1}/{num_subjects} | {seg_name}")
        if out_path.exists() and not overwrite:
            print(f"[INFO] Skipping exam {subj_idx+1}: {out_path} already exists.")
            continue

        mask_nib = nib.load(mask_path)
        affine = mask_nib.affine
        (oh, ow, od) = mask_nib.shape
        plane_outputs = []

        for m_idx, model in enumerate(loaded_models):
            plane = model_slicing_planes[m_idx]
            in_sl = model_num_input_slices[m_idx]
            out_sl = model_num_output_slices[m_idx]
            n_cls = model_num_classes[m_idx]
            min_HW = model_min_hw[m_idx]

            print(f"  [Model {m_idx+1}] plane={plane}, in={in_sl}, out={out_sl}, classes={n_cls}, minHW={min_HW}")

            (X_data, _, M_data, z_indices, transform_infos) = load_val_data(
                scan_indexes=[subj_idx],
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
            with torch.no_grad():
                for x_batch, m_batch in val_gen:
                    x_tensor = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
                    x_tensor = x_tensor.permute(0, 3, 1, 2).contiguous()
                    batch_logits = model(x_tensor)
                    if isinstance(batch_logits, tuple):
                        batch_logits = batch_logits[0]
                    batch_pred = F.softmax(batch_logits, dim=-1).cpu().numpy()
                    all_preds.append(batch_pred)

            if not all_preds:
                print(f"[WARNING] No slices processed for exam {subj_idx+1} ({base_name}).")
                # Instead of using the post-alignment shape, use the original mask dimensions.
                reassembled_4d = np.zeros((oh, ow, od, n_cls), dtype=np.float32)
                # Bypass the transformation steps for an empty prediction.
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
            if (Xf != oh) or (Yf != ow) or (Zf != od):
                raise ValueError(f"Mismatch after transforms! Expected {(oh, ow, od)}, got {(Xf, Yf, Zf)}. Check orientation logic.")
            final_arr = reoriented_4d
            mask_original = (mask_nib.get_fdata() > 0.5)
            for c in range(n_cls):
                final_arr[..., c] *= mask_original
            if debug_models:
                dbg_lbl = np.argmax(final_arr, axis=-1).astype(np.uint8)
                dbg_path = out_dir / f"Mod{m_idx+1}_debug.nii.gz"
                nib.save(nib.Nifti1Image(dbg_lbl, affine), str(dbg_path))
                print(f"  [debug] Wrote per-model debug label => {dbg_path}")
            plane_outputs.append(final_arr)

        print(f"[INFO] Merging via {merging_method}...")
        merged_label = merge_predictions(plane_outputs).astype(np.uint8)
        nib.save(nib.Nifti1Image(merged_label, affine), str(out_path))
        print(f"[INFO] Final seg => {out_path}")

    print("[INFO] All exams done.")


########################################################################
# Main entry point for command-line usage.
########################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Segment with training-like 2.5D pipeline."
    )
    parser.add_argument("--config", required=True,
                        help="Path to .cfg from create_segmentation_config")
    parser.add_argument("--slice_batch_size", type=int, default=1,
                        help="Batch size for 2.5D slices")
    parser.add_argument("--tiebreaker_model", type=int, default=0,
                        help="Tie-breaker model index (1-based). 0 => background on ties.")
    parser.add_argument("--overwrite_existing_outputs", action="store_true", default=False,
                        help="Overwrite existing outputs instead of skipping.")
    parser.add_argument("--use_cpu", action="store_true", default=False,
                        help="Force PyTorch to use CPU.")
    parser.add_argument("--debug_models", action="store_true", default=False,
                        help="Save each model's pre-merge label for debug.")

    args = parser.parse_args()
    if args.use_cpu:
        print("[INFO] Forcing PyTorch to use CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run_segmentation(
        segmentation_config_file=args.config,
        slice_batch_size=args.slice_batch_size,
        overwrite=args.overwrite_existing_outputs,
        tiebreaker_model=args.tiebreaker_model,
        debug_models=args.debug_models
    )

if __name__ == "__main__":
    main()

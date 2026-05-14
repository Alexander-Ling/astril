import os
import argparse
import configparser
import astril.config as config
from pathlib import Path


def parse_train_parameters(config_file_path):
    """
    Parses the train_parameters.cfg file and sets the parameters in the config module.
    """
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read(config_file_path)

    config.output_dir = cfg_parser.get("DEFAULT", "output_dir")
    config.n_cores = cfg_parser.getint("DEFAULT", "n_cores")
    config.dataloader_num_workers = cfg_parser.getint("DEFAULT", "dataloader_num_workers", fallback=2)
    config.slicing_plane = cfg_parser.get("DEFAULT", "slicing_plane")
    config.image_paths_files = cfg_parser.get("DEFAULT", "image_paths_files").split(',')
    config.gt_paths_file = cfg_parser.get("DEFAULT", "ground_truth_paths_files")
    config.mask_paths_file = cfg_parser.get("DEFAULT", "mask_paths_files")
    config.num_classes = cfg_parser.getint("DEFAULT", "num_classes")
    config.epochs = cfg_parser.getint("DEFAULT", "epochs")
    config.num_input_slices = cfg_parser.getint("DEFAULT", "num_input_slices")
    config.num_output_slices = cfg_parser.getint("DEFAULT", "num_output_slices")
    config.training_schedule_file = cfg_parser.get("DEFAULT", "training_schedule_file")
    raw_pretrained = cfg_parser.get("DEFAULT", "pretrained_model_path", fallback=None)
    if raw_pretrained is None or raw_pretrained.strip().lower() in {"", "none", "null", "na"}:
        config.pretrained_model_path = None
    else:
        config.pretrained_model_path = raw_pretrained
    config.print_every_n_subbatches = cfg_parser.getint("DEFAULT", "print_every_n_subbatches")
    config.minimum_height_width = cfg_parser.getint("DEFAULT", "minimum_height_width")
    config.num_channels = len(config.image_paths_files)

    if cfg_parser.has_option("DEFAULT", "base_num_filters"):
        config.base_num_filters = cfg_parser.getint("DEFAULT", "base_num_filters")
    else:
        config.base_num_filters = 32
    if cfg_parser.has_option("DEFAULT", "center_depth"):
        config.center_depth = cfg_parser.getint("DEFAULT", "center_depth")
    else:
        config.center_depth = 1
    if cfg_parser.has_option("DEFAULT", "encoder_level_factors"):
        factors_str = cfg_parser.get("DEFAULT", "encoder_level_factors")
        config.encoder_level_factors = [int(x.strip()) for x in factors_str.split(",") if x.strip()]
    else:
        config.encoder_level_factors = [1, 2, 4, 8]

    if cfg_parser.has_option("DEFAULT", "val_image_paths_files"):
        config.val_image_paths_files = cfg_parser.get("DEFAULT", "val_image_paths_files").split(',')
    if cfg_parser.has_option("DEFAULT", "val_ground_truth_paths_files"):
        config.val_gt_paths_file = cfg_parser.get("DEFAULT", "val_ground_truth_paths_files")
    if cfg_parser.has_option("DEFAULT", "val_mask_paths_files"):
        config.val_mask_paths_file = cfg_parser.get("DEFAULT", "val_mask_paths_files")

    # Architecture flags
    config.use_se_blocks = cfg_parser.getboolean("DEFAULT", "use_se_blocks", fallback=False)
    config.use_deep_supervision = cfg_parser.getboolean("DEFAULT", "use_deep_supervision", fallback=False)
    ds_weights_str = cfg_parser.get("DEFAULT", "deep_supervision_weights", fallback="0.5,0.25")
    try:
        config.deep_supervision_weights = [float(x.strip()) for x in ds_weights_str.split(",") if x.strip()]
    except ValueError:
        config.deep_supervision_weights = [0.5, 0.25]

    # Augmentation flags
    config.use_flip_augmentation = cfg_parser.getboolean("DEFAULT", "use_flip_augmentation", fallback=False)
    config.use_intensity_augmentation = cfg_parser.getboolean("DEFAULT", "use_intensity_augmentation", fallback=False)
    config.intensity_augmentation_strength = cfg_parser.getfloat("DEFAULT", "intensity_augmentation_strength", fallback=0.1)

    # Mixed precision
    config.use_mixed_precision = cfg_parser.getboolean("DEFAULT", "use_mixed_precision", fallback=False)

    # BrainIAC
    config.use_brainiac_embeddings = cfg_parser.getboolean("DEFAULT", "use_brainiac_embeddings", fallback=False)
    config.brainiac_embedding_type = cfg_parser.get("DEFAULT", "brainiac_embedding_type", fallback="encoder_fusion")

    def _none_str(raw):
        return None if raw in (None, "", "None", "none", "na") else raw

    train_bif = _none_str(cfg_parser.get("DEFAULT", "brainiac_feature_paths_files", fallback=None))
    val_bif = _none_str(cfg_parser.get("DEFAULT", "val_brainiac_feature_paths_files", fallback=None))
    config.brainiac_feature_paths_files = train_bif.split(",") if train_bif else None
    config.val_brainiac_feature_paths_files = val_bif.split(",") if val_bif else None
    config.brainiac_encoder_input_channels = cfg_parser.getint("DEFAULT", "brainiac_encoder_input_channels", fallback=0)
    config.brainiac_encode_channels = cfg_parser.get("DEFAULT", "brainiac_encode_channels", fallback="all")
    config.brainiac_encode_channel_indices = None


def _brainiac_channel_label(channel_cfg_file, index):
    """Stable label for per-channel BrainIAC feature caches."""
    stem = Path(channel_cfg_file).stem.lower()
    for prefix in ("trainchannels_", "valchannels_", "segchannels_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return f"ch{index}_{safe}" if safe else f"ch{index}"


def _resolve_brainiac_channel_indices(channel_cfg_files, selector):
    """Resolve 'all', comma-separated indices, or channel labels to indices."""
    if selector is None or str(selector).strip().lower() in {"", "all", "none"}:
        return list(range(len(channel_cfg_files)))

    labels = {}
    for idx, cfg_file in enumerate(channel_cfg_files):
        label = _brainiac_channel_label(cfg_file, idx).lower()
        short = label.split("_", 1)[1] if "_" in label else label
        labels[label] = idx
        labels[short] = idx
        labels[f"ch{idx}"] = idx

    resolved = []
    for raw_token in str(selector).split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
        else:
            if token not in labels:
                raise ValueError(
                    f"Unknown BrainIAC channel selector '{raw_token}'. "
                    f"Use 'all', numeric indices, or one of: {sorted(labels)}"
                )
            idx = labels[token]
        if idx < 0 or idx >= len(channel_cfg_files):
            raise ValueError(
                f"BrainIAC channel index {idx} is out of range for "
                f"{len(channel_cfg_files)} configured input channels."
            )
        if idx not in resolved:
            resolved.append(idx)

    if not resolved:
        raise ValueError("brainiac_encode_channels resolved to no input channels.")
    return resolved


def main():
    parser = argparse.ArgumentParser(description="MRI slice-based segmentation training script.")
    parser.add_argument("--config", required=True, help="Path to train_parameters.cfg file.")
    parser.add_argument("--epochs", type=int, help="Override the number of training epochs.")
    parser.add_argument("--n_cores", type=int, help="Override the number of CPU cores.")
    parser.add_argument("--output_dir", type=str, help="Override the output directory.")
    parser.add_argument("--slicing_plane", type=str, choices=["axial", "sagittal", "coronal"],
                        help="Override the slicing plane.")
    parser.add_argument("--training_schedule_file", type=str,
                        help="Override the training schedule file path.")
    parser.add_argument("--print_every_n_subbatches", type=int,
                        help="Override the subbatch logging frequency.")
    parser.add_argument("--minimum_height_width", type=int,
                        help="Override the minimum height and width required for training slices (in pixels).")

    # Architecture flags (override values in .cfg)
    parser.add_argument("--Use_SE_Blocks", action="store_true",
                        help="Enable Squeeze-and-Excitation channel attention in residual blocks.")
    parser.add_argument("--Use_Deep_Supervision", action="store_true",
                        help="Enable deep supervision auxiliary loss heads.")

    # Augmentation flags (override values in .cfg)
    parser.add_argument("--Use_Flip_Augmentation", action="store_true",
                        help="Enable random horizontal/vertical flip augmentation.")
    parser.add_argument("--Use_Intensity_Augmentation", action="store_true",
                        help="Enable random intensity (noise + contrast) augmentation.")

    # Mixed precision
    parser.add_argument("--Use_Mixed_Precision", action="store_true",
                        help="Enable float16 mixed precision for RTX tensor cores. "
                             "Halves activation VRAM and enables tensor-core utilisation.")

    # BrainIAC
    parser.add_argument("--BrainIAC_Weights_Path", type=str, default=None,
                        help=(
                            "Optional path to a locally downloaded BrainIAC .ckpt file. "
                            "If not provided, weights are downloaded automatically from Dropbox "
                            "on first use and cached for subsequent runs. "
                            "All other BrainIAC settings are set in train_parameters.cfg."
                        ))

    args = parser.parse_args()
    parse_train_parameters(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.n_cores is not None:
        config.n_cores = args.n_cores
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.slicing_plane is not None:
        config.slicing_plane = args.slicing_plane
    if args.training_schedule_file is not None:
        config.training_schedule_file = args.training_schedule_file
    if args.print_every_n_subbatches is not None:
        config.print_every_n_subbatches = args.print_every_n_subbatches
    if args.minimum_height_width is not None:
        config.minimum_height_width = args.minimum_height_width
    if args.Use_Mixed_Precision:
        config.use_mixed_precision = True
    if args.Use_SE_Blocks:
        config.use_se_blocks = True
    if args.Use_Deep_Supervision:
        config.use_deep_supervision = True
    if args.Use_Flip_Augmentation:
        config.use_flip_augmentation = True
    if args.Use_Intensity_Augmentation:
        config.use_intensity_augmentation = True

    # Write any CLI-overridden architecture/augmentation flags back to the .cfg
    # so the saved config accurately reflects what was actually used for training.
    from .create_config_files import update_train_config_flags
    cfg_updates = {}
    if args.Use_Mixed_Precision:
        cfg_updates["use_mixed_precision"] = "true"
    if args.Use_SE_Blocks:
        cfg_updates["use_se_blocks"] = "true"
    if args.Use_Deep_Supervision:
        cfg_updates["use_deep_supervision"] = "true"
    if args.Use_Flip_Augmentation:
        cfg_updates["use_flip_augmentation"] = "true"
    if args.Use_Intensity_Augmentation:
        cfg_updates["use_intensity_augmentation"] = "true"

    # BrainIAC pre-computation (runs when use_brainiac_embeddings=true in the config)
    if config.use_brainiac_embeddings:
        from .brainiac_utils import (
            ensure_brainiac_weights,
            compute_brainiac_encoder_features,
            BrainIACWeightsNotFoundError,
        )
        from .data_loading import read_paths_from_file

        if config.brainiac_embedding_type != "encoder_fusion":
            raise ValueError(
                "BrainIAC now supports only brainiac_embedding_type = encoder_fusion. "
                f"Found: {config.brainiac_embedding_type}"
            )

        weights_path = ensure_brainiac_weights(weights_path=args.BrainIAC_Weights_Path)
        brainiac_output_dir = Path(config.output_dir) / "brainiac_features"
        if len(config.image_paths_files) != len(config.val_image_paths_files):
            raise ValueError(
                "BrainIAC encoder fusion requires the same number of training and "
                "validation image channel cfg files."
            )
        selected_indices = _resolve_brainiac_channel_indices(
            config.image_paths_files,
            config.brainiac_encode_channels,
        )
        config.brainiac_encode_channel_indices = selected_indices
        print(f"[brainiac] Encoding input channel indices: {selected_indices}")

        train_feature_cfgs = []
        val_feature_cfgs = []
        for idx in selected_indices:
            train_cfg = config.image_paths_files[idx]
            val_cfg = config.val_image_paths_files[idx]
            label = _brainiac_channel_label(train_cfg, idx)
            train_paths = read_paths_from_file(train_cfg)
            val_paths = read_paths_from_file(val_cfg)
            print(f"[brainiac] Encoder-fusion embeddings for channel '{label}' "
                  f"({len(train_paths)} train, {len(val_paths)} val scans)...")

            train_features = compute_brainiac_encoder_features(
                nifti_paths=train_paths,
                weights_path=weights_path,
                output_dir=brainiac_output_dir / "train" / f"{label}_encoder",
                sequence_label=label,
            )
            val_features = compute_brainiac_encoder_features(
                nifti_paths=val_paths,
                weights_path=weights_path,
                output_dir=brainiac_output_dir / "val" / f"{label}_encoder",
                sequence_label=label,
            )

            train_feature_cfg = Path(config.output_dir) / f"trainBrainIAC_encoder_{label}.cfg"
            val_feature_cfg = Path(config.output_dir) / f"valBrainIAC_encoder_{label}.cfg"
            train_feature_cfg.write_text("\n".join(train_features))
            val_feature_cfg.write_text("\n".join(val_features))
            train_feature_cfgs.append(str(train_feature_cfg))
            val_feature_cfgs.append(str(val_feature_cfg))

        config.brainiac_feature_paths_files = train_feature_cfgs
        config.val_brainiac_feature_paths_files = val_feature_cfgs
        config.brainiac_encoder_input_channels = len(train_feature_cfgs) * 768
        config.brainiac_weights_path = str(weights_path)

        cfg_updates["use_brainiac_embeddings"] = "true"
        cfg_updates["brainiac_embedding_type"] = "encoder_fusion"
        cfg_updates["brainiac_encode_channels"] = ",".join(str(i) for i in selected_indices)
        cfg_updates["brainiac_feature_paths_files"] = ",".join(config.brainiac_feature_paths_files)
        cfg_updates["val_brainiac_feature_paths_files"] = ",".join(config.val_brainiac_feature_paths_files)
        cfg_updates["brainiac_encoder_input_channels"] = str(config.brainiac_encoder_input_channels)

        print(f"[brainiac] Done. Encoder-fusion input channels: {config.brainiac_encoder_input_channels}")

    # Persist flag changes to the .cfg so the saved model config reflects reality
    if cfg_updates:
        update_train_config_flags(args.config, **cfg_updates)

    from .train import train_model
    train_model()


if __name__ == "__main__":
    main()

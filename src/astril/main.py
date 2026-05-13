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

    # BrainIAC (informational; main() sets these if --Use_BrainIAC_Embeddings passed)
    config.use_brainiac_embeddings = cfg_parser.getboolean("DEFAULT", "use_brainiac_embeddings", fallback=False)


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
    parser.add_argument("--Use_BrainIAC_Embeddings", action="store_true",
                        help="Pre-compute BrainIAC saliency maps and use as an extra input channel.")
    parser.add_argument("--BrainIAC_Weights_Path", type=str, default=None,
                        help=(
                            "Optional path to a locally downloaded BrainIAC .ckpt file. "
                            "If not provided, weights are downloaded automatically from Dropbox "
                            "on first use and cached for subsequent runs."
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

    # BrainIAC pre-computation (must happen before train_model() so channels are set)
    if args.Use_BrainIAC_Embeddings:
        from .brainiac_utils import (
            ensure_brainiac_weights,
            compute_brainiac_saliency_maps,
            BrainIACWeightsNotFoundError,
        )
        from .data_loading import read_paths_from_file

        weights_path = ensure_brainiac_weights(
            weights_path=args.BrainIAC_Weights_Path,
        )

        brainiac_output_dir = Path(config.output_dir) / "brainiac_features"

        # Use first training channel as reference MRI for BrainIAC input
        first_train_paths = read_paths_from_file(config.image_paths_files[0])
        print(f"[brainiac] Pre-computing saliency maps for {len(first_train_paths)} training scans...")
        train_saliency_paths = compute_brainiac_saliency_maps(
            first_train_paths, weights_path, brainiac_output_dir / "train"
        )

        first_val_paths = read_paths_from_file(config.val_image_paths_files[0])
        print(f"[brainiac] Pre-computing saliency maps for {len(first_val_paths)} validation scans...")
        val_saliency_paths = compute_brainiac_saliency_maps(
            first_val_paths, weights_path, brainiac_output_dir / "val"
        )

        # Write saliency paths as new channel config files
        train_sal_cfg = Path(config.output_dir) / "trainChannels_brainiac.cfg"
        val_sal_cfg = Path(config.output_dir) / "valChannels_brainiac.cfg"
        train_sal_cfg.write_text("\n".join(train_saliency_paths))
        val_sal_cfg.write_text("\n".join(val_saliency_paths))

        # Append as the last channel
        config.image_paths_files.append(str(train_sal_cfg))
        config.val_image_paths_files.append(str(val_sal_cfg))
        config.num_channels += 1
        config.use_brainiac_embeddings = True
        config.brainiac_weights_path = str(weights_path)

        cfg_updates["use_brainiac_embeddings"] = "true"
        cfg_updates["brainiac_embedding_type"] = "saliency_map"

        print(f"[brainiac] Done. Total input channels: {config.num_channels}")

    # Persist flag changes to the .cfg so the saved model config reflects reality
    if cfg_updates:
        update_train_config_flags(args.config, **cfg_updates)

    from .train import train_model
    train_model()


if __name__ == "__main__":
    main()

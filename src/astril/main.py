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

    # BrainIAC
    config.use_brainiac_embeddings = cfg_parser.getboolean("DEFAULT", "use_brainiac_embeddings", fallback=False)

    def _none_str(raw):
        return None if raw in (None, "", "None", "none", "na") else raw

    config.brainiac_n_pcs        = cfg_parser.getint("DEFAULT", "brainiac_n_pcs", fallback=3)
    config.brainiac_pca_save_dir = _none_str(cfg_parser.get("DEFAULT", "brainiac_pca_save_dir", fallback=None))
    config.brainiac_pca_t1c_path = _none_str(cfg_parser.get("DEFAULT", "brainiac_pca_t1c_path", fallback=None))
    config.brainiac_pca_t2_path  = _none_str(cfg_parser.get("DEFAULT", "brainiac_pca_t2_path",  fallback=None))


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
                            "All other BrainIAC settings (use_brainiac_embeddings, brainiac_n_pcs, "
                            "etc.) are set in train_parameters.cfg."
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

    # BrainIAC PCA pre-computation (runs when use_brainiac_embeddings=true in the config)
    if config.use_brainiac_embeddings:
        import pickle
        from .brainiac_utils import (
            ensure_brainiac_weights,
            compute_brainiac_pca_features,
            BrainIACWeightsNotFoundError,
        )
        from .data_loading import read_paths_from_file

        weights_path = ensure_brainiac_weights(weights_path=args.BrainIAC_Weights_Path)
        n_pcs = config.brainiac_n_pcs
        pca_save_dir = Path(
            config.brainiac_pca_save_dir
            if config.brainiac_pca_save_dir
            else Path(config.output_dir) / "brainiac_features"
        )
        pca_save_dir.mkdir(parents=True, exist_ok=True)
        brainiac_output_dir = Path(config.output_dir) / "brainiac_features"

        # Channels 0 = T1c, 1 = T2 (as set up by create_config_files with T1c+T2)
        train_t1c_paths = read_paths_from_file(config.image_paths_files[0])
        train_t2_paths  = read_paths_from_file(config.image_paths_files[1])
        val_t1c_paths   = read_paths_from_file(config.val_image_paths_files[0])
        val_t2_paths    = read_paths_from_file(config.val_image_paths_files[1])

        # --- T1c PCA: fit on training, apply to val ---
        print(f"[brainiac] T1c embeddings → PCA ({n_pcs} components) for "
              f"{len(train_t1c_paths)} training scans...")
        train_t1c_pc_paths, pca_t1c = compute_brainiac_pca_features(
            nifti_paths=train_t1c_paths,
            weights_path=weights_path,
            output_dir=brainiac_output_dir / "train" / "t1c",
            sequence_label="t1c",
            n_components=n_pcs,
            pca=None,
        )
        pca_t1c_pkl = pca_save_dir / "pca_t1c.pkl"
        with open(pca_t1c_pkl, "wb") as _f:
            pickle.dump(pca_t1c, _f)

        print(f"[brainiac] Applying T1c PCA to {len(val_t1c_paths)} validation scans...")
        val_t1c_pc_paths, _ = compute_brainiac_pca_features(
            nifti_paths=val_t1c_paths,
            weights_path=weights_path,
            output_dir=brainiac_output_dir / "val" / "t1c",
            sequence_label="t1c",
            n_components=n_pcs,
            pca=pca_t1c,
        )

        # --- T2 PCA: fit on training (T2f and/or T2w already in train_t2_paths) ---
        print(f"[brainiac] T2 embeddings → PCA ({n_pcs} components) for "
              f"{len(train_t2_paths)} training scans (may include T2f and T2w)...")
        train_t2_pc_paths, pca_t2 = compute_brainiac_pca_features(
            nifti_paths=train_t2_paths,
            weights_path=weights_path,
            output_dir=brainiac_output_dir / "train" / "t2",
            sequence_label="t2",
            n_components=n_pcs,
            pca=None,
        )
        pca_t2_pkl = pca_save_dir / "pca_t2.pkl"
        with open(pca_t2_pkl, "wb") as _f:
            pickle.dump(pca_t2, _f)

        print(f"[brainiac] Applying T2 PCA to {len(val_t2_paths)} validation scans...")
        val_t2_pc_paths, _ = compute_brainiac_pca_features(
            nifti_paths=val_t2_paths,
            weights_path=weights_path,
            output_dir=brainiac_output_dir / "val" / "t2",
            sequence_label="t2",
            n_components=n_pcs,
            pca=pca_t2,
        )

        # --- Write one cfg file per PC per sequence, append to channel lists ---
        for k in range(n_pcs):
            t1c_train_cfg = Path(config.output_dir) / f"trainChannels_brainiac_t1c_pc{k}.cfg"
            t1c_val_cfg   = Path(config.output_dir) / f"valChannels_brainiac_t1c_pc{k}.cfg"
            t1c_train_cfg.write_text("\n".join(train_t1c_pc_paths[k]))
            t1c_val_cfg.write_text("\n".join(val_t1c_pc_paths[k]))
            config.image_paths_files.append(str(t1c_train_cfg))
            config.val_image_paths_files.append(str(t1c_val_cfg))

        for k in range(n_pcs):
            t2_train_cfg = Path(config.output_dir) / f"trainChannels_brainiac_t2_pc{k}.cfg"
            t2_val_cfg   = Path(config.output_dir) / f"valChannels_brainiac_t2_pc{k}.cfg"
            t2_train_cfg.write_text("\n".join(train_t2_pc_paths[k]))
            t2_val_cfg.write_text("\n".join(val_t2_pc_paths[k]))
            config.image_paths_files.append(str(t2_train_cfg))
            config.val_image_paths_files.append(str(t2_val_cfg))

        config.num_channels += 2 * n_pcs
        config.brainiac_weights_path = str(weights_path)

        cfg_updates["use_brainiac_embeddings"] = "true"
        cfg_updates["brainiac_embedding_type"] = "pca_embeddings"
        cfg_updates["brainiac_n_pcs"] = str(n_pcs)
        cfg_updates["brainiac_pca_t1c_path"] = str(pca_t1c_pkl)
        cfg_updates["brainiac_pca_t2_path"]  = str(pca_t2_pkl)

        print(f"[brainiac] Done. Total input channels: {config.num_channels}")

    # Persist flag changes to the .cfg so the saved model config reflects reality
    if cfg_updates:
        update_train_config_flags(args.config, **cfg_updates)

    from .train import train_model
    train_model()


if __name__ == "__main__":
    main()

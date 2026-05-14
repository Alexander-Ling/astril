import configparser
import os
import argparse
import glob
import itertools
from pathlib import Path
import multiprocessing


MODEL1_AXIAL_4MRI_CHANNELS = ["t1c", "t1n", "t2f", "t2w"]
MODEL1_AXIAL_4MRI_PATTERNS = [
    "_T1c_brain-norm.nii.gz|_T1c_normalized.nii.gz",
    "_T1n_brain-norm.nii.gz|_T1n_normalized.nii.gz",
    "_T2f_brain-norm.nii.gz|_T2f_normalized.nii.gz",
    "_T2w_brain-norm.nii.gz|_T2w_normalized.nii.gz",
]


def write_model1_axial_short_schedule(path):
    """
    Write a conservative 60-epoch axial Model 1 schedule.

    Validation begins at epoch 10, runs every 2 epochs through epoch 30, then
    every 5 epochs. Learning rate warms up at epochs 1 and 5, then decays at
    epochs 30, 40, and 50.
    """
    columns = [
        "epoch",
        "scan_batch_size",
        "slice_sub_batch_size",
        "accumulate_n_sub_batches",
        "conduct_validation",
        "validation_frequency",
        "learning_rate",
        "wce_loss_weight",
        "tversky_gamma",
        "class_weights",
        "epochs_per_new_training_data",
        "class_multiplication_factors",
        "require_classes",
        "tversky_alpha_values",
        "gradient_clip_norm",
        "label_smoothing",
        "deep_supervision_loss_weight",
    ]
    rows = [
        [1, "NA", 16, 1, "FALSE", "NA", 0.0001, 0.5, 1.0, "NA", 1, "NA", "NA", "NA", 1.0, 0.0, 0.5],
        [5, "NA", 16, 1, "FALSE", "NA", 0.0010, 0.5, 1.0, "NA", 1, "NA", "NA", "NA", 1.0, 0.0, 0.5],
        [10, "NA", 16, 1, "TRUE", 2, 0.0010, 0.5, 1.0, "NA", 1, "NA", "NA", "NA", 1.0, 0.0, 0.5],
        [30, "NA", 16, 1, "TRUE", 2, 0.0005, 0.5, 1.0, "NA", 1, "NA", "NA", "NA", 1.0, 0.0, 0.5],
        [31, "NA", 16, 1, "TRUE", 5, 0.0005, 0.5, 1.0, "NA", 1, "NA", "NA", "NA", 1.0, 0.0, 0.5],
        [40, "NA", 16, 1, "TRUE", 5, 0.0002, 0.5, 1.0, "NA", 1, "NA", "NA", "NA", 1.0, 0.0, 0.5],
        [50, "NA", 16, 1, "TRUE", 5, 0.0001, 0.5, 1.0, "NA", 1, "NA", "NA", "NA", 1.0, 0.0, 0.5],
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")
    return path


def update_train_config_flags(config_path, **flags):
    """
    Update boolean/string flags in an existing train_parameters.cfg file.
    E.g. update_train_config_flags(path, use_brainiac_embeddings='true',
                                         use_se_blocks='true')
    Uses configparser so existing values are preserved and only named keys change.
    """
    cfg = configparser.ConfigParser()
    cfg.read(str(config_path))
    for key, value in flags.items():
        cfg["DEFAULT"][key] = str(value)
    with open(str(config_path), "w") as f:
        cfg.write(f)


def create_config_files(
    workingDirectory=".",
    trainDataDirectory=None,
    valDataDirectory=None,
    trainChannels=None,
    trainPatterns=None,
    gtPattern=None,
    maskPattern=None,
    channel_alt_patterns=None,
    nCpuCores=None,
    numClasses=None,
    nEpochs=400,
    trainingSchedulePath=None,
    preTrainedModelPath=None,
    subbatchLogFrequency=10,
    numInputSlices=3,
    numOutputSlices=1,
    slicingPlane="axial",
    minimum_height_width=240,
    base_num_filters=32,
    center_depth=1,
    encoder_level_factors=[1, 2, 4, 8],
    use_brainiac_embeddings=False,
    brainiac_embedding_type="encoder_fusion",
    brainiac_encode_channels="all",
    use_flip_augmentation=False,
    use_intensity_augmentation=False,
    intensity_augmentation_strength=0.1,
    use_rotation_augmentation=False,
    rotation_degrees=10.0,
):
    """
    Creates config files in `workingDirectory/Configs/` for training and validation data.
    Note: Instead of a single data directory and random splitting,
    separate directories for training and validation are now required.

    channel_alt_patterns: optional list of fallback filename patterns, one per channel
        (use None at a position for channels with no fallback). At training time, subjects
        that match BOTH the primary and alt pattern for a channel are added twice — once
        per match — enabling dataset doubling (e.g. T2f and T2w as interchangeable T2
        channels). At validation time only the primary pattern is used, with the alt as a
        fallback when the primary is absent (no doubling).
    """
    # Validate inputs
    if trainDataDirectory is None:
        raise ValueError("trainDataDirectory must be provided.")
    if valDataDirectory is None:
        raise ValueError("valDataDirectory must be provided.")
    if trainChannels is None or trainPatterns is None:
        raise ValueError("Both trainChannels and trainPatterns must be provided.")
    if len(trainChannels) != len(trainPatterns):
        raise ValueError("The number of trainChannels must match the number of trainPatterns.")
    if gtPattern is None or maskPattern is None:
        raise ValueError("gtPattern and maskPattern must be provided.")
    if channel_alt_patterns is not None and len(channel_alt_patterns) != len(trainChannels):
        raise ValueError("channel_alt_patterns must be the same length as trainChannels.")

    # Determine number of CPU cores
    if nCpuCores is None:
        nCpuCores = max(multiprocessing.cpu_count() - 1, 1)
    dataloader_num_workers = min(8, max(int(nCpuCores), 0))

    # Create directories
    workingDirectory = Path(workingDirectory).resolve()
    configs_dir = workingDirectory / "Configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Generate paths for .cfg files for training data
    train_channel_cfg_files = [configs_dir / f"trainChannels_{channel}.cfg" for channel in trainChannels]
    train_gt_cfg_file = configs_dir / "trainGtLabels.cfg"
    train_mask_cfg_file = configs_dir / "trainRoiMasks.cfg"

    # Generate paths for .cfg files for validation data
    val_channel_cfg_files = [configs_dir / f"valChannels_{channel}.cfg" for channel in trainChannels]
    val_gt_cfg_file = configs_dir / "valGtLabels.cfg"
    val_mask_cfg_file = configs_dir / "valRoiMasks.cfg"

    # Gather timepoint directories for training data
    trainDataDirectory = Path(trainDataDirectory).resolve()
    train_timepoint_dirs = [p for p in trainDataDirectory.rglob("*") if p.is_dir()]

    # Gather timepoint directories for validation data
    valDataDirectory = Path(valDataDirectory).resolve()
    val_timepoint_dirs = [p for p in valDataDirectory.rglob("*") if p.is_dir()]

    def match_pattern(directory, pattern):
        """
        Matches a pattern in the given directory, returning the first unambiguous hit.
        Pattern may contain '|'-separated fallbacks tried left-to-right
        (e.g. "_T2f_normalized.nii.gz|_T2f_brain-norm.nii.gz").
        If multiple files match, prefer a NIfTI whose name starts with the
        current exam directory name. This preserves legacy DFCI folders that
        contain duplicate numeric-ID and DFCI-prefixed masks/labels.
        Returns None if no pattern yields a usable match.
        """
        if pattern is None:
            return None
        all_files = list(directory.iterdir())
        for pat in pattern.split("|"):
            pat = pat.strip()
            matches = [f for f in all_files if pat in f.name]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                nii_matches = [f for f in matches if f.name.endswith((".nii.gz", ".nii"))]
                if len(nii_matches) == 1:
                    return nii_matches[0]
                prefix_matches = [f for f in nii_matches if f.name.startswith(directory.name)]
                if len(prefix_matches) == 1:
                    return prefix_matches[0]
        return None

    # Process training directories
    # For channels with an alt pattern, subjects matching BOTH primary and alt are added
    # twice (once per match), enabling automatic dataset doubling.
    train_channel_file_paths = {channel: [] for channel in trainChannels}
    train_gt_file_paths = []
    train_mask_file_paths = []

    for timepoint_dir in train_timepoint_dirs:
        gt_file = match_pattern(timepoint_dir, gtPattern)
        if gt_file is None:
            print(f"Warning (train): Missing or ambiguous match for ground truth in {timepoint_dir}. Skipping.")
            continue
        mask_file = match_pattern(timepoint_dir, maskPattern)
        if mask_file is None:
            print(f"Warning (train): Missing or ambiguous match for mask file in {timepoint_dir}. Skipping.")
            continue

        # Collect options per channel: list of matched paths (1 or 2 for alt-pattern channels)
        options_per_channel = []
        skip = False
        for i, (channel, pattern) in enumerate(zip(trainChannels, trainPatterns)):
            alt_pat = (channel_alt_patterns[i] if channel_alt_patterns else None)
            primary = match_pattern(timepoint_dir, pattern)
            alt = match_pattern(timepoint_dir, alt_pat) if alt_pat else None

            if alt_pat:
                opts = []
                if primary is not None:
                    opts.append(str(primary))
                if alt is not None and str(alt) != str(primary):
                    opts.append(str(alt))
                if not opts:
                    print(f"Warning (train): No match for channel '{channel}' (primary or alt) in {timepoint_dir}. Skipping.")
                    skip = True
                    break
                options_per_channel.append(opts)
            else:
                if primary is None:
                    print(f"Warning (train): Missing or ambiguous match for pattern '{pattern}' in {timepoint_dir}. Skipping.")
                    skip = True
                    break
                options_per_channel.append([str(primary)])

        if skip:
            continue

        # itertools.product produces one row per combination; in practice this is 1 or 2 rows
        for combo in itertools.product(*options_per_channel):
            for channel, file_path in zip(trainChannels, combo):
                train_channel_file_paths[channel].append(file_path)
            train_gt_file_paths.append(str(gt_file))
            train_mask_file_paths.append(str(mask_file))

    # Process validation directories
    # For channels with an alt pattern, prefer primary; fall back to alt if primary absent.
    # No doubling at validation time.
    val_channel_file_paths = {channel: [] for channel in trainChannels}
    val_gt_file_paths = []
    val_mask_file_paths = []

    for timepoint_dir in val_timepoint_dirs:
        gt_file = match_pattern(timepoint_dir, gtPattern)
        if gt_file is None:
            print(f"Warning (val): Missing or ambiguous match for ground truth in {timepoint_dir}. Skipping.")
            continue
        mask_file = match_pattern(timepoint_dir, maskPattern)
        if mask_file is None:
            print(f"Warning (val): Missing or ambiguous match for mask file in {timepoint_dir}. Skipping.")
            continue

        matched_channel_files = []
        skip = False
        for i, (channel, pattern) in enumerate(zip(trainChannels, trainPatterns)):
            alt_pat = (channel_alt_patterns[i] if channel_alt_patterns else None)
            matched = match_pattern(timepoint_dir, pattern)
            if matched is None and alt_pat:
                matched = match_pattern(timepoint_dir, alt_pat)
            if matched is None:
                print(f"Warning (val): No match for channel '{channel}' (primary or alt) in {timepoint_dir}. Skipping.")
                skip = True
                break
            matched_channel_files.append(str(matched))

        if skip:
            continue

        for channel, file_path in zip(trainChannels, matched_channel_files):
            val_channel_file_paths[channel].append(file_path)
        val_gt_file_paths.append(str(gt_file))
        val_mask_file_paths.append(str(mask_file))

    # Write training channel .cfg files
    for channel, cfg_file in zip(trainChannels, train_channel_cfg_files):
        with cfg_file.open("w") as f:
            f.write("\n".join(train_channel_file_paths[channel]))

    # Write training gt and mask .cfg files
    with train_gt_cfg_file.open("w") as f:
        f.write("\n".join(train_gt_file_paths))
    with train_mask_cfg_file.open("w") as f:
        f.write("\n".join(train_mask_file_paths))

    # Write validation channel .cfg files
    for channel, cfg_file in zip(trainChannels, val_channel_cfg_files):
        with cfg_file.open("w") as f:
            f.write("\n".join(val_channel_file_paths[channel]))

    # Write validation gt and mask .cfg files
    with val_gt_cfg_file.open("w") as f:
        f.write("\n".join(val_gt_file_paths))
    with val_mask_cfg_file.open("w") as f:
        f.write("\n".join(val_mask_file_paths))

    # Write train_parameters.cfg (note: trainFraction has been removed)
    params_cfg_file = configs_dir / "train_parameters.cfg"
    with params_cfg_file.open("w") as f:
        f.write("[DEFAULT]\n")
        f.write(f"output_dir = {workingDirectory}\n")
        f.write(f"n_cores = {nCpuCores}\n")
        f.write(f"dataloader_num_workers = {dataloader_num_workers}\n")
        f.write("dataloader_prefetch_factor = 4\n")
        f.write("dataloader_persistent_workers = true\n")
        f.write(f"slicing_plane = {slicingPlane}\n")
        f.write(f"image_paths_files = {','.join(map(str, train_channel_cfg_files))}\n")
        f.write(f"ground_truth_paths_files = {train_gt_cfg_file}\n")
        f.write(f"mask_paths_files = {train_mask_cfg_file}\n")
        f.write(f"val_image_paths_files = {','.join(map(str, val_channel_cfg_files))}\n")
        f.write(f"val_ground_truth_paths_files = {val_gt_cfg_file}\n")
        f.write(f"val_mask_paths_files = {val_mask_cfg_file}\n")
        f.write(f"num_classes = {numClasses}\n")
        f.write(f"epochs = {nEpochs}\n")
        f.write(f"num_input_slices = {numInputSlices}\n")
        f.write(f"num_output_slices = {numOutputSlices}\n")
        f.write(f"minimum_height_width = {minimum_height_width}\n")
        f.write(f"training_schedule_file = {trainingSchedulePath}\n")
        f.write(f"pretrained_model_path = {preTrainedModelPath}\n")
        f.write(f"print_every_n_subbatches = {subbatchLogFrequency}\n")
        f.write(f"base_num_filters = {base_num_filters}\n")
        f.write(f"center_depth = {center_depth}\n")
        encoder_factors_str = ",".join(str(x) for x in encoder_level_factors)
        f.write(f"encoder_level_factors = {encoder_factors_str}\n")
        # Architecture flags (default off; main.py may update these after config generation)
        f.write("use_se_blocks = false\n")
        f.write("use_deep_supervision = false\n")
        f.write("deep_supervision_weights = 0.5,0.25\n")
        # BrainIAC flags. When enabled, BrainIAC is used as a frozen encoder-fusion branch.
        f.write(f"use_brainiac_embeddings = {str(bool(use_brainiac_embeddings)).lower()}\n")
        f.write(f"brainiac_embedding_type = {brainiac_embedding_type}\n")
        f.write(f"brainiac_encode_channels = {brainiac_encode_channels}\n")
        f.write("brainiac_feature_paths_files = none\n")
        f.write("val_brainiac_feature_paths_files = none\n")
        f.write("brainiac_encoder_input_channels = 0\n")
        # Mixed precision (default off; safe to enable on CUDA hardware)
        f.write("use_mixed_precision = false\n")
        # Augmentation flags
        f.write(f"use_flip_augmentation = {str(bool(use_flip_augmentation)).lower()}\n")
        f.write(f"use_intensity_augmentation = {str(bool(use_intensity_augmentation)).lower()}\n")
        f.write(f"intensity_augmentation_strength = {float(intensity_augmentation_strength)}\n")
        f.write(f"use_rotation_augmentation = {str(bool(use_rotation_augmentation)).lower()}\n")
        f.write(f"rotation_degrees = {float(rotation_degrees)}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate configuration files for MRI segmentation training.")
    parser.add_argument("--workingDirectory", default=".", help="Directory to store generated config files.")
    parser.add_argument("--trainDataDirectory", required=True, help="Directory with training data.")
    parser.add_argument("--valDataDirectory", required=True, help="Directory with validation data.")
    parser.add_argument("--model1_axial_4mri_no_brainiac_recipe", action="store_true",
                        help="Use the axial Model 1 T1c/T1n/T2f/T2w no-BrainIAC overfit-reduction recipe.")
    parser.add_argument("--trainChannels", nargs="+", default=None, help="Names of training channels.")
    parser.add_argument("--trainPatterns", nargs="+", default=None, help="Patterns for training channels.")
    parser.add_argument("--channel_alt_patterns", nargs="+", default=None,
                        help="Optional fallback patterns, one per channel. Use 'none' for channels with no "
                             "fallback (e.g. --channel_alt_patterns none _T2w_brain-norm.nii.gz). "
                             "Training: subjects matching both primary and alt are added twice (dataset "
                             "doubling). Validation: primary preferred, alt used as fallback.")
    parser.add_argument("--gtPattern", required=True, help="Pattern for ground truth files.")
    parser.add_argument("--maskPattern", required=True, help="Pattern for mask files.")
    parser.add_argument("--nCpuCores", type=int, default=None, help="Number of CPU cores to use for data loading.")
    parser.add_argument("--numClasses", type=int, required=True, help="Number of segmentation classes, including background.")
    parser.add_argument("--nEpochs", type=int, default=400, help="Number of training epochs.")
    parser.add_argument("--trainingSchedulePath", default=None, help="Path to training schedule file.")
    parser.add_argument("--preTrainedModelPath", default=None, help="Optional path to a migrated PyTorch .pt checkpoint.")
    parser.add_argument("--subbatchLogFrequency", type=int, default=10, help="Log training outputs every this many sub-batches.")
    parser.add_argument("--numInputSlices", type=int, default=3, help="Number of adjacent slices input to cnn model each cycle.")
    parser.add_argument("--numOutputSlices", type=int, default=1, help="Number of segmented slices output from cnn model each cycle.")
    parser.add_argument("--slicingPlane", default="axial", choices=["axial", "sagittal", "coronal"], help="Slicing plane.")
    parser.add_argument("--minimum_height_width", type=int, default=240, help="Minimum height or width of slice for training (in pixels).")
    parser.add_argument("--base_num_filters", type=int, default=32, help="Base number of filters in first encoder layer.")
    parser.add_argument("--center_depth", type=int, default=1, help="Number of center bottleneck blocks to include in UNET model.")
    parser.add_argument("--encoder_level_factors", type=str, default="1,2,4,8",
                        help="Comma-separated expansions for each encoder level (e.g. 1,2,4,8).")
    parser.add_argument("--use_brainiac_embeddings", action="store_true",
                        help="Enable BrainIAC feature integration in the generated training config.")
    parser.add_argument("--brainiac_embedding_type", default="encoder_fusion",
                        choices=["encoder_fusion"],
                        help="BrainIAC integration mode to write when --use_brainiac_embeddings is set.")
    parser.add_argument("--brainiac_encode_channels", default="all",
                        help="Comma-separated channel indices or names to encode with BrainIAC, or 'all'.")
    parser.add_argument("--Use_Flip_Augmentation", action="store_true",
                        help="Enable random horizontal/vertical flip augmentation in the generated config.")
    parser.add_argument("--Use_Intensity_Augmentation", action="store_true",
                        help="Enable random intensity augmentation in the generated config.")
    parser.add_argument("--intensity_augmentation_strength", type=float, default=0.1,
                        help="Noise/contrast strength for intensity augmentation.")
    parser.add_argument("--Use_Rotation_Augmentation", action="store_true",
                        help="Enable random in-plane rotation augmentation in the generated config.")
    parser.add_argument("--rotation_degrees", type=float, default=10.0,
                        help="Maximum absolute random in-plane rotation angle in degrees.")

    args = parser.parse_args()

    if args.model1_axial_4mri_no_brainiac_recipe:
        args.trainChannels = args.trainChannels or MODEL1_AXIAL_4MRI_CHANNELS
        args.trainPatterns = args.trainPatterns or MODEL1_AXIAL_4MRI_PATTERNS
        args.slicingPlane = "axial"
        args.nEpochs = 60 if args.nEpochs == parser.get_default("nEpochs") else args.nEpochs
        args.Use_Flip_Augmentation = True
        args.Use_Intensity_Augmentation = True
        args.Use_Rotation_Augmentation = True
        args.use_brainiac_embeddings = False
        if args.trainingSchedulePath is None:
            args.trainingSchedulePath = str(
                Path(args.workingDirectory).resolve()
                / "Configs"
                / "model1_axial_short_schedule.tsv"
            )
            write_model1_axial_short_schedule(args.trainingSchedulePath)

    if args.trainChannels is None or args.trainPatterns is None:
        raise ValueError(
            "--trainChannels and --trainPatterns are required unless "
            "--model1_axial_4mri_no_brainiac_recipe is used."
        )
    if args.trainingSchedulePath is None:
        raise ValueError("--trainingSchedulePath is required unless a recipe writes a default schedule.")

    encoder_level_factors = [int(x) for x in args.encoder_level_factors.split(",") if x.strip()]

    if args.preTrainedModelPath and not str(args.preTrainedModelPath).lower().endswith(".pt"):
        raise ValueError("--preTrainedModelPath must point to a PyTorch .pt checkpoint.")

    # Convert "none" sentinels to None in channel_alt_patterns
    channel_alt_patterns = None
    if args.channel_alt_patterns is not None:
        channel_alt_patterns = [
            None if p.strip().lower() in ("none", "na", "") else p.strip()
            for p in args.channel_alt_patterns
        ]

    create_config_files(
        workingDirectory=args.workingDirectory,
        trainDataDirectory=args.trainDataDirectory,
        valDataDirectory=args.valDataDirectory,
        trainChannels=args.trainChannels,
        trainPatterns=args.trainPatterns,
        gtPattern=args.gtPattern,
        maskPattern=args.maskPattern,
        channel_alt_patterns=channel_alt_patterns,
        nCpuCores=args.nCpuCores,
        numClasses=args.numClasses,
        nEpochs=args.nEpochs,
        trainingSchedulePath=args.trainingSchedulePath,
        preTrainedModelPath=args.preTrainedModelPath,
        subbatchLogFrequency=args.subbatchLogFrequency,
        numInputSlices=args.numInputSlices,
        numOutputSlices=args.numOutputSlices,
        slicingPlane=args.slicingPlane,
        minimum_height_width=args.minimum_height_width,
        base_num_filters=args.base_num_filters,
        center_depth=args.center_depth,
        encoder_level_factors=encoder_level_factors,
        use_brainiac_embeddings=args.use_brainiac_embeddings,
        brainiac_embedding_type=args.brainiac_embedding_type,
        brainiac_encode_channels=args.brainiac_encode_channels,
        use_flip_augmentation=args.Use_Flip_Augmentation,
        use_intensity_augmentation=args.Use_Intensity_Augmentation,
        intensity_augmentation_strength=args.intensity_augmentation_strength,
        use_rotation_augmentation=args.Use_Rotation_Augmentation,
        rotation_degrees=args.rotation_degrees,
    )


if __name__ == "__main__":
    main()

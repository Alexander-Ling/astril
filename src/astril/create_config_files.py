import configparser
import os
import argparse
import glob
import itertools
from pathlib import Path
import multiprocessing


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
    encoder_level_factors=[1, 2, 4, 8]
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
        Matches a pattern in the given directory.
        If a pattern is missing or ambiguous, returns None.
        """
        all_files = list(directory.iterdir())
        matches = [file for file in all_files if pattern in file.name]
        if len(matches) != 1:
            return None
        return matches[0]

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
        # BrainIAC flags (default off; set use_brainiac_embeddings=true in this file to enable)
        f.write("use_brainiac_embeddings = false\n")
        f.write("brainiac_embedding_type = pca_embeddings\n")
        f.write("brainiac_n_pcs = 3\n")
        f.write("brainiac_pca_save_dir = none\n")
        f.write("brainiac_pca_t1c_path = none\n")
        f.write("brainiac_pca_t2_path = none\n")
        # Mixed precision (default off; safe to enable on CUDA hardware)
        f.write("use_mixed_precision = false\n")
        # Augmentation flags (default off; set via --Use_Flip_Augmentation / --Use_Intensity_Augmentation)
        f.write("use_flip_augmentation = false\n")
        f.write("use_intensity_augmentation = false\n")
        f.write("intensity_augmentation_strength = 0.1\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate configuration files for MRI segmentation training.")
    parser.add_argument("--workingDirectory", default=".", help="Directory to store generated config files.")
    parser.add_argument("--trainDataDirectory", required=True, help="Directory with training data.")
    parser.add_argument("--valDataDirectory", required=True, help="Directory with validation data.")
    parser.add_argument("--trainChannels", nargs="+", required=True, help="Names of training channels.")
    parser.add_argument("--trainPatterns", nargs="+", required=True, help="Patterns for training channels.")
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
    parser.add_argument("--trainingSchedulePath", required=True, help="Path to training schedule file.")
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

    args = parser.parse_args()

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
        encoder_level_factors=encoder_level_factors
    )

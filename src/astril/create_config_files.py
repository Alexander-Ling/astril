import os
import argparse
import glob
from pathlib import Path
import multiprocessing

def create_config_files(
    workingDirectory=".",
    trainDataDirectory=None,
    valDataDirectory=None,
    trainChannels=None,
    trainPatterns=None,
    gtPattern=None,
    maskPattern=None,
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
    train_channel_file_paths = {channel: [] for channel in trainChannels}
    train_gt_file_paths = []
    train_mask_file_paths = []

    for timepoint_dir in train_timepoint_dirs:
        matched_channel_files = []
        skip = False
        for channel, pattern in zip(trainChannels, trainPatterns):
            matched_file = match_pattern(timepoint_dir, pattern)
            if matched_file is None:
                print(f"Warning (train): Missing or ambiguous match for pattern '{pattern}' in {timepoint_dir}. Skipping.")
                skip = True
                break
            matched_channel_files.append(str(matched_file))
        if not skip:
            gt_file = match_pattern(timepoint_dir, gtPattern)
            if gt_file is None:
                print(f"Warning (train): Missing or ambiguous match for ground truth in {timepoint_dir}. Skipping.")
                continue
            mask_file = match_pattern(timepoint_dir, maskPattern)
            if mask_file is None:
                print(f"Warning (train): Missing or ambiguous match for mask file in {timepoint_dir}. Skipping.")
                continue
            for channel, file_path in zip(trainChannels, matched_channel_files):
                train_channel_file_paths[channel].append(file_path)
            train_gt_file_paths.append(str(gt_file))
            train_mask_file_paths.append(str(mask_file))

    # Process validation directories
    val_channel_file_paths = {channel: [] for channel in trainChannels}
    val_gt_file_paths = []
    val_mask_file_paths = []

    for timepoint_dir in val_timepoint_dirs:
        matched_channel_files = []
        skip = False
        for channel, pattern in zip(trainChannels, trainPatterns):
            matched_file = match_pattern(timepoint_dir, pattern)
            if matched_file is None:
                print(f"Warning (val): Missing or ambiguous match for pattern '{pattern}' in {timepoint_dir}. Skipping.")
                skip = True
                break
            matched_channel_files.append(str(matched_file))
        if not skip:
            gt_file = match_pattern(timepoint_dir, gtPattern)
            if gt_file is None:
                print(f"Warning (val): Missing or ambiguous match for ground truth in {timepoint_dir}. Skipping.")
                continue
            mask_file = match_pattern(timepoint_dir, maskPattern)
            if mask_file is None:
                print(f"Warning (val): Missing or ambiguous match for mask file in {timepoint_dir}. Skipping.")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate configuration files for MRI segmentation training.")
    parser.add_argument("--workingDirectory", default=".", help="Directory to store generated config files.")
    parser.add_argument("--trainDataDirectory", required=True, help="Directory with training data.")
    parser.add_argument("--valDataDirectory", required=True, help="Directory with validation data.")
    parser.add_argument("--trainChannels", nargs="+", required=True, help="Names of training channels.")
    parser.add_argument("--trainPatterns", nargs="+", required=True, help="Patterns for training channels.")
    parser.add_argument("--gtPattern", required=True, help="Pattern for ground truth files.")
    parser.add_argument("--maskPattern", required=True, help="Pattern for mask files.")
    parser.add_argument("--nCpuCores", type=int, default=None, help="Number of CPU cores to use for data loading.")
    parser.add_argument("--numClasses", type=int, required=True, help="Number of segmentation classes, including background.")
    parser.add_argument("--nEpochs", type=int, default=400, help="Number of training epochs.")
    parser.add_argument("--trainingSchedulePath", required=True, help="Path to training schedule file.")
    parser.add_argument("--preTrainedModelPath", default=None, help="Optional path to pre-trained model.")
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

    create_config_files(
        workingDirectory=args.workingDirectory,
        trainDataDirectory=args.trainDataDirectory,
        valDataDirectory=args.valDataDirectory,
        trainChannels=args.trainChannels,
        trainPatterns=args.trainPatterns,
        gtPattern=args.gtPattern,
        maskPattern=args.maskPattern,
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

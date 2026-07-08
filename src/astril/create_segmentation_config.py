import os
import argparse
import glob
from pathlib import Path
import configparser

MISSING_CHANNEL_SENTINEL = "__MISSING__"

def parse_train_config_for_model_parameters(train_config_path):
    """
    Reads a train_parameters.cfg (created by create_config_files)
    and extracts relevant parameters, e.g.:
      - slicing_plane
      - num_input_slices
      - num_output_slices
      - minimum_height_width
      - num_classes
      - (optionally) base_num_filters, encoder_level_factors, etc.

    Returns
    -------
    dict
        e.g. {
          "slicing_plane": "axial",
          "num_input_slices": 3,
          "num_output_slices": 1,
          "minimum_height_width": 240,
          "num_classes": 2
        }
    """
    parser = configparser.ConfigParser()
    parser.read(train_config_path)

    # The create_config_files script typically writes these in [DEFAULT].
    section = "DEFAULT"
    
    # Read slicing_plane
    slicing_plane = parser.get(section, "slicing_plane", fallback="axial")

    # Read integers
    num_input_slices     = parser.getint(section, "num_input_slices", fallback=3)
    num_output_slices    = parser.getint(section, "num_output_slices", fallback=1)
    minimum_height_width = parser.getint(section, "minimum_height_width", fallback=240)
    num_classes          = parser.getint(section, "num_classes", fallback=2)
    
    return {
        "slicing_plane": slicing_plane,
        "num_input_slices": num_input_slices,
        "num_output_slices": num_output_slices,
        "minimum_height_width": minimum_height_width,
        "num_classes": num_classes
    }


def create_segmentation_config(
    workingDirectory=".",
    inputChannels=None,
    channelPatterns=None,
    maskPattern=None,
    channel_alt_patterns=None,
    model_paths=None,
    modelTrainConfigFiles=None,
    merging_method="majority_vote",
    inputVolumeDirectory=None,
    outputVolumeDirectory=None,
    segmentSuffix="_seg.nii.gz",
    output_config_filename="segmentation_parameters.cfg",
    silent=False,
    optional_channels=None,
    allow_missing_optional_channels=None,
):
    """
    Creates config files for segmentation in `workingDirectory/Configs/`.

    1. For each channel specified in inputChannels with corresponding pattern in channelPatterns,
       searches subdirectories of `inputVolumeDirectory` to find a single matching file per subject/timepoint.
       Writes matched paths to `segChannels_<channelName>.cfg`.
    2. Similarly, searches for mask files (using maskPattern) and writes them to `segMask.cfg`.
    3. Accepts a set of model_paths and corresponding modelTrainConfigFiles (one per model).
       For each model's training config, extracts relevant 2.5D parameters (e.g. slicing_plane, 
       num_input_slices, etc.) and stores them in segmentation_parameters.cfg, so that 
       run_segmentation can use them.
    4. Creates a central `segmentation_parameters.cfg` describing:
       - The .cfg files for channels & mask
       - The PyTorch .pt model checkpoints & their *training* parameters
       - The merging method (e.g. "majority_vote")
       - The output volume directory or "in_place" if segmentation results go alongside each mask

    Parameters
    ----------
    workingDirectory : str
        Directory where the "Configs/" folder will be created (if not found).
    inputChannels : list of str
        Names/identifiers of each MRI channel, e.g. ["T1", "T2", "FLAIR"].
    channelPatterns : list of str
        Filename patterns for each channel, e.g. ["-T1.nii.gz", "-T2.nii.gz", "-FLAIR.nii.gz"].
        Must be the same length as `inputChannels`.
    maskPattern : str
        Pattern to identify mask files, e.g. "-brainmask.nii.gz".
    model_paths : list of str
        Paths to migrated PyTorch .pt checkpoints.
    modelTrainConfigFiles : list of str
        Paths to the train_parameters.cfg used to train each corresponding model in model_paths.
    merging_method : str
        How predictions from multiple models are merged, e.g. "majority_vote".
    inputVolumeDirectory : str
        Directory containing subdirectories (or files) for each subject/timepoint.
    outputVolumeDirectory : str
        Either a path to a directory where final segmentations will be stored,
        or the string "in_place" meaning output files go beside each mask file.
    segmentSuffix : str
        Suffix to use when saving segmentation volumes.
    output_config_filename : str
        Filename for the main segmentation config file.
    silent : bool
        If True, suppresses print statements.

    Returns
    -------
    str
        Path to the newly-created `segmentation_parameters.cfg` file.
    """

    # ------------------------------
    # 1) Validate inputs
    # ------------------------------
    if inputChannels is None or channelPatterns is None:
        raise ValueError("Both inputChannels and channelPatterns must be provided.")
    if len(inputChannels) != len(channelPatterns):
        raise ValueError("The number of inputChannels must match the number of channelPatterns.")
    if maskPattern is None:
        raise ValueError("maskPattern must be provided.")

    if channel_alt_patterns is not None and len(channel_alt_patterns) != len(inputChannels):
        raise ValueError("channel_alt_patterns must be the same length as inputChannels.")
    if not model_paths or not modelTrainConfigFiles:
        raise ValueError("You must provide both model_paths and modelTrainConfigFiles (one per model).")
    if len(model_paths) != len(modelTrainConfigFiles):
        raise ValueError("Mismatch: the number of model_paths must match the number of modelTrainConfigFiles.")

    if optional_channels is None:
        inferred_optional = []
        for cfg_path in modelTrainConfigFiles:
            cp = configparser.ConfigParser()
            cp.read(str(cfg_path))
            raw = cp["DEFAULT"].get("optional_channels", "")
            for token in raw.split(","):
                token = token.strip()
                if token and token not in inferred_optional:
                    inferred_optional.append(token)
        optional_channels = inferred_optional
    else:
        optional_channels = list(optional_channels or [])
    unknown_optional = sorted(set(optional_channels) - set(inputChannels))
    if unknown_optional:
        raise ValueError(f"optional_channels contains unknown channel(s): {unknown_optional}")
    optional_channel_set = set(optional_channels)
    if allow_missing_optional_channels is None:
        allow_missing_optional_channels = bool(optional_channel_set)

    invalid_model_paths = [m for m in model_paths if not str(m).lower().endswith(".pt")]
    if invalid_model_paths:
        raise ValueError(
            "Migrated astril segmentation expects PyTorch .pt checkpoints. "
            f"Invalid model_paths: {invalid_model_paths}"
        )

    if inputVolumeDirectory is None:
        raise ValueError("`inputVolumeDirectory` must be provided.")
    if outputVolumeDirectory is None:
        raise ValueError("`outputVolumeDirectory` must be provided (or 'in_place').")

    # ------------------------------
    # 2) Prepare directories
    # ------------------------------
    working_dir = Path(workingDirectory).resolve()
    configs_dir = working_dir / "Configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(inputVolumeDirectory).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory.")

    # Gather all subdirectories (including the root) to search for patterns
    all_dirs = [p for p in input_dir.rglob("*") if p.is_dir()]
    if input_dir.is_dir():
        all_dirs.append(input_dir)
    all_dirs = list(set(all_dirs))
    all_dirs.sort()

    # For each channel, store matched file paths
    channel_file_paths = {chan: [] for chan in inputChannels}
    mask_file_paths = []

    def match_pattern(directory, pattern):
        """
        Search for exactly one file in `directory` whose name contains `pattern`.
        Pattern may contain '|'-separated fallbacks tried left-to-right
        (e.g. "_T2f_normalized.nii.gz|_T2f_brain-norm.nii.gz").
        If multiple files match, prefer a NIfTI whose name starts with the
        current exam directory name. This keeps legacy DFCI folders with
        duplicate numeric-ID and DFCI-prefixed masks from being skipped.
        Returns None if no pattern yields a usable match.
        """
        if pattern is None:
            return None
        for pat in pattern.split("|"):
            pat = pat.strip()
            matches = [f for f in directory.iterdir() if pat in f.name]
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

    # ------------------------------
    # 3) Loop over subdirectories, match each channel & mask
    #    For channels with an alt pattern: prefer primary, fall back to alt.
    #    No doubling — pick one match per subject.
    # ------------------------------
    for d in all_dirs:
        matched_channel_files = []
        skip_this_dir = False

        for i, (chan, pat) in enumerate(zip(inputChannels, channelPatterns)):
            fpath = match_pattern(d, pat)
            if fpath is None and channel_alt_patterns and channel_alt_patterns[i]:
                fpath = match_pattern(d, channel_alt_patterns[i])
            if fpath is None:
                if allow_missing_optional_channels and chan in optional_channel_set:
                    matched_channel_files.append(MISSING_CHANNEL_SENTINEL)
                    continue
                else:
                    skip_this_dir = True
                    break
            matched_channel_files.append(str(fpath))

        if skip_this_dir:
            continue

        # Match mask
        fmask = match_pattern(d, maskPattern)
        if fmask is None:
            continue

        # If we get here, we have all channels + mask
        for chan, f in zip(inputChannels, matched_channel_files):
            channel_file_paths[chan].append(f)
        mask_file_paths.append(str(fmask))

    # ------------------------------
    # 4) Write out channel .cfg files & mask .cfg
    # ------------------------------
    channel_cfg_files = []
    for chan in inputChannels:
        cfg_file = configs_dir / f"segChannels_{chan}.cfg"
        with cfg_file.open("w") as f:
            f.write("\n".join(channel_file_paths[chan]))
        channel_cfg_files.append(str(cfg_file))

    mask_cfg_file = configs_dir / "segMask.cfg"
    with mask_cfg_file.open("w") as f:
        f.write("\n".join(mask_file_paths))

    # ------------------------------
    # 5) Parse each model's training config to gather parameters
    # ------------------------------
    model_slicing_planes = []
    model_num_input_slices = []
    model_num_output_slices = []
    model_min_height_widths = []
    model_num_classes = []

    for train_cfg_path in modelTrainConfigFiles:
        train_params = parse_train_config_for_model_parameters(train_cfg_path)
        model_slicing_planes.append(train_params["slicing_plane"])
        model_num_input_slices.append(train_params["num_input_slices"])
        model_num_output_slices.append(train_params["num_output_slices"])
        model_min_height_widths.append(train_params["minimum_height_width"])
        model_num_classes.append(train_params["num_classes"])

    if len(set(model_num_classes)) != 1:
        raise ValueError("All models must have the same num_classes to merge predictions meaningfully.")

    # ------------------------------
    # 6) Create the segmentation_parameters.cfg
    # ------------------------------
    config_parser = configparser.ConfigParser()
    config_parser["DEFAULT"] = {}

    # (a) References to newly created .cfg files
    config_parser["DEFAULT"]["channel_paths_files"] = ",".join(channel_cfg_files)
    config_parser["DEFAULT"]["mask_paths_file"] = str(mask_cfg_file)
    config_parser["DEFAULT"]["channel_names"] = ",".join(inputChannels)
    config_parser["DEFAULT"]["optional_channels"] = ",".join(optional_channels)
    config_parser["DEFAULT"]["allow_missing_optional_channels"] = str(bool(allow_missing_optional_channels)).lower()
    config_parser["DEFAULT"]["missing_channel_fill"] = "zero"

    # (b) Model paths
    config_parser["DEFAULT"]["model_paths"] = ",".join(str(Path(m).resolve()) for m in model_paths)

    # (c) Merging method + output dir
    config_parser["DEFAULT"]["merging_method"] = merging_method
    config_parser["DEFAULT"]["output_directory"] = outputVolumeDirectory

    # (d) Store extracted parameters from the model training configs in parallel arrays
    config_parser["DEFAULT"]["model_train_slicing_planes"]   = ",".join(model_slicing_planes)
    config_parser["DEFAULT"]["model_train_num_input_slices"]   = ",".join(str(x) for x in model_num_input_slices)
    config_parser["DEFAULT"]["model_train_num_output_slices"]  = ",".join(str(x) for x in model_num_output_slices)
    config_parser["DEFAULT"]["model_train_minimum_hw"]         = ",".join(str(x) for x in model_min_height_widths)
    config_parser["DEFAULT"]["model_train_num_classes"]        = ",".join(str(x) for x in model_num_classes)

    # (e) Optionally store the path to each train_config:
    config_parser["DEFAULT"]["model_train_config_files"] = ",".join(str(Path(cfg).resolve()) for cfg in modelTrainConfigFiles)

    # (f) Save maskPattern and segmentSuffix for use in determining segmented filenames:
    config_parser["DEFAULT"]["maskPattern"] = str(maskPattern)
    config_parser["DEFAULT"]["segmentSuffix"] = str(segmentSuffix)

    # (g) Write to .cfg file
    seg_cfg_path = configs_dir / output_config_filename
    with seg_cfg_path.open("w") as cfg:
        config_parser.write(cfg)

    if not silent:
        print(f"Segmentation config file created: {seg_cfg_path}")
    return str(seg_cfg_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate config files for MRI segmentation using multiple channels & models, with training configs for each model.")
    parser.add_argument("--workingDirectory", default=".", help="Directory to store generated config files.")
    parser.add_argument("--inputChannels", nargs="+", required=True,
                        help="Names of input channels (e.g. T1n, T2w, T2f).")
    parser.add_argument("--channelPatterns", nargs="+", required=True,
                        help="Patterns for each channel's input data (e.g. T1c_brain_norm.nii.gz, T2w_brain_norm.nii.gz, etc.).")
    parser.add_argument("--maskPattern", required=True,
                        help="Pattern to identify mask files (e.g. brainmask.nii.gz).")
    parser.add_argument("--channel_alt_patterns", nargs="+", default=None,
                        help="Optional fallback patterns, one per channel. Use 'none' for channels "
                             "with no fallback (e.g. --channel_alt_patterns none _T2w_brain-norm.nii.gz). "
                             "Primary pattern is preferred; alt is used when primary is absent. "
                             "No dataset doubling at inference time.")
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="Paths to PyTorch .pt model checkpoints. One per slicing plane.")
    parser.add_argument("--modelTrainConfigFiles", nargs="+", required=True,
                        help="Paths to the train_parameters.cfg used to train each model. Must match length of model_paths.")
    parser.add_argument("--merging_method", default="majority_vote",
                        help="Method for merging model predictions.")
    parser.add_argument("--inputVolumeDirectory", required=True,
                        help="Directory containing subdirectories or volumes for segmentation.")
    parser.add_argument("--outputVolumeDirectory", required=True,
                        help='Directory for saving segmentation volumes OR "in_place".')
    parser.add_argument("--segmentSuffix", default="_seg.nii.gz", help="Suffix to use when saving segmentation volumes--will replace maskPattern in mask file names if maskPattern includes .nii.gz. Otherwise, will be appended to end of mask filenames.")
    parser.add_argument("--output_config_filename", default="segmentation_parameters.cfg",
                        help="Name of the segmentation config file to produce (in Configs/).")
    parser.add_argument("--optional_channels", nargs="*", default=None,
                        help="Channel names that may be missing at inference time.")
    parser.add_argument("--allow_missing_optional_channels", action="store_true",
                        help="Allow optional channels to be absent and zero-filled at inference time.")
    parser.add_argument("--silent", action="store_true", help="Suppress output messages.")
    
    args = parser.parse_args(argv)

    channel_alt_patterns = None
    if args.channel_alt_patterns is not None:
        channel_alt_patterns = [
            None if p.strip().lower() in ("none", "na", "") else p.strip()
            for p in args.channel_alt_patterns
        ]

    create_segmentation_config(
        workingDirectory=args.workingDirectory,
        inputChannels=args.inputChannels,
        channelPatterns=args.channelPatterns,
        maskPattern=args.maskPattern,
        channel_alt_patterns=channel_alt_patterns,
        model_paths=args.model_paths,
        modelTrainConfigFiles=args.modelTrainConfigFiles,
        merging_method=args.merging_method,
        inputVolumeDirectory=args.inputVolumeDirectory,
        outputVolumeDirectory=args.outputVolumeDirectory,
        segmentSuffix=args.segmentSuffix,
        output_config_filename=args.output_config_filename,
        silent=args.silent,
        optional_channels=args.optional_channels,
        allow_missing_optional_channels=args.allow_missing_optional_channels or bool(args.optional_channels),
    )


if __name__ == "__main__":
    main()

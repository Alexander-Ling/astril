import os
import argparse
import numpy as np
from pathlib import Path
import nibabel as nib
import ast

def match_pattern(directory, pattern):
    """
    Matches a pattern in the given directory. Returns matched files.

    Args:
        directory (Path): Directory to search for files.
        pattern (str): Substring to match in file names.

    Returns:
        list[Path]: List of matching files.
    """
    all_files = list(directory.iterdir())
    matches = [file for file in all_files if pattern in file.name]
    return matches

def parse_dict_string(dict_str):
    """
    Safely parse a dictionary-like string (e.g., "{(1,2,3):5, (1,3):2}")
    into a Python dictionary.

    If dict_str is empty, 'nan', invalid, or 'None', returns empty dict.
    """
    if not dict_str or dict_str in ["{}", "None", "null", "nan"]:
        return {}
    try:
        parsed = ast.literal_eval(dict_str)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed object is not a dictionary.")
        for k in parsed.keys():
            if not isinstance(k, tuple):
                raise ValueError(f"Invalid key {k}. Keys must be tuples.")
        return parsed
    except Exception as e:
        raise ValueError(f"Error parsing dictionary string '{dict_str}': {e}")

def remap_gt_classes(trainDataDirectory, gtPattern, classRemapDict, outputPattern, overwrite_existing_outputs=False):
    """
    Remap ground truth classes in segmentation files based on a mapping dictionary.

    Args:
        trainDataDirectory (str): Directory containing training data.
        gtPattern (str): Pattern for ground truth files.
        classRemapDict (str): Dictionary-like string defining the remapping.
        outputPattern (str): Pattern to replace gtPattern in output file names.
        overwrite_existing_outputs (bool): If True, overwrite outputs that already exist.
                                             Defaults to False (skip if exists).
    """
    # Parse classRemapDict
    remap_dict = parse_dict_string(classRemapDict)
    if not remap_dict:
        raise ValueError("Invalid classRemapDict format. Ensure it is a valid dictionary-like string.")

    # Convert trainDataDirectory to Path object
    train_data_dir = Path(trainDataDirectory)

    # Gather files matching the pattern
    gt_files = [p for p in train_data_dir.rglob("*") if p.is_file() and gtPattern in p.name]

    if not gt_files:
        raise FileNotFoundError(f"No files matching pattern '{gtPattern}' found in directory '{trainDataDirectory}'.")

    # Process each ground truth file
    for gt_file in gt_files:
        # Determine the output file name before any heavy processing
        output_file = gt_file.with_name(gt_file.name.replace(gtPattern, outputPattern))
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Skip processing if output file exists and we're not set to overwrite
        if output_file.exists() and not overwrite_existing_outputs:
            print(f"Skipping {gt_file} because output file {output_file} already exists and overwrite is disabled.")
            continue

        print(f"Processing: {gt_file}")

        # Load ground truth file
        img = nib.load(str(gt_file))
        gt_data = np.asarray(img.dataobj, dtype=np.int32)
        affine = img.affine  # Preserve the affine matrix

        # Remap classes
        remapped_data = np.zeros_like(gt_data, dtype=np.int32)
        for src_classes, target_class in remap_dict.items():
            for src_class in src_classes:
                remapped_data[gt_data == src_class] = target_class

        # Save remapped data with the original affine matrix
        nib.save(nib.Nifti1Image(remapped_data, affine), str(output_file))
        print(f"Saved remapped file to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap ground truth classes in segmentation files.")
    parser.add_argument("--trainDataDirectory", required=True, help="Directory containing training data.")
    parser.add_argument("--gtPattern", required=True, help="Pattern for ground truth files.")
    parser.add_argument("--classRemapDict", required=True, help="Dictionary-like string defining the class remapping.")
    parser.add_argument("--outputPattern", required=True, help="Pattern to replace gtPattern in output file names.")
    parser.add_argument(
        "--overwrite_existing_outputs",
        action="store_true",
        help="If set, overwrite outputs that already exist. By default, existing outputs are skipped."
    )

    args = parser.parse_args()

    remap_gt_classes(
        trainDataDirectory=args.trainDataDirectory,
        gtPattern=args.gtPattern,
        classRemapDict=args.classRemapDict,
        outputPattern=args.outputPattern,
        overwrite_existing_outputs=args.overwrite_existing_outputs
    )

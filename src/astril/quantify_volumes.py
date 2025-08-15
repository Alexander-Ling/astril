"""
Module: quantify_volumes
This module calculates segmentation volumes for MRI segmentation files.
Files are recursively searched based on a provided filename pattern,
and the results are written to an output file.

Command Line Usage:
    python -m astril.quantify_volumes <root_directory> <filename_pattern> <output_file> --total_levels 3 [--number_of_threads 4] [--append_to_output]

Example:
    python -m astril.quantify_volumes /data/mri "*_seg_UNet.nii.gz" 2025-02-03_UNet_Volumes.txt --total_levels 3 --number_of_threads 4 --append_to_output
"""

import os
import nibabel as nib
import numpy as np
import fnmatch
from pathlib import Path
import argparse
import concurrent.futures

def file_matches_pattern(file, pattern):
    """
    Returns True if the file's name matches the provided pattern.
    
    If the pattern contains wildcard characters ('*', '?', or '['),
    then it is interpreted using shell-style matching via fnmatch.
    Otherwise, the pattern is treated as a substring that must appear in the filename.
    """
    if any(ch in pattern for ch in "*?[]"):
        return fnmatch.fnmatch(file.name, pattern)
    else:
        return pattern in file.name

def calculate_volume_for_file(input_file_path, total_levels):
    """
    Loads a segmentation file, computes the volume for each segmentation level (1..total_levels),
    and returns a formatted string with the file path, file name, and volumes.
    """
    try:
        img = nib.load(input_file_path)
        data = img.get_fdata()
    except Exception as e:
        print(f"Error processing file {input_file_path}: {e}")
        return None

    # Calculate voxel volume (in mm^3)
    voxel_dims = img.header.get_zooms()
    voxel_volume = np.prod(voxel_dims)

    # Compute volume for each segmentation level (levels 1..total_levels)
    volumes_mm3 = [
        round((data == level).sum() * voxel_volume)
        for level in range(1, total_levels + 1)
    ]

    file_name = os.path.basename(input_file_path)
    output_line = f"{input_file_path}\t{file_name}\t" + "\t".join(str(v) for v in volumes_mm3)
    return output_line

def quantify_segmentation_volumes(root_directory, filename_pattern, output_file, total_levels,
                                  number_of_threads=1, append_to_output=False):
    """
    Recursively searches for segmentation files in `root_directory` that match `filename_pattern`,
    calculates the segmentation volumes, and writes the results to `output_file`.
    """
    root_dir = Path(root_directory)
    # Get all files and filter using our custom matching function.
    all_files = list(root_dir.rglob('*'))
    files = [f for f in all_files if f.is_file() and file_matches_pattern(f, filename_pattern)]
    
    if not files:
        print(f"No files found in {root_directory} matching pattern '{filename_pattern}'.")
        return

    results = []
    total_jobs = len(files)
    completed_jobs = 0
    print(f"Found {total_jobs} file(s). Starting processing with {number_of_threads} thread(s)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_threads) as executor:
        future_to_file = {
            executor.submit(calculate_volume_for_file, str(file), total_levels): file
            for file in files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            if result is not None:
                results.append(result)
            completed_jobs += 1
            print(f"\rProgress: Completed {completed_jobs} out of {total_jobs} jobs", end="")
    print("\nAll files processed.")

    output_path = Path(output_file)
    write_header = True
    if append_to_output and output_path.exists() and output_path.stat().st_size > 0:
        write_header = False

    mode = 'a' if append_to_output else 'w'
    with open(output_path, mode) as f:
        if write_header:
            header = ["File_path", "File_name"] + [f"Segment{i}Volume_mm3" for i in range(1, total_levels + 1)]
            f.write("\t".join(header) + "\n")
        for line in results:
            f.write(line + "\n")
    print(f"Results written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Calculate segmentation volumes for MRI segmentation files in a directory recursively.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_directory", help="Root directory to search for segmentation files.")
    parser.add_argument("filename_pattern", help="Filename pattern or partial filename to match segmentation files. "
                                                   "Wildcards (*, ?, [) are supported if provided.")
    parser.add_argument("output_file", help="Output text file to save segmentation volume results.")
    parser.add_argument("--total_levels", type=int, required=True,
                        help="Total number of expected segmentation levels (excluding background).")
    parser.add_argument("--number_of_threads", type=int, default=1,
                        help="Number of threads to use for parallel processing.")
    parser.add_argument("--append_to_output", action="store_true",
                        help="Append results to the output file if it exists; otherwise, overwrite.")

    args = parser.parse_args()
    quantify_segmentation_volumes(
        root_directory=args.root_directory,
        filename_pattern=args.filename_pattern,
        output_file=args.output_file,
        total_levels=args.total_levels,
        number_of_threads=args.number_of_threads,
        append_to_output=args.append_to_output
    )

if __name__ == "__main__":
    main()

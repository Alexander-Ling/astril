import os
from pathlib import Path
import nibabel as nib
import numpy as np
import argparse


def match_pattern(directory, pattern):
    """
    Search for exactly one file in `directory` whose name contains `pattern`.
    If none or more than one match is found, return None.
    """
    matches = [f for f in directory.iterdir() if pattern in f.name]
    if len(matches) != 1:
        return None
    return matches[0]


def merge_seg_volumes(
    inputVolumePatterns,
    inputVolumeDirectory,
    outputVolumeDirectory,
    outputVolumePatterns,
    overwrite_existing_outputs=False
):
    """
    Merge segmentation volumes by overlaying non-zero voxels sequentially.

    Parameters
    ----------
    inputVolumePatterns : list of str
        List of patterns to match input files (order matters for merging).
    inputVolumeDirectory : str
        Directory to recursively search for volumes (subdirectories at all depths).
    outputVolumeDirectory : str
        Directory to save merged volumes, or "in_place" to save beside input files.
    outputVolumePatterns : str
        Pattern for output filenames, replacing the first input pattern in the matched file name.
    overwrite_existing_outputs : bool
        If True, existing output files will be overwritten; otherwise skipped.

    Outputs
    -------
    One merged segmentation volume (.nii.gz) per valid directory in `inputVolumeDirectory` (recursively).
    """
    input_dir = Path(inputVolumeDirectory).resolve()

    # If we're saving in a separate directory, prepare that path
    if outputVolumeDirectory != "in_place":
        output_dir = Path(outputVolumeDirectory).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    # Gather all subdirectories (recursively) to process
    # rglob('*') returns every file/folder under input_dir; we'll filter to only directories.
    valid_subdirs = [p for p in input_dir.rglob('*') if p.is_dir()]

    for subdir in valid_subdirs:
        # Match files for each pattern
        input_files = []
        for pattern in inputVolumePatterns:
            matched_file = match_pattern(subdir, pattern)
            if matched_file is None:
                print(f"Skipping {subdir} (pattern '{pattern}' did not match exactly one file).")
                break
            input_files.append(matched_file)
        else:
            # All patterns matched exactly one file; proceed to validate shapes/affines
            loaded_imgs = []
            shapes = []
            affines = []
            for file_path in input_files:
                img = nib.load(str(file_path))
                loaded_imgs.append(img)
                shapes.append(img.shape)
                affines.append(img.affine)

            # Check shape/affine consistency
            ref_shape = shapes[0]
            ref_affine = affines[0]
            if any(s != ref_shape for s in shapes[1:]) or any(not np.allclose(a, ref_affine) for a in affines[1:]):
                print(f"Skipping {subdir}: Shape or affine mismatch among volumes.")
                continue

            # Merge volumes by overlaying non-zero voxels
            merged_data = np.zeros(ref_shape, dtype=np.int16)
            for img in loaded_imgs:
                img_data = img.get_fdata()
                # Replace only zeros in merged_data with non-zero from the new image
                mask = (merged_data == 0) & (img_data != 0)
                merged_data[mask] = img_data[mask]

            # Construct output file name/path
            original_filename = input_files[0].name
            new_filename = original_filename.replace(inputVolumePatterns[0], outputVolumePatterns)

            if output_dir:
                # If a separate output directory is specified, replicate subdir structure if desired.
                # Right now, you could either place everything directly in output_dir,
                # or replicate subdir structure (uncomment next lines if needed):
                #
                # relative_subdir = subdir.relative_to(input_dir)
                # full_output_dir = output_dir / relative_subdir
                # full_output_dir.mkdir(parents=True, exist_ok=True)
                #
                # output_file = full_output_dir / new_filename

                output_file = output_dir / new_filename
            else:
                # "in_place" -> save in the same subdirectory
                output_file = subdir / new_filename

            # Check overwrite conditions
            if output_file.exists() and not overwrite_existing_outputs:
                print(f"Skipping {output_file} (file exists, overwrite disabled).")
                continue

            # Save merged volume
            out_img = nib.Nifti1Image(merged_data, ref_affine)
            nib.save(out_img, str(output_file))
            print(f"Merged volume saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Merge segmentation volumes recursively.")
    parser.add_argument(
        "--inputVolumePatterns", nargs="+", required=True,
        help="Patterns to match input volumes. Order matters for merge priority."
    )
    parser.add_argument(
        "--inputVolumeDirectory", required=True,
        help="Directory to recursively search for volumes (subdirectories at all depths)."
    )
    parser.add_argument(
        "--outputVolumeDirectory", required=True,
        help="Directory to save merged volumes or 'in_place' for saving in each found subdirectory."
    )
    parser.add_argument(
        "--outputVolumePatterns", required=True,
        help="Pattern for output files, replacing the first input pattern in the matched filename."
    )
    parser.add_argument(
        "--overwrite_existing_outputs", action="store_true",
        help="Allow overwriting of existing output files."
    )
    args = parser.parse_args()

    merge_seg_volumes(
        args.inputVolumePatterns,
        args.inputVolumeDirectory,
        args.outputVolumeDirectory,
        args.outputVolumePatterns,
        args.overwrite_existing_outputs
    )

if __name__ == "__main__":
    main()

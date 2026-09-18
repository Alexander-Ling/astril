"""
Module: quantify_volumes
This module calculates segmentation volumes for MRI segmentation files.
Files are recursively searched based on a provided filename pattern,
and the results are written to an output file.

Which levels to quantify is always driven by an explicit contract, never inferred by scanning
voxel data for whichever values happen to appear -- that would be fragile (data-dependent,
non-deterministic across batches) and could silently disagree with what the segmentation model
was actually designed to output. The level set/labels come from, in priority order:
  1. an explicit `level_labels` mapping (or `--labels_json` contract file) passed by the caller,
  2. each matched file's own `segmentation_provenance.json` sidecar (written automatically by a
     segmentation step, e.g. `segment_GBM.py`, naming the model family and its labels), or
  3. an explicit `total_levels` count (levels 1..N, generically named).
At least one of these must resolve to a non-empty level set, or the run fails loudly rather than
silently producing an empty/wrong table.

Command Line Usage:
    python -m astril.quantify_volumes <root_directory> <filename_pattern> <output_file> [--total_levels 3] [--labels_json labels.json] [--number_of_threads 4] [--append_to_output]

Example:
    python -m astril.quantify_volumes /data/mri "*_seg_UNet.nii.gz" 2025-02-03_UNet_Volumes.txt --total_levels 3 --number_of_threads 4 --append_to_output
"""

import json
import os
import fnmatch
from pathlib import Path
import argparse
import concurrent.futures

PROVENANCE_FILENAME = "segmentation_provenance.json"


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


def _sanitize_label(label: str) -> str:
    """Make a model-provided label safe to use as a TSV column name."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(label).strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "Unlabeled"


def _labels_from_contract(data: dict) -> dict[int, str]:
    """Pull a {level: name} mapping out of a parsed labels.json contract, dropping background/0.
    Accepts either the wrapped astril shape ({"labels": {...}, ...}) or a bare {level: name} map.
    """
    labels = data.get("labels", data) if isinstance(data, dict) else {}
    return {int(level): str(name) for level, name in labels.items() if int(level) != 0}


def load_labels_json(path) -> dict[int, str]:
    """Load an explicit labels.json contract file. Unlike the best-effort provenance sidecar
    lookup below, a path given explicitly by the caller is expected to exist and be valid --
    failures here are raised, not swallowed."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _labels_from_contract(data)


def load_provenance_labels(segmentation_file_path) -> dict[int, str] | None:
    """
    Look for a `segmentation_provenance.json` sidecar in the same directory as a segmentation
    file (written by `segment_GBM.py`/future model pipelines) and return its level->label
    mapping, or None if no sidecar is present / it can't be read.
    """
    sidecar = Path(segmentation_file_path).parent / PROVENANCE_FILENAME
    if not sidecar.is_file():
        return None
    try:
        with sidecar.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _labels_from_contract(data)
    except Exception as e:
        print(f"[quantify_volumes] WARNING: could not read {sidecar}: {e}")
        return None


def calculate_volume_for_file(input_file_path, levels):
    """Load one segmentation file and compute its volume (mm3) at each of `levels`, in order."""
    import nibabel as nib
    import numpy as np
    try:
        img = nib.load(input_file_path)
        data = img.get_fdata()
    except Exception as e:
        print(f"Error processing file {input_file_path}: {e}")
        return None
    voxel_dims = img.header.get_zooms()
    voxel_volume = np.prod(voxel_dims)
    volumes_mm3 = [round((data == level).sum() * voxel_volume) for level in levels]
    file_name = os.path.basename(input_file_path)
    return f"{input_file_path}\t{file_name}\t" + "\t".join(str(v) for v in volumes_mm3)


def _resolve_levels_and_labels(files, total_levels=None, level_labels=None, labels_json=None):
    """
    Determine the level set (and any known names for it) to quantify, per the priority order
    documented at the top of this module. Never reads voxel data -- only small JSON sidecars.
    """
    resolved_labels: dict[int, str] = {}

    explicit_contract = None
    if level_labels:
        explicit_contract = {int(k): str(v) for k, v in level_labels.items() if int(k) != 0}
    elif labels_json:
        explicit_contract = load_labels_json(labels_json)

    if explicit_contract:
        resolved_labels.update(explicit_contract)
    else:
        # Fall back to whatever each matched file's own segmentation_provenance.json declares --
        # still an explicit contract, just supplied automatically by the segmentation step
        # instead of by hand. Different files are allowed to carry different contracts (e.g. a
        # mixed batch across model families); a genuine disagreement on one level's meaning is
        # surfaced as a warning, not silently picked for you.
        label_conflicts = set()
        for f in files:
            provenance_labels = load_provenance_labels(str(f))
            if not provenance_labels:
                continue
            for level, name in provenance_labels.items():
                existing = resolved_labels.get(level)
                if existing is not None and existing != name:
                    label_conflicts.add(level)
                else:
                    resolved_labels[level] = name
        if label_conflicts:
            print(
                "[quantify_volumes] WARNING: different segmentation_provenance.json sidecars "
                f"disagree on the label for level(s) {sorted(label_conflicts)}; using the first "
                "one encountered for each. Pass an explicit level_labels/labels_json to force one."
            )

    if total_levels:
        for level in range(1, int(total_levels) + 1):
            resolved_labels.setdefault(level, None)

    return resolved_labels


def quantify_segmentation_volumes(root_directory, filename_pattern, output_file, total_levels=None,
                                  number_of_threads=1, append_to_output=False, level_labels=None,
                                  labels_json=None):
    """
    Recursively searches for segmentation files in `root_directory` that match `filename_pattern`,
    calculates the segmentation volumes, and writes the results to `output_file`.

    The set of levels to quantify must be pinned down by a contract, not inferred from the data:
    pass `total_levels` (levels 1..N, generic "SegmentN" column names), `level_labels`/
    `labels_json` (an explicit {level: name} contract), or ensure the matched files carry
    `segmentation_provenance.json` sidecars (written automatically by the segmentation step) --
    at least one is required. Every declared level is reported for every file, even at 0 mm3, so
    the output table has a stable column set regardless of which levels happen to be absent from
    any one exam.
    """
    root_dir = Path(root_directory)
    all_files = list(root_dir.rglob('*'))
    files = [f for f in all_files if f.is_file() and file_matches_pattern(f, filename_pattern)]

    if not files:
        print(f"No files found in {root_directory} matching pattern '{filename_pattern}'.")
        return

    resolved_labels = _resolve_levels_and_labels(
        files, total_levels=total_levels, level_labels=level_labels, labels_json=labels_json
    )
    if not resolved_labels:
        raise ValueError(
            "quantify_segmentation_volumes: could not determine which segmentation levels to "
            "quantify. Pass --total_levels, --labels_json (or level_labels=), or ensure the "
            f"matched files have a {PROVENANCE_FILENAME} sidecar."
        )

    levels = sorted(resolved_labels.keys())
    column_names = [
        f"{_sanitize_label(resolved_labels[level])}_Volume_mm3" if resolved_labels.get(level)
        else f"Segment{level}Volume_mm3"
        for level in levels
    ]

    total_jobs = len(files)
    print(f"Found {total_jobs} file(s). Starting processing with {number_of_threads} thread(s)...")
    results = []
    completed_jobs = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_threads) as executor:
        future_to_file = {
            executor.submit(calculate_volume_for_file, str(file), levels): file
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
            header = ["File_path", "File_name"] + column_names
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
    parser.add_argument("--total_levels", type=int, default=None,
                        help="Number of non-background levels to quantify (1..N, generic column names). "
                             "Not needed if --labels_json is given or matched files carry a "
                             "segmentation_provenance.json sidecar.")
    parser.add_argument("--labels_json", type=str, default=None,
                        help="Path to a labels.json contract ({\"labels\": {\"1\": \"...\", ...}}) naming "
                             "what each level means; output columns are named accordingly. One of "
                             "--total_levels/--labels_json is required unless matched files already "
                             "carry a segmentation_provenance.json sidecar.")
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
        append_to_output=args.append_to_output,
        labels_json=args.labels_json,
    )

if __name__ == "__main__":
    main()

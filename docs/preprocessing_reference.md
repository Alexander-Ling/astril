# DICOM preprocessing reference

This document is the reference for Astril's DICOM-library preprocessing
functions. The command-line interface exposes the same functions through
`python -m astril.preprocess`.

The metadata-indexed workflow is:

```text
DICOM library (any structure)
  -> organize_dicoms                 -> organized DICOM library + DICOM index
  -> create_patient_metadata         -> patient metadata table (user fills in patientID, day0Date)
  -> plan_dicom_to_nifti_conversion  -> conversion plan (user reviews/edits)
  -> convert_dicom_plan              -> NIfTI library
```

`organize_dicoms` is the entry point for every DICOM library. It reads DICOM
metadata recursively and creates a normalized Patient/Exam/Series tree, so the
input folders do not need to follow any naming or nesting convention.
Downstream functions use the generated index instead of inferring identity
from folder names.

The organized library and the DICOM index are inputs to both
`create_patient_metadata` and `plan_dicom_to_nifti_conversion`. The planner
requires a patient metadata table with `Directory`, `patientID`, and
`day0Date` columns; `create_patient_metadata` is the supported way to produce
one. `convert_dicom_plan` reads only the plan, which records the source series
folder for each conversion.

`demix_dicoms` is a legacy cleanup step and is not part of this workflow; see
[`demix_dicoms`](#demix_dicoms) below.

## `organize_dicoms`

### Purpose

Scan a DICOM library recursively, copy readable DICOM files into a normalized
directory hierarchy, and write a machine-readable organization index.

Patient, exam, and series identity are metadata-based:

- Patient: `PatientID` (with a deterministic fallback when absent)
- Exam: `StudyInstanceUID` exclusively
- Series: `SeriesInstanceUID` (with a deterministic fallback when absent)

The input may be nested differently for different patients and may contain
suffixless DICOM files. Because files are routed by `SeriesInstanceUID`, files
from different series that share a source folder are separated, and files from
one series that are split across source folders are combined. The source
library is never modified. Existing destination files are not overwritten:
identical files are skipped and conflicting files are recorded as conflicts.

### Python API

```python
organize_dicoms(
    root_dir: str,
    out_dir: str,
    index_out: str | None = None,
    quarantine_dir: str | None = None,
    log_out: str | None = None,
    unresolved_out: str | None = None,
    dry_run: bool = False,
    show_progress: bool = True,
    n_workers: int | None = None,
)
```

`root_dir` is the source library. `out_dir` is the normalized output root.
`index_out` writes the organization index and must be a CSV or TSV path; XLSX
is not supported for the organization index. The CLI requires `--indexOut`,
and the downstream functions cannot run without it.

`quarantine_dir` receives files that are unreadable, missing required identity
metadata, or have UID conflicts; it defaults to `out_dir/_quarantine`.
Quarantined files keep their path relative to `root_dir`.

`unresolved_out` sets the unresolved-file report path. By default the report is
written beside the index as `<index stem>_unresolved.csv`, or beneath `out_dir`
as `unresolved_dicom_report.csv` when no index path is given. `log_out` writes
an additional CSV copy of the complete index.

`dry_run=True` reads metadata and writes the index without copying files.
Dry-run rows have status `planned`, which the downstream functions do not
accept, so rerun without `dry_run` before continuing.

`n_workers` controls metadata reads and copying, and `show_progress` controls
progress display.

The function returns the generated organization records as a pandas DataFrame.
A Ctrl+C interruption is handled cooperatively: completed records and the
unresolved report are preserved when possible.

### CLI

```text
python -m astril.preprocess organize_dicoms --dir DIR --outDir OUTDIR --indexOut INDEXOUT
    [--quarantineDir QUARANTINEDIR] [--logOut LOGOUT] [--unresolvedOut UNRESOLVEDOUT]
    [--dryRun] [--n_workers N_WORKERS] [--noProgress]
```

### Normalized output

The destination uses the following structure:

```text
out_dir/
  Patient_<PatientID>_<hash>/
    Exam_<StudyInstanceUID hash>/
      Series_<SeriesNumber>_<SeriesDescription>_<SeriesInstanceUID hash>/
        DICOM files
  _quarantine/
    <source-relative paths of quarantined files>
```

The exact filesystem-safe labels are implementation details; the index is the
authoritative mapping back to source files. Patient folder names include the
DICOM `PatientID`, which may be PHI. NIfTI outputs use the `patientID` values
assigned in the patient metadata table instead.

### Organization index columns

| Column | Meaning |
|---|---|
| `source_path` | Original source file path. |
| `source_relpath` | Source path relative to `root_dir`. |
| `normalized_path` | Destination path for a successfully organized file. |
| `quarantine_path` | Destination path, when an unresolved file is quarantined. |
| `patient_id` | DICOM `PatientID`, or deterministic fallback. |
| `patient_name` | DICOM `PatientName`, when available. |
| `study_instance_uid` | DICOM `StudyInstanceUID`; defines exam identity. |
| `study_id` | Study identifier retained for display or traceability. |
| `study_date` | DICOM study date, if present; not used for exam identity. |
| `series_instance_uid` | DICOM `SeriesInstanceUID`; defines series identity. |
| `series_number` | DICOM series number, if present. |
| `series_description` | DICOM series description, if present. |
| `protocol_name` | DICOM protocol name, if present. |
| `modality` | DICOM `Modality` value. Common values are `MR` (magnetic resonance), `CT` (computed tomography), `US` (ultrasound), `PT` or `PET` (positron emission tomography), `NM` (nuclear medicine), `CR` (computed radiography), `DX` (digital radiography), `MG` (mammography), `XA` (X-ray angiography), and `OT` (other). `PR` means Presentation State (display/annotation instructions), and `SR` means Structured Report (structured clinical/reporting data); these are generally non-image objects and are not usually suitable for NIfTI conversion. The value is preserved from DICOM metadata, so any valid DICOM modality code may occur; it is not inferred from the folder name. If the tag is absent, the column is empty. |
| `sop_class_uid` | DICOM SOP Class UID. |
| `sequence_type` | Best-effort normalized type, such as `MRI`, `CT`, `Ultrasound`, or `PET`. |
| `sequence_type_source` | Metadata field or rule used for the sequence-type assignment. |
| `sop_instance_uid` | DICOM SOP Instance UID. |
| `manufacturer` | DICOM manufacturer. |
| `acq_dt_iso` | Acquisition date/time in ISO-like form, when available. |
| `sequence_name` | DICOM sequence name. |
| `image_type` | DICOM `ImageType` values, serialized with semicolon separators. |
| `imgtype_flags` | Normalized image-type flags used by series classification. |
| `echo_time` | Echo time. |
| `repetition_time` | Repetition time. |
| `inversion_time` | Inversion time. |
| `flip_angle` | Flip angle. |
| `b_value` | Diffusion b-value. |
| `primary_secondary` | `PRIMARY`, `SECONDARY`, or empty based on `ImageType`. |
| `is_fspgr` | Whether the series looks like an FSPGR acquisition. |
| `plane` | Acquisition plane inferred from series text or image orientation. |
| `rows_px`, `cols_px` | Image matrix dimensions. |
| `pixel_spacing` | DICOM pixel-spacing values. |
| `pixdim_row_mm`, `pixdim_col_mm` | In-plane pixel spacing in millimeters. |
| `slice_thickness_mm` | Slice thickness in millimeters. |
| `spacing_between_slices_mm`, `spacing_between_slices` | Spacing between slices (parsed millimeters and raw DICOM value). |
| `z_spacing_mm` | Through-plane spacing: spacing between slices, or slice thickness when absent. |
| `n_slices_per_vol_est` | Estimated number of slices per volume. |
| `num_frames` | Number of frames, when present. |
| `images_in_acq`, `locations_in_acq` | DICOM acquisition/location counts. |
| `mr_acq_type` | MR acquisition type. |
| `pulse_sequence_name`, `scanning_sequence`, `sequence_variant`, `scan_options` | Vendor and sequence descriptors. |
| `acquisition_contrast`, `contrast_agent`, `contrast_volume` | Contrast metadata. |
| `study_description`, `procedure_step_description` | Study/procedure descriptions. |
| `status` | Organization result; see the status values below. |
| `error` | Error or diagnostic message for non-success records. |

The index can contain one row per DICOM file. A series may therefore have many
rows, while all rows for one `StudyInstanceUID` map to the same exam directory.
The expanded fields are the raw inputs needed by series classification; the
planner aggregates these rows into one classifier record per series.

### Status values

| `status` | Meaning | Used downstream |
|---|---|---|
| `copied` | File was copied into the organized tree. | Yes |
| `skipped_existing` | An identical file already existed at the destination. | Yes |
| `duplicate_sop` | The same SOP Instance UID occurred more than once; one copy is retained. | Yes |
| `planned` | Dry run only; the file would have been copied. | No |
| `quarantined` | File was unreadable, missing identity metadata, or had a UID conflict; see `error`. | No |
| `conflict` | A different file already existed at the destination. | No |

### Unresolved-file report

The unresolved report is a filtered copy of the organization records containing
statuses other than `copied`, `skipped_existing`, or `planned`, or a nonempty
`error`. It is intended for triage and reruns. It retains source paths and all
available DICOM context, so unresolved files can be investigated without
relying on the original folder layout.

## `create_patient_metadata`

### Purpose

Create a patient-level table for assigning Astril patient identifiers and
reference dates before conversion planning.

### Python API

```python
create_patient_metadata(
    root_dir: str,
    out_path: str,
    previous_paths: list[str] | None = None,
    omit_previous: bool = False,
    subdirs: list[str] | None = None,
    exclude_empty: bool = False,
    show_progress: bool = True,
    n_workers: int | None = None,
    index_path: str | None = None,
)
```

For the indexed workflow, pass the organized library from `organize_dicoms` as
`root_dir` and its index as `index_path` (`--index` on the CLI). Patient rows
are then grouped by organized patient directory, and `subdirs` is not used.
If multiple names or DICOM patient IDs occur in one directory, they are
combined into one value separated by `; `.

Without `index_path`, the function falls back to the legacy folder scan, which
expects `{Patient}/.../MR/{series}` folders (configurable with `subdirs`). That
mode does not work on the organized tree.

`previous_paths` prefills `patientID` and `day0Date` from earlier tables with
matching `Directory` values; `omit_previous=True` drops those rows instead.

### CLI

```text
python -m astril.preprocess create_patient_metadata --dir DIR --metadataOut METADATAOUT
    [--index INDEX] [--previousMetadata [PREVIOUSMETADATA ...]] [--omitPrevious]
    [--subdirs SUBDIRS [SUBDIRS ...]] [--excludeEmpty] [--n_workers N_WORKERS]
```

### Output

The table (CSV, TSV, or XLSX) has these columns:

| Column | Meaning |
|---|---|
| `Directory` | Patient folder relative to `root_dir`. Must match an existing folder. |
| `patientID` | User-assigned identifier used in NIfTI output paths. Blank unless prefilled. |
| `patientName` | Lowercased DICOM patient name(s). |
| `dicomPatientID` | Lowercased DICOM patient ID(s). |
| `day0Date` | User-assigned reference date (mm/dd/yyyy) for timepoint calculation. Blank unless prefilled. |

Users must fill in `patientID` and `day0Date`. The planner skips rows missing
either value. If you edit the table in Excel, disable automatic data
conversion so `Directory` values are not altered.

## `plan_dicom_to_nifti_conversion`

### Purpose

Associate indexed DICOM series with patient metadata, infer conversion labels,
and produce an editable DICOM-to-NIfTI conversion plan.

### Python API

```python
plan_dicom_to_nifti_conversion(
    patient_metadata: str,
    root_dir: str,
    out_dir: str,
    dicom_index: str,
    plan_out: str,
    n_workers: int | None = None,
    show_progress: bool = True,
    previous_plans: list[str] | None = None,
    ignore_previous: bool = False,
    min_slices: int = 10,
    use_actual_exam_ids: bool = False,
    add_missing_derived: bool = False,
    make_derived_from_scratch: bool = False,
    unexpected_multiframe_policy: str = "keep_first",
)
```

Pass the organization index with `dicom_index` and the organized root with
`root_dir`. The index is required; the planner no longer supports legacy
folder-based DICOM discovery. Do not use the removed legacy `--mrSubdirs`
option. The planner uses the index to select organized series and performs no
DICOM reads; the patient metadata table supplies assigned patient IDs and
day-zero dates. `plan_out` is required and must be a CSV or TSV path. Only
index rows with status `copied`, `skipped_existing`, or `duplicate_sop` are
planned.

The generated plan includes the discovered DICOM-series metadata and the
conversion decisions, especially `selected_for_conversion` and
`proposed_nifti_path`. Review and edit these columns before conversion; only
selected rows with valid proposed paths are converted. Proposed paths follow
`{out_dir}/{patientID}/{patientID}_{timepoint_days}_{ExamAlias}/{patientID}_{timepoint_days}_{final_label}.nii.gz`.

The Python function returns a summary dictionary with `plan_out`, `n_exams`,
`n_series`, `n_selected`, `n_derived`, and `n_errors`; it does not return the
complete plan table.

### CLI

```text
python -m astril.preprocess plan_dicom_to_nifti_conversion --patientMetadata PATIENTMETADATA
    --dir DIR --dicomIndex DICOMINDEX --outDir OUTDIR --planOut PLANOUT
    [--n_workers N_WORKERS] [--noProgress] [--previousPlan [PREVIOUSPLAN ...]]
    [--ignorePrevious] [--minSlices MINSLICES] [--use_actual_exam_ids]
    [--add_missing_derived] [--make_derived_from_scratch]
    [--unexpectedMultiframePolicy {keep_first,skip}]
```

## `convert_dicom_plan`

### Purpose

Run the conversions described by a reviewed plan.

### Python API

```python
convert_dicom_plan(
    plan_path: str,
    n_workers: int | None = None,
    overwrite: bool = False,
    log_out: str | None = None,
    show_progress: bool = True,
    unexpected_multiframe_policy: str = "keep_first",
    debug: bool = False,
)
```

The plan is the only data input; it records the source series folder and
target path for each row. Rows are converted when `selected_for_conversion` is
true and `proposed_nifti_path` is non-empty and not `-`. `CONVERT` rows run
before `DERIVE` rows. Existing outputs are kept unless `overwrite=True`.

`unexpected_multiframe_policy` controls series expected to be 3D that convert
to multi-frame/4D: `keep_first` keeps frame 0 with a warning, and `skip`
deletes the output with a warning. A per-row CSV/TSV log is streamed to
`log_out`, and the function returns the log as a pandas DataFrame.

### CLI

```text
python -m astril.preprocess convert_dicom_plan --plan PLAN [--n_workers N_WORKERS]
    [--overwrite] [--logOut LOGOUT] [--unexpectedMultiframePolicy {keep_first,skip}] [--debug]
```

## `demix_dicoms`

### Purpose

Separate files that were placed in the wrong series directory by examining
series-related DICOM metadata. This is a legacy cleanup step and is not needed
in the indexed workflow, because `organize_dicoms` already routes files by
`SeriesInstanceUID`.

### Python API

```python
demix_dicoms(
    root_dir: str,
    show_progress: bool = True,
    log_out: str | None = None,
    out_dir: str | None = None,
    dry_run: bool = False,
    n_workers: int | None = None,
    in_place: bool = False,
)
```

Use `out_dir` for a de-mixed copy. In-place moves require explicit
`in_place=True`. This function expects the legacy patient/exam/series-style
layout closely enough to identify the series directories; it is not the
metadata-indexed entry point. Use `organize_dicoms` when the input layout is
unknown or inconsistent.

## Recommended indexed command sequence

```powershell
python -m astril.preprocess organize_dicoms `
  --dir D:\DICOM_Library `
  --outDir D:\Organized_DICOM `
  --indexOut D:\Organized_DICOM\dicom_index.csv

# Review D:\Organized_DICOM\dicom_index_unresolved.csv for quarantined or conflicting files.

python -m astril.preprocess create_patient_metadata `
  --dir D:\Organized_DICOM `
  --index D:\Organized_DICOM\dicom_index.csv `
  --metadataOut D:\Organized_DICOM\patient_metadata.csv

# Fill in patientID and day0Date in patient_metadata.csv.

python -m astril.preprocess plan_dicom_to_nifti_conversion `
  --patientMetadata D:\Organized_DICOM\patient_metadata.csv `
  --dir D:\Organized_DICOM `
  --dicomIndex D:\Organized_DICOM\dicom_index.csv `
  --outDir D:\NIFTI_Library `
  --planOut D:\Organized_DICOM\conversion_plan.csv

# Review selected_for_conversion and proposed_nifti_path in conversion_plan.csv.

python -m astril.preprocess convert_dicom_plan `
  --plan D:\Organized_DICOM\conversion_plan.csv
```

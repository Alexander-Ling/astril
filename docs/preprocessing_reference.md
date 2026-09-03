# DICOM preprocessing reference

This document is the reference for Astril's DICOM-library preprocessing
functions. The command-line interface exposes the same functions through
`python -m astril.preprocess`.

The metadata-indexed workflow is:

```text
organize_dicoms -> demix_dicoms (optional) -> create_patient_metadata (optional)
                 -> plan_dicom_to_nifti_conversion
```

`organize_dicoms` is the preferred entry point for a library whose input
folders may not use a consistent naming convention. It reads DICOM metadata
recursively and creates a normalized Patient/Exam/Series tree. Downstream
functions can then use the generated index instead of inferring identity from
folder names.

## `organize_dicoms`

### Purpose

Scan a DICOM library recursively, copy readable DICOM files into a normalized
directory hierarchy, and write a machine-readable organization index.

Patient, exam, and series identity are metadata-based:

- Patient: `PatientID` (with a deterministic fallback when absent)
- Exam: `StudyInstanceUID` exclusively
- Series: `SeriesInstanceUID` (with a deterministic fallback when absent)

The input may be nested differently for different patients and may contain
suffixless DICOM files. Existing destination files are not overwritten:
identical files are skipped and conflicting files are recorded as conflicts.

### Python API

```python
organize_dicoms(
    root_dir: str,
    out_dir: str,
    index_out: str | None = None,
    n_workers: int | None = None,
    show_progress: bool = True,
    unresolved_out: str | None = None,
)
```

`root_dir` is the source library. `out_dir` is the normalized output root.
`index_out` writes the organization index as CSV or TSV. CSV/TSV are required
for scalable indexed processing; XLSX is not supported for the organization
index. If supplied,
the default unresolved report is written beside it as
`<index stem>_unresolved.csv`; otherwise it is written beneath `out_dir` as
`unresolved_dicom_report.csv`. `n_workers` controls metadata reads and
copying, and `show_progress` controls progress display.

The function returns the generated organization records for programmatic
callers. A Ctrl+C interruption is handled cooperatively: completed records
and the unresolved report are preserved when possible.

### Normalized output

The destination uses the following logical structure:

```text
out_dir/
  Patient_<patient key>/
    Exam_<StudyInstanceUID key>/
      <sequence or series label>/
        DICOM files
```

The exact filesystem-safe labels are implementation details; the index is the
authoritative mapping back to source files.

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
| `echo_time` | Echo time. |
| `repetition_time` | Repetition time. |
| `inversion_time` | Inversion time. |
| `flip_angle` | Flip angle. |
| `b_value` | Diffusion b-value. |
| `primary_secondary` | `PRIMARY`, `SECONDARY`, or empty based on `ImageType`. |
| `rows_px`, `cols_px` | Image matrix dimensions. |
| `pixel_spacing` | DICOM pixel-spacing values. |
| `slice_thickness` | Slice thickness. |
| `spacing_between_slices` | Spacing between slices. |
| `num_frames` | Number of frames, when present. |
| `images_in_acq`, `locations_in_acq` | DICOM acquisition/location counts. |
| `mr_acq_type` | MR acquisition type. |
| `pulse_sequence_name`, `scanning_sequence`, `sequence_variant`, `scan_options` | Vendor and sequence descriptors. |
| `acquisition_contrast`, `contrast_agent`, `contrast_volume` | Contrast metadata. |
| `study_description`, `procedure_step_description` | Study/procedure descriptions. |
| `status` | Organization result, such as success, skipped, conflict, or unresolved. |
| `error` | Error or diagnostic message for non-success records. |

The index can contain one row per DICOM file. A series may therefore have many
rows, while all rows for one `StudyInstanceUID` map to the same exam directory.
The expanded fields are the raw inputs needed by series classification; the
planner aggregates these rows into one classifier record per series.

### Unresolved-file report

The unresolved report is a filtered copy of the organization records containing
non-success statuses or a nonempty `error`. It is intended for triage and
reruns. It retains source paths and all available DICOM context, so unresolved
files can be investigated without relying on the original folder layout.

## `demix_dicoms`

### Purpose

Separate files that were placed in the wrong series directory by examining
series-related DICOM metadata. This is an optional legacy cleanup step.

### Python API

```python
demix_dicoms(
    dir: str,
    out_dir: str | None = None,
    log_out: str | None = None,
    n_workers: int = 12,
    dry_run: bool = False,
    in_place: bool = False,
    show_progress: bool = True,
)
```

Use `out_dir` for a de-mixed copy. In-place moves require explicit
`in_place=True`. This function expects the legacy patient/exam/series-style
layout closely enough to identify the series directories; it is not the
metadata-indexed entry point. Prefer `organize_dicoms` when the input layout is
unknown or inconsistent.

## `create_patient_metadata`

### Purpose

Create a patient-level table for assigning Astril patient identifiers and
reference dates before conversion planning.

### Python API

```python
create_patient_metadata(
    root_dir: str,
    metadata_out: str,
    previous_metadata: list[str] | None = None,
    omit_previous: bool = False,
    subdirs: list[str] | None = None,
    exclude_empty: bool = False,
    n_workers: int | None = None,
    dicom_index: str,
)
```

With `dicom_index`, patient rows are grouped by normalized patient directory,
not by individual `patientName` values. If multiple names occur in one
directory, they are combined into one `patientName` value separated by `; `.

The output table contains the patient directory, assigned patient ID, DICOM
patient ID/name, and the day-zero date used for timepoint calculation. The
exact output column names are preserved by the selected CSV, TSV, or XLSX
writer and should be treated as the input contract for the planner.

## `plan_dicom_to_nifti_conversion`

### Purpose

Inspect DICOM series, associate them with patient metadata, infer conversion
labels, and produce an editable DICOM-to-NIfTI conversion plan.

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
day-zero dates. `plan_out` is required and must be a CSV or TSV path.

The generated plan includes the discovered DICOM-series metadata and the
conversion decisions, especially `selected_for_conversion` and
`proposed_nifti_path`. Only selected rows with valid proposed paths are
intended for conversion. The Python function returns a summary dictionary
with `plan_out`, `n_exams`, `n_series`, `n_selected`, `n_derived`, and
`n_errors`; it does not return the complete plan table.

## Recommended indexed command sequence

```powershell
python -m astril.preprocess organize_dicoms `
  --dir D:\DICOM_Library `
  --outDir D:\Organized_DICOM `
  --indexOut D:\Organized_DICOM\dicom_index.csv

python -m astril.preprocess create_patient_metadata `
  --dir D:\Organized_DICOM `
  --metadataOut D:\Organized_DICOM\patient_metadata.csv `
  --dicomIndex D:\Organized_DICOM\dicom_index.csv

python -m astril.preprocess plan_dicom_to_nifti_conversion `
  --patientMetadata D:\Organized_DICOM\patient_metadata.csv `
  --dir D:\Organized_DICOM `
  --outDir D:\NIFTI_Library `
  --planOut D:\Organized_DICOM\conversion_plan.csv `
  --dicomIndex D:\Organized_DICOM\dicom_index.csv
```

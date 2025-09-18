from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

# We import and call your existing function directly (no subprocess).
# If your package layout differs (e.g., installed as astril), adjust the import:
from .preprocess_single_brain_mri import preprocess_single_brain_mri  # type: ignore


# ----------------------------
# Helpers
# ----------------------------

REQ_LABELS = ("T1c", "T1n", "T2f", "T2w")

try:
    # auto picks the right frontend (console/notebook) and tends to behave better on Windows/PwSh
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # fallback to a simple counter

def _find_required_scans(exam_dir: Path) -> dict[str, Path] | None:
    """
    Return dict with keys T1c,T1n,T2f,T2w -> Path if all present, else None.
    Prefers .nii.gz over .nii if both exist.
    """
    found: dict[str, Path] = {}
    for lbl in REQ_LABELS:
        # Prefer .nii.gz
        gz = list(exam_dir.glob(f"*_{lbl}.nii.gz"))
        ni = list(exam_dir.glob(f"*_{lbl}.nii"))
        cand = gz[0] if gz else (ni[0] if ni else None)
        if cand is None:
            return None
        found[lbl] = cand
    return found


def _parse_patient_exam(exam_dir: Path) -> tuple[str, str]:
    """
    Parse patient and exam names from the directory structure:
        {in_dir}/{patient}/{exam}/
    """
    patient = exam_dir.parent.name
    exam = exam_dir.name
    return patient, exam

def _exam_prefix_from_dir(exam_dir: Path) -> str:
    """
    Expected exam dir name: {patient}_{timepoint}_{ExamAlias}
    We want output prefix == "{patient}_{timepoint}" to match the single-run outputs.
    """
    parts = exam_dir.name.split("_", 2)
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    # Fallback: try to derive from any T1c file name
    t1c = list(exam_dir.glob("*_T1c.nii.gz")) or list(exam_dir.glob("*_T1c.nii"))
    if t1c:
        stem = t1c[0].name
        if stem.lower().endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.lower().endswith(".nii"):
            stem = stem[:-4]
        # drop trailing "_T1c"
        if stem.endswith("_T1c"):
            stem = stem[:-5]
        return stem
    # Worst case, return dir name
    return exam_dir.name

def _outputs_complete(out_exam: Path, prefix: str) -> bool:
    """
    Check for the full set of expected single-run outputs in out_exam:
      - brainmask
      - for each modality in REQ_LABELS: *_brain.nii.gz and *_brain_norm.nii.gz
      - transform_record.json per modality
    """
    expected = [
        out_exam / f"{prefix}_brainmask.nii.gz",
    ]
    for mod in REQ_LABELS:
        expected += [
            out_exam / f"{prefix}_{mod}_brain.nii.gz",
            out_exam / f"{prefix}_{mod}_brain_norm.nii.gz",
            out_exam / f"{prefix}_{mod}_transform_record.json",
        ]
    return all(p.exists() for p in expected)

def _parse_timepoint_from_exam_name(exam_name: str) -> tuple[int, bool]:
    """
    Exam dir format is typically: {patient}_{timepoint}_{ExamAlias}
    e.g. '032_d1690_SH8JJOMZ' or '020_d-1539_2NLZ8ZMA'

    Returns (day_number, has_day), where day_number is an integer (can be negative).
    If parsing fails, returns (0, False) so those sort after valid day entries.
    """
    parts = exam_name.split("_", 2)
    if len(parts) < 2:
        return (0, False)
    tp = parts[1]  # e.g., 'd1690' or 'd-1539'
    if not tp or tp[0].lower() != "d":
        return (0, False)
    try:
        day = int(tp[1:])
        return (day, True)
    except ValueError:
        return (0, False)


def _choose_reference_for_patient(
    patient_dir: Path,
    in_dir: Path,
    old_coreg_maps: dict[str, Path] | None,
) -> Path | None:
    """
    If old_coreg_maps has an entry for this patient, reuse it (if it exists).
    Else pick earliest timepoint (smallest 'day') that has a T1c.
    Returns absolute Path to the reference T1c, or None if not found.
    """
    patient = patient_dir.name

    # A) Old logs preference
    if old_coreg_maps and patient in old_coreg_maps:
        candidate = in_dir / old_coreg_maps[patient]
        if candidate.exists():
            return candidate

    # B) Earliest timepoint with T1c
    exams = [p for p in patient_dir.iterdir() if p.is_dir()]
    # Sort by parsed day; invalids go last
    exams_sorted = sorted(
        exams,
        key=lambda p: (_parse_timepoint_from_exam_name(p.name)[1] is False,
                       _parse_timepoint_from_exam_name(p.name)[0])
    )
    for ex in exams_sorted:
        t1c = list(ex.glob("*_T1c.nii.gz")) or list(ex.glob("*_T1c.nii"))
        if t1c:
            return t1c[0]
    return None


def _read_old_coreg_logs(paths: list[Path]) -> dict[str, Path]:
    """
    Read one or more prior coreg logs and build patient -> relative path mapping.
    If duplicates occur, the first file wins (earlier in the provided list).
    Expected headers include at least: Patient, ReferenceRelativePath
    """
    mapping: dict[str, Path] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Flexible header handling
            patient_key = None
            path_key = None
            for k in reader.fieldnames or []:
                lk = k.strip().lower()
                if lk in ("patient", "patientid", "patient_id"):
                    patient_key = k
                if lk in ("referencerelativepath", "reference_relative_path", "relative_ref_path"):
                    path_key = k
            if not patient_key or not path_key:
                continue
            for row in reader:
                pat = (row.get(patient_key) or "").strip()
                rel = (row.get(path_key) or "").strip()
                if pat and rel and pat not in mapping:
                    mapping[pat] = Path(rel)
    return mapping


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------------
# Core batch function
# ----------------------------

def preprocess_library(
    in_dir: str | Path,
    out_dir: str | Path,
    preprocess_log: str | Path | None = None,
    coreg_log: str | Path | None = None,
    old_coreg_log: list[str | Path] | None = None,
    dont_coregister: bool = False,
    n_workers: int = 1,
    overwrite: bool = False,
    # Passthrough options to preprocess_single_brain_mri
    final_dims: list[int] | None = None,
    final_voxels: list[float] | None = None,
    save_scans_with_skulls: bool = False,
    debug: bool = False,
) -> tuple[Path, Path]:
    """
    Walk {in_dir}/{patient}/{exam}, find T1c/T1n/T2f/T2w in each exam,
    optionally pick a patient-level reference T1c for co-registration,
    and run preprocess_single_brain_mri on each exam into {out_dir}/{patient}/{exam}.

    Returns (preprocess_log_path, coreg_log_path).
    """
    in_dir = Path(in_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default log paths
    if preprocess_log is None:
        preprocess_log = out_dir / f"preprocess_log_{_now_stamp()}.csv"
    else:
        preprocess_log = Path(preprocess_log)
        preprocess_log.parent.mkdir(parents=True, exist_ok=True)

    if coreg_log is None:
        coreg_log = out_dir / f"coreg_log_{_now_stamp()}.csv"
    else:
        coreg_log = Path(coreg_log)
        coreg_log.parent.mkdir(parents=True, exist_ok=True)

    # Build patient list
    patient_dirs = [p for p in in_dir.iterdir() if p.is_dir()]

    # Read previous reference logs (optional)
    old_maps: dict[str, Path] | None = None
    if old_coreg_log:
        old_paths = [Path(p) for p in old_coreg_log]
        old_maps = _read_old_coreg_logs(old_paths)

    # Determine a reference for each patient (or None)
    patient_ref: dict[str, Path | None] = {}
    for pd in patient_dirs:
        if dont_coregister:
            patient_ref[pd.name] = None
        else:
            patient_ref[pd.name] = _choose_reference_for_patient(pd, in_dir, old_maps)

    # Write the coreg log up front (what we will try to use)
    with coreg_log.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Patient", "ReferenceRelativePath"])
        for pat, ref in sorted(patient_ref.items(), key=lambda kv: kv[0]):
            rel = ""
            if ref is not None:
                # Relative to in_dir as requested
                try:
                    rel = str(ref.resolve().relative_to(in_dir))
                except Exception:
                    # If ref isn't under in_dir, still record absolute path
                    rel = str(ref)
            w.writerow([pat, rel])

    # Collect exam tasks
    tasks: list[tuple[Path, Path, dict[str, Path], Path | None]] = []
    for pd in patient_dirs:
        for ex in sorted(p for p in pd.iterdir() if p.is_dir()):
            scans = _find_required_scans(ex)
            if scans is None:
                # We will log a skip later
                tasks.append((pd, ex, {}, patient_ref[pd.name]))
            else:
                tasks.append((pd, ex, scans, patient_ref[pd.name]))

    # Prepare the preprocess log (delimiter from extension; line-buffered)
    delim = "\t" if str(preprocess_log).lower().endswith(".tsv") else ","
    with preprocess_log.open("w", newline="", encoding="utf-8", buffering=1) as f:
        w = csv.writer(f, delimiter=delim)
        w.writerow([
            "Patient", "Exam", "Status", "Message",
            "InExamDir", "OutExamDir",
            "T1c", "T1n", "T2f", "T2w",
            "CoregisterRefUsed",
            "final_dims", "final_voxels", "save_scans_with_skulls"
        ])
        f.flush()
        os.fsync(f.fileno())

    # Progress setup
    total = len(tasks)
    lock = threading.Lock()
    if tqdm:
        # tie to stdout explicitly; dynamic_ncols helps with odd terminals; leave=False keeps console tidy
        pbar = tqdm(total=total, desc="Preprocessing exams", unit="exam",
                    dynamic_ncols=True, leave=False, file=sys.stdout)
    else:
        pbar = None
        done_count = 0

    def _advance():
        nonlocal done_count
        if pbar:
            pbar.update(1)
            # Force a redraw so progress is visible even when other processes are spamming stdout
            pbar.refresh()
        else:
            with lock:
                done_count += 1
                print(f"[Progress] {done_count}/{total} exams completed", flush=True)

    # Worker function
    def _run_exam(pd: Path, ex: Path, scans: dict[str, Path], ref: Path | None) -> tuple[str, list[str]]:
        patient, exam = _parse_patient_exam(ex)
        out_exam = out_dir / patient / exam

        if not scans:
            msg = "Missing required scans; skipped."
            return "SKIPPED", [patient, exam, "SKIPPED", msg, str(ex), str(out_exam), "", "", "", "", str(ref or "NONE"), str(final_dims), str(final_voxels), save_scans_with_skulls]

        # Skip if outputs already complete and not overwriting
        prefix = _exam_prefix_from_dir(ex)
        if (not overwrite) and _outputs_complete(out_exam, prefix):
            msg = "Already processed (all expected outputs present); skipped."
            return "SKIPPED", [patient, exam, "SKIPPED", msg, str(ex), str(out_exam),
                               str(scans['T1c']), str(scans['T1n']), str(scans['T2f']), str(scans['T2w']),
                               str(ref or "NONE"), str(final_dims), str(final_voxels), save_scans_with_skulls]

        out_exam.mkdir(parents=True, exist_ok=True)

        t1c = str(scans["T1c"])
        t1n = str(scans["T1n"])
        t2f = str(scans["T2f"])
        t2w = str(scans["T2w"])

        started = time.time()
        try:
            preprocess_single_brain_mri(
                t1c_path=t1c,
                t1n_path=t1n,
                t2f_path=t2f,
                t2w_path=t2w,
                output_dir=str(out_exam),
                co_register_path=None if dont_coregister else (str(ref) if ref else None),
                # passthroughs
                final_dims=final_dims,
                final_voxels=final_voxels,
                save_scans_with_skulls=save_scans_with_skulls,
                debug=debug,
            )
            dur = f"{time.time() - started:.1f}s"
            return "OK", [patient, exam, "OK", dur, str(ex), str(out_exam), t1c, t1n, t2f, t2w, str(ref or "NONE"), str(final_dims), str(final_voxels), save_scans_with_skulls]
        except Exception as e:
            msg = f"ERROR: {e.__class__.__name__}: {e}"
            return "ERROR", [patient, exam, "ERROR", msg, str(ex), str(out_exam), t1c, t1n, t2f, t2w, str(ref or "NONE"), str(final_dims), str(final_voxels), save_scans_with_skulls]

    # Execute (optionally parallel)
    if n_workers and n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool, \
             preprocess_log.open("a", newline="", encoding="utf-8", buffering=1) as f:
            w = csv.writer(f, delimiter=delim)
            futs = {pool.submit(_run_exam, *t): t for t in tasks}
            for fut in as_completed(futs):
                try:
                    status, row = fut.result()
                    w.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    _advance()
    else:
        with preprocess_log.open("a", newline="", encoding="utf-8", buffering=1) as f:
            w = csv.writer(f, delimiter=delim)
            for t in tasks:
                try:
                    status, row = _run_exam(*t)
                    w.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    _advance()

    if pbar:
        pbar.close()
    return Path(preprocess_log), Path(coreg_log)


# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser(
        description="Batch brain MRI preprocessing over a NIFTI library (patient/exam tree)."
    )
    p.add_argument("--in_dir", required=True, help="Path to NIFTI library root (converted by convert_dicom_plan).")
    p.add_argument("--out_dir", required=True, help="Path to write preprocessed outputs (patient/exam structure).")
    p.add_argument("--preprocess_log", default=None, help="CSV path for per-exam outcomes. Auto-generated if omitted.")
    p.add_argument("--coreg_log", default=None, help="CSV path for chosen per-patient reference. Auto-generated if omitted.")
    p.add_argument("--old_coreg_log", nargs="*", default=None, help="One or more prior coreg logs to reuse references.")
    p.add_argument("--dont_coregister", action="store_true", help="Disable co-registration; run each exam independently.")
    p.add_argument("--n_workers", type=int, default=1, help="Parallel workers (threaded). Use 1 to run serially.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs instead of skipping completed exams.")


    # Passthroughs to preprocess_single_brain_mri
    p.add_argument("--final_dims", type=int, nargs=3, default=(240, 240, 155), metavar=("NX", "NY", "NZ"),
                   help="Final dimensions (e.g., 192 224 160).")
    p.add_argument("--final_voxels", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("SX", "SY", "SZ"),
                   help="Final voxel sizes in mm (e.g., 1.0 1.0 1.0).")
    p.add_argument("--save_scans_with_skulls", action="store_true", help="Also save skull-on registered scans (PHI risk).")
    p.add_argument("--debug", action="store_true", help="Keep temp dirs from underlying preprocessing.")

    args = p.parse_args()

    preprocess_library(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        preprocess_log=args.preprocess_log,
        coreg_log=args.coreg_log,
        old_coreg_log=args.old_coreg_log,
        dont_coregister=args.dont_coregister,
        n_workers=args.n_workers,
        overwrite=args.overwrite,
        final_dims=args.final_dims,
        final_voxels=args.final_voxels,
        save_scans_with_skulls=args.save_scans_with_skulls,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()

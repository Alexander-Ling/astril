from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# We import and call your existing function directly (no subprocess).
# If your package layout differs (e.g., installed as astril), adjust the import:
from .preprocess_single_brain_mri import preprocess_single_brain_mri  # type: ignore


try:
    # auto picks the right frontend (console/notebook) and tends to behave better on Windows/PwSh
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # fallback to a simple counter


# ----------------------------
# Helpers
# ----------------------------

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    # Fallback: use dir name
    return exam_dir.name


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


def _estimate_anchor_quality(nifti_path: Path) -> tuple[float, dict]:
    """
    Heuristic quality score for picking a per-patient co-registration reference.

    Higher is better.

    Returns:
        (score, info_dict)

    info_dict contains lightweight metadata that we can log for auditing:
        - shape
        - zooms_mm
        - voxel_volume_mm3
        - anisotropy
        - sample_std
        - sample_nonzero_frac
        - error (optional)
    """
    info: dict = {
        "shape": "",
        "zooms_mm": "",
        "voxel_volume_mm3": "",
        "anisotropy": "",
        "sample_std": "",
        "sample_nonzero_frac": "",
    }
    try:
        import math
        import numpy as np
        import nibabel as nib

        img = nib.load(os.fspath(nifti_path))
        shape = tuple(int(x) for x in img.shape)
        if len(shape) != 3:
            info["error"] = f"ndim={len(shape)}"
            return float("-inf"), info

        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        if any((not np.isfinite(z)) or (z <= 0) for z in zooms):
            info["error"] = f"bad_zooms={zooms}"
            return float("-inf"), info

        vx_vol = float(zooms[0] * zooms[1] * zooms[2])
        aniso = float(max(zooms) / max(1e-6, min(zooms)))

        # Sample a few central slices without loading the whole volume
        dataobj = img.dataobj
        z = shape[2]
        zs = sorted(set([
            int(z * 0.50),
            int(z * 0.40),
            int(z * 0.60),
            int(z * 0.33),
            int(z * 0.67),
        ]))
        zs = [zi for zi in zs if 0 <= zi < z]

        samples = []
        for zi in zs:
            sl = np.asanyarray(dataobj[:, :, zi]).astype(np.float32, copy=False)
            if sl.size:
                samples.append(sl)
        if not samples:
            info["error"] = "no_samples"
            return float("-inf"), info

        samp = np.concatenate([s.ravel() for s in samples])
        if samp.size < 1024:
            info["error"] = "small_sample"
            return float("-inf"), info
        if not np.isfinite(samp).all():
            info["error"] = "nonfinite"
            return float("-inf"), info

        std = float(np.std(samp))
        nonzero_frac = float(np.mean(np.abs(samp) > 1e-6))

        if std < 1e-4 or nonzero_frac < 0.01:
            info["error"] = f"low_signal std={std:.3g} nonzero_frac={nonzero_frac:.3g}"
            return float("-inf"), info

        # Populate info for logging
        info.update(
            {
                "shape": "x".join(map(str, shape)),
                "zooms_mm": ",".join(f"{z:.6g}" for z in zooms),
                "voxel_volume_mm3": f"{vx_vol:.6g}",
                "anisotropy": f"{aniso:.6g}",
                "sample_std": f"{std:.6g}",
                "sample_nonzero_frac": f"{nonzero_frac:.6g}",
            }
        )

        # Score components (simple, fast, and fairly robust)
        nvox = float(shape[0] * shape[1] * shape[2])

        score = 0.0
        # Resolution bonus: smaller voxel volume => higher
        score += 50.0 * (1.0 / max(vx_vol, 1e-6))
        # Mild preference for larger volumes (coverage)
        score += 1e-7 * nvox
        # Penalize anisotropy (thick slices)
        score -= 2.5 * (aniso - 1.0)
        # Mild contrast/texture bonus
        score += 2.0 * math.log1p(std)
        # Prefer non-empty samples
        score += 2.0 * nonzero_frac

        return float(score), info

    except Exception as e:
        info["error"] = str(e)
        return float("-inf"), info


def _choose_reference_for_patient(
    patient_dir: Path,
    in_dir: Path,
    old_coreg_maps: dict[str, Path] | None,
    anchor_label: str,
    reference_selection: str = "highest_quality",
) -> tuple[Path | None, dict]:
    """
    Choose the per-patient reference image used for co-registration.

    Priority:
      1) If old_coreg_maps has an entry for this patient, reuse it (if it exists).
      2) Otherwise:
           - reference_selection == 'highest_quality': score all candidate anchor volumes and pick the best
           - reference_selection == 'earliest': pick earliest timepoint that has an anchor (legacy behavior)

    Returns:
      (reference_path_or_none, info_dict_for_coreg_log)
    """
    patient = patient_dir.name
    info: dict = {
        "SelectionMethod": "",
        "QualityScore": "",
        "Shape": "",
        "ZoomsMM": "",
        "VoxelVolumeMM3": "",
        "Anisotropy": "",
        "SampleStd": "",
        "SampleNonzeroFrac": "",
        "Error": "",
    }

    # A) Old logs preference
    if old_coreg_maps and patient in old_coreg_maps:
        candidate = in_dir / old_coreg_maps[patient]
        if candidate.exists():
            info["SelectionMethod"] = "old_coreg_log"
            return candidate.resolve(), info

    exams = [p for p in patient_dir.iterdir() if p.is_dir()]
    patterns = (f"*_{anchor_label}.nii.gz", f"*_{anchor_label}.nii")

    # B) Highest quality (default)
    if (reference_selection or "").lower() in ("highest_quality", "best", "quality"):
        candidates: list[Path] = []
        for ex in exams:
            for pat in patterns:
                candidates.extend(list(ex.glob(pat)))

        best: Path | None = None
        best_score = float("-inf")
        best_info: dict | None = None

        for c in candidates:
            s, meta = _estimate_anchor_quality(c)
            if s > best_score:
                best_score = s
                best = c
                best_info = meta

        if best is not None and best_score != float("-inf") and best_info is not None:
            info["SelectionMethod"] = "highest_quality"
            info["QualityScore"] = f"{best_score:.6g}"
            info["Shape"] = best_info.get("shape", "")
            info["ZoomsMM"] = best_info.get("zooms_mm", "")
            info["VoxelVolumeMM3"] = best_info.get("voxel_volume_mm3", "")
            info["Anisotropy"] = best_info.get("anisotropy", "")
            info["SampleStd"] = best_info.get("sample_std", "")
            info["SampleNonzeroFrac"] = best_info.get("sample_nonzero_frac", "")
            if "error" in best_info:
                info["Error"] = best_info.get("error", "")
            return best.resolve(), info
        # else fall through to earliest

    # C) Earliest timepoint with anchor
    exams_sorted = sorted(
        exams,
        key=lambda p: (
            _parse_timepoint_from_exam_name(p.name)[1] is False,
            _parse_timepoint_from_exam_name(p.name)[0],
        ),
    )
    for ex in exams_sorted:
        for pat in patterns:
            cand = list(ex.glob(pat))
            if cand:
                info["SelectionMethod"] = "earliest"
                return cand[0].resolve(), info

    info["SelectionMethod"] = "none"
    info["Error"] = "no_anchor_found"
    return None, info



def _find_exam_dir_for_file(patient_dir: Path, file_path: Path) -> Path | None:
    """
    Given a {in_dir}/{patient} directory and a file path that (likely) lives under it,
    attempt to locate the {exam} directory that contains the file.
    """
    try:
        file_path = file_path.resolve()
    except Exception:
        file_path = Path(file_path)

    for ex in patient_dir.iterdir():
        if not ex.is_dir():
            continue
        try:
            # Fast containment check
            if file_path.is_relative_to(ex.resolve()):  # py>=3.9
                return ex
        except Exception:
            # Fallback: compare parent chain
            try:
                exr = ex.resolve()
                if str(file_path).startswith(str(exr) + os.sep):
                    return ex
            except Exception:
                pass
    return None


def _is_exam_already_processed(out_exam: Path, prefix: str) -> bool:
    """
    Heuristic "done" check for the new preprocessing pipeline:
      - brainmask exists OR at least one transform record exists OR at least one *_brain.nii.gz exists.
    This is intentionally loose (better to skip than to re-run unintentionally).
    Use --overwrite to force reruns.
    """
    if (out_exam / f"{prefix}_brainmask.nii.gz").exists():
        return True
    if list(out_exam.glob(f"{prefix}_*_transform_record.json")):
        return True
    if list(out_exam.glob(f"{prefix}_*_brain.nii.gz")):
        return True
    return False

def _find_existing_patient_brainmask(patient_out_dir: Path, patient: str) -> Path | None:
    """
    Look for an existing per-patient brainmask, first at the canonical location, then by scanning
    all exam outputs for *_brainmask.nii.gz and choosing the newest one.
    """
    canonical = patient_out_dir / f"{patient}_brainmask_ref.nii.gz"
    if canonical.exists():
        return canonical
    else:
        return None

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
    n_workers_per_registration_process: int | None = None,
    n_workers_per_hd_bet_process: int | None = None,
    overwrite: bool = False,
    reuse_patient_brainmask: bool = True,
    reference_selection: str = "highest_quality",
    # Passthrough options to preprocess_single_brain_mri
    modalities: list[str] | None = None,
    anchor_label: str = "T1c",
    registration_metric: str = "mi",
    registration_strategy: str = "medium",
    family_parent_map: dict[str, str] | None = None,
    final_dims: tuple[int, int, int] = (240, 240, 155),
    final_voxels: tuple[float, float, float] = (1.0, 1.0, 1.0),
    save_scans_with_skulls: bool = False,
    use_gpu: bool = False,
    enable_tta: bool = False,
    debug: bool = False,
    quiet: bool = False,
) -> tuple[Path, Path]:
    """
    Walk {in_dir}/{patient}/{exam} and run preprocess_single_brain_mri on each exam into
    {out_dir}/{patient}/{exam}.

    New behavior (no legacy compatibility):
      - No required-label checking here; preprocess_single_brain_mri handles detection/validation.
      - Optional per-patient co-registration reference selection (anchor_label file in earliest timepoint).
      - Optional per-patient brainmask reuse:
          * If a patient-level brainmask exists in the output tree, pass it via brainmask_path.
          * Otherwise, create exactly one brainmask per patient (from the chosen reference exam if possible)
            and reuse it for all other exams.

    Returns (preprocess_log_path, coreg_log_path).
    """
    in_dir = Path(in_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Thread budgeting for nested tools
    # ----------------------------
    # Batch mode runs multiple exam pipelines in parallel (n_workers). Each pipeline will call into
    # registration (SimpleITK) and possibly HD-BET. If each nested tool uses "all threads", CPU
    # oversubscription kills throughput. We therefore allocate a per-pipeline budget.
    #
    # HD-BET on CPU typically doesn't scale well beyond a modest thread count; we cap at 8.
    HD_BET_CPU_THREAD_CAP = 8

    def _norm_pos_int_or_none(x, name):
        if x is None:
            return None
        try:
            x = int(x)
        except Exception:
            raise ValueError(f"{name} must be int or None; got {x!r}")
        if x <= 0:
            raise ValueError(f"{name} must be >= 1 or None; got {x}")
        return x

    n_workers = int(n_workers) if int(n_workers) > 0 else 1
    n_workers_per_registration_process = _norm_pos_int_or_none(n_workers_per_registration_process, "n_workers_per_registration_process")
    n_workers_per_hd_bet_process = _norm_pos_int_or_none(n_workers_per_hd_bet_process, "n_workers_per_hd_bet_process")

    threads_available = os.cpu_count() or 1
    auto_budget = max(1, int(threads_available // max(1, n_workers)))

    if n_workers_per_registration_process is None:
        n_workers_per_registration_process = auto_budget

    if n_workers_per_hd_bet_process is None:
        n_workers_per_hd_bet_process = min(HD_BET_CPU_THREAD_CAP, auto_budget)
    else:
        # Even if user overrides, still enforce the cap in batch mode as requested.
        n_workers_per_hd_bet_process = min(HD_BET_CPU_THREAD_CAP, n_workers_per_hd_bet_process)

    if not quiet:
        print(
            f"[preprocess_brain_mris] CPU threads available={threads_available}; "
            f"pipelines(n_workers)={n_workers}; "
            f"n_workers_per_registration_process={n_workers_per_registration_process}",
            f"n_workers_per_hd_bet_process={n_workers_per_hd_bet_process}",
            flush=True,
        )

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
    patient_ref_info: dict[str, dict] = {}

    for pd in patient_dirs:
        if dont_coregister:
            patient_ref[pd.name] = None
            patient_ref_info[pd.name] = {
                "SelectionMethod": "dont_coregister",
                "QualityScore": "",
                "Shape": "",
                "ZoomsMM": "",
                "VoxelVolumeMM3": "",
                "Anisotropy": "",
                "SampleStd": "",
                "SampleNonzeroFrac": "",
                "Error": "",
            }
        else:
            ref, info = _choose_reference_for_patient(
                patient_dir=pd,
                in_dir=in_dir,
                old_coreg_maps=old_maps,
                anchor_label=anchor_label,
                reference_selection=reference_selection,
            )
            patient_ref[pd.name] = ref
            patient_ref_info[pd.name] = info

    # Write the coreg log up front (what we will try to use)
    with coreg_log.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Patient",
            "ReferenceRelativePath",
            "SelectionMethod",
            "QualityScore",
            "Shape",
            "ZoomsMM",
            "VoxelVolumeMM3",
            "Anisotropy",
            "SampleStd",
            "SampleNonzeroFrac",
            "Error",
        ])
        for pat in sorted(patient_ref.keys()):
            ref = patient_ref.get(pat)
            info = patient_ref_info.get(pat, {})
            rel = ""
            if ref is not None:
                try:
                    rel = str(ref.resolve().relative_to(in_dir))
                except Exception:
                    rel = str(ref)
            w.writerow([
                pat,
                rel,
                info.get("SelectionMethod", ""),
                info.get("QualityScore", ""),
                info.get("Shape", ""),
                info.get("ZoomsMM", ""),
                info.get("VoxelVolumeMM3", ""),
                info.get("Anisotropy", ""),
                info.get("SampleStd", ""),
                info.get("SampleNonzeroFrac", ""),
                info.get("Error", ""),
            ])

    # Prepare the preprocess log (delimiter from extension; line-buffered)
    delim = "\t" if str(preprocess_log).lower().endswith(".tsv") else ","
    log_lock = threading.Lock()
    with preprocess_log.open("w", newline="", encoding="utf-8", buffering=1) as f:
        w = csv.writer(f, delimiter=delim)
        w.writerow([
            "Patient", "Exam", "Status", "Message",
            "InExamDir", "OutExamDir",
            "CoregisterRefUsed",
            "PatientBrainmaskUsed",
            "anchor_label", "modalities",
            "registration_metric", "registration_strategy",
            "final_dims", "final_voxels",
            "save_scans_with_skulls", "use_gpu", "enable_tta", "debug",
        ])
        f.flush()
        os.fsync(f.fileno())

    # Precompute total exams for progress
    all_exam_dirs: list[Path] = []
    for pd in patient_dirs:
        all_exam_dirs.extend([p for p in pd.iterdir() if p.is_dir()])
    total_exams = len(all_exam_dirs)

    done_count_box = [0]

    if tqdm:
        pbar = tqdm(total=total_exams, desc="Preprocessing exams", unit="exam",
                    dynamic_ncols=True, leave=False, file=sys.stdout)
    else:
        pbar = None

    def _advance(n: int = 1):
        if pbar:
            pbar.update(n)
            pbar.refresh()
        else:
            done_count_box[0] += n
            print(f"[Progress] {done_count_box[0]}/{total_exams} exams completed", flush=True)
    def _log_row(row: list[object]):
        with log_lock:
            with preprocess_log.open("a", newline="", encoding="utf-8", buffering=1) as f:
                w = csv.writer(f, delimiter=delim)
                w.writerow([str(x) for x in row])
                f.flush()
                os.fsync(f.fileno())

    def _process_single_exam(
        patient: str,
        exam_dir: Path,
        out_exam: Path,
        coreg_ref: Path | None,
        brainmask_path: Path | None,
        coreg_ref_used_for_exam: Path | None,
    ) -> tuple[str, str, Path | None]:
        """
        Run preprocess_single_brain_mri for one exam. Returns (status, message, exam_brainmask_path_if_any).
        """
        prefix = _exam_prefix_from_dir(exam_dir)

        if (not overwrite) and _is_exam_already_processed(out_exam, prefix):
            return "SKIPPED", "Already processed (outputs present); skipped.", (out_exam / f"{prefix}_brainmask.nii.gz" if (out_exam / f"{prefix}_brainmask.nii.gz").exists() else None)

        out_exam.mkdir(parents=True, exist_ok=True)

        started = time.time()
        try:
            preprocess_single_brain_mri(
                output_dir=str(out_exam),
                scan_dir=str(exam_dir),
                modalities=modalities,
                anchor_label=anchor_label,
                registration_metric=registration_metric,
                registration_strategy=registration_strategy,
                co_register_path=None if dont_coregister else (str(coreg_ref_used_for_exam) if coreg_ref_used_for_exam else None),
                save_scans_with_skulls=save_scans_with_skulls,
                final_dims=final_dims,
                final_voxels=final_voxels,
                debug=debug,
                brainmask_path=str(brainmask_path) if (reuse_patient_brainmask and brainmask_path) else None,
                family_parent_map=family_parent_map,
                use_gpu=use_gpu,
                enable_tta=enable_tta,
                n_workers_per_registration_process=n_workers_per_registration_process,
                n_workers_per_hd_bet_process=n_workers_per_hd_bet_process,
                verbose=not quiet,
            )
            dur = f"{time.time() - started:.1f}s"

            bm = out_exam / f"{prefix}_brainmask.nii.gz"
            return "OK", dur, (bm if bm.exists() else None)
        except Exception as e:
            msg = f"ERROR: {e.__class__.__name__}: {e}"
            return "ERROR", msg, None

    def _process_patient(pd: Path) -> int:
        """
        Process all exams for a patient sequentially (so we can create/reuse a single patient brainmask).
        Returns number of exams processed (including skipped/errored).
        """
        patient = pd.name
        patient_out_dir = out_dir / patient
        patient_out_dir.mkdir(parents=True, exist_ok=True)

        coreg_ref = patient_ref.get(patient)
        ref_exam_dir: Path | None = None
        if coreg_ref is not None:
            ref_exam_dir = _find_exam_dir_for_file(pd, coreg_ref)

        # Determine existing patient brainmask (from prior runs)
        patient_mask: Path | None = None
        if debug:
            print(f"reuse_patient_brainmask = {reuse_patient_brainmask}")
        if reuse_patient_brainmask:
            patient_mask = _find_existing_patient_brainmask(patient_out_dir, patient)
            if debug:
                print(f"[Debug] For {patient_out_dir} reusing brainmask {patient_mask}")
                if patient_mask is not None:
                    import nibabel as nib
                    import os as os
                    mask_file = nib.load(os.fspath(patient_mask))
                    temp_data = mask_file.get_fdata()
                    print(f"[Debug] brainmask shape = {temp_data.shape}")

        # Determine exam processing order:
        #   1) If we need to create a patient mask and we have a reference exam dir, do that first.
        #   2) Otherwise, do timepoint-sorted order (earliest day first).
        exam_dirs = [p for p in pd.iterdir() if p.is_dir()]
        exam_dirs_sorted = sorted(
            exam_dirs,
            key=lambda p: (
                _parse_timepoint_from_exam_name(p.name)[1] is False,
                _parse_timepoint_from_exam_name(p.name)[0],
                p.name,
            ),
        )

        ordered: list[Path] = []
        if reuse_patient_brainmask and (patient_mask is None) and (ref_exam_dir is not None) and (ref_exam_dir in exam_dirs_sorted):
            ordered.append(ref_exam_dir)
            ordered.extend([e for e in exam_dirs_sorted if e != ref_exam_dir])
        else:
            ordered = exam_dirs_sorted

        processed_count = 0

        for ex in ordered:
            _, exam = _parse_patient_exam(ex)
            out_exam = patient_out_dir / exam

            # Decide co-registration reference to use for this exam:
            # - If dont_coregister: None
            # - If this is the reference exam itself: skip coregistration (None), since it is already in that space.
            # - Otherwise: use the patient-level reference file.
            coreg_ref_used = None
            if not dont_coregister:
                if (coreg_ref is not None) and (ref_exam_dir is not None) and (ex == ref_exam_dir):
                    coreg_ref_used = None
                else:
                    coreg_ref_used = coreg_ref

            status, msg, exam_bm = _process_single_exam(
                patient=patient,
                exam_dir=ex,
                out_exam=out_exam,
                coreg_ref=coreg_ref,
                brainmask_path=patient_mask,
                coreg_ref_used_for_exam=coreg_ref_used,
            )

            _log_row([
                patient, exam, status, msg,
                str(ex), str(out_exam),
                str(coreg_ref or "NONE") if not dont_coregister else "NONE",
                str(patient_mask or "NONE") if reuse_patient_brainmask else "DISABLED",
                anchor_label, ",".join(modalities) if modalities else "AUTO",
                registration_metric, registration_strategy,
                final_dims, final_voxels,
                save_scans_with_skulls, use_gpu, enable_tta, debug,
            ])

            processed_count += 1
            _advance(1)

        return processed_count

    # Execute: parallelize across patients (safe for patient-level brainmask creation/reuse)
    if n_workers and n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_process_patient, pd): pd for pd in patient_dirs}
            for fut in as_completed(futs):
                # Ensure exceptions don't kill the whole run without being logged
                pd = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    # Log a patient-level failure (one row per exam dir to keep progress consistent)
                    patient = pd.name
                    exam_dirs = [p for p in pd.iterdir() if p.is_dir()]
                    for ex in exam_dirs:
                        _, exam = _parse_patient_exam(ex)
                        out_exam = out_dir / patient / exam
                        _log_row([
                            patient, exam, "ERROR",
                            f"Patient worker failed: {e.__class__.__name__}: {e}",
                            str(ex), str(out_exam),
                            str(patient_ref.get(patient) or "NONE"),
                            "UNKNOWN",
                            anchor_label, ",".join(modalities) if modalities else "AUTO",
                            registration_metric, registration_strategy,
                            final_dims, final_voxels,
                            save_scans_with_skulls, use_gpu, enable_tta, debug,
                        ])
                        _advance(1)
    else:
        for pd in patient_dirs:
            try:
                _process_patient(pd)
            except Exception as e:
                patient = pd.name
                exam_dirs = [p for p in pd.iterdir() if p.is_dir()]
                for ex in exam_dirs:
                    _, exam = _parse_patient_exam(ex)
                    out_exam = out_dir / patient / exam
                    _log_row([
                        patient, exam, "ERROR",
                        f"Patient worker failed: {e.__class__.__name__}: {e}",
                        str(ex), str(out_exam),
                        str(patient_ref.get(patient) or "NONE"),
                        "UNKNOWN",
                        anchor_label, ",".join(modalities) if modalities else "AUTO",
                        registration_metric, registration_strategy,
                        final_dims, final_voxels,
                        save_scans_with_skulls, use_gpu, enable_tta, debug,
                    ])
                    _advance(1)

    if pbar:
        pbar.close()

    return Path(preprocess_log), Path(coreg_log)


# ----------------------------
# CLI
# ----------------------------

def _parse_family_parent_map(val: str | None) -> dict[str, str] | None:
    if not val:
        return None
    try:
        if os.path.isfile(val):
            with open(val, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        else:
            obj = json.loads(val)
        if not isinstance(obj, dict):
            raise ValueError("family_parent_map must be a JSON object/dict.")
        return {str(k): str(v) for k, v in obj.items()}
    except Exception as e:
        raise SystemExit(f"Failed to parse --family_parent_map: {e}")


def main():
    import __main__
    module = getattr(__main__, "__spec__", None)
    prog = f"python -m {module.name}" if module and module.name else None
    p = argparse.ArgumentParser(
        prog=prog,
        description="Batch brain MRI preprocessing over a NIFTI library (patient/exam tree)."
    )
    p.add_argument("--in_dir", required=True, help="Path to NIFTI library root (converted by convert_dicom_plan).")
    p.add_argument("--out_dir", required=True, help="Path to write preprocessed outputs (patient/exam structure).")
    p.add_argument("--preprocess_log", default=None, help="CSV/TSV path for per-exam outcomes. Auto-generated if omitted.")
    p.add_argument("--coreg_log", default=None, help="CSV path for chosen per-patient reference. Auto-generated if omitted.")
    p.add_argument("--old_coreg_log", nargs="*", default=None, help="One or more prior coreg logs to reuse references.")
    p.add_argument("--dont_coregister", action="store_true", help="Disable co-registration; run each exam independently.")
    p.add_argument("--n_workers", type=int, default=1, help="Parallel workers across patients (threaded). Use 1 to run serially.")
    p.add_argument(
    "--reference_selection",
    default="highest_quality",
    choices=["highest_quality", "earliest"],
    help=(
        "How to choose the per-patient co-registration reference anchor. "
        "'highest_quality' scores all candidate anchor volumes and picks the best; "
        "'earliest' uses the smallest timepoint day (legacy behavior). "
        "(default: highest_quality)"
    ),
    )
    p.add_argument(
        "--n_workers_per_registration_process",
        type=int,
        default=None,
        help="Optional override: CPU threads to allow per SimpleITK/ITK process within each pipeline. "
             "Default: floor(cpu_threads / n_workers).",
    )
    p.add_argument(
        "--n_workers_per_hd_bet_process",
        type=int,
        default=None,
       help="Optional override: CPU threads to allow for HD-BET when device=cpu, per pipeline. "
             "Default: min(8, floor(cpu_threads / n_workers)). (Capped at 8 in batch mode.)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs instead of skipping processed exams.")
    p.add_argument("--no_reuse_patient_brainmask", action="store_true", help="Disable per-patient brainmask reuse; compute per-exam.")

    # Passthroughs to preprocess_single_brain_mri
    p.add_argument(
        "--modalities",
        default=None,
        help=(
            "Comma-separated list of modality labels to auto-detect per exam (non-recursive). "
            "If omitted, preprocess_single_brain_mri auto-detects all supported scans in each exam."
        ),
    )
    p.add_argument("--anchor_label", default="T1c", help="Anchor label for registration/skull stripping (default: T1c).")
    p.add_argument("--registration_metric", default="mi", help="Registration similarity metric (default: mi).")
    p.add_argument("--registration_strategy", default="medium", help="Registration preset: accurate|medium|fast (default: medium).")
    p.add_argument(
        "--family_parent_map",
        default=None,
        help=(
            "JSON string or path to JSON file mapping {family -> parent_label}. "
            "Example: '{\"DWI\":\"DWI\", \"SWI\":\"SWI\"}'."
        ),
    )
    p.add_argument("--final_dims", type=int, nargs=3, default=(240, 240, 155), metavar=("NX", "NY", "NZ"),
                   help="Final dimensions (default: 240 240 155).")
    p.add_argument("--final_voxels", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("SX", "SY", "SZ"),
                   help="Final voxel sizes in mm (default: 1.0 1.0 1.0).")
    p.add_argument("--save_scans_with_skulls", action="store_true", help="Also save skull-on registered scans (PHI risk).")
    p.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration for hd-bet skull stripping.")
    p.add_argument("--enable_tta", action="store_true", help="Enable test-time augmentation (TTA) for hd-bet skull stripping.")
    p.add_argument("--skip_qc", action="store_true", help="Skip creation of PDF QC files showing the center axial slice from each preprocessed volume.")
    p.add_argument("--debug", action="store_true", help="Keep temp dirs from underlying preprocessing.")
    p.add_argument("--verbose", action="store_true", help="Print logging from underlying preprocessing.")

    args = p.parse_args()

    modalities = None
    if args.modalities:
        modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]

    family_parent_map = _parse_family_parent_map(args.family_parent_map)

    preprocess_library(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        preprocess_log=args.preprocess_log,
        coreg_log=args.coreg_log,
        old_coreg_log=args.old_coreg_log,
        dont_coregister=args.dont_coregister,
        n_workers=args.n_workers,
        n_workers_per_registration_process=args.n_workers_per_registration_process,
        n_workers_per_hd_bet_process=args.n_workers_per_hd_bet_process,
        overwrite=args.overwrite,
        reuse_patient_brainmask=not args.no_reuse_patient_brainmask,
        reference_selection=args.reference_selection,
        modalities=modalities,
        anchor_label=args.anchor_label,
        registration_metric=args.registration_metric,
        registration_strategy=args.registration_strategy,
        family_parent_map=family_parent_map,
        final_dims=tuple(args.final_dims),
        final_voxels=tuple(args.final_voxels),
        save_scans_with_skulls=args.save_scans_with_skulls,
        use_gpu=args.use_gpu,
        enable_tta=args.enable_tta,
        debug=args.debug,
        quiet=not args.verbose,
    )

    if not args.skip_qc:
        from .preprocess import generate_preprocessing_qc_pdfs

        if args.verbose:
            print(f"[QC] Preparing PDF QC files to visualize preprocessed MRI volumes...")

        generate_preprocessing_qc_pdfs(
            root_dir=args.out_dir,
            n_workers=args.n_workers,
            out_dir=args.out_dir + "/QC/"
        )


if __name__ == "__main__":
    main()
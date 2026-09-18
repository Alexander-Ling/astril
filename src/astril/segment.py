#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module: segment
Generic, contract-driven segmentation engine. Reads a model's `pipeline.json` (see
`pipeline_contract.py`) -- an ordered list of inference stages, each naming its own per-plane
models, which named inputs it consumes (from the model's declared `initial_inputs`, or a prior
stage's own derived output), how its per-plane predictions are merged, and what it produces --
and runs it using the exact same primitives GBM_seg_v2's hand-written cascade already used:
`create_segmentation_config`, `load_models_for_config`, and `process_subject_with_models`'s four
merge functions. No new segmentation math is introduced; per-stage settings that used to be
Python constants (channel lists, merge methods, gate threshold/dilation) are now contract data.

A model can be selected two ways, mutually exclusive:
  - `model_family`: a name registered in astril's own `models.json`/OSF download system, resolved
    via `locate_models_dir()/<family>/` (the same directory `astril-download-models` populates).
  - `model_dir`: a direct path to any directory containing its own `pipeline.json` + checkpoints,
    for a model that hasn't been (or never will be) integrated into astril's model manifest --
    e.g. one a user trained and wants to run immediately without publishing it anywhere first.
Either way, the model's own `pipeline.json` is the single source of truth for its identity
(`model_family` in the contract) and labels, used for the `segmentation_provenance.json` sidecar
regardless of which lookup path found the model.

Scope/assumption this engine currently makes: stage 0 must depend only on `initial_inputs` (true
by construction -- there's no earlier stage for it to depend on), and is the only stage built
once over the whole input directory; every later stage is reconfigured per-subject, since in
practice a later stage typically needs a prior stage's per-subject derived output before it can
be configured at all. A stage that could technically run batch-wide but comes after stage 0 is
still (correctly, just less efficiently) reconfigured per-subject under this engine.

This is an ADDITIVE, not-yet-default engine: `segment_GBM.py`'s hand-written GBM_seg_v2 cascade
(the `segment-gbm` console script) is unchanged and is still what the Nextflow pipeline calls.
Compare this engine's output against `segment-gbm`'s on the same input before switching over.
"""
import argparse
import json
import os
from pathlib import Path

from .pipeline_contract import PIPELINE_FILENAME


def _resolve_model_root(model_family: str | None, model_dir) -> Path:
    """
    Resolve to the one directory holding a model's pipeline.json + checkpoints, from either a
    registered family name (looked up via locate_models_dir()) or a direct path.
    """
    if bool(model_family) == bool(model_dir):
        raise ValueError("Specify exactly one of model_family or model_dir")
    if model_dir:
        root = Path(model_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"model_dir does not exist or is not a directory: {root}")
        return root
    from .models_download import locate_models_dir
    return Path(locate_models_dir()) / model_family


def _load_pipeline_contract(model_root: Path) -> dict:
    contract_path = model_root / PIPELINE_FILENAME
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"No {PIPELINE_FILENAME} found at {contract_path}. Every model directory -- whether "
            "resolved via --model_family or --model_dir -- must ship its own pipeline.json "
            "describing its stages, inputs, and labels."
        )
    with contract_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_inputs_summary(initial_inputs: dict) -> str:
    """Human-readable listing of every declared input: name, kind, required/optional, pattern --
    printed upfront so a user pointing this at their own files can see exactly what's expected
    without having to go read pipeline.json themselves."""
    lines = []
    name_width = max((len(n) for n in initial_inputs), default=0)
    for name, spec in initial_inputs.items():
        requirement = "required" if spec.get("required") else "optional"
        lines.append(f"  {name.ljust(name_width)}  ({spec.get('kind', 'channel')}, {requirement})  pattern: '{spec['pattern']}'")
    return "\n".join(lines)


def _resolve_ref(ref: str, initial_inputs: dict, stage_patterns: dict) -> tuple[str, str]:
    """
    Resolve an input/mask/intersect_with reference -- a bare `initial_inputs` name, or a
    "stage_name.output_name" dotted reference to a prior stage's own derived output -- to
    (logical_name, filename_pattern). Patterns, not per-subject file paths: a later stage's
    config is rebuilt from these patterns against that one subject's own directory.
    """
    if "." in ref:
        stage_name, output_name = ref.split(".", 1)
        key = (stage_name, output_name)
        if key not in stage_patterns:
            raise ValueError(
                f"Reference '{ref}' does not match any prior stage's derived_outputs "
                f"(available: {sorted('.'.join(k) for k in stage_patterns)})"
            )
        return output_name, stage_patterns[key]
    if ref not in initial_inputs:
        raise ValueError(f"Reference '{ref}' is not a declared initial_input or prior stage output")
    return ref, initial_inputs[ref]["pattern"]


def _resolve_stage_models(model_root: Path, stage: dict):
    """For one stage's `models` list, return parallel lists: model_paths (.pt), train_cfg_paths, planes."""
    model_paths, train_cfgs, planes = [], [], []
    for m in stage["models"]:
        pt_file = model_root / f"{m['dir']}.pt"
        nested_pt_file = model_root / m["dir"] / f"{m['dir']}.pt"
        model_paths.append(str(pt_file if pt_file.is_file() else nested_pt_file))
        train_cfgs.append(str(model_root / m["cfg"]))
        planes.append(m["plane"])
    return model_paths, train_cfgs, planes


def _tiebreaker_index(stage: dict) -> int:
    """`merge.tiebreaker` may be a plane name (e.g. "Axial") or a literal model index."""
    tb = stage["merge"].get("tiebreaker", 0)
    if isinstance(tb, int):
        return tb
    planes = [m["plane"] for m in stage["models"]]
    return planes.index(tb)


def _find_matching_file(directory, pattern: str) -> Path:
    matches = [p for p in Path(directory).iterdir() if p.is_file() and pattern in p.name]
    if not matches:
        raise FileNotFoundError(f"No file matching pattern '{pattern}' found in {directory}")
    if len(matches) > 1:
        raise ValueError(f"Multiple files matching pattern '{pattern}' found in {directory}: {matches}")
    return matches[0]


def _derived_output_path(base_file: Path, base_pattern: str, derived_pattern: str) -> Path:
    """Name a derived-output file after the pattern it replaces in its reference file's name,
    same convention `compute_final_segmentation_path` uses for the pipeline's final output."""
    base_name = base_file.name
    if base_pattern in base_name:
        derived_name = base_name.replace(base_pattern, derived_pattern)
    else:
        derived_name = base_name.replace(".nii.gz", derived_pattern)
    return base_file.parent / derived_name


def _compute_final_output_path(primary_file: Path, candidate_patterns: list[str], segment_suffix: str) -> Path:
    """Generalized `compute_final_segmentation_path`: try each pattern this pipeline could have
    named its primary per-subject file after (the first stage's mask, any derived output), in
    order, and replace the first one found; fall back to appending before the extension."""
    base_name = primary_file.name
    for pattern in candidate_patterns:
        if pattern in base_name:
            return primary_file.parent / base_name.replace(pattern, segment_suffix)
    return primary_file.parent / base_name.replace(".nii.gz", segment_suffix)


def _apply_threshold_gate(probabilities, reference_mask_path, output_path,
                          background_index=0, threshold=0.30, dilation_voxels=5):
    """Threshold a merged probability volume's foreground mass, dilate, and intersect with a
    reference mask -- the same math GBM_seg_v2's Model-A-gate derivation already used, lifted
    into a reusable, contract-parameterized primitive (`op: "threshold_gate"`)."""
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import binary_dilation

    ref_image = nib.load(str(reference_mask_path))
    ref_mask = ref_image.get_fdata() > 0.5
    if probabilities is None or probabilities.ndim != 4:
        raise ValueError("threshold_gate requires a 4-D probability array")
    if probabilities.shape[:3] != ref_mask.shape:
        raise ValueError(
            f"threshold_gate: probability/mask shape mismatch: {probabilities.shape[:3]} vs {ref_mask.shape}"
        )
    foreground_probability = 1.0 - probabilities[..., background_index]
    gate = foreground_probability >= float(threshold)
    if dilation_voxels > 0:
        gate = binary_dilation(gate, structure=np.ones((3, 3, 3), dtype=bool), iterations=dilation_voxels)
    gate &= ref_mask
    nib.save(nib.Nifti1Image(gate.astype(np.uint8), ref_image.affine, ref_image.header), str(output_path))
    return output_path


_DERIVED_OUTPUT_OPS = {
    "threshold_gate": _apply_threshold_gate,
}


def _write_segmentation_provenance(subject_dir, model_family_label: str, labels: dict) -> None:
    """
    Same sidecar contract as segment_GBM.py's -- see that module's docstring for consumers.
    `model_family_label`/`labels` come straight from the contract that was actually loaded (its
    own declared `model_family` and `labels`), not from a fresh family-name lookup -- so this
    works identically whether the model was found via --model_family or --model_dir.
    """
    payload = {
        "model_family": model_family_label,
        "labels": {str(level): name for level, name in labels.items()} if labels else {},
    }
    out_path = Path(subject_dir) / "segmentation_provenance.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not write segmentation provenance to {out_path}: {e}")


def _cleanup_intermediate_files(subject_dir, intermediate_patterns: list[str], cfg_extensions=("parameters.cfg",)) -> None:
    for file_path in Path(subject_dir).rglob("*"):
        if not file_path.is_file():
            continue
        name = file_path.name
        if any(pat in name for pat in intermediate_patterns) or name.endswith(cfg_extensions):
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Warning: could not remove {file_path}: {e}")


def run_segmentation(input_dir, model_family=None, model_dir=None, slice_batch_size=1,
                     overwrite_existing_outputs=False, input_overrides=None,
                     segment_suffix="_seg.nii.gz", debug_models=False):
    """
    Run a pipeline.json-described segmentation pipeline over every subject discovered under
    `input_dir` (via stage 0's mask pattern). Exactly one of `model_family` (a name registered in
    astril's models.json) or `model_dir` (a direct path to a model directory -- e.g. a
    self-trained model not published anywhere) must be given.

    input_overrides: optional {initial_input_name: pattern} to replace any of the contract's
    default file patterns (e.g. {"brainmask": "_my_brainmask.nii.gz"}), the generic equivalent
    of segment_GBM.py's `channel_patterns`/`brainmask_pattern` arguments.
    """
    from .create_segmentation_config import create_segmentation_config, parse_train_config_for_model_parameters
    from .data_loading import read_paths_from_file
    from .run_segmentation import load_models_for_config

    model_root = _resolve_model_root(model_family, model_dir)
    contract = _load_pipeline_contract(model_root)
    model_family_label = contract.get("model_family") or model_family or model_root.name
    contract_labels = {
        int(level): str(name) for level, name in contract.get("labels", {}).items() if int(level) != 0
    }

    initial_inputs = {
        name: {**spec, "pattern": (input_overrides or {}).get(name, spec["pattern"])}
        for name, spec in contract["initial_inputs"].items()
    }
    stages = contract["stages"]
    if not stages:
        raise ValueError(f"Pipeline contract at {model_root} declares no stages")

    print(f"[INFO] Model '{model_family_label}' expects the following inputs under {input_dir}:")
    print(_format_inputs_summary(initial_inputs))

    working_dir = os.path.join(input_dir, "Segmentation_Configs")
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    # Every pattern this pipeline could plausibly have named a subject's primary file after,
    # for resolving the final output filename regardless of which stage's file we're looking at.
    candidate_patterns = [spec["pattern"] for spec in initial_inputs.values()]
    for stage in stages:
        for derived in stage.get("derived_outputs", []):
            candidate_patterns.append(derived["pattern"])

    stage_patterns: dict[tuple[str, str], str] = {}
    stage_models_cache: dict[str, list] = {}
    stage_model_paths: dict[str, list[str]] = {}
    stage_train_cfgs: dict[str, list[str]] = {}

    # ---- Stage 0: build one full-batch config over the whole input_dir; discover subjects ----
    stage0 = stages[0]
    stage0_channel_names = list(stage0["inputs"])
    stage0_channel_patterns = [initial_inputs[name]["pattern"] for name in stage0_channel_names]
    _, stage0_mask_pattern = _resolve_ref(stage0["mask"], initial_inputs, stage_patterns)
    stage0_model_paths, stage0_train_cfgs, _ = _resolve_stage_models(model_root, stage0)
    stage0_out_suffix = segment_suffix if stage0.get("is_final") else stage0["label_output_suffix"]

    stage0_config = create_segmentation_config(
        workingDirectory=working_dir, inputChannels=stage0_channel_names,
        channelPatterns=stage0_channel_patterns, maskPattern=stage0_mask_pattern,
        model_paths=stage0_model_paths, modelTrainConfigFiles=stage0_train_cfgs,
        merging_method=stage0["merge"]["method"], inputVolumeDirectory=input_dir,
        outputVolumeDirectory="in_place", segmentSuffix=stage0_out_suffix,
        output_config_filename=f"{stage0['name']}_parameters.cfg", silent=False,
    )
    import configparser
    cp = configparser.ConfigParser()
    cp.read(stage0_config)
    cfg0 = cp["DEFAULT"]
    stage0_num_in = list(map(int, cfg0["model_train_num_input_slices"].split(",")))
    stage0_min_hw = list(map(int, cfg0["model_train_minimum_hw"].split(",")))
    stage_models_cache[stage0["name"]] = load_models_for_config(
        model_paths=stage0_model_paths, model_train_config_files=stage0_train_cfgs,
        model_num_input_slices=stage0_num_in, model_min_hw=stage0_min_hw,
        num_modal_channels=len(stage0_channel_names),
    )
    stage_model_paths[stage0["name"]] = stage0_model_paths
    stage_train_cfgs[stage0["name"]] = stage0_train_cfgs

    mask_paths = read_paths_from_file(cfg0["mask_paths_file"])
    if not mask_paths:
        stage0_inputs_summary = _format_inputs_summary(
            {name: initial_inputs[name] for name in stage0_channel_names}
            | {stage0["mask"]: initial_inputs[stage0["mask"]]}
        )
        raise RuntimeError(
            f"No exams found under {input_dir}: no subdirectory (or the directory itself) has "
            f"every required input for stage '{stage0['name']}' matching its expected pattern.\n"
            f"{stage0_inputs_summary}\n"
            "A directory only counts as an exam if its required inputs above are all present; "
            "missing an optional one is fine. If your files use different naming, pass "
            "--input_override NAME=PATTERN (Python API: input_overrides={...}) to override any "
            "of the patterns above."
        )
    print(f"[INFO] Found {len(mask_paths)} exam(s) in {input_dir}.")

    # ---- Load models for every later stage up front (dims read directly from each checkpoint's
    # own train_parameters.cfg -- these stages' inputs don't exist as files until per-subject
    # processing reaches them, so a full config can't be built for them yet). ----
    for stage in stages[1:]:
        model_paths, train_cfgs, _ = _resolve_stage_models(model_root, stage)
        dims = [parse_train_config_for_model_parameters(c) for c in train_cfgs]
        stage_models_cache[stage["name"]] = load_models_for_config(
            model_paths=model_paths, model_train_config_files=train_cfgs,
            model_num_input_slices=[d["num_input_slices"] for d in dims],
            model_min_hw=[d["minimum_height_width"] for d in dims],
            num_modal_channels=len(stage["inputs"]),
        )
        stage_model_paths[stage["name"]] = model_paths
        stage_train_cfgs[stage["name"]] = train_cfgs

    intermediate_patterns = [s["label_output_suffix"] for s in stages if not s.get("is_final") and s.get("label_output_suffix")]
    intermediate_patterns += [d["pattern"] for s in stages for d in s.get("derived_outputs", [])]

    # ---- Per-subject: run every stage in order, deriving each stage's declared outputs before
    # moving to the next. ----
    for subject_index, primary_path in enumerate(mask_paths):
        primary_path = Path(primary_path)
        subject_dir = primary_path.parent
        final_output_path = _compute_final_output_path(primary_path, candidate_patterns, segment_suffix)
        if final_output_path.exists() and not overwrite_existing_outputs:
            print(f"[INFO] Skipping exam {subject_index + 1}: final segmentation file {final_output_path} already exists.")
            continue

        print(f"\n==============================\n[INFO] Exam {subject_index + 1} of {len(mask_paths)}: {subject_dir}")
        stage_patterns.clear()

        for stage_idx, stage in enumerate(stages):
            is_final = bool(stage.get("is_final"))
            out_suffix = segment_suffix if is_final else stage["label_output_suffix"]
            needs_probs = bool(stage.get("derived_outputs"))
            tiebreaker = _tiebreaker_index(stage)
            loaded_models = stage_models_cache[stage["name"]]

            if stage_idx == 0:
                config_path = stage0_config
                idx_in_config = subject_index
                overwrite_for_stage = True  # only the skip-check above gates a whole subject
            else:
                channel_names, channel_patterns = [], []
                for inp in stage["inputs"]:
                    name, pattern = _resolve_ref(inp, initial_inputs, stage_patterns)
                    channel_names.append(name)
                    channel_patterns.append(pattern)
                _, mask_pattern = _resolve_ref(stage["mask"], initial_inputs, stage_patterns)
                stage_working_dir = os.path.join(working_dir, stage["name"], f"Exam_{subject_index + 1}")
                config_path = create_segmentation_config(
                    workingDirectory=stage_working_dir, inputChannels=channel_names,
                    channelPatterns=channel_patterns, maskPattern=mask_pattern,
                    model_paths=stage_model_paths[stage["name"]], modelTrainConfigFiles=stage_train_cfgs[stage["name"]],
                    merging_method=stage["merge"]["method"], inputVolumeDirectory=str(subject_dir),
                    outputVolumeDirectory="in_place", segmentSuffix=out_suffix,
                    output_config_filename=f"{stage['name']}_parameters.cfg", silent=True,
                )
                idx_in_config = 0
                overwrite_for_stage = overwrite_existing_outputs if is_final else True

            print(f"[INFO] Exam {subject_index + 1}: running stage '{stage['name']}'...")
            result = process_subject_with_models(
                config_path, idx_in_config, loaded_models, slice_batch_size,
                overwrite_for_stage, out_suffix, tiebreaker_model=tiebreaker,
                debug_models=debug_models, return_merged=needs_probs,
            )

            for derived in stage.get("derived_outputs", []):
                op_fn = _DERIVED_OUTPUT_OPS.get(derived["op"])
                if op_fn is None:
                    raise ValueError(f"Unknown derived_outputs op '{derived['op']}' in stage '{stage['name']}'")
                merged_label, merged_probabilities, _affine = result
                source_array = merged_probabilities if derived.get("source") == "probabilities" else merged_label
                _, ref_pattern = _resolve_ref(derived["intersect_with"], initial_inputs, stage_patterns)
                ref_path = _find_matching_file(subject_dir, ref_pattern)
                out_path = _derived_output_path(ref_path, ref_pattern, derived["pattern"])
                op_fn(
                    source_array, ref_path, out_path,
                    background_index=derived.get("background_index", 0),
                    threshold=derived.get("threshold", 0.5),
                    dilation_voxels=derived.get("dilation_voxels", 0),
                )
                stage_patterns[(stage["name"], derived["name"])] = derived["pattern"]
                print(f"[INFO] Exam {subject_index + 1}: derived '{stage['name']}.{derived['name']}' -> {out_path}")

        for op in contract.get("final_postprocessing", []):
            raise NotImplementedError(f"final_postprocessing op '{op.get('op')}' is not implemented yet")

        _write_segmentation_provenance(subject_dir, model_family_label, contract_labels)
        _cleanup_intermediate_files(subject_dir, intermediate_patterns)

    print("[INFO] Segmentation pipeline complete.")


# Deferred import placed near point of use to keep CLI --help instant, same convention as the
# rest of astril's segmentation modules.
def process_subject_with_models(*args, **kwargs):
    from .segment_GBM import process_subject_with_models as _impl
    return _impl(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Run a contract-driven (pipeline.json) segmentation pipeline for a given model."
    )
    parser.add_argument("input_directory", help="Directory containing input scans for segmentation.")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_family", help="Model family name registered in astril's models.json, e.g. GBM_seg_v2.")
    model_group.add_argument("--model_dir", help="Path to a model directory (its own pipeline.json + checkpoints), "
                                                 "for a model not published through astril's model manifest.")
    parser.add_argument("--segment_suffix", type=str, default="_seg.nii.gz",
                        help="Suffix for the final segmentation output file name.")
    parser.add_argument("--slice_batch_size", type=int, default=1)
    parser.add_argument("--overwrite_existing_outputs", action="store_true")
    parser.add_argument("--debug_models", action="store_true")
    parser.add_argument("--input_override", action="append", default=[], metavar="NAME=PATTERN",
                        help="Override an initial_input's default file pattern, e.g. brainmask=_my_mask.nii.gz. Repeatable.")
    args = parser.parse_args()

    input_overrides = {}
    for item in args.input_override:
        if "=" not in item:
            parser.error(f"--input_override must be NAME=PATTERN, got: {item}")
        name, pattern = item.split("=", 1)
        input_overrides[name] = pattern

    run_segmentation(
        input_dir=args.input_directory,
        model_family=args.model_family,
        model_dir=args.model_dir,
        slice_batch_size=args.slice_batch_size,
        overwrite_existing_outputs=args.overwrite_existing_outputs,
        input_overrides=input_overrides or None,
        segment_suffix=args.segment_suffix,
        debug_models=args.debug_models,
    )


if __name__ == "__main__":
    main()

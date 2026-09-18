"""
Module: model_labels
Convenience accessor for just the level->label portion of a model family's `pipeline.json`
contract (see `pipeline_contract.py`) -- what each non-zero integer in that model's output means.

A model family's segmentation entry point (`segment.py`, driven by that family's `pipeline.json`)
writes this information into a `segmentation_provenance.json` sidecar next to its output;
`quantify_volumes.py` reads that sidecar back to label volume columns meaningfully instead of
with generic "SegmentN" names.
"""
from .pipeline_contract import available_pipeline_families, try_load_pipeline_contract

available_label_families = available_pipeline_families


def load_model_labels(family: str) -> dict[int, str] | None:
    """
    Return {level: label} (excluding background/level 0) for `family`, or None if its downloaded
    archive has no pipeline.json, or the contract has no `labels` section. Not an error --
    quantify_volumes.py falls back to generic "SegmentN" column names when no labels are available.
    """
    contract = try_load_pipeline_contract(family)
    if not contract:
        return None
    try:
        return {
            int(level): str(name)
            for level, name in contract.get("labels", {}).items()
            if int(level) != 0
        }
    except Exception as e:
        print(f"[model_labels] WARNING: could not parse labels for '{family}': {e}")
        return None

"""
Module: pipeline_contract
Loads a segmentation model family's full pipeline contract from `models/<family>/pipeline.json`
-- shipped as part of that model's own download archive (alongside its weights, README,
CITATIONS), not as astril package data. Each model family is meant to be a complete, standalone
download that integrates into astril on its own; `pipeline.json` is just one more file inside
`models/<family>/`, listed in that family's `models.json` `expect` entry like its README/CITATIONS
already are, so a normal `astril-download-models` run verifies it's present.

Resolved via `locate_models_dir()` (the same download-cache-aware directory `segment.py`'s
`_family_root()` and `segment_GBM.py`'s `_resolve_gbm_family_root()` use) rather than a path
relative to this package's own install location, since a non-editable install can cache
downloaded models in a per-user directory outside the package entirely.

The contract describes a family's segmentation level->label meanings (`labels`), its initial
input channels/mask (`initial_inputs`), and its ordered multi-stage inference pipeline (`stages`,
`final_postprocessing`) -- see `segment.py` for the generic engine that executes it, and
`model_labels.py` for the narrower labels-only accessor `quantify_volumes.py` uses.
"""
import json
from pathlib import Path

PIPELINE_FILENAME = "pipeline.json"


def _models_dir() -> Path:
    # Lazy import so importing this module (e.g. from quantify_volumes.py's hot path) stays cheap.
    from .models_download import locate_models_dir
    return Path(locate_models_dir())


def available_pipeline_families() -> list[str]:
    """Model families whose downloaded archive included a pipeline.json."""
    models_dir = _models_dir()
    if not models_dir.is_dir():
        return []
    return sorted(p.parent.name for p in models_dir.glob(f"*/{PIPELINE_FILENAME}") if p.is_file())


def load_pipeline_contract(family: str) -> dict:
    """
    Return the parsed pipeline.json contract for `family`.

    Raises FileNotFoundError if that family's downloaded archive has no pipeline.json (not yet
    added to its OSF archive, its archive hasn't been downloaded yet, or the family name is wrong).
    """
    path = _models_dir() / family / PIPELINE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No pipeline contract found for model family '{family}' (expected {path}). "
            f"Run `astril-download-models --family {family}` if its archive hasn't been fetched "
            "yet, or confirm this family actually ships a pipeline.json."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def try_load_pipeline_contract(family: str) -> dict | None:
    """Like load_pipeline_contract, but returns None instead of raising when missing/unreadable."""
    try:
        return load_pipeline_contract(family)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[pipeline_contract] WARNING: could not read pipeline.json for '{family}': {e}")
        return None

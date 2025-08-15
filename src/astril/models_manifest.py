# src/astril/models_manifest.py
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Union

# -------- helpers --------

def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _iso8601_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)

def _default_models_dir_from_this_module() -> Path:
    # This file lives in src/astril/, so models/ is a sibling
    return Path(__file__).resolve().parent / "models"

def _default_url_map_path() -> Path:
    # Default URL map: src/astril/models/osf_urls.json (installed path: astril/models/osf_urls.json)
    return _default_models_dir_from_this_module() / "osf_urls.json"

def _load_url_map(url_mapping: Optional[Union[Dict[str, str], str, Path]]) -> Dict[str, str]:
    """
    Load a filename->URL mapping.
    Behavior:
      - If url_mapping is a dict, use it.
      - If it's a path (str/Path), load JSON from that path.
      - If None, try the default osf_urls.json path; if it exists, load it; otherwise return {}.
    """
    if isinstance(url_mapping, dict):
        return dict(url_mapping)
    if isinstance(url_mapping, (str, Path)) and url_mapping:
        p = Path(url_mapping)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        # If explicitly given but missing, raise to surface the mistake.
        raise FileNotFoundError(f"URL map not found: {p}")

    # url_mapping is None -> auto-detect default osf_urls.json
    default_map = _default_url_map_path()
    if default_map.exists():
        return json.loads(default_map.read_text(encoding="utf-8"))
    return {}

# -------- core --------

def generate_models_json(
    models_dir: Union[str, Path, None] = None,
    output_path: Union[str, Path, None] = None,
    url_mapping: Optional[Union[Dict[str, str], str, Path]] = None,
    include: Optional[Sequence[str]] = ("*.h5", "*.cfg"),
    exclude: Optional[Sequence[str]] = (".git*", "*.gitignore", "*.gitkeep", "models.json", "osf_urls.json", "*.txt"),
    overwrite: bool = True,
    pretty: bool = True,
) -> Path:
    """
    Create models.json with entries like:
      {
        "Axial_1.h5": {"url": "...", "sha256": "...", "bytes": 160392920, "version": "2025-04-01T20:45:00Z"},
        ...
      }

    Defaults:
      - models_dir: astril/models (next to this module)
      - url_mapping: auto-loads astril/models/osf_urls.json if present; otherwise no URLs
    """
    # Resolve models dir
    if models_dir is None:
        models_dir = _default_models_dir_from_this_module()
    models_dir = Path(models_dir).resolve()
    if not models_dir.is_dir():
        raise NotADirectoryError(f"Models directory not found: {models_dir}")

    # Load URL mapping with default auto-detection
    url_map: Dict[str, str] = _load_url_map(url_mapping)

    include = include or ("*",)
    exclude = exclude or tuple()

    # Collect files
    files: List[Path] = []
    for p in sorted(models_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        if _matches_any(name, exclude):
            continue
        if include and not _matches_any(name, include):
            continue
        files.append(p)

    if not files:
        raise FileNotFoundError(
            f"No files found in {models_dir} matching include={include} with exclude={exclude}"
        )

    # Build manifest
    manifest: Dict[str, Dict[str, object]] = {}
    for p in files:
        st = p.stat()
        manifest[p.name] = {
            "url": url_map.get(p.name, ""),   # empty if not in map (and no default map)
            "sha256": _sha256(p),
            "bytes": st.st_size,
            "version": _iso8601_utc(st.st_mtime),
        }

    # Write output
    out = Path(output_path).resolve() if output_path else (models_dir / "models.json")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        else:
            json.dump(manifest, f, separators=(",", ":"))
    return out

# -------- CLI --------

def cli_make_models_json(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="astril-make-models-json")
    p.add_argument("--models-dir", type=str, default=None,
                   help="Directory to scan (default: astril/models beside this file)")
    p.add_argument("--out", type=str, default=None,
                   help="Output path for models.json (default: <models_dir>/models.json)")
    p.add_argument("--url-map", type=str, default=None,
                   help=("Path to filename->URL JSON. If omitted, "
                         "defaults to <models_dir>/osf_urls.json when present."))
    p.add_argument("--include", type=str, default="*.h5,*.cfg",
                   help="Comma-separated include globs")
    p.add_argument("--exclude", type=str, default=".git*,*.gitignore,*.gitkeep,models.json,osf_urls.json,*.txt",
                   help="Comma-separated exclude globs")
    p.add_argument("--no-pretty", action="store_true", help="Disable pretty printing")
    p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing output")
    args = p.parse_args(argv)

    include = [s.strip() for s in args.include.split(",") if s.strip()]
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]

    out = generate_models_json(
        models_dir=args.models_dir,
        output_path=args.out,
        url_mapping=args.url_map,       # None triggers auto default; path is honored; missing path raises
        include=include,
        exclude=exclude,
        overwrite=not args.no_overwrite,
        pretty=not args.no_pretty,
    )
    print(out)

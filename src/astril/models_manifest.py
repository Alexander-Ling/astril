from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Union

PKG_NAME = "astril"

def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _iso8601_utc(ts: float) -> str:
    # Example: "2025-08-15T14:02:33Z"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)

def generate_models_json(
    models_dir: Union[str, Path, None] = None,
    output_path: Union[str, Path, None] = None,
    url_mapping: Optional[Union[Dict[str, str], str, Path]] = None,
    include: Optional[Sequence[str]] = ("*.h5", "*.cfg", "*.json", "*.bin", "*.pt", "*.onnx"),
    exclude: Optional[Sequence[str]] = (".git*", "*.gitignore", "*.gitkeep", "*.txt"),
    overwrite: bool = True,
    pretty: bool = True,
) -> Path:
    """
    Scan a models directory and write a models.json manifest with entries:
      { "<filename>": { "url": "<optional>", "sha256": "<hash>", "bytes": <int>, "version": "<ISO8601>" }, ... }

    Parameters
    ----------
    models_dir : path-like or None
        Directory to scan. Defaults to the package-local "src/astril/models" if present,
        otherwise "<this file>/models".
    output_path : path-like or None
        Where to write models.json. Defaults to <models_dir>/models.json.
    url_mapping : dict or path to a JSON file mapping filename -> URL (e.g., OSF /download links).
        If provided, URLs are filled from this mapping; unknown files get an empty URL.
    include : list of glob patterns to include (default picks common weight/config types).
    exclude : list of glob patterns to exclude (git files, .txt tracking by default).
    overwrite : if False and output exists, raises FileExistsError.
    pretty : pretty-print JSON with indentation.

    Returns
    -------
    Path to the written JSON.
    """
    # Resolve models_dir
    if models_dir is None:
        # Prefer source-layout path in development
        candidate = Path(__file__).resolve().parent / "models"
        if candidate.exists():
            models_dir = candidate
        else:
            # Fallback: two levels up if module moved
            models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir = Path(models_dir).resolve()

    if not models_dir.exists() or not models_dir.is_dir():
        raise NotADirectoryError(f"Models directory not found: {models_dir}")

    # Load URL mapping if provided
    url_map: Dict[str, str] = {}
    if isinstance(url_mapping, (str, Path)):
        with Path(url_mapping).open("r", encoding="utf-8") as f:
            url_map = json.load(f)
    elif isinstance(url_mapping, dict):
        url_map = dict(url_mapping)

    # Discover files
    include = include or ("*",)
    exclude = exclude or tuple()

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

    manifest: Dict[str, Dict[str, object]] = {}
    for p in files:
        stat = p.stat()
        entry = {
            "url": url_map.get(p.name, ""),          # fill later or supply via url_mapping
            "sha256": _sha256(p),
            "bytes": stat.st_size,
            "version": _iso8601_utc(stat.st_mtime),  # per-file version; you can switch to a global if you prefer
        }
        manifest[p.name] = entry

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

# Optional CLI wrapper
def cli_make_models_json(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="astril-make-models-json")
    p.add_argument("--models-dir", type=str, default=None, help="Directory to scan (default: astril/models)")
    p.add_argument("--out", type=str, default=None, help="Output path for models.json (default: <models_dir>/models.json)")
    p.add_argument("--url-map", type=str, default=None, help="JSON file mapping filename -> URL (e.g., OSF /download)")
    p.add_argument("--include", type=str, default="*.h5,*.cfg,*.json,*.bin,*.pt,*.onnx", help="Comma-separated include globs")
    p.add_argument("--exclude", type=str, default=".git*,*.gitignore,*.gitkeep,*.txt", help="Comma-separated exclude globs")
    p.add_argument("--no-pretty", action="store_true", help="Disable pretty printing")
    p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing output")
    args = p.parse_args(argv)

    include = [s.strip() for s in args.include.split(",") if s.strip()]
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]
    out = generate_models_json(
        models_dir=args.models_dir,
        output_path=args.out,
        url_mapping=args.url_map,
        include=include,
        exclude=exclude,
        overwrite=not args.no_overwrite,
        pretty=not args.no_pretty,
    )
    print(out)
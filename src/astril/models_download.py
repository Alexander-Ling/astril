from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import hashlib, json, os, urllib.request
from importlib import resources as ir
from tqdm import tqdm

from .paths import preferred_models_dir

@dataclass
class ModelEntry:
    filename: str
    url: str
    sha256: Optional[str]
    bytes: Optional[int]

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _urlopen_with_optional_osf_auth(url: str):
    req = urllib.request.Request(url)
    if "osf.io" in url:
        token = os.environ.get("OSF_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req)

def _load_manifest() -> Dict[str, Dict]:
    ref = ir.files("astril") / "models" / "models.json"
    with ir.as_file(ref) as p:
        return json.loads(Path(p).read_text(encoding="utf-8"))

def _iter_entries(manifest: Dict[str, Dict]) -> List[ModelEntry]:
    out: List[ModelEntry] = []
    for fname, meta in manifest.items():
        out.append(
            ModelEntry(
                filename=fname,
                url=meta.get("url", "") or "",
                sha256=meta.get("sha256"),
                bytes=meta.get("bytes"),
            )
        )
    return out

def locate_models_dir() -> Path:
    """Directory astril will use for model binaries."""
    return preferred_models_dir(create=True)

def locate_model(filename: str) -> Path:
    p = locate_models_dir() / filename
    if not p.exists():
        raise FileNotFoundError(
            f"Model '{filename}' not found at {p}.\n"
            f"Run 'astril-download-models' to fetch required files."
        )
    return p

def download_models(overwrite: bool = False, only: Optional[List[str]] = None) -> Path:
    """Download (subset of) models listed in packaged models.json; verify SHA256; atomic writes."""
    target = locate_models_dir()
    manifest = _load_manifest()
    entries = _iter_entries(manifest)

    if only:
        only_set = set(only)
        entries = [e for e in entries if e.filename in only_set]
        missing = [x for x in only if x not in manifest]
        if missing:
            raise ValueError(f"Unknown model(s) in models.json: {missing}")

    for e in entries:
        if not e.url:
            raise RuntimeError(f"No URL configured for '{e.filename}' in models.json")

        dst = target / e.filename
        if dst.exists() and not overwrite:
            if e.sha256:
                try:
                    if _sha256(dst) == e.sha256:
                        print(f"[skip] {e.filename} (present, checksum OK)")
                        continue
                    else:
                        print(f"[warn] {e.filename} present but checksum mismatch; re-downloading...")
                except Exception:
                    print(f"[warn] Could not checksum {e.filename}; re-downloading...")
            else:
                print(f"[skip] {e.filename} (present, no checksum in manifest)")
                continue

        tmp = dst.with_suffix(".part")
        if tmp.exists():
            tmp.unlink()

        print(f"[get]  {e.filename} ? {e.url}")
        with _urlopen_with_optional_osf_auth(e.url) as resp:
            # Content length may be missing; tqdm handles None.
            total = getattr(resp, "length", None) or e.bytes
            with tqdm(total=total, unit="B", unit_scale=True, desc=e.filename) as bar:
                with tmp.open("wb") as out:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk: break
                        out.write(chunk)
                        bar.update(len(chunk))

        if e.sha256:
            got = _sha256(tmp)
            if got != e.sha256:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum mismatch for {e.filename}: got {got}, expected {e.sha256}")

        tmp.replace(dst)

    print(f"\nModels available at: {target}")
    return target

# ---- CLI ----
def cli_download(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="astril-download-models")
    p.add_argument("--overwrite", action="store_true", help="Replace existing files")
    p.add_argument("--only", type=str, help="Comma-separated list of filenames to fetch")
    args = p.parse_args(argv)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    out = download_models(overwrite=args.overwrite, only=only)
    print(out)

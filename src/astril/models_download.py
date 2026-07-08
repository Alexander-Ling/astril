from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import hashlib, json, os, zipfile, shutil, time, random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from importlib import resources as ir
from tqdm import tqdm

from .paths import preferred_models_dir

@dataclass
class ModelEntry:
    filename: str
    url: str
    sha256: Optional[str]
    bytes: Optional[int]
    # Optional extra metadata (forward-compatible with richer models.json)
    kind: Optional[str] = None            # e.g. "pytorch_zip    "
    extract_to: Optional[str] = None      # target subdir name under models dir
    expect: Optional[List[str]] = None    # sanity-check paths after extract

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _maybe_normalize_osf_url(url: str) -> str:
    # Encourage direct download on OSF to avoid HTML intermediates
    if "osf.io" in url and "download=1" not in url:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}download=1"
    return url

def _make_session() -> requests.Session:
    s = requests.Session()
    # Limit retries to at most 10 attempts, even if env asks for more
    _cfg_total = int(os.environ.get("ASTRIL_DOWNLOAD_RETRIES", "10"))
    _total = min(_cfg_total, 10)
    retry = Retry(
        total=_total,
        backoff_factor=1.0,  # 1,2,4,8,... seconds
        status_forcelist=(500, 502, 503, 504, 522, 524),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    s.headers.update({
        "User-Agent": "astril-model-downloader/1.0 (+https://github.com/your-org/astril)",
        "Accept": "*/*",
    })
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    token = os.environ.get("OSF_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s

def _supports_range(headers: requests.structures.CaseInsensitiveDict) -> bool:
    return "bytes" in headers.get("Accept-Ranges", "").lower()

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
                kind=meta.get("kind"),
                extract_to=meta.get("extract_to"),
                expect=meta.get("expect"),
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
        if "REPLACE_WITH" in e.url:
            raise RuntimeError(
                f"Placeholder URL configured for '{e.filename}' in models.json. "
                "Upload the packaged model archive to OSF and replace this with the OSF direct-download URL."
            )

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
        if overwrite and tmp.exists():
            tmp.unlink()

        url = _maybe_normalize_osf_url(e.url)
        sess = _make_session()
        chunk_size = 1 << 20  # 1 MiB

        # Loop handles resume automatically across retries managed by session
        while True:
            bytes_on_disk = tmp.stat().st_size if tmp.exists() else 0
            headers = {}
            # Probe if server supports Range
            probe = sess.head(url, timeout=30)
            can_range = _supports_range(probe.headers)
            expected_total = e.bytes
            try:
                if expected_total is None:
                    # Try to infer from HEAD
                    cl = probe.headers.get("Content-Length")
                    if cl is not None and cl.isdigit():
                        expected_total = int(cl)
            except Exception:
                pass

            if can_range and bytes_on_disk > 0:
                headers["Range"] = f"bytes={bytes_on_disk}-"

            print(f"[get]  {e.filename} ? {url}" + (f" (resume at {bytes_on_disk}B)" if "Range" in headers else ""))
            with sess.get(url, stream=True, headers=headers, timeout=60) as resp:
                status_ok = resp.status_code in (200, 206)
                if not status_ok:
                    raise RuntimeError(f"HTTP {resp.status_code} fetching {url}")

                mode = "ab" if (resp.status_code == 206 and bytes_on_disk > 0) else "wb"
                if mode == "wb":
                    bytes_on_disk = 0
                total_for_bar = expected_total if expected_total is not None else None
                with tqdm(total=total_for_bar, initial=bytes_on_disk, unit="B", unit_scale=True, desc=e.filename) as bar:
                    with tmp.open(mode) as out:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            out.write(chunk)
                            bar.update(len(chunk))

            final_size = tmp.stat().st_size
            if expected_total is not None and final_size != expected_total:
                # short read; sleep a bit and try to resume
                print(f"[warn] Short read for {e.filename}: got {final_size} of {expected_total} bytes; retrying...")
                time.sleep(1.0 + random.random() * 0.5)
                continue
            break

        if e.sha256:
            got = _sha256(tmp)
            if got != e.sha256:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum mismatch for {e.filename}: got {got}, expected {e.sha256}")

        tmp.replace(dst)

        # ---- Post-processing: unzip archives if needed ----
        if dst.suffix.lower() == ".zip" or (e.kind and "zip" in e.kind):
            extract_dir = target / (e.extract_to or dst.stem)
            if extract_dir.exists() and overwrite:
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"[unzip] {dst.name} -> {extract_dir}")
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(extract_dir)

                # --- Flatten a single top-level directory, if present ---
                # e.g., archive contains "GBM_seg_v1/<files>" and extract_dir is ".../GBM_seg_v1"
                # which would produce ".../GBM_seg_v1/GBM_seg_v1/<files>"; flatten that.
                names = zf.namelist()
                top_levels = {n.split("/", 1)[0] for n in names if "/" in n}
                if len(top_levels) == 1:
                    inner = extract_dir / next(iter(top_levels))
                   # Only flatten if the inner directory actually exists and is not the same path
                    if inner.is_dir() and inner.resolve() != extract_dir.resolve():
                        print(f"[fixup] Flattening single top-level folder: {inner.name} -> {extract_dir}")
                        for child in inner.iterdir():
                            shutil.move(str(child), extract_dir / child.name)
                        # remove now-empty inner folder
                        try:
                           inner.rmdir()
                        except OSError:
                            # best effort; ignore if non-empty hidden files remain
                            pass

            # Optional sanity-check: confirm expected files exist
            if e.expect:
                missing = [p for p in e.expect if not (extract_dir / p).exists()]
                if missing:
                    raise RuntimeError(f"Archive '{e.filename}' extracted but missing expected items: {missing}")
            # Unless user chose to keep archives, remove the zip
            if not getattr(download_models, "_keep_archives", False):
                try:
                    dst.unlink()
                except Exception:
                    print(f"[warn] Could not remove archive {dst}")

    print(f"\nModels available at: {target}")
    return target

# ---- CLI ----
def cli_download(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="astril-download-models")
    p.add_argument("--overwrite", action="store_true", help="Replace existing files")
    p.add_argument("--only", type=str, help="Comma-separated list of filenames to fetch")
    p.add_argument("--keep-archives", action="store_true", help="Keep .zip archives after extraction")
    args = p.parse_args(argv)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    # Stash flag on the function for simple propagation without changing signature
    download_models._keep_archives = bool(args.keep_archives)  # type: ignore[attr-defined]
    out = download_models(overwrite=args.overwrite, only=only)
    print(out)

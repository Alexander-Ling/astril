from __future__ import annotations
from pathlib import Path
import os
from platformdirs import user_data_dir

_PKG = "astril"

def package_root() -> Path:
    return Path(__file__).resolve().parent

def package_models_dir() -> Path:
    return package_root() / "models"

def preferred_models_dir(create: bool = True) -> Path:
    """
    Prefer a writable package-local models/ (e.g., editable installs).
    Else use a per-user cache: <platformdirs>/astril/models.
    """
    pkg = package_models_dir()
    try:
        if pkg.exists() and os.access(pkg, os.W_OK):
            if create: pkg.mkdir(parents=True, exist_ok=True)
            return pkg
        if not pkg.exists() and os.access(pkg.parent, os.W_OK):
            if create: pkg.mkdir(parents=True, exist_ok=True)
            return pkg
    except Exception:
        pass

    user_dir = Path(user_data_dir(_PKG)) / "models"
    if create: user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir
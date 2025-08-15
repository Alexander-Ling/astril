"""
astril: Automated Segmentation Toolkit for Radiology Image Libraries
(See pyproject.toml for project metadata)
"""

import sys
if sys.version_info < (3, 11):
    raise RuntimeError(
        "astril requires Python 3.11 or newer. "
        "On Windows, try:  py -3.11 -m venv .venv && .venv\\Scripts\\pip install astril"
    )

# Optional: keep version in sync with pyproject.toml without hardcoding:
try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("astril")
except Exception:
    __version__ = "unknown"

"""
astril: Automated Segmentation Toolkit for Radiology Image Libraries
(See pyproject.toml for project metadata)
"""

try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except Exception:  # pragma: no cover
    version = None
    PackageNotFoundError = Exception

try:
    __version__ = version("astril") if version else "unknown"
except PackageNotFoundError:
    __version__ = "unknown"

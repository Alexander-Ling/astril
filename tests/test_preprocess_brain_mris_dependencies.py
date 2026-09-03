from pathlib import Path
from types import SimpleNamespace

import pytest

import astril.preprocess_brain_mris as batch_preprocess


def test_dcm2niix_check_accepts_executable_on_path(monkeypatch):
    monkeypatch.setattr(batch_preprocess.shutil, "which", lambda name: r"C:\tools\dcm2niix.exe")

    assert batch_preprocess._ensure_dcm2niix_accessible().endswith("dcm2niix.exe")


def test_dcm2niix_check_adds_python_package_directory(monkeypatch, tmp_path):
    executable = tmp_path / "dcm2niix.exe"
    executable.write_bytes(b"")
    monkeypatch.setattr(batch_preprocess.shutil, "which", lambda name: str(executable) if str(tmp_path) in batch_preprocess.os.environ.get("PATH", "") else None)
    monkeypatch.setattr(
        batch_preprocess.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(tmp_path)], origin=None),
    )
    monkeypatch.setattr(batch_preprocess.sysconfig, "get_path", lambda name: str(tmp_path / "Scripts"))

    assert Path(batch_preprocess._ensure_dcm2niix_accessible()) == executable


def test_dcm2niix_check_reports_actionable_install_instructions(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_preprocess.shutil, "which", lambda name: None)
    monkeypatch.setattr(batch_preprocess.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(batch_preprocess.sysconfig, "get_path", lambda name: str(tmp_path / "Scripts"))

    with pytest.raises(RuntimeError, match="python -m pip install dcm2niix"):
        batch_preprocess._ensure_dcm2niix_accessible()

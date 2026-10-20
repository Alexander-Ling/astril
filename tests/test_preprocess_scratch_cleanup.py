"""Unmasked (face-containing) preprocessing intermediates must never be left in the output tree.

Covers: the scratch dir lives outside output_dir and is registered for cleanup; stale
`.astril_tmp_*` workspaces from older versions are swept; and the CLI stop handlers delete
registered scratch dirs before exiting on SIGINT/SIGTERM.
"""
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from astril import preprocessing_utils
from astril.preprocess_single_brain_mri import _remove_stale_output_temp_dirs, _unmasked_scratch_dir


class ScratchDirTests(unittest.TestCase):
    def test_scratch_dir_is_outside_output_dir_registered_and_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            out.mkdir()
            with _unmasked_scratch_dir(str(out)) as scratch:
                self.assertTrue(os.path.isdir(scratch))
                self.assertFalse(Path(scratch).resolve().is_relative_to(out.resolve()))
                self.assertIn(os.path.abspath(scratch), preprocessing_utils._TEMP_DIRS_FOR_ATEXIT_CLEANUP)
            self.assertFalse(os.path.exists(scratch))
            self.assertNotIn(os.path.abspath(scratch), preprocessing_utils._TEMP_DIRS_FOR_ATEXIT_CLEANUP)

    def test_scratch_dir_honors_astril_scratch_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            out, scratch_root = Path(temp) / "out", Path(temp) / "scratch"
            out.mkdir()
            with patch.dict(os.environ, {"ASTRIL_SCRATCH_DIR": str(scratch_root)}):
                with _unmasked_scratch_dir(str(out)) as scratch:
                    self.assertEqual(Path(scratch).parent.resolve(), scratch_root.resolve())

    def test_stale_output_temp_dirs_are_swept_and_others_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / ".astril_tmp_abc").mkdir()
            (out / ".astril_tmp_abc" / "T1c_reg.nii.gz").write_bytes(b"x")
            (out / "transforms").mkdir()
            (out / ".other").mkdir()
            _remove_stale_output_temp_dirs(str(out))
            self.assertFalse((out / ".astril_tmp_abc").exists())
            self.assertTrue((out / "transforms").exists())
            self.assertTrue((out / ".other").exists())


@unittest.skipIf(os.name == "nt", "POSIX signal delivery")
class CliStopHandlerTests(unittest.TestCase):
    def _run_child(self, sig):
        with tempfile.TemporaryDirectory() as temp:
            scratch = Path(temp) / "scratch"
            child = textwrap.dedent(f"""
                import os, signal
                from astril.preprocessing_utils import _register_temp_dir_for_atexit
                from astril.preprocess_brain_mris import _install_cli_stop_handlers
                os.makedirs({str(scratch)!r})
                _register_temp_dir_for_atexit({str(scratch)!r})
                _install_cli_stop_handlers()
                signal.raise_signal(signal.{sig.name})
                print("handler did not exit")
            """)
            proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=60)
            return proc, scratch.exists()

    def test_sigint_removes_scratch_and_exits_130(self):
        proc, exists = self._run_child(signal.SIGINT)
        self.assertEqual(proc.returncode, 130, proc.stdout + proc.stderr)
        self.assertFalse(exists)

    def test_sigterm_removes_scratch_and_exits_143(self):
        proc, exists = self._run_child(signal.SIGTERM)
        self.assertEqual(proc.returncode, 143, proc.stdout + proc.stderr)
        self.assertFalse(exists)


if __name__ == "__main__":
    unittest.main()

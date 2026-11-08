"""An empty brainmask must fail the exam, not report success with no outputs.

Previously every masked scan came out all-zero, Step 4 dropped each label with a verbose-only warning, and the exam
returned "RAN" having written only its brainmask.

No patient data is used.
"""
import os
import tempfile
import unittest

import numpy as np

try:
    import SimpleITK as sitk
    from astril.preprocess_single_brain_mri import preprocess_single_brain_mri
    from astril.preprocessing_utils import count_mask_voxels
except ImportError:  # preprocessing extras not installed
    sitk = None


def _write(arr, path):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteImage(img, path)
    return path


@unittest.skipIf(sitk is None, "preprocessing extras (SimpleITK) not installed")
class EmptyBrainmaskTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        zz, yy, xx = np.mgrid[:40, :48, :48]
        head = (((zz - 20) / 16.0) ** 2 + ((yy - 24) / 20.0) ** 2 + ((xx - 24) / 20.0) ** 2) <= 1.0
        self.t1c = _write(np.where(head, 800.0, 0.0).astype(np.float32), os.path.join(self.root, "SYN_d0_T1c.nii.gz"))
        self.empty_mask = _write(np.zeros(head.shape, np.uint8), os.path.join(self.root, "empty_mask.nii.gz"))
        self.out_dir = os.path.join(self.root, "out")

    def tearDown(self):
        self._tmp.cleanup()

    def test_count_mask_voxels(self):
        self.assertEqual(count_mask_voxels(self.empty_mask), 0)
        arr = np.zeros((4, 4, 4), np.uint8)
        arr[1:3, 1:3, 1:3] = 1
        self.assertEqual(count_mask_voxels(_write(arr, os.path.join(self.root, "m.nii.gz"))), 8)

    def test_empty_provided_brainmask_raises(self):
        with self.assertRaisesRegex(RuntimeError, "Brainmask is empty"):
            preprocess_single_brain_mri(
                output_dir=self.out_dir,
                scans={"T1c": self.t1c},
                anchor_label="T1c",
                brainmask_path=self.empty_mask,
                registration_backend="itk",
                final_dims=(48, 48, 40),
                final_voxels=(2.0, 2.0, 2.0),
                patientID="SYN",
                timepoint="d0",
                verbose=False,
            )
        produced = [f for f in os.listdir(self.out_dir) if f.endswith("_brain.nii.gz") or f.endswith("_brain-norm.nii.gz")] \
            if os.path.isdir(self.out_dir) else []
        self.assertEqual(produced, [])


if __name__ == "__main__":
    unittest.main()

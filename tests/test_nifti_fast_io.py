"""nibabel NIfTI fast path (preprocessing_utils.read_nifti_fast / write_nifti_fast) must match SimpleITK exactly.

Threaded batch preprocessing serialises all ITK file I/O behind one lock (ITK's I/O factories are not thread-safe),
so 3D scalar NIfTI goes through nibabel instead. These checks pin the contract: whatever the fast path returns is
voxel- and geometry-identical to sitk.ReadImage, what it writes reads back identically through ITK, and anything it
cannot reproduce (scaled voxels, 4D, NRRD) is handed back to ITK.
"""
import os
import tempfile
import unittest

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from astril.preprocessing_utils import read_image, read_nifti_fast, write_image, write_nifti_fast


def _oblique_image(dtype, size=(9, 7, 5), seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        arr = rng.integers(0, 200, size=size[::-1]).astype(dtype)
    else:
        arr = rng.standard_normal(size[::-1]).astype(dtype) * 100
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.9, 1.1, 2.5))
    img.SetOrigin((-100.25, 33.5, 12.0))
    t = sitk.Euler3DTransform((0, 0, 0), 0.1, -0.2, 0.3, (0, 0, 0))
    img.SetDirection(t.GetMatrix())
    return img


def _assert_same(tc, a, b):
    tc.assertEqual(a.GetSize(), b.GetSize())
    tc.assertEqual(a.GetPixelID(), b.GetPixelID())
    np.testing.assert_allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(sitk.GetArrayViewFromImage(a), sitk.GetArrayViewFromImage(b))


class NiftiFastIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_itk_written_files_like_itk(self):
        for dtype in (np.float32, np.uint8, np.int16, np.float64):
            for ext in (".nii", ".nii.gz"):
                with self.subTest(dtype=dtype.__name__, ext=ext):
                    p = os.path.join(self.d, f"a_{dtype.__name__}{ext}")
                    sitk.WriteImage(_oblique_image(dtype), p)
                    fast = read_nifti_fast(p)
                    self.assertIsNotNone(fast)
                    _assert_same(self, sitk.ReadImage(p), fast)

    def test_write_reads_back_identically_through_itk_and_nibabel(self):
        for dtype in (np.float32, np.uint8, np.int16):
            with self.subTest(dtype=dtype.__name__):
                img = _oblique_image(dtype)
                p = os.path.join(self.d, f"w_{dtype.__name__}.nii.gz")
                self.assertTrue(write_nifti_fast(img, p))
                _assert_same(self, img, sitk.ReadImage(p))
                # Same header conventions as ITK's writer, so nibabel-based consumers see no difference either.
                ref = os.path.join(self.d, "ref.nii.gz")
                sitk.WriteImage(img, ref)
                h_new, h_itk = nib.load(p).header, nib.load(ref).header
                np.testing.assert_allclose(nib.load(p).affine, nib.load(ref).affine, rtol=0, atol=1e-5)
                for key in ("sform_code", "qform_code", "xyzt_units", "datatype", "bitpix"):
                    self.assertEqual(h_new[key], h_itk[key], key)
                np.testing.assert_array_equal(h_new["pixdim"][1:4], h_itk["pixdim"][1:4])

    def test_pixel_type_argument_casts_like_itk(self):
        p = os.path.join(self.d, "u8.nii.gz")
        sitk.WriteImage(_oblique_image(np.uint8), p)
        _assert_same(self, sitk.ReadImage(p, sitk.sitkFloat32), read_image(p, sitk.sitkFloat32))

    def test_scaled_voxels_fall_back_to_itk(self):
        # ITK applies scl_slope/scl_inter and promotes integers to float; the fast path must not try to mimic that.
        p = os.path.join(self.d, "scaled.nii")
        sitk.WriteImage(_oblique_image(np.uint8), p)
        with open(p, "r+b") as fh:
            fh.seek(112)
            fh.write(np.array([1.0 / 255.0, 0.0], dtype="<f4").tobytes())
        self.assertIsNone(read_nifti_fast(p))
        via_wrapper = read_image(p)
        self.assertEqual(via_wrapper.GetPixelID(), sitk.sitkFloat32)
        _assert_same(self, sitk.ReadImage(p), via_wrapper)

    def test_4d_and_nrrd_fall_back_to_itk(self):
        img4d = sitk.JoinSeries([_oblique_image(np.float32, seed=s) for s in range(3)])
        p4 = os.path.join(self.d, "vol4d.nii.gz")
        write_image(img4d, p4)
        self.assertIsNone(read_nifti_fast(p4))
        self.assertFalse(write_nifti_fast(img4d, os.path.join(self.d, "x.nii.gz")))
        _assert_same(self, img4d, read_image(p4))
        pn = os.path.join(self.d, "vol.nrrd")
        img = _oblique_image(np.int16)
        self.assertFalse(write_nifti_fast(img, pn))
        write_image(img, pn)
        self.assertIsNone(read_nifti_fast(pn))
        _assert_same(self, img, read_image(pn))

    def test_missing_orientation_falls_back_to_itk(self):
        # qform_code == sform_code == 0: nibabel centres its fallback affine, ITK does not.
        p = os.path.join(self.d, "noform.nii")
        sitk.WriteImage(_oblique_image(np.float32), p)
        with open(p, "r+b") as fh:
            fh.seek(252)
            fh.write(np.array([0, 0], dtype="<i2").tobytes())
        self.assertIsNone(read_nifti_fast(p))
        _assert_same(self, sitk.ReadImage(p), read_image(p))


if __name__ == "__main__":
    unittest.main()

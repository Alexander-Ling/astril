"""GPU registration backend (astril.gpu) vs ITK on a synthetic pair. Skipped without CUDA.

Checks: Triton self-test; recursive-Gaussian smoothing and cubic B-spline resampling match ITK; a full registration
(Mattes MI, 3-level pyramid, RSGD + PhysicalShift scales, deterministic NONE sampling) lands on the same pose as ITK;
register_images(registration_backend="gpu") runs end to end.
"""
import os
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from astril import gpu

HAVE_CUDA = gpu.cuda_available()


def _phantom(size=64, spacing=2.0):
    """Smooth anatomical-ish volume: ellipsoids of different intensity + mild texture."""
    z, y, x = np.mgrid[:size, :size, :size].astype(np.float32) - size / 2
    arr = 100.0 * ((x / 24) ** 2 + (y / 20) ** 2 + (z / 18) ** 2 < 1)
    arr += 80.0 * (((x - 6) / 8) ** 2 + ((y + 4) / 6) ** 2 + (z / 10) ** 2 < 1)
    arr += 50.0 * (((x + 9) / 5) ** 2 + ((y - 7) / 9) ** 2 + ((z - 3) / 6) ** 2 < 1)
    arr += 5.0 * np.sin(x / 3.0) * np.cos(y / 4.0)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((spacing,) * 3)
    img.SetOrigin((-size * spacing / 2,) * 3)
    return sitk.SmoothingRecursiveGaussian(img, 1.5)


def _moved(img, tfm):
    return sitk.Resample(img, img, tfm, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def _configure(r, init):
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([4.0, 2.0, 0.0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetMetricSamplingStrategy(r.NONE)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=150,
                                               gradientMagnitudeTolerance=1e-6)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetInitialTransform(init, inPlace=False)


@unittest.skipUnless(HAVE_CUDA, "CUDA not available")
class GpuBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed = _phantom()
        true = sitk.Euler3DTransform((0.0, 0.0, 0.0), 0.08, -0.06, 0.1, (3.0, -2.0, 4.0))
        cls.moving = _moved(cls.fixed, true.GetInverse())
        cls.true = true

    def test_triton_self_test(self):
        self.assertTrue(gpu.triton_usable(), "Triton self-test failed (see warning for the reason)")

    def test_smoothing_matches_itk(self):
        import torch
        from astril.gpu import filters as gf
        ref = sitk.SmoothingRecursiveGaussian(self.moving, [3.0] * 3, False)
        g = gf.smoothing_recursive_gaussian(torch.as_tensor(sitk.GetArrayFromImage(self.moving), device="cuda"),
                                            self.moving.GetSpacing(), [3.0] * 3)
        a, b = g.cpu().numpy(), sitk.GetArrayFromImage(ref)
        self.assertLess(np.abs(a - b).max() / np.abs(b).max(), 1e-6)

    @unittest.skipUnless(HAVE_CUDA and gpu.triton_usable(), "Triton not usable")
    def test_bspline_resample_matches_itk(self):
        from astril.gpu.resample import gpu_resample_image
        ref = sitk.Resample(self.moving, self.fixed, self.true, sitk.sitkBSpline, 0.0, sitk.sitkFloat32)
        out = gpu_resample_image(self.moving, self.fixed, self.true, sitk.sitkBSpline, 0.0, sitk.sitkFloat32)
        self.assertIsNotNone(out)
        a, b = sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(ref)
        self.assertLess(np.abs(a - b).max() / np.abs(b).max(), 1e-6)

    def test_registration_matches_itk(self):
        from astril.gpu.registration import GpuImageRegistrationMethod
        init = sitk.CenteredTransformInitializer(self.fixed, self.moving, sitk.Euler3DTransform(),
                                                 sitk.CenteredTransformInitializerFilter.GEOMETRY)
        outs = {}
        for name, cls in [("itk", sitk.ImageRegistrationMethod), ("gpu", GpuImageRegistrationMethod)]:
            r = cls()
            _configure(r, sitk.Euler3DTransform(init))
            outs[name] = r.Execute(self.fixed, self.moving)
        pts = [(x, y, z) for x in (-30, 0, 30) for y in (-30, 0, 30) for z in (-30, 0, 30)]
        d = max(np.linalg.norm(np.subtract(outs["itk"].TransformPoint(p), outs["gpu"].TransformPoint(p))) for p in pts)
        err = max(np.linalg.norm(np.subtract(outs["gpu"].TransformPoint(p), self.true.TransformPoint(p))) for p in pts)
        self.assertLess(d, 0.1, f"GPU vs ITK final pose differ by {d:.3f} mm")
        self.assertLess(err, 1.0, f"GPU pose error {err:.3f} mm")

    def test_register_images_gpu_backend(self):
        from astril.preprocess import register_images
        with tempfile.TemporaryDirectory() as td:
            fp, mp = os.path.join(td, "f.nii.gz"), os.path.join(td, "m.nii.gz")
            sitk.WriteImage(self.fixed, fp)
            sitk.WriteImage(self.moving, mp)
            register_images(fp, mp, os.path.join(td, "o.nii.gz"), transform_path=os.path.join(td, "t.tfm"),
                            registration_strategy="accurate", metric_sampling_seed=1, interpolation=3,
                            registration_voxel_mm="2,2,2", registration_backend="gpu")
            t = sitk.ReadTransform(os.path.join(td, "t.tfm"))
            err = max(np.linalg.norm(np.subtract(t.TransformPoint(p), self.true.TransformPoint(p)))
                      for p in [(-30, 0, 20), (25, -25, 0), (0, 30, -30)])
            self.assertLess(err, 1.0)
            self.assertTrue(os.path.exists(os.path.join(td, "o.nii.gz")))


if __name__ == "__main__":
    unittest.main()

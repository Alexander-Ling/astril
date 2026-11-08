"""FSL b-vector reorientation after resampling (register_images -> update_fsl_vectors_after_transform).

Independent check: each input image contains a thin bright line along a known world direction g, and its .bvec is g
expressed in the input's FSL voxel frame. The image is resampled with a rigid transform (saved as a CompositeTransform
file, as astril writes them) onto a differently oriented output grid. The line's direction in the output is measured by
PCA of the bright voxels' physical coordinates, and the written .bvec must point the same way. Covers oblique grids and
both voxel-to-world determinant signs (FSL's x flip).
"""
import os
import tempfile
import unittest

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from astril.preprocess import register_images
from astril.preprocessing_utils import _fsl_bvec_frame

P = np.diag([-1.0, -1.0, 1.0])  # LPS <-> RAS


def _rot(ax, ay, az):
    t = sitk.Euler3DTransform((0, 0, 0), ax, ay, az, (0, 0, 0))
    return np.asarray(t.GetMatrix()).reshape(3, 3)


def _grid(size, spacing, direction, centre):
    img = sitk.Image(list(size), sitk.sitkFloat32)
    img.SetSpacing(spacing)
    img.SetDirection(tuple(np.asarray(direction).reshape(-1)))
    half = (np.asarray(size) - 1) / 2.0
    img.SetOrigin(tuple(np.asarray(centre) - np.asarray(direction) @ (half * np.asarray(spacing))))
    return img


def _line_image(ref, g_lps, centre, radius=1.2, half_len=28.0):
    """Bright cylinder along unit LPS direction g through `centre` on ref's grid."""
    size = ref.GetSize()
    k, j, i = np.meshgrid(*[np.arange(n) for n in size[::-1]], indexing="ij")
    idx = np.stack([i.ravel(), j.ravel(), k.ravel()], 1).astype(float)
    M = np.asarray(ref.GetDirection()).reshape(3, 3) @ np.diag(ref.GetSpacing())
    pts = np.asarray(ref.GetOrigin()) + idx @ M.T - centre
    along = pts @ g_lps
    perp = np.linalg.norm(pts - np.outer(along, g_lps), axis=1)
    arr = ((perp < radius) & (np.abs(along) < half_len)).astype(np.float32) * 1000.0
    out = sitk.GetImageFromArray(arr.reshape(size[::-1]))
    out.CopyInformation(ref)
    return out


def _principal_direction_lps(img):
    arr = sitk.GetArrayFromImage(img)
    kji = np.argwhere(arr > 500.0)
    M = np.asarray(img.GetDirection()).reshape(3, 3) @ np.diag(img.GetSpacing())
    pts = np.asarray(img.GetOrigin()) + kji[:, ::-1].astype(float) @ M.T
    pts -= pts.mean(0)
    return np.linalg.svd(pts, full_matrices=False)[2][0]


class BvecReorientationTests(unittest.TestCase):
    def _case(self, in_dir, out_dir, rot_angles, g_lps):
        centre = np.array([5.0, -12.0, 20.0])
        g_lps = np.asarray(g_lps, float) / np.linalg.norm(g_lps)
        with tempfile.TemporaryDirectory() as td:
            src = _line_image(_grid((72, 72, 72), (1.0, 1.0, 1.0), in_dir, centre), g_lps, centre)
            src_path = os.path.join(td, "dwi.nii.gz")
            sitk.WriteImage(src, src_path)
            b_in = _fsl_bvec_frame(src_path).T @ (P @ g_lps)      # g in the input's FSL voxel frame
            np.savetxt(os.path.join(td, "dwi.bvec"), np.c_[[0.0, 0.0, 0.0], b_in], fmt="%.10f")
            np.savetxt(os.path.join(td, "dwi.bval"), [[0, 1000]], fmt="%d")

            ref = _grid((72, 72, 72), (1.0, 1.0, 1.0), out_dir, centre)
            ref_path = os.path.join(td, "ref.nii.gz")
            sitk.WriteImage(ref, ref_path)
            euler = sitk.Euler3DTransform(tuple(centre), *rot_angles, (2.0, -1.5, 1.0))
            comp = sitk.CompositeTransform(3)
            comp.AddTransform(euler)
            tfm_path = os.path.join(td, "t.tfm")
            sitk.WriteTransform(comp, tfm_path)

            out_path = os.path.join(td, "out.nii.gz")
            register_images(ref_path, src_path, out_path, transform_path=tfm_path, apply_only=True,
                            interpolation="linear", registration_backend="itk")
            b_out = np.loadtxt(os.path.join(td, "out.bvec"))
            self.assertTrue(np.allclose(b_out[:, 0], 0.0), "b0 vector must stay zero")
            self.assertTrue(os.path.exists(os.path.join(td, "out.bval")))

            u_lps = _principal_direction_lps(sitk.ReadImage(out_path))
            b_meas = _fsl_bvec_frame(out_path).T @ (P @ u_lps)
            cos = abs(float(b_meas @ b_out[:, 1]) / np.linalg.norm(b_out[:, 1]))
            self.assertGreater(cos, 0.999, f"bvec {b_out[:, 1]} vs measured line direction {b_meas} (|cos| {cos:.5f})")

    def test_oblique_rotation_grids(self):
        # Rotation-like sitk (LPS) directions -> RAS affine det > 0 -> FSL x flip on both frames
        self._case(_rot(0.2, -0.1, 0.3), _rot(-0.15, 0.25, -0.05), (0.3, -0.2, 0.25), [1.0, 0.4, -0.3])

    def test_negative_determinant_input_grid(self):
        # one flipped axis -> RAS det < 0 -> no FSL x flip on the input frame (output frame still flipped)
        flip = np.diag([-1.0, 1.0, 1.0])
        self._case(_rot(0.1, 0.2, -0.25) @ flip, _rot(0.05, -0.2, 0.1), (-0.25, 0.15, 0.35), [0.2, -1.0, 0.5])

    def test_negative_determinant_output_grid(self):
        flip = np.diag([1.0, 1.0, -1.0])
        self._case(_rot(-0.2, 0.1, 0.05), _rot(0.3, 0.1, -0.2) @ flip, (0.2, 0.3, -0.1), [-0.6, 0.3, 1.0])


if __name__ == "__main__":
    unittest.main()

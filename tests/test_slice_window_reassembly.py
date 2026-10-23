import tempfile
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib

from astril.data_loading import (
    load_val_slices,
    reassemble_window_predictions,
    select_window_centers,
)


def _covered(centers, num_output_slices):
    start = -(num_output_slices // 2)
    return {c + start + i for c in centers for i in range(num_output_slices)}


class TestSelectWindowCenters(unittest.TestCase):
    def test_average_mode_uses_every_relevant_slice(self):
        self.assertEqual(select_window_centers([5, 3, 4], 3, "average"), [3, 4, 5])

    def test_tile_mode_covers_every_slice_with_fewer_windows(self):
        for relevant in (range(3, 20), range(3, 21), range(3, 22), [3, 4, 9, 10, 11, 30]):
            relevant = list(relevant)
            centers = select_window_centers(relevant, 3, "tile")
            self.assertTrue(set(relevant) <= _covered(centers, 3), (relevant, centers))
            self.assertLessEqual(max(centers), max(relevant))
            self.assertLessEqual(len(centers), -(-len(relevant) // 3) + 2)

    def test_tile_mode_is_non_overlapping_on_contiguous_range(self):
        self.assertEqual(select_window_centers(range(0, 9), 3, "tile"), [1, 4, 7])

    def test_single_output_slice_tile_equals_average(self):
        self.assertEqual(select_window_centers([2, 3, 4], 1, "tile"), [2, 3, 4])

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            select_window_centers([1, 2], 3, "overwrite")


class TestReassembleWindowPredictions(unittest.TestCase):
    def test_overlapping_outputs_are_averaged_not_overwritten(self):
        # Each window predicts a constant so its contribution is identifiable. Slice 5 is
        # covered by all three windows; the old reassembly kept only the last write (6.0).
        centers = [4, 5, 6]
        preds = np.zeros((3, 2, 2, 3, 1), dtype=np.float32)
        for w, bias in enumerate((0.0, 3.0, 6.0)):
            preds[w] = bias
        out = reassemble_window_predictions(preds, centers, 3, (2, 2, 10))
        self.assertAlmostEqual(float(out[0, 0, 5, 0]), 3.0)   # mean(0, 3, 6)
        self.assertAlmostEqual(float(out[0, 0, 4, 0]), 1.5)   # mean(0, 3)
        self.assertAlmostEqual(float(out[0, 0, 3, 0]), 0.0)   # only window 4
        self.assertAlmostEqual(float(out[0, 0, 7, 0]), 6.0)   # only window 6
        self.assertEqual(float(out[0, 0, 0, 0]), 0.0)         # uncovered stays zero

    def test_out_of_range_slots_are_dropped(self):
        preds = np.ones((1, 1, 1, 3, 2), dtype=np.float32)
        out = reassemble_window_predictions(preds, [0], 3, (1, 1, 2))
        self.assertEqual(out.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(out, 1.0)


class TestLoadValSlicesOverlapMode(unittest.TestCase):
    def test_tile_mode_builds_fewer_windows_covering_the_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mask = np.zeros((8, 8, 12), dtype=np.uint8)
            mask[2:6, 2:6, 2:11] = 1
            image = np.random.default_rng(0).random((8, 8, 12)).astype(np.float32)
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(tmp / "mask.nii.gz"))
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(tmp / "t1c.nii.gz"))

            def load(mode):
                return load_val_slices(
                    0, [[str(tmp / "t1c.nii.gz")]], [str(tmp / "mask.nii.gz")],
                    [str(tmp / "mask.nii.gz")], "axial", 7, 3,
                    return_transform_info=True, target_height=8, target_width=8,
                    slice_overlap_mode=mode,
                )

            x_avg, _, _, z_avg, info = load("average")
            x_tile, _, _, z_tile, _ = load("tile")
            self.assertEqual(len(x_avg), 9)
            self.assertEqual(len(x_tile), 3)
            self.assertTrue(set(z_avg) <= _covered(z_tile, 3))

            # Reassembled predictions from the tile windows fill every in-mask slice.
            preds = np.ones((len(z_tile), 8, 8, 3, 2), dtype=np.float32)
            volume = reassemble_window_predictions(preds, z_tile, 3, info["post_alignment_shape"])
            for z in z_avg:
                self.assertTrue(np.all(volume[:, :, z] == 1.0))


if __name__ == "__main__":
    unittest.main()

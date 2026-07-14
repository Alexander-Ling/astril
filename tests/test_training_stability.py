import unittest

import torch

from astril.train import (
    _accumulation_window_size,
    _tensors_are_finite,
)
from astril.utils import combined_focal_tversky_wce_loss


class TrainingStabilityTests(unittest.TestCase):
    def test_accumulation_windows_include_short_final_window(self):
        sizes = [
            _accumulation_window_size(batch, 23, 10)
            for batch in range(1, 24)
        ]
        self.assertEqual(sizes[:10], [10] * 10)
        self.assertEqual(sizes[10:20], [10] * 10)
        self.assertEqual(sizes[20:], [3] * 3)

    def test_checkpoint_tensor_validation_rejects_nonfinite_values(self):
        self.assertTrue(_tensors_are_finite({"weights": torch.ones(2)}))
        self.assertFalse(_tensors_are_finite({"weights": torch.tensor([float("nan")])}))

    def test_custom_loss_is_finite_for_large_spatial_reductions(self):
        logits = torch.zeros((1, 256, 256, 1, 2), dtype=torch.float16, requires_grad=True)
        targets = torch.zeros_like(logits, dtype=torch.float32)
        targets[..., 0] = 1.0
        mask = torch.ones((1, 256, 256, 1, 1), dtype=torch.float32)

        loss, per_class = combined_focal_tversky_wce_loss(
            targets,
            logits,
            mask,
            [0.15, 0.85],
            [0.5, 0.5],
            [0.5, 0.5],
            gamma=2.0,
            wce_weight=0.1,
        )

        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(per_class).all().item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())


if __name__ == "__main__":
    unittest.main()

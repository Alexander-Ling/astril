import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

import astril.config as config
from astril.data_loading import sample_optional_channel_dropout
from astril.model_architecture import (
    ResidualContextUNeXt25D,
    create_dynamic_unet_from_metadata,
)
from astril.run_segmentation import average_logit
from astril.train import (
    ExponentialMovingAverage,
    _compute_model_loss,
    _initialize_model_from_compatible_checkpoint,
)


class ResidualContextUNeXt25DTests(unittest.TestCase):
    def make_model(self):
        return ResidualContextUNeXt25D(
            input_channels=18,
            num_modalities=3,
            num_input_slices=5,
            base_num_filters=8,
            encoder_level_factors=[1, 2, 4],
            num_output_slices=1,
            out_channels=2,
            center_depth=1,
            blocks_per_level=1,
            use_deep_supervision=True,
        )

    def test_new_architecture_is_the_default(self):
        self.assertEqual(config.architecture_type, "residual_context_unext_25d")
        self.assertTrue(config.use_deep_supervision)
        self.assertTrue(config.use_modality_presence_encoding)
        self.assertTrue(config.use_ema)

    def test_forward_returns_native_half_and_quarter_auxiliary_logits(self):
        model = self.make_model().train()
        main, auxiliary = model(torch.randn(2, 18, 32, 32))
        self.assertEqual(tuple(main.shape), (2, 32, 32, 1, 2))
        self.assertEqual([tuple(item.shape) for item in auxiliary], [
            (2, 16, 16, 1, 2),
            (2, 8, 8, 1, 2),
        ])

    def test_metadata_round_trip_preserves_architecture(self):
        model = self.make_model()
        restored = create_dynamic_unet_from_metadata(model.architecture_config())
        self.assertIsInstance(restored, ResidualContextUNeXt25D)
        self.assertEqual(restored.expected_input_channels, 18)
        self.assertEqual(restored.num_input_slices, 5)

    def test_presence_maps_are_required_when_enabled(self):
        model = self.make_model().eval()
        with self.assertRaisesRegex(ValueError, "expected input"):
            model(torch.randn(1, 15, 32, 32))

    def test_deep_supervision_loss_resizes_targets_at_native_scales(self):
        model = self.make_model().train()
        x = torch.randn(1, 18, 32, 32)
        labels = torch.zeros((1, 32, 32, 1, 2), dtype=torch.float32)
        labels[..., 0] = 1.0
        mask = torch.ones((1, 32, 32, 1, 1), dtype=torch.float32)
        logits, loss, _ = _compute_model_loss(
            model,
            x,
            None,
            labels,
            mask,
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            1.0,
            0.5,
            0.0,
            False,
            True,
            1.0,
        )
        self.assertEqual(tuple(logits.shape), (1, 32, 32, 1, 2))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

    def test_required_only_subset_drops_every_optional_modality(self):
        dropped = sample_optional_channel_dropout(
            [1, 2, 3],
            strategy="subset",
            subset_probabilities={"required_only": 1.0},
        )
        self.assertEqual(dropped, [1, 2, 3])

    def test_ema_updates_and_average_logit_honors_model_weights(self):
        source = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            source.weight.fill_(1.0)
        ema = ExponentialMovingAverage(source, decay=0.5)
        with torch.no_grad():
            source.weight.fill_(3.0)
        ema.update(source)
        self.assertTrue(torch.allclose(ema.model.weight, torch.full_like(ema.model.weight, 2.0)))

        first = np.asarray([[[[4.0, 0.0]]]], dtype=np.float32)
        second = np.asarray([[[[0.0, 3.0]]]], dtype=np.float32)
        result = average_logit([first, second], weights=[1.0, 2.0])
        self.assertEqual(int(result.item()), 1)

    def test_compatible_transfer_copies_backbone_and_reinitializes_five_class_heads(self):
        source = self.make_model()
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "binary.pt"
            torch.save(
                {
                    "architecture": source.architecture_config(),
                    "model_state_dict": source.state_dict(),
                },
                checkpoint_path,
            )
            original = {
                name: getattr(config, name)
                for name in (
                    "architecture_type", "num_channels", "num_input_slices",
                    "num_output_slices", "num_classes", "base_num_filters",
                    "encoder_level_factors", "center_depth", "blocks_per_level",
                    "context_stem_channels", "skip_attention_type",
                    "use_modality_presence_encoding", "use_deep_supervision",
                    "use_brainiac_embeddings", "use_dinov3_embeddings",
                )
            }
            try:
                config.architecture_type = "residual_context_unext_25d"
                config.num_channels = 3
                config.num_input_slices = 5
                config.num_output_slices = 1
                config.num_classes = 5
                config.base_num_filters = 8
                config.encoder_level_factors = [1, 2, 4]
                config.center_depth = 1
                config.blocks_per_level = 1
                config.context_stem_channels = None
                config.skip_attention_type = "residual"
                config.use_modality_presence_encoding = True
                config.use_deep_supervision = True
                config.use_brainiac_embeddings = False
                config.use_dinov3_embeddings = False
                transferred = _initialize_model_from_compatible_checkpoint(
                    checkpoint_path, torch.device("cpu")
                )
            finally:
                for name, value in original.items():
                    setattr(config, name, value)

        self.assertEqual(transferred.out_channels, 5)
        self.assertTrue(torch.equal(
            transferred.state_dict()["encoders.0.0.conv1.weight"],
            source.state_dict()["encoders.0.0.conv1.weight"],
        ))
        self.assertEqual(transferred.final_conv.out_channels, 5)
        self.assertEqual(transferred.aux_heads[0].out_channels, 5)


if __name__ == "__main__":
    unittest.main()

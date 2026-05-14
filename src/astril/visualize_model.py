#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspect a migrated PyTorch astril model checkpoint and optionally trace it.
"""

import argparse
import configparser
from pathlib import Path


def build_model_from_train_config(train_config_path):
    from astril.create_segmentation_config import parse_train_config_for_model_parameters
    from astril.model_architecture import DynamicAttentionResUNet, BrainIACEncoderFusionUNet
    cp = configparser.ConfigParser()
    cp.read(train_config_path)
    cfg = cp["DEFAULT"]
    params = parse_train_config_for_model_parameters(train_config_path)

    num_channels_val = cfg.get("num_channels", None)
    if num_channels_val is None or num_channels_val.strip() == "":
        ips = cfg.get("image_paths_files", "")
        num_channels = len(ips.split(",")) if ips.strip() else 1
    else:
        num_channels = int(num_channels_val)

    use_brainiac = cfg.getboolean("use_brainiac_embeddings", fallback=False)
    brainiac_mode = cfg.get("brainiac_embedding_type", fallback="encoder_fusion")
    if use_brainiac and brainiac_mode != "encoder_fusion":
        raise ValueError(
            "BrainIAC now supports only brainiac_embedding_type = encoder_fusion. "
            f"Found: {brainiac_mode}"
        )
    model_cls = BrainIACEncoderFusionUNet if use_brainiac else DynamicAttentionResUNet
    model_kwargs = {}
    if model_cls is BrainIACEncoderFusionUNet:
        model_kwargs["brainiac_input_channels"] = cfg.getint("brainiac_encoder_input_channels", fallback=0)

    model = model_cls(
        input_channels=num_channels * params["num_input_slices"],
        base_num_filters=cfg.getint("base_num_filters", fallback=32),
        encoder_level_factors=[
            int(x.strip())
            for x in cfg.get("encoder_level_factors", fallback="1,2,4,8").split(",")
            if x.strip()
        ],
        num_output_slices=params["num_output_slices"],
        out_channels=params["num_classes"],
        center_depth=cfg.getint("center_depth", fallback=1),
        use_se_blocks=cfg.getboolean("use_se_blocks", fallback=False),
        use_deep_supervision=cfg.getboolean("use_deep_supervision", fallback=False),
        **model_kwargs,
    )
    return model, params


def load_model_for_visualization(model_path, train_config_path=None, device=None):
    import torch
    from astril.model_architecture import create_dynamic_unet_from_metadata
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if "architecture" in checkpoint:
        model = create_dynamic_unet_from_metadata(checkpoint["architecture"])
        params = None
    elif train_config_path:
        model, params = build_model_from_train_config(train_config_path)
    else:
        raise ValueError("Checkpoint lacks architecture metadata; provide --train_config.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, params, device


def visualize_model(model, output_path, dummy_input_shape, device):
    import torch
    dummy_input = torch.zeros(dummy_input_shape, dtype=torch.float32, device=device)
    if getattr(model, "brainiac_input_channels", 0) > 0:
        dummy_brainiac = torch.zeros(
            (
                dummy_input_shape[0],
                model.brainiac_input_channels,
                max(1, dummy_input_shape[2] // 16),
                max(1, dummy_input_shape[3] // 16),
            ),
            dtype=torch.float32,
            device=device,
        )
        traced = torch.jit.trace(model, (dummy_input, dummy_brainiac))
    else:
        traced = torch.jit.trace(model, dummy_input)
    traced.save(str(output_path))
    print(model)
    print(f"Traced TorchScript model saved to '{output_path}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a migrated PyTorch astril .pt checkpoint."
    )
    parser.add_argument("--model_path", required=True, help="Path to a PyTorch .pt checkpoint.")
    parser.add_argument("--train_config", default=None, help="Optional training config for legacy metadata recovery.")
    parser.add_argument("--log_dir", default="./logs", help="Directory for the traced TorchScript artifact.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the dummy input tensor.")
    parser.add_argument("--dummy_hw", type=int, default=None, help="Optional dummy input height/width override.")
    args = parser.parse_args()

    model, params, device = load_model_for_visualization(args.model_path, args.train_config)
    arch = model.architecture_config()
    dummy_hw = args.dummy_hw or (params["minimum_height_width"] if params else 128)
    dummy_input_shape = (args.batch_size, arch["input_channels"], dummy_hw, dummy_hw)
    print(f"Using dummy input shape: {dummy_input_shape}")

    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visualize_model(model, out_dir / "model_trace.pt", dummy_input_shape, device)


if __name__ == "__main__":
    main()

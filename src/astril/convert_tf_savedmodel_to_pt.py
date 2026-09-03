import argparse
import configparser
from pathlib import Path

import numpy as np
import torch

from .model_architecture import TFDynamicAttentionResUNet


def _import_tensorflow():
    try:
        import tensorflow as tf
        if not hasattr(tf, "keras"):
            raise ImportError("import tensorflow succeeded, but tf.keras is unavailable")
        return tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required to read legacy SavedModel weights. "
            "The active Python environment cannot import a complete TensorFlow package."
        ) from exc


def _legacy_tf_custom_objects(tf):
    Conv2D = tf.keras.layers.Conv2D
    SeparableConv2D = tf.keras.layers.SeparableConv2D
    MaxPooling2D = tf.keras.layers.MaxPooling2D
    UpSampling2D = tf.keras.layers.UpSampling2D
    Add = tf.keras.layers.Add
    Multiply = tf.keras.layers.Multiply
    Activation = tf.keras.layers.Activation
    Concatenate = tf.keras.layers.Concatenate
    Model = tf.keras.Model

    class ResidualConvBlock(tf.keras.layers.Layer):
        def __init__(self, num_filters, input_channels=None, name="ResidualConvBlock", **kwargs):
            super().__init__(name=name, **kwargs)
            self.num_filters = num_filters
            self.input_channels = input_channels
            self.adjust_channels = input_channels is not None and input_channels != num_filters
            if self.adjust_channels:
                self.channel_adjust_conv = Conv2D(
                    num_filters, kernel_size=1, padding="same", use_bias=False,
                    name="RCB_channel_adjust_conv",
                )
            self.conv1 = SeparableConv2D(
                num_filters, kernel_size=3, padding="same", activation="relu",
                name="RCB_conv1",
            )
            self.conv2 = SeparableConv2D(
                num_filters, kernel_size=3, padding="same", activation="relu",
                name="RCB_conv2",
            )
            self.add = Add(name="RCB_add")

        def call(self, x):
            skip = self.channel_adjust_conv(x) if self.adjust_channels else x
            x = self.conv1(x)
            x = self.conv2(x)
            return self.add([x, skip])

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"num_filters": self.num_filters, "input_channels": self.input_channels})
            return cfg

    class AttentionBlock(tf.keras.layers.Layer):
        def __init__(self, F_g, F_l, F_int, name="AttentionBlock", **kwargs):
            super().__init__(name=name, **kwargs)
            self.F_g = F_g
            self.F_l = F_l
            self.F_int = F_int
            self.W_g = Conv2D(
                F_int, kernel_size=1, strides=1, padding="same", activation="relu",
                name="AB_W_g",
            )
            self.W_x = Conv2D(
                F_int, kernel_size=1, strides=1, padding="same", activation="relu",
                name="AB_W_x",
            )
            self.psi = Conv2D(
                1, kernel_size=1, strides=1, padding="same", activation="sigmoid",
                name="AB_psi",
            )
            self.relu = Activation("relu", name="AB_relu")
            self.add = Add(name="AB_add")
            self.multiply = Multiply(name="AB_multiply")

        def call(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(self.add([g1, x1]))
            psi = self.psi(psi)
            return self.multiply([x, psi])

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"F_g": self.F_g, "F_l": self.F_l, "F_int": self.F_int})
            return cfg

    class DynamicAttentionResUNet(Model):
        def __init__(
            self,
            input_channels,
            base_num_filters=32,
            encoder_level_factors=(1, 2, 4, 8),
            num_output_slices=1,
            out_channels=4,
            center_depth=1,
            name="DynamicAttentionResUNet",
            **kwargs,
        ):
            super().__init__(name=name, **kwargs)
            self.input_channels = input_channels
            self.base_num_filters = base_num_filters
            self.encoder_level_factors = list(encoder_level_factors)
            self.num_output_slices = num_output_slices
            self.out_channels = out_channels
            self.center_depth = center_depth
            self.num_encoder_levels = len(self.encoder_level_factors)

            self.encoders = []
            self.pools = []
            prev_channels = input_channels
            for factor in self.encoder_level_factors:
                nf = base_num_filters * factor
                self.encoders.append(ResidualConvBlock(nf, input_channels=prev_channels))
                self.pools.append(MaxPooling2D(pool_size=2, padding="same"))
                prev_channels = nf

            center_filters = base_num_filters * self.encoder_level_factors[-1] * 2
            self.center_blocks = []
            in_channels = prev_channels
            for _ in range(center_depth):
                self.center_blocks.append(ResidualConvBlock(center_filters, input_channels=in_channels))
                in_channels = center_filters

            self.upsamples = []
            self.attention_blocks = []
            self.decoders = []
            self.concat_layers = []
            prev_dec_channels = center_filters
            for factor in reversed(self.encoder_level_factors):
                self.upsamples.append(UpSampling2D(size=(2, 2)))
                skip_filters = base_num_filters * factor
                self.attention_blocks.append(
                    AttentionBlock(
                        F_g=prev_dec_channels,
                        F_l=skip_filters,
                        F_int=skip_filters // 2,
                    )
                )
                self.concat_layers.append(Concatenate())
                self.decoders.append(
                    ResidualConvBlock(
                        skip_filters,
                        input_channels=prev_dec_channels + skip_filters,
                    )
                )
                prev_dec_channels = skip_filters

            self.final_conv = Conv2D(
                filters=self.num_output_slices * self.out_channels,
                kernel_size=1,
                activation="softmax",
                name="Final_Conv",
            )

        def call(self, x):
            skips = []
            e_in = x
            for encoder, pool in zip(self.encoders, self.pools):
                e_out = encoder(e_in)
                skips.append(e_out)
                e_in = pool(e_out)
            d_in = e_in
            for block in self.center_blocks:
                d_in = block(d_in)
            for i in range(self.num_encoder_levels):
                d_up = self.upsamples[i](d_in)
                skip = skips[self.num_encoder_levels - 1 - i]
                att = self.attention_blocks[i](d_up, skip)
                d_in = self.decoders[i](self.concat_layers[i]([d_up, att]))
            out = self.final_conv(d_in)
            if self.num_output_slices >= 1:
                shape = tf.shape(out)
                out = tf.reshape(
                    out,
                    (shape[0], shape[1], shape[2], self.num_output_slices, self.out_channels),
                )
            return out

        def get_config(self):
            cfg = super().get_config()
            cfg.update(
                {
                    "input_channels": self.input_channels,
                    "base_num_filters": self.base_num_filters,
                    "encoder_level_factors": self.encoder_level_factors,
                    "num_output_slices": self.num_output_slices,
                    "out_channels": self.out_channels,
                    "center_depth": self.center_depth,
                }
            )
            return cfg

    return {
        "ResidualConvBlock": ResidualConvBlock,
        "AttentionBlock": AttentionBlock,
        "DynamicAttentionResUNet": DynamicAttentionResUNet,
    }


def _read_train_config(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    d = cfg["DEFAULT"]
    image_paths = [x for x in d.get("image_paths_files", "").split(",") if x.strip()]
    channel_names = [x.strip() for x in d.get("channel_names", "").split(",") if x.strip()]
    if not channel_names:
        channel_names = [f"ch{i}" for i in range(len(image_paths))]
    return {
        "num_channels": len(image_paths),
        "channel_names": channel_names,
        "num_input_slices": d.getint("num_input_slices"),
        "num_output_slices": d.getint("num_output_slices"),
        "num_classes": d.getint("num_classes"),
        "base_num_filters": d.getint("base_num_filters", fallback=32),
        "center_depth": d.getint("center_depth", fallback=1),
        "encoder_level_factors": [
            int(x.strip())
            for x in d.get("encoder_level_factors", "1,2,4,8").split(",")
            if x.strip()
        ],
    }


def _copy_conv2d(tf_layer, torch_layer):
    kernel, bias = tf_layer.get_weights()
    torch_layer.weight.data.copy_(torch.from_numpy(np.transpose(kernel, (3, 2, 0, 1))))
    torch_layer.bias.data.copy_(torch.from_numpy(bias))


def _copy_conv2d_no_bias(tf_layer, torch_layer):
    (kernel,) = tf_layer.get_weights()
    torch_layer.weight.data.copy_(torch.from_numpy(np.transpose(kernel, (3, 2, 0, 1))))


def _copy_separable(tf_layer, torch_layer):
    depthwise, pointwise, bias = tf_layer.get_weights()
    if depthwise.shape[3] != 1:
        raise ValueError(f"Only depth_multiplier=1 is supported, got {depthwise.shape}.")
    torch_layer.depthwise.weight.data.copy_(
        torch.from_numpy(np.transpose(depthwise[:, :, :, 0], (2, 0, 1))[:, None, :, :])
    )
    torch_layer.pointwise.weight.data.copy_(torch.from_numpy(np.transpose(pointwise, (3, 2, 0, 1))))
    torch_layer.pointwise.bias.data.copy_(torch.from_numpy(bias))


def _copy_residual_block(tf_block, torch_block):
    if torch_block.channel_adjust_conv is not None:
        _copy_conv2d_no_bias(tf_block.channel_adjust_conv, torch_block.channel_adjust_conv)
    _copy_separable(tf_block.conv1, torch_block.conv1)
    _copy_separable(tf_block.conv2, torch_block.conv2)


def _copy_attention_block(tf_block, torch_block):
    _copy_conv2d(tf_block.W_g, torch_block.W_g)
    _copy_conv2d(tf_block.W_x, torch_block.W_x)
    _copy_conv2d(tf_block.psi, torch_block.psi)


def convert(saved_model_dir, train_config, output_checkpoint):
    tf = _import_tensorflow()
    params = _read_train_config(train_config)
    custom_objects = _legacy_tf_custom_objects(tf)
    tf_model = tf.keras.models.load_model(saved_model_dir, custom_objects=custom_objects)

    model = TFDynamicAttentionResUNet(
        input_channels=params["num_channels"] * params["num_input_slices"],
        base_num_filters=params["base_num_filters"],
        encoder_level_factors=params["encoder_level_factors"],
        num_output_slices=params["num_output_slices"],
        out_channels=params["num_classes"],
        center_depth=params["center_depth"],
    )

    for tf_block, torch_block in zip(tf_model.encoders, model.encoders):
        _copy_residual_block(tf_block, torch_block)
    for tf_block, torch_block in zip(tf_model.center_blocks, model.center_blocks):
        _copy_residual_block(tf_block, torch_block)
    for tf_block, torch_block in zip(tf_model.attention_blocks, model.attention_blocks):
        _copy_attention_block(tf_block, torch_block)
    for tf_block, torch_block in zip(tf_model.decoders, model.decoders):
        _copy_residual_block(tf_block, torch_block)
    _copy_conv2d(tf_model.final_conv, model.final_conv)

    output_checkpoint = Path(output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 0,
            "architecture": model.architecture_config(),
            "channel_metadata": {
                "channel_names": params["channel_names"],
                "optional_channels": [],
                "channel_dropout_probabilities": {},
            },
            "model_state_dict": model.state_dict(),
        },
        output_checkpoint,
    )
    return output_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Convert legacy TensorFlow astril SavedModel to PyTorch .pt.")
    parser.add_argument("--saved_model_dir", required=True, help="Legacy TensorFlow SavedModel directory.")
    parser.add_argument("--train_config", required=True, help="Matching legacy train_parameters.cfg.")
    parser.add_argument("--output_checkpoint", required=True, help="Output PyTorch .pt checkpoint path.")
    args = parser.parse_args()
    out = convert(args.saved_model_dir, args.train_config, args.output_checkpoint)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

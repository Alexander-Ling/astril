import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualConvBlock(nn.Module):
    def __init__(self, num_filters, input_channels=None, use_se=False, se_reduction=4):
        super().__init__()
        if input_channels is None:
            input_channels = num_filters
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.use_se = use_se
        self.se_reduction = se_reduction

        self.channel_adjust_conv = None
        if input_channels != num_filters:
            self.channel_adjust_conv = nn.Conv2d(
                input_channels, num_filters, kernel_size=1, bias=False
            )

        self.conv1 = SeparableConv2d(input_channels, num_filters)
        self.conv2 = SeparableConv2d(num_filters, num_filters)

        if use_se:
            reduced = max(1, num_filters // se_reduction)
            self.se_fc1 = nn.Linear(num_filters, reduced)
            self.se_fc2 = nn.Linear(reduced, num_filters)

    def forward(self, x):
        skip = self.channel_adjust_conv(x) if self.channel_adjust_conv is not None else x
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        if self.use_se:
            se = F.adaptive_avg_pool2d(x, 1).flatten(1)
            se = F.relu(self.se_fc1(se), inplace=True)
            se = torch.sigmoid(self.se_fc2(se)).view(x.shape[0], self.num_filters, 1, 1)
            x = x * se

        return x + skip


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)

    def forward(self, g, x):
        psi = F.relu(self.W_g(g) + self.W_x(x), inplace=True)
        psi = torch.sigmoid(self.psi(psi))
        return x * psi


class TFSeparableConv2d(nn.Module):
    """PyTorch equivalent of Keras SeparableConv2D with use_bias=True."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class TFResidualConvBlock(nn.Module):
    """Residual block matching the legacy TensorFlow astril implementation."""

    def __init__(self, num_filters, input_channels=None):
        super().__init__()
        if input_channels is None:
            input_channels = num_filters
        self.num_filters = num_filters
        self.input_channels = input_channels

        self.channel_adjust_conv = None
        if input_channels != num_filters:
            self.channel_adjust_conv = nn.Conv2d(
                input_channels, num_filters, kernel_size=1, bias=False
            )

        self.conv1 = TFSeparableConv2d(input_channels, num_filters)
        self.conv2 = TFSeparableConv2d(num_filters, num_filters)

    def forward(self, x):
        skip = self.channel_adjust_conv(x) if self.channel_adjust_conv is not None else x
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x + skip


class TFAttentionBlock(nn.Module):
    """Attention block matching legacy TF Conv2D(activation='relu') behavior."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)

    def forward(self, g, x):
        g1 = F.relu(self.W_g(g), inplace=True)
        x1 = F.relu(self.W_x(x), inplace=True)
        psi = F.relu(g1 + x1, inplace=True)
        psi = torch.sigmoid(self.psi(psi))
        return x * psi


class TFDynamicAttentionResUNet(nn.Module):
    """
    PyTorch target for legacy TensorFlow astril DynamicAttentionResUNet weights.

    This keeps the same output/logit contract as current PyTorch training while
    matching the old TF layer math before the final softmax.
    """

    def __init__(
        self,
        input_channels,
        base_num_filters=32,
        encoder_level_factors=None,
        num_output_slices=1,
        out_channels=4,
        center_depth=1,
        **_,
    ):
        nn.Module.__init__(self)
        if encoder_level_factors is None:
            encoder_level_factors = [1, 2, 4, 8]

        self.input_channels = input_channels
        self.base_num_filters = base_num_filters
        self.encoder_level_factors = list(encoder_level_factors)
        self.num_output_slices = num_output_slices
        self.out_channels = out_channels
        self.center_depth = center_depth
        self.use_se_blocks = False
        self.use_deep_supervision = False
        self.num_encoder_levels = len(self.encoder_level_factors)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = input_channels
        for factor in self.encoder_level_factors:
            nf = base_num_filters * factor
            self.encoders.append(TFResidualConvBlock(nf, input_channels=prev_channels))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            prev_channels = nf

        center_filters = base_num_filters * self.encoder_level_factors[-1] * 2
        self.center_blocks = nn.ModuleList()
        in_channels = prev_channels
        for _ in range(center_depth):
            self.center_blocks.append(TFResidualConvBlock(center_filters, input_channels=in_channels))
            in_channels = center_filters

        self.attention_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_dec_channels = center_filters
        for factor in reversed(self.encoder_level_factors):
            skip_filters = base_num_filters * factor
            self.attention_blocks.append(
                TFAttentionBlock(
                    F_g=prev_dec_channels,
                    F_l=skip_filters,
                    F_int=max(1, skip_filters // 2),
                )
            )
            self.decoders.append(
                TFResidualConvBlock(
                    skip_filters,
                    input_channels=prev_dec_channels + skip_filters,
                )
            )
            prev_dec_channels = skip_filters

        final_channels = self.num_output_slices * self.out_channels
        self.final_conv = nn.Conv2d(prev_dec_channels, final_channels, kernel_size=1)
        self.aux_heads = nn.ModuleList()

    def architecture_config(self):
        return {
            "architecture_type": "tf_dynamic_attention_resunet",
            "input_channels": self.input_channels,
            "base_num_filters": self.base_num_filters,
            "encoder_level_factors": self.encoder_level_factors,
            "num_output_slices": self.num_output_slices,
            "out_channels": self.out_channels,
            "center_depth": self.center_depth,
            "use_se_blocks": False,
            "use_deep_supervision": False,
        }

    def _format_output(self, logits):
        n, _, h, w = logits.shape
        logits = logits.view(n, self.num_output_slices, self.out_channels, h, w)
        return logits.permute(0, 3, 4, 1, 2).contiguous()

    def forward(self, x):
        skip_connections = []
        e_in = x
        for encoder, pool in zip(self.encoders, self.pools):
            e_out = encoder(e_in)
            skip_connections.append(e_out)
            e_in = pool(e_out)

        d_in = e_in
        for block in self.center_blocks:
            d_in = block(d_in)

        for i in range(self.num_encoder_levels):
            skip = skip_connections[self.num_encoder_levels - 1 - i]
            d_up = F.interpolate(d_in, size=skip.shape[-2:], mode="nearest")
            att = self.attention_blocks[i](d_up, skip)
            d_in = self.decoders[i](torch.cat([d_up, att], dim=1))

        return self._format_output(self.final_conv(d_in))


class DynamicAttentionResUNet(nn.Module):
    """
    PyTorch implementation of astril's 2.5D Attention ResUNet.

    Inputs are channels-first tensors: (N, C, H, W).
    Outputs preserve astril's historical semantic layout:
    (N, H, W, num_output_slices, out_channels).
    """

    def __init__(
        self,
        input_channels,
        base_num_filters=32,
        encoder_level_factors=None,
        num_output_slices=1,
        out_channels=4,
        center_depth=1,
        use_se_blocks=False,
        use_deep_supervision=False,
        **_,
    ):
        super().__init__()
        if encoder_level_factors is None:
            encoder_level_factors = [1, 2, 4, 8]

        self.input_channels = input_channels
        self.base_num_filters = base_num_filters
        self.encoder_level_factors = list(encoder_level_factors)
        self.num_output_slices = num_output_slices
        self.out_channels = out_channels
        self.center_depth = center_depth
        self.use_se_blocks = use_se_blocks
        self.use_deep_supervision = use_deep_supervision
        self.num_encoder_levels = len(self.encoder_level_factors)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = input_channels
        for factor in self.encoder_level_factors:
            nf = base_num_filters * factor
            self.encoders.append(
                ResidualConvBlock(nf, input_channels=prev_channels, use_se=use_se_blocks)
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            prev_channels = nf

        center_filters = base_num_filters * self.encoder_level_factors[-1] * 2
        self.center_blocks = nn.ModuleList()
        in_channels = prev_channels
        for _ in range(center_depth):
            self.center_blocks.append(
                ResidualConvBlock(
                    center_filters,
                    input_channels=in_channels,
                    use_se=use_se_blocks,
                )
            )
            in_channels = center_filters

        self.attention_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_channels = []
        reversed_factors = list(reversed(self.encoder_level_factors))
        prev_dec_channels = center_filters
        for factor in reversed_factors:
            skip_filters = base_num_filters * factor
            self.attention_blocks.append(
                AttentionBlock(
                    F_g=prev_dec_channels,
                    F_l=skip_filters,
                    F_int=max(1, skip_filters // 2),
                )
            )
            self.decoders.append(
                ResidualConvBlock(
                    skip_filters,
                    input_channels=prev_dec_channels + skip_filters,
                    use_se=use_se_blocks,
                )
            )
            decoder_channels.append(skip_filters)
            prev_dec_channels = skip_filters

        final_channels = self.num_output_slices * self.out_channels
        self.final_conv = nn.Conv2d(prev_dec_channels, final_channels, kernel_size=1)
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList(
                [
                    nn.Conv2d(in_ch, final_channels, kernel_size=1)
                    for in_ch in reversed(decoder_channels[-min(2, self.num_encoder_levels):])
                ]
            )
        else:
            self.aux_heads = nn.ModuleList()

    def architecture_config(self):
        return {
            "input_channels": self.input_channels,
            "base_num_filters": self.base_num_filters,
            "encoder_level_factors": self.encoder_level_factors,
            "num_output_slices": self.num_output_slices,
            "out_channels": self.out_channels,
            "center_depth": self.center_depth,
            "use_se_blocks": self.use_se_blocks,
            "use_deep_supervision": self.use_deep_supervision,
        }

    def _format_output(self, logits):
        n, _, h, w = logits.shape
        logits = logits.view(n, self.num_output_slices, self.out_channels, h, w)
        return logits.permute(0, 3, 4, 1, 2).contiguous()

    def forward(self, x):
        skip_connections = []
        e_in = x
        for encoder, pool in zip(self.encoders, self.pools):
            e_out = encoder(e_in)
            skip_connections.append(e_out)
            e_in = pool(e_out)

        d_in = e_in
        for block in self.center_blocks:
            d_in = block(d_in)

        decoder_outputs = []
        for i in range(self.num_encoder_levels):
            skip = skip_connections[self.num_encoder_levels - 1 - i]
            d_up = F.interpolate(d_in, size=skip.shape[-2:], mode="nearest")
            att = self.attention_blocks[i](d_up, skip)
            d_out = self.decoders[i](torch.cat([d_up, att], dim=1))
            decoder_outputs.append(d_out)
            d_in = d_out

        main_out = self._format_output(self.final_conv(d_in))

        if self.use_deep_supervision and self.training:
            aux_outputs = []
            for i, aux_head in enumerate(self.aux_heads):
                aux = aux_head(decoder_outputs[-(i + 1)])
                if aux.shape[-2:] != d_in.shape[-2:]:
                    aux = F.interpolate(aux, size=d_in.shape[-2:], mode="nearest")
                aux_outputs.append(self._format_output(aux))
            return main_out, aux_outputs

        return main_out


class BrainIACEncoderFusionUNet(DynamicAttentionResUNet):
    """2.5D U-Net with frozen BrainIAC patch-token features fused at encoder level 2."""

    def __init__(
        self,
        input_channels,
        brainiac_input_channels,
        base_num_filters=32,
        encoder_level_factors=None,
        num_output_slices=1,
        out_channels=4,
        center_depth=1,
        use_se_blocks=False,
        use_deep_supervision=False,
        brainiac_fusion_level=1,
        **_,
    ):
        if encoder_level_factors is None:
            encoder_level_factors = [1, 2, 4, 8]
        if int(brainiac_fusion_level) != 1:
            raise ValueError("BrainIAC encoder fusion v1 supports only fusion level 1 (second encoder).")
        if int(brainiac_input_channels) <= 0:
            raise ValueError("brainiac_input_channels must be positive for BrainIAC encoder fusion.")

        nn.Module.__init__(self)
        self.input_channels = input_channels
        self.brainiac_input_channels = int(brainiac_input_channels)
        self.base_num_filters = base_num_filters
        self.encoder_level_factors = list(encoder_level_factors)
        self.num_output_slices = num_output_slices
        self.out_channels = out_channels
        self.center_depth = center_depth
        self.use_se_blocks = use_se_blocks
        self.use_deep_supervision = use_deep_supervision
        self.brainiac_fusion_level = int(brainiac_fusion_level)
        self.num_encoder_levels = len(self.encoder_level_factors)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = input_channels
        for level, factor in enumerate(self.encoder_level_factors):
            nf = base_num_filters * factor
            encoder_input_channels = prev_channels
            if level == self.brainiac_fusion_level:
                encoder_input_channels += nf
                self.brainiac_projection = nn.Sequential(
                    nn.Conv2d(self.brainiac_input_channels, nf, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                )
                self.brainiac_fusion_mix = nn.Conv2d(encoder_input_channels, encoder_input_channels, kernel_size=1)
            self.encoders.append(
                ResidualConvBlock(nf, input_channels=encoder_input_channels, use_se=use_se_blocks)
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            prev_channels = nf

        center_filters = base_num_filters * self.encoder_level_factors[-1] * 2
        self.center_blocks = nn.ModuleList()
        in_channels = prev_channels
        for _ in range(center_depth):
            self.center_blocks.append(
                ResidualConvBlock(
                    center_filters,
                    input_channels=in_channels,
                    use_se=use_se_blocks,
                )
            )
            in_channels = center_filters

        self.attention_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_channels = []
        reversed_factors = list(reversed(self.encoder_level_factors))
        prev_dec_channels = center_filters
        for factor in reversed_factors:
            skip_filters = base_num_filters * factor
            self.attention_blocks.append(
                AttentionBlock(
                    F_g=prev_dec_channels,
                    F_l=skip_filters,
                    F_int=max(1, skip_filters // 2),
                )
            )
            self.decoders.append(
                ResidualConvBlock(
                    skip_filters,
                    input_channels=prev_dec_channels + skip_filters,
                    use_se=use_se_blocks,
                )
            )
            decoder_channels.append(skip_filters)
            prev_dec_channels = skip_filters

        final_channels = self.num_output_slices * self.out_channels
        self.final_conv = nn.Conv2d(prev_dec_channels, final_channels, kernel_size=1)
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList(
                [
                    nn.Conv2d(in_ch, final_channels, kernel_size=1)
                    for in_ch in reversed(decoder_channels[-min(2, self.num_encoder_levels):])
                ]
            )
        else:
            self.aux_heads = nn.ModuleList()

    def architecture_config(self):
        cfg = super().architecture_config()
        cfg.update(
            {
                "architecture_type": "brainiac_encoder_fusion",
                "brainiac_input_channels": self.brainiac_input_channels,
                "brainiac_fusion_level": self.brainiac_fusion_level,
            }
        )
        return cfg

    def forward(self, x, brainiac_features=None):
        if isinstance(x, (tuple, list)):
            x, brainiac_features = x
        if brainiac_features is None:
            raise ValueError("BrainIAC encoder-fusion model requires brainiac_features.")

        skip_connections = []
        e_in = x
        for level, (encoder, pool) in enumerate(zip(self.encoders, self.pools)):
            if level == self.brainiac_fusion_level:
                b = self.brainiac_projection(brainiac_features)
                b = F.interpolate(b, size=e_in.shape[-2:], mode="bilinear", align_corners=False)
                e_in = self.brainiac_fusion_mix(torch.cat([e_in, b], dim=1))
            e_out = encoder(e_in)
            skip_connections.append(e_out)
            e_in = pool(e_out)

        d_in = e_in
        for block in self.center_blocks:
            d_in = block(d_in)

        decoder_outputs = []
        for i in range(self.num_encoder_levels):
            skip = skip_connections[self.num_encoder_levels - 1 - i]
            d_up = F.interpolate(d_in, size=skip.shape[-2:], mode="nearest")
            att = self.attention_blocks[i](d_up, skip)
            d_out = self.decoders[i](torch.cat([d_up, att], dim=1))
            decoder_outputs.append(d_out)
            d_in = d_out

        main_out = self._format_output(self.final_conv(d_in))
        if self.use_deep_supervision and self.training:
            aux_outputs = []
            for i, aux_head in enumerate(self.aux_heads):
                aux = aux_head(decoder_outputs[-(i + 1)])
                if aux.shape[-2:] != d_in.shape[-2:]:
                    aux = F.interpolate(aux, size=d_in.shape[-2:], mode="nearest")
                aux_outputs.append(self._format_output(aux))
            return main_out, aux_outputs

        return main_out


def create_dynamic_unet_from_config():
    from .config import (
        architecture_type,
        num_channels,
        num_input_slices,
        num_output_slices,
        num_classes,
        base_num_filters,
        encoder_level_factors,
        center_depth,
        use_se_blocks,
        use_deep_supervision,
        use_brainiac_embeddings,
        brainiac_embedding_type,
        brainiac_encoder_input_channels,
    )

    if use_brainiac_embeddings and brainiac_embedding_type != "encoder_fusion":
        raise ValueError(
            "BrainIAC now supports only brainiac_embedding_type = encoder_fusion. "
            f"Found: {brainiac_embedding_type}"
        )

    if use_brainiac_embeddings:
        return BrainIACEncoderFusionUNet(
            input_channels=num_channels * num_input_slices,
            brainiac_input_channels=brainiac_encoder_input_channels,
            base_num_filters=base_num_filters,
            encoder_level_factors=encoder_level_factors,
            num_output_slices=num_output_slices,
            out_channels=num_classes,
            center_depth=center_depth,
            use_se_blocks=use_se_blocks,
            use_deep_supervision=use_deep_supervision,
        )

    if architecture_type == "tf_dynamic_attention_resunet":
        return TFDynamicAttentionResUNet(
            input_channels=num_channels * num_input_slices,
            base_num_filters=base_num_filters,
            encoder_level_factors=encoder_level_factors,
            num_output_slices=num_output_slices,
            out_channels=num_classes,
            center_depth=center_depth,
        )

    if architecture_type not in {"dynamic_attention_resunet", ""}:
        raise ValueError(f"Unsupported architecture_type: {architecture_type}")

    return DynamicAttentionResUNet(
        input_channels=num_channels * num_input_slices,
        base_num_filters=base_num_filters,
        encoder_level_factors=encoder_level_factors,
        num_output_slices=num_output_slices,
        out_channels=num_classes,
        center_depth=center_depth,
        use_se_blocks=use_se_blocks,
        use_deep_supervision=use_deep_supervision,
    )


def create_dynamic_unet_from_metadata(metadata: dict):
    metadata = dict(metadata)
    architecture_type = metadata.pop("architecture_type", "dynamic_attention_resunet")
    if architecture_type == "brainiac_encoder_fusion":
        return BrainIACEncoderFusionUNet(**metadata)
    if architecture_type == "tf_dynamic_attention_resunet":
        return TFDynamicAttentionResUNet(**metadata)
    if architecture_type not in {"dynamic_attention_resunet", ""}:
        raise ValueError(f"Unsupported architecture_type: {architecture_type}")
    return DynamicAttentionResUNet(**metadata)

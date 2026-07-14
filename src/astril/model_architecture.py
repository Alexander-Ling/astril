import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib


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


class DinoV3EncoderFusionUNet(DynamicAttentionResUNet):
    """
    2.5D U-Net with DINOv3 ViT features fused at multiple encoder levels
    via DPT-style projection heads (Dense Prediction Transformer).

    DINOv3 runs online during the forward pass on the center MRI slice only.
    The full multi-slice input flows through the U-Net encoder unchanged.

    Supported loading modes:
      - torch.hub from a local DINOv3 clone: set dinov3_hub_repo + dinov3_weights
      - HuggingFace transformers: set dinov3_hf_model_id

    dinov3_fusion_levels  : encoder level indices (0-based) where DINOv3 features are injected
    dinov3_hook_blocks    : DINOv3 ViT block indices whose outputs are captured (one per fusion level)
    dinov3_num_input_channels : number of MRI modalities to project to 3 channels (defaults to all)
    dinov3_frozen         : if True, DINOv3 weights are frozen and forward runs under no_grad
    """

    _EMBED_DIM = {
        "dinov3_vits16": 384,
        "dinov3_vitb16": 768,
        "dinov3_vitl16": 1024,
        "dinov3_vith16": 1280,
    }
    _PATCH_SIZE = 16
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        input_channels,
        dinov3_num_input_channels,
        base_num_filters=32,
        encoder_level_factors=None,
        num_output_slices=1,
        out_channels=4,
        center_depth=1,
        use_se_blocks=False,
        use_deep_supervision=False,
        dinov3_model_name="dinov3_vitb16",
        dinov3_hub_repo=None,
        dinov3_weights=None,
        dinov3_hf_model_id=None,
        dinov3_fusion_levels=None,
        dinov3_hook_blocks=None,
        dinov3_frozen=True,
        **_,
    ):
        if encoder_level_factors is None:
            encoder_level_factors = [1, 2, 4, 8]

        n_levels = len(encoder_level_factors)
        # Default: fuse at all levels except level 0 (full resolution)
        if dinov3_fusion_levels is None:
            dinov3_fusion_levels = list(range(1, n_levels))
        # Default hook blocks for 12-block ViT-S/B; evenly spaced across the model
        if dinov3_hook_blocks is None:
            n_fuse = len(dinov3_fusion_levels)
            # [2, 5, 8, 11] for 4 levels on a 12-block model
            dinov3_hook_blocks = [
                int(round((i + 1) * 11 / n_fuse)) for i in range(n_fuse)
            ]

        if len(dinov3_fusion_levels) != len(dinov3_hook_blocks):
            raise ValueError(
                f"dinov3_fusion_levels and dinov3_hook_blocks must have the same length; "
                f"got {len(dinov3_fusion_levels)} and {len(dinov3_hook_blocks)}."
            )
        if any(lvl >= n_levels for lvl in dinov3_fusion_levels):
            raise ValueError(
                f"All dinov3_fusion_levels must be < len(encoder_level_factors)={n_levels}."
            )

        nn.Module.__init__(self)

        self.input_channels = input_channels
        self.dinov3_num_input_channels = int(dinov3_num_input_channels)
        self.base_num_filters = base_num_filters
        self.encoder_level_factors = list(encoder_level_factors)
        self.num_output_slices = num_output_slices
        self.out_channels = out_channels
        self.center_depth = center_depth
        self.use_se_blocks = use_se_blocks
        self.use_deep_supervision = use_deep_supervision
        self.dinov3_model_name = dinov3_model_name
        self.dinov3_hub_repo = dinov3_hub_repo
        self.dinov3_weights = dinov3_weights
        self.dinov3_hf_model_id = dinov3_hf_model_id
        self.dinov3_fusion_levels = list(dinov3_fusion_levels)
        self.dinov3_hook_blocks = list(dinov3_hook_blocks)
        self.dinov3_frozen = dinov3_frozen
        self.num_encoder_levels = n_levels

        embed_dim = self._EMBED_DIM.get(dinov3_model_name, 768)

        # --- DINOv3 backbone ---
        self.dinov3 = self._load_dinov3(
            dinov3_model_name, dinov3_hub_repo, dinov3_weights, dinov3_hf_model_id
        )
        if dinov3_frozen:
            for p in self.dinov3.parameters():
                p.requires_grad_(False)

        # Hook storage (populated during forward; keyed by block index)
        self._dinov3_hook_outputs: dict = {}
        self._dinov3_hooks: list = []
        self._register_dinov3_hooks()

        # ImageNet normalization constants as persistent buffers
        self.register_buffer(
            "dinov3_mean",
            torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "dinov3_std",
            torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1),
        )

        # 1×1 conv: MRI center-slice channels → 3 channels for DINOv3
        self.dinov3_input_proj = nn.Conv2d(
            self.dinov3_num_input_channels, 3, kernel_size=1, bias=True
        )

        # Per-fusion-level DPT projection heads: embed_dim → encoder channel count
        self.dinov3_proj_heads = nn.ModuleList()
        for lvl in self.dinov3_fusion_levels:
            enc_ch = base_num_filters * encoder_level_factors[lvl]
            self.dinov3_proj_heads.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, enc_ch, kernel_size=1, bias=True),
                    nn.GELU(),
                    nn.Conv2d(enc_ch, enc_ch, kernel_size=1, bias=True),
                )
            )

        # Per-fusion-level mix convs: concat(enc, dino) → enc channels
        self.dinov3_fusion_convs = nn.ModuleList()
        for lvl in self.dinov3_fusion_levels:
            enc_ch = base_num_filters * encoder_level_factors[lvl]
            self.dinov3_fusion_convs.append(
                nn.Conv2d(enc_ch * 2, enc_ch, kernel_size=1, bias=True)
            )

        # --- U-Net encoder (identical structure to DynamicAttentionResUNet) ---
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
                ResidualConvBlock(center_filters, input_channels=in_channels, use_se=use_se_blocks)
            )
            in_channels = center_filters

        self.attention_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_channels = []
        prev_dec_channels = center_filters
        for factor in reversed(self.encoder_level_factors):
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
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(in_ch, final_channels, kernel_size=1)
                for in_ch in reversed(decoder_channels[-min(2, self.num_encoder_levels):])
            ])
        else:
            self.aux_heads = nn.ModuleList()

    # ------------------------------------------------------------------
    # DINOv3 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dinov3(model_name, hub_repo, weights, hf_model_id):
        if hub_repo is not None:
            return torch.hub.load(
                hub_repo,
                model_name,
                source="local",
                weights=weights,
            )
        if hf_model_id is not None:
            from transformers import AutoModel
            return AutoModel.from_pretrained(hf_model_id)
        raise ValueError(
            "Either dinov3_hub_repo (for torch.hub local clone) or "
            "dinov3_hf_model_id (for HuggingFace) must be provided."
        )

    def _register_dinov3_hooks(self):
        for h in self._dinov3_hooks:
            h.remove()
        self._dinov3_hooks = []

        # torch.hub DINOv3/DINOv2: model.blocks
        # HuggingFace DINOv2: model.encoder.layer
        # HuggingFace DINOv3: model.model.layer
        blocks = getattr(self.dinov3, "blocks", None)
        if blocks is None:
            for enc_attr in ("encoder", "model"):
                enc = getattr(self.dinov3, enc_attr, None)
                if enc is not None:
                    blocks = getattr(enc, "layer", None)
                    if blocks is not None:
                        break
        if blocks is None:
            raise ValueError(
                "Cannot locate transformer blocks in the DINOv3 model. "
                "Expected 'model.blocks' (torch.hub), 'model.encoder.layer' "
                "(HuggingFace DINOv2), or 'model.model.layer' (HuggingFace DINOv3)."
            )

        for block_idx in self.dinov3_hook_blocks:
            def _make_hook(idx):
                def _hook(module, input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    self._dinov3_hook_outputs[idx] = out
                return _hook
            self._dinov3_hooks.append(
                blocks[block_idx].register_forward_hook(_make_hook(block_idx))
            )

    def set_dinov3_frozen(self, frozen: bool):
        """Freeze or unfreeze DINOv3 backbone weights. Call before adding params to optimizer."""
        self.dinov3_frozen = frozen
        for p in self.dinov3.parameters():
            p.requires_grad_(not frozen)

    def _extract_dinov3_features(self, center_slice):
        """
        Run DINOv3 on the center MRI slice and return one spatial feature map per
        fusion level (ordered to match self.dinov3_fusion_levels).

        center_slice : (N, dinov3_num_input_channels, H, W)
        returns      : list of (N, embed_dim, H_p, W_p) tensors
        """
        N, C, H, W = center_slice.shape

        # Project MRI channels → 3-channel RGB-like input
        x_proj = self.dinov3_input_proj(center_slice)   # (N, 3, H, W)
        x_proj = (x_proj - self.dinov3_mean) / self.dinov3_std

        # Pad to nearest multiple of patch_size so patch grid is integral
        pad_h = (self._PATCH_SIZE - H % self._PATCH_SIZE) % self._PATCH_SIZE
        pad_w = (self._PATCH_SIZE - W % self._PATCH_SIZE) % self._PATCH_SIZE
        if pad_h > 0 or pad_w > 0:
            x_proj = F.pad(x_proj, (0, pad_w, 0, pad_h))

        H_p = x_proj.shape[-2] // self._PATCH_SIZE
        W_p = x_proj.shape[-1] // self._PATCH_SIZE

        self._dinov3_hook_outputs.clear()
        ctx = torch.no_grad() if self.dinov3_frozen else contextlib.nullcontext()
        with ctx:
            self.dinov3(x_proj)

        # Reshape token sequences → spatial feature maps
        feature_maps = []
        for block_idx in self.dinov3_hook_blocks:
            tokens = self._dinov3_hook_outputs[block_idx]   # (N, n_special+H_p*W_p, D)
            n_special = tokens.shape[1] - H_p * W_p         # CLS + register tokens
            patch_tokens = tokens[:, n_special:, :]
            _N, _L, D = patch_tokens.shape
            feat = patch_tokens.transpose(1, 2).reshape(N, D, H_p, W_p)
            feature_maps.append(feat)

        return feature_maps

    # ------------------------------------------------------------------
    # architecture_config / forward
    # ------------------------------------------------------------------

    def architecture_config(self):
        cfg = super().architecture_config()
        cfg.update({
            "architecture_type": "dinov3_encoder_fusion",
            "dinov3_num_input_channels": self.dinov3_num_input_channels,
            "dinov3_model_name": self.dinov3_model_name,
            "dinov3_hub_repo": self.dinov3_hub_repo,
            "dinov3_weights": self.dinov3_weights,
            "dinov3_hf_model_id": self.dinov3_hf_model_id,
            "dinov3_fusion_levels": self.dinov3_fusion_levels,
            "dinov3_hook_blocks": self.dinov3_hook_blocks,
            "dinov3_frozen": self.dinov3_frozen,
        })
        return cfg

    def forward(self, x):
        N, C_total, H, W = x.shape

        # Extract center-slice channels for DINOv3
        C = self.dinov3_num_input_channels
        S = C_total // C
        center_start = C * (S // 2)
        center = x[:, center_start:center_start + C, :, :]

        # DINOv3 features: one map per fusion level
        dino_features = self._extract_dinov3_features(center)
        level_to_dino = {
            lvl: feat
            for lvl, feat in zip(self.dinov3_fusion_levels, dino_features)
        }
        fusion_level_index = {lvl: i for i, lvl in enumerate(self.dinov3_fusion_levels)}

        # U-Net encoder with DINOv3 fusion after each designated level
        skip_connections = []
        e_in = x
        for level, (encoder, pool) in enumerate(zip(self.encoders, self.pools)):
            e_out = encoder(e_in)
            if level in level_to_dino:
                fi = fusion_level_index[level]
                dino_feat = F.interpolate(
                    level_to_dino[level], size=e_out.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                dino_proj = self.dinov3_proj_heads[fi](dino_feat)
                e_out = self.dinov3_fusion_convs[fi](torch.cat([e_out, dino_proj], dim=1))
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
        use_dinov3_embeddings,
        dinov3_model_name,
        dinov3_hub_repo,
        dinov3_weights,
        dinov3_hf_model_id,
        dinov3_fusion_levels,
        dinov3_hook_blocks,
        dinov3_num_input_channels,
        dinov3_frozen,
    )

    if use_brainiac_embeddings and use_dinov3_embeddings:
        raise ValueError("use_brainiac_embeddings and use_dinov3_embeddings cannot both be True.")

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

    if use_dinov3_embeddings:
        return DinoV3EncoderFusionUNet(
            input_channels=num_channels * num_input_slices,
            dinov3_num_input_channels=dinov3_num_input_channels or num_channels,
            base_num_filters=base_num_filters,
            encoder_level_factors=encoder_level_factors,
            num_output_slices=num_output_slices,
            out_channels=num_classes,
            center_depth=center_depth,
            use_se_blocks=use_se_blocks,
            use_deep_supervision=use_deep_supervision,
            dinov3_model_name=dinov3_model_name,
            dinov3_hub_repo=dinov3_hub_repo,
            dinov3_weights=dinov3_weights,
            dinov3_hf_model_id=dinov3_hf_model_id,
            dinov3_fusion_levels=dinov3_fusion_levels,
            dinov3_hook_blocks=dinov3_hook_blocks,
            dinov3_frozen=dinov3_frozen,
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
    if architecture_type == "dinov3_encoder_fusion":
        return DinoV3EncoderFusionUNet(**metadata)
    if architecture_type == "tf_dynamic_attention_resunet":
        return TFDynamicAttentionResUNet(**metadata)
    if architecture_type not in {"dynamic_attention_resunet", ""}:
        raise ValueError(f"Unsupported architecture_type: {architecture_type}")
    return DynamicAttentionResUNet(**metadata)

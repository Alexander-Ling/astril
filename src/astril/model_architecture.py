import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, MaxPooling2D, UpSampling2D,
    Add, Multiply, Activation, Concatenate,
    GlobalAveragePooling2D, Dense, Reshape
)
from tensorflow.keras.optimizers import Adam

########################################################
# 1) ResidualConvBlock
########################################################
class ResidualConvBlock(tf.keras.layers.Layer):
    """
    Basic residual block using SeparableConv2D, plus optional 1x1
    channel adjust if input != num_filters.
    Optionally includes a Squeeze-and-Excitation (SE) channel attention module.
    """
    def __init__(self, num_filters, input_channels=None, use_se=False, se_reduction=4,
                 name="ResidualConvBlock"):
        super().__init__(name=name)
        self._base_name = self.name if self.name is not None else "ResidualConvBlock"
        self.num_filters = num_filters
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.adjust_channels = (input_channels is not None and input_channels != num_filters)

        if self.adjust_channels:
            self.channel_adjust_conv = Conv2D(
                num_filters, kernel_size=1, padding='same', use_bias=False,
                name="RCB_channel_adjust_conv"
            )

        self.conv1 = SeparableConv2D(
            num_filters, kernel_size=3, padding='same', activation='relu',
            name="RCB_conv1"
        )
        self.conv2 = SeparableConv2D(
            num_filters, kernel_size=3, padding='same', activation='relu',
            name="RCB_conv2"
        )
        self.add = Add(name="RCB_add")

        if use_se:
            reduced = max(1, num_filters // se_reduction)
            self.se_gap = GlobalAveragePooling2D(name="RCB_SE_gap")
            self.se_fc1 = Dense(reduced, activation='relu', name="RCB_SE_fc1")
            self.se_fc2 = Dense(num_filters, activation='sigmoid', name="RCB_SE_fc2")
            self.se_reshape = Reshape((1, 1, num_filters), name="RCB_SE_reshape")
            self.se_multiply = Multiply(name="RCB_SE_multiply")

    def call(self, x):
        if self.adjust_channels:
            skip = self.channel_adjust_conv(x)
        else:
            skip = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.use_se:
            se = self.se_gap(x)
            se = self.se_fc1(se)
            se = self.se_fc2(se)
            se = self.se_reshape(se)
            x = self.se_multiply([x, se])

        out = self.add([x, skip])
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "use_se": self.use_se,
            "se_reduction": self.se_reduction,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


########################################################
# 2) AttentionBlock
########################################################
class AttentionBlock(tf.keras.layers.Layer):
    """
    Standard attention block from Attention U-Net.
    """
    def __init__(self, F_g, F_l, F_int, name="AttentionBlock"):
        super().__init__(name=name)
        self._base_name = self.name if self.name is not None else "AttentionBlock"
        self.W_g = Conv2D(
            F_int, kernel_size=1, strides=1, padding='same', activation='relu',
            name="AB_W_g"
        )
        self.W_x = Conv2D(
            F_int, kernel_size=1, strides=1, padding='same', activation='relu',
            name="AB_W_x"
        )
        self.psi = Conv2D(
            1, kernel_size=1, strides=1, padding='same', activation='sigmoid',
            name="AB_psi"
        )
        self.relu = Activation('relu', name="AB_relu")
        self.add = Add(name="AB_add")
        self.multiply = Multiply(name="AB_multiply")

    def call(self, g, x):
        # g => gating signal, x => skip connection
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(self.add([g1, x1]))
        psi = self.psi(psi)
        return self.multiply([x, psi])


########################################################
# 3) DynamicAttentionResUNet
########################################################
class DynamicAttentionResUNet(Model):
    """
    A U-Net style architecture that builds encoder & decoder
    with dynamic expansions for each level, plus attention blocks.
    Supports:
    - Multiple center blocks via 'center_depth'
    - SE channel attention in residual blocks via 'use_se_blocks'
    - Deep supervision via 'use_deep_supervision' (returns tuple at training time)
    """
    def __init__(
        self,
        input_channels,
        base_num_filters=32,
        encoder_level_factors=[1, 2, 4, 8],
        num_output_slices=1,
        out_channels=4,
        center_depth=1,
        use_se_blocks=False,
        use_deep_supervision=False,
        name="DynamicAttentionResUNet",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.input_channels = input_channels
        self.base_num_filters = base_num_filters
        self.encoder_level_factors = encoder_level_factors
        self.num_output_slices = num_output_slices
        self.out_channels = out_channels
        self.center_depth = center_depth
        self.use_se_blocks = use_se_blocks
        self.use_deep_supervision = use_deep_supervision

        self.num_encoder_levels = len(encoder_level_factors)

        # 3.1. Build Encoder Blocks
        self.encoders = []
        self.pools = []
        prev_channels = input_channels
        for i, factor in enumerate(encoder_level_factors):
            nf = base_num_filters * factor
            block = ResidualConvBlock(
                num_filters=nf, input_channels=prev_channels,
                use_se=use_se_blocks,
                name="Encoder_ResidualConvBlock"
            )
            self.encoders.append(block)
            pool = MaxPooling2D(pool_size=2, padding='same', name="Encoder_MaxPool")
            self.pools.append(pool)
            prev_channels = nf

        # 3.2. Build the "Center" (bottleneck) blocks
        center_filters = base_num_filters * encoder_level_factors[-1] * 2
        self.center_blocks = []
        in_channels = prev_channels
        for i in range(center_depth):
            block = ResidualConvBlock(
                center_filters, input_channels=in_channels,
                use_se=use_se_blocks,
                name="Center_ResidualConvBlock"
            )
            self.center_blocks.append(block)
            in_channels = center_filters

        # 3.3. Build Decoder
        self.upsamples = []
        self.attention_blocks = []
        self.decoders = []
        self.concat_layers = []

        reversed_factors = list(reversed(encoder_level_factors))
        prev_dec_channels = center_filters

        for i, factor in enumerate(reversed_factors):
            up = UpSampling2D(size=(2, 2), name="Decoder_UpSampling")
            self.upsamples.append(up)

            skip_filters = base_num_filters * factor
            att_block = AttentionBlock(
                F_g=prev_dec_channels,
                F_l=skip_filters,
                F_int=skip_filters // 2,
                name="Decoder_AttentionBlock"
            )
            self.attention_blocks.append(att_block)

            concat_layer = Concatenate(name="Decoder_Concat")
            self.concat_layers.append(concat_layer)

            dec_filters = skip_filters
            dec_block = ResidualConvBlock(
                dec_filters,
                input_channels=(prev_dec_channels + skip_filters),
                use_se=use_se_blocks,
                name="Decoder_ResidualConvBlock"
            )
            self.decoders.append(dec_block)

            prev_dec_channels = dec_filters

        # 3.4. Final conv => num_output_slices * out_channels
        self.final_conv = Conv2D(
            filters=self.num_output_slices * self.out_channels,
            kernel_size=1,
            activation='softmax',
            name="Final_Conv"
        )

        # 3.5. Auxiliary heads for deep supervision (2 shallowest decoder levels)
        if use_deep_supervision:
            self.aux_heads = [
                Conv2D(
                    self.num_output_slices * self.out_channels,
                    kernel_size=1,
                    activation='softmax',
                    name=f"Aux_Conv_{i}"
                )
                for i in range(min(2, self.num_encoder_levels))
            ]
        else:
            self.aux_heads = []

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_channels": self.input_channels,
            "base_num_filters": self.base_num_filters,
            "encoder_level_factors": self.encoder_level_factors,
            "num_output_slices": self.num_output_slices,
            "out_channels": self.out_channels,
            "center_depth": self.center_depth,
            "use_se_blocks": self.use_se_blocks,
            "use_deep_supervision": self.use_deep_supervision,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, training=False):
        # Encoder forward
        skip_connections = []
        e_in = x
        for i in range(self.num_encoder_levels):
            e_out = self.encoders[i](e_in)
            skip_connections.append(e_out)
            e_in = self.pools[i](e_out)

        # Center forward
        center = e_in
        for block in self.center_blocks:
            center = block(center)

        # Decoder forward — collect outputs for deep supervision
        d_in = center
        decoder_outputs = []
        for i in range(self.num_encoder_levels):
            d_up = self.upsamples[i](d_in)
            skip = skip_connections[self.num_encoder_levels - 1 - i]
            att = self.attention_blocks[i](g=d_up, x=skip)
            d_concat = self.concat_layers[i]([d_up, att])
            d_out = self.decoders[i](d_concat)
            decoder_outputs.append(d_out)
            d_in = d_out

        # Final output
        out_flat = self.final_conv(d_in)
        new_shape = (
            -1,
            tf.shape(out_flat)[1],
            tf.shape(out_flat)[2],
            self.num_output_slices,
            self.out_channels
        )
        main_out = tf.reshape(out_flat, new_shape)

        # Deep supervision: auxiliary outputs from the 2 shallowest decoder levels
        if self.use_deep_supervision and training:
            aux_outputs = []
            # decoder_outputs[-1] is shallowest (last decoder level),
            # decoder_outputs[-2] is second shallowest
            for i, aux_head in enumerate(self.aux_heads):
                aux_flat = aux_head(decoder_outputs[-(i + 1)])
                aux_out = tf.reshape(aux_flat, new_shape)
                aux_outputs.append(aux_out)
            return (main_out, aux_outputs)

        return main_out


# --------------------------------------------------------------------------
# Helper function to dynamically create unet model from config
# --------------------------------------------------------------------------
def create_dynamic_unet_from_config():
    """
    Reads from config.py and dynamically builds the U-Net model.
    """
    from .config import (
        num_channels,
        num_input_slices,
        num_output_slices,
        num_classes,
        base_num_filters,
        encoder_level_factors,
        center_depth,
        use_se_blocks,
        use_deep_supervision,
    )

    input_channels = num_channels * num_input_slices
    model = DynamicAttentionResUNet(
        input_channels=input_channels,
        base_num_filters=base_num_filters,
        encoder_level_factors=encoder_level_factors,
        num_output_slices=num_output_slices,
        out_channels=num_classes,
        center_depth=center_depth,
        use_se_blocks=use_se_blocks,
        use_deep_supervision=use_deep_supervision,
        name="DynamicAttentionResUNet_Model"
    )

    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',  # placeholder; real loss applied in training loop
        metrics=['accuracy']
    )
    return model

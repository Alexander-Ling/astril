import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, MaxPooling2D, UpSampling2D,
    Add, Multiply, Activation, Concatenate
)
from tensorflow.keras.optimizers import Adam

########################################################
# 1) ResidualConvBlock
########################################################
class ResidualConvBlock(tf.keras.layers.Layer):
    """
    Basic residual block using SeparableConv2D, plus optional 1x1
    channel adjust if input != num_filters.
    """
    def __init__(self, num_filters, input_channels=None, name=f"ResidualConvBlock"):
        super().__init__(name=name)
        # Use the provided name or default to the class name.
        self._base_name = self.name if self.name is not None else "ResidualConvBlock"
        self.num_filters = num_filters
        self.adjust_channels = (input_channels is not None and input_channels != num_filters)

        if self.adjust_channels:
            self.channel_adjust_conv = Conv2D(
                num_filters, kernel_size=1, padding='same', use_bias=False,
                name=f"RCB_channel_adjust_conv"
            )

        self.conv1 = SeparableConv2D(
            num_filters, kernel_size=3, padding='same', activation='relu',
            name=f"RCB_conv1"
        )
        self.conv2 = SeparableConv2D(
            num_filters, kernel_size=3, padding='same', activation='relu',
            name=f"RCB_conv2"
        )
        # Precreate the addition layer to give it a name.
        self.add = Add(name=f"RCB_add")

    def call(self, x):
        if self.adjust_channels:
            skip = self.channel_adjust_conv(x)
        else:
            skip = x

        x = self.conv1(x)
        x = self.conv2(x)
        out = self.add([x, skip])
        return out


########################################################
# 2) AttentionBlock
########################################################
class AttentionBlock(tf.keras.layers.Layer):
    """
    Standard attention block from Attention U-Net.
    """
    def __init__(self, F_g, F_l, F_int, name=f"AttentionBlock"):
        super().__init__(name=name)
        self._base_name = self.name if self.name is not None else "AttentionBlock"
        self.W_g = Conv2D(
            F_int, kernel_size=1, strides=1, padding='same', activation='relu',
            name=f"AB_W_g"
        )
        self.W_x = Conv2D(
            F_int, kernel_size=1, strides=1, padding='same', activation='relu',
            name=f"AB_W_x"
        )
        self.psi = Conv2D(
            1, kernel_size=1, strides=1, padding='same', activation='sigmoid',
            name=f"AB_psi"
        )
        self.relu = Activation('relu', name=f"AB_relu")
        self.add = Add(name=f"AB_add")
        self.multiply = Multiply(name=f"AB_multiply")

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
    Now supports multiple center blocks via 'center_depth'.
    """
    def __init__(
        self,
        input_channels,          # e.g. base_channels * num_input_slices
        base_num_filters=32,     # e.g. 32 or 64
        encoder_level_factors=[1, 2, 4, 8],  # expansions for each encoder level
        num_output_slices=1,     # how many slices in final output
        out_channels=4,          # number of segmentation classes
        center_depth=1,          # number of bottleneck (center) blocks
        name="DynamicAttentionResUNet",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # Store these as instance attributes so get_config() can access them
        self.input_channels = input_channels
        self.base_num_filters = base_num_filters
        self.encoder_level_factors = encoder_level_factors
        self.num_output_slices = num_output_slices
        self.out_channels = out_channels
        self.center_depth = center_depth

        self.num_encoder_levels = len(encoder_level_factors)

        # 3.1. Build Encoder Blocks
        self.encoders = []
        self.pools = []
        prev_channels = input_channels
        for i, factor in enumerate(encoder_level_factors):
            nf = base_num_filters * factor
            block = ResidualConvBlock(
                num_filters=nf, input_channels=prev_channels,
                name=f"Encoder_ResidualConvBlock"
            )
            self.encoders.append(block)
            pool = MaxPooling2D(pool_size=2, padding='same', name=f"Encoder_MaxPool")
            self.pools.append(pool)
            prev_channels = nf

        # 3.2. Build the "Center" (bottleneck) blocks
        center_filters = base_num_filters * encoder_level_factors[-1] * 2
        self.center_blocks = []
        in_channels = prev_channels
        for i in range(center_depth):
            block = ResidualConvBlock(
                center_filters, input_channels=in_channels,
                name=f"Center_ResidualConvBlock"
            )
            self.center_blocks.append(block)
            in_channels = center_filters

        # 3.3. Build Decoder
        self.upsamples = []
        self.attention_blocks = []
        self.decoders = []
        self.concat_layers = []  # precreate concatenation layers with names

        reversed_factors = list(reversed(encoder_level_factors))
        prev_dec_channels = center_filters  # after the last center block

        for i, factor in enumerate(reversed_factors):
            up = UpSampling2D(size=(2, 2), name=f"Decoder_UpSampling")
            self.upsamples.append(up)

            skip_filters = base_num_filters * factor
            att_block = AttentionBlock(
                F_g=prev_dec_channels,
                F_l=skip_filters,
                F_int=skip_filters // 2,
                name=f"Decoder_AttentionBlock"
            )
            self.attention_blocks.append(att_block)

            # Create a named Concatenate layer for merging the skip and upsampled signals
            concat_layer = Concatenate(name=f"Decoder_Concat")
            self.concat_layers.append(concat_layer)

            dec_filters = skip_filters
            dec_block = ResidualConvBlock(
                dec_filters,
                input_channels=(prev_dec_channels + skip_filters),
                name=f"Decoder_ResidualConvBlock"
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

    def get_config(self):
        """
        Return a dict of parameters needed to recreate this model.
        """
        config = super().get_config()
        config.update({
            "input_channels": self.input_channels,
            "base_num_filters": self.base_num_filters,
            "encoder_level_factors": self.encoder_level_factors,
            "num_output_slices": self.num_output_slices,
            "out_channels": self.out_channels,
            "center_depth": self.center_depth
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Reconstruct an instance from the config dictionary.
        """
        return cls(**config)

    def call(self, x):
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

        # Decoder forward
        d_in = center
        for i in range(self.num_encoder_levels):
            d_up = self.upsamples[i](d_in)
            skip = skip_connections[self.num_encoder_levels - 1 - i]

            att = self.attention_blocks[i](g=d_up, x=skip)
            d_concat = self.concat_layers[i]([d_up, att])
            d_out = self.decoders[i](d_concat)
            d_in = d_out

        # Final
        out_flat = self.final_conv(d_in)
        if self.num_output_slices >= 1:
            new_shape = (
                -1,
                tf.shape(out_flat)[1],
                tf.shape(out_flat)[2],
                self.num_output_slices,
                self.out_channels
            )
            out = tf.reshape(out_flat, new_shape)
        else:
            out = out_flat

        return out


# --------------------------------------------------------------------------
# Helper function to dynamically create unet model from config
# --------------------------------------------------------------------------
def create_dynamic_unet_from_config():
    """
    Example: read from config.py or pass as parameters
    to dynamically build a U-Net model.
    """
    from .config import (
        num_channels,
        num_input_slices,
        num_output_slices,
        num_classes,
        base_num_filters,
        encoder_level_factors,
        center_depth
    )

    input_channels = num_channels * num_input_slices
    model = DynamicAttentionResUNet(
        input_channels=input_channels,
        base_num_filters=base_num_filters,
        encoder_level_factors=encoder_level_factors,
        num_output_slices=num_output_slices,
        out_channels=num_classes,
        center_depth=center_depth,
        name="DynamicAttentionResUNet_Model"
    )

    # compile
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',  # default loss that gets replaced
        metrics=['accuracy']  # default metric that gets replaced
    )
    return model

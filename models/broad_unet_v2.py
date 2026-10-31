"""
models/broad_unet_v2.py
=======================
Enhanced Broad-UNet architecture supporting:
- Residual temperature prediction: Yhat = X_{t0} + Delta Y
- Cyclical temporal coordinates (10-channel input)
- Post-ASPP bottleneck enhancements: MLP, KAN, Sequential, Parallel, Control
- Decoder skip fusion enhancements at merge6, merge7, merge8, merge9
- Encoder stage enhancements at conv2, conv3, conv4

Strictly preserves Broad-UNet's factorized multi-scale 3D kernel structure.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Add, Dropout, concatenate, Activation, Reshape,
    Conv3D, MaxPool3D, UpSampling3D, GlobalAveragePooling3D
)

from models.modules import (
    AnchorResidualBroadcast, MLPBlock, KANBlock, SequentialMLPKANBlock,
    ParallelMLPKANBlock, SkipFusionBlock, Conv3DControlBlock
)

def build_broad_unet_v2(
    lags: int = 6,
    latitude: int = 32,
    longitude: int = 48,
    features: int = 6,
    features_output: int = 5,
    filters: int = 16,
    dropout: float = 0.2,
    activation: str = "linear",
    residual_head: bool = True,
    target_channel_idx: int = 0,
    post_aspp_module: str = "none",  # "none", "mlp", "kan", "sequential", "parallel", "control"
    decoder_fusion: str = "none",     # "none", "merge6", "merge67", "merge6789", "merge9"
    encoder_stages: list = None,      # ["conv2", "conv3", "conv4"]
    encoder_module: str = "none",     # "none", "mlp", "kan"
    seed: int = 50
):
    """
    Constructs the enhanced Broad-UNet model (v2 and experimental architectures).
    """
    if encoder_stages is None:
        encoder_stages = []

    kernel_init = tf.keras.initializers.GlorotUniform(seed=seed)

    def image_level_feature_pooling(x, f):
        up_size = x.shape[2:4] if hasattr(x, "shape") else x.get_shape().as_list()[2:4]
        x = GlobalAveragePooling3D()(x)
        x = Reshape((1, 1, 1, f))(x)
        x = UpSampling3D(size=(1, up_size[0], up_size[1]))(x)
        return x

    def ASPP_block(x, f):
        x1 = Conv3D(f, 1, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x2 = Conv3D(f, (1, 3, 3), activation="relu", padding="same", dilation_rate=6, kernel_initializer=kernel_init)(x)
        x3 = Conv3D(f, (1, 3, 3), activation="relu", padding="same", dilation_rate=12, kernel_initializer=kernel_init)(x)
        x4 = Conv3D(f, (1, 3, 3), activation="relu", padding="same", dilation_rate=18, kernel_initializer=kernel_init)(x)
        x5 = image_level_feature_pooling(x, f)
        x = concatenate([x1, x2, x3, x4, x5], axis=-1)
        x = Conv3D(f, 1, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        return x

    def convolutional_block(x, f, k):
        shortcut = x
        x1 = Conv3D(f, 1, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x2 = Conv3D(f, (1, 1, 3), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x2 = Conv3D(f, (1, 3, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x2)
        x2 = Conv3D(f, (3, 1, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x2)
        x3 = Conv3D(f, (1, 1, 5), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x3 = Conv3D(f, (1, 5, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x3)
        x3 = Conv3D(f, (5, 1, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x3)
        x4 = Conv3D(f, (1, 1, 3), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x4 = Conv3D(f, (1, 3, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x4)
        x4 = Conv3D(f, (3, 1, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x4)
        x5 = Conv3D(f, (1, 1, 3), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x5 = Conv3D(f, (1, 3, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x5)
        x5 = Conv3D(f, (3, 1, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x5)
        x = concatenate([x1, x2, x3, x4, x5], axis=-1)
        x = Conv3D(f, 1, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        return x

    def apply_encoder_enhancement(x, channels, stage_name):
        if stage_name in encoder_stages:
            if encoder_module == "mlp":
                return MLPBlock(channels, name=f"{stage_name}_mlp")(x)
            elif encoder_module == "kan":
                return KANBlock(channels, name=f"{stage_name}_kan")(x)
        return x

    # --- Contracting part / Encoder ---
    inputs = Input(shape=(lags, latitude, longitude, features), name="meteorological_input")

    # Extract t0 anchor frame for residual prediction: X[:, -1, :, :, target_channel_idx]
    if residual_head:
        t0_anchor = AnchorResidualBroadcast(
            target_idx=target_channel_idx,
            num_outputs=features_output,
            name="anchor_residual_broadcast"
        )(inputs)

    # conv1
    conv1 = Conv3D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="conv1")(inputs)
    conv1 = convolutional_block(conv1, filters, 3)
    conv1 = apply_encoder_enhancement(conv1, filters, "conv1")
    pool1 = MaxPool3D(pool_size=(1, 2, 2), name="pool1")(conv1)

    # conv2
    conv2 = Conv3D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="conv2")(pool1)
    conv2 = convolutional_block(conv2, 2 * filters, 3)
    conv2 = apply_encoder_enhancement(conv2, 2 * filters, "conv2")
    pool2 = MaxPool3D(pool_size=(1, 2, 2), name="pool2")(conv2)

    # conv3
    conv3 = Conv3D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="conv3")(pool2)
    conv3 = convolutional_block(conv3, 4 * filters, 3)
    conv3 = apply_encoder_enhancement(conv3, 4 * filters, "conv3")
    pool3 = MaxPool3D(pool_size=(1, 2, 2), name="pool3")(conv3)

    # conv4
    conv4 = Conv3D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="conv4")(pool3)
    conv4 = convolutional_block(conv4, 8 * filters, 3)
    conv4 = apply_encoder_enhancement(conv4, 8 * filters, "conv4")
    drop4 = Dropout(dropout, name="drop4")(conv4)

    # --- Bottleneck part ---
    pool4 = MaxPool3D(pool_size=(1, 2, 2), name="pool4")(drop4)
    conv5 = Conv3D(16 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="conv5")(pool4)
    conv5 = convolutional_block(conv5, 16 * filters, 3)
    compressLags = Conv3D(16 * filters, (lags, 1, 1), activation="relu", padding="valid", name="compress_lags5")(conv5)
    aspp = ASPP_block(compressLags, 16 * filters)

    # Post-ASPP bottleneck enhancements
    bottleneck_c = 16 * filters
    if post_aspp_module == "mlp":
        aspp = MLPBlock(bottleneck_c, name="aspp_mlp")(aspp)
    elif post_aspp_module == "kan":
        aspp = KANBlock(bottleneck_c, name="aspp_kan")(aspp)
    elif post_aspp_module == "sequential":
        aspp = SequentialMLPKANBlock(bottleneck_c, name="aspp_seq_mlp_kan")(aspp)
    elif post_aspp_module == "parallel":
        aspp = ParallelMLPKANBlock(bottleneck_c, name="aspp_par_mlp_kan")(aspp)
    elif post_aspp_module == "control":
        aspp = Conv3DControlBlock(bottleneck_c, name="aspp_control")(aspp)

    drop5 = Dropout(dropout, name="drop5")(aspp)

    # --- Expanding part / Decoder ---
    # Merge 6
    up6 = Conv3D(8 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init, name="up6")(
        UpSampling3D(size=(1, 2, 2))(drop5)
    )
    compressLags4 = Conv3D(8 * filters, (lags, 1, 1), activation="relu", padding="valid", name="compress_lags4")(drop4)
    merge6 = concatenate([compressLags4, up6], axis=-1, name="merge6")
    if decoder_fusion in ["merge6", "merge67", "merge6789"]:
        merge6 = SkipFusionBlock(16 * filters, 8 * filters, mode="kan", name="fusion_merge6")(merge6)
    conv6 = Conv3D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="conv6")(merge6)
    conv6 = convolutional_block(conv6, 8 * filters, 3)

    # Merge 7
    up7 = Conv3D(4 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init, name="up7")(
        UpSampling3D(size=(1, 2, 2))(conv6)
    )
    compressLags3 = Conv3D(4 * filters, (lags, 1, 1), activation="relu", padding="valid", name="compress_lags3")(conv3)
    merge7 = concatenate([compressLags3, up7], axis=-1, name="merge7")
    if decoder_fusion in ["merge67", "merge6789"]:
        merge7 = SkipFusionBlock(8 * filters, 4 * filters, mode="kan", name="fusion_merge7")(merge7)
    conv7 = Conv3D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="conv7")(merge7)
    conv7 = convolutional_block(conv7, 4 * filters, 3)

    # Merge 8
    up8 = Conv3D(2 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init, name="up8")(
        UpSampling3D(size=(1, 2, 2))(conv7)
    )
    compressLags2 = Conv3D(2 * filters, (lags, 1, 1), activation="relu", padding="valid", name="compress_lags2")(conv2)
    merge8 = concatenate([compressLags2, up8], axis=-1, name="merge8")
    if decoder_fusion in ["merge6789"]:
        merge8 = SkipFusionBlock(4 * filters, 2 * filters, mode="kan", name="fusion_merge8")(merge8)
    conv8 = Conv3D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="conv8")(merge8)
    conv8 = convolutional_block(conv8, 2 * filters, 3)

    # Merge 9
    up9 = Conv3D(filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init, name="up9")(
        UpSampling3D(size=(1, 2, 2))(conv8)
    )
    compressLags1 = Conv3D(filters, (lags, 1, 1), activation="relu", padding="valid", name="compress_lags1")(conv1)
    merge9 = concatenate([compressLags1, up9], axis=-1, name="merge9")
    if decoder_fusion in ["merge6789", "merge9"]:
        merge9 = SkipFusionBlock(2 * filters, filters, mode="kan", name="fusion_merge9")(merge9)
    conv9 = Conv3D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="conv9")(merge9)
    conv9 = convolutional_block(conv9, filters, 3)
    conv9 = Conv3D(2 * features_output, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="conv9_proj")(conv9)

    # conv10 output projection
    conv10 = Conv3D(
        features_output, 1,
        activation=activation,
        padding="same",
        name="conv10"
    )(conv9)

    if residual_head:
        final_output = Add(name="residual_temperature_prediction")([t0_anchor, conv10])
    else:
        final_output = conv10

    return Model(inputs=inputs, outputs=final_output, name="Broad_UNet_v2")

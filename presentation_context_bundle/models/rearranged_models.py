"""
models/rearranged_models.py
===========================
Factory and architectural definitions for the 15 Layer Rearrangement Experiments (L0 to L14)
investigating the impact of moving layers up/down, reordering convolutions,
ASPP repositioning, norm/activation order, and skip-fusion processing.

All models support 7 forecast horizons: [1, 2, 4, 8, 12, 24, 36] steps
corresponding to [+2h, +4h, +8h, +16h, +24h, +48h, +72h] lead times.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Add, Dropout, concatenate, Activation, Reshape,
    Conv3D, MaxPool3D, UpSampling3D, GlobalAveragePooling3D, BatchNormalization
)

from models.modules import (
    AnchorResidualBroadcast, MLPBlock, KANBlock,
    SequentialMLPKANBlock, ParallelMLPKANBlock, SkipFusionBlock
)

EXPERIMENT_NAMES = {
    "L0_original": "Original Broad-UNet Reference",
    "L1_conv_rearrangement": "MultiScale Before Conv3D in Encoder",
    "L2_conv_before_aspp": "Additional Conv3D Processing Before ASPP",
    "L3_conv_after_aspp": "Additional Conv3D Processing After ASPP",
    "L4_mlp_before_aspp": "MLP Bottleneck Processing Before ASPP",
    "L5_mlp_after_aspp": "MLP Bottleneck Processing After ASPP",
    "L6_kan_before_aspp": "KAN Bottleneck Processing Before ASPP",
    "L7_kan_after_aspp": "KAN Bottleneck Processing After ASPP",
    "L8_mlp_then_kan": "Sequential MLP -> KAN After ASPP",
    "L9_kan_then_mlp": "Sequential KAN -> MLP After ASPP",
    "L10_before_skip_fusion": "Pre-Skip Fusion Processing on Encoder Features",
    "L11_after_skip_fusion": "Post-Skip KAN Fusion at Merge6 & Merge7",
    "L12_encoder_rearrangement": "Encoder Conv -> BatchNorm -> ReLU Ordering",
    "L13_decoder_rearrangement": "Decoder Convolution Before UpSampling",
    "L14_best_combined_arrangement": "Best Combined Architecture (Norm-Act + Pre-ASPP + KAN-Skips)"
}

def build_rearranged_model(
    exp_id: str = "L0_original",
    lags: int = 6,
    latitude: int = 32,
    longitude: int = 48,
    features: int = 10,
    features_output: int = 7,
    filters: int = 16,
    dropout: float = 0.2,
    residual_head: bool = True,
    seed: int = 50
) -> Model:
    """
    Constructs the requested rearranged architecture.
    """
    kernel_init = tf.keras.initializers.GlorotUniform(seed=seed)

    def image_level_feature_pooling(x, f):
        up_size = x.shape[2:4] if hasattr(x, "shape") else x.get_shape().as_list()[2:4]
        x = GlobalAveragePooling3D()(x)
        x = Reshape((1, 1, 1, f))(x)
        x = UpSampling3D(size=(1, int(up_size[0]), int(up_size[1])))(x)
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

    def convolutional_block(x, f, k, norm=False):
        shortcut = x
        x1 = Conv3D(f, 1, activation=None if norm else "relu", padding="same", kernel_initializer=kernel_init)(x)
        if norm:
            x1 = BatchNormalization()(x1)
            x1 = Activation("relu")(x1)

        x2 = Conv3D(f, (1, 1, 3), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x2 = Conv3D(f, (1, 3, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x2)
        x2 = Conv3D(f, (3, 1, 1), activation=None if norm else "relu", padding="same", kernel_initializer=kernel_init)(x2)
        if norm:
            x2 = BatchNormalization()(x2)
            x2 = Activation("relu")(x2)

        x3 = Conv3D(f, (1, 1, 5), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x3 = Conv3D(f, (1, 5, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x3)
        x3 = Conv3D(f, (5, 1, 1), activation=None if norm else "relu", padding="same", kernel_initializer=kernel_init)(x3)
        if norm:
            x3 = BatchNormalization()(x3)
            x3 = Activation("relu")(x3)

        x4 = Conv3D(f, (1, 1, 3), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x4 = Conv3D(f, (1, 3, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x4)
        x4 = Conv3D(f, (3, 1, 1), activation=None if norm else "relu", padding="same", kernel_initializer=kernel_init)(x4)
        if norm:
            x4 = BatchNormalization()(x4)
            x4 = Activation("relu")(x4)

        x5 = Conv3D(f, (1, 1, 3), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x5 = Conv3D(f, (1, 3, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x5)
        x5 = Conv3D(f, (3, 1, 1), activation=None if norm else "relu", padding="same", kernel_initializer=kernel_init)(x5)
        if norm:
            x5 = BatchNormalization()(x5)
            x5 = Activation("relu")(x5)

        x = concatenate([x1, x2, x3, x4, x5], axis=-1)
        x = Conv3D(f, 1, activation=None if norm else "relu", padding="same", kernel_initializer=kernel_init)(x)
        if norm:
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        return x

    # --- Inputs & Residual Anchor ---
    inputs = Input(shape=(lags, latitude, longitude, features), name="meteorological_input")
    if residual_head:
        t0_anchor = AnchorResidualBroadcast(target_idx=0, num_outputs=features_output, name="anchor_residual_broadcast")(inputs)

    use_norm = (exp_id in ["L12_encoder_rearrangement", "L14_best_combined_arrangement"])

    # --- Encoder Stage 1 ---
    if exp_id == "L1_conv_rearrangement":
        # MultiScale first, then Conv3D
        conv1_ms = convolutional_block(inputs, features, 3)
        conv1 = Conv3D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv1_ms)
    elif use_norm:
        conv1 = Conv3D(filters, 3, activation=None, padding="same", kernel_initializer=kernel_init)(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = convolutional_block(conv1, filters, 3, norm=True)
    else:
        conv1 = Conv3D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(inputs)
        conv1 = convolutional_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)

    # --- Encoder Stage 2 ---
    if exp_id == "L1_conv_rearrangement":
        conv2_ms = convolutional_block(pool1, filters, 3)
        conv2 = Conv3D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv2_ms)
    elif use_norm:
        conv2 = Conv3D(2 * filters, 3, activation=None, padding="same", kernel_initializer=kernel_init)(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = convolutional_block(conv2, 2 * filters, 3, norm=True)
    else:
        conv2 = Conv3D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool1)
        conv2 = convolutional_block(conv2, 2 * filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)

    # --- Encoder Stage 3 ---
    if exp_id == "L1_conv_rearrangement":
        conv3_ms = convolutional_block(pool2, 2 * filters, 3)
        conv3 = Conv3D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv3_ms)
    elif use_norm:
        conv3 = Conv3D(4 * filters, 3, activation=None, padding="same", kernel_initializer=kernel_init)(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3 = convolutional_block(conv3, 4 * filters, 3, norm=True)
    else:
        conv3 = Conv3D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool2)
        conv3 = convolutional_block(conv3, 4 * filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)

    # --- Encoder Stage 4 ---
    if exp_id == "L1_conv_rearrangement":
        conv4_ms = convolutional_block(pool3, 4 * filters, 3)
        conv4 = Conv3D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv4_ms)
    elif use_norm:
        conv4 = Conv3D(8 * filters, 3, activation=None, padding="same", kernel_initializer=kernel_init)(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        conv4 = convolutional_block(conv4, 8 * filters, 3, norm=True)
    else:
        conv4 = Conv3D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool3)
        conv4 = convolutional_block(conv4, 8 * filters, 3)
    drop4 = Dropout(dropout)(conv4)

    # --- Bottleneck Stage 5 ---
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool4)
    conv5 = convolutional_block(conv5, 16 * filters, 3)
    compressLags = Conv3D(16 * filters, (lags, 1, 1), activation="relu", padding="valid")(conv5)

    # Pre-ASPP Modifications
    bneck_in = compressLags
    if exp_id in ["L2_conv_before_aspp", "L14_best_combined_arrangement"]:
        bneck_in = Conv3D(16 * filters, (1, 3, 3), activation="relu", padding="same", kernel_initializer=kernel_init, name="pre_aspp_conv")(bneck_in)
    elif exp_id == "L4_mlp_before_aspp":
        bneck_in = MLPBlock(16 * filters, name="pre_aspp_mlp")(bneck_in)
    elif exp_id == "L6_kan_before_aspp":
        bneck_in = KANBlock(16 * filters, name="pre_aspp_kan")(bneck_in)

    # ASPP Multi-Scale Block
    aspp = ASPP_block(bneck_in, 16 * filters)

    # Post-ASPP Modifications
    bneck_out = aspp
    if exp_id == "L3_conv_after_aspp":
        bneck_out = Conv3D(16 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init, name="post_aspp_conv")(bneck_out)
    elif exp_id == "L5_mlp_after_aspp":
        bneck_out = MLPBlock(16 * filters, name="post_aspp_mlp")(bneck_out)
    elif exp_id == "L7_kan_after_aspp":
        bneck_out = KANBlock(16 * filters, name="post_aspp_kan")(bneck_out)
    elif exp_id == "L8_mlp_then_kan":
        bneck_out = SequentialMLPKANBlock(16 * filters, name="post_aspp_seq_mlp_kan")(bneck_out)
    elif exp_id == "L9_kan_then_mlp":
        # KAN then MLP
        bneck_out = KANBlock(16 * filters, name="post_aspp_kan")(bneck_out)
        bneck_out = MLPBlock(16 * filters, name="post_aspp_mlp")(bneck_out)

    drop5 = Dropout(dropout)(bneck_out)

    # --- Decoder Stage 6 ---
    if exp_id == "L13_decoder_rearrangement":
        # Convolution before UpSampling
        dec6_proc = Conv3D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(drop5)
        up6 = Conv3D(8 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
            UpSampling3D(size=(1, 2, 2))(dec6_proc)
        )
    else:
        up6 = Conv3D(8 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
            UpSampling3D(size=(1, 2, 2))(drop5)
        )

    skip4 = Conv3D(8 * filters, (lags, 1, 1), activation="relu", padding="valid")(drop4)
    if exp_id == "L10_before_skip_fusion":
        skip4 = Conv3D(8 * filters, 1, activation="relu", padding="same", kernel_initializer=kernel_init, name="pre_skip4_conv")(skip4)

    merge6 = concatenate([skip4, up6], axis=-1)
    if exp_id in ["L11_after_skip_fusion", "L14_best_combined_arrangement"]:
        merge6 = SkipFusionBlock(16 * filters, 8 * filters, mode="kan", name="post_skip_merge6")(merge6)

    conv6 = Conv3D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge6)
    conv6 = convolutional_block(conv6, 8 * filters, 3)

    # --- Decoder Stage 7 ---
    if exp_id == "L13_decoder_rearrangement":
        dec7_proc = Conv3D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv6)
        up7 = Conv3D(4 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
            UpSampling3D(size=(1, 2, 2))(dec7_proc)
        )
    else:
        up7 = Conv3D(4 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
            UpSampling3D(size=(1, 2, 2))(conv6)
        )

    skip3 = Conv3D(4 * filters, (lags, 1, 1), activation="relu", padding="valid")(conv3)
    if exp_id == "L10_before_skip_fusion":
        skip3 = Conv3D(4 * filters, 1, activation="relu", padding="same", kernel_initializer=kernel_init, name="pre_skip3_conv")(skip3)

    merge7 = concatenate([skip3, up7], axis=-1)
    if exp_id in ["L11_after_skip_fusion", "L14_best_combined_arrangement"]:
        merge7 = SkipFusionBlock(8 * filters, 4 * filters, mode="kan", name="post_skip_merge7")(merge7)

    conv7 = Conv3D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge7)
    conv7 = convolutional_block(conv7, 4 * filters, 3)

    # --- Decoder Stage 8 ---
    up8 = Conv3D(2 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv7)
    )
    skip2 = Conv3D(2 * filters, (lags, 1, 1), activation="relu", padding="valid")(conv2)
    merge8 = concatenate([skip2, up8], axis=-1)
    conv8 = Conv3D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge8)
    conv8 = convolutional_block(conv8, 2 * filters, 3)

    # --- Decoder Stage 9 ---
    up9 = Conv3D(filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv8)
    )
    skip1 = Conv3D(filters, (lags, 1, 1), activation="relu", padding="valid")(conv1)
    merge9 = concatenate([skip1, up9], axis=-1)
    conv9 = Conv3D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge9)
    conv9 = convolutional_block(conv9, filters, 3)
    conv9 = Conv3D(2 * features_output, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv9)

    # Output projection for 7 forecast horizons
    conv10 = Conv3D(features_output, 1, activation="linear", padding="same", name="conv10")(conv9)

    if residual_head:
        final_output = Add(name="residual_temperature_prediction")([t0_anchor, conv10])
    else:
        final_output = conv10

    model = Model(inputs=inputs, outputs=final_output, name=f"Broad_UNet_{exp_id}")
    return model


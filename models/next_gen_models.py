"""
models/next_gen_models.py
=========================
Next-Generation Architectures extending beyond L15 (Champion Hybrid):
1. L16_full_hierarchy_kan_skips:
   - Extends KAN B-spline Skip Fusion across ALL 4 decoder merge levels (merge6, merge7, merge8, merge9).
   - Dynamically aligns fine-scale coastal and mountainous terrain features at high spatial resolutions (16x24 and 32x48).
2. L17_decoupled_diurnal_synoptic_heads:
   - Decoupled Multi-Horizon Prediction Heads:
     * Diurnal Head (+2h, +4h, +8h, +16h): Tanh-bounded oscillating perturbation.
     * Synoptic Head (+24h, +48h, +72h): Unconstrained linear advection projection.
   - Eliminates multi-task negative gradient interference between diurnal solar and baroclinic frontal objectives.
3. L18_spatiotemporal_se_multiscale:
   - Spatio-Temporal Squeeze-and-Excitation (ST-SE) channel-lag attention inside each MultiScale Conv3D block.
   - Dynamically re-weights atmospheric channels (e.g. amplifying solar radiation during daytime, amplifying wind advection during frontal passages).
4. L19_apex_hybrid_kan_se_decoupled:
   - The Ultimate Apex Synthesis:
     * Full 4-Stage KAN Skip Fusion (merge6, 7, 8, 9).
     * Spatio-Temporal Squeeze-and-Excitation in MultiScale blocks.
     * Decoupled Diurnal + Synoptic prediction heads.
     * Global LeakyReLU (alpha=0.2) backbone.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Add, Dropout, concatenate, Activation, Reshape,
    Conv3D, MaxPool3D, UpSampling3D, GlobalAveragePooling3D, LeakyReLU
)
from models.modules import AnchorResidualBroadcast, SkipFusionBlock, SpatioTemporalSEBlock


def build_next_gen_model(
    model_name: str,
    lags: int = 6,
    latitude: int = 32,
    longitude: int = 48,
    features: int = 10,
    features_output: int = 7,
    filters: int = 16,
    dropout: float = 0.2,
    seed: int = 42
) -> Model:
    kernel_init = tf.keras.initializers.GlorotUniform(seed=seed)
    act_name = "leaky_relu"

    use_all_kan = model_name in ["L16_full_hierarchy_kan_skips", "L19_apex_hybrid_kan_se_decoupled"]
    use_decoupled = model_name in ["L17_decoupled_diurnal_synoptic_heads", "L19_apex_hybrid_kan_se_decoupled"]
    use_se = model_name in ["L18_spatiotemporal_se_multiscale", "L19_apex_hybrid_kan_se_decoupled"]

    def image_level_feature_pooling(x, f):
        up_size = x.shape[2:4] if hasattr(x, "shape") else x.get_shape().as_list()[2:4]
        x = GlobalAveragePooling3D()(x)
        x = Reshape((1, 1, 1, f))(x)
        x = UpSampling3D(size=(1, int(up_size[0]), int(up_size[1])))(x)
        return x

    def ASPP_block(x, f):
        x1 = Conv3D(f, 1, activation=act_name, padding="same", kernel_initializer=kernel_init)(x)
        x2 = Conv3D(f, (1, 3, 3), activation=act_name, padding="same", dilation_rate=6, kernel_initializer=kernel_init)(x)
        x3 = Conv3D(f, (1, 3, 3), activation=act_name, padding="same", dilation_rate=12, kernel_initializer=kernel_init)(x)
        x4 = Conv3D(f, (1, 3, 3), activation=act_name, padding="same", dilation_rate=18, kernel_initializer=kernel_init)(x)
        x5 = image_level_feature_pooling(x, f)
        x = concatenate([x1, x2, x3, x4, x5], axis=-1)
        x = Conv3D(f, 1, activation=act_name, padding="same", kernel_initializer=kernel_init)(x)
        return x

    def convolutional_block(x, f, k, block_id=""):
        shortcut = x
        x1 = Conv3D(f, 1, activation=act_name, padding="same", kernel_initializer=kernel_init)(x)

        x2 = Conv3D(f, (1, 1, 3), activation=act_name, padding="same", kernel_initializer=kernel_init)(x)
        x2 = Conv3D(f, (1, 3, 1), activation=act_name, padding="same", kernel_initializer=kernel_init)(x2)
        x2 = Conv3D(f, (3, 1, 1), activation=act_name, padding="same", kernel_initializer=kernel_init)(x2)

        x3 = Conv3D(f, (1, 1, 5), activation=act_name, padding="same", kernel_initializer=kernel_init)(x)
        x3 = Conv3D(f, (1, 5, 1), activation=act_name, padding="same", kernel_initializer=kernel_init)(x3)
        x3 = Conv3D(f, (5, 1, 1), activation=act_name, padding="same", kernel_initializer=kernel_init)(x3)

        x4 = Conv3D(f, (1, 1, 3), activation=act_name, padding="same", kernel_initializer=kernel_init)(x)
        x4 = Conv3D(f, (1, 3, 1), activation=act_name, padding="same", kernel_initializer=kernel_init)(x4)
        x4 = Conv3D(f, (3, 1, 1), activation=act_name, padding="same", kernel_initializer=kernel_init)(x4)

        x5 = Conv3D(f, (1, 1, 3), activation=act_name, padding="same", kernel_initializer=kernel_init)(x)
        x5 = Conv3D(f, (1, 3, 1), activation=act_name, padding="same", kernel_initializer=kernel_init)(x5)
        x5 = Conv3D(f, (3, 1, 1), activation=act_name, padding="same", kernel_initializer=kernel_init)(x5)

        x = concatenate([x1, x2, x3, x4, x5], axis=-1)
        x = Conv3D(f, 1, activation=act_name, padding="same", kernel_initializer=kernel_init)(x)

        if use_se:
            x = SpatioTemporalSEBlock(f, reduction=4, name=f"se_{block_id}_{f}")(x)

        x = Add()([x, shortcut])
        x = Activation(act_name)(x)
        return x

    # --- Inputs & Anchor Residual Broadcast ---
    inputs = Input(shape=(lags, latitude, longitude, features), name="meteorological_input")
    t0_anchor = AnchorResidualBroadcast(target_idx=0, num_outputs=features_output, name="anchor_t0_broadcast")(inputs)

    # --- Encoder Stage 1 ---
    conv1 = Conv3D(filters, 3, activation=act_name, padding="same", kernel_initializer=kernel_init)(inputs)
    conv1 = convolutional_block(conv1, filters, 3, "enc1")
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)

    # --- Encoder Stage 2 ---
    conv2 = Conv3D(2 * filters, 3, activation=act_name, padding="same", kernel_initializer=kernel_init)(pool1)
    conv2 = convolutional_block(conv2, 2 * filters, 3, "enc2")
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)

    # --- Encoder Stage 3 ---
    conv3 = Conv3D(4 * filters, 3, activation=act_name, padding="same", kernel_initializer=kernel_init)(pool2)
    conv3 = convolutional_block(conv3, 4 * filters, 3, "enc3")
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)

    # --- Encoder Stage 4 ---
    conv4 = Conv3D(8 * filters, 3, activation=act_name, padding="same", kernel_initializer=kernel_init)(pool3)
    conv4 = convolutional_block(conv4, 8 * filters, 3, "enc4")
    drop4 = Dropout(dropout)(conv4)

    # --- Bottleneck Stage 5 ---
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16 * filters, 3, activation=act_name, padding="same", kernel_initializer=kernel_init)(pool4)
    conv5 = convolutional_block(conv5, 16 * filters, 3, "bneck")
    compressLags = Conv3D(16 * filters, (lags, 1, 1), activation=act_name, padding="valid")(conv5)

    aspp = ASPP_block(compressLags, 16 * filters)
    drop5 = Dropout(dropout)(aspp)

    # --- Decoder Stage 6 with KAN Skip Fusion ---
    up6 = Conv3D(8 * filters, 2, activation=act_name, padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(drop5)
    )
    skip4 = Conv3D(8 * filters, (lags, 1, 1), activation=act_name, padding="valid")(drop4)
    merge6 = concatenate([skip4, up6], axis=-1)
    merge6 = SkipFusionBlock(16 * filters, 8 * filters, mode="kan", name="post_skip_merge6")(merge6)

    conv6 = Conv3D(8 * filters, 3, activation=act_name, padding="same", kernel_initializer=kernel_init)(merge6)
    conv6 = convolutional_block(conv6, 8 * filters, 3, "dec6")

    # --- Decoder Stage 7 with KAN Skip Fusion ---
    up7 = Conv3D(4 * filters, 2, activation=act_name, padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv6)
    )
    skip3 = Conv3D(4 * filters, (lags, 1, 1), activation=act_name, padding="valid")(conv3)
    merge7 = concatenate([skip3, up7], axis=-1)
    merge7 = SkipFusionBlock(8 * filters, 4 * filters, mode="kan", name="post_skip_merge7")(merge7)

    conv7 = Conv3D(4 * filters, 3, activation=act_name, padding="same", kernel_initializer=kernel_init)(merge7)
    conv7 = convolutional_block(conv7, 4 * filters, 3, "dec7")

    # --- Decoder Stage 8 ---
    up8 = Conv3D(2 * filters, 2, activation=act_name, padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv7)
    )
    skip2 = Conv3D(2 * filters, (lags, 1, 1), activation=act_name, padding="valid")(conv2)
    merge8 = concatenate([skip2, up8], axis=-1)
    if use_all_kan:
        merge8 = SkipFusionBlock(4 * filters, 2 * filters, mode="kan", name="post_skip_merge8")(merge8)

    conv8 = Conv3D(2 * filters, 3, activation=act_name, padding="same", kernel_initializer=kernel_init)(merge8)
    conv8 = convolutional_block(conv8, 2 * filters, 3, "dec8")

    # --- Decoder Stage 9 ---
    up9 = Conv3D(filters, 2, activation=act_name, padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv8)
    )
    skip1 = Conv3D(filters, (lags, 1, 1), activation=act_name, padding="valid")(conv1)
    merge9 = concatenate([skip1, up9], axis=-1)
    if use_all_kan:
        merge9 = SkipFusionBlock(2 * filters, filters, mode="kan", name="post_skip_merge9")(merge9)

    conv9 = Conv3D(filters, 3, activation=act_name, padding="same", kernel_initializer=kernel_init)(merge9)
    conv9 = convolutional_block(conv9, filters, 3, "dec9")
    conv9 = Conv3D(2 * features_output, 3, activation=act_name, padding="same", kernel_initializer=kernel_init)(conv9)

    # --- Output Projection Head ---
    if use_decoupled:
        # Decouple into Diurnal Head (4 horizons: +2h, +4h, +8h, +16h) and Synoptic Head (3 horizons: +24h, +48h, +72h)
        # Scaled Tanh for Diurnal (+-6.0 K anomaly range)
        head_diurnal = Conv3D(4, 1, activation="tanh", padding="same", name="head_diurnal_tanh")(conv9)
        head_diurnal = head_diurnal * 6.0
        # Linear Conv3D for Synoptic (+24h, +48h, +72h)
        head_synoptic = Conv3D(3, 1, activation="linear", padding="same", name="head_synoptic_linear")(conv9)
        delta_T = concatenate([head_diurnal, head_synoptic], axis=-1, name="decoupled_delta_t")
    else:
        delta_T = Conv3D(features_output, 1, activation="linear", padding="same", name="delta_t_linear")(conv9)

    final_output = Add(name="residual_temperature_prediction")([t0_anchor, delta_T])

    model = Model(inputs=inputs, outputs=final_output, name=model_name)
    return model


NEXT_GEN_EXPERIMENTS = {
    "L16_full_hierarchy_kan_skips": {
        "name": "Full-Hierarchy 4-Stage KAN Skip Fusion",
        "desc": "Extends KAN B-spline Skip Fusion across ALL 4 decoder merges (merge6, 7, 8, 9). Cuts more parameters while aligning fine coastal textures.",
        "category": "Skip Connection Mastery"
    },
    "L17_decoupled_diurnal_synoptic_heads": {
        "name": "Decoupled Diurnal-Synoptic Prediction Heads",
        "desc": "Decouples the output projection into a Tanh-scaled diurnal head (+2h..+16h) and linear synoptic head (+24h..+72h), eliminating multi-task gradient conflict.",
        "category": "Multi-Horizon Head Decoupling"
    },
    "L18_spatiotemporal_se_multiscale": {
        "name": "Spatio-Temporal Squeeze-and-Excitation",
        "desc": "Injects ST-SE channel-lag attention into each MultiScale Conv3D block, dynamically prioritizing wind vs solar channels based on global meteorological context.",
        "category": "Attention & Channel Recalibration"
    },
    "L19_apex_hybrid_kan_se_decoupled": {
        "name": "APEX HYBRID: Full KAN + ST-SE + Decoupled Heads",
        "desc": "The ultimate synthesis: Full 4-Stage KAN Skips + Spatio-Temporal Squeeze-and-Excitation + Decoupled Diurnal/Synoptic Heads + LeakyReLU backbone.",
        "category": "Apex Next-Gen Synthesis"
    }
}


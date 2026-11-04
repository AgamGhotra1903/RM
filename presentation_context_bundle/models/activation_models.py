"""
models/activation_models.py
===========================
Factory for Broad-UNet architecture with configurable global backbone activation function:
['relu', 'gelu', 'silu', 'tanh', 'leaky_relu'].

Every internal non-linear activation in encoder, multiscale blocks, ASPP, and decoder
is switched to the specified activation function, while keeping parameters identical (~10.64M)
to isolate the empirical impact of the activation function on multi-horizon temperature forecasting.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Add, Dropout, concatenate, Activation, Reshape,
    Conv3D, MaxPool3D, UpSampling3D, GlobalAveragePooling3D
)
from models.modules import AnchorResidualBroadcast

ACTIVATION_DESCRIPTIONS = {
    "relu": "Standard Rectified Linear Unit (Baseline Reference, piecewise linear max(0, x))",
    "gelu": "Gaussian Error Linear Unit (Smooth probabilistic gating x * Phi(x), standard in Transformers)",
    "silu": "Sigmoid Linear Unit / Swish (Smooth self-gated x * sigmoid(x), used in KAN base branch)",
    "tanh": "Hyperbolic Tangent (Symmetric bounded non-linearity (-1, 1), tests vanishing gradients/saturation)",
    "leaky_relu": "Leaky ReLU (Non-zero slope 0.2 for x < 0, prevents dying neurons)"
}

def build_broad_unet_with_activation(
    activation: str = "relu",
    lags: int = 6,
    latitude: int = 32,
    longitude: int = 48,
    features: int = 10,
    features_output: int = 7,
    filters: int = 16,
    dropout: float = 0.2,
    residual_head: bool = True,
    seed: int = 42
) -> Model:
    """
    Builds the official Broad-UNet backbone parameterized by a global activation function.
    """
    kernel_init = tf.keras.initializers.GlorotUniform(seed=seed)

    def image_level_feature_pooling(x, f):
        up_size = x.shape[2:4] if hasattr(x, "shape") else x.get_shape().as_list()[2:4]
        x = GlobalAveragePooling3D()(x)
        x = Reshape((1, 1, 1, f))(x)
        x = UpSampling3D(size=(1, int(up_size[0]), int(up_size[1])))(x)
        return x

    def ASPP_block(x, f):
        x1 = Conv3D(f, 1, activation=activation, padding="same", kernel_initializer=kernel_init)(x)
        x2 = Conv3D(f, (1, 3, 3), activation=activation, padding="same", dilation_rate=6, kernel_initializer=kernel_init)(x)
        x3 = Conv3D(f, (1, 3, 3), activation=activation, padding="same", dilation_rate=12, kernel_initializer=kernel_init)(x)
        x4 = Conv3D(f, (1, 3, 3), activation=activation, padding="same", dilation_rate=18, kernel_initializer=kernel_init)(x)
        x5 = image_level_feature_pooling(x, f)
        x = concatenate([x1, x2, x3, x4, x5], axis=-1)
        x = Conv3D(f, 1, activation=activation, padding="same", kernel_initializer=kernel_init)(x)
        return x

    def convolutional_block(x, f, k):
        shortcut = x
        x1 = Conv3D(f, 1, activation=activation, padding="same", kernel_initializer=kernel_init)(x)

        x2 = Conv3D(f, (1, 1, 3), activation=activation, padding="same", kernel_initializer=kernel_init)(x)
        x2 = Conv3D(f, (1, 3, 1), activation=activation, padding="same", kernel_initializer=kernel_init)(x2)
        x2 = Conv3D(f, (3, 1, 1), activation=activation, padding="same", kernel_initializer=kernel_init)(x2)

        x3 = Conv3D(f, (1, 1, 5), activation=activation, padding="same", kernel_initializer=kernel_init)(x)
        x3 = Conv3D(f, (1, 5, 1), activation=activation, padding="same", kernel_initializer=kernel_init)(x3)
        x3 = Conv3D(f, (5, 1, 1), activation=activation, padding="same", kernel_initializer=kernel_init)(x3)

        x4 = Conv3D(f, (1, 1, 3), activation=activation, padding="same", kernel_initializer=kernel_init)(x)
        x4 = Conv3D(f, (1, 3, 1), activation=activation, padding="same", kernel_initializer=kernel_init)(x4)
        x4 = Conv3D(f, (3, 1, 1), activation=activation, padding="same", kernel_initializer=kernel_init)(x4)

        x5 = Conv3D(f, (1, 1, 3), activation=activation, padding="same", kernel_initializer=kernel_init)(x)
        x5 = Conv3D(f, (1, 3, 1), activation=activation, padding="same", kernel_initializer=kernel_init)(x5)
        x5 = Conv3D(f, (3, 1, 1), activation=activation, padding="same", kernel_initializer=kernel_init)(x5)

        x = concatenate([x1, x2, x3, x4, x5], axis=-1)
        x = Conv3D(f, 1, activation=activation, padding="same", kernel_initializer=kernel_init)(x)
        x = Add()([x, shortcut])
        x = Activation(activation)(x)
        return x

    # --- Input & Anchor Residual Broadcast ---
    inputs = Input(shape=(lags, latitude, longitude, features), name="meteorological_input")
    if residual_head:
        t0_anchor = AnchorResidualBroadcast(target_idx=0, num_outputs=features_output, name="anchor_residual_broadcast")(inputs)

    # --- Encoder Stage 1 ---
    conv1 = Conv3D(filters, 3, activation=activation, padding="same", kernel_initializer=kernel_init)(inputs)
    conv1 = convolutional_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)

    # --- Encoder Stage 2 ---
    conv2 = Conv3D(2 * filters, 3, activation=activation, padding="same", kernel_initializer=kernel_init)(pool1)
    conv2 = convolutional_block(conv2, 2 * filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)

    # --- Encoder Stage 3 ---
    conv3 = Conv3D(4 * filters, 3, activation=activation, padding="same", kernel_initializer=kernel_init)(pool2)
    conv3 = convolutional_block(conv3, 4 * filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)

    # --- Encoder Stage 4 ---
    conv4 = Conv3D(8 * filters, 3, activation=activation, padding="same", kernel_initializer=kernel_init)(pool3)
    conv4 = convolutional_block(conv4, 8 * filters, 3)
    drop4 = Dropout(dropout)(conv4)

    # --- Bottleneck Stage 5 ---
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16 * filters, 3, activation=activation, padding="same", kernel_initializer=kernel_init)(pool4)
    conv5 = convolutional_block(conv5, 16 * filters, 3)
    compressLags = Conv3D(16 * filters, (lags, 1, 1), activation=activation, padding="valid")(conv5)
    aspp = ASPP_block(compressLags, 16 * filters)
    drop5 = Dropout(dropout)(aspp)

    # --- Decoder Stage 6 ---
    up6 = Conv3D(8 * filters, 2, activation=activation, padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(drop5)
    )
    skip4 = Conv3D(8 * filters, (lags, 1, 1), activation=activation, padding="valid")(drop4)
    merge6 = concatenate([skip4, up6], axis=-1)
    conv6 = Conv3D(8 * filters, 3, activation=activation, padding="same", kernel_initializer=kernel_init)(merge6)
    conv6 = convolutional_block(conv6, 8 * filters, 3)

    # --- Decoder Stage 7 ---
    up7 = Conv3D(4 * filters, 2, activation=activation, padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv6)
    )
    skip3 = Conv3D(4 * filters, (lags, 1, 1), activation=activation, padding="valid")(conv3)
    merge7 = concatenate([skip3, up7], axis=-1)
    conv7 = Conv3D(4 * filters, 3, activation=activation, padding="same", kernel_initializer=kernel_init)(merge7)
    conv7 = convolutional_block(conv7, 4 * filters, 3)

    # --- Decoder Stage 8 ---
    up8 = Conv3D(2 * filters, 2, activation=activation, padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv7)
    )
    skip2 = Conv3D(2 * filters, (lags, 1, 1), activation=activation, padding="valid")(conv2)
    merge8 = concatenate([skip2, up8], axis=-1)
    conv8 = Conv3D(2 * filters, 3, activation=activation, padding="same", kernel_initializer=kernel_init)(merge8)
    conv8 = convolutional_block(conv8, 2 * filters, 3)

    # --- Decoder Stage 9 ---
    up9 = Conv3D(filters, 2, activation=activation, padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv8)
    )
    skip1 = Conv3D(filters, (lags, 1, 1), activation=activation, padding="valid")(conv1)
    merge9 = concatenate([skip1, up9], axis=-1)
    conv9 = Conv3D(filters, 3, activation=activation, padding="same", kernel_initializer=kernel_init)(merge9)
    conv9 = convolutional_block(conv9, filters, 3)
    conv9 = Conv3D(2 * features_output, 3, activation=activation, padding="same", kernel_initializer=kernel_init)(conv9)

    # --- Final Output Layer (Linear) ---
    conv10 = Conv3D(features_output, 1, activation="linear", padding="same", name="conv10")(conv9)

    if residual_head:
        final_output = Add(name="residual_temperature_prediction")([t0_anchor, conv10])
    else:
        final_output = conv10

    model = Model(inputs=inputs, outputs=final_output, name=f"Broad_UNet_act_{activation}")
    return model


"""
models/broad_unet.py
====================
Broad-UNet architecture module conforming strictly to:
Fernández & Mehrkanoon, "Broad-UNet: Multi-scale feature learning for nowcasting tasks",
Neural Networks, 144, 2021, pp. 419-427.

Contains:
- build_broad_unet: 3D Multi-scale factorized conv blocks + ASPP bottleneck + temporal skip reduction.
- build_unet_baseline: 2D U-Net baseline (UNet_original).
- MaskedMSELoss: Loss function ignoring artificial edge padding.
- MaskedMAE: Evaluation metric ignoring artificial edge padding.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Add, Dropout, concatenate, Activation, Reshape,
    Conv3D, MaxPool3D, UpSampling3D, GlobalAveragePooling3D,
    Conv2D, MaxPool2D, UpSampling2D
)

def build_broad_unet(
    lags: int = 6,
    latitude: int = 32,
    longitude: int = 48,
    features: int = 6,
    features_output: int = 1,
    filters: int = 16,
    dropout: float = 0.5,
    activation: str = "linear",
    seed: int = 50
):
    """
    Constructs the official Broad-UNet model.
    
    Args:
        lags: Number of antecedent time steps (T = 6)
        latitude: Padded latitude dimension (32)
        longitude: Padded longitude dimension (48)
        features: Number of meteorological input channels (6)
        features_output: Number of prediction targets / horizons (1 or 5)
        filters: Base convolutional filter count (16)
        dropout: Spatial dropout probability (0.5)
        activation: Output layer activation ('linear' for continuous temperature regression)
        seed: Random initializer seed (50)
    """
    kernel_init = tf.keras.initializers.GlorotUniform(seed=seed)

    def image_level_feature_pooling(x, f):
        # Keras 2 / Keras 3 compatibility
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

    def convolutional_block(x, f, k):
        shortcut = x
        x1 = Conv3D(f, 1, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        
        # Branch 2: Factorized 3x3x3
        x2 = Conv3D(f, (1, 1, 3), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x2 = Conv3D(f, (1, 3, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x2)
        x2 = Conv3D(f, (3, 1, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x2)
        
        # Branch 3: Factorized 5x5x5
        x3 = Conv3D(f, (1, 1, 5), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x3 = Conv3D(f, (1, 5, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x3)
        x3 = Conv3D(f, (5, 1, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x3)
        
        # Branch 4: Factorized 3x3x3
        x4 = Conv3D(f, (1, 1, 3), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x4 = Conv3D(f, (1, 3, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x4)
        x4 = Conv3D(f, (3, 1, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x4)
        
        # Branch 5: Factorized 3x3x3
        x5 = Conv3D(f, (1, 1, 3), activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x5 = Conv3D(f, (1, 3, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x5)
        x5 = Conv3D(f, (3, 1, 1), activation="relu", padding="same", kernel_initializer=kernel_init)(x5)
        
        x = concatenate([x1, x2, x3, x4, x5], axis=-1)
        x = Conv3D(f, 1, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        return x

    # --- Contracting Part (Encoder) ---
    inputs = Input(shape=(lags, latitude, longitude, features), name="input_frames")
    
    conv1 = Conv3D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(inputs)
    conv1 = convolutional_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool1)
    conv2 = convolutional_block(conv2, 2 * filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool2)
    conv3 = convolutional_block(conv3, 4 * filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool3)
    conv4 = convolutional_block(conv4, 8 * filters, 3)
    drop4 = Dropout(dropout)(conv4)
    
    # --- Bottleneck Part ---
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool4)
    conv5 = convolutional_block(conv5, 16 * filters, 3)
    compressLags = Conv3D(16 * filters, (lags, 1, 1), activation="relu", padding="valid")(conv5)
    aspp = ASPP_block(compressLags, 16 * filters)
    drop5 = Dropout(dropout)(aspp)
    
    # --- Expanding Part (Decoder) with Temporal Skip Reductions ---
    up6 = Conv3D(8 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(drop5)
    )
    compressLags6 = Conv3D(8 * filters, (lags, 1, 1), activation="relu", padding="valid")(drop4)
    merge6 = concatenate([compressLags6, up6], axis=-1)
    conv6 = Conv3D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge6)
    conv6 = convolutional_block(conv6, 8 * filters, 3)

    up7 = Conv3D(4 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv6)
    )
    compressLags7 = Conv3D(4 * filters, (lags, 1, 1), activation="relu", padding="valid")(conv3)
    merge7 = concatenate([compressLags7, up7], axis=-1)
    conv7 = Conv3D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge7)
    conv7 = convolutional_block(conv7, 4 * filters, 3)

    up8 = Conv3D(2 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv7)
    )
    compressLags8 = Conv3D(2 * filters, (lags, 1, 1), activation="relu", padding="valid")(conv2)
    merge8 = concatenate([compressLags8, up8], axis=-1)
    conv8 = Conv3D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge8)
    conv8 = convolutional_block(conv8, 2 * filters, 3)

    up9 = Conv3D(filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
        UpSampling3D(size=(1, 2, 2))(conv8)
    )
    compressLags9 = Conv3D(filters, (lags, 1, 1), activation="relu", padding="valid")(conv1)
    merge9 = concatenate([compressLags9, up9], axis=-1)
    conv9 = Conv3D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge9)
    conv9 = convolutional_block(conv9, filters, 3)
    conv9 = Conv3D(2 * features_output, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv9)
    
    # Final projection layer (shape: (batch, 1, lat, lon, features_output))
    conv10 = Conv3D(features_output, 1, activation=activation, padding="same", name="predicted_temperature_map")(conv9)

    model = Model(inputs=inputs, outputs=conv10, name="Broad_UNet")
    return model


def build_unet_baseline(
    lags: int = 6,
    latitude: int = 32,
    longitude: int = 48,
    features: int = 6,
    features_output: int = 1,
    filters: int = 16,
    dropout: float = 0.5,
    activation: str = "linear",
    seed: int = 50
):
    """
    Constructs the 2D U-Net baseline from the official repository (UNet_original).
    Takes flattened temporal features as 2D channels: (batch, lat, lon, lags * features).
    """
    kernel_init = tf.keras.initializers.GlorotUniform(seed=seed)
    in_channels = lags * features
    inputs = Input(shape=(latitude, longitude, in_channels), name="input_frames_2d")
    
    conv1 = Conv2D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(inputs)
    conv1 = Conv2D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool1)
    conv2 = Conv2D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool2)
    conv3 = Conv2D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool3)
    conv4 = Conv2D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv4)
    drop4 = Dropout(dropout)(conv4)
    
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(16 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool4)
    conv5 = Conv2D(16 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(pool4)
    drop5 = Dropout(dropout)(conv5)
    
    up6 = Conv2D(8 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
        UpSampling2D(size=(2, 2))(drop5)
    )
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv2D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge6)
    conv6 = Conv2D(8 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv6)

    up7 = Conv2D(4 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
        UpSampling2D(size=(2, 2))(conv6)
    )
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv2D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge7)
    conv7 = Conv2D(4 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv7)

    up8 = Conv2D(2 * filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
        UpSampling2D(size=(2, 2))(conv7)
    )
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv2D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge8)
    conv8 = Conv2D(2 * filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv8)

    up9 = Conv2D(filters, 2, activation="relu", padding="same", kernel_initializer=kernel_init)(
        UpSampling2D(size=(2, 2))(conv8)
    )
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv2D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(merge9)
    conv9 = Conv2D(filters, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv9)
    conv9 = Conv2D(2 * features_output, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(conv9)
    
    conv10 = Conv2D(features_output, 1, activation=activation, padding="same", name="predicted_temperature_map_2d")(conv9)
    
    model = Model(inputs=inputs, outputs=conv10, name="UNet_original")
    return model


class MaskedMSELoss(tf.keras.losses.Loss):
    """
    Computes Mean Squared Error strictly over the valid unpadded geographic region.
    Padded boundary cells are masked out and do not generate loss or backpropagate gradients.
    """
    def __init__(self, mask_2d, name="masked_mse"):
        super().__init__(name=name)
        # mask_2d shape: (32, 48) with 1.0 on inner (28, 34) and 0.0 on padding
        # Reshape to (1, 1, 32, 48, 1) for 3D broadcasting
        self.mask = tf.cast(tf.constant(mask_2d), tf.float32)
        self.mask_3d = tf.reshape(self.mask, (1, 1, mask_2d.shape[0], mask_2d.shape[1], 1))
        self.mask_2d_loss = tf.reshape(self.mask, (1, mask_2d.shape[0], mask_2d.shape[1], 1))

    def call(self, y_true, y_pred):
        if len(y_pred.shape) == 5:
            mask = self.mask_3d
        else:
            mask = self.mask_2d_loss
        diff_sq = tf.square(y_true - y_pred) * mask
        return tf.reduce_sum(diff_sq) / tf.maximum(tf.reduce_sum(mask) * tf.cast(tf.shape(y_true)[0], tf.float32), 1.0)


class MaskedMAE(tf.keras.metrics.Metric):
    """
    Computes Mean Absolute Error strictly over the valid unpadded geographic region.
    """
    def __init__(self, mask_2d, name="masked_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mask = tf.cast(tf.constant(mask_2d), tf.float32)
        self.mask_3d = tf.reshape(self.mask, (1, 1, mask_2d.shape[0], mask_2d.shape[1], 1))
        self.mask_2d_m = tf.reshape(self.mask, (1, mask_2d.shape[0], mask_2d.shape[1], 1))
        self.total_abs_err = self.add_weight(name="total_abs_err", initializer="zeros")
        self.total_valid_pixels = self.add_weight(name="total_valid_pixels", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_pred.shape) == 5:
            mask = self.mask_3d
        else:
            mask = self.mask_2d_m
        abs_err = tf.abs(y_true - y_pred) * mask
        self.total_abs_err.assign_add(tf.reduce_sum(abs_err))
        self.total_valid_pixels.assign_add(tf.reduce_sum(mask) * tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.total_abs_err / tf.maximum(self.total_valid_pixels, 1.0)

    def reset_state(self):
        self.total_abs_err.assign(0.0)
        self.total_valid_pixels.assign(0.0)


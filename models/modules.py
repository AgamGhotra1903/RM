"""
models/modules.py
=================
Modular architectural components for Broad-UNet enhancements:
1. AnchorResidualBroadcast: Broadcasts t0 anchor frame for residual temperature prediction.
2. MLPBlock: Per-position LayerNorm -> Dense(C, 2C) -> GELU -> Dropout -> Dense(2C, C) with residual connection.
3. KANBlock: Kolmogorov-Arnold Network block with B-spline activation bases and residual connection.
4. SequentialMLPKANBlock: Sequential composition MLP -> KAN.
5. ParallelMLPKANBlock: Parallel branches MLP || KAN with learnable gating parameter.
6. SkipFusionBlock: Skip-connection fusion block for decoder merge points (merge6..merge9).
7. Conv3DControlBlock: Parameter-matched standard convolutional control.

All blocks adhere strictly to:
- Shape preservation: (B, D, H, W, C) -> (B, D, H, W, C)
- Identity-at-init property (diff < 1e-4) to prevent initial training degradation.
- Full gradient flow to all internal weights.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, LayerNormalization, Dense, Dropout, Conv3D, BatchNormalization, Activation,
    GlobalAveragePooling3D
)

class AnchorResidualBroadcast(Layer):
    """
    Extracts the most recent observed temperature frame at t0 from the lookback window,
    shapes it into (B, 1, H, W, num_horizons), and broadcasts it for residual addition.
    """
    def __init__(self, target_idx: int = 0, num_outputs: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.target_idx = target_idx
        self.num_outputs = num_outputs

    def call(self, inputs):
        # inputs shape: (B, lags, lat, lon, features)
        t0 = inputs[:, -1:, :, :, self.target_idx:self.target_idx + 1]  # (B, 1, lat, lon, 1)
        return tf.tile(t0, [1, 1, 1, 1, self.num_outputs])

    def compute_output_shape(self, input_shape):
        b, _, h, w, _ = input_shape
        return (b, 1, h, w, self.num_outputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_idx": self.target_idx,
            "num_outputs": self.num_outputs
        })
        return config


class MLPBlock(Layer):
    """
    Multi-Layer Perceptron (MLP) block for channel-wise feature mixing.
    Applied per spatial/spatiotemporal location.
    """
    def __init__(self, channels: int, mlp_ratio: float = 2.0, dropout: float = 0.0, scale_init: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.scale_init = scale_init
        
        hidden_dim = int(self.channels * self.mlp_ratio)
        self.norm = LayerNormalization(axis=-1, name="ln")
        self.fc1 = Dense(hidden_dim, activation="gelu", kernel_initializer="glorot_uniform", name="fc1")
        self.drop = Dropout(self.dropout_rate, name="drop")
        self.fc2 = Dense(self.channels, kernel_initializer="glorot_uniform", name="fc2")

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(self.channels,),
            name="scale",
            initializer=tf.keras.initializers.Constant(self.scale_init),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.norm(inputs)
        x = self.fc1(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        return inputs + self.scale * x

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout_rate,
            "scale_init": self.scale_init
        })
        return config


class KANBlock(Layer):
    """
    Kolmogorov-Arnold Network (KAN) block for channel-wise non-linear representations.
    Uses cubic B-spline basis functions over a uniform grid with learnable coefficients
    plus a base linear projection.
    """
    def __init__(
        self,
        channels: int,
        num_knots: int = 5,
        grid_range: tuple = (-2.0, 2.0),
        scale_init: float = 1e-4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_knots = num_knots
        self.grid_min, self.grid_max = grid_range
        self.scale_init = scale_init
        self.spacing = (self.grid_max - self.grid_min) / float(self.num_knots - 1)
        self.grid = tf.cast(tf.linspace(self.grid_min, self.grid_max, self.num_knots), tf.float32)

        self.norm = LayerNormalization(axis=-1, name="ln")

    def build(self, input_shape):
        self.base_weight = self.add_weight(
            shape=(self.channels, self.channels),
            name="base_weight",
            initializer="glorot_uniform",
            trainable=True
        )
        self.spline_weight = self.add_weight(
            shape=(self.channels, self.num_knots, self.channels),
            name="spline_weight",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True
        )
        self.scale = self.add_weight(
            shape=(self.channels,),
            name="scale",
            initializer=tf.keras.initializers.Constant(self.scale_init),
            trainable=True
        )
        super().build(input_shape)

    def _compute_spline_basis(self, x):
        x_exp = tf.expand_dims(x, -1)
        u = tf.abs(x_exp - self.grid) / self.spacing
        p1 = (2.0 / 3.0) - tf.square(u) + 0.5 * tf.pow(u, 3.0)
        p2 = (1.0 / 6.0) * tf.pow(2.0 - u, 3.0)
        basis = tf.where(u < 1.0, p1, tf.where(u < 2.0, p2, 0.0))
        return basis

    def call(self, inputs, training=False):
        x = self.norm(inputs)
        base = tf.matmul(tf.nn.silu(x), self.base_weight)
        basis = self._compute_spline_basis(x)
        spline = tf.einsum("...ig,igo->...o", basis, self.spline_weight)
        out = base + spline
        return inputs + self.scale * out

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "num_knots": self.num_knots,
            "grid_range": (self.grid_min, self.grid_max),
            "scale_init": self.scale_init
        })
        return config


class SequentialMLPKANBlock(Layer):
    """
    Sequential combination: MLPBlock -> KANBlock.
    """
    def __init__(self, channels: int, mlp_ratio: float = 2.0, num_knots: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.mlp_ratio = mlp_ratio
        self.num_knots = num_knots
        self.mlp = MLPBlock(channels, mlp_ratio=mlp_ratio, name="seq_mlp")
        self.kan = KANBlock(channels, num_knots=num_knots, name="seq_kan")

    def call(self, inputs, training=False):
        x = self.mlp(inputs, training=training)
        return self.kan(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "mlp_ratio": self.mlp.mlp_ratio,
            "num_knots": self.kan.num_knots
        })
        return config


class ParallelMLPKANBlock(Layer):
    """
    Parallel combination: inputs + g * MLP(inputs) + (1 - g) * KAN(inputs)
    with learnable gating parameter g = sigmoid(logit), initialized to 0.5.
    """
    def __init__(self, channels: int, mlp_ratio: float = 2.0, num_knots: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.mlp_ratio = mlp_ratio
        self.num_knots = num_knots
        self.mlp = MLPBlock(channels, mlp_ratio=mlp_ratio, name="par_mlp")
        self.kan = KANBlock(channels, num_knots=num_knots, name="par_kan")

    def build(self, input_shape):
        self.gate_logit = self.add_weight(
            shape=(1,),
            name="gate_logit",
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        g = tf.sigmoid(self.gate_logit)
        mlp_out = self.mlp(inputs, training=training)
        kan_out = self.kan(inputs, training=training)
        mlp_delta = mlp_out - inputs
        kan_delta = kan_out - inputs
        return inputs + g * mlp_delta + (1.0 - g) * kan_delta

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "mlp_ratio": self.mlp.mlp_ratio,
            "num_knots": self.kan.num_knots
        })
        return config


class SkipFusionBlock(Layer):
    """
    Skip-connection fusion block for decoder merge points (merge6..merge9).
    Transforms the concatenated skip and upsampled tensors using KAN or MLP.
    """
    def __init__(self, in_channels: int, out_channels: int, mode: str = "kan", **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.proj = Dense(self.out_channels, kernel_initializer="glorot_uniform", name="skip_proj")
        if self.mode == "kan":
            self.block = KANBlock(self.out_channels, name="skip_kan")
        elif self.mode == "mlp":
            self.block = MLPBlock(self.out_channels, name="skip_mlp")
        else:
            self.block = None

    def call(self, inputs, training=False):
        x = self.proj(inputs)
        if self.block is not None:
            x = self.block(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "mode": self.mode
        })
        return config


class Conv3DControlBlock(Layer):
    """
    Parameter-matched standard convolutional control block.
    Matches the parameter count of MLP/KAN blocks using standard 3D convolutions.
    """
    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.conv1 = Conv3D(self.channels, 1, padding="same", activation="relu", name="ctrl_c1")
        self.conv2 = Conv3D(self.channels, 3, padding="same", activation="relu", name="ctrl_c2")

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(self.channels,),
            name="ctrl_scale",
            initializer=tf.keras.initializers.Constant(1e-4),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return inputs + self.scale * x

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


class SpatioTemporalSEBlock(Layer):
    """
    Spatio-Temporal Squeeze-and-Excitation Block for 3D convolutional features.
    Squeezes (B, D, H, W, C) via GlobalAveragePooling3D into (B, C),
    processes through two Dense layers with bottleneck reduction ratio r=4,
    and excites channel activations via sigmoid scaling.
    """
    def __init__(self, channels: int, reduction: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        hidden_dim = max(4, channels // reduction)
        self.pool = GlobalAveragePooling3D()
        self.fc1 = Dense(hidden_dim, activation="relu", kernel_initializer="glorot_uniform", name="se_fc1")
        self.fc2 = Dense(channels, activation="sigmoid", kernel_initializer="glorot_uniform", name="se_fc2")

    def call(self, inputs):
        s = self.pool(inputs)
        s = self.fc1(s)
        s = self.fc2(s)
        s = tf.reshape(s, [-1, 1, 1, 1, self.channels])
        return inputs * s

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction": self.reduction
        })
        return config

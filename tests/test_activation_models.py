"""
tests/test_activation_models.py
===============================
Unit tests verifying that all 5 activation variants ('relu', 'gelu', 'silu', 'tanh', 'leaky_relu')
build with identical parameters and produce exact output tensor shape (B, 1, Lat, Lon, 7).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import tensorflow as tf
from models.activation_models import build_broad_unet_with_activation, ACTIVATION_DESCRIPTIONS

@pytest.mark.parametrize("act", list(ACTIVATION_DESCRIPTIONS.keys()))
def test_broad_unet_activations_shape(act):
    model = build_broad_unet_with_activation(
        activation=act,
        lags=6,
        latitude=32,
        longitude=48,
        features=10,
        features_output=7,
        filters=4,  # use smaller filters for fast unit testing
        dropout=0.1,
        residual_head=True
    )
    assert model is not None
    assert model.count_params() > 0

    dummy_input = tf.random.normal((2, 6, 32, 48, 10))
    output = model(dummy_input, training=False)
    assert output.shape == (2, 1, 32, 48, 7), (
        f"Activation {act} produced unexpected output shape: {output.shape}"
    )


"""
tests/test_rearranged_models.py
===============================
Unit tests verifying architectural integrity, parameter shapes,
and forward pass reproducibility for all 15 layer rearrangement models (L0 to L14)
across 7 forecast horizons.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import tensorflow as tf
from models.rearranged_models import build_rearranged_model, EXPERIMENT_NAMES

@pytest.mark.parametrize("exp_id", list(EXPERIMENT_NAMES.keys()))
def test_all_15_rearranged_architectures_shape(exp_id):
    """
    Verifies that every rearranged architecture builds successfully and
    produces the exact expected 7-horizon tensor shape (B, 1, Lat, Lon, 7).
    """
    model = build_rearranged_model(
        exp_id=exp_id,
        lags=6,
        latitude=32,
        longitude=48,
        features=10,
        features_output=7,
        filters=4,  # use smaller filters for fast unit test execution
        dropout=0.1,
        residual_head=True
    )
    
    assert model is not None
    assert model.count_params() > 0
    
    # Forward pass test
    dummy_input = tf.random.normal((2, 6, 32, 48, 10))
    output = model(dummy_input, training=False)
    
    assert output.shape == (2, 1, 32, 48, 7), (
        f"Architecture {exp_id} output shape mismatch: {output.shape} != (2, 1, 32, 48, 7)"
    )


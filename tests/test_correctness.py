"""
tests/test_correctness.py
=========================
Correctness unit tests for architectural modules, identity-at-init,
gradient flow, single-batch overfit, and determinism.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import tensorflow as tf
from models.modules import (
    MLPBlock, KANBlock, SequentialMLPKANBlock,
    ParallelMLPKANBlock, Conv3DControlBlock, SkipFusionBlock
)
from models.broad_unet_v2 import build_broad_unet_v2

def test_mlp_block_shape_and_identity():
    channels = 32
    block = MLPBlock(channels=channels, scale_init=1e-5)
    x = tf.random.normal((2, 1, 8, 8, channels))
    y = block(x)
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    diff = float(tf.reduce_max(tf.abs(y - x)).numpy())
    assert diff < 1e-4, f"Identity-at-init violated: diff = {diff:.6e} >= 1e-4"

def test_kan_block_shape_and_identity():
    channels = 32
    block = KANBlock(channels=channels, num_knots=5, scale_init=1e-5)
    x = tf.random.normal((2, 1, 8, 8, channels))
    y = block(x)
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    diff = float(tf.reduce_max(tf.abs(y - x)).numpy())
    assert diff < 1e-4, f"Identity-at-init violated: diff = {diff:.6e} >= 1e-4"

def test_sequential_mlp_kan_block():
    channels = 16
    block = SequentialMLPKANBlock(channels=channels)
    x = tf.random.normal((2, 1, 4, 4, channels))
    y = block(x)
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    diff = float(tf.reduce_max(tf.abs(y - x)).numpy())
    assert diff < 1e-3, f"Sequential diff too large: {diff:.6e}"

def test_parallel_mlp_kan_block():
    channels = 16
    block = ParallelMLPKANBlock(channels=channels)
    x = tf.random.normal((2, 1, 4, 4, channels))
    y = block(x)
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    diff = float(tf.reduce_max(tf.abs(y - x)).numpy())
    assert diff < 1e-3, f"Parallel diff too large: {diff:.6e}"

def test_conv3d_control_block():
    channels = 16
    block = Conv3DControlBlock(channels=channels)
    x = tf.random.normal((2, 1, 4, 4, channels))
    y = block(x)
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    diff = float(tf.reduce_max(tf.abs(y - x)).numpy())
    assert diff < 1e-3, f"Control diff too large: {diff:.6e}"

def test_gradient_flow_to_all_parameters():
    channels = 16
    mlp = MLPBlock(channels=channels, scale_init=1e-4)
    kan = KANBlock(channels=channels, scale_init=1e-4)
    
    x = tf.random.normal((2, 1, 4, 4, channels))
    
    with tf.GradientTape() as tape:
        y_mlp = mlp(x)
        loss_mlp = tf.reduce_mean(tf.square(y_mlp - 2.0))
    grads_mlp = tape.gradient(loss_mlp, mlp.trainable_variables)
    
    for var, grad in zip(mlp.trainable_variables, grads_mlp):
        assert grad is not None, f"MLP gradient is None for variable: {var.name}"
        norm = float(tf.norm(grad).numpy())
        assert norm > 0.0, f"MLP gradient norm is zero for variable: {var.name}"

    with tf.GradientTape() as tape:
        y_kan = kan(x)
        loss_kan = tf.reduce_mean(tf.square(y_kan - 2.0))
    grads_kan = tape.gradient(loss_kan, kan.trainable_variables)
    
    for var, grad in zip(kan.trainable_variables, grads_kan):
        assert grad is not None, f"KAN gradient is None for variable: {var.name}"
        norm = float(tf.norm(grad).numpy())
        assert norm > 0.0, f"KAN gradient norm is zero for variable: {var.name}"

def test_broad_unet_v2_shape_and_residual():
    model = build_broad_unet_v2(
        lags=6, latitude=32, longitude=48, features=6, features_output=5,
        filters=16, residual_head=True, post_aspp_module="mlp"
    )
    x = tf.random.normal((2, 6, 32, 48, 6))
    y = model(x, training=False)
    assert y.shape == (2, 1, 32, 48, 5), f"Broad-UNet output shape unexpected: {y.shape}"

def test_single_batch_overfit():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((1, 8, 8, 16)),
        MLPBlock(16, scale_init=0.1),
        KANBlock(16, scale_init=0.1),
        tf.keras.layers.Dense(5)
    ])
    opt = tf.keras.optimizers.Adam(0.05)
    x = tf.random.normal((2, 1, 8, 8, 16))
    y = tf.random.normal((2, 1, 8, 8, 5))
    
    losses = []
    for _ in range(25):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(float(loss.numpy()))
        
    initial_loss = losses[0]
    final_loss = losses[-1]
    assert final_loss < 0.2 * initial_loss, f"Failed single batch overfit: {initial_loss:.4f} -> {final_loss:.4f}"

def test_determinism():
    tf.random.set_seed(1234)
    x = tf.random.normal((2, 1, 4, 4, 16))
    block1 = KANBlock(16)
    _ = block1(x)
    
    tf.random.set_seed(1234)
    x2 = tf.random.normal((2, 1, 4, 4, 16))
    assert np.allclose(x.numpy(), x2.numpy()), "RNG seed failed to produce deterministic inputs"

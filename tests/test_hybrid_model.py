"""
tests/test_hybrid_model.py
==========================
Unit tests for the Hybrid KAN + LeakyReLU Broad-UNet architecture.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.hybrid_model import build_hybrid_kan_leaky_relu

class TestHybridModel(unittest.TestCase):
    def test_hybrid_forward_pass_and_shape(self):
        model = build_hybrid_kan_leaky_relu(
            lags=6, latitude=32, longitude=48, features=10, features_output=7,
            filters=16, dropout=0.2, residual_head=True
        )
        dummy_x = tf.random.normal((2, 6, 32, 48, 10))
        out = model(dummy_x, training=False)
        self.assertEqual(out.shape, (2, 1, 32, 48, 7))
        print(f"Hybrid Model Parameter Count: {model.count_params():,}")
        self.assertTrue(10_000_000 < model.count_params() < 10_500_000)

    def test_gradient_flow(self):
        model = build_hybrid_kan_leaky_relu(
            lags=6, latitude=32, longitude=48, features=10, features_output=7,
            filters=16, dropout=0.2, residual_head=True
        )
        dummy_x = tf.random.normal((2, 6, 32, 48, 10))
        dummy_y = tf.random.normal((2, 1, 32, 48, 7))

        with tf.GradientTape() as tape:
            pred = model(dummy_x, training=True)
            loss = tf.reduce_mean(tf.square(pred - dummy_y))

        grads = tape.gradient(loss, model.trainable_weights)
        self.assertEqual(len(grads), len(model.trainable_weights))
        none_grads = [w.name for w, g in zip(model.trainable_weights, grads) if g is None]
        self.assertEqual(len(none_grads), 0, f"Variables with None gradient: {none_grads}")
        print("All trainable weights received non-None gradients successfully!")

if __name__ == "__main__":
    unittest.main()

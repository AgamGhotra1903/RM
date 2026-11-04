"""
scripts/print_model_graph.py
============================
A pure-Python graph visualizer and architectural explainer for:
1. The basic (original upstream) Broad-UNet model.
2. All derivative models (L0 through L14 layer rearrangements).
3. The Champion Hybrid model (L15: KAN Skip Fusion + LeakyReLU backbone).
4. The global activation variants (ReLU, GELU, SiLU, Tanh, LeakyReLU).

Includes:
- Textual and physical architectural explanations.
- Structured ASCII/Unicode computational graph trees with tensor shapes.
- Live model inspection (parameter counts, layer types, exact Keras graph connections).
- High-resolution 16:9 presentation diagram rendering.
- Slide deck generation for academic / TA presentations.

Usage:
  python scripts/print_model_graph.py --model basic --plot
  python scripts/print_model_graph.py --model L0 --plot
  python scripts/print_model_graph.py --model L11 --plot
  python scripts/print_model_graph.py --model L15 --plot
  python scripts/print_model_graph.py --presentation
  python scripts/print_model_graph.py --model all
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.broad_unet import build_broad_unet
from models.rearranged_models import build_rearranged_model, EXPERIMENT_NAMES
from models.activation_models import build_broad_unet_with_activation, ACTIVATION_DESCRIPTIONS
from models.hybrid_model import build_hybrid_kan_leaky_relu
from models.next_gen_models import build_next_gen_model
from models.modules import SkipFusionBlock, KANBlock, MLPBlock, SpatioTemporalSEBlock

MODELS_CATALOG = {
    "basic": {
        "name": "Basic Original Broad-UNet (jesusgf96 upstream)",
        "builder": lambda: build_broad_unet(lags=6, latitude=32, longitude=48, features=6, features_output=7, filters=16),
        "desc": "The raw reference architecture without residual anchor head (direct absolute prediction).",
        "category": "Baseline Benchmark",
        "has_anchor": False,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 10642887,
        "val_rmse": "7.340 K"
    },
    "L0": {
        "name": "L0_original: Hardened Baseline Broad-UNet",
        "builder": lambda: build_rearranged_model("L0_original", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Official Broad-UNet backbone equipped with the Anchor Residual Prediction Head (Yhat = T_t0 + Delta T).",
        "category": "Reference Baseline",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 10642935,
        "val_rmse": "3.520 K"
    },
    "L1": {
        "name": "L1_conv_rearrangement",
        "builder": lambda: build_rearranged_model("L1_conv_rearrangement", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Reordered encoder blocks: Multi-scale branches evaluated BEFORE initial Conv3D.",
        "category": "Encoder Rearrangement",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 10642935,
        "val_rmse": "3.535 K"
    },
    "L2": {
        "name": "L2_conv_before_aspp",
        "builder": lambda: build_rearranged_model("L2_conv_before_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Additional 3D Convolution layer injected BEFORE the ASPP bottleneck.",
        "category": "Bottleneck Rearrangement",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 11233015,
        "val_rmse": "3.518 K"
    },
    "L3": {
        "name": "L3_conv_after_aspp",
        "builder": lambda: build_rearranged_model("L3_conv_after_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Additional 3D Convolution layer injected AFTER the ASPP bottleneck.",
        "category": "Bottleneck Rearrangement",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 11233015,
        "val_rmse": "3.515 K"
    },
    "L4": {
        "name": "L4_mlp_before_aspp",
        "builder": lambda: build_rearranged_model("L4_mlp_before_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Multi-Layer Perceptron (MLP) channel-mixing block inserted BEFORE ASPP.",
        "category": "Bottleneck Rearrangement",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 10905591,
        "val_rmse": "3.512 K"
    },
    "L5": {
        "name": "L5_mlp_after_aspp",
        "builder": lambda: build_rearranged_model("L5_mlp_after_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Multi-Layer Perceptron (MLP) channel-mixing block inserted AFTER ASPP.",
        "category": "Bottleneck Rearrangement",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 10905591,
        "val_rmse": "3.509 K"
    },
    "L6": {
        "name": "L6_kan_before_aspp",
        "builder": lambda: build_rearranged_model("L6_kan_before_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Kolmogorov-Arnold Network (KAN) block with B-spline bases inserted BEFORE ASPP.",
        "category": "Bottleneck Rearrangement",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 10839543,
        "val_rmse": "3.501 K"
    },
    "L7": {
        "name": "L7_kan_after_aspp",
        "builder": lambda: build_rearranged_model("L7_kan_after_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Kolmogorov-Arnold Network (KAN) block with B-spline bases inserted AFTER ASPP.",
        "category": "Bottleneck Rearrangement",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 10839543,
        "val_rmse": "3.498 K"
    },
    "L8": {
        "name": "L8_mlp_then_kan",
        "builder": lambda: build_rearranged_model("L8_mlp_then_kan", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Sequential composition: ASPP -> MLP -> KAN.",
        "category": "Bottleneck Compound",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 11102199,
        "val_rmse": "3.492 K"
    },
    "L9": {
        "name": "L9_kan_then_mlp",
        "builder": lambda: build_rearranged_model("L9_kan_then_mlp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Sequential composition: ASPP -> KAN -> MLP.",
        "category": "Bottleneck Compound",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 11102199,
        "val_rmse": "3.495 K"
    },
    "L10": {
        "name": "L10_before_skip_fusion",
        "builder": lambda: build_rearranged_model("L10_before_skip_fusion", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Pre-skip linear projection applied to encoder skip features before merge.",
        "category": "Skip Connection Rearrangement",
        "has_anchor": True,
        "skip_type": "pre_skip_linear",
        "activation": "ReLU",
        "params": 10661419,
        "val_rmse": "3.497 K"
    },
    "L11": {
        "name": "L11_after_skip_fusion",
        "builder": lambda: build_rearranged_model("L11_after_skip_fusion", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Post-skip non-linear KAN Fusion with B-spline bases at merge6 & merge7.",
        "category": "Skip Connection Rearrangement (Rank 2 Overall)",
        "has_anchor": True,
        "skip_type": "kan_fusion",
        "activation": "ReLU",
        "params": 10254583,
        "val_rmse": "3.427 K"
    },
    "L12": {
        "name": "L12_encoder_rearrangement",
        "builder": lambda: build_rearranged_model("L12_encoder_rearrangement", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Reordered normalization/activation: Conv3D -> BatchNorm -> ReLU.",
        "category": "Norm/Act Rearrangement",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 10643959,
        "val_rmse": "3.519 K"
    },
    "L13": {
        "name": "L13_decoder_rearrangement",
        "builder": lambda: build_rearranged_model("L13_decoder_rearrangement", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Reordered decoder blocks: Convolution evaluated BEFORE UpSampling3D.",
        "category": "Decoder Rearrangement",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "ReLU",
        "params": 10642935,
        "val_rmse": "3.684 K"
    },
    "L14": {
        "name": "L14_best_combined_arrangement",
        "builder": lambda: build_rearranged_model("L14_best_combined_arrangement", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Compound arrangement: Pre-ASPP Conv + KAN Skip Fusion + Norm-Act reordering.",
        "category": "Compound Architecture",
        "has_anchor": True,
        "skip_type": "kan_fusion",
        "activation": "ReLU",
        "params": 10845687,
        "val_rmse": "3.411 K"
    },
    "L15": {
        "name": "L15_hybrid_kan_leaky_relu",
        "builder": lambda: build_hybrid_kan_leaky_relu(lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Hybrid Synthesis: KAN Skip Fusion (merge6 & merge7) + LeakyReLU (alpha=0.2) backbone.",
        "category": "Tournament Hybrid (Rank 4 / 20)",
        "has_anchor": True,
        "skip_type": "kan_fusion",
        "activation": "LeakyReLU (0.2)",
        "params": 10254583,
        "val_rmse": "3.369 K"
    },
    "L16": {
        "name": "L16_full_hierarchy_kan_skips",
        "builder": lambda: build_next_gen_model("L16_full_hierarchy_kan_skips", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Full-Hierarchy 4-Stage KAN Skip Fusion across merges 6, 7, 8, 9 down to full spatial resolution.",
        "category": "Next-Gen Skip Hierarchy",
        "has_anchor": True,
        "skip_type": "kan_fusion",
        "activation": "LeakyReLU (0.2)",
        "params": 10230455,
        "val_rmse": "3.507 K"
    },
    "L17": {
        "name": "L17_decoupled_diurnal_synoptic_heads",
        "builder": lambda: build_next_gen_model("L17_decoupled_diurnal_synoptic_heads", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "APEX CHAMPION: Decoupled Diurnal (Tanh, +2h..+16h) and Synoptic (Linear, +24h..+72h) heads + KAN Skips.",
        "category": "Apex Champion (Rank 1 / 20)",
        "has_anchor": True,
        "skip_type": "kan_fusion",
        "activation": "LeakyReLU (0.2)",
        "params": 10254583,
        "val_rmse": "3.211 K"
    },
    "L18": {
        "name": "L18_spatiotemporal_se_multiscale",
        "builder": lambda: build_next_gen_model("L18_spatiotemporal_se_multiscale", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Spatio-Temporal Squeeze-and-Excitation (ST-SE) channel attention inside MultiScale Conv3D blocks.",
        "category": "Next-Gen Channel Attention (Rank 3 / 20)",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "LeakyReLU (0.2)",
        "params": 10310031,
        "val_rmse": "3.318 K"
    },
    "L19": {
        "name": "L19_apex_hybrid_kan_se_decoupled",
        "builder": lambda: build_next_gen_model("L19_apex_hybrid_kan_se_decoupled", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "APEX SYNTHESIS: Full KAN Skips + ST-SE Attention + Decoupled Diurnal/Synoptic Heads + LeakyReLU.",
        "category": "Apex Synthesis (Rank 2 / 20)",
        "has_anchor": True,
        "skip_type": "kan_fusion",
        "activation": "LeakyReLU (0.2)",
        "params": 10285903,
        "val_rmse": "3.241 K"
    },
    "act_tanh": {
        "name": "Broad-UNet with Tanh Backbone",
        "builder": lambda: build_broad_unet_with_activation("tanh", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Global Tanh activation (-1, 1). Best short-range diurnal model (+7.56% val gain).",
        "category": "Activation Variant",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "Tanh",
        "params": 10642935,
        "val_rmse": "3.374 K"
    },
    "act_leaky_relu": {
        "name": "Broad-UNet with LeakyReLU Backbone",
        "builder": lambda: build_broad_unet_with_activation("leaky_relu", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Global LeakyReLU (alpha=0.2). Best robust activation across all 72 hours (+4.17% gain).",
        "category": "Activation Variant",
        "has_anchor": True,
        "skip_type": "naive_concat",
        "activation": "LeakyReLU (0.2)",
        "params": 10642935,
        "val_rmse": "3.373 K"
    }
}


def explain_graph_structure(model_key: str):
    """
    Prints a detailed textual and physical explanation of the computational graph structure.
    """
    info = MODELS_CATALOG.get(model_key, MODELS_CATALOG.get(f"L{model_key}", None))
    if not info:
        print(f"Unknown model key: {model_key}")
        return

    name = info["name"]
    desc = info["desc"]
    cat = info["category"]

    print("=" * 88)
    print(f"COMPUTATIONAL GRAPH EXPLANATION: {name}")
    print(f"Category: {cat}")
    print(f"Summary : {desc}")
    print(f"Parameters: {info['params']:,} | Val RMSE: {info['val_rmse']}")
    print("=" * 88)

    print("\n1. TOPOLOGICAL OVERVIEW OF THE GRAPH:")
    print("   The Broad-UNet architecture forms a Directed Acyclic Graph (DAG) with five key topological regions:")
    print("   " + "-" * 84)
    print("   [Input Tensor] ---> [Encoder: 4 Downsampling Stages] ---> [Temporal Compression]")
    print("          |                         |                                  |")
    print("          |               (4 Long Skip Connections)                    v")
    print("          |                         |                           [Bottleneck / ASPP]")
    print("          |                         v                                  |")
    print("          |              [Decoder: 4 Upsampling Stages] <--------------+")
    print("          |                         |")
    print("          +------------------------>+---> [Output Head: 7 Forecast Horizons]")
    print("   " + "-" * 84)

    print("\n2. DEEP-DIVE INTO GRAPH MACRO-STAGES & TENSOR FLOW:")
    print("   Stage 0: Input & Anchor Broadcast")
    print("     - Meteorological input tensor: shape (B, lags=6, lat=32, lon=48, channels=10).")
    if info["has_anchor"]:
        print("     - AnchorResidualBroadcast node extracts latest frame T_t0 at index -1,")
        print("       broadcasting it as (B, 1, 32, 48, 7) directly to the loss head.")
    else:
        print("     - [!] NO ANCHOR BROADCAST: Directly regresses absolute values (causes severe cold collapse).")

    print("\n   Stage 1 to 4: Hierarchical Spatio-Temporal Encoder")
    print(f"     - Activation function: {info['activation']}")
    print("     - Each stage contains an initial Conv3D followed by a MultiScale Factorized Block.")
    print("     - The MultiScale Block splits feature flow into 5 parallel factorized branches:")
    print("       * Branch 1: 1x1x1 Conv3D (identity/channel mixing)")
    print("       * Branch 2: Factorized 1x1x3 -> 1x3x1 -> 3x1x1 (spatial height/width + temporal depth)")
    print("       * Branch 3: Factorized 1x1x5 -> 1x5x1 -> 5x1x1 (synoptic scale receptive field)")
    print("       * Branch 4 & 5: Auxiliary factorized 3D convolutions")
    print("       * Depth Concatenation + 1x1x1 Conv3D projection + Residual Shortcut Addition.")
    print("     - Spatial downsampling: MaxPool3D(pool_size=(1, 2, 2)) preserves temporal dimension (depth=6)")
    print("       while halving spatial resolution: (32, 48) -> (16, 24) -> (8, 12) -> (4, 6).")

    print("\n   Stage 5: Temporal Compression & Bottleneck (ASPP)")
    print("     - CompressLags node: Conv3D(256, kernel_size=(6, 1, 1), padding='valid') crushes temporal lags")
    print("       from 6 down to 1: (B, 6, 4, 6, 128) -> (B, 1, 4, 6, 256).")
    print("     - ASPP (Atrous Spatial Pyramid Pooling): Evaluates 5 parallel branches with dilation rates:")
    print("       * Rate 1 (1x1x1), Rate 6 (1x3x3), Rate 12 (1x3x3), Rate 18 (1x3x3), and Global Pooling.")
    print("       * Captures multi-scale synoptic weather systems simultaneously without losing spatial resolution.")

    print("\n   Stage 6 to 9: Symmetrical Upsampling Decoder & Skip Merges")
    print(f"     - Skip Fusion Mode: {info['skip_type'].upper()}")
    print("     - 4 Long Skip Connections bridge encoder stages to decoder stages:")
    print("       * Skip 4 (from Stage 4) fuses with Up6 at merge6  -> spatial (8, 12)")
    print("       * Skip 3 (from Stage 3) fuses with Up7 at merge7  -> spatial (16, 24)")
    print("       * Skip 2 (from Stage 2) fuses with Up8 at merge8  -> spatial (32, 48)")
    print("       * Skip 1 (from Stage 1) fuses with Up9 at merge9  -> spatial (32, 48)")
    if info["skip_type"] == "kan_fusion":
        print("       * KAN Skip Fusion: Cubic B-splines dynamically adapt activation curves at merge6 and merge7!")
    else:
        print("       * Naive Concatenation: Rigid linear blending followed by standard activation.")

    print("\n   Stage 10: Multi-Horizon Forecast Projection")
    print("     - Final Conv3D(7, kernel_size=1) projects features to the 7 lead times.")
    if info["has_anchor"]:
        print("     - Residual addition: Yhat = T_t0 + Delta_T.")
    else:
        print("     - Direct output: Yhat = Conv3D (Absolute Temperature).")


def print_ascii_graph(model_key: str):
    """
    Renders a clean, structured ASCII / Unicode tree graph of the model architecture.
    """
    info = MODELS_CATALOG.get(model_key, MODELS_CATALOG.get(f"L{model_key}", None))
    if not info:
        return

    print("\n" + "=" * 88)
    print(f"ASCII COMPUTATIONAL GRAPH TREE: {info['name']}")
    print("=" * 88)

    is_kan = info["skip_type"] == "kan_fusion"
    has_anchor = info["has_anchor"]
    act_str = info["activation"]

    graph_lines = [
        "  [Input: Meteorological ERA5 Windows]",
        "  Tensor Shape: (Batch, Lags=6, Lat=32, Lon=48, Channels=10)",
        "       │",
    ]

    if has_anchor:
        graph_lines.extend([
            "       ├───> [AnchorResidualBroadcast] ──────────────────────────────────────────────┐",
            "       │     Extracts latest t0 frame: (B, 1, 32, 48, 7)                             │",
            "       │                                                                             │",
        ])
    else:
        graph_lines.extend([
            "       │     [!] NO RESIDUAL ANCHOR (Direct absolute regression)                     │",
        ])

    graph_lines.extend([
        "       v",
        f"  [Encoder Stage 1] Filters: 16 | Activation: {act_str}",
        "       ├── Initial Conv3D (16f, 3x3x3)",
        "       ├── MultiScale Factorized Convolutions (1x1x3, 1x3x1, 3x1x1)",
        "       ├── Output Shape: (B, 6, 32, 48, 16)",
        "       ├───> [Skip 1 Branch] ──> Conv3D(16f, (6,1,1), valid) ──────> (B, 1, 32, 48, 16) ────────┐",
        "       │                                                                                      │",
        "       v [MaxPool3D (1, 2, 2)]                                                                │",
        f"  [Encoder Stage 2] Filters: 32 | Shape: (B, 6, 16, 24, 32) | Activation: {act_str}          │",
        "       ├── Initial Conv3D + MultiScale Factorized Convolutions                                │",
        "       ├───> [Skip 2 Branch] ──> Conv3D(32f, (6,1,1), valid) ──────> (B, 1, 16, 24, 32) ──┐  │",
        "       │                                                                                │  │",
        "       v [MaxPool3D (1, 2, 2)]                                                          │  │",
        f"  [Encoder Stage 3] Filters: 64 | Shape: (B, 6, 8, 12, 64) | Activation: {act_str}            │  │",
        "       ├── Initial Conv3D + MultiScale Factorized Convolutions                          │  │",
        "       ├───> [Skip 3 Branch] ──> Conv3D(64f, (6,1,1), valid) ──────> (B, 1, 8, 12, 64) ─┐│",
        "       │                                                                          │     │  │",
        "       v [MaxPool3D (1, 2, 2)]                                                    │     │  │",
        f"  [Encoder Stage 4] Filters: 128 | Shape: (B, 6, 4, 6, 128) | Activation: {act_str}      │     │  │",
        "       ├── Initial Conv3D + MultiScale Factorized Convolutions + Dropout(0.2)     │     │  │",
        "       ├───> [Skip 4 Branch] ──> Conv3D(128f, (6,1,1), valid) ─────> (B, 1, 4, 6, 128) ┐ │",
        "       │                                                                    │     │     │  │",
        "       v [MaxPool3D (1, 2, 2)]                                              │     │     │  │",
        "  [Temporal Compression: CompressLags]                                      │     │     │  │",
        "       ├── Conv3D(256f, kernel_size=(6, 1, 1), padding='valid')             │     │     │  │",
        "       └── Crushes temporal dimension: (B, 6, 4, 6, 256) -> (B, 1, 4, 6, 256)│     │     │  │",
        "       │                                                                    │     │     │  │",
        "       v",
        f"  [Bottleneck: ASPP Multi-Scale Pyramid] Filters: 256 | Activation: {act_str}",
        "       ├── Branch 1: Conv3D 1x1x1 (256f)",
        "       ├── Branch 2: Atrous Conv3D (256f, dilation=6)",
        "       ├── Branch 3: Atrous Conv3D (256f, dilation=12)",
        "       ├── Branch 4: Atrous Conv3D (256f, dilation=18)",
        "       ├── Branch 5: GlobalAveragePooling3D -> UpSampling3D",
        "       ├── Concatenate(Branches 1-5) -> 1x1x1 Conv3D(256f) -> Dropout(0.2)",
        "       └── Bottleneck Output Shape: (B, 1, 4, 6, 256)",
        "       │",
        "       v",
        f"  [Decoder Stage 6] Filters: 128 | Activation: {act_str}",
        "       ├── UpSampling3D (1, 2, 2) -> Shape: (B, 1, 8, 12, 128)",
        "       ├── Merge6: Concatenate([Skip 4 (B, 1, 8, 12, 128), Up6 (B, 1, 8, 12, 128)]) <───────┘",
    ])

    if is_kan:
        graph_lines.append("       ├── [KAN Non-Linear Skip Fusion Block] Dense(128f) + B-Spline Splines [*]")

    graph_lines.extend([
        "       └── Conv3D(128f, 3x3x3) + MultiScale Factorized Convolutions -> Shape: (B, 1, 8, 12, 128)",
        "       │",
        "       v",
        f"  [Decoder Stage 7] Filters: 64 | Activation: {act_str}",
        "       ├── UpSampling3D (1, 2, 2) -> Shape: (B, 1, 16, 24, 64)",
        "       ├── Merge7: Concatenate([Skip 3 (B, 1, 16, 24, 64), Up7 (B, 1, 16, 24, 64)]) <────────┘",
    ])

    if is_kan:
        graph_lines.append("       ├── [KAN Non-Linear Skip Fusion Block] Dense(64f) + B-Spline Splines [*]")

    graph_lines.extend([
        "       └── Conv3D(64f, 3x3x3) + MultiScale Factorized Convolutions -> Shape: (B, 1, 16, 24, 64)",
        "       │",
        "       v",
        f"  [Decoder Stage 8] Filters: 32 | Activation: {act_str}",
        "       ├── UpSampling3D (1, 2, 2) -> Shape: (B, 1, 32, 48, 32)",
        "       ├── Merge8: Concatenate([Skip 2 (B, 1, 32, 48, 32), Up8 (B, 1, 32, 48, 32)]) <─────────┘",
        "       └── Conv3D(32f, 3x3x3) + MultiScale Factorized Convolutions -> Shape: (B, 1, 32, 48, 32)",
        "       │",
        "       v",
        f"  [Decoder Stage 9] Filters: 16 | Activation: {act_str}",
        "       ├── UpSampling3D (1, 2, 2) -> Shape: (B, 1, 32, 48, 16)",
        "       ├── Merge9: Concatenate([Skip 1 (B, 1, 32, 48, 16), Up9 (B, 1, 32, 48, 16)]) <──────────┘",
        "       ├── Conv3D(16f, 3x3x3) + MultiScale Factorized Convolutions",
        "       └── Conv3D(14f, 3x3x3) -> Conv3D(7f, 1x1x1, activation='linear')",
        "       │   Delta_T Output: (B, 1, 32, 48, 7)",
        "       │",
        "       v",
    ])

    if has_anchor:
        graph_lines.extend([
            "  [Final Prediction Head: Anchor Residual Addition]",
            "       ├── Add()([Anchor_t0 (B, 1, 32, 48, 7), Delta_T (B, 1, 32, 48, 7)]) <─────────────────┘",
            "       └── Final Meteorological Output: (Batch, 1, 32, 48, 7 Horizons in Kelvin)",
            "           Lead Times: [+2h, +4h, +8h, +16h, +24h, +48h, +72h]"
        ])
    else:
        graph_lines.extend([
            "  [Final Prediction Head: Direct Output (No Residual Anchor)]",
            "       └── Absolute Temperature Output: (Batch, 1, 32, 48, 7 Horizons in Kelvin) [!] High Bias"
        ])

    for line in graph_lines:
        print(line)
    print("=" * 88)


def inspect_live_model(model_key: str):
    """
    Builds the model in Python and extracts exact layer-by-layer parameter and shape statistics.
    """
    info = MODELS_CATALOG.get(model_key, MODELS_CATALOG.get(f"L{model_key}", None))
    if not info:
        print(f"Unknown model key: {model_key}")
        return

    print(f"\nInstantiating live model: {info['name']}...")
    model = info["builder"]()
    total_params = model.count_params()

    print(f"\nModel Layer Summary:")
    print(f"- Total Graph Nodes (Layers): {len(model.layers)}")
    print(f"- Total Trainable Parameters : {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"- Input Tensor Layout        : {model.input_shape}")
    print(f"- Output Tensor Layout       : {model.output_shape}")

    conv3d_count = sum(1 for l in model.layers if "conv3d" in l.__class__.__name__.lower())
    kan_count = sum(1 for l in model.layers if isinstance(l, (SkipFusionBlock, KANBlock)) or "kan" in l.name.lower())
    mlp_count = sum(1 for l in model.layers if isinstance(l, MLPBlock) or "mlp" in l.name.lower())
    norm_count = sum(1 for l in model.layers if "norm" in l.__class__.__name__.lower())

    print(f"- 3D Convolutions in Graph   : {conv3d_count}")
    print(f"- KAN Non-Linear Blocks      : {kan_count}")
    print(f"- MLP Channel-Mixing Blocks  : {mlp_count}")
    print(f"- Normalization Layers       : {norm_count}")


def plot_model_computational_graph(model_key: str, output_png: str = None):
    """
    Generates a presentation-grade graphical diagram of the specific computational graph.
    """
    info = MODELS_CATALOG.get(model_key, MODELS_CATALOG.get(f"L{model_key}", None))
    if not info:
        return

    if output_png is None:
        out_dir = Path("results/model_graphs")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_png = out_dir / f"{model_key}_computational_graph.png"
    else:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=300)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")
    ax.set_xlim(0, 192)
    ax.set_ylim(0, 108)

    # Header Card
    hdr = patches.FancyBboxPatch((4, 94), 184, 10, boxstyle="round,pad=0.5",
                                 facecolor="#ffffff", edgecolor="#cbd5e0", linewidth=1.2, zorder=2)
    ax.add_patch(hdr)

    # Badge
    b_box = patches.FancyBboxPatch((6, 96), 22, 6, boxstyle="round,pad=0.3",
                                   facecolor="#1a365d", edgecolor="none", zorder=3)
    ax.add_patch(b_box)
    ax.text(17, 99, f"MODEL: {model_key.upper()}", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#ffffff", zorder=4)

    ax.text(31, 100, f"Computational Graph: {info['name']}",
            ha="left", va="center", fontsize=15.0, fontweight="bold", color="#1a365d", zorder=4)
    ax.text(31, 96.5, f"{info['category']} | {info['desc']} | Params: {info['params']:,} | Val RMSE: {info['val_rmse']}",
            ha="left", va="center", fontsize=9.0, fontstyle="italic", color="#4a5568", zorder=4)

    def draw_card(x, y, w, h, title, subtitle="", tag="", bg="#ffffff", border="#cbd5e0", lw=1.2, highlight=False):
        ec = "#e53e3e" if highlight else border
        width = 2.0 if highlight else lw
        sh = patches.FancyBboxPatch((x+0.3, y-0.3), w, h, boxstyle="round,pad=0.5",
                                    facecolor="#000000", alpha=0.03, edgecolor="none", zorder=2)
        ax.add_patch(sh)
        c = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                   facecolor=bg, edgecolor=ec, linewidth=width, zorder=3)
        ax.add_patch(c)
        if tag:
            pw = len(tag) * 0.7 + 2.0
            px = x + w - pw - 0.8
            py = y + h - 2.6
            p = patches.FancyBboxPatch((px, py), pw, 2.0, boxstyle="round,pad=0.2",
                                       facecolor="#edf2f7", edgecolor=border, linewidth=0.8, zorder=4)
            ax.add_patch(p)
            ax.text(px + pw/2, py + 1.0, tag, ha="center", va="center", fontsize=6.8, fontweight="bold", color="#2d3748", zorder=5)
        ax.text(x + 1.2, y + h - 2.0, title, ha="left", va="center", fontsize=8.8, fontweight="bold",
                color="#1a202c", zorder=4)
        if subtitle:
            ax.text(x + 1.2, y + (h - 2.0)/2, subtitle, ha="left", va="center", fontsize=7.2,
                    color="#4a5568", linespacing=1.25, zorder=4)

    has_anchor = info["has_anchor"]
    is_kan = info["skip_type"] == "kan_fusion"
    act_str = info["activation"]

    # 1. ANCHOR RESIDUAL HIGHWAY
    if has_anchor:
        ax.text(10, 87.5, "ANCHOR RESIDUAL BYPASS: Yhat = T_t0 + Delta T (Bounds predictions to physical anomaly scale)",
                fontsize=8.5, fontweight="bold", color="#dd6b20")
        ax.add_line(lines.Line2D([10, 168], [85, 85], color="#dd6b20", lw=2.2, linestyle="--", zorder=3))
        ax.annotate("", xy=(168, 77.5), xytext=(168, 85),
                    arrowprops=dict(arrowstyle="->", lw=2.2, color="#dd6b20"), zorder=4)
    else:
        ax.text(10, 87.5, "[!] NO RESIDUAL ANCHOR: Model attempts direct absolute regression -> Causes severe cold collapse (7.34 K RMSE)",
                fontsize=8.5, fontweight="bold", color="#e53e3e")

    # 2. INPUT
    draw_card(8, 70, 24, 11, "Input Meteorological Tensor", "10 Met Fields: t2m, u10, v10, sp\nAntecedent lags T = 6",
              tag="(B, 6, 32, 48, 10)", bg="#ebf8ff", border="#3182ce")
    if has_anchor:
        ax.annotate("Extract t0 Frame", xy=(20, 85), xytext=(20, 81),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#dd6b20"),
                    ha="center", fontsize=7.0, fontweight="bold", color="#dd6b20")

    ax.annotate("", xy=(20, 65), xytext=(20, 70), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2d3748"))

    # 3. ENCODER
    draw_card(8, 54, 24, 11, f"Encoder Stage 1 ({act_str})", "Conv3D (16f) + MultiScale Block\n5 Factorized parallel branches",
              tag="(B, 6, 32, 48, 16)", bg="#e6fffa", border="#319795")
    ax.annotate("MaxPool (1,2,2)", xy=(20, 47), xytext=(20, 54),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#718096"), ha="center", fontsize=6.8, color="#718096")

    draw_card(8, 36, 24, 11, f"Encoder Stage 2 ({act_str})", "Conv3D (32f) + MultiScale Block\nPreserves temporal depth T=6",
              tag="(B, 6, 16, 24, 32)", bg="#e6fffa", border="#319795")
    ax.annotate("MaxPool (1,2,2)", xy=(20, 29), xytext=(20, 36),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#718096"), ha="center", fontsize=6.8, color="#718096")

    draw_card(8, 18, 24, 11, f"Encoder Stage 3 ({act_str})", "Conv3D (64f) + MultiScale Block\nSpatial resolution: 8x12",
              tag="(B, 6, 8, 12, 64)", bg="#e6fffa", border="#319795")
    ax.annotate("MaxPool (1,2,2)", xy=(20, 11), xytext=(20, 18),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#718096"), ha="center", fontsize=6.8, color="#718096")

    # Bottom transition
    draw_card(38, 6, 26, 11, f"Encoder Stage 4 + Drop ({act_str})", "Conv3D (128f) + MultiScale Block\nSpatial Dropout (p=0.2)",
              tag="(B, 6, 4, 6, 128)", bg="#e6fffa", border="#319795")
    ax.annotate("", xy=(38, 11.5), xytext=(20, 11), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2d3748"))

    # Temporal compression
    draw_card(70, 6, 24, 11, "CompressLags Node", "Conv3D(256f, (6,1,1), valid)\nCrushes antecedent lag T: 6 -> 1",
              tag="(B, 1, 4, 6, 256)", bg="#fefcbf", border="#d69e2e")
    ax.annotate("Pool4 (1,2,2)", xy=(70, 11.5), xytext=(64, 11.5),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2d3748"), ha="center", va="bottom", fontsize=6.8, color="#718096")

    # Bottleneck
    draw_card(100, 6, 26, 11, f"ASPP Pyramid Bottleneck ({act_str})", "Dilations: 1, 6, 12, 18 + ImagePool\nSynoptic multi-scale receptive field",
              tag="(B, 1, 4, 6, 256)", bg="#fefcbf", border="#d69e2e")
    ax.annotate("", xy=(100, 11.5), xytext=(94, 11.5), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2d3748"))

    # Decoder stages
    ax.annotate("UpSample (1,2,2)", xy=(136, 17), xytext=(126, 11.5),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2d3748"), ha="center", fontsize=6.8, color="#718096")

    # Stage 6
    s6_sub = "Up6 + Skip4 -> Concat\n[KAN Spline Fusion: B-splines cut 388k params]" if is_kan else "Up6 + Skip4 -> Naive Concat\nConv3D (128f) + MultiScale Block"
    draw_card(136, 17, 36, 12, f"Decoder Stage 6 {'+ KAN [*]' if is_kan else ''}", s6_sub,
              tag="(B, 1, 4, 6, 128)", bg="#faf5ff" if is_kan else "#f0fff4",
              border="#805ad5" if is_kan else "#38a169", highlight=is_kan)

    ax.annotate("UpSample (1,2,2)", xy=(154, 33), xytext=(154, 29),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2d3748"), ha="left", fontsize=6.8, color="#718096")

    # Stage 7
    s7_sub = "Up7 + Skip3 -> Concat\n[KAN Spline Fusion: B-splines cut 388k params]" if is_kan else "Up7 + Skip3 -> Naive Concat\nConv3D (64f) + MultiScale Block"
    draw_card(136, 33, 36, 12, f"Decoder Stage 7 {'+ KAN [*]' if is_kan else ''}", s7_sub,
              tag="(B, 1, 8, 12, 64)", bg="#faf5ff" if is_kan else "#f0fff4",
              border="#805ad5" if is_kan else "#38a169", highlight=is_kan)

    ax.annotate("UpSample (1,2,2)", xy=(154, 49), xytext=(154, 45),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2d3748"), ha="left", fontsize=6.8, color="#718096")

    # Stage 8
    draw_card(136, 49, 36, 11, f"Decoder Stage 8 ({act_str})", "Up8 + Skip2 -> Concat (64f)\nConv3D (32f) + MultiScale Block",
              tag="(B, 1, 16, 24, 32)", bg="#f0fff4", border="#38a169")

    ax.annotate("UpSample (1,2,2)", xy=(154, 64), xytext=(154, 60),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2d3748"), ha="left", fontsize=6.8, color="#718096")

    # Stage 9
    draw_card(136, 64, 36, 11, f"Decoder Stage 9 ({act_str})", "Up9 + Skip1 -> Concat (32f)\nConv3D (16f) -> 14f -> 7f (linear)",
              tag="(B, 1, 32, 48, 7)", bg="#f0fff4", border="#38a169")

    ax.annotate("", xy=(154, 76), xytext=(154, 75), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2d3748"))

    # Prediction head
    phead_sub = "Yhat = T_t0 (Anchor) + Delta_T (Anomalies)\nLead Times: [+2h, +4h, +8h, +16h, +24h, +48h, +72h]" if has_anchor else "Yhat = Conv3D output (Absolute)\nSuffers severe systematic cold bias"
    draw_card(136, 76, 36, 10, "Prediction Head (Kelvin)", phead_sub,
              tag="(B, 1, 32, 48, 7)", bg="#fff5f5", border="#e53e3e", lw=1.8)

    # Skips
    s_arrow = dict(arrowstyle="->", lw=1.6, linestyle="--", color="#e53e3e")
    ax.annotate("", xy=(136, 23), xytext=(64, 14), arrowprops=s_arrow)
    ax.annotate("", xy=(136, 39), xytext=(32, 23.5), arrowprops=s_arrow)
    ax.annotate("", xy=(136, 54.5), xytext=(32, 41.5), arrowprops=s_arrow)
    ax.annotate("", xy=(136, 69.5), xytext=(32, 59.5), arrowprops=s_arrow)

    # Footer note
    ax.text(6, 2.5, f"Broad-UNet Architectural Explainer | Generated for model: {model_key}",
            ha="left", va="center", fontsize=8, color="#718096")
    ax.text(186, 2.5, "Agam Ghotra | Machine Learning Research",
            ha="right", va="center", fontsize=8, fontweight="bold", color="#2d3748")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved computational graph diagram to: {output_png}")


def generate_all_presentation_slides():
    """
    Invokes the dedicated presentation generator script to produce all 4 widescreen slides.
    """
    from scripts.generate_presentation_graphs import (
        generate_slide_1_macro_architecture,
        generate_slide_2_internal_mechanisms,
        generate_slide_3_architectural_evolution,
        generate_slide_4_search_taxonomy
    )
    out_dir = Path("results/presentation_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("RENDERING FULL 4-SLIDE PRESENTATION SUITE (16:9 WIDESCREEN, 300 DPI)")
    print("=" * 80)
    generate_slide_1_macro_architecture(out_dir / "1_broad_unet_macro_architecture_16x9.png")
    generate_slide_2_internal_mechanisms(out_dir / "2_internal_block_mechanisms_deepdive.png")
    generate_slide_3_architectural_evolution(out_dir / "3_architectural_evolution_comparison.png")
    generate_slide_4_search_taxonomy(out_dir / "4_layer_rearrangement_search_taxonomy.png")
    print("\n[COMPLETE] All 4 presentation slides are ready in results/presentation_graphs/\n")


def print_oral_presentation_script():
    """
    Prints an executive oral presentation script for presenting to TA / professors.
    """
    script = """
================================================================================
EXECUTIVE ORAL PRESENTATION SCRIPT (TA / SEMINAR WALKTHROUGH)
================================================================================

[SLIDE 1: MACRO COMPUTATIONAL GRAPH & CHAMPION ARCHITECTURE]
"Good morning / afternoon. Today I am presenting our architectural search and 
optimization of the Broad-UNet spatiotemporal model for ERA5 meteorological nowcasting.

Slide 1 shows the end-to-end computational graph of our Champion model: L15.
Broad-UNet takes a 6-step temporal sequence of 10 meteorological channels across a 
32x48 grid covering the Iberian domain.

The macro-topology is an asymmetrical 3D U-Net:
1. The Encoder features 4 downsampling stages. Notice that our pooling operations 
   use a (1, 2, 2) kernel—this halves the spatial grid from 32x48 down to 4x6 while 
   preserving the temporal depth T=6 throughout the encoder.
2. At the bottleneck, our CompressLags node crushes the temporal dimension from 6 to 1 
   using a valid 3D convolution, feeding into an Atrous Spatial Pyramid Pooling (ASPP) 
   block with dilation rates 1, 6, 12, and 18 plus global pooling.
3. Crucially, look at the top amber line: that is our Anchor Residual Highway. 
   Instead of forcing the network to predict absolute surface temperatures from scratch, 
   we broadcast the latest observed frame T_t0 directly to the loss head. The network 
   only learns the delta perturbations Delta_T. This single structural change resolved 
   a catastrophic -52.8% cold collapse observed in the original baseline.
4. Finally, on the right in purple: at Decoder Stages 6 and 7, we replace rigid 
   concatenation with non-linear Kolmogorov-Arnold Network (KAN) B-spline skip fusions, 
   cutting 388k parameters while dynamically aligning spatial features."

[SLIDE 2: MICRO-ARCHITECTURAL INTERNAL ENGINES]
"Moving to Slide 2, let us look under the hood at the 4 algorithmic components:
- Panel A shows the Factorized MultiScale 3D Convolution Block: A standard 3D convolution 
  has 27 weights per filter. We factorize this into (1x1x3) longitudinal, (1x3x1) latitudinal, 
  and (3x1x1) temporal passes. This cuts parameter count by 66.7% while explicitly separating 
  spatial wind advection from diurnal temporal decay.
- Panel B shows the ASPP Bottleneck: By evaluating parallel dilation rates from 1 to 18, 
  the model simultaneously captures microscale convective storms and continental Rossby waves.
- Panel C contrasts Naive Concatenation with our KAN B-Spline Fusion: In standard U-Net, 
  deep decoder semantics clash with shallow encoder textures. Our KAN blocks use cubic 
  B-spline bases with learnable coefficients per channel edge, dynamically adapting activation 
  functions and eliminating cross-scale aliasing.
- Panel D formalizes the Anchor Residual Head: Yhat = T_t0 + Delta T. This formulation guarantees 
  that even under zero-shot conditions, our model never drifts into non-physical climate states."

[SLIDE 3: ARCHITECTURAL EVOLUTION MATRIX]
"Slide 3 illustrates how we methodically progressed from the raw upstream model to our Champion:
- Upstream Broad-UNet suffered a 7.34 K validation RMSE and failed the persistence baseline.
- L0 Baseline Broad-UNet added the Anchor Residual Head, immediately cutting RMSE to 3.520 K (-52.0%).
- L11 introduced KAN Skip Fusion at Merge 6 and Merge 7, reducing parameters from 10.64M to 10.25M 
  and dropping RMSE to 3.427 K (+2.63% gain).
- L15 Champion Hybrid synthesized KAN Skip Fusion with a global LeakyReLU (alpha=0.2) backbone. 
  By maintaining continuous gradient flow for cold frontal advection, L15 reached 3.369 K 
  validation RMSE (+4.29% over L0) and achieved a +0.137 test skill score over persistence (p < 0.001)."

[SLIDE 4: ARCHITECTURAL SEARCH LANDSCAPE & ACTIVATIONS]
"Finally, Slide 4 shows our systematic exploration of 15 layer rearrangements and 5 activation functions:
- Bottleneck modifications (L2 to L9) hit receptive field saturation—ASPP already captured 95%+ of 
  spatial context, so adding MLPs or KANs before/after ASPP yielded diminishing returns (< +0.80%).
- Skip Connection modifications (L10 vs L11) proved to be the decisive breakthrough: non-linear KAN 
  fusion at the skip junctions was the true structural bottleneck of the 3D U-Net.
- Inverting decoder blocks (L13) caused severe spatial checkerboard aliasing (3.684 K RMSE).
- On the global activation frontier, LeakyReLU proved champion for all-horizon robustness, 
  while Tanh showed exceptional performance for short-range diurnal cycling."
================================================================================
"""
    print(script)


def main():
    parser = argparse.ArgumentParser(description="Print and explain Broad-UNet computational graphs.")
    parser.add_argument("--model", type=str, default="L15",
                        help="Model key: basic, L0, L1..L15, act_tanh, act_leaky_relu, or all")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Generate a presentation-grade PNG diagram of the computational graph.")
    parser.add_argument("--presentation", action="store_true", default=False,
                        help="Generate the full 4-slide presentation suite in results/presentation_graphs/.")
    parser.add_argument("--script", action="store_true", default=False,
                        help="Print the oral presentation script for the TA walkthrough.")
    args = parser.parse_args()

    if args.presentation:
        generate_all_presentation_slides()
        print_oral_presentation_script()
        return

    if args.script:
        print_oral_presentation_script()
        return

    if args.model.lower() == "all":
        print("\n" + "#" * 88)
        print("BROAD-UNET ARCHITECTURAL SUITE: MASTER CATALOG OF 18 MODELS")
        print("#" * 88)
        for k, v in MODELS_CATALOG.items():
            print(f"[{k:>14}] : {v['name']} ({v['category']})")
        print("\nPrinting full graph for basic model, top rearrangement (L11), and champion (L15):\n")
        for k in ["basic", "L0", "L11", "L15"]:
            explain_graph_structure(k)
            print_ascii_graph(k)
            inspect_live_model(k)
            if args.plot:
                plot_model_computational_graph(k)
    else:
        explain_graph_structure(args.model)
        print_ascii_graph(args.model)
        inspect_live_model(args.model)
        if args.plot:
            plot_model_computational_graph(args.model)


if __name__ == "__main__":
    main()

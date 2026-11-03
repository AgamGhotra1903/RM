"""
scripts/print_model_graph.py
============================
A pure-Python graph visualizer and architectural explainer for:
1. The basic (original) Broad-UNet model.
2. All derivative models (L1 through L14 layer rearrangements).
3. The Champion Hybrid model (L15: KAN Skip Fusion + LeakyReLU backbone).
4. The global activation variants (ReLU, GELU, SiLU, Tanh, LeakyReLU).

Usage:
  python scripts/print_model_graph.py --model basic
  python scripts/print_model_graph.py --model L0
  python scripts/print_model_graph.py --model L11
  python scripts/print_model_graph.py --model L15
  python scripts/print_model_graph.py --model all
  python scripts/print_model_graph.py --model L15 --plot
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.broad_unet import build_broad_unet
from models.rearranged_models import build_rearranged_model, EXPERIMENT_NAMES
from models.activation_models import build_broad_unet_with_activation, ACTIVATION_DESCRIPTIONS
from models.hybrid_model import build_hybrid_kan_leaky_relu
from models.modules import SkipFusionBlock, KANBlock, MLPBlock

MODELS_CATALOG = {
    "basic": {
        "name": "Basic Original Broad-UNet (jesusgf96 upstream)",
        "builder": lambda: build_broad_unet(lags=6, latitude=32, longitude=48, features=6, features_output=7, filters=16),
        "desc": "The raw reference architecture without residual anchor head (direct absolute prediction).",
        "category": "Baseline"
    },
    "L0": {
        "name": "L0_original: Hardened Baseline Broad-UNet",
        "builder": lambda: build_rearranged_model("L0_original", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Official Broad-UNet backbone equipped with the Anchor Residual Prediction Head (Yhat = T_t0 + Delta T).",
        "category": "Reference Baseline"
    },
    "L1": {
        "name": "L1_conv_rearrangement",
        "builder": lambda: build_rearranged_model("L1_conv_rearrangement", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Reordered encoder blocks: Multi-scale branches evaluated BEFORE the initial Conv3D.",
        "category": "Encoder Rearrangement"
    },
    "L2": {
        "name": "L2_conv_before_aspp",
        "builder": lambda: build_rearranged_model("L2_conv_before_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Additional 3D Convolution layer injected BEFORE the ASPP bottleneck.",
        "category": "Bottleneck Rearrangement"
    },
    "L3": {
        "name": "L3_conv_after_aspp",
        "builder": lambda: build_rearranged_model("L3_conv_after_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Additional 3D Convolution layer injected AFTER the ASPP bottleneck.",
        "category": "Bottleneck Rearrangement"
    },
    "L4": {
        "name": "L4_mlp_before_aspp",
        "builder": lambda: build_rearranged_model("L4_mlp_before_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Multi-Layer Perceptron (MLP) channel-mixing block inserted BEFORE ASPP.",
        "category": "Bottleneck Rearrangement"
    },
    "L5": {
        "name": "L5_mlp_after_aspp",
        "builder": lambda: build_rearranged_model("L5_mlp_after_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Multi-Layer Perceptron (MLP) channel-mixing block inserted AFTER ASPP.",
        "category": "Bottleneck Rearrangement"
    },
    "L6": {
        "name": "L6_kan_before_aspp",
        "builder": lambda: build_rearranged_model("L6_kan_before_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Kolmogorov-Arnold Network (KAN) block with B-spline bases inserted BEFORE ASPP.",
        "category": "Bottleneck Rearrangement"
    },
    "L7": {
        "name": "L7_kan_after_aspp",
        "builder": lambda: build_rearranged_model("L7_kan_after_aspp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Kolmogorov-Arnold Network (KAN) block with B-spline bases inserted AFTER ASPP.",
        "category": "Bottleneck Rearrangement"
    },
    "L8": {
        "name": "L8_mlp_then_kan",
        "builder": lambda: build_rearranged_model("L8_mlp_then_kan", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Sequential composition: ASPP -> MLP -> KAN.",
        "category": "Bottleneck Compound"
    },
    "L9": {
        "name": "L9_kan_then_mlp",
        "builder": lambda: build_rearranged_model("L9_kan_then_mlp", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Sequential composition: ASPP -> KAN -> MLP.",
        "category": "Bottleneck Compound"
    },
    "L10": {
        "name": "L10_before_skip_fusion",
        "builder": lambda: build_rearranged_model("L10_before_skip_fusion", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Pre-skip linear projection applied to encoder skip features before merge.",
        "category": "Skip Connection Rearrangement"
    },
    "L11": {
        "name": "L11_after_skip_fusion",
        "builder": lambda: build_rearranged_model("L11_after_skip_fusion", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Post-skip non-linear KAN Fusion with B-spline bases at merge6 & merge7.",
        "category": "Skip Connection Rearrangement (Rank 2 Overall)"
    },
    "L12": {
        "name": "L12_encoder_rearrangement",
        "builder": lambda: build_rearranged_model("L12_encoder_rearrangement", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Reordered normalization/activation: Conv3D -> BatchNorm -> ReLU.",
        "category": "Norm/Act Rearrangement"
    },
    "L13": {
        "name": "L13_decoder_rearrangement",
        "builder": lambda: build_rearranged_model("L13_decoder_rearrangement", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Reordered decoder blocks: Convolution evaluated BEFORE UpSampling3D.",
        "category": "Decoder Rearrangement"
    },
    "L14": {
        "name": "L14_best_combined_arrangement",
        "builder": lambda: build_rearranged_model("L14_best_combined_arrangement", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Compound arrangement: Pre-ASPP Conv + KAN Skip Fusion + Norm-Act reordering.",
        "category": "Compound Architecture"
    },
    "L15": {
        "name": "L15_hybrid_kan_leaky_relu",
        "builder": lambda: build_hybrid_kan_leaky_relu(lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "CHAMPION MODEL: KAN Skip Fusion (merge6 & merge7) + LeakyReLU (alpha=0.2) backbone.",
        "category": "Overall Champion (Rank 1 / 16)"
    },
    "act_tanh": {
        "name": "Broad-UNet with Tanh Backbone",
        "builder": lambda: build_broad_unet_with_activation("tanh", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Global Tanh activation (-1, 1). Best short-range diurnal model (+7.56% val gain).",
        "category": "Activation Variant"
    },
    "act_leaky_relu": {
        "name": "Broad-UNet with LeakyReLU Backbone",
        "builder": lambda: build_broad_unet_with_activation("leaky_relu", lags=6, latitude=32, longitude=48, features=10, features_output=7, filters=16),
        "desc": "Global LeakyReLU (alpha=0.2). Best robust activation across all 72 hours (+4.17% gain).",
        "category": "Activation Variant"
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
    print("     - In L0-L15: An AnchorResidualBroadcast node extracts the latest frame T_t0 at index -1,")
    print("       broadcasting it as (B, 1, 32, 48, 7) to create a residual skip directly to the loss head.")

    print("\n   Stage 1 to 4: Hierarchical Spatio-Temporal Encoder")
    print("     - Each stage contains an initial Conv3D followed by a MultiScale Factorized Block.")
    print("     - The MultiScale Block splits feature flow into 5 parallel factorized branches:")
    print("       * Branch 1: 1x1x1 Conv3D (identity/channel mixing)")
    print("       * Branch 2: Factorized 1x1x3 -> 1x3x1 -> 3x1x1 (spatial height/width + temporal depth)")
    print("       * Branch 3: Factorized 1x1x5 -> 1x5x1 -> 5x1x1 (wider receptive field)")
    print("       * Branch 4 & 5: Auxiliary factorized 3D convolutions")
    print("       * Concatenation + 1x1x1 Conv3D projection + Residual Shortcut Addition.")
    print("     - Spatial downsampling: MaxPool3D(pool_size=(1, 2, 2)) preserves temporal dimension (depth=6)")
    print("       while halving spatial resolution: (32, 48) -> (16, 24) -> (8, 12) -> (4, 6).")

    print("\n   Stage 5: Temporal Compression & Bottleneck (ASPP)")
    print("     - CompressLags node: Conv3D(256, kernel_size=(6, 1, 1), padding='valid') crushes temporal lags")
    print("       from 6 down to 1: (B, 6, 4, 6, 128) -> (B, 1, 4, 6, 256).")
    print("     - ASPP (Atrous Spatial Pyramid Pooling): Evaluates 5 parallel branches with dilation rates:")
    print("       * Rate 1 (1x1x1), Rate 6 (1x3x3), Rate 12 (1x3x3), Rate 18 (1x3x3), and Global Pooling.")
    print("       * Captures multi-scale synoptic weather systems simultaneously without losing spatial resolution.")

    print("\n   Stage 6 to 9: Symmetrical Upsampling Decoder & Skip Merges")
    print("     - Each stage uses UpSampling3D(size=(1, 2, 2)) followed by Conv3D.")
    print("     - 4 Long Skip Connections bridge encoder stages to decoder stages:")
    print("       * Skip 4 (from Stage 4) fuses with Up6 at merge6  -> spatial (8, 12)")
    print("       * Skip 3 (from Stage 3) fuses with Up7 at merge7  -> spatial (16, 24)")
    print("       * Skip 2 (from Stage 2) fuses with Up8 at merge8  -> spatial (32, 48)")
    print("       * Skip 1 (from Stage 1) fuses with Up9 at merge9  -> spatial (32, 48)")
    print("     - Each skip tensor passes through a temporal compression Conv3D(valid) before merge.")

    print("\n   Stage 10: Multi-Horizon Forecast Projection")
    print("     - Final Conv3D(7, kernel_size=1) projects features to the 7 lead times.")
    print("     - Residual addition: Yhat = T_t0 + Delta_T.")

    # Model-Specific Architectural Explanation
    print("\n3. SPECIFIC GRAPH MODIFICATIONS IN THIS MODEL:")
    if model_key == "basic":
        print("   - Basic Upstream Model: Direct absolute temperature prediction (no residual anchor).")
        print("   - All activations: standard piecewise linear ReLU.")
        print("   - All skip connections: naive concatenation without non-linear projection.")
    elif model_key == "L0":
        print("   - Baseline Reference: Standard Broad-UNet backbone + Anchor Residual Head.")
        print("   - Resolves the systematic cold bias (-52.8% error drop vs basic upstream).")
    elif model_key == "L1":
        print("   - Reordered Encoder: Swaps the evaluation order so MultiScale branches run BEFORE Conv3D.")
        print("   - Graph Effect: Removes upfront spatio-temporal regularization, causing error accumulation on +48h/+72h.")
    elif model_key in ["L2", "L3"]:
        loc = "BEFORE" if model_key == "L2" else "AFTER"
        print(f"   - Extra Conv3D injected {loc} ASPP.")
        print("   - Graph Effect: Adds ~590k params to the bottleneck. ASPP dilations are already saturated, yielding marginal gains.")
    elif model_key in ["L4", "L5"]:
        loc = "BEFORE" if model_key == "L4" else "AFTER"
        print(f"   - Dense MLP channel-mixing block inserted {loc} ASPP.")
        print("   - Graph Effect: LayerNorm -> Dense(256, 512) -> GELU -> Dense(512, 256) + Residual.")
    elif model_key in ["L6", "L7"]:
        loc = "BEFORE" if model_key == "L6" else "AFTER"
        print(f"   - KAN Block with learnable B-spline activation curves inserted {loc} ASPP.")
        print("   - Graph Effect: Replaces fixed activation functions with spline-parameterized edges.")
    elif model_key in ["L8", "L9"]:
        order = "ASPP -> MLP -> KAN" if model_key == "L8" else "ASPP -> KAN -> MLP"
        print(f"   - Sequential Bottleneck: {order}.")
        print("   - Graph Effect: Evaluates both channel-mixing and spline adaptation in tandem.")
    elif model_key == "L10":
        print("   - Pre-Skip Processing: Linear Conv3D(1x1x1) applied to encoder skip branches before merge.")
        print("   - Graph Effect: Minimal change (+0.65% gain); linear projection alone is insufficient for multi-scale alignment.")
    elif model_key == "L11":
        print("   - Post-Skip KAN Fusion: KAN Blocks replace naive concatenation at merge6 and merge7.")
        print("   - Graph Effect: Learnable B-splines smoothly fuse high-level decoder features with low-level encoder features.")
        print("   - Parameter Impact: Cuts 388,352 parameters while reducing validation RMSE by +2.63% (Rank 2 overall).")
    elif model_key == "L12":
        print("   - Norm/Act Reordering: Evaluates Conv3D -> BatchNorm -> ReLU inside encoder blocks.")
        print("   - Graph Effect: Adds batch statistics normalization; achieves stable but neutral performance.")
    elif model_key == "L13":
        print("   - Decoder Reordering: Evaluates Conv3D BEFORE UpSampling3D.")
        print("   - Graph Effect: Convolutions operate on un-interpolated features, causing spatial checkerboard aliasing (Rank 15).")
    elif model_key == "L14":
        print("   - Compound Best Combination: Pre-ASPP Conv + KAN Skip Fusion + Norm-Act reordering.")
    elif model_key == "L15":
        print("   - CHAMPION HYBRID MODEL: Combines Post-Skip KAN Fusion (merge6 & merge7) with a LeakyReLU (alpha=0.2) backbone.")
        print("   - Graph Effect: Eliminates dying neurons across all 3D convolutions while KAN non-linearly aligns skip features.")
        print("   - Achievement: Lowest Validation RMSE (3.369 K, +4.29% over L0), cuts 388k params, and achieves double-digit skill over persistence.")
    elif "tanh" in model_key:
        print("   - Global Tanh Backbone: Replaces ReLU with bounded Hyperbolic Tangent (-1, 1).")
        print("   - Graph Effect: Perfectly bounds diurnal residual fluctuations (huge -1.18 K error drop on +16h).")
    elif "leaky_relu" in model_key:
        print("   - Global LeakyReLU Backbone: Replaces ReLU with non-zero slope (alpha=0.2) for negative pre-activations.")
        print("   - Graph Effect: Uninterrupted gradient flow for cold advection anomalies without saturation.")


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

    is_l11_or_l15 = model_key in ["L11", "L15", "L14"]
    is_l1 = model_key == "L1"
    is_l10 = model_key == "L10"
    is_l13 = model_key == "L13"
    act_str = "LeakyReLU(0.2)" if model_key in ["L15", "act_leaky_relu"] else ("Tanh" if "tanh" in model_key else "ReLU")

    graph_lines = [
        "  [Input: Meteorological ERA5 Windows]",
        "  Tensor Shape: (Batch, Lags=6, Lat=32, Lon=48, Channels=10)",
        "       │",
        "       ├───> [AnchorResidualBroadcast] ──────────────────────────────────────────────┐",
        "       │     Extracts latest t0 frame: (B, 1, 32, 48, 7)                             │",
        "       │                                                                             │",
        "       v                                                                             │",
        f"  [Encoder Stage 1] Filters: 16 | Activation: {act_str}",
        f"       │  ├── Initial Conv3D (16f, 3x3x3)" if not is_l1 else f"       │  ├── MultiScale Factorized Convolutions [L1 Reordered]",
        f"       │  └── MultiScale Factorized Convolutions (1x1x3, 1x3x1, 3x1x1)" if not is_l1 else f"       │  └── Conv3D (16f, 3x3x3)",
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
        f"       ├───> [Skip 3 Branch] ──> Conv3D(64f, (6,1,1), valid) {'[Pre-Skip Conv]' if is_l10 else ''} -> (B, 1, 8, 12, 64) ─┐│",
        "       │                                                                          │     │  │",
        "       v [MaxPool3D (1, 2, 2)]                                                    │     │  │",
        f"  [Encoder Stage 4] Filters: 128 | Shape: (B, 6, 4, 6, 128) | Activation: {act_str}      │     │  │",
        "       ├── Initial Conv3D + MultiScale Factorized Convolutions + Dropout(0.2)     │     │  │",
        f"       ├───> [Skip 4 Branch] ──> Conv3D(128f, (6,1,1), valid) {'[Pre-Skip Conv]' if is_l10 else ''} -> (B, 1, 4, 6, 128) ┐ │",
        "       │                                                                    │     │     │  │",
        "       v [MaxPool3D (1, 2, 2)]                                              │     │     │  │",
        "  [Temporal Compression: CompressLags]                                      │     │     │  │",
        "       ├── Conv3D(256f, kernel_size=(6, 1, 1), padding='valid')             │     │     │  │",
        "       └── Crushes temporal dimension: (B, 6, 4, 6, 256) -> (B, 1, 4, 6, 256)│     │     │  │",
        "       │                                                                    │     │     │  │",
        "       v",
    ]

    # Bottleneck Section
    if model_key in ["L2", "L14"]:
        graph_lines.append("  [Bottleneck Pre-ASPP] Conv3D(256f, 1x3x3) [L2 Pre-ASPP Modification]")
    elif model_key == "L4":
        graph_lines.append("  [Bottleneck Pre-ASPP] Dense MLP Block (256 -> 512 -> 256) [L4 MLP Modification]")
    elif model_key == "L6":
        graph_lines.append("  [Bottleneck Pre-ASPP] KAN Block (B-spline edge activations) [L6 KAN Modification]")

    graph_lines.extend([
        f"  [Bottleneck: ASPP Multi-Scale Pyramid] Filters: 256 | Activation: {act_str}",
        "       ├── Branch 1: Conv3D 1x1x1 (256f)",
        "       ├── Branch 2: Atrous Conv3D (256f, dilation=6)",
        "       ├── Branch 3: Atrous Conv3D (256f, dilation=12)",
        "       ├── Branch 4: Atrous Conv3D (256f, dilation=18)",
        "       ├── Branch 5: GlobalAveragePooling3D -> UpSampling3D",
        "       ├── Concatenate(Branches 1-5) -> 1x1x1 Conv3D(256f) -> Dropout(0.2)",
        "       └── Bottleneck Output Shape: (B, 1, 4, 6, 256)",
        "       │",
    ])

    if model_key == "L3":
        graph_lines.append("  [Bottleneck Post-ASPP] Conv3D(256f, 3x3x3) [L3 Post-ASPP Modification]\n       │")
    elif model_key == "L5":
        graph_lines.append("  [Bottleneck Post-ASPP] Dense MLP Block (256 -> 512 -> 256) [L5 Post-ASPP Modification]\n       │")
    elif model_key == "L7":
        graph_lines.append("  [Bottleneck Post-ASPP] KAN Block (B-spline edge activations) [L7 Post-ASPP Modification]\n       │")
    elif model_key == "L8":
        graph_lines.append("  [Bottleneck Post-ASPP] Sequential MLP -> KAN Blocks [L8 Compound Modification]\n       │")
    elif model_key == "L9":
        graph_lines.append("  [Bottleneck Post-ASPP] Sequential KAN -> MLP Blocks [L9 Compound Modification]\n       │")

    # Decoder Section
    graph_lines.extend([
        "       v",
        f"  [Decoder Stage 6] Filters: 128 | Activation: {act_str}",
        "       ├── UpSampling3D (1, 2, 2) -> Shape: (B, 1, 8, 12, 128)" if not is_l13 else "       ├── Conv3D(128f) BEFORE UpSampling3D (1, 2, 2) [L13 Reordered]",
        "       ├── Merge6: Concatenate([Skip 4 (B, 1, 8, 12, 128), Up6 (B, 1, 8, 12, 128)]) <───────┘",
    ])

    if is_l11_or_l15:
        graph_lines.append("       ├── [KAN Non-Linear Skip Fusion Block] Dense(128f) + B-Spline Splines 🌟")
    
    graph_lines.extend([
        "       └── Conv3D(128f, 3x3x3) + MultiScale Factorized Convolutions -> Shape: (B, 1, 8, 12, 128)",
        "       │",
        "       v",
        f"  [Decoder Stage 7] Filters: 64 | Activation: {act_str}",
        "       ├── UpSampling3D (1, 2, 2) -> Shape: (B, 1, 16, 24, 64)",
        "       ├── Merge7: Concatenate([Skip 3 (B, 1, 16, 24, 64), Up7 (B, 1, 16, 24, 64)]) <────────┘",
    ])

    if is_l11_or_l15:
        graph_lines.append("       ├── [KAN Non-Linear Skip Fusion Block] Dense(64f) + B-Spline Splines 🌟")

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
        "       │   Delta_T Residual Output: (B, 1, 32, 48, 7)",
        "       │",
        "       v",
        "  [Final Prediction Head: Anchor Residual Addition]",
        "       ├── Add()([Anchor_t0 (B, 1, 32, 48, 7), Delta_T (B, 1, 32, 48, 7)]) <─────────────────┘",
        "       └── Final Meteorological Output: (Batch, 1, 32, 48, 7 Horizons in Kelvin)",
        "           Lead Times: [+2h, +4h, +8h, +16h, +24h, +48h, +72h]"
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

    # Inspect key layer categories
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
    Generates a publication-grade graphical diagram of the computational graph using matplotlib.
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

    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Title Banner
    ax.text(50, 96.5, f"Computational Graph: {info['name']}",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#1a365d")
    ax.text(50, 93.5, f"{info['category']} | {info['desc']}",
            ha="center", va="center", fontsize=10, fontstyle="italic", color="#4a5568")

    def draw_box(x, y, w, h, title, subtitle, color, border="#2b6cb0", highlight=False):
        lw = 2.4 if highlight else 1.2
        ec = "#e53e3e" if highlight else border
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                      facecolor=color, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 1.2, title, ha="center", va="center", fontsize=8.5,
                fontweight="bold" if highlight else "normal", color="#1a202c", zorder=4)
        ax.text(x + w/2, y + h/2 - 1.2, subtitle, ha="center", va="center", fontsize=7.2,
                color="#2d3748", zorder=4)

    # Stage 0: Input & Anchor
    draw_box(4, 78, 14, 8, "Input Windows", "(B, 6, 32, 48, 10)\nERA5 Met Fields", "#bee3f8")
    draw_box(4, 90, 14, 6, "Anchor Residual Head", "Broadcast T_t0\n(B, 1, 32, 48, 7)", "#feebc8", border="#dd6b20")

    # Encoder Stages (Left Downwards)
    draw_box(4, 62, 14, 8, "Encoder Stage 1", "Conv3D + MultiScale\n16 filters (6, 32, 48)", "#c3dafe")
    draw_box(4, 48, 14, 6, "MaxPool3D (1,2,2)", "(B, 6, 16, 24, 16)", "#ebf8ff")
    draw_box(4, 34, 14, 8, "Encoder Stage 2", "Conv3D + MultiScale\n32 filters (6, 16, 24)", "#c3dafe")
    draw_box(4, 20, 14, 6, "MaxPool3D (1,2,2)", "(B, 6, 8, 12, 32)", "#ebf8ff")
    draw_box(4, 6, 14, 8, "Encoder Stage 3", "Conv3D + MultiScale\n64 filters (6, 8, 12)", "#c3dafe")

    # Bottom Row: Stage 4, Temporal Compress, Bottleneck
    draw_box(24, 6, 14, 8, "Encoder Stage 4", "Conv3D + MultiScale\n128 filters (6, 4, 6)", "#c3dafe")
    draw_box(42, 6, 16, 8, "CompressLags", "Conv3D (6,1,1) valid\n(B, 1, 4, 6, 256)", "#fefcbf")
    
    # Bottleneck Box
    bneck_color = "#fed7d7" if "kan" in model_key or "mlp" in model_key else "#feebc8"
    draw_box(62, 6, 16, 8, "ASPP Bottleneck", "Dilations 1,6,12,18\n+ ImagePool (256f)", bneck_color, border="#c53030")

    # Decoder Stages (Right Upwards)
    is_kan = model_key in ["L11", "L15", "L14"]
    dec6_sub = "UpSample (1,2,2)\n+ KAN Skip Fusion [*]" if is_kan else "UpSample (1,2,2)\n+ Merge6 (128f)"
    dec7_sub = "UpSample (1,2,2)\n+ KAN Skip Fusion [*]" if is_kan else "UpSample (1,2,2)\n+ Merge7 (64f)"

    draw_box(82, 18, 15, 8, "Decoder Stage 6", dec6_sub, "#c6f6d5", highlight=is_kan)
    draw_box(82, 36, 15, 8, "Decoder Stage 7", dec7_sub, "#c6f6d5", highlight=is_kan)
    draw_box(82, 54, 15, 8, "Decoder Stage 8", "UpSample (1,2,2)\n+ Merge8 (32f)", "#c6f6d5")
    draw_box(82, 72, 15, 8, "Decoder Stage 9", "UpSample (1,2,2)\n+ Merge9 (16f)", "#c6f6d5")

    # Output Projection
    draw_box(82, 88, 15, 7, "Output Projection", "Conv3D 1x1x1 (7f)\n+ Anchor Residual", "#fed7e2", border="#b83280")

    # Connective Arrows
    arrow = dict(arrowstyle="->", lw=1.6, color="#2d3748")
    ax.annotate("", xy=(11, 62), xytext=(11, 78), arrowprops=arrow)
    ax.annotate("", xy=(11, 48), xytext=(11, 62), arrowprops=arrow)
    ax.annotate("", xy=(11, 34), xytext=(11, 48), arrowprops=arrow)
    ax.annotate("", xy=(11, 20), xytext=(11, 34), arrowprops=arrow)
    ax.annotate("", xy=(11, 6), xytext=(11, 20), arrowprops=arrow)

    ax.annotate("", xy=(24, 10), xytext=(18, 10), arrowprops=arrow)
    ax.annotate("", xy=(42, 10), xytext=(38, 10), arrowprops=arrow)
    ax.annotate("", xy=(62, 10), xytext=(58, 10), arrowprops=arrow)
    ax.annotate("", xy=(82, 18), xytext=(78, 10), arrowprops=arrow)

    ax.annotate("", xy=(89.5, 36), xytext=(89.5, 26), arrowprops=arrow)
    ax.annotate("", xy=(89.5, 54), xytext=(89.5, 44), arrowprops=arrow)
    ax.annotate("", xy=(89.5, 72), xytext=(89.5, 62), arrowprops=arrow)
    ax.annotate("", xy=(89.5, 88), xytext=(89.5, 80), arrowprops=arrow)

    # Long Skip Connections (Dotted Red Lines)
    skip_arrow = dict(arrowstyle="->", lw=1.5, color="#e53e3e", linestyle="--")
    ax.annotate("Skip 4 (128f)", xy=(82, 22), xytext=(38, 14), arrowprops=skip_arrow, fontsize=7.5, color="#c53030", fontweight="bold")
    ax.annotate("Skip 3 (64f)", xy=(82, 40), xytext=(18, 12), arrowprops=skip_arrow, fontsize=7.5, color="#c53030", fontweight="bold")
    ax.annotate("Skip 2 (32f)", xy=(82, 58), xytext=(18, 38), arrowprops=skip_arrow, fontsize=7.5, color="#c53030", fontweight="bold")
    ax.annotate("Skip 1 (16f)", xy=(82, 76), xytext=(18, 66), arrowprops=skip_arrow, fontsize=7.5, color="#c53030", fontweight="bold")

    # Anchor Residual Skip Connection (Dotted Orange Line across top)
    anchor_arrow = dict(arrowstyle="->", lw=1.8, color="#dd6b20", linestyle=":")
    ax.annotate("Anchor Residual Addition: Yhat = T_t0 + Delta_T", xy=(82, 92), xytext=(18, 92),
                arrowprops=anchor_arrow, fontsize=8.5, color="#c05621", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved computational graph diagram to: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Print and explain Broad-UNet computational graphs.")
    parser.add_argument("--model", type=str, default="L15",
                        help="Model key: basic, L0, L1..L15, act_tanh, act_leaky_relu, or all")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Generate a high-resolution PNG diagram of the computational graph.")
    args = parser.parse_args()

    if args.model.lower() == "all":
        print("\n" + "#" * 88)
        print("BROAD-UNET ARCHITECTURAL SUITE: MASTER CATALOG OF 18 MODELS")
        print("#" * 88)
        for k, v in MODELS_CATALOG.items():
            print(f"[{k:>14}] : {v['name']} ({v['category']})")
        print("\nPrinting full graph for basic model, top rearrangement (L11), and champion (L15):\n")
        for k in ["basic", "L11", "L15"]:
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

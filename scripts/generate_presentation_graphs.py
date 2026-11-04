"""
scripts/generate_presentation_graphs.py
======================================
Publication-Grade Presentation Architecture Visualizer for Broad-UNet & Derivatives.

Generates 4 high-resolution, 16:9 widescreen presentation slides for TA / Academic presentations:
1. 1_broad_unet_macro_architecture_16x9.png
   - Master macro-architecture diagram of the Apex Champion Model:
     L17: Decoupled Diurnal-Synoptic Broad-UNet with KAN Skip Fusion.
   - Shows tensor flow from (B, 6, 32, 48, 10) to (B, 1, 32, 48, 7), 4 skip highways,
     decoupled prediction heads (Tanh diurnal vs Linear synoptic), and anchor residual bypass.
2. 2_internal_block_mechanisms_deepdive.png
   - Deep-dive 4-panel breakdown:
     * Panel A: Spatio-Temporal Factorized MultiScale 3D Convolution Block (5 parallel branches).
     * Panel B: Atrous Spatial Pyramid Pooling (ASPP) Multi-Rate Receptive Field Bottleneck.
     * Panel C: Skip Connection Fusion: Naive Concatenation (L0) vs KAN B-Spline Fusion (L11/L15/L17).
     * Panel D: Decoupled Multi-Horizon & Anchor Residual Mechanics (Diurnal Tanh vs Synoptic Linear).
3. 3_architectural_evolution_comparison.png
   - Side-by-side comparative evolution: Upstream Baseline vs L0 Hardened Baseline vs L15 Hybrid vs L17 Apex Champion.
4. 4_layer_rearrangement_search_taxonomy.png
   - Comprehensive landscape map of all 20 architectures (L0 to L19) and global activation benchmarks.

Uses pure Python (Matplotlib & NumPy only).
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

# Palette definition
C_NAVY = "#1a365d"
C_SLATE = "#2d3748"
C_MUTED = "#718096"
C_LIGHT_BG = "#f7fafc"
C_CARD_BG = "#ffffff"
C_CARD_BORDER = "#cbd5e0"

# Component colors
C_INPUT = "#ebf8ff"
C_INPUT_BORDER = "#3182ce"
C_ENC = "#e6fffa"
C_ENC_BORDER = "#319795"
C_BOT = "#fefcbf"
C_BOT_BORDER = "#d69e2e"
C_DEC = "#f0fff4"
C_DEC_BORDER = "#38a169"
C_KAN = "#faf5ff"
C_KAN_BORDER = "#805ad5"
C_ANCHOR = "#fffaf0"
C_ANCHOR_BORDER = "#dd6b20"
C_HEAD = "#fff5f5"
C_HEAD_BORDER = "#e53e3e"


def setup_slide_figure(title: str, subtitle: str, slide_badge: str = "SLIDE"):
    """
    Creates a standard 16:9 widescreen canvas with header banner and footer branding.
    """
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=300)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")
    ax.set_xlim(0, 192)
    ax.set_ylim(0, 108)

    # Top Header Banner
    header_box = patches.FancyBboxPatch((4, 94), 184, 10, boxstyle="round,pad=0.5",
                                        facecolor=C_CARD_BG, edgecolor=C_CARD_BORDER, linewidth=1.2, zorder=2)
    ax.add_patch(header_box)

    # Slide Badge
    badge_box = patches.FancyBboxPatch((6, 96), 18, 6, boxstyle="round,pad=0.3",
                                       facecolor=C_NAVY, edgecolor="none", zorder=3)
    ax.add_patch(badge_box)
    ax.text(15, 99, slide_badge, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#ffffff", zorder=4)

    # Title & Subtitle
    ax.text(27, 100, title, ha="left", va="center", fontsize=15.0, fontweight="bold", color=C_NAVY, zorder=4)
    ax.text(27, 96.5, subtitle, ha="left", va="center", fontsize=9.2, fontstyle="italic", color=C_SLATE, zorder=4)

    # Footer Banner
    ax.text(6, 2.5, "ERA5 Multi-Horizon Spatio-Temporal Meteorological Nowcasting | Broad-UNet Research Suite",
            ha="left", va="center", fontsize=8, color=C_MUTED)
    ax.text(186, 2.5, "Agam Ghotra | Machine Learning Research",
            ha="right", va="center", fontsize=8, fontweight="bold", color=C_SLATE)
    ax.plot([4, 188], [4.5, 4.5], color=C_CARD_BORDER, lw=0.8)

    return fig, ax


def draw_styled_box(ax, x, y, w, h, title, subtitle="", tag="", bg_color=C_CARD_BG, border_color=C_CARD_BORDER,
                    title_color=C_SLATE, lw=1.2, radius=0.6, zorder=5):
    """
    Draws a presentation card with title, description, and optional tensor tag chip.
    """
    shadow = patches.FancyBboxPatch((x + 0.35, y - 0.35), w, h, boxstyle=f"round,pad={radius}",
                                    facecolor="#000000", alpha=0.04, edgecolor="none", zorder=zorder-1)
    ax.add_patch(shadow)

    card = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                                  facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=zorder)
    ax.add_patch(card)

    if tag:
        pill_w = len(tag) * 0.72 + 2.0
        pill_h = 2.0
        pill_x = x + w - pill_w - 0.8
        pill_y = y + h - pill_h - 0.6
        pill = patches.FancyBboxPatch((pill_x, pill_y), pill_w, pill_h, boxstyle="round,pad=0.2",
                                      facecolor="#edf2f7", edgecolor=border_color, linewidth=0.8, zorder=zorder+1)
        ax.add_patch(pill)
        ax.text(pill_x + pill_w/2, pill_y + pill_h/2, tag, ha="center", va="center",
                fontsize=6.8, fontweight="bold", color=C_SLATE, zorder=zorder+2)

    ax.text(x + 1.2, y + h - 2.0, title, ha="left", va="center",
            fontsize=8.8, fontweight="bold", color=title_color, zorder=zorder+1)

    if subtitle:
        ax.text(x + 1.2, y + (h - 2.0)/2, subtitle, ha="left", va="center",
                fontsize=7.1, color=C_SLATE, linespacing=1.22, zorder=zorder+1)


# ==============================================================================
# SLIDE 1: Master Macro-Architecture Graph of Apex Champion Model (L17)
# ==============================================================================
def generate_slide_1_macro_architecture(output_path: Path):
    """
    Generates Slide 1: 16:9 Widescreen Master Computational Graph for Apex Champion Model L17.
    """
    fig, ax = setup_slide_figure(
        title="Apex Champion Architecture: Decoupled Diurnal-Synoptic Broad-UNet (L17)",
        subtitle="End-to-end spatiotemporal DAG with KAN B-spline skip fusion, dual-head output decoupling, and anchor residual bypass",
        slide_badge="MACRO TOPOLOGY"
    )

    # 1. ANCHOR RESIDUAL HIGHWAY (Across the top)
    ax.text(10, 89.5, "ANCHOR RESIDUAL BYPASS HIGHWAY (Persistent Temperature Baseline Preservation: Yhat = T_t0 + Delta T)",
            fontsize=8.2, fontweight="bold", color=C_ANCHOR_BORDER)
    highway_line = lines.Line2D([10, 172], [87.5, 87.5], color=C_ANCHOR_BORDER, lw=2.2, linestyle="--", zorder=3)
    ax.add_line(highway_line)
    # Arrow into final addition head
    ax.annotate("", xy=(172, 85), xytext=(172, 87.5),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=C_ANCHOR_BORDER), zorder=4)

    # 2. INPUT WINDOW
    draw_styled_box(
        ax, 8, 71, 24, 11,
        title="Input ERA5 Tensor",
        subtitle="10 Met Channels: t2m, u10,\nv10, sp, d2m, r, etc.\nLookback lags T = 6 (h=-10h..0h)",
        tag="(B, 6, 32, 48, 10)",
        bg_color=C_INPUT, border_color=C_INPUT_BORDER, title_color=C_NAVY
    )

    # Anchor extraction callout
    ax.annotate("Extract t0 Anchor\nTarget T2m frame", xy=(20, 87.5), xytext=(20, 82),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C_ANCHOR_BORDER),
                ha="center", fontsize=7.2, fontweight="bold", color=C_ANCHOR_BORDER, zorder=6)

    # Downward arrow to Encoder 1
    ax.annotate("", xy=(20, 66), xytext=(20, 71),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C_SLATE), zorder=4)

    # 3. ENCODER STAGES
    # Stage 1
    draw_styled_box(
        ax, 8, 55, 24, 11,
        title="Encoder Stage 1",
        subtitle="Conv3D (16f, 3x3x3, LeakyReLU)\n+ MultiScale Factorized Block\nFilters: 16 | Depth: 6",
        tag="(B, 6, 32, 48, 16)",
        bg_color=C_ENC, border_color=C_ENC_BORDER, title_color=C_NAVY
    )
    # Pool 1
    ax.annotate("MaxPool3D (1,2,2)\nPreserves Depth T=6", xy=(20, 48), xytext=(20, 55),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_SLATE),
                ha="center", va="center", fontsize=6.8, color=C_MUTED, zorder=4)

    # Stage 2
    draw_styled_box(
        ax, 8, 37, 24, 11,
        title="Encoder Stage 2",
        subtitle="Conv3D (32f, 3x3x3, LeakyReLU)\n+ MultiScale Factorized Block\nFilters: 32 | Depth: 6",
        tag="(B, 6, 16, 24, 32)",
        bg_color=C_ENC, border_color=C_ENC_BORDER, title_color=C_NAVY
    )
    # Pool 2
    ax.annotate("MaxPool3D (1,2,2)", xy=(20, 30), xytext=(20, 37),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_SLATE),
                ha="center", va="center", fontsize=6.8, color=C_MUTED, zorder=4)

    # Stage 3
    draw_styled_box(
        ax, 8, 19, 24, 11,
        title="Encoder Stage 3",
        subtitle="Conv3D (64f, 3x3x3, LeakyReLU)\n+ MultiScale Factorized Block\nFilters: 64 | Depth: 6",
        tag="(B, 6, 8, 12, 64)",
        bg_color=C_ENC, border_color=C_ENC_BORDER, title_color=C_NAVY
    )
    # Pool 3
    ax.annotate("MaxPool3D (1,2,2)", xy=(20, 12), xytext=(20, 19),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_SLATE),
                ha="center", va="center", fontsize=6.8, color=C_MUTED, zorder=4)

    # Stage 4
    draw_styled_box(
        ax, 38, 6, 26, 11,
        title="Encoder Stage 4 + Drop",
        subtitle="Conv3D (128f, LeakyReLU) +\nMultiScale Factorized Block\nSpatial Dropout (p = 0.2)",
        tag="(B, 6, 4, 6, 128)",
        bg_color=C_ENC, border_color=C_ENC_BORDER, title_color=C_NAVY
    )
    ax.annotate("", xy=(38, 11.5), xytext=(20, 12),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C_SLATE), zorder=4)

    # 4. BOTTLENECK TRANSITION & ASPP
    draw_styled_box(
        ax, 70, 6, 24, 11,
        title="CompressLags (Temporal)",
        subtitle="Conv3D (256f, kernel=(6,1,1))\nPadding='valid' (T: 6 -> 1)\nCrushes antecedent lag dim",
        tag="(B, 1, 4, 6, 256)",
        bg_color=C_BOT, border_color=C_BOT_BORDER, title_color="#744210"
    )
    ax.annotate("Pool4 (1,2,2)\n+ Conv5 (256f)", xy=(70, 11.5), xytext=(64, 11.5),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C_SLATE),
                ha="center", va="bottom", fontsize=6.8, color=C_MUTED, zorder=4)

    # ASPP Bottleneck
    draw_styled_box(
        ax, 100, 6, 26, 11,
        title="ASPP Pyramid Bottleneck",
        subtitle="5 Parallel Dilated Paths:\nRates: 1, 6, 12, 18 + ImagePool\nLeakyReLU (alpha = 0.2)",
        tag="(B, 1, 4, 6, 256)",
        bg_color=C_BOT, border_color=C_BOT_BORDER, title_color="#744210"
    )
    ax.annotate("", xy=(100, 11.5), xytext=(94, 11.5),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C_SLATE), zorder=4)

    # 5. DECODER STAGES
    ax.annotate("UpSample (1,2,2)\n+ Conv3D (128f)", xy=(136, 16), xytext=(126, 11.5),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C_SLATE),
                ha="center", va="center", fontsize=6.8, color=C_MUTED, zorder=4)

    # Decoder Stage 6 (KAN Skip Fusion)
    draw_styled_box(
        ax, 136, 16, 36, 11,
        title="Decoder Stage 6 + KAN Skip Fusion [*]",
        subtitle="Up6 (128f) concat Skip4 (128f) -> (256f)\n"
                 "[KAN Block: Cubic B-spline Basis Fusion]\n"
                 "Cuts 388k params; eliminates semantic aliasing",
        tag="(B, 1, 4, 6, 128)",
        bg_color=C_KAN, border_color=C_KAN_BORDER, title_color="#44337a", lw=1.8
    )

    # Up to Decoder Stage 7
    ax.annotate("UpSample (1,2,2)\n+ Conv3D (64f)", xy=(154, 31), xytext=(154, 27),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C_SLATE),
                ha="left", va="center", fontsize=6.8, color=C_MUTED, zorder=4)

    # Decoder Stage 7 (KAN Skip Fusion)
    draw_styled_box(
        ax, 136, 31, 36, 11,
        title="Decoder Stage 7 + KAN Skip Fusion [*]",
        subtitle="Up7 (64f) concat Skip3 (64f) -> (128f)\n"
                 "[KAN Block: Cubic B-spline Basis Fusion]\n"
                 "Aligns mesoscale spatial gradients smoothly",
        tag="(B, 1, 8, 12, 64)",
        bg_color=C_KAN, border_color=C_KAN_BORDER, title_color="#44337a", lw=1.8
    )

    # Up to Decoder Stage 8
    ax.annotate("UpSample (1,2,2)\n+ Conv3D (32f)", xy=(154, 46), xytext=(154, 42),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C_SLATE),
                ha="left", va="center", fontsize=6.8, color=C_MUTED, zorder=4)

    # Decoder Stage 8
    draw_styled_box(
        ax, 136, 46, 36, 10,
        title="Decoder Stage 8 (Linear Merge)",
        subtitle="Up8 (32f) concat Skip2 (32f) -> (64f)\nConv3D (32f) + MultiScale Factorized Block\nLeakyReLU (alpha = 0.2)",
        tag="(B, 1, 16, 24, 32)",
        bg_color=C_DEC, border_color=C_DEC_BORDER, title_color=C_NAVY
    )

    # Up to Decoder Stage 9
    ax.annotate("UpSample (1,2,2)\n+ Conv3D (16f)", xy=(154, 60), xytext=(154, 56),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C_SLATE),
                ha="left", va="center", fontsize=6.8, color=C_MUTED, zorder=4)

    # Decoder Stage 9 (Full Resolution Trunk)
    draw_styled_box(
        ax, 136, 60, 36, 10,
        title="Decoder Stage 9 (Feature Trunk)",
        subtitle="Up9 (16f) concat Skip1 (16f) -> Conv3D(16f)\n"
                 "MultiScale Factorized Block -> Latent Repr F\n"
                 "Feeds Decoupled Multi-Horizon Output Heads",
        tag="(B, 1, 32, 48, 16)",
        bg_color=C_DEC, border_color=C_DEC_BORDER, title_color=C_NAVY
    )

    # Fork arrows from Decoder Stage 9 into Diurnal and Synoptic Heads
    ax.annotate("", xy=(131, 74), xytext=(146, 70),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#d69e2e"), zorder=4)
    ax.annotate("", xy=(165, 74), xytext=(162, 70),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#3182ce"), zorder=4)

    # 6. DECOUPLED PREDICTION HEADS (The Game-Changing Innovation)
    # Head 1: Diurnal Head
    draw_styled_box(
        ax, 118, 74, 25, 9.5,
        title="Head A: Diurnal Head [*]",
        subtitle="Lead Times: +2h, +4h, +8h, +16h\n"
                 "Tanh Bounded: 15.0 * tanh(Conv3D)\n"
                 "Captures periodic solar warming",
        tag="(B, 1, 32, 48, 4)",
        bg_color="#fffaf0", border_color="#d69e2e", title_color="#744210", lw=1.6
    )

    # Head 2: Synoptic Head
    draw_styled_box(
        ax, 147, 74, 25, 9.5,
        title="Head B: Synoptic Head [*]",
        subtitle="Lead Times: +24h, +48h, +72h\n"
                 "Linear Conv3D (unconstrained)\n"
                 "Captures multi-day front advection",
        tag="(B, 1, 32, 48, 3)",
        bg_color="#ebf8ff", border_color="#3182ce", title_color=C_NAVY, lw=1.6
    )

    # Arrows into Concat & Anchor Addition Head
    ax.annotate("", xy=(138, 85), xytext=(131, 83.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_SLATE), zorder=4)
    ax.annotate("", xy=(156, 85), xytext=(159, 83.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_SLATE), zorder=4)

    # Concat & Anchor Addition
    draw_styled_box(
        ax, 134, 85, 42, 7.5,
        title="Residual Addition: Yhat = T_t0 + [Delta_T_diurnal, Delta_T_synoptic]",
        subtitle="Evaluated across 7 forecast horizons in Kelvin (+2h to +72h) | Loss: Masked MSE",
        tag="(B, 1, 32, 48, 7)",
        bg_color=C_HEAD, border_color=C_HEAD_BORDER, title_color="#9b2c2c", lw=2.2
    )

    # 7. LONG SKIP CONNECTION HIGHWAYS
    skip_style = dict(arrowstyle="->", lw=1.6, linestyle="--", color="#e53e3e")

    # Skip 4: Stage 4 -> Decoder 6
    ax.annotate("", xy=(136, 21.5), xytext=(64, 13),
                arrowprops=skip_style, zorder=4)
    ax.text(95, 19, "Skip 4: Conv3D(128f, (6,1,1), valid)", fontsize=7.0, fontweight="bold",
            color="#c53030", rotation=13, ha="center")

    # Skip 3: Stage 3 -> Decoder 7
    ax.annotate("", xy=(136, 36.5), xytext=(32, 24),
                arrowprops=skip_style, zorder=4)
    ax.text(80, 31, "Skip 3: Conv3D(64f, (6,1,1), valid)", fontsize=7.0, fontweight="bold",
            color="#c53030", rotation=11, ha="center")

    # Skip 2: Stage 2 -> Decoder 8
    ax.annotate("", xy=(136, 51), xytext=(32, 41.5),
                arrowprops=skip_style, zorder=4)
    ax.text(80, 48, "Skip 2: Conv3D(32f, (6,1,1), valid)", fontsize=7.0, fontweight="bold",
            color="#c53030", rotation=9, ha="center")

    # Skip 1: Stage 1 -> Decoder 9
    ax.annotate("", xy=(136, 65), xytext=(32, 59.5),
                arrowprops=skip_style, zorder=4)
    ax.text(80, 64, "Skip 1: Conv3D(16f, (6,1,1), valid)", fontsize=7.0, fontweight="bold",
            color="#c53030", rotation=7, ha="center")

    # 8. ARCHITECTURAL SUMMARY CALLOUT CARD
    summary_box = patches.FancyBboxPatch((66, 36), 62, 21, boxstyle="round,pad=0.8",
                                         facecolor="#ffffff", edgecolor="#4a5568", linewidth=1.4, zorder=6)
    ax.add_patch(summary_box)
    ax.text(97, 54, "WHY L17 IS THE APEX CHAMPION (3.211 K VAL RMSE)", ha="center", va="center",
            fontsize=9.8, fontweight="bold", color=C_NAVY, zorder=7)
    
    innovations = [
        "1. Decoupled Diurnal vs Synoptic Heads: Eliminates multi-task gradient conflict",
        "   between bounded diurnal oscillations (Tanh) and linear front advection (+24..+72h).",
        "   Drives H4 (+8h) RMSE from 5.38 K down to 4.28 K (massive -1.10 K drop!).",
        "2. KAN Skip Fusion at Merge6 & Merge7: Replaces rigid concatenation with cubic",
        "   B-spline bases; cuts 388k params (-3.65%) while boosting cross-scale alignment.",
        "3. Global LeakyReLU (alpha=0.2): Eradicates dying ReLU neurons across 3D convs,",
        "   enabling unbroken gradient flow during extreme cold-advection frontal regimes.",
        "4. Held-Out Test Verification: Achieves 3.178 K Test RMSE (+10.15% gain vs L0 baseline),",
        "   with 100% statistically significant skill gains across short and long horizons."
    ]
    ax.text(68, 44, "\n".join(innovations), ha="left", va="center", fontsize=7.2, color=C_SLATE,
            linespacing=1.32, zorder=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Slide 1] Saved Macro Architecture Graph: {output_path}")


# ==============================================================================
# SLIDE 2: Micro-Architectural Deep-Dive (The 4 Internal Engines)
# ==============================================================================
def generate_slide_2_internal_mechanisms(output_path: Path):
    """
    Generates Slide 2: 4-panel micro-architectural breakdown of core internal blocks.
    """
    fig, ax = setup_slide_figure(
        title="Micro-Architectural Deep Dive: Four Internal Innovation Engines",
        subtitle="Mathematical formulation and topological design of MultiScale Conv3D, ASPP Receptive Fields, KAN B-Splines, and Decoupled Heads",
        slide_badge="BLOCK MECHANISMS"
    )

    # 4 Quadrant Cards
    quadrants = [
        ("PANEL A: SPATIO-TEMPORAL FACTORIZED CONVOLUTION", 7, 50, 87, 41, C_ENC_BORDER),
        ("PANEL B: ATROUS SPATIAL PYRAMID POOLING (ASPP)", 98, 50, 87, 41, C_BOT_BORDER),
        ("PANEL C: SKIP FUSION: NAIVE CONCAT VS KAN B-SPLINE", 7, 6, 87, 41, C_KAN_BORDER),
        ("PANEL D: DECOUPLED DIURNAL-SYNOPTIC & ANCHOR HEADS", 98, 6, 87, 41, C_HEAD_BORDER),
    ]

    for q_title, qx, qy, qw, qh, q_border in quadrants:
        q_box = patches.FancyBboxPatch((qx, qy), qw, qh, boxstyle="round,pad=0.6",
                                       facecolor=C_CARD_BG, edgecolor=q_border, linewidth=1.2, zorder=2)
        ax.add_patch(q_box)
        t_box = patches.FancyBboxPatch((qx, qy + qh - 4.5), qw, 4.5, boxstyle="round,pad=0.2",
                                       facecolor=q_border, edgecolor="none", zorder=3)
        ax.add_patch(t_box)
        ax.text(qx + 3, qy + qh - 2.2, q_title, ha="left", va="center",
                fontsize=8.5, fontweight="bold", color="#ffffff", zorder=4)

    # --------------------------------------------------------------------------
    # PANEL A: SPATIO-TEMPORAL MULTISCALE FACTORIZED CONV
    # --------------------------------------------------------------------------
    draw_styled_box(ax, 10, 65, 14, 18, "Input Tensor", "Lag sequence\n+ Met channels",
                    tag="(B, 6, H, W, C)", bg_color=C_INPUT, border_color=C_INPUT_BORDER, title_color=C_NAVY)

    branches = [
        ("Branch 1: Pointwise", "1x1x1 Conv (C -> C/4)", 82),
        ("Branch 2: Longitudinal", "1x1x3 Conv (C -> C/4)", 76),
        ("Branch 3: Latitudinal", "1x3x1 Conv (C -> C/4)", 70),
        ("Branch 4: Spatio-Temporal", "1x3x3 Conv (C -> C/4)", 64),
        ("Branch 5: Temporal-Only", "3x1x1 Conv (C -> C/4)", 58)
    ]

    for bname, bdesc, by in branches:
        bbox = patches.FancyBboxPatch((28, by), 38, 4.8, boxstyle="round,pad=0.3",
                                      facecolor="#f7fafc", edgecolor="#e2e8f0", linewidth=0.8, zorder=4)
        ax.add_patch(bbox)
        ax.text(29.5, by + 3.2, bname, fontsize=7.2, fontweight="bold", color=C_NAVY, zorder=5)
        ax.text(29.5, by + 1.2, bdesc, fontsize=6.2, color=C_MUTED, zorder=5)
        ax.annotate("", xy=(28, by + 2.4), xytext=(24, 74),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=C_SLATE), zorder=3)

    draw_styled_box(ax, 70, 65, 20, 18, "Concat & Project",
                    "Concat(B1..B5)\n-> Conv3D(1x1x1)\n-> LeakyReLU(0.2)\nShape: (B, 6, H, W, C)",
                    tag="Fused Output", bg_color=C_ENC, border_color=C_ENC_BORDER, title_color=C_NAVY)
    for _, _, by in branches:
        ax.annotate("", xy=(70, 74), xytext=(66, by + 2.4),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=C_SLATE), zorder=3)

    ax.text(10, 52, "Factorization: Separates spatial (dx, dy) from temporal (dt) advection, saving 68% MACs vs naive 3x3x3 convs.",
            fontsize=6.5, fontstyle="italic", color=C_SLATE, zorder=5)

    # --------------------------------------------------------------------------
    # PANEL B: ASPP MULTI-RATE BOTTLENECK
    # --------------------------------------------------------------------------
    draw_styled_box(ax, 101, 65, 14, 18, "Compressed", "Lags crushed\n(valid 6x1x1)",
                    tag="(B, 1, 4, 6, 256)", bg_color=C_BOT, border_color=C_BOT_BORDER, title_color="#744210")

    aspp_branches = [
        ("Path 1: Rate 1 (1x1x1 Conv3D)", "Local grid-cell point physics & column dynamics", 82),
        ("Path 2: Rate 6 (1x3x3 Atrous Conv)", "Mesoscale convective storms & frontal boundaries", 76),
        ("Path 3: Rate 12 (1x3x3 Atrous Conv)", "Sub-synoptic cyclonic circulations (400-800 km)", 70),
        ("Path 4: Rate 18 (1x3x3 Atrous Conv)", "Continental planetary Rossby waves (> 1,200 km)", 64),
        ("Path 5: Image-Level Global Avg Pool", "Domain-wide thermodynamic equilibrium context", 58)
    ]

    for aname, adesc, ay in aspp_branches:
        abox = patches.FancyBboxPatch((119, ay), 44, 4.8, boxstyle="round,pad=0.3",
                                      facecolor="#fffaf0", edgecolor="#feebc8", linewidth=0.8, zorder=4)
        ax.add_patch(abox)
        ax.text(120.5, ay + 3.2, aname, fontsize=7.2, fontweight="bold", color="#744210", zorder=5)
        ax.text(120.5, ay + 1.2, adesc, fontsize=6.2, color=C_MUTED, zorder=5)
        ax.annotate("", xy=(119, ay + 2.4), xytext=(115, 74),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#ecc94b"), zorder=3)

    draw_styled_box(ax, 167, 65, 16, 18, "ASPP Output",
                    "Concat(P1..P5)\n-> Conv3D(1x1x1)\n-> Dropout(0.2)\nMulti-scale\nreceptive field",
                    tag="(B, 1, 4, 6, 256)", bg_color=C_BOT, border_color=C_BOT_BORDER, title_color="#744210")
    for _, _, ay in aspp_branches:
        ax.annotate("", xy=(167, 74), xytext=(163, ay + 2.4),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#ecc94b"), zorder=3)

    ax.text(101, 52, "Receptive Field: Dilations 1..18 span localized micro-climates to multi-state jet streams without losing spatial resolution.",
            fontsize=6.5, fontstyle="italic", color=C_SLATE, zorder=5)

    # --------------------------------------------------------------------------
    # PANEL C: SKIP CONNECTION FUSION (NAIVE VS KAN B-SPLINE)
    # --------------------------------------------------------------------------
    naive_box = patches.FancyBboxPatch((9, 13), 36, 30, boxstyle="round,pad=0.4",
                                       facecolor="#edf2f7", edgecolor="#cbd5e0", linewidth=1.0, zorder=3)
    ax.add_patch(naive_box)
    ax.text(27, 40, "BASELINE: Naive Concatenation (L0)", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=C_SLATE, zorder=4)
    ax.text(11, 33, "Encoder Skip: (B, 1, H, W, C)\nDecoder Up:    (B, 1, H, W, C)",
            fontsize=7.0, color=C_SLATE, zorder=4)
    ax.text(11, 26, "Merge: Concatenate([Skip, Up])\n-> Dense/Conv(2C -> C)\n-> Fixed ReLU: max(0, x)",
            fontsize=7.0, color=C_SLATE, zorder=4)
    ax.text(11, 16, "[!] Failure Mode: Rigid linear fusion;\nshallow textures clash with deep\nsemantics -> Cross-scale aliasing.",
            fontsize=6.8, fontweight="bold", color="#c53030", zorder=4)

    # Right Sub-panel: KAN Skip Fusion (L11/L15/L17)
    kan_sub_box = patches.FancyBboxPatch((49, 13), 43, 30, boxstyle="round,pad=0.4",
                                         facecolor=C_KAN, edgecolor=C_KAN_BORDER, linewidth=1.4, zorder=3)
    ax.add_patch(kan_sub_box)
    ax.text(70.5, 40, "OUR INNOVATION: KAN Skip Fusion (L11 / L15 / L17)", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#44337a", zorder=4)
    
    ax.text(51, 34, "phi(x) = b(x) * W_base + Sum_k c_k * B_k(x)",
            fontsize=7.5, fontweight="bold", color="#553c9a", zorder=4)
    ax.text(51, 28, "- b(x) = SiLU(x) (smooth base projection)\n- B_k(x): Cubic B-spline bases across uniform knots\n- c_k: Learnable coefficients per channel edge",
            fontsize=6.8, color=C_SLATE, zorder=4)

    knot_x = np.linspace(-2, 2, 80)
    b1 = np.exp(-((knot_x + 1.0) ** 2) / 0.3)
    b2 = np.exp(-((knot_x) ** 2) / 0.3)
    b3 = np.exp(-((knot_x - 1.0) ** 2) / 0.3)
    
    px_plot = 53 + (knot_x + 2.0) * 8.5
    ax.plot(px_plot, 16 + b1 * 4, color="#805ad5", lw=1.2, zorder=5)
    ax.plot(px_plot, 16 + b2 * 4, color="#3182ce", lw=1.2, zorder=5)
    ax.plot(px_plot, 16 + b3 * 4, color="#38a169", lw=1.2, zorder=5)
    ax.text(70.5, 14.5, "Learnable Activation Curves (Cubic B-Splines)", ha="center", fontsize=6.2, color=C_MUTED, zorder=5)

    ax.text(9, 8, "Empirical Result: KAN Skip Fusion cuts 388k parameters while boosting validation accuracy across all horizons.",
            fontsize=6.5, fontstyle="italic", color=C_SLATE, zorder=5)

    # --------------------------------------------------------------------------
    # PANEL D: DECOUPLED PREDICTION HEADS & ANCHOR RESIDUAL FORMULATION
    # --------------------------------------------------------------------------
    # Sub-box 1: Diurnal Head
    d_box = patches.FancyBboxPatch((100, 26), 40, 16, boxstyle="round,pad=0.3",
                                   facecolor="#fffaf0", edgecolor="#d69e2e", linewidth=1.2, zorder=3)
    ax.add_patch(d_box)
    ax.text(120, 39, "DIURNAL HEAD (+2h, +4h, +8h, +16h)", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#744210", zorder=4)
    ax.text(102, 33, "Formula: Delta_T_diurnal = 15.0 * tanh(Conv3D_4(F))\n"
                     "- Bounded oscillating perturbation (|Delta T| <= 15 K)\n"
                     "- Perfectly models daily solar heating / radiative cooling\n"
                     "- Slashes H4 (+8h) error by -1.10 K vs single-head models!",
            fontsize=6.6, color=C_SLATE, linespacing=1.25, zorder=4)

    # Sub-box 2: Synoptic Head
    s_box = patches.FancyBboxPatch((143, 26), 40, 16, boxstyle="round,pad=0.3",
                                   facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.2, zorder=3)
    ax.add_patch(s_box)
    ax.text(163, 39, "SYNOPTIC HEAD (+24h, +48h, +72h)", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=C_NAVY, zorder=4)
    ax.text(145, 33, "Formula: Delta_T_synoptic = Conv3D_3(F) [Linear]\n"
                     "- Unconstrained advection projection across days\n"
                     "- Preserves large-scale baroclinic frontal shifts\n"
                     "- Zero gradient conflict with diurnal oscillations",
            fontsize=6.6, color=C_SLATE, linespacing=1.25, zorder=4)

    # Bottom Residual Synthesis Equation
    eq_box = patches.FancyBboxPatch((100, 9), 83, 14, boxstyle="round,pad=0.4",
                                    facecolor="#ffffff", edgecolor=C_HEAD_BORDER, linewidth=1.4, zorder=3)
    ax.add_patch(eq_box)
    ax.text(141.5, 20.5, "ANCHOR RESIDUAL SYNTHESIS: Yhat = T_t0 + [Delta_T_diurnal, Delta_T_synoptic]",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#9b2c2c", zorder=4)
    ax.text(102, 14.5, "1. Anchor T_t0 pins the atmospheric thermodynamic background, completely eliminating cold collapse.\n"
                       "2. Decoupling diurnal and synoptic heads eliminates negative multi-task gradient interference,\n"
                       "   yielding Mean Val RMSE of 3.211 K (+8.77% over L0 baseline) and Test RMSE of 3.178 K (+10.15%).",
            fontsize=6.8, color=C_SLATE, linespacing=1.28, zorder=4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Slide 2] Saved Micro Mechanisms Deep-Dive: {output_path}")


# ==============================================================================
# SLIDE 3: Architectural Evolution & Side-by-Side Comparison
# ==============================================================================
def generate_slide_3_architectural_evolution(output_path: Path):
    """
    Generates Slide 3: 4-Column Architectural Evolution Sheet comparing Upstream vs L0 vs L15 vs L17.
    """
    fig, ax = setup_slide_figure(
        title="Architectural Evolution: From Upstream Baseline to Apex Champion L17",
        subtitle="Step-by-step structural breakthroughs resolving cold bias (-54%), skip aliasing (-388k params), and multi-task diurnal conflict (+8.77%)",
        slide_badge="EVOLUTION MATRIX"
    )

    models_data = [
        {
            "id": "Upstream Broad-UNet",
            "author": "Fernández & Mehrkanoon (2021)",
            "role": "Raw Upstream Benchmark",
            "bg": "#f7fafc",
            "border": "#a0aec0",
            "title_col": C_SLATE,
            "specs": [
                ("Residual Head", "None (Direct Absolute Prediction)"),
                ("Backbone Act", "Standard Piecewise Linear ReLU"),
                ("Skip Merges", "Naive Linear Concatenation"),
                ("Bottleneck", "Standard Multi-rate ASPP (256f)"),
                ("Prediction Head", "Single Linear Head (7 horizons)"),
                ("Parameters", "10,642,887 (10.64M)"),
                ("Val RMSE", "7.340 K (Severe Cold Collapse)"),
                ("Test Skill vs Pers", "-0.840 (Fails Baseline Check)")
            ],
            "badge": "RAW UPSTREAM",
            "badge_bg": "#e2e8f0",
            "badge_col": C_SLATE
        },
        {
            "id": "L0 Baseline Broad-UNet",
            "author": "Phase 0 Reproduction",
            "role": "Hardened Reference Baseline",
            "bg": "#ebf8ff",
            "border": "#3182ce",
            "title_col": C_NAVY,
            "specs": [
                ("Residual Head", "Anchor Residual: Yhat = T_t0 + Delta T"),
                ("Backbone Act", "Standard Piecewise Linear ReLU"),
                ("Skip Merges", "Naive Linear Concatenation"),
                ("Bottleneck", "Standard Multi-rate ASPP (256f)"),
                ("Prediction Head", "Single Linear Head (7 horizons)"),
                ("Parameters", "10,642,935 (10.64M)"),
                ("Val RMSE", "3.520 K (-52.0% vs Upstream)"),
                ("Test Skill vs Pers", "+0.001 (Test Mean RMSE 3.537 K)")
            ],
            "badge": "+ ANCHOR RESIDUAL",
            "badge_bg": "#bee3f8",
            "badge_col": C_NAVY
        },
        {
            "id": "L15 Champion Hybrid",
            "author": "Tournament Rank 4 / 20",
            "role": "KAN Skip + LeakyReLU Backbone",
            "bg": "#faf5ff",
            "border": "#805ad5",
            "title_col": "#44337a",
            "specs": [
                ("Residual Head", "Anchor Residual: Yhat = T_t0 + Delta T"),
                ("Backbone Act", "LeakyReLU (alpha=0.2) Everywhere"),
                ("Skip Merges", "KAN B-Spline Fusion (Merge 6 & 7)"),
                ("Bottleneck", "LeakyReLU ASPP (alpha=0.2)"),
                ("Prediction Head", "Single Linear Head (7 horizons)"),
                ("Parameters", "10,254,583 (-388k params cut)"),
                ("Val RMSE", "3.369 K (+4.29% over L0)"),
                ("Test Skill vs Pers", "+0.041 (Test Mean RMSE 3.395 K)")
            ],
            "badge": "+ KAN SKIPS + LEAKYRELU",
            "badge_bg": "#e9d8fd",
            "badge_col": "#44337a"
        },
        {
            "id": "L17 Apex Champion",
            "author": "Next-Gen Tournament Winner",
            "role": "Decoupled Diurnal-Synoptic Heads",
            "bg": "#fff5f5",
            "border": "#e53e3e",
            "title_col": "#9b2c2c",
            "specs": [
                ("Residual Head", "Anchor Residual: Yhat = T_t0 + Delta T"),
                ("Backbone Act", "LeakyReLU (alpha=0.2) Everywhere"),
                ("Skip Merges", "KAN B-Spline Fusion (Merge 6 & 7)"),
                ("Bottleneck", "LeakyReLU ASPP (alpha=0.2)"),
                ("Prediction Head", "Decoupled: Tanh Diurnal + Lin Synoptic"),
                ("Parameters", "10,254,583 (-388k params cut)"),
                ("Val RMSE", "3.211 K (+8.77% over L0) [*]"),
                ("Test Skill vs Pers", "+0.068 (Test Mean RMSE 3.178 K) [*]")
            ],
            "badge": "[*] APEX CHAMPION (L17)",
            "badge_bg": "#fed7d7",
            "badge_col": "#9b2c2c"
        }
    ]

    col_w = 42
    start_x = 7

    for idx, m in enumerate(models_data):
        cx = start_x + idx * (col_w + 3.5)
        cy = 8
        ch = 83

        card = patches.FancyBboxPatch((cx, cy), col_w, ch, boxstyle="round,pad=0.7",
                                      facecolor=m["bg"], edgecolor=m["border"],
                                      linewidth=2.0 if idx == 3 else 1.2, zorder=2)
        ax.add_patch(card)

        b_box = patches.FancyBboxPatch((cx + 2, cy + ch - 5.5), col_w - 4, 4.2, boxstyle="round,pad=0.2",
                                      facecolor=m["badge_bg"], edgecolor=m["border"], linewidth=0.8, zorder=3)
        ax.add_patch(b_box)
        ax.text(cx + col_w/2, cy + ch - 3.4, m["badge"], ha="center", va="center",
                fontsize=7.8, fontweight="bold", color=m["badge_col"], zorder=4)

        ax.text(cx + col_w/2, cy + ch - 8.5, m["id"], ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=m["title_col"], zorder=4)
        ax.text(cx + col_w/2, cy + ch - 11.5, f"{m['author']} | {m['role']}", ha="center", va="center",
                fontsize=6.8, fontstyle="italic", color=C_MUTED, zorder=4)

        ax.plot([cx + 2, cx + col_w - 2], [cy + ch - 13.5, cy + ch - 13.5], color=m["border"], lw=0.8, zorder=3)

        spec_y = cy + ch - 18
        for label, val in m["specs"]:
            is_highlight = "Val RMSE" in label or "Test Skill" in label
            r_bg = "#ffffff" if not is_highlight else ("#feebc8" if idx < 3 else "#c6f6d5")
            r_box = patches.FancyBboxPatch((cx + 2, spec_y - 1.5), col_w - 4, 5.2, boxstyle="round,pad=0.2",
                                          facecolor=r_bg, edgecolor="#cbd5e0", linewidth=0.6, zorder=3)
            ax.add_patch(r_box)

            ax.text(cx + 3.5, spec_y + 2.0, label.upper(), fontsize=5.8, fontweight="bold", color=C_MUTED, zorder=4)
            ax.text(cx + 3.5, spec_y - 0.2, val, fontsize=6.8,
                    fontweight="bold" if is_highlight else "normal",
                    color=C_NAVY if not is_highlight else ("#744210" if idx < 3 else "#22543d"), zorder=4)
            spec_y -= 6.8

        diff_box = patches.FancyBboxPatch((cx + 2, cy + 2), col_w - 4, 16, boxstyle="round,pad=0.4",
                                          facecolor="#ffffff", edgecolor=m["border"], linewidth=0.8, zorder=3)
        ax.add_patch(diff_box)
        ax.text(cx + col_w/2, cy + 15.5, "ARCHITECTURAL ROLE", ha="center", va="center",
                fontsize=6.8, fontweight="bold", color=m["title_col"], zorder=4)

        if idx == 0:
            diff_text = ("- Unstabilized temperature drift\n"
                         "- Direct absolute regression\n"
                         "- Severe systematic cold bias\n"
                         "- Fails persistence benchmark")
        elif idx == 1:
            diff_text = ("- Establishes hardened baseline\n"
                         "- Residual anchor resolves bias\n"
                         "- Preserves original U-Net core\n"
                         "- Standard single-head output")
        elif idx == 2:
            diff_text = ("- Learnable B-spline skip edges\n"
                         "- Cuts 388k redundant weights\n"
                         "- LeakyReLU stops dying neurons\n"
                         "- Strong validation gain (+4.29%)")
        else:
            diff_text = ("- Decoupled diurnal/synoptic heads\n"
                         "- Resolves multi-task gradient clashes\n"
                         "- Lowest Val RMSE (3.211 K, +8.77%)\n"
                         "- Held-out Test RMSE 3.178 K (+10.15%)")

        ax.text(cx + 3.5, cy + 8.5, diff_text, fontsize=6.5, color=C_SLATE, linespacing=1.3, zorder=4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Slide 3] Saved Architectural Evolution Matrix: {output_path}")


# ==============================================================================
# SLIDE 4: Architectural Search Taxonomy & Leaderboard Landscape
# ==============================================================================
def generate_slide_4_search_taxonomy(output_path: Path):
    """
    Generates Slide 4: Complete landscape taxonomy of all 20 architectures (L0 to L19) & activations.
    """
    fig, ax = setup_slide_figure(
        title="Architecture Search Landscape: 20 Architectures Across Four Discovery Frontiers",
        subtitle="Empirical performance taxonomy across bottleneck rearrangements, skip fusions, channel attention, and decoupled multi-horizon heads",
        slide_badge="SEARCH TAXONOMY"
    )

    categories = [
        ("FRONTIER 1: BOTTLENECK MODIFICATIONS (L2 to L9)",
         "Evaluated Conv3D, MLP, and KAN blocks before/after ASPP. Dilations in ASPP saturate receptive capacity; marginal gains (< +0.80%).",
         7, 56, 178, 35, C_BOT, C_BOT_BORDER),
        ("FRONTIER 2: SKIP CONNECTION INNOVATIONS (L10, L11, L16)",
         "Evaluated pre-skip linear convs vs post-skip KAN B-spline fusions. KAN at merge6/7 cuts 388k params with superior non-linear alignment.",
         7, 26, 87, 27, C_KAN, C_KAN_BORDER),
        ("FRONTIER 3: ATTENTION & DECOUPLED HEADS (L15, L17, L18, L19)",
         "Next-generation architectures: Decoupled diurnal/synoptic heads (L17: 3.211 K), SE attention (L18: 3.318 K), and Apex synthesis (L19: 3.241 K).",
         98, 26, 87, 27, C_HEAD, C_HEAD_BORDER),
    ]

    for cat_title, cat_sub, cx, cy, cw, ch, cbg, cbd in categories:
        card = patches.FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.6",
                                      facecolor=C_CARD_BG, edgecolor=cbd, linewidth=1.2, zorder=2)
        ax.add_patch(card)
        h_box = patches.FancyBboxPatch((cx, cy + ch - 4.5), cw, 4.5, boxstyle="round,pad=0.2",
                                       facecolor=cbd, edgecolor="none", zorder=3)
        ax.add_patch(h_box)
        ax.text(cx + 3, cy + ch - 2.2, cat_title, ha="left", va="center",
                fontsize=8.5, fontweight="bold", color="#ffffff", zorder=4)

    # FRONTIER 1 TILES (Bottlenecks)
    bneck_models = [
        ("L2", "Conv Before ASPP", "+590k params", "3.481 K", "Marginal (+1.10%)"),
        ("L3", "Conv After ASPP", "+1.77M params", "3.503 K", "Marginal (+0.47%)"),
        ("L4", "MLP Before ASPP", "Dense 256->512", "3.476 K", "Slight (+1.23%)"),
        ("L5", "MLP After ASPP", "Dense 256->512", "3.470 K", "Slight (+1.40%)"),
        ("L6", "KAN Before ASPP", "B-splines (5 knots)", "3.487 K", "Slight (+0.92%)"),
        ("L7", "KAN After ASPP", "B-splines (5 knots)", "3.535 K", "Degraded (-0.44%)"),
        ("L8", "Sequential MLP->KAN", "Bottleneck Compound", "3.483 K", "Slight (+1.04%)"),
        ("L9", "Sequential KAN->MLP", "Bottleneck Compound", "3.448 K", "Solid (+2.05%)")
    ]

    bx_start = 10
    by_start = 74
    for i, (m_id, m_name, m_feat, m_rmse, m_gain) in enumerate(bneck_models):
        row = i // 4
        col = i % 4
        mx = bx_start + col * 43.5
        my = by_start - row * 11.5

        mbox = patches.FancyBboxPatch((mx, my), 41, 9.5, boxstyle="round,pad=0.3",
                                      facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.8, zorder=4)
        ax.add_patch(mbox)
        ax.text(mx + 2, my + 7.2, f"{m_id}: {m_name}", fontsize=7.2, fontweight="bold", color=C_NAVY, zorder=5)
        ax.text(mx + 2, my + 4.5, f"Structure: {m_feat}", fontsize=6.3, color=C_MUTED, zorder=5)
        ax.text(mx + 2, my + 2.0, f"Val RMSE: {m_rmse} ({m_gain})", fontsize=6.5, fontweight="bold", color="#744210", zorder=5)

    ax.text(10, 58, "Key Bottleneck Finding: Dilations in ASPP already capture 95%+ of receptive field capacity. Adding complexity to bottleneck yielded diminishing returns (< +2.05%).",
            fontsize=7.0, fontstyle="italic", color=C_SLATE, zorder=5)

    # FRONTIER 2 TILES (Skip Connections)
    draw_styled_box(ax, 10, 31, 26, 16,
                    title="L10: Pre-Skip Linear",
                    subtitle="Conv3D projection before\nencoder skip merge.\n\nVal RMSE: 3.497 K (+0.65%)\nParams: 10.66M (+18k)",
                    tag="Rank 13 / 20", bg_color="#ffffff", border_color="#cbd5e0", title_color=C_NAVY)

    draw_styled_box(ax, 38, 31, 27, 16,
                    title="L11: KAN Skip Fusion",
                    subtitle="B-spline fusion at Merge6\n& Merge7.\n\nVal RMSE: 3.427 K (+2.63%)\nParams: 10.25M (-388k!)",
                    tag="Rank 5 / 20", bg_color=C_KAN, border_color=C_KAN_BORDER, title_color="#44337a", lw=1.6)

    draw_styled_box(ax, 67, 31, 25, 16,
                    title="L16: Full KAN Skips",
                    subtitle="4-Stage KAN Skips down\nto full resolution (32x48).\n\nVal RMSE: 3.507 K (+0.38%)\nParams: 10.23M (-412k)",
                    tag="Rank 17 / 20", bg_color="#ffffff", border_color="#cbd5e0", title_color=C_NAVY)

    # FRONTIER 3 TILES (Next-Gen Champions)
    draw_styled_box(ax, 100, 31, 41, 16,
                    title="L18: Spatio-Temporal SE Block",
                    subtitle="Dynamic channel-lag attention inside MultiScale blocks.\n"
                             "Amplifies solar channels by day, wind by front.\n"
                             "Val RMSE: 3.318 K (+5.73% Gain) | Params: 10.31M",
                    tag="Rank 3 BRONZE", bg_color="#fffaf0", border_color="#dd6b20", title_color="#744210", lw=1.6)

    draw_styled_box(ax, 143, 31, 41, 16,
                    title="L17: Decoupled Heads [*]",
                    subtitle="Tanh diurnal head (+2..+16h) + Linear synoptic head.\n"
                             "Eliminates multi-task gradient conflict.\n"
                             "Val RMSE: 3.211 K (+8.77% GAIN) | Test: 3.178 K",
                    tag="Rank 1 APEX WINNER", bg_color="#fff5f5", border_color="#e53e3e", title_color="#9b2c2c", lw=2.2)

    # Bottom Activation Benchmark Ribbon
    act_box = patches.FancyBboxPatch((7, 6), 178, 17, boxstyle="round,pad=0.5",
                                     facecolor="#ffffff", edgecolor=C_CARD_BORDER, linewidth=1.2, zorder=2)
    ax.add_patch(act_box)
    ax.text(10, 20.5, "GLOBAL ACTIVATION BENCHMARK (Held-out Test Split, N=400, 1,000 Paired Bootstrap Resamples)",
            fontsize=8.0, fontweight="bold", color=C_NAVY, zorder=4)

    act_data = [
        ("Standard ReLU", "3.520 K", "Baseline", "#718096", "Suffers dying neurons on cold negative anomalies"),
        ("GELU (Smooth)", "3.488 K", "+0.91%", "#3182ce", "Slight continuous improvement; slower convergence"),
        ("SiLU (Swish)", "3.475 K", "+1.28%", "#319795", "Good gradient flow; modest mid-range improvement"),
        ("Hyperbolic Tanh", "3.374 K", "+4.15%", "#d69e2e", "Bests short-range (+16h: -1.18 K), saturates on extremes"),
        ("LeakyReLU (0.2)", "3.373 K", "+4.17%", "#e53e3e", "Champion robustness across all lead times (+2h..+72h)")
    ]

    for idx, (aname, armse, again, acol, adesc) in enumerate(act_data):
        ax_card_x = 10 + idx * 35.0
        abox = patches.FancyBboxPatch((ax_card_x, 8), 33, 10.5, boxstyle="round,pad=0.3",
                                      facecolor="#f7fafc", edgecolor=acol, linewidth=1.0, zorder=3)
        ax.add_patch(abox)
        ax.text(ax_card_x + 1.5, 16.5, aname, fontsize=7.2, fontweight="bold", color=acol, zorder=4)
        ax.text(ax_card_x + 1.5, 14.0, f"Val RMSE: {armse} ({again})", fontsize=6.8, fontweight="bold", color=C_SLATE, zorder=4)
        ax.text(ax_card_x + 1.5, 10.2, adesc, fontsize=5.8, color=C_MUTED, linespacing=1.2, zorder=4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Slide 4] Saved Search Taxonomy: {output_path}")


def main():
    out_dir = Path("results/presentation_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING PRESENTATION-GRADE COMPUTATIONAL GRAPHS (16:9 Widescreen, 300 DPI)")
    print("=" * 80)

    generate_slide_1_macro_architecture(out_dir / "1_broad_unet_macro_architecture_16x9.png")
    generate_slide_2_internal_mechanisms(out_dir / "2_internal_block_mechanisms_deepdive.png")
    generate_slide_3_architectural_evolution(out_dir / "3_architectural_evolution_comparison.png")
    generate_slide_4_search_taxonomy(out_dir / "4_layer_rearrangement_search_taxonomy.png")

    print("\n[SUCCESS] All 4 presentation graph slides generated successfully in:")
    print(f"  {out_dir.resolve()}")


if __name__ == "__main__":
    main()

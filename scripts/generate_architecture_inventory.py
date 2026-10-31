"""
scripts/generate_architecture_inventory.py
=========================================
Generates the comprehensive, layer-level architectural inventory of original Broad-UNet:
- Layer name, type, position index
- Input shape, output shape
- Parameter counts
- Activation function
- Inbound connected layers
- Generates publication architecture diagram
"""

import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.broad_unet import build_broad_unet

def extract_layer_inventory(model):
    inventory = []
    
    for idx, layer in enumerate(model.layers):
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        params = layer.count_params()
        
        # Activation
        act = "None"
        if hasattr(layer, "activation"):
            act = layer.activation.__name__ if hasattr(layer.activation, "__name__") else str(layer.activation)
            
        # Inbound layers
        inbound = []
        if hasattr(layer, "_inbound_nodes"):
            for node in layer._inbound_nodes:
                if hasattr(node, "inbound_layers"):
                    in_layers = node.inbound_layers
                    if isinstance(in_layers, list):
                        inbound.extend([l.name for l in in_layers])
                    elif in_layers is not None:
                        inbound.append(in_layers.name)
        elif hasattr(layer, "inbound_nodes"):
            for node in layer.inbound_nodes:
                if hasattr(node, "parent_layers"):
                    inbound.extend([l.name for l in node.parent_layers])

        inbound_str = ", ".join(list(dict.fromkeys(inbound))) if inbound else "Input"

        # Shapes
        in_shape = str(layer.input_shape) if hasattr(layer, "input_shape") else "N/A"
        out_shape = str(layer.output_shape) if hasattr(layer, "output_shape") else "N/A"
        
        inventory.append({
            "index": idx,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "input_shape": in_shape,
            "output_shape": out_shape,
            "parameters": params,
            "activation": act,
            "connected_from": inbound_str
        })
        
    return pd.DataFrame(inventory)


def draw_broad_unet_diagram(output_png="results/original_broad_unet_diagram.png"):
    """
    Renders a clean, high-resolution architectural diagram of the original Broad-UNet.
    """
    out_p = Path(output_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Title
    ax.text(50, 96, "Broad-UNet Original Architecture (García Fernández & Mehrkanoon, 2021)",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#1a252f")
    ax.text(50, 92.5, "Factorized Multi-Scale 3D Convolutions + Temporal Compression + Atrous Spatial Pyramid Pooling (ASPP)",
            ha="center", va="center", fontsize=11, fontstyle="italic", color="#555")

    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, title, subtitle, color, border_color="#2c3e50"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                      facecolor=color, edgecolor=border_color, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 1.2, title, ha="center", va="center", fontsize=9, fontweight="bold", color="#2c3e50", zorder=4)
        ax.text(x + w/2, y + h/2 - 1.2, subtitle, ha="center", va="center", fontsize=7.5, color="#34495e", zorder=4)

    # 1. Input
    draw_box(4, 76, 12, 8, "Input Frames", "(B, 6, 32, 48, C)", "#d6eaf8")

    # 2. Encoder stages (Left column downwards)
    # Stage 1
    draw_box(4, 60, 14, 8, "Encoder Stage 1", "Conv3D + MultiScale\n16 filters (6, 32, 48)", "#aed6f1")
    # Pool 1
    draw_box(4, 46, 14, 6, "MaxPool3D (1,2,2)", "(B, 6, 16, 24, 16)", "#ebf5fb")
    # Stage 2
    draw_box(4, 32, 14, 8, "Encoder Stage 2", "Conv3D + MultiScale\n32 filters (6, 16, 24)", "#aed6f1")
    # Pool 2
    draw_box(4, 18, 14, 6, "MaxPool3D (1,2,2)", "(B, 6, 8, 12, 32)", "#ebf5fb")
    # Stage 3
    draw_box(4, 4, 14, 8, "Encoder Stage 3", "Conv3D + MultiScale\n64 filters (6, 8, 12)", "#aed6f1")

    # Stage 4 & Bottleneck (Bottom horizontal)
    draw_box(24, 4, 14, 8, "Encoder Stage 4", "Conv3D + MultiScale\n128 filters (6, 4, 6)", "#aed6f1")
    draw_box(42, 4, 16, 8, "Bottleneck Stage 5", "Conv3D (256 filters)\nCompressLags (1, 2, 3)", "#f9e79f")
    draw_box(62, 4, 16, 8, "ASPP Multi-Scale", "Dilations: 1, 6, 12, 18\n+ Image Pool (256f)", "#f5b041")

    # 3. Decoder stages (Right column upwards)
    # Stage 6
    draw_box(82, 18, 15, 8, "Decoder Stage 6", "UpSampling (1,2,2)\nMerge6 + MultiScale (128f)", "#a9dfbf")
    # Stage 7
    draw_box(82, 36, 15, 8, "Decoder Stage 7", "UpSampling (1,2,2)\nMerge7 + MultiScale (64f)", "#a9dfbf")
    # Stage 8
    draw_box(82, 54, 15, 8, "Decoder Stage 8", "UpSampling (1,2,2)\nMerge8 + MultiScale (32f)", "#a9dfbf")
    # Stage 9
    draw_box(82, 72, 15, 8, "Decoder Stage 9", "UpSampling (1,2,2)\nMerge9 + MultiScale (16f)", "#a9dfbf")
    # Final Output
    draw_box(82, 86, 15, 6, "Output Projection", "Conv3D 1x1x1\n(B, 1, 32, 48, H)", "#f5b7b1")

    # Connective arrows
    arrow = dict(arrowstyle="->", lw=1.8, color="#2c3e50")
    # Encoder down
    ax.annotate("", xy=(11, 60), xytext=(11, 76), arrowprops=arrow)
    ax.annotate("", xy=(11, 46), xytext=(11, 60), arrowprops=arrow)
    ax.annotate("", xy=(11, 32), xytext=(11, 46), arrowprops=arrow)
    ax.annotate("", xy=(11, 18), xytext=(11, 32), arrowprops=arrow)
    ax.annotate("", xy=(11, 4), xytext=(11, 18), arrowprops=arrow)

    # Bottom across
    ax.annotate("", xy=(24, 8), xytext=(18, 8), arrowprops=arrow)
    ax.annotate("", xy=(42, 8), xytext=(38, 8), arrowprops=arrow)
    ax.annotate("", xy=(62, 8), xytext=(58, 8), arrowprops=arrow)
    ax.annotate("", xy=(82, 18), xytext=(78, 8), arrowprops=arrow)

    # Decoder up
    ax.annotate("", xy=(89.5, 36), xytext=(89.5, 26), arrowprops=arrow)
    ax.annotate("", xy=(89.5, 54), xytext=(89.5, 44), arrowprops=arrow)
    ax.annotate("", xy=(89.5, 72), xytext=(89.5, 62), arrowprops=arrow)
    ax.annotate("", xy=(89.5, 86), xytext=(89.5, 80), arrowprops=arrow)

    # Skip connections with temporal lag compression
    skip_arrow = dict(arrowstyle="->", lw=1.6, color="#e74c3c", linestyle="--")
    # Skip 4 -> merge6
    ax.annotate("CompressLags4\n(Temporal Skip)", xy=(82, 22), xytext=(38, 12),
                arrowprops=skip_arrow, fontsize=8, color="#c0392b", fontweight="bold")
    # Skip 3 -> merge7
    ax.annotate("CompressLags3", xy=(82, 40), xytext=(18, 10),
                arrowprops=skip_arrow, fontsize=8, color="#c0392b", fontweight="bold")
    # Skip 2 -> merge8
    ax.annotate("CompressLags2", xy=(82, 58), xytext=(18, 36),
                arrowprops=skip_arrow, fontsize=8, color="#c0392b", fontweight="bold")
    # Skip 1 -> merge9
    ax.annotate("CompressLags1", xy=(82, 76), xytext=(18, 64),
                arrowprops=skip_arrow, fontsize=8, color="#c0392b", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved architecture diagram to {output_png}")


def main():
    print("Building original Broad-UNet model...")
    model = build_broad_unet(lags=6, latitude=32, longitude=48, features=6, features_output=7, filters=16)
    print(f"Total layers: {len(model.layers)} | Total Parameters: {model.count_params():,}")
    
    df_inv = extract_layer_inventory(model)
    csv_path = Path("results/original_architecture_inventory.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_inv.to_csv(csv_path, index=False)
    print(f"Saved inventory CSV to {csv_path}")

    # Generate Markdown table
    md_path = Path("results/original_architecture_inventory.md")
    with open(md_path, "w") as f:
        f.write("# Broad-UNet Original Architecture Inventory\n\n")
        f.write(f"- **Total Model Layers:** {len(model.layers)}\n")
        f.write(f"- **Total Trainable Parameters:** {model.count_params():,}\n")
        f.write("- **Input Tensor Layout:** `(Batch, Lags=6, Lat=32, Lon=48, Features=6)`\n")
        f.write("- **Output Forecast Tensor:** `(Batch, 1, Lat=32, Lon=48, Horizons=7)`\n\n")
        f.write(df_inv.to_markdown(index=False))
    print(f"Saved inventory Markdown to {md_path}")

    draw_broad_unet_diagram("results/original_broad_unet_diagram.png")

if __name__ == "__main__":
    main()


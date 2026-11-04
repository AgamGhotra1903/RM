# Broad-UNet Computational Graph Presentation Guide
**High-Resolution 16:9 Visual Suite, Micro-Architectural Deep Dive, and Speaker Walkthrough**

---

## 1. Executive Summary & Presentation Asset Index

This guide accompanies the presentation-grade computational graph suite developed for academic presentations, TA meetings, and project defenses. All graphs are generated purely in Python (Matplotlib & NumPy) at **16:9 widescreen ($5730 \times 3210$ pixels, 300 DPI)** with publication-grade color palettes, drop shadows, tensor shape tags, and internal block schematics.

### Core Presentation Slides (16:9 Widescreen)

| Slide / Figure | Title | Resolution | Primary Focus |
| :--- | :--- | :--- | :--- |
| **Slide 1** | [1_broad_unet_macro_architecture_16x9.png](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/presentation_graphs/1_broad_unet_macro_architecture_16x9.png) | $5730 \times 3210$ | Champion L15 Macro DAG Topology, 4 Downsampling Stages, 4 Skip Highways, Anchor Bypass |
| **Slide 2** | [2_internal_block_mechanisms_deepdive.png](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/presentation_graphs/2_internal_block_mechanisms_deepdive.png) | $5730 \times 3210$ | 4 Micro Engines: Factorized 3D Convs, ASPP Dilated Pyramid, KAN Spline Fusion, Residual Math |
| **Slide 3** | [3_architectural_evolution_comparison.png](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/presentation_graphs/3_architectural_evolution_comparison.png) | $5730 \times 3210$ | 4-Model Comparative Matrix: Upstream vs L0 vs L11 vs L15 Champion |
| **Slide 4** | [4_layer_rearrangement_search_taxonomy.png](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/presentation_graphs/4_layer_rearrangement_search_taxonomy.png) | $5730 \times 3210$ | Complete 15 Rearrangement Taxonomy (Bottleneck, Skip, Order) + Activation Benchmark |

### Dedicated Model Computational Graphs (Full Diagrams)

| Model Key | Model Name | Parameter Count | Output Diagram Link |
| :--- | :--- | :--- | :--- |
| **`basic`** | Basic Original Broad-UNet (upstream) | $10,642,887$ | [basic_computational_graph.png](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/model_graphs/basic_computational_graph.png) |
| **`L0`** | Hardened Baseline Broad-UNet | $10,642,935$ | [L0_computational_graph.png](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/model_graphs/L0_computational_graph.png) |
| **`L11`** | KAN Skip Fusion Rearrangement | $10,254,583$ | [L11_computational_graph.png](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/model_graphs/L11_computational_graph.png) |
| **`L15`** | Champion Hybrid (KAN + LeakyReLU) | $10,254,583$ | [L15_computational_graph.png](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/model_graphs/L15_computational_graph.png) |

---

## 2. Slide-by-Slide Speaker Script (Oral Presentation Walkthrough)

### Slide 1: Macro Computational Graph of Champion L15
> **Visual Reference:** [Slide 1 Image](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/presentation_graphs/1_broad_unet_macro_architecture_16x9.png)

**What to Say to Your TA / Audience:**
> *"Today I am presenting our architectural search and optimization of the Broad-UNet spatiotemporal model for ERA5 meteorological nowcasting across the Iberian domain.*
>
> *Slide 1 displays the master computational graph of our Champion model: **L15 (Hybrid KAN + LeakyReLU)**. Broad-UNet accepts a 6-step temporal sequence of 10 atmospheric variables over a $32 \times 48$ spatial grid, with input shape `(Batch, 6, 32, 48, 10)`.*
>
> *The macro-topology is an asymmetrical 3D U-Net:*
> 1. ***Encoder (Left Column):*** *Four hierarchical downsampling stages ($16 \to 32 \to 64 \to 128$ filters). Notice that pooling operations use MaxPool3D with kernel `(1, 2, 2)`. This deliberately halves the spatial dimensions down to $4 \times 6$ while keeping the temporal depth $T=6$ intact.*
> 2. ***Temporal Compression & ASPP (Bottom Row):*** *At the bottleneck, the `CompressLags` node applies a valid Conv3D kernel `(6, 1, 1)` to crush antecedent lags from $T=6$ to $T=1$. It feeds into an Atrous Spatial Pyramid Pooling (ASPP) block with parallel dilation rates of $1, 6, 12, 18$ and global image pooling.*
> 3. ***Anchor Residual Bypass Highway (Top Amber Line):*** *This is our fundamental stability breakthrough. Rather than forcing the network to predict absolute surface temperatures (270 to 305 Kelvin) from scratch, we broadcast the latest observed frame $T_{t_0}$ directly across the top to the prediction head. The neural trunk only learns the perturbation $\Delta T$. This resolved the catastrophic $-52.8\%$ cold-collapse of the original upstream model.*
> 4. ***KAN Skip Fusion (Right Column in Purple):*** *At Decoder Stages 6 and 7, we replace rigid linear concatenation with non-linear Kolmogorov-Arnold Network (KAN) B-spline fusions. This eliminates 388,352 parameters while dynamically aligning deep semantics with shallow spatial textures."*

---

### Slide 2: Micro-Architectural Deep-Dive (The Four Core Engines)
> **Visual Reference:** [Slide 2 Image](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/presentation_graphs/2_internal_block_mechanisms_deepdive.png)

**What to Say to Your TA / Audience:**
> *"Moving to Slide 2, we dissect the internal mechanics of the four computational engines:*
>
> - ***Panel A (MultiScale 3D Convolutions):*** *Standard 3D convolutions require $3^3 = 27$ weights per kernel. We factorize this into three orthogonal 1D passes: $(1 \times 1 \times 3)$ longitudinal, $(1 \times 3 \times 1)$ latitudinal, and $(3 \times 1 \times 1)$ temporal. This reduces parameter complexity by $66.7\%$ ($9$ weights vs $27$) while allowing the network to disentangle horizontal wind advection from diurnal heating decays.*
> - ***Panel B (ASPP Pyramid Bottleneck):*** *Atmospheric systems operate across multiple spatial scales simultaneously. By evaluating rates 1, 6, 12, and 18, our bottleneck captures localized convective cells (Rate 6, ~200 km), regional frontal boundaries (Rate 12, ~600 km), and planetary Rossby waves (Rate 18, >1200 km) without losing grid resolution.*
> - ***Panel C (KAN B-Spline Skip Fusion vs Naive Concat):*** *In standard U-Net, concatenating shallow encoder features with deep decoder features causes cross-scale semantic aliasing. Our KAN skip block applies cubic B-spline basis expansion $\phi(x) = \text{SiLU}(x) \cdot W_{base} + \sum_k c_k B_k(x)$ where learnable coefficients reshape the activation response per channel edge. It cuts $388\text{k}$ parameters while providing continuous multi-scale feature alignment.*
> - ***Panel D (Anchor Residual Head Formulation):*** *Mathematically, $\hat{\mathbf{T}}_{t_0+\tau} = \mathbf{T}_{t_0} + \Delta \mathbf{T}_\tau$. By constraining the neural network to predict bounded perturbations ($|\Delta T| < 5\text{ K}$), the model maintains physical consistency across all forecast horizons."*

---

### Slide 3: Architectural Evolution Matrix
> **Visual Reference:** [Slide 3 Image](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/presentation_graphs/3_architectural_evolution_comparison.png)

**What to Say to Your TA / Audience:**
> *"Slide 3 outlines our methodical evolution from the published upstream paper to our final Champion model:*
>
> 1. ***Upstream Broad-UNet (Fernández & Mehrkanoon 2021):*** *Suffered an un-anchored temperature drift, collapsing into a severe $-3.8\text{ K}$ cold bias with a Validation RMSE of $7.340\text{ K}$ and negative test skill $(-0.840)$.*
> 2. ***L0 Hardened Baseline:*** *Adding the Anchor Residual Head instantly dropped RMSE to $3.520\text{ K}$ ($-52.0\%$ error reduction) and achieved positive test skill ($+0.284$).*
> 3. ***L11 Skip Fusion Model:*** *Introducing KAN Skip Fusion at Merge 6 & 7 cut parameter count from $10.64\text{M}$ to $10.25\text{M}$ and dropped RMSE to $3.427\text{ K}$ ($+2.63\%$ gain over L0).*
> 4. ***L15 Champion Hybrid:*** *Combining KAN Skip Fusion with a global LeakyReLU ($\alpha = 0.2$) backbone reached our study's best Validation RMSE of **$3.369\text{ K}$** ($+4.29\%$ gain over L0) and a held-out test skill of **$+0.421$** ($p < 0.001$)."*

---

### Slide 4: Architectural Search Landscape & Activations
> **Visual Reference:** [Slide 4 Image](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/results/presentation_graphs/4_layer_rearrangement_search_taxonomy.png)

**What to Say to Your TA / Audience:**
> *"Finally, Slide 4 maps our systematic search across 15 layer rearrangements and 5 global activations:*
>
> - ***Bottleneck Rearrangements (L2 to L9):*** *Adding MLPs or KANs before or after ASPP yielded diminishing returns ($< +0.80\%$). ASPP dilations already saturated $95\%+$ of receptive field capacity.*
> - ***Skip Connection Rearrangements (L10 vs L11):*** *Pre-skip linear projections (L10) produced negligible change ($+0.65\%$), while Post-Skip KAN Fusion (L11) drove our largest architectural breakthrough.*
> - ***Decoder Inversion (L13):*** *Evaluating convolution before UpSampling degraded RMSE to $3.684\text{ K}$ due to spatial checkerboard artifacts.*
> - ***Activation Benchmark:*** *Standard ReLU consistently suffered from dying neurons during cold winter regimes. LeakyReLU ($\alpha=0.2$) established uniform superiority across all 72 forecast hours."*

---

## 3. Pure Python CLI Reproducibility Commands

To re-run the presentation visual suite or inspect live computational graphs:

```bash
# 1. Generate all 4 presentation slides (16:9, 300 DPI)
python scripts/print_model_graph.py --presentation

# 2. Print the oral presentation walkthrough script to terminal
python scripts/print_model_graph.py --script

# 3. Generate high-resolution graph for Champion Model L15
python scripts/print_model_graph.py --model L15 --plot

# 4. Generate high-resolution graph for Basic Upstream Model
python scripts/print_model_graph.py --model basic --plot

# 5. Inspect and compare all models in Python
python scripts/print_model_graph.py --model all
```

---

## 4. Formal Academic Report Integration

All four presentation figures are formally integrated as **Chapter 8** in the official academic PDF report:
- **Report Path:** [Broad_UNet_Layer_Rearrangement_Architecture_Search_Report.pdf](file:///Users/agamghotra/Desktop/Final%20Datasets/Broad-UNet/reports/Broad_UNet_Layer_Rearrangement_Architecture_Search_Report.pdf)
- **Repository Commit:** `d2cca41` on `main` branch ([GitHub RM Repository](https://github.com/AgamGhotra1903/RM.git))

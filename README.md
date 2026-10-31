# Broad-UNet for Regional Weather Nowcasting (ERA5 Temperature Prediction)

Official implementation adaptation of **Broad-UNet** for multi-variable meteorological nowcasting using European Centre for Medium-Range Weather Forecasts (ECMWF) **ERA5 Reanalysis** data.

> **Scientific Reference:**  
> Jesús García Fernández & Siamak Mehrkanoon, *"Broad-UNet: Multi-scale feature learning for nowcasting tasks"*, **Neural Networks**, Volume 144, 2021, pp. 419–427. [DOI: 10.1016/j.neunet.2021.08.018](https://doi.org/10.1016/j.neunet.2021.08.018)

---

## 1. Overview & Adaptation Scope

This repository adapts the official Broad-UNet architecture to predict 2-metre surface air temperature (**$t2m$**, in Kelvin) 2 hours ahead ($L = 1$) based on 12 hours of antecedent history ($T = 6$ frames at 2-hour intervals) across 6 surface and atmospheric variables.

### Architecture Preservation
The core Broad-UNet architecture from García Fernández & Mehrkanoon (2021) is preserved intact:
- **Multi-Scale Factorized Convolutional Blocks**: Depthwise-separable 3D spatial convolutions with factorized branches ($1\times1$, $3\times3$, $5\times5$, $7\times7$).
- **Atrous Spatial Pyramid Pooling (ASPP)**: Multi-scale dilated convolutions at the bottleneck ($r \in \{1, 2, 4, 8\}$).
- **Temporal Skip Reductions**: Compresses 3D temporal sequence features into 2D skip connections to bridge encoder and decoder feature maps.
- **Divisibility Handling**: The regional domain ($28 \times 34$) is padded to $32 \times 48$ to accommodate the 4 successive $2\times2$ spatial pooling layers ($2^4 = 16$), and inversely unpadded back to the exact $28 \times 34$ physical grid before metric computation.

---

## 2. Dataset Specifications

- **Domain:** Northern India / Uttar Pradesh regional bounding box ($30.5^\circ\text{N} - 23.75^\circ\text{N}$, $76.75^\circ\text{E} - 85.0^\circ\text{E}$).
- **Spatial Grid:** $28$ latitude $\times 34$ longitude points at $0.25^\circ$ resolution ($952$ grid cells).
- **Temporal Coverage:** 15 continuous years (2011–2025, 65,748 total 2-hourly timestamps, 0 missing timestamps, 0 NaNs).
- **Target Variable:** 2-metre temperature (`t2m`, Kelvin).
- **Input Features (6 Variables):**
  1. `msl`: Mean sea level pressure ($\text{Pa}$)
  2. `sp`: Surface pressure ($\text{Pa}$)
  3. `d2m`: 2-metre dewpoint temperature ($\text{K}$)
  4. `t2m`: 2-metre air temperature ($\text{K}$)
  5. `slhf`: Surface latent heat flux ($\text{J m}^{-2}$)
  6. `ssr`: Surface net solar radiation ($\text{J m}^{-2}$)
- **Unified NetCDF Directory:** `~/ERA5_Radiation/final/` containing `2011_final.nc` through `2025_final.nc`.

---

## 3. Strict Non-Leakage Experimental Protocol

1. **Chronological Splitting:**
   - **Training Set:** 2011-01-01 00:00 to 2020-08-07 08:00 (42,022 timesteps, ~80% of pre-test period).
   - **Validation Set:** 2020-08-07 10:00 to 2022-12-31 22:00 (10,508 timesteps, ~20% of pre-test period).
   - **Held-out Test Set:** 2023-01-01 00:00 to 2025-12-31 22:00 (13,140 timesteps, 3 full unseen years).
2. **Zero Information Leakage:**
   - Min-max scaling statistics for all 6 input variables and target are computed **exclusively** on the training partition.
   - Validation and test sets are scaled using frozen training parameters.
   - Hard automated assertions in `train_era5.py` verify zero overlap and strict chronological ordering.

---

## 4. Setup and Quick Start

### Environment Setup
```bash
cd Broad-UNet

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (TensorFlow, netCDF4, xarray, matplotlib, reportlab, pandas)
pip install -r requirements.txt
```

### 1. Dataset Audit
Verify temporal continuity, 0 NaNs, and grid alignment across all 15 years (2011–2025):
```bash
python scripts/audit_dataset.py
```

### 2. Multi-Horizon Training
Train Broad-UNet across all 5 forecast horizons ($H \in \{1, 2, 4, 8, 12\}$):
```bash
python scripts/train.py --horizon all --model broad_unet --epochs 4 --steps 35 --batch-size 4
```

### 3. Held-Out Test Partition Evaluation
Evaluate on the untouched test partition (2024–2025) in physical Kelvin against Persistence and Climatology:
```bash
python scripts/evaluate.py --checkpoint outputs/multi_horizon/best_broad_unet_all.keras --samples 400
```

### 4. Publication Visualizations
Generate horizon comparison curves, 4-panel spatial error heatmaps, and residual scatter plots:
```bash
python scripts/visualize.py
```

### 5. Generate Academic PDF Report
Compile the formal multi-page research report:
```bash
python scripts/generate_pdf_report.py
```

---

## 5. Directory Structure

```
Broad-UNet/
├── configs/
│   └── config.py              # Master configuration (paths, grids, horizons, splits)
├── models/
│   ├── __init__.py            # Module exports
│   └── broad_unet.py          # Broad-UNet & U-Net baseline with Masked MSE / MAE losses
├── scripts/
│   ├── audit_dataset.py       # Comprehensive 15-year NetCDF integrity verification
│   ├── era5_data.py           # ERA5 data manager, spatial padding, Z-score scaling, sequence generator
│   ├── train.py               # Multi-horizon training pipeline with Masked MSE loss & checkpointing
│   ├── evaluate.py            # Independent test evaluation script (physical Kelvin metrics)
│   ├── visualize.py           # Publication figures (horizon curves, spatial maps, residuals)
│   └── generate_pdf_report.py # Formal academic PDF report builder
├── outputs/
│   ├── multi_horizon/         # Trained model checkpoints & loss curves
│   │   ├── best_broad_unet_all.keras
│   │   └── loss_curve_all.png
│   ├── figures/               # 300 DPI publication plots and spatial maps
│   │   ├── horizon_comparison_curves.png
│   │   ├── scatter_and_residuals.png
│   │   └── spatial_forecast_maps/
│   ├── test_predictions/      # Compressed sample prediction arrays
│   └── normalization_stats.json # Training set Z-score and min-max statistics
├── reports/
│   ├── dataset_audit.md       # Full 15-year dataset audit report
│   ├── metrics_summary.csv    # Quantitative evaluation table across all horizons
│   ├── metrics_summary.json   # JSON formatted metrics
│   └── Broad_UNet_ERA5_Weather_Forecasting_Report.pdf # Publication-grade PDF report
├── models.py                  # Official repo model definitions (Keras 3 compatible)
├── requirements.txt           # Python environment dependencies
└── README.md                  # Project documentation
```

---

## 6. Citation

```bibtex
@article{Broad-UNet,
  title={Broad-UNet: Multi-scale feature learning for nowcasting tasks},
  author={Garc{\'\i}a Fern{\'a}ndez, Jes{\'u}s and Mehrkanoon, Siamak},
  journal={Neural Networks},
  volume={144},
  pages={419--427},
  year={2021},
  publisher={Elsevier}
}
```


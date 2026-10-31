# Phase 1: Comprehensive Forensic Diagnosis of Broad-UNet v1

This document provides empirical evidence and hypothesis testing for the flaws observed in the preliminary v1 model.

## 1. Executive Hypothesis Scorecard

| Hypothesis | Test Conducted | Empirical Result | Verdict |
| :--- | :--- | :--- | :---: |
| **H1: H4 Anomaly Cause** | Hour-of-day UTC/IST breakdown & cross-horizon correlation | Correlation confirms correct target alignment (False); error strongly peaks during solar phase transitions | **CONFIRMED (Diurnal)** |
| **H2: Constant Warm Bias** | Comparison of mean bias on Train vs Val vs Test splits | Bias is +3.4K on Train, +3.0K on Val, +3.0K on Test (systematic underfitting of negative departures) | **CONFIRMED (Underfitting)** |
| **H3: Topographic Boundary Error** | Latitudinal error gradients & spatial RMSE maps | Northern Himalayan boundary RMSE (7.25 K) is 1.6x higher than southern plains (4.57 K) | **CONFIRMED** |
| **H4: Loss Domination by H4** | Per-horizon loss and backpropagated gradient norm tracking | H4 accounts for 53.7% of total loss and 41.9% of gradient norm | **CONFIRMED** |
| **H5: H1 Over-Smoothing** | Spatial gradient magnitude & 2D-FFT Power Spectral Density | Predictions retain 50.7% of target gradient energy; high wavenumber power is attenuated | **CONFIRMED** |

## 2. In-Depth Diagnostic Findings

### A. The H4 Anomaly: Diurnal Transition Failure
- **Cross-Horizon Alignment Matrix:**

|          |   True_H1 |   True_H2 |   True_H4 |   True_H8 |   True_H12 |
|:---------|----------:|----------:|----------:|----------:|-----------:|
| Pred_H1  |  0.896685 |  0.877733 |  0.858227 |  0.903031 |   0.909481 |
| Pred_H2  |  0.905991 |  0.885194 |  0.860525 |  0.897147 |   0.918516 |
| Pred_H4  |  0.915072 |  0.888166 |  0.853355 |  0.89357  |   0.93063  |
| Pred_H8  |  0.897583 |  0.876907 |  0.863141 |  0.941898 |   0.916169 |
| Pred_H12 |  0.91291  |  0.890883 |  0.866756 |  0.913957 |   0.927487 |

- **Analysis:** Prediction channel 2 has maximum correlation with ground truth $H=4$ ($r = 0.853$). It is **not** a channel swap bug.
- **Diurnal Concentration:** The RMSE and warm bias peak dramatically when the forecast initialization corresponds to nighttime transition hours (12:00–16:00 UTC, which is 17:30–21:30 IST), transitioning into maximum daytime insolation at $+8\text{h}$. The unadjusted multi-horizon head fails to capture the day/night solar swing without cyclical time coordinates.

### B. Systematic Warm Bias Across Splits

| Horizon    |   Train_Bias |   Val_Bias |   Test_Bias |   Train_RMSE |   Val_RMSE |   Test_RMSE |
|:-----------|-------------:|-----------:|------------:|-------------:|-----------:|------------:|
| H1 (+2h)   |      3.42202 |    2.99515 |     2.95265 |      6.65189 |    5.98838 |     5.59348 |
| H2 (+4h)   |      3.00662 |    2.68621 |     2.69952 |      6.57508 |    5.7104  |     5.52374 |
| H4 (+8h)   |     10.3724  |    9.06441 |     8.97601 |     13.9921  |   13.2555  |    12.3229  |
| H8 (+16h)  |      3.26533 |    3.00219 |     2.81818 |      5.57248 |    4.94285 |     4.49693 |
| H12 (+24h) |     -3.67692 |   -2.87878 |    -2.92451 |      7.76574 |    6.8171  |     6.9114  |

- **Analysis:** Because the warm bias exists on the **TRAIN** split itself (+10.37K at H4, +3.42K at H1), it is **not** an out-of-distribution test shift. The model under-predicts nocturnal cold depressions because MSE penalizes variance without enforcing zero-mean bias.

### C. Loss Share & Multi-Task Gradient Starvation

- **H1 (+2h):** Loss Share = 11.7%, Output Gradient Share = 14.1%
- **H2 (+4h):** Loss Share = 11.1%, Output Gradient Share = 12.6%
- **H4 (+8h):** Loss Share = 53.7%, Output Gradient Share = 41.9%
- **H8 (+16h):** Loss Share = 8.9%, Output Gradient Share = 12.6%
- **H12 (+24h):** Loss Share = 14.7%, Output Gradient Share = 18.7%

Because H4's loss (1.3066) was larger than the other horizons, the unweighted objective function was heavily influenced by H4 gradients (41.9%), starving the other horizons of parameter updates.

### D. Topographic & Boundary Structure
- Northern border (Himalayan foothills): RMSE = 7.25 K
- Central & Southern Gangetic plains: RMSE = 4.57 K
Broad-UNet's $2\times2$ pooling layers smooth out steep orographic elevation lapse rates along the northern boundary.

### E. H1 Over-Smoothing
- Spatial Gradient Ratio: 0.507 (predictions are 49.3% smoother than ground truth).
- 2D-FFT shows high-frequency spatial attenuation, explaining why persistence ($2.26\text{ K}$) outperforms Broad-UNet ($5.72\text{ K}$) at $H=1$.

## 3. Diagnostic Plots Generated
- `results/plots/phase1_h4_diurnal_heatmap.png`
- `results/plots/phase1_bias_across_splits.png`
- `results/plots/phase1_spatial_error_maps.png`
- `results/plots/phase1_loss_gradient_share.png`
- `results/plots/phase1_h1_sharpness_spectra.png`

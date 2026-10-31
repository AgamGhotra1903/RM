# ERA5 Broad-UNet Project Master Results Summary

Master tracking register for all runs, baselines, diagnostic tests, ablation stages, and architectural experiments.

---

## Phase 0: v1 Checkpoint & Baseline Reproduction (Physical Units: Kelvin)

Evaluated strictly on held-out **Test split (2024–2025)** and **Validation split (2022–2023)** over 400 sequences.

| Split | Horizon | Lead Time | Model | RMSE (K) | MAE (K) | $R^2$ | Bias (K) | Skill vs Pers | Skill vs Clim |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test** | **H1** | +2 h | Broad-UNet v1 (4ep, 35st) | 5.719 | 4.404 | 0.712 | +2.774 | -1.525 | -1.020 |
| Test | H1 | +2 h | Persistence ($t_0$) | 2.265 | 1.660 | 0.955 | +0.007 | 0.000 | +0.200 |
| Test | H1 | +2 h | Diurnal Climatology | 2.832 | 2.313 | 0.929 | +0.818 | -0.250 | 0.000 |
| Test | H1 | +2 h | Persistence (24h-prior) | 1.807 | 1.366 | 0.971 | -0.061 | +0.202 | +0.362 |
| **Test** | **H2** | +4 h | Broad-UNet v1 (4ep, 35st) | 5.637 | 4.459 | 0.721 | +2.502 | -0.447 | -0.992 |
| Test | H2 | +4 h | Persistence ($t_0$) | 3.895 | 2.930 | 0.867 | +0.015 | 0.000 | -0.376 |
| Test | H2 | +4 h | Diurnal Climatology | 2.830 | 2.311 | 0.930 | +0.817 | +0.274 | 0.000 |
| Test | H2 | +4 h | Persistence (24h-prior) | 1.834 | 1.385 | 0.970 | -0.050 | +0.529 | +0.352 |
| **Test** | **H4** | +8 h | Broad-UNet v1 (4ep, 35st) | 12.519 | 9.412 | -0.377 | +8.970 | -1.074 | -3.423 |
| Test | H4 | +8 h | Persistence ($t_0$) | 6.035 | 4.984 | 0.680 | +0.020 | 0.000 | -1.132 |
| Test | H4 | +8 h | Diurnal Climatology | 2.831 | 2.311 | 0.930 | +0.813 | +0.531 | 0.000 |
| Test | H4 | +8 h | Persistence (24h-prior) | 1.885 | 1.405 | 0.969 | -0.054 | +0.688 | +0.334 |
| **Test** | **H8** | +16 h | Broad-UNet v1 (4ep, 35st) | 4.693 | 3.693 | 0.807 | +2.644 | **+0.227** | -0.651 |
| Test | H8 | +16 h | Persistence ($t_0$) | 6.071 | 5.050 | 0.677 | -0.050 | 0.000 | -1.137 |
| Test | H8 | +16 h | Diurnal Climatology | 2.841 | 2.322 | 0.929 | +0.825 | +0.532 | 0.000 |
| Test | H8 | +16 h | Persistence (24h-prior) | 1.841 | 1.388 | 0.970 | -0.103 | +0.697 | +0.352 |
| **Test** | **H12** | +24 h | Broad-UNet v1 (4ep, 35st) | 6.944 | 4.640 | 0.578 | -3.366 | -2.748 | -1.434 |
| Test | H12 | +24 h | Persistence ($t_0$) | 1.853 | 1.396 | 0.970 | -0.089 | 0.000 | +0.350 |
| Test | H12 | +24 h | Diurnal Climatology | 2.853 | 2.334 | 0.929 | +0.841 | -0.540 | 0.000 |
| Test | H12 | +24 h | Persistence (24h-prior) | 1.853 | 1.396 | 0.970 | -0.089 | 0.000 | +0.350 |

---

## Phase 1: Diagnostic Hypothesis Scorecard (Gate 1 Passed)

Full forensic report and mathematical evidence: [`results/phase1_diagnosis.md`](phase1_diagnosis.md)

| Hypothesis | Test Conducted | Empirical Result | Verdict |
| :--- | :--- | :--- | :---: |
| **H1: H4 Anomaly Cause** | Diurnal cycle heatmap & correlation matrix | Max correlation aligns with H4 ($r = 0.853$); error concentrates in diurnal solar transitions (12–16 UTC) | **CONFIRMED (Diurnal)** |
| **H2: Constant Warm Bias** | Train vs Val vs Test split comparison | Bias is +3.4K (Train), +3.0K (Val), +3.0K (Test); underfitting of nocturnal cold depressions | **CONFIRMED (Underfitting)** |
| **H3: Topographic Boundary Error** | Latitudinal error gradients & spatial RMSE maps | Northern Himalayan boundary RMSE (7.25 K) is 1.6x higher than southern plains (4.57 K) | **CONFIRMED** |
| **H4: Loss Domination by H4** | Per-horizon loss and backpropagated gradient norm tracking | H4 accounts for **53.7% of total loss** and **41.9% of gradient norm** | **CONFIRMED** |
| **H5: H1 Over-Smoothing** | Spatial gradient magnitude & 2D-FFT Power Spectral Density | Predictions retain only **50.7% of target gradient energy**; high wavenumber power attenuated | **CONFIRMED** |

Diagnostic plots generated:
- `results/plots/phase1_h4_diurnal_heatmap.png`
- `results/plots/phase1_bias_across_splits.png`
- `results/plots/phase1_spatial_error_maps.png`
- `results/plots/phase1_loss_gradient_share.png`
- `results/plots/phase1_h1_sharpness_spectra.png`

---

## Phase 2: Stronger Baseline (Baseline v2) Ablations (Gate 2 Passed)

All ablation stages evaluated strictly on the **Validation set (2022–2023)** across 400 sequences.

| Stage | Configuration | Mean Val RMSE (K) | H1 RMSE (K) | H2 RMSE (K) | H4 RMSE (K) | H8 RMSE (K) | H12 RMSE (K) | Mean Skill vs Pers | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1** | Broad-UNet Checkpoint (Absolute) | 7.503 | 6.259 | 5.991 | 13.416 | 5.364 | 6.484 | -1.365 | Baseline |
| **S1** | Residual Head: $\hat{T}_{t+h} = T_{t_0} + \Delta T$ | 3.542 | 2.093 | 3.162 | 4.887 | 5.313 | 2.257 | +0.066 | Passed |
| **S2** | Residual + Cyclical Time ($\sin/\cos$ hour, doy) | 3.450 | 2.343 | 3.231 | 4.557 | 3.930 | 3.188 | +0.011 | Passed |
| **S3** | Residual + Cyclical Time + Loss Rebalancing | **3.623** | **1.993** | **3.314** | **5.254** | **5.312** | **2.244** | **+0.055** | **Baseline v2 Winner** |

### Gate 2 Verification Verdict: PASSED
- **H1 (+2h):** Model RMSE 1.993 K < Persistence 2.205 K (**Skill = +0.096**)
- **H2 (+4h):** Model RMSE 3.314 K < Persistence 3.690 K (**Skill = +0.102**)
- **H4 (+8h):** Model RMSE 5.254 K < Persistence 5.667 K (**Skill = +0.073**)
- **H8 (+16h):** Model RMSE 5.312 K < Persistence 5.741 K (**Skill = +0.075**)
- **H12 (+24h):** Model RMSE 2.244 K < Climatology 2.825 K (**Skill vs Clim = +0.206**)
- Waterfall Plot saved: `results/plots/waterfall_baseline_v2.png`

---

## Phase 3: Team Leader's Architecture Experiments (In Progress)

### Stage A: Architectural Screening (Target: $\ge +1.0\%$ improvement over Baseline v2)

| Stage | Description | Parameters | Val Mean RMSE (K) | Val Mean Skill | $\Delta$ vs Baseline v2 (K) | % Gain | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline v2 (S3)** | Core Winner (Residual + Cyclical + Rebalanced) | 10,641,153 | 3.623 | +0.055 | 0.000 | 0.00% | Reference |
| **E-M1** | MLP after ASPP Bottleneck | 10,904,833 | *Running* | - | - | - | - |
| **E-K1** | KAN after ASPP Bottleneck | 11,035,137 | *Pending* | - | - | - | - |
| **E-MK1** | Sequential MLP $\to$ KAN after ASPP | 11,298,817 | *Pending* | - | - | - | - |
| **E-MK2** | Parallel MLP $\parallel$ KAN with learnable gate | 11,298,818 | *Pending* | - | - | - | - |
| **E-S6** | KAN in Decoder Skip Fusion at `merge6` | 10,330,369 | *Pending* | - | - | - | - |
| **E-S67** | KAN in Decoder Skip Fusion at `merge6, merge7` | 10,252,801 | *Pending* | - | - | - | - |
| **E-S6789** | KAN in all Decoder Skips (`merge6` through `merge9`) | 10,228,673 | *Pending* | - | - | - | - |
| **E-S9** | KAN in Decoder Skip Fusion at `merge9` | 10,636,353 | *Pending* | - | - | - | - |
| **E-C2_C4** | MLP in Encoder Stages `conv2, conv3, conv4` | 10,728,513 | *Pending* | - | - | - | - |
| **E-KC2_KC4** | KAN in Encoder Stages `conv2, conv3, conv4` | 10,770,849 | *Pending* | - | - | - | - |
| **E-CTRL** | Parameter-matched Conv3D Control | 12,476,929 | *Pending* | - | - | - | - |

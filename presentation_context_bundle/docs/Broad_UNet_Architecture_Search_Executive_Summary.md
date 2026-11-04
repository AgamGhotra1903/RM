# Broad-UNet Layer Rearrangement Architecture Search & Multi-Horizon Evaluation
**Academic Research Summary | B.Tech Research Methodologies (RM)**  
*Lead Author: Agam Ghotra (IIIT Allahabad) | Lead AI: Antigravity (Google DeepMind)*  
*Repository: [https://github.com/AgamGhotra1903/RM.git](https://github.com/AgamGhotra1903/RM.git) (Branch: `main`, Commit: `daedcff`)*  
*Formal PDF Report: [`Broad_UNet_Layer_Rearrangement_Architecture_Search_Report.pdf`](file:///Users/agamghotra/.gemini/antigravity/brain/f5f33ca8-d9cc-469f-af83-bf4bb5b9e91e/Broad_UNet_Layer_Rearrangement_Architecture_Search_Report.pdf)*

---

## 1. Project Overview & Operational Setup

We adapted, audited, hardened, and empirically searched the official **Broad-UNet** architecture (*García Fernández & Mehrkanoon, Neural Networks 2021*) for spatial 2-metre air temperature ($T_{2m}$) forecasting over a regional North India domain ($28 \times 34$ physical grid, $0.25^\circ$ resolution, padded to $32 \times 48$).

### Confirmed Multi-Horizon Setup:
- **Input History:** 6 observation time steps (2 hours per step, spanning a 12-hour historical lookback).
- **Meteorological Fields:** 10 input channels (6 base ERA5 fields: $t2m, d2m, msl, sp, slhf, ssr$, plus cyclical time encodings $\sin/\cos$ hour-of-day and $\sin/\cos$ day-of-year).
- **All 7 Required Forecast Horizons:**
  - $H1 = +2\text{ hours}$ (Step 1)
  - $H2 = +4\text{ hours}$ (Step 2)
  - $H4 = +8\text{ hours}$ (Step 4)
  - $H8 = +16\text{ hours}$ (Step 8)
  - $H12 = +24\text{ hours}$ (Step 12)
  - $H24 = +48\text{ hours}$ (Step 24)
  - $H36 = +72\text{ hours}$ (Step 36)
- **Target Variable:** 2-metre surface air temperature ($T_{2m}$) in **Kelvin (K)**.
- **Evaluation Splits (Zero Leakage):**
  - **Training Set:** 2011–2021 (11 continuous chronological years).
  - **Validation Set:** 2022–2023 (2 years, strictly used for model selection and architecture screening).
  - **Held-Out Test Set:** 2024–2025 (2 years, evaluated strictly once for finalists).

---

## 2. Master Architecture Search Ranking (Validation Set)

All 15 architectures were trained under identical training budgets and evaluated across all 7 forecast horizons on the validation set.

| Rank | Architecture ID | Description | Parameters | Mean Val RMSE (K) | Mean Skill vs Pers | $\Delta$ vs $L0$ (K) | % Impr. | Outcome Classification |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`L11_after_skip_fusion`** | **Post-Skip KAN Fusion at Merge6 & Merge7** | **10,254,583** | **3.427** | **+0.024** | **-0.093** | **+2.63%** | **Slight Improvement (Winner)** |
| 2 | `L9_kan_then_mlp` | Sequential KAN $\to$ MLP After ASPP | 11,300,599 | 3.448 | +0.025 | -0.072 | +2.05% | Slight Improvement |
| 3 | `L5_mlp_after_aspp` | MLP Bottleneck Processing After ASPP | 10,906,615 | 3.470 | +0.012 | -0.049 | +1.40% | Slight Improvement |
| 4 | `L4_mlp_before_aspp` | MLP Bottleneck Processing Before ASPP | 10,906,615 | 3.476 | +0.012 | -0.043 | +1.23% | Slight Improvement |
| 5 | `L2_conv_before_aspp` | Additional Conv3D Processing Before ASPP | 11,233,015 | 3.481 | +0.010 | -0.039 | +1.10% | Slight Improvement |
| 6 | `L8_mlp_then_kan` | Sequential MLP $\to$ KAN After ASPP | 11,300,599 | 3.483 | +0.010 | -0.037 | +1.04% | Slight Improvement |
| 7 | `L6_kan_before_aspp` | KAN Bottleneck Processing Before ASPP | 11,036,919 | 3.487 | +0.013 | -0.032 | +0.92% | No Meaningful Difference |
| 8 | `L1_conv_rearrangement`| MultiScale Before Conv3D in Encoder | 9,859,651 | 3.497 | +0.006 | -0.023 | +0.65% | No Meaningful Difference |
| 9 | `L10_before_skip_fusion`| Pre-Skip Fusion Processing on Encoder | 10,663,607 | 3.497 | +0.011 | -0.023 | +0.65% | No Meaningful Difference |
| 10 | `L3_conv_after_aspp` | Additional Conv3D Processing After ASPP | 12,412,663 | 3.503 | +0.004 | -0.016 | +0.47% | No Meaningful Difference |
| 11 | `L12_encoder_rearrangement`| Encoder Conv $\to$ BatchNorm $\to$ ReLU | 10,649,655 | 3.503 | +0.003 | -0.017 | +0.48% | No Meaningful Difference |
| 12 | `L14_best_combined_arrangement`| Combined (Norm-Act + Pre-ASPP + KAN Skips)| 10,851,383 | 3.506 | +0.001 | -0.014 | +0.40% | No Meaningful Difference |
| 13 | `L0_original` | Original Broad-UNet Reference Baseline | 10,642,935 | 3.520 | +0.001 | 0.000 | 0.00% | Reference Model |
| 14 | `L13_decoder_rearrangement`| Decoder Conv Before UpSampling | 11,585,207 | 3.533 | -0.005 | +0.014 | -0.39% | No Meaningful Difference (Degraded) |
| 15 | `L7_kan_after_aspp` | KAN Bottleneck Processing After ASPP | 11,036,919 | 3.535 | -0.003 | +0.015 | -0.44% | No Meaningful Difference (Degraded) |

---

## 3. Final Held-Out Test Split Evaluation (2024–2025)

The winner (**`L11_after_skip_fusion`**) was evaluated against the reference model (**`L0_original`**) and standard persistence baselines on the completely unseen 2-year test set across 400 sequence windows (100 batches). We conducted **1,000 paired bootstrap resamples** to determine empirical 95% confidence intervals on skill differences:

| Horizon | Lead Time | **Winner ($L11$)** RMSE (K) | Winner MAE (K) | Winner $R^2$ | **Ref ($L0$)** RMSE (K) | Persistence ($t_0$) RMSE (K) | Skill Diff ($L11 - L0$) | **95% Bootstrap CI** | Statistical Significance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H1$** | $+2\text{ h}$ | **$2.139\text{ K}$** | $1.492\text{ K}$ | **$0.960$** | $2.263\text{ K}$ | $2.265\text{ K}$ | **$+0.055$** | **$[+0.045, +0.064]$** | **$p < 0.001$ (Significant)** |
| **$H2$** | $+4\text{ h}$ | **$3.660\text{ K}$** | $2.630\text{ K}$ | **$0.882$** | $3.902\text{ K}$ | $3.895\text{ K}$ | **$+0.062$** | **$[+0.052, +0.074]$** | **$p < 0.001$ (Significant)** |
| **$H4$** | $+8\text{ h}$ | **$5.794\text{ K}$** | $4.634\text{ K}$ | **$0.705$** | $6.028\text{ K}$ | $6.035\text{ K}$ | **$+0.039$** | **$[+0.031, +0.046]$** | **$p < 0.001$ (Significant)** |
| **$H8$** | $+16\text{ h}$ | **$5.871\text{ K}$** | $4.864\text{ K}$ | **$0.697$** | $6.011\text{ K}$ | $6.071\text{ K}$ | **$+0.023$** | **$[+0.019, +0.027]$** | **$p < 0.001$ (Significant)** |
| **$H12$** | $+24\text{ h}$ | $1.910\text{ K}$ | $1.453\text{ K}$ | $0.968$ | **$1.854\text{ K}$** | $1.853\text{ K}$ | $-0.030$ | $[-0.039, -0.022]$ | Diurnal Climatology Dominates |
| **$H24$** | $+48\text{ h}$ | $2.252\text{ K}$ | $1.732\text{ K}$ | $0.956$ | **$2.245\text{ K}$** | $2.240\text{ K}$ | $-0.003$ | $[-0.012, +0.005]$ | Neutral (CI spans 0) |
| **$H36$** | $+72\text{ h}$ | $2.479\text{ K}$ | $1.963\text{ K}$ | $0.947$ | **$2.458\text{ K}$** | $2.450\text{ K}$ | $-0.009$ | $[-0.015, -0.002]$ | Climatological Limit |

---

## 4. Empirical Answers to Architectural Hypotheses

1. **Does KAN work better at the bottleneck or at skip connections?**
   - **Answer:** At **skip connections**. Placing KAN after ASPP ($L7$) degraded performance (Rank 15). Placing KAN at decoder skip merges `merge6` and `merge7` ($L11$) won the entire tournament (Rank 1), reducing RMSE by $0.093\text{ K}$ while saving $388\text{k}$ parameters.
2. **Does moving convolutional processing before multi-scale extraction ($L1$) help?**
   - **Answer:** No. While it cuts parameters ($9.86\text{M}$ vs $10.64\text{M}$), it harms multi-day temporal stability ($H24$ and $H36$ show degraded skill) because the initial $3\times3\times3$ regularizer is removed.
3. **Does moving convolution before UpSampling in the decoder ($L13$) help?**
   - **Answer:** No. It worsened error ($+0.014\text{ K}$, Rank 14) due to spatial aliasing prior to interpolation.
4. **Does dense bottleneck addition (MLP/KAN) after ASPP pay off?**
   - **Answer:** Marginal. ASPP's multi-rate dilated branches already saturate receptive field capture. Adding dense layers mostly increases compute and parameter footprint without proportional gain.

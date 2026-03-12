# 🫀 CardioSense AI
### Early Cardiovascular Disease Risk Detection · Hack4Health 2026

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![AUC](https://img.shields.io/badge/AUC-0.8003-success?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![GPU](https://img.shields.io/badge/GPU-RTX%204050-76b900?style=flat-square&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A machine learning system for early detection of cardiovascular disease, trained on 70,000 de-identified patient records. Achieves AUC **0.8003** using RealMLP (v3 digit decomposition) with GPU acceleration, SHAP explainability, and an interactive Streamlit web app.

---

## 📊 Results

| Model | Version | AUC Score | Notes |
|-------|---------|-----------|-------|
| Logistic Regression | Baseline | 0.730 | Pre-project baseline |
| XGBoost | v3 | 0.7901 | Best XGBoost version |
| XGBoost | v1 | 0.7920 | |
| LightGBM | v4 | 0.7667 | |
| LightGBM | v1 | 0.7945 | Best LightGBM version |
| RealMLP | v2 | 0.7932 | |
| RealMLP | v1 | 0.7996 | |
| RealMLP | v5 | 0.7998 | Genetic programming features |
| RealMLP | v4 | 0.7999 | |
| **RealMLP** | **v3** | **0.8003** | **✅ Best — Selected & Deployed** |

**Improvement: +9.6% AUC over baseline**

---

## 🗂️ Project Structure

```
CardioSense-AI/
├── cardiac_failure.ipynb         # Full training pipeline (EDA → model → SHAP)
├── app.py                        # Streamlit demo app
├── cardiac_failure_model.pkl     # Trained RealMLP model
├── shap_values.npy               # Precomputed SHAP values (200 test patients)
├── X_test.csv                    # Test set for SHAP display
├── CardioSense_AI_Report.pdf     # Hackathon report (3 pages)
└── Data/
    └── cardiac_failure_processed.csv
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- **70,000** patient records → **65,846** after cleaning (shape: 65846 × 13)
- Memory optimized: `6.4 MB → 1.6 MB` (75% reduction) via dtype casting to uint8/float32
- Outlier removal: `ap_hi` [80–180], `ap_lo` [60–120], `weight` [45–105 kg], `height` [145–190 cm]
- Logical filter: rows where diastolic BP > systolic BP removed
- 24 duplicate rows dropped
- Skewness analysis: `ap_hi` positively skewed (1.11); all others roughly symmetric

### 2. Feature Engineering (5 Versions Tested)

| Version | Strategy | Key Features |
|---------|----------|-------------|
| **v1 (best)** | Baseline + BMI | `bmi = weight / (height/100)²` |
| v2 | Binned bands | `age_band`, `ap_hi_band`, `bmi_band` |
| v3 | Digit decomposition | `ap_hi_tens`, `ap_hi_units`, `height_tens` |
| v4 | Categorical | All features → string → category dtype |
| v5 | Genetic programming | gplearn SymbolicTransformer (10 evolved features) |

Additional domain interactions tested (`bp_diff`, `bp_ratio`, `bmi_age`, `chol_age`, `smoke_age`, `alco_age`, `active_bmi`) — degraded AUC by −0.00047, confirming base features were informationally sufficient.

### 3. Models & Training
- **5-fold Stratified Cross-Validation** (AUC-ROC metric)
- Out-of-fold (OOF) predictions to prevent data leakage
- XGBoost, LightGBM, **RealMLP** (pytabkit) all benchmarked across all 5 versions

### 4. RealMLP Architecture (Best Model)

```python
RealMLP_TD_Classifier(
    device='cuda',                              # NVIDIA RTX 4050 Laptop GPU
    random_state=42,
    n_epochs=100,
    batch_size=128,
    n_ens=8,                                    # 8-model ensemble
    val_metric_name='1-auc_ovr',
    use_early_stopping=True,
    early_stopping_additive_patience=20,
    early_stopping_multiplicative_patience=1,
    act='mish',                                 # Mish activation
    embedding_size=8,
    first_layer_lr_factor=0.5962121993798933,
    hidden_sizes='rectangular',
    hidden_width=384,
    lr=0.04,
    ls_eps=0.011498317194338772,
    ls_eps_sched='coslog4',
    max_one_hot_cat_size=18,
    n_hidden_layers=4,
    p_drop=0.07301419697186451,
    p_drop_sched='flat_cos',
    plr_hidden_1=16,
    plr_hidden_2=8,
    plr_lr_factor=0.1151437622270563,
    plr_sigma=2.3316811282666916,
    scale_lr_factor=2.244801835541429,
    sq_mom=0.9881659450444177,                  # 1.0 - 0.011834054955582318
    wd=0.02369230879235962,
)
```

> Parameters derived from a Kaggle top-3 cardiovascular disease solution. Optuna tuning (50 trials) confirmed these were already near-optimal — no trial exceeded AUC 0.800034.

### 5. Hyperparameter Tuning
- Optuna Bayesian optimisation — 50 trials
- Search space: `lr`, `hidden_width` [256–512], `n_hidden_layers` [2–6], `p_drop` [0–0.3], `batch_size` {64,128,256}, `wd`, `n_ens` {4,8,16}, `n_epochs` [50–200]
- Result: No improvement — best trial 0.8000435 vs baseline 0.800034

### 6. Explainability (SHAP)
- SHAP KernelExplainer on RealMLP (background=50 samples, 200 test patients, 100 perturbations each)
- Precomputed and saved as `shap_values.npy` for instant app display

| Feature | Impact | Interpretation |
|---------|--------|----------------|
| `ap_hi` | 🔴 Strong positive | High systolic BP = strongest CVD predictor |
| `age` | 🔴 Positive | Risk rises sharply after 45 |
| `cholesterol` | 🔴 Positive | Elevated levels linked to arterial plaque |
| `bmi` | 🔴 Positive | Obesity significantly elevates risk |
| `active` | 🟢 Negative | Physical activity is protective |
| `gluc`, `alco` | ⚪ Minimal | Near-zero SHAP — model largely ignores these |

---

## 🚀 Run the App

```bash
pip install streamlit pandas numpy scikit-learn shap matplotlib pytabkit
streamlit run app.py
```

**Required files in same folder:**
```
cardiac_failure_model.pkl   shap_values.npy   X_test.csv
```

---

## 📦 Full Install

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost \
            pytabkit shap optuna streamlit matplotlib seaborn \
            category_encoders gplearn torch
```

---

## 📋 Dataset

- **Source**: Cardiovascular Disease Dataset — 70,000 patients, 12 features + 1 target
- **Final training shape**: 65,846 × 13 (after cleaning)
- **Features**: age (normalized), gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi
- **Target**: `cardio` — binary (0 = No CVD, 1 = CVD present), ~50/50 class balance

---

## 🏆 Hackathon

Built for **[Byte 2 Beat · Hack4Health 2026](https://hack4health.devpost.com)**

**Judging criteria addressed:**
- ✅ **Creativity** — 5 feature engineering strategies + SHAP + Genetic programming + interactive app
- ✅ **Technical Complexity** — RealMLP neural net + GPU + 8 ensembles + Optuna + KernelExplainer
- ✅ **Practicality** — Deployable Streamlit app, real clinical workflow, medically coherent SHAP findings
- ✅ **Presentation** — Full 3-page PDF report, SHAP plots, risk gauge UI

---

## ⚠️ Disclaimer

CardioSense AI is an educational research tool built for Hack4Health 2026. It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## ⚠️ Known Limitations & Fixes

### Smoking & Alcohol Risk Adjustment
The core RealMLP model shows near-zero SHAP importance for `smoke` and `alco` features. This is a **known dataset limitation** — only 8.7% of patients reported smoking (self-reporting bias), so the model couldn't learn a strong signal from these features.

**Fix applied in app:** A clinically-informed manual adjustment is added on top of the model output:
- Smoker = Yes → +5% risk
- Alcohol = Yes → +3% risk  
- Both = Yes → +8% risk combined

This keeps the core model intact while ensuring medically correct behavior in the app.
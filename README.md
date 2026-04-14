# 🎯 AI/ML-Based Predictive Model for Ablation Zone Estimation in Tumor Treatment

> A machine learning system that predicts microwave ablation zone dimensions from treatment parameters — enabling real-time clinical decision support in milliseconds vs. hours for traditional physics-based simulations.

**Bachelor of Technology (Honours) Project** · IIIT Kottayam · April 2026  
**Author:** Rishi Sharma (Roll No. 2023BCD0023)  
**Supervisor:** Dr. Debarati Ganguly

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Models](#-models)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Limitations & Future Work](#-limitations--future-work)
- [References](#-references)

---

## 🔬 Overview

Microwave Ablation (MWA) is a minimally invasive alternative to surgery for treating solid tumors in the liver, lung, kidney, and adrenal glands. The procedure works by delivering electromagnetic energy through an interstitial antenna to heat and destroy cancerous tissue — creating a region of necrosis called the **ablation zone**.

Accurately predicting the **diameter** and **length** of this zone before treatment is critical: too small, and the tumor survives; too large, and healthy tissue is unnecessarily destroyed. Current methods are either:

- ⏳ **Physics-based simulations (FEM/FDTD)** — accurate but take *hours* per case
- 📋 **Manufacturer lookup tables** — fast but don't generalize across antenna designs
- 📐 **Empirical formulas** — simple but poor generalization

This project builds an **ML-based predictive model** that generalizes across multiple antenna types, power levels, and treatment durations — delivering predictions in **milliseconds**.

---

## 🎯 Problem Statement

> Develop an AI/ML-based predictive model that accurately estimates the ablation zone produced during tumor treatment using microwave ablation parameters, enabling faster and more accurate treatment planning while minimizing damage to surrounding healthy tissues.

---

## 📊 Dataset

A comprehensive dataset of **326 samples** was curated from **30+ published research papers (2004–2025)**, covering both experimental and simulation data.

| Source Type | Samples | Papers |
|-------------|---------|--------|
| Experimental (ex vivo / in vivo) | 222 | ~25 papers |
| Simulation (FEM / computational) | 104 | ~15 papers |
| **Total** | **326** | **30+** |

**Coverage:** 15 antenna types · Power range: 20–160 W · Duration: 1–40 minutes · Frequencies: 915 MHz and 2.45 GHz

### Target Variables

| Target | Availability | Mean |
|--------|-------------|------|
| Effective Diameter (mm) | 91.7% | ~28 mm |
| Length (mm) | 62.0% | ~38 mm |

---

## ⚙️ Feature Engineering

Raw text fields from heterogeneous papers were parsed using regex extraction into structured numerical features:

| Feature | Formula / Description |
|---------|----------------------|
| `power_watts` | Input power (W) |
| `time_minutes` | Treatment duration (min) |
| `energy_joules` | `power × time × 60` — total energy delivered |
| `power_time_product` | `power × time` |
| `log_power` | `ln(1 + power)` — accounts for saturation |
| `log_time` | `ln(1 + time)` — accounts for saturation |
| `log_energy` | `ln(1 + energy)` |
| `sqrt_time` | `√time` — models thermal diffusion growth |
| `is_simulated` | Binary flag: `0` = experimental, `1` = simulated |
| `antenna_encoded` | Label-encoded antenna category (0–14) |

**Key insight from correlation analysis:** `energy_joules` (power × time) is a significantly stronger predictor (r = +0.591) than either power (r = +0.430) or time (r = +0.343) alone.

---

## 🤖 Models

Six regression models were trained and evaluated using **10-fold cross-validation** with **GridSearchCV** hyperparameter optimization:

| Model | Type |
|-------|------|
| Ridge Regression | Linear baseline with L2 regularization |
| K-Nearest Neighbors (KNN) | Instance-based, non-parametric |
| Support Vector Regression (SVR) | RBF kernel, non-linear |
| Random Forest | Ensemble of decision trees (bagging) |
| Gradient Boosting (GBR) | Sequential ensemble (boosting) |
| Multi-Layer Perceptron (MLP) | Feedforward neural network |

---

## 📈 Results

### Effective Diameter Prediction

| Model | CV R² | Test R² | MAE (mm) | RMSE (mm) |
|-------|-------|---------|----------|-----------|
| **Random Forest ⭐** | 0.5293 | **0.6952** | **6.02** | **7.70** |
| Gradient Boosting | 0.5047 | 0.6789 | 6.23 | 7.90 |
| Ridge Regression | 0.3542 | 0.5313 | 7.90 | 9.55 |
| KNN | 0.4804 | 0.5255 | 7.74 | 9.60 |
| SVR | 0.4262 | 0.5138 | 6.91 | 9.72 |
| MLP | 0.4542 | -0.0102 | 11.59 | 14.01 |

### Length Prediction

| Model | CV R² | Test R² | MAE (mm) | RMSE (mm) |
|-------|-------|---------|----------|-----------|
| **Gradient Boosting ⭐** | 0.6690 | **0.5118** | **10.93** | **13.76** |
| Random Forest | 0.6420 | 0.5032 | 10.67 | 13.88 |
| MLP | 0.5119 | 0.4875 | 9.97 | 14.10 |
| SVR | 0.6180 | 0.3445 | 10.74 | 15.95 |
| KNN | 0.6008 | 0.2954 | 11.33 | 16.53 |
| Ridge Regression | 0.3994 | 0.2580 | 12.50 | 16.97 |

### Feature Importance (Random Forest — Diameter)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `log_energy` | 24.5% |
| 2 | `energy_joules` | 21.9% |
| 3 | `power_time_product` | 20.3% |
| 4 | `antenna_encoded` | 12.1% |
| 5 | `is_simulated` | 6.1% |

> **Energy-related features collectively account for 66.7% of prediction power**, confirming total energy delivered is the dominant factor in ablation zone size.

---

## 📁 Project Structure

```
ablation-zone-prediction/
├── data/
│   ├── raw/                    # Raw data from published papers
│   └── processed/              # Cleaned and feature-engineered dataset
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_preprocessing.ipynb  # Feature engineering & cleaning
│   ├── 03_modeling.ipynb       # Model training & evaluation
│   └── 04_analysis.ipynb       # Results analysis & plots
├── src/
│   ├── preprocessing.py        # Data parsing and feature engineering
│   ├── models.py               # Model training and evaluation
│   └── predict.py              # Interactive prediction system
├── models/
│   ├── random_forest_diameter.pkl
│   └── gradient_boosting_length.pkl
├── reports/
│   └── honours_report.pdf
├── requirements.txt
└── README.md
```

---

## 🚀 Usage

### Installation

```bash
git clone https://github.com/yourusername/ablation-zone-prediction.git
cd ablation-zone-prediction
pip install -r requirements.txt
```

### Requirements

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

### Interactive Prediction

```python
from src.predict import AblationPredictor

predictor = AblationPredictor()

result = predictor.predict(
    power_watts=50,
    time_minutes=5,
    antenna_type="Dual Slot",
    is_simulated=False
)

print(f"Predicted Diameter: {result['diameter_mm']:.1f} mm")
print(f"Predicted Length:   {result['length_mm']:.1f} mm")
print(f"Estimated Volume:   {result['volume_mm3']:.0f} mm³")
print(f"Sphericity Index:   {result['sphericity_index']:.2f}")
```

### Sample Output

```
Predicted Diameter: 27.3 mm
Predicted Length:   38.1 mm
Estimated Volume:   16,538 mm³
Sphericity Index:   0.72
```

### Sample Predictions

| Power (W) | Time (min) | Antenna | Pred. Diameter | Pred. Length |
|-----------|------------|---------|----------------|--------------|
| 50 | 5 | Dual Slot | ~27 mm | ~38 mm |
| 50 | 10 | Dual Slot | ~33 mm | ~47 mm |
| 100 | 5 | Monopole | ~35 mm | ~50 mm |
| 100 | 10 | Monopole | ~42 mm | ~58 mm |
| 20 | 5 | Dipole | ~20 mm | ~30 mm |

---

## ⚠️ Limitations & Future Work

### Current Limitations

- **Dataset size:** 326 samples is small for ML; more data would improve generalization
- **Missing features:** Antenna-specific parameters (slot dimensions, frequency, length) not consistently available
- **Tissue variability:** Different tissue types (liver, lung, kidney, bovine) combined without tissue-type encoding
- **Temperature data:** Only 29.1% of entries had temperature data, limiting its use as a feature

### Future Work

- [ ] Expand dataset with more recent publications and additional ablation modalities
- [ ] Include tissue-specific features (perfusion rate, dielectric properties, tissue type)
- [ ] Explore physics-informed neural networks with larger datasets
- [ ] 3D ablation zone shape prediction using image-based ML
- [ ] Web-based clinical tool with a user-friendly interface
- [ ] Multi-output models predicting diameter, length, and volume simultaneously

---

## 📚 References

Key references from 30+ papers spanning 2004–2025. Selected highlights:

1. Dual-slot antennas for microwave tissue heating (2011) — *Medical Physics*
2. A Minimally Invasive Antenna for Microwave Ablation Therapies (2011) — *IEEE TBME*
3. Triple Coaxial-Half-Slot Antenna With Deep Learning-Based Temperature Prediction (2021) — *IEEE Access*
4. A Minimally Invasive Microwave Ablation Antenna With Highly Localized Ablation Zone (2022) — *IEEE TBME*
5. Design of Microwave Ablation Antenna for Flexible Omnidirectional/Directional Ablation Zone Control (2025) — *IEEE TBME*

Full reference list available in the [project report](reports/honours_report.pdf).

---

## 📄 License

This project was developed as an academic honours project at IIIT Kottayam. Please cite appropriately if you use this work.

---

<div align="center">
  <sub>Department of Computer Science and Engineering (AI & DS) · IIIT Kottayam · 2026</sub>
</div>

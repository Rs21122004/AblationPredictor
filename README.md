# Ablation Zone Prediction for Cancer Treatment using Machine Learning

## 🧠 Overview
Accurate prediction of the ablation zone is critical in **microwave ablation (MWA)** for tumor treatment. Traditional approaches rely on computationally expensive simulations or limited lookup tables, making real-time clinical decision-making difficult.

This project presents a **machine learning-based predictive system** that estimates ablation zone dimensions (diameter and length) directly from treatment parameters, enabling **instant predictions (milliseconds)** instead of hours required by physics-based simulations.

---

## 🎯 Problem Statement
Predict the **ablation zone dimensions** (diameter, length, volume) from treatment parameters such as:
- Input power
- Treatment duration
- Antenna type

👉 Without relying on:
- FEM simulations  
- Manufacturer-specific lookup tables  

---

## 📊 Dataset
- 📄 **326 samples** curated from **30+ research papers (2004–2025)** :contentReference[oaicite:1]{index=1}  
- 🔬 Includes:
  - Experimental (ex vivo / in vivo)
  - Simulated (FEM / computational)
- ⚙️ Covers:
  - **15+ antenna types**
  - Wide range of power & time values

### Key Challenges
- Unstructured data → extracted using **regex parsing**
- Missing values (temperature, etc.)
- Heterogeneous sources across studies

---

## ⚙️ Feature Engineering
Domain-driven features based on **thermal physics principles**:

- `energy_joules = power × time × 60` (most important feature)
- `power_time_product`
- Log transforms (`log_power`, `log_energy`)
- `sqrt_time` (models thermal diffusion)
- Encoded antenna type
- Simulation flag (`is_simulated`)

👉 Energy-related features contributed **~66.7% of model importance** :contentReference[oaicite:2]{index=2}  

---

## 🧠 Machine Learning Models
Trained and benchmarked **6 regression models**:

- Ridge Regression
- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)
- Random Forest ⭐
- Gradient Boosting ⭐
- Multi-Layer Perceptron (MLP)

### Training Strategy
- ✅ 10-fold cross-validation  
- ✅ GridSearchCV for hyperparameter tuning  
- ✅ Separate models for:
  - Diameter prediction
  - Length prediction  

---

## 📈 Results

### Diameter Prediction
- **Best Model:** Random Forest  
- **R² Score:** 0.695  
- **MAE:** 6.02 mm  

### Length Prediction
- **Best Model:** Gradient Boosting  
- **R² Score:** 0.512  
- **MAE:** 10.93 mm  

👉 Achieves **clinically meaningful accuracy** :contentReference[oaicite:3]{index=3}  

---

## 🔍 Key Insights

- ⚡ **Energy (power × time)** is the dominant predictor  
- 🌳 Tree-based models outperform neural networks on tabular data  
- 📉 MLP overfits due to small dataset size  
- 🔬 Antenna type significantly impacts ablation geometry  

---

## 🛠️ System Design

A **real-time prediction system** was built:

### Input:
- Power (W)
- Time (minutes)
- Antenna type

### Output:
- Predicted diameter (mm)
- Predicted length (mm)
- Estimated volume
- Sphericity index

👉 Predictions generated in **milliseconds vs hours for simulations** :contentReference[oaicite:4]{index=4}  

---

## 📂 Project Structure

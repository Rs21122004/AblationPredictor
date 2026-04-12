# Tumor Ablation Zone Prediction System

An end-to-end machine learning system and full-stack web application designed for predicting tumor ablation zone dimensions (effective diameter and length) based on experimental and simulated data. This project was developed as part of an Honours Project, providing both a robust predictive model and an intuitive dashboard interface for clinicians and researchers.

## Overview

The project is structured into three main components:
1. **Machine Learning Pipeline**: Advanced data preprocessing, feature engineering, and model training scripts to process ablation data. Includes experimental and simulated datasets to construct predictive models for the expected ablation dimensions based on applied power (Watts) and time (Minutes).
2. **Backend API (FastAPI)**: A robust REST API serving the pre-trained machine learning models. Built to deliver quick, scalable predictions on the fly.
3. **Frontend Dashboard (React)**: An interactive, aesthetically pleasing frontend application where users can input ablation parameters and instantly view predicted outcomes in an intuitive dashboard.

## Key Features

- **Predictive Modeling**: Provides accurate estimates for both `effective_diameter_mm` and `length_mm` of ablation zones.
- **Model Evaluation & Results**: Comparative evaluation of multiple regression models to find the most accurate algorithm.
- **Data Preprocessing**: Comprehensive handling of missing data, outlier removal (e.g., extreme power values), antenna label encoding, and scaling.
- **Full-Stack Application**: A production-grade web application (React frontend + FastAPI backend) allowing hands-on interaction with the models.
- **Academic Research Foundation**: Includes the complete LaTeX source for the Honours Project report, reflecting the rigorous methodology applied.

## Directory Structure

```
├── ablation-prediction-app/
│   ├── backend/             # FastAPI server and trained models endpoints
│   └── frontend/            # React-based interactive web dashboard
├── data_preprocessing.py    # Cleans and scales raw data, handles train/test splits
├── eda_analysis.py          # Exploratory Data Analysis & visual generation
├── feature_engineering.py   # Feature extraction and transformation
├── model_training.py        # Trains the ML models based on the processed datasets
├── model_results.py         # Evaluates model performance and metrics
├── predict_demo.py          # Command-line demonstration of predictions
├── report.tex (and others)  # LaTeX files for the Honours Project Report
└── README.md                # This documentation file
```

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js (for the frontend app)

### Running the Machine Learning Pipeline

If you want to re-train the models or run the data scripts:
```bash
# 1. Feature Engineering
python feature_engineering.py

# 2. Preprocess the Data
python data_preprocessing.py

# 3. Train the Models
python model_training.py

# 4. View Model Results
python model_results.py
```

### Running the Web Application

**1. Start the Backend API**
```bash
cd ablation-prediction-app/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**2. Start the Frontend Dashboard**
```bash
cd ablation-prediction-app/frontend
npm install
npm run dev
```

## Data Assets
The project utilizes historical datasets containing applied settings, antenna types, and resulting ablation dimensions:
- `Ablation Zone Model - Experimental data.csv`
- `Ablation Zone Model2 - Simulated data.csv`

*Note: Cleaned and engineered `.csv` and `.pkl` outputs are generated automatically by running the processing scripts.*

## Academic Context
This repository contains the codebase and final report contents for an Honours Review submission. The accompanying presentation, documentation, and thesis files detail the rationale, theoretical background, and systematic results obtained throughout the project's duration.

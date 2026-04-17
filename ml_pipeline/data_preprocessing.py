"""
Phase 2: Data Preprocessing
Honours Project - Ablation Zone Prediction

Handles missing values, removes outliers, scales features, and creates
train-test splits ready for model training.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import pickle
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load feature-engineered data ───
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'combined_data_engineered.csv'))
print(f"Loaded: {len(df)} samples\n")

# ═══════════════════════════════════════════════
# 1. REMOVE OUTLIERS
# ═══════════════════════════════════════════════
# Remove pulsed microwave entries (25 kW) — different treatment modality
print("Step 1: Removing outliers...")
df = df[df['power_watts'] <= 200].copy()
print(f"  After removing extreme power outliers: {len(df)} samples")

# ═══════════════════════════════════════════════
# 2. SELECT FEATURES AND TARGETS
# ═══════════════════════════════════════════════
print("\nStep 2: Selecting features and targets...")

# Primary input features
input_features = [
    'power_watts',
    'time_minutes',
    'energy_joules',
    'power_time_product',
    'log_power',
    'log_time',
    'log_energy',
    'sqrt_time',
    'is_simulated',
]

# We want to predict effective_diameter_mm (available for 91.7% of data)
# We'll also build a model for length_mm where available

# ─── Dataset A: Predict Effective Diameter ───
print("\n--- Dataset A: Effective Diameter Prediction ---")
mask_diam = df['effective_diameter_mm'].notna() & df['power_watts'].notna() & df['time_minutes'].notna()
df_diam = df[mask_diam].copy()
print(f"  Samples with effective_diameter_mm: {len(df_diam)}")

# ─── Dataset B: Predict Length ───
print("\n--- Dataset B: Length Prediction ---")
mask_len = df['length_mm'].notna() & df['power_watts'].notna() & df['time_minutes'].notna()
df_len = df[mask_len].copy()
print(f"  Samples with length_mm: {len(df_len)}")

# ─── Dataset C: Predict Both (samples that have both) ───
mask_both = mask_diam & mask_len
df_both = df[mask_both].copy()
print(f"\n  Samples with BOTH targets: {len(df_both)}")

# ═══════════════════════════════════════════════
# 3. ENCODE ANTENNA CATEGORY
# ═══════════════════════════════════════════════
print("\nStep 3: Encoding antenna categories...")

le = LabelEncoder()
df['antenna_encoded'] = le.fit_transform(df['antenna_category'])
df_diam['antenna_encoded'] = le.transform(df_diam['antenna_category'])
df_len['antenna_encoded'] = le.transform(df_len['antenna_category'])
df_both['antenna_encoded'] = le.transform(df_both['antenna_category'])

# Add antenna encoding to input features
all_features = input_features + ['antenna_encoded']
print(f"  Antenna categories: {list(le.classes_)}")
print(f"  Total input features: {len(all_features)}")

# ═══════════════════════════════════════════════
# 4. PREPARE FEATURE MATRICES
# ═══════════════════════════════════════════════
print("\nStep 4: Preparing feature matrices...")

# Dataset A: Diameter prediction
X_diam = df_diam[all_features].values
y_diam = df_diam['effective_diameter_mm'].values

# Dataset B: Length prediction
X_len = df_len[all_features].values
y_len = df_len['length_mm'].values

# Dataset C: Multi-output (both)
X_both = df_both[all_features].values
y_both = df_both[['effective_diameter_mm', 'length_mm']].values

# ═══════════════════════════════════════════════
# 5. TRAIN-TEST SPLIT (80/20)
# ═══════════════════════════════════════════════
print("\nStep 5: Train-test split (80/20)...")

# Diameter split
X_diam_train, X_diam_test, y_diam_train, y_diam_test = train_test_split(
    X_diam, y_diam, test_size=0.2, random_state=42
)
print(f"  Diameter — Train: {len(X_diam_train)}, Test: {len(X_diam_test)}")

# Length split
X_len_train, X_len_test, y_len_train, y_len_test = train_test_split(
    X_len, y_len, test_size=0.2, random_state=42
)
print(f"  Length   — Train: {len(X_len_train)}, Test: {len(X_len_test)}")

# Multi-output split
X_both_train, X_both_test, y_both_train, y_both_test = train_test_split(
    X_both, y_both, test_size=0.2, random_state=42
)
print(f"  Both     — Train: {len(X_both_train)}, Test: {len(X_both_test)}")

# ═══════════════════════════════════════════════
# 6. FEATURE SCALING
# ═══════════════════════════════════════════════
print("\nStep 6: Feature scaling (StandardScaler)...")

# Fit scaler on training data only
scaler_diam = StandardScaler()
X_diam_train_scaled = scaler_diam.fit_transform(X_diam_train)
X_diam_test_scaled = scaler_diam.transform(X_diam_test)

scaler_len = StandardScaler()
X_len_train_scaled = scaler_len.fit_transform(X_len_train)
X_len_test_scaled = scaler_len.transform(X_len_test)

scaler_both = StandardScaler()
X_both_train_scaled = scaler_both.fit_transform(X_both_train)
X_both_test_scaled = scaler_both.transform(X_both_test)

print("  Scaling complete (fitted on training data only)")

# ═══════════════════════════════════════════════
# 7. SAVE PREPROCESSED DATA
# ═══════════════════════════════════════════════
print("\nStep 7: Saving preprocessed data...")

preprocessed = {
    'feature_names': all_features,
    'label_encoder': le,
    # Diameter dataset
    'X_diam_train': X_diam_train, 'X_diam_test': X_diam_test,
    'y_diam_train': y_diam_train, 'y_diam_test': y_diam_test,
    'X_diam_train_scaled': X_diam_train_scaled, 'X_diam_test_scaled': X_diam_test_scaled,
    'scaler_diam': scaler_diam,
    # Length dataset
    'X_len_train': X_len_train, 'X_len_test': X_len_test,
    'y_len_train': y_len_train, 'y_len_test': y_len_test,
    'X_len_train_scaled': X_len_train_scaled, 'X_len_test_scaled': X_len_test_scaled,
    'scaler_len': scaler_len,
    # Multi-output dataset
    'X_both_train': X_both_train, 'X_both_test': X_both_test,
    'y_both_train': y_both_train, 'y_both_test': y_both_test,
    'X_both_train_scaled': X_both_train_scaled, 'X_both_test_scaled': X_both_test_scaled,
    'scaler_both': scaler_both,
}

output_path = os.path.join(SCRIPT_DIR, 'preprocessed_data.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(preprocessed, f)

print(f"  Saved to: {output_path}")

# ═══════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("PREPROCESSING COMPLETE")
print(f"{'='*60}")
print(f"\nFeatures ({len(all_features)}): {all_features}")
print(f"\nDatasets ready:")
print(f"  A) Diameter prediction:  {len(X_diam_train)} train + {len(X_diam_test)} test")
print(f"  B) Length prediction:    {len(X_len_train)} train + {len(X_len_test)} test")
print(f"  C) Multi-output (both):  {len(X_both_train)} train + {len(X_both_test)} test")
print(f"\nAll data saved to preprocessed_data.pkl")
print("Done!")

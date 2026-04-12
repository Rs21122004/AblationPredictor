"""
Phase 3: Model Training & Comparison
Honours Project - Ablation Zone Prediction

Trains 6 ML models with 10-fold cross-validation and hyperparameter tuning.
Predicts: effective_diameter_mm and length_mm
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore
from sklearn.svm import SVR  # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore

from sklearn.model_selection import cross_val_score, GridSearchCV  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer  # type: ignore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load preprocessed data ───
with open(os.path.join(SCRIPT_DIR, 'preprocessed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

print("="*70)
print("PHASE 3: MODEL TRAINING & COMPARISON")
print("="*70)
print(f"\nFeatures: {data['feature_names']}")


def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE metric."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       X_train_scaled, X_test_scaled, target_name):
    """Train all 6 models on a given target and return results."""

    print(f"\n{'─'*70}")
    print(f"TARGET: {target_name}")
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"{'─'*70}")

    # Define models with hyperparameter grids
    models = {
        'Ridge Regression': {
            'model': Ridge(),
            'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
            'use_scaled': True,
        },
        'KNN': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'use_scaled': True,
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf'],
                'epsilon': [0.01, 0.1, 0.5]
            },
            'use_scaled': True,
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'use_scaled': False,  # Trees don't need scaling
        },
        'XGBoost (GBR)': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0],
            },
            'use_scaled': False,
        },
        'MLP Neural Network': {
            'model': MLPRegressor(max_iter=1000, random_state=42, early_stopping=True),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.001, 0.01, 0.1],
                'learning_rate': ['adaptive'],
            },
            'use_scaled': True,
        },
    }

    results = {}

    for name, config in models.items():
        print(f"\n  ▶ Training: {name}...")

        X_tr = X_train_scaled if config['use_scaled'] else X_train
        X_te = X_test_scaled if config['use_scaled'] else X_test

        # GridSearchCV with 10-fold cross-validation
        grid = GridSearchCV(
            config['model'],
            config['params'],
            cv=10,
            scoring='r2',
            n_jobs=-1,
            refit=True,
            error_score='raise'
        )
        grid.fit(X_tr, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        cv_r2 = grid.best_score_

        # Predict on test set
        y_pred = best_model.predict(X_te)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        results[name] = {
            'model': best_model,
            'best_params': best_params,
            'cv_r2': cv_r2,
            'test_r2': r2,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_mape': mape,
            'y_pred': y_pred,
            'y_test': y_test,
            'use_scaled': config['use_scaled'],
        }

        print(f"    Best params: {best_params}")
        print(f"    CV R² = {cv_r2:.4f}  |  Test R² = {r2:.4f}")
        print(f"    MAE = {mae:.2f} mm  |  RMSE = {rmse:.2f} mm  |  MAPE = {mape:.1f}%")

    return results


# ═══════════════════════════════════════════════
# TRAIN FOR TARGET: EFFECTIVE DIAMETER
# ═══════════════════════════════════════════════
print("\n\n" + "="*70)
print("TRAINING MODELS FOR EFFECTIVE DIAMETER (mm)")
print("="*70)

results_diam = train_and_evaluate(
    data['X_diam_train'], data['X_diam_test'],
    data['y_diam_train'], data['y_diam_test'],
    data['X_diam_train_scaled'], data['X_diam_test_scaled'],
    'effective_diameter_mm'
)


# ═══════════════════════════════════════════════
# TRAIN FOR TARGET: LENGTH
# ═══════════════════════════════════════════════
print("\n\n" + "="*70)
print("TRAINING MODELS FOR LENGTH (mm)")
print("="*70)

results_len = train_and_evaluate(
    data['X_len_train'], data['X_len_test'],
    data['y_len_train'], data['y_len_test'],
    data['X_len_train_scaled'], data['X_len_test_scaled'],
    'length_mm'
)


# ═══════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════
def print_comparison_table(results, target_name):
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON — {target_name}")
    print(f"{'='*70}")
    print(f"\n{'Model':<25s} {'CV R²':>8s} {'Test R²':>8s} {'MAE(mm)':>8s} {'RMSE(mm)':>9s} {'MAPE(%)':>8s}")
    print("─" * 70)

    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    for name, r in sorted_models:
        marker = " ⭐" if name == sorted_models[0][0] else ""
        print(f"{name:<25s} {r['cv_r2']:>8.4f} {r['test_r2']:>8.4f} {r['test_mae']:>8.2f} {r['test_rmse']:>9.2f} {r['test_mape']:>8.1f}{marker}")

    best_name = sorted_models[0][0]
    best = sorted_models[0][1]
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Test R² = {best['test_r2']:.4f}, MAE = {best['test_mae']:.2f} mm, RMSE = {best['test_rmse']:.2f} mm")

print_comparison_table(results_diam, 'EFFECTIVE DIAMETER')
print_comparison_table(results_len, 'LENGTH')


# ═══════════════════════════════════════════════
# FEATURE IMPORTANCE (from tree models)
# ═══════════════════════════════════════════════
print(f"\n{'='*70}")
print("FEATURE IMPORTANCE (Random Forest — Diameter)")
print(f"{'='*70}")
rf_model = results_diam['Random Forest']['model']
importances = rf_model.feature_importances_
feat_names = data['feature_names']
sorted_idx = np.argsort(importances)[::-1]
for i in sorted_idx:
    print(f"  {feat_names[i]:<25s}  {importances[i]:.4f}")


# ═══════════════════════════════════════════════
# SAVE ALL RESULTS
# ═══════════════════════════════════════════════
print(f"\nSaving trained models and results...")

training_results = {
    'diameter': {},
    'length': {},
    'feature_names': data['feature_names'],
}

for target, results in [('diameter', results_diam), ('length', results_len)]:
    for name, r in results.items():
        training_results[target][name] = {  # type: ignore
            'model': r['model'],
            'best_params': r['best_params'],
            'cv_r2': r['cv_r2'],
            'test_r2': r['test_r2'],
            'test_mae': r['test_mae'],
            'test_rmse': r['test_rmse'],
            'test_mape': r['test_mape'],
            'y_pred': r['y_pred'],
            'y_test': r['y_test'],
            'use_scaled': r['use_scaled'],
        }

output_path = os.path.join(SCRIPT_DIR, 'training_results.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(training_results, f)

print(f"  Saved to: {output_path}")
print("\n✅ Phase 3 Complete — All models trained and evaluated!")

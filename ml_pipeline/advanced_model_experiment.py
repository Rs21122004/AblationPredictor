"""
Advanced Model Experiment — Ablation Zone Prediction
Honours Project — Rishi Sharma

Tries additional boosting algorithms and ensemble strategies
to improve on the baseline R² of 0.695 (Random Forest, diameter).

New models:
  - XGBoost          (GPU-optional, highly tuned boosting)
  - LightGBM         (GBDT variant, fast leaf-wise)
  - CatBoost         (handles categoricals natively)
  - Extra Trees      (more randomised than Random Forest)
  - Stacking         (blends best base models with a Ridge meta-learner)

Run from the ml_pipeline/ directory:
  python advanced_model_experiment.py
"""

import os
import pickle
import warnings
import time
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    ExtraTreesRegressor,
    StackingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    print(f"⚠️  XGBoost unavailable ({e.__class__.__name__}) — skipping.")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except Exception as e:
    HAS_LGB = False
    print(f"⚠️  LightGBM unavailable ({e.__class__.__name__}) — skipping.")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception as e:
    HAS_CAT = False
    print(f"⚠️  CatBoost unavailable ({e.__class__.__name__}) — skipping.")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)   # HonoursReview2/

# ─── Load preprocessed splits (same as model_training.py) ───
with open(os.path.join(ROOT_DIR, 'preprocessed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

FEATURE_NAMES = data['feature_names']

print("=" * 70)
print("ADVANCED MODEL EXPERIMENT — Ablation Zone Prediction")
print("=" * 70)
print(f"Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}\n")


# ─── Utility ───────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(name, model, X_tr, y_tr, X_te, y_te, cv=5):
    """Fit, predict, return dict of metrics."""
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    y_pred = model.predict(X_te)
    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='r2').mean()

    return {
        'name': name,
        'cv_r2': cv_r2,
        'test_r2': r2_score(y_te, y_pred),
        'mae': mean_absolute_error(y_te, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_te, y_pred)),
        'mape': mape(y_te, y_pred),
        'time_s': elapsed,
        'model': model,
        'y_pred': y_pred,
    }


def run_experiments(X_train, X_test, y_train, y_test,
                    X_train_sc, X_test_sc, label):
    """Run all advanced models for one target variable."""

    print(f"\n{'═'*70}")
    print(f"TARGET: {label}")
    print(f"  Train = {len(X_train)} | Test = {len(X_test)}")
    print(f"{'═'*70}\n")

    results = []

    # ── 1. Extra Trees ────────────────────────────────────────────────────
    print("▶ Extra Trees ...")
    et = ExtraTreesRegressor(
        n_estimators=300, max_depth=None,
        min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    results.append(evaluate("Extra Trees", et, X_train, y_train, X_test, y_test))

    # ── 2. XGBoost ───────────────────────────────────────────────────────
    if HAS_XGB:
        print("▶ XGBoost (tuned) ...")
        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        results.append(evaluate("XGBoost", xgb, X_train, y_train, X_test, y_test))

    # ── 3. LightGBM ──────────────────────────────────────────────────────
    if HAS_LGB:
        print("▶ LightGBM (tuned) ...")
        lgb = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        results.append(evaluate("LightGBM", lgb, X_train, y_train, X_test, y_test))

    # ── 4. CatBoost ──────────────────────────────────────────────────────
    if HAS_CAT:
        print("▶ CatBoost (tuned) ...")
        cat = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=0,
        )
        results.append(evaluate("CatBoost", cat, X_train, y_train, X_test, y_test))

    # ── 5. XGBoost fine-tuned via GridSearchCV ────────────────────────────
    if HAS_XGB:
        print("▶ XGBoost (GridSearchCV) ...")
        xgb_grid = GridSearchCV(
            XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            param_grid={
                'n_estimators': [300, 500],
                'max_depth': [4, 5, 6],
                'learning_rate': [0.03, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.7, 0.9],
            },
            cv=5, scoring='r2', n_jobs=-1, refit=True,
        )
        xgb_grid.fit(X_train, y_train)
        best_xgb = xgb_grid.best_estimator_
        r = evaluate("XGBoost (CV-tuned)", best_xgb, X_train, y_train, X_test, y_test)
        r['best_params'] = xgb_grid.best_params_
        results.append(r)
        print(f"    Best params: {xgb_grid.best_params_}")

    # ── 6. Stacking Ensemble ─────────────────────────────────────────────
    print("▶ Stacking Ensemble ...")
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
    ]
    if HAS_XGB:
        base_estimators.append(
            ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05,
                                  max_depth=5, random_state=42, n_jobs=-1, verbosity=0))
        )
    if HAS_LGB:
        base_estimators.append(
            ('lgb', LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=6, random_state=42, n_jobs=-1, verbose=-1))
        )

    stacker = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5, n_jobs=-1,
    )
    results.append(evaluate("Stacking Ensemble", stacker, X_train, y_train, X_test, y_test))

    # ── Print comparison table ────────────────────────────────────────────
    BASELINE_R2 = 0.695 if "diameter" in label.lower() else 0.512

    print(f"\n{'─'*70}")
    print(f"{'Model':<25} {'CV R²':>7} {'Test R²':>8} {'MAE':>7} {'RMSE':>7} {'MAPE%':>7}  Time")
    print("─" * 70)

    results.sort(key=lambda x: x['test_r2'], reverse=True)
    for r in results:
        flag = ""
        if r['test_r2'] > BASELINE_R2:
            flag = " 🚀 BEATS BASELINE"
        elif abs(r['test_r2'] - BASELINE_R2) < 0.005:
            flag = " ≈ tie"
        print(
            f"{r['name']:<25} {r['cv_r2']:>7.4f} {r['test_r2']:>8.4f} "
            f"{r['mae']:>7.2f} {r['rmse']:>7.2f} {r['mape']:>7.1f}  {r['time_s']:.1f}s{flag}"
        )

    best = results[0]
    print(f"\n🏆  Best: {best['name']}")
    print(f"    Test R² = {best['test_r2']:.4f}  |  MAE = {best['mae']:.2f} mm")
    print(f"    Baseline (Random Forest): R² = {BASELINE_R2}")
    if best['test_r2'] > BASELINE_R2:
        improvement = (best['test_r2'] - BASELINE_R2) / BASELINE_R2 * 100
        print(f"    ✅ Improvement over baseline: +{best['test_r2'] - BASELINE_R2:.4f} ({improvement:.1f}%)")
    else:
        print(f"    ⚠️  Did not beat baseline (difference = {best['test_r2'] - BASELINE_R2:+.4f})")

    return results


# ─── Run for diameter ────────────────────────────────────────────────────────
diam_results = run_experiments(
    data['X_diam_train'], data['X_diam_test'],
    data['y_diam_train'], data['y_diam_test'],
    data['X_diam_train_scaled'], data['X_diam_test_scaled'],
    label='Effective Diameter (mm)',
)

# ─── Run for length ──────────────────────────────────────────────────────────
len_results = run_experiments(
    data['X_len_train'], data['X_len_test'],
    data['y_len_train'], data['y_len_test'],
    data['X_len_train_scaled'], data['X_len_test_scaled'],
    label='Length (mm)',
)

# ─── Save best models to disk ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SAVING BEST ADVANCED MODELS")
print(f"{'='*70}")

output = {
    'diameter_best': diam_results[0],
    'length_best': len_results[0],
    'diameter_all': diam_results,
    'length_all': len_results,
    'feature_names': FEATURE_NAMES,
}

save_path = os.path.join(ROOT_DIR, 'advanced_training_results.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(output, f)
print(f"  Saved → {save_path}")

print("\n✅ Experiment complete!")

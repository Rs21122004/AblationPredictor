"""
Phase 4: Model Results & Visualization
Honours Project - Ablation Zone Prediction

Generates publication-quality comparison charts, predicted vs actual plots,
feature importance, and residual analysis from trained models.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import pickle
import os
import matplotlib  # type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# Load results
with open(os.path.join(SCRIPT_DIR, 'training_results.pkl'), 'rb') as f:
    results = pickle.load(f)

print("="*60)
print("PHASE 4: RESULTS VISUALIZATION")
print("="*60)


# ═══════════════════════════════════════════════
# PLOT 1: Model Comparison Bar Charts
# ═══════════════════════════════════════════════
print("\n1/5  Model comparison bar charts...")

for target, target_label in [('diameter', 'Effective Diameter'), ('length', 'Length')]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Model Comparison — {target_label} Prediction', fontsize=15, fontweight='bold')

    models = list(results[target].keys())
    r2_vals = [results[target][m]['test_r2'] for m in models]
    mae_vals = [results[target][m]['test_mae'] for m in models]
    rmse_vals = [results[target][m]['test_rmse'] for m in models]

    colors = sns.color_palette('Set2', len(models))

    # Sort by R²
    sorted_idx = np.argsort(r2_vals)[::-1]
    models_sorted = [models[i] for i in sorted_idx]
    colors_sorted = [colors[i] for i in sorted_idx]

    # R² Score
    ax = axes[0]
    r2_sorted = [r2_vals[i] for i in sorted_idx]
    bars = ax.barh(models_sorted, r2_sorted, color=colors_sorted, edgecolor='white')
    ax.set_xlabel('R² Score')
    ax.set_title('R² Score (higher = better)')
    ax.set_xlim(-0.1, 1.0)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, r2_sorted):
        x_pos = max(bar.get_width(), 0) + 0.02
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
    ax.invert_yaxis()

    # MAE
    ax = axes[1]
    mae_sorted = [mae_vals[i] for i in sorted_idx]
    bars = ax.barh(models_sorted, mae_sorted, color=colors_sorted, edgecolor='white')
    ax.set_xlabel('MAE (mm)')
    ax.set_title('Mean Absolute Error (lower = better)')
    for bar, val in zip(bars, mae_sorted):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=9)
    ax.invert_yaxis()

    # RMSE
    ax = axes[2]
    rmse_sorted = [rmse_vals[i] for i in sorted_idx]
    bars = ax.barh(models_sorted, rmse_sorted, color=colors_sorted, edgecolor='white')
    ax.set_xlabel('RMSE (mm)')
    ax.set_title('Root Mean Squared Error (lower = better)')
    for bar, val in zip(bars, rmse_sorted):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'model_comparison_{target}.png'))
    plt.close()


# ═══════════════════════════════════════════════
# PLOT 2: Predicted vs Actual (Best Models)
# ═══════════════════════════════════════════════
print("2/5  Predicted vs Actual scatter plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Predicted vs Actual — Best Models', fontsize=15, fontweight='bold')

for ax, (target, target_label) in zip(axes, [('diameter', 'Effective Diameter (mm)'), ('length', 'Length (mm)')]):
    # Find best model
    best_name = max(results[target], key=lambda m: results[target][m]['test_r2'])
    best = results[target][best_name]
    y_test = best['y_test']
    y_pred = best['y_pred']

    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='gray', s=60, color='steelblue')

    # Perfect prediction line
    lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')

    ax.set_xlabel(f'Actual {target_label}')
    ax.set_ylabel(f'Predicted {target_label}')
    ax.set_title(f'{best_name}\nR² = {best["test_r2"]:.4f}, MAE = {best["test_mae"]:.2f} mm')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'predicted_vs_actual.png'))
plt.close()


# ═══════════════════════════════════════════════
# PLOT 3: Feature Importance
# ═══════════════════════════════════════════════
print("3/5  Feature importance plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Feature Importance — Random Forest & Gradient Boosting', fontsize=15, fontweight='bold')

feature_names = results['feature_names']

for ax, (model_name, target) in zip(axes, [('Random Forest', 'diameter'), ('XGBoost (GBR)', 'diameter')]):
    model = results[target][model_name]['model']
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx],
            color=sns.color_palette('viridis', len(feature_names)), edgecolor='white')
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{model_name} — Diameter Prediction')

    for i, idx in enumerate(sorted_idx):
        ax.text(importances[idx] + 0.005, i, f'{importances[idx]:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'))
plt.close()


# ═══════════════════════════════════════════════
# PLOT 4: Residual Analysis
# ═══════════════════════════════════════════════
print("4/5  Residual analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Residual Analysis — Best Models', fontsize=15, fontweight='bold')

for row, (target, target_label) in enumerate([('diameter', 'Diameter'), ('length', 'Length')]):
    best_name = max(results[target], key=lambda m: results[target][m]['test_r2'])
    best = results[target][best_name]
    y_test = best['y_test']
    y_pred = best['y_pred']
    residuals = y_test - y_pred

    # Residuals vs Predicted
    ax = axes[row, 0]
    ax.scatter(y_pred, residuals, alpha=0.7, edgecolors='gray', s=50, color='coral')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel(f'Predicted {target_label} (mm)')
    ax.set_ylabel('Residual (mm)')
    ax.set_title(f'{best_name} — Residuals vs Predicted ({target_label})')

    # Residual Distribution
    ax = axes[row, 1]
    ax.hist(residuals, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(residuals.mean(), color='black', linestyle=':', linewidth=1.2,
               label=f'Mean: {residuals.mean():.2f} mm')
    ax.set_xlabel('Residual (mm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Residual Distribution ({target_label})')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'residual_analysis.png'))
plt.close()


# ═══════════════════════════════════════════════
# PLOT 5: Cross-Validation vs Test R² Comparison
# ═══════════════════════════════════════════════
print("5/5  CV vs Test R² comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Cross-Validation R² vs Test R² — Overfitting Check', fontsize=15, fontweight='bold')

for ax, (target, target_label) in zip(axes, [('diameter', 'Diameter'), ('length', 'Length')]):
    models = list(results[target].keys())
    cv_r2 = [results[target][m]['cv_r2'] for m in models]
    test_r2 = [results[target][m]['test_r2'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, cv_r2, width, label='CV R² (10-fold)', color='steelblue', edgecolor='white')
    bars2 = ax.bar(x + width/2, test_r2, width, label='Test R²', color='coral', edgecolor='white')

    ax.set_ylabel('R² Score')
    ax.set_title(f'{target_label} Prediction')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax.legend()
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylim(-0.2, 1.0)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', fontsize=8)
    for bar in bars2:
        y_pos = max(bar.get_height(), 0) + 0.02
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{bar.get_height():.2f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'cv_vs_test_comparison.png'))
plt.close()


# ═══════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════
print(f"\n{'='*75}")
print("COMPLETE RESULTS SUMMARY")
print(f"{'='*75}")

for target, target_label in [('diameter', 'EFFECTIVE DIAMETER'), ('length', 'LENGTH')]:
    print(f"\n── {target_label} ──")
    print(f"{'Model':<22s} {'CV R²':>7s} {'Test R²':>8s} {'MAE(mm)':>8s} {'RMSE(mm)':>9s} {'MAPE(%)':>8s}")
    print("─" * 75)
    sorted_models = sorted(results[target].items(), key=lambda x: x[1]['test_r2'], reverse=True)
    for name, r in sorted_models:
        marker = " ⭐" if name == sorted_models[0][0] else ""
        print(f"{name:<22s} {r['cv_r2']:>7.4f} {r['test_r2']:>8.4f} {r['test_mae']:>8.2f} {r['test_rmse']:>9.2f} {r['test_mape']:>8.1f}{marker}")

print(f"\n✅ All plots saved to: {PLOTS_DIR}/")
print("Done!")

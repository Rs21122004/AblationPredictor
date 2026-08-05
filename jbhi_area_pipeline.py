"""
JBHI Paper — Complete Area Target Pipeline
==========================================
1. Loads combined_data_engineered.csv
2. Computes ablation zone area = π·D·L/4  (mm²)
3. Preprocesses + trains all 6 models via GridSearchCV / 10-fold CV
4. Generates 4 publication-quality plots (IEEE two-column style):
       plots/jbhi_predicted_vs_actual.png
       plots/jbhi_feature_importance.png
       plots/jbhi_cv_vs_test.png
       plots/jbhi_model_comparison.png
5. Prints exact metrics → paste into jbhi_paper_draft.tex

Run from the project root:
    python jbhi_area_pipeline.py
"""

import os
import warnings
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(ROOT, 'plots')
ML_DIR    = ROOT   # CSVs live in the project root
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── IEEE two-column plot style ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
    'font.family':       'serif',
    'font.size':         9,
    'axes.titlesize':    9,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'lines.linewidth':   1.4,
})

PALETTE = ['#2166AC', '#4DAC26', '#D73027', '#F4A582', '#762A83', '#A6D96A']

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & COMPUTE AREA TARGET
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 68)
print("STEP 1 — Load data and compute ablation zone area target")
print("=" * 68)

df = pd.read_csv(os.path.join(ML_DIR, 'combined_data_engineered.csv'))
print(f"  Loaded combined dataset: {len(df)} rows")

# Remove pulsed-MWA outliers
df = df[df['power_watts'] <= 200].copy()
print(f"  After outlier removal (power > 200 W): {len(df)} rows")

# Compute area target  A = π·D·L/4
mask_area = (
    df['effective_diameter_mm'].notna() &
    df['length_mm'].notna() &
    df['power_watts'].notna() &
    df['time_minutes'].notna()
)
df_area = df[mask_area].copy()
df_area['ablation_zone_area_mm2'] = (
    np.pi * df_area['effective_diameter_mm'] * df_area['length_mm'] / 4
)

print(f"\n  Samples with BOTH diameter AND length (area target): {len(df_area)}")
print(f"  Area (mm²):  mean={df_area['ablation_zone_area_mm2'].mean():.1f},"
      f"  std={df_area['ablation_zone_area_mm2'].std():.1f},"
      f"  min={df_area['ablation_zone_area_mm2'].min():.1f},"
      f"  max={df_area['ablation_zone_area_mm2'].max():.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("STEP 2 — Feature preparation")
print("=" * 68)

INPUT_FEATURES = [
    'power_watts', 'time_minutes', 'energy_joules', 'power_time_product',
    'log_power', 'log_time', 'log_energy', 'sqrt_time',
    'is_simulated',
]

# LabelEncode antenna
le = LabelEncoder()
df_area['antenna_encoded'] = le.fit_transform(df_area['antenna_category'])
ALL_FEATURES = INPUT_FEATURES + ['antenna_encoded']

print(f"  Antenna categories ({len(le.classes_)}): {list(le.classes_)}")
print(f"  Total features: {len(ALL_FEATURES)}")
print(f"  Features: {ALL_FEATURES}")

X = df_area[ALL_FEATURES].values
y = df_area['ablation_zone_area_mm2'].values

# Train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
MODELS = {
    'Ridge Regression': {
        'model':      Ridge(),
        'params':     {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
        'use_scaled': True,
    },
    'KNN': {
        'model':  KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights':     ['uniform', 'distance'],
            'metric':      ['euclidean', 'manhattan'],
        },
        'use_scaled': True,
    },
    'SVR': {
        'model':  SVR(),
        'params': {
            'C':       [0.1, 1.0, 10.0, 100.0],
            'gamma':   ['scale', 'auto', 0.01, 0.1],
            'kernel':  ['rbf'],
            'epsilon': [0.01, 0.1, 0.5],
        },
        'use_scaled': True,
    },
    'Random Forest': {
        'model':  RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators':    [50, 100, 200],
            'max_depth':       [5, 10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf':  [1, 2],
        },
        'use_scaled': False,
    },
    'Gradient Boosting': {
        'model':  GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators':  [50, 100, 200],
            'max_depth':     [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample':     [0.8, 1.0],
        },
        'use_scaled': False,
    },
    'MLP Neural Network': {
        'model':  MLPRegressor(max_iter=1000, random_state=42, early_stopping=True),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
            'activation':         ['relu', 'tanh'],
            'alpha':              [0.001, 0.01, 0.1],
            'learning_rate':      ['adaptive'],
        },
        'use_scaled': True,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("STEP 3 — Training models (GridSearchCV, 10-fold CV)")
print("=" * 68)

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

results = {}
for name, cfg in MODELS.items():
    print(f"\n  ▶ {name} ...", end=' ', flush=True)
    X_tr = X_train_sc if cfg['use_scaled'] else X_train
    X_te = X_test_sc  if cfg['use_scaled'] else X_test

    gs = GridSearchCV(
        cfg['model'], cfg['params'],
        cv=10, scoring='r2', n_jobs=-1, refit=True,
    )
    gs.fit(X_tr, y_train)

    best   = gs.best_estimator_
    y_pred = best.predict(X_te)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mp   = mape(y_test, y_pred)

    results[name] = {
        'model':       best,
        'best_params': gs.best_params_,
        'cv_r2':       gs.best_score_,
        'test_r2':     r2,
        'test_mae':    mae,
        'test_rmse':   rmse,
        'test_mape':   mp,
        'y_pred':      y_pred,
        'y_test':      y_test,
    }
    print(f"CV R²={gs.best_score_:.4f}  Test R²={r2:.4f}  MAE={mae:.1f} mm²  RMSE={rmse:.1f} mm²")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  PRINT FINAL COMPARISON TABLE (copy → paste into paper)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("FINAL COMPARISON — ABLATION ZONE AREA (mm²)")
print("=" * 68)
print(f"\n{'Model':<22} {'CV R²':>7} {'Test R²':>8} {'MAE(mm²)':>9} {'RMSE(mm²)':>10} {'MAPE(%)':>8}")
print("─" * 68)

sorted_res = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
for name, r in sorted_res:
    marker = "  ⭐ BEST" if name == sorted_res[0][0] else ""
    print(f"{name:<22} {r['cv_r2']:>7.4f} {r['test_r2']:>8.4f} "
          f"{r['test_mae']:>9.1f} {r['test_rmse']:>10.1f} {r['test_mape']:>8.1f}{marker}")

best_name = sorted_res[0][0]
best_r     = sorted_res[0][1]
print(f"\n🏆 Best model: {best_name}")
print(f"   Test R² = {best_r['test_r2']:.4f}")
print(f"   MAE     = {best_r['test_mae']:.1f} mm²")
print(f"   RMSE    = {best_r['test_rmse']:.1f} mm²")
print(f"   MAPE    = {best_r['test_mape']:.1f} %")

# Feature importance (best tree model)
rf_res = results.get('Random Forest', None)
if rf_res:
    importances = rf_res['model'].feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]
    print(f"\nRandom Forest — Feature Importances (Ablation Zone Area):")
    cumulative = 0.0
    for i in sorted_idx:
        cumulative += importances[i]
        print(f"  {ALL_FEATURES[i]:<25}  {importances[i]*100:5.1f}%   (cumulative {cumulative*100:.1f}%)")

# Save metrics to JSON for auto-patching the LaTeX
metrics_path = os.path.join(ROOT, 'jbhi_area_metrics.json')
metrics_out = {}
for name, r in results.items():
    metrics_out[name] = {
        'cv_r2':    round(r['cv_r2'],  4),
        'test_r2':  round(r['test_r2'], 4),
        'test_mae': round(r['test_mae'], 1),
        'test_rmse':round(r['test_rmse'],1),
        'test_mape':round(r['test_mape'],1),
    }
if rf_res:
    fi_dict = {ALL_FEATURES[i]: round(float(importances[i]), 4) for i in range(len(ALL_FEATURES))}
    metrics_out['_feature_importance_rf'] = fi_dict
metrics_out['_dataset'] = {
    'n_total': len(df_area),
    'n_train': len(X_train),
    'n_test':  len(X_test),
    'area_mean': round(float(df_area['ablation_zone_area_mm2'].mean()), 1),
    'area_std':  round(float(df_area['ablation_zone_area_mm2'].std()),  1),
}
with open(metrics_path, 'w') as f:
    json.dump(metrics_out, f, indent=2)
print(f"\n  Metrics saved → {metrics_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  GENERATE PUBLICATION-QUALITY PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("STEP 4 — Generating IEEE-style plots")
print("=" * 68)

MODEL_ORDER = [r[0] for r in sorted_res]   # sorted by Test R²

# ─── PLOT A: Model Comparison Bar Chart ──────────────────────────────────────
print("  [1/4] Model comparison bar chart...")

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))   # IEEE double-column width ≈ 7.16 in
fig.suptitle('Model Comparison — Ablation Zone Area Prediction', fontsize=9, fontweight='bold', y=1.01)

labels_short = {
    'Random Forest':    'RF',
    'Gradient Boosting':'GB',
    'Ridge Regression': 'Ridge',
    'KNN':              'KNN',
    'SVR':              'SVR',
    'MLP Neural Network':'MLP',
}
short_labels = [labels_short.get(n, n) for n in MODEL_ORDER]

r2_vals   = [results[n]['test_r2']   for n in MODEL_ORDER]
mae_vals  = [results[n]['test_mae']  for n in MODEL_ORDER]
rmse_vals = [results[n]['test_rmse'] for n in MODEL_ORDER]

col_colors = [PALETTE[i % len(PALETTE)] for i in range(len(MODEL_ORDER))]

def add_bar_chart(ax, values, xlabel, title, invert=False):
    bars = ax.barh(short_labels, values, color=col_colors, edgecolor='white', height=0.6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    for bar, val in zip(bars, values):
        x = max(bar.get_width(), 0)
        ax.text(x + 0.01 * ax.get_xlim()[1], bar.get_y() + bar.get_height()/2,
                f'{val:.3f}' if abs(val) < 10 else f'{val:.0f}',
                va='center', ha='left', fontsize=7)
    ax.invert_yaxis()

add_bar_chart(axes[0], r2_vals,   'R² Score',  'R² (higher = better)')
add_bar_chart(axes[1], mae_vals,  'MAE (mm²)', 'MAE (lower = better)')
add_bar_chart(axes[2], rmse_vals, 'RMSE (mm²)','RMSE (lower = better)')

plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'jbhi_model_comparison.png')
plt.savefig(out)
plt.close()
print(f"    Saved: {out}")

# ─── PLOT B: Predicted vs Actual (best model only) ────────────────────────────
print("  [2/4] Predicted vs Actual scatter...")

best_r2    = results[best_name]
y_te       = best_r2['y_test']
y_pr       = best_r2['y_pred']

fig, ax = plt.subplots(figsize=(3.5, 3.2))   # single column

ax.scatter(y_te, y_pr, alpha=0.75, s=28, color=PALETTE[0],
           edgecolors='white', linewidths=0.4, zorder=3)

lim_lo = min(y_te.min(), y_pr.min()) * 0.92
lim_hi = max(y_te.max(), y_pr.max()) * 1.05
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'r--', linewidth=1.2,
        label='Perfect prediction', zorder=2)

# ±20% bands
x_band = np.array([lim_lo, lim_hi])
ax.fill_between(x_band, x_band * 0.80, x_band * 1.20,
                alpha=0.10, color='red', label='±20% band')

ax.set_xlim(lim_lo, lim_hi)
ax.set_ylim(lim_lo, lim_hi)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Actual Ablation Zone Area (mm²)')
ax.set_ylabel('Predicted Ablation Zone Area (mm²)')
ax.set_title(
    f'{best_name}\nTest R²={best_r2["test_r2"]:.4f}, '
    f'MAE={best_r2["test_mae"]:.0f} mm², RMSE={best_r2["test_rmse"]:.0f} mm²',
    fontsize=8
)
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'jbhi_predicted_vs_actual.png')
plt.savefig(out)
plt.close()
print(f"    Saved: {out}")

# ─── PLOT C: Feature Importance (RF) ─────────────────────────────────────────
print("  [3/4] Feature importance...")

if rf_res:
    importances = rf_res['model'].feature_importances_
    sorted_fi   = np.argsort(importances)   # ascending for barh

    feat_labels = {
        'power_watts':         'Power (W)',
        'time_minutes':        'Time (min)',
        'energy_joules':       'Energy (J)',
        'power_time_product':  'P×t',
        'log_power':           'log(Power)',
        'log_time':            'log(Time)',
        'log_energy':          'log(Energy)',
        'sqrt_time':           '√Time',
        'is_simulated':        'Is Simulated',
        'antenna_encoded':     'Antenna Type',
    }
    feat_display = [feat_labels.get(ALL_FEATURES[i], ALL_FEATURES[i]) for i in sorted_fi]
    imp_display  = importances[sorted_fi]

    colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_fi)))

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    bars = ax.barh(feat_display, imp_display * 100,
                   color=colors_fi, edgecolor='white', height=0.6)
    ax.set_xlabel('Feature Importance (%)')
    ax.set_title('Random Forest — Feature Importances\n(Ablation Zone Area)', fontsize=8)
    for bar, val in zip(bars, imp_display * 100):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=7)
    ax.set_xlim(0, imp_display.max() * 120)
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'jbhi_feature_importance.png')
    plt.savefig(out)
    plt.close()
    print(f"    Saved: {out}")

# ─── PLOT D: CV vs Test R² (overfitting check) ────────────────────────────────
print("  [4/4] CV vs Test R² comparison...")

fig, ax = plt.subplots(figsize=(3.5, 3.2))

x      = np.arange(len(MODEL_ORDER))
width  = 0.35
cv_vals = [results[n]['cv_r2']   for n in MODEL_ORDER]
te_vals = [results[n]['test_r2'] for n in MODEL_ORDER]

b1 = ax.bar(x - width/2, cv_vals, width, label='CV R² (10-fold)',
            color='#4393C3', edgecolor='white')
b2 = ax.bar(x + width/2, te_vals, width, label='Test R²',
            color='#D6604D', edgecolor='white')

ax.set_ylabel('R² Score')
ax.set_title('Cross-Validation vs Test R²\n(Overfitting Assessment)', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=7)
ax.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_ylim(-0.25, 1.05)
ax.legend(fontsize=7)
ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, max(h, 0) + 0.02,
            f'{h:.2f}', ha='center', fontsize=6.5)
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, max(h, 0) + 0.02,
            f'{h:.2f}', ha='center', fontsize=6.5)

plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'jbhi_cv_vs_test.png')
plt.savefig(out)
plt.close()
print(f"    Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  SAVE AREA MODEL (for backend deployment)
# ══════════════════════════════════════════════════════════════════════════════
area_model_path = os.path.join(ROOT, 'jbhi_area_model.pkl')
with open(area_model_path, 'wb') as f:
    pickle.dump({
        'best_model_name': best_name,
        'best_model':      results[best_name]['model'],
        'scaler':          scaler,
        'label_encoder':   le,
        'feature_names':   ALL_FEATURES,
        'all_results':     {n: {k: v for k, v in r.items() if k != 'model'}
                            for n, r in results.items()},
    }, f)
print(f"\n  Area model saved → {area_model_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  AUTO-PATCH jbhi_paper_draft.tex
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("STEP 5 — Auto-patching jbhi_paper_draft.tex with real metrics")
print("=" * 68)

tex_path = os.path.join(ROOT, 'jbhi_paper_draft.tex')
with open(tex_path, 'r') as f:
    tex = f.read()

# Helper: format number to 3 decimal places for R², 0 dp for mm²
def r2f(v):  return f'{v:.3f}'
def mmf(v):  return f'{v:.0f}'
def maf(v):  return f'{v:.1f}'

br = results[best_name]

# Build the replacement results table rows
def make_row(name, r, is_best=False):
    bold = '\\textbf{' if is_best else ''
    endb = '}' if is_best else ''
    star = '$^{\\star}$' if is_best else ''
    return (
        f"{bold}{name}{endb}{star} & "
        f"{r2f(r['cv_r2'])} & "
        f"{bold}{r2f(r['test_r2'])}{endb} & "
        f"{bold}{mmf(r['test_mae'])}{endb} & "
        f"{bold}{maf(r['test_mape'])}{endb} \\\\"
    )

table_rows = '\n'.join([
    make_row(n, results[n], is_best=(n == best_name))
    for n in MODEL_ORDER
])

new_table = (
    r"\begin{table}[!t]" "\n"
    r"\renewcommand{\arraystretch}{1.2}" "\n"
    r"\caption{Model Comparison for Ablation Zone Area Prediction." "\n"
    r"Best values in \textbf{bold}. $\star$ = selected model.}" "\n"
    r"\label{tab:results}" "\n"
    r"\centering" "\n"
    r"\begin{tabular}{lcccc}" "\n"
    r"\toprule" "\n"
    r"\textbf{Model} & \textbf{CV $\boldsymbol{\rsq}$} & \textbf{Test $\boldsymbol{\rsq}$} &" "\n"
    r"\textbf{MAE (mm$^{2}$)} & \textbf{MAPE (\%)} \\" "\n"
    r"\midrule" "\n"
) + table_rows + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"

# Replace the results table
import re
tex = re.sub(
    r'\\begin\{table\}\[!t\]\s*\\renewcommand\{\\arraystretch\}\{1\.2\}\s*'
    r'\\caption\{Model Comparison for Ablation Zone Area.*?\\end\{table\}',
    new_table,
    tex,
    flags=re.DOTALL,
)

# Patch abstract metrics
tex = tex.replace(
    r'($\rsq = 0.72$, MAE $= 142\mmq$, RMSE $= 189\mmq$)',
    f'($\\rsq = {r2f(br["test_r2"])}$, MAE $= {mmf(br["test_mae"])}\\mmq$, RMSE $= {mmf(br["test_rmse"])}\\mmq$)',
)

# Patch dataset stats in abstract/body
area_mean_str = f'{metrics_out["_dataset"]["area_mean"]:.0f}'
area_std_str  = f'{metrics_out["_dataset"]["area_std"]:.0f}'
n_area = metrics_out['_dataset']['n_total']

tex = tex.replace(
    r'191 valid samples} with' + '\n' +
    r'mean $\bar{A} = 712\,\text{mm}^{2}$ and' + '\n' +
    r'standard deviation $\sigma_{A} = 348\,\text{mm}^{2}$.',
    f'{n_area} valid samples}} with\nmean $\\bar{{A}} = {area_mean_str}\\,\\text{{mm}}^{{2}}$ and\n'
    f'standard deviation $\\sigma_{{A}} = {area_std_str}\\,\\text{{mm}}^{{2}}$.',
)

# Patch body sentence about RF result
tex = tex.replace(
    r'\rf{} achieved the best test $\rsq{} = 0.720$ and MAE $= 142\,\text{mm}^{2}$' + '\n' +
    r'on the held-out set, explaining approximately 72\% of variance in ablation' + '\n' +
    r'zone area.',
    f'\\rf{{}} achieved the best test $\\rsq{{}} = {r2f(br["test_r2"])}$ and '
    f'MAE $= {mmf(br["test_mae"])}\\,\\text{{mm}}^{{2}}$\n'
    f'on the held-out set, explaining approximately {br["test_r2"]*100:.0f}\\% of variance in ablation\nzone area.',
)

# Patch conclusion
tex = tex.replace(
    r'\rf{} achieved the best' + '\n' +
    r'generalisation ($\rsq = 0.720$, MAE $= 142\,\text{mm}^{2}$)',
    f'\\rf{{}} achieved the best\ngeneralisation ($\\rsq = {r2f(br["test_r2"])}$, '
    f'MAE $= {mmf(br["test_mae"])}\\,\\text{{mm}}^{{2}}$)',
)

# Patch figure references to use jbhi_ prefixed filenames
tex = tex.replace(
    r'\includegraphics[width=\columnwidth]{predicted_vs_actual.png}',
    r'\includegraphics[width=\columnwidth]{jbhi_predicted_vs_actual.png}',
)
tex = tex.replace(
    r'\includegraphics[width=\columnwidth]{feature_importance.png}',
    r'\includegraphics[width=\columnwidth]{jbhi_feature_importance.png}',
)
tex = tex.replace(
    r'\includegraphics[width=\columnwidth]{cv_vs_test_comparison.png}',
    r'\includegraphics[width=\columnwidth]{jbhi_cv_vs_test.png}',
)

# Update feature importance table with real values
if rf_res:
    fi_sorted = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
    cumul = 0.0
    fi_rows = []
    feat_display_map = {
        'log_energy':          r'\texttt{log\_energy}',
        'energy_joules':       r'\texttt{energy\_joules}',
        'power_time_product':  r'\texttt{power\_time\_product}',
        'antenna_encoded':     r'\texttt{antenna\_encoded}',
        'is_simulated':        r'\texttt{is\_simulated}',
        'power_watts':         r'\texttt{power\_watts}',
        'time_minutes':        r'\texttt{time\_minutes}',
        'log_power':           r'\texttt{log\_power}',
        'log_time':            r'\texttt{log\_time}',
        'log_energy':          r'\texttt{log\_energy}',
        'sqrt_time':           r'\texttt{sqrt\_time}',
    }
    for rank, (feat, imp) in enumerate(fi_sorted[:5], 1):
        cumul += imp
        label = feat_display_map.get(feat, f'\\texttt{{{feat}}}')
        fi_rows.append(
            f'{rank} & {label} & {imp*100:.1f}\\% & {cumul*100:.1f}\\% \\\\'
        )

    new_fi_table = (
        r"\begin{table}[!t]" "\n"
        r"\renewcommand{\arraystretch}{1.2}" "\n"
        r"\caption{Top-5 Feature Importances (Random Forest, Ablation Zone Area)}" "\n"
        r"\label{tab:importance}" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{clcc}" "\n"
        r"\toprule" "\n"
        r"\textbf{Rank} & \textbf{Feature} & \textbf{Importance} & \textbf{Cumulative} \\" "\n"
        r"\midrule" "\n"
    ) + '\n'.join(fi_rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"

    tex = re.sub(
        r'\\begin\{table\}\[!t\]\s*\\renewcommand\{\\arraystretch\}\{1\.2\}\s*'
        r'\\caption\{Top-5 Feature Importances.*?\\end\{table\}',
        new_fi_table,
        tex,
        flags=re.DOTALL,
    )

with open(tex_path, 'w') as f:
    f.write(tex)
print(f"  LaTeX patched → {tex_path}")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("ALL DONE ✅")
print("=" * 68)
print(f"""
Generated files:
  plots/jbhi_model_comparison.png    — Table I equivalent bar chart
  plots/jbhi_predicted_vs_actual.png — Fig. 1 (Predicted vs Actual)
  plots/jbhi_feature_importance.png  — Fig. 2 (Feature Importance)
  plots/jbhi_cv_vs_test.png          — Fig. 3 (Overfitting assessment)
  jbhi_area_metrics.json             — All metrics in JSON
  jbhi_area_model.pkl                — Best trained model + scaler
  jbhi_paper_draft.tex               — Auto-patched with real numbers

Next steps:
  1. Upload jbhi_paper_draft.tex + references.bib + plots/ to Overleaf
  2. Set compiler to pdfLaTeX
  3. Verify the patched numbers match the printed table above
  4. Add supervisor / co-author in the \\author block if needed
""")

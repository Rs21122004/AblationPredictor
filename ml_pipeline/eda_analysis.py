"""
Phase 1: Exploratory Data Analysis (EDA)
Honours Project - Ablation Zone Prediction

Generates all EDA plots and statistics from the feature-engineered dataset.
Outputs saved to plots/ directory.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib  # type: ignore
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import os

# ─── Setup ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# Load data
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'combined_data_engineered.csv'))
print(f"Loaded {len(df)} samples with {len(df.columns)} columns\n")


# ═══════════════════════════════════════════════
# PLOT 1: Correlation Heatmap
# ═══════════════════════════════════════════════
print("1/6  Generating correlation heatmap...")

numeric_cols = [
    'power_watts', 'time_minutes', 'temperature_celsius',
    'length_mm', 'width_mm', 'diameter_mm', 'effective_diameter_mm',
    'energy_joules', 'power_time_product',
    'estimated_volume_mm3', 'sphericity_index'
]
# Filter out extreme power values for better correlation
df_filtered = df[df['power_watts'] <= 200].copy()
corr_df = df_filtered[numeric_cols].dropna(how='all', axis=1)
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            square=True, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap — Ablation Zone Features', fontsize=14, fontweight='bold', pad=15)
plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
plt.close()


# ═══════════════════════════════════════════════
# PLOT 2: Feature Distributions
# ═══════════════════════════════════════════════
print("2/6  Generating feature distributions...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribution of Key Features', fontsize=15, fontweight='bold')

features = [
    ('power_watts', 'Power (W)', 'steelblue'),
    ('time_minutes', 'Time (minutes)', 'coral'),
    ('temperature_celsius', 'Temperature (°C)', 'seagreen'),
    ('effective_diameter_mm', 'Effective Diameter (mm)', 'mediumpurple'),
    ('length_mm', 'Length (mm)', 'goldenrod'),
    ('sphericity_index', 'Sphericity Index', 'indianred')
]

for ax, (col, label, color) in zip(axes.flat, features):
    data = df_filtered[col].dropna()
    if len(data) > 0:
        ax.hist(data, bins=25, color=color, edgecolor='white', alpha=0.8)
        ax.axvline(data.mean(), color='black', linestyle='--', linewidth=1.2, label=f'Mean: {data.mean():.1f}')
        ax.axvline(data.median(), color='red', linestyle=':', linewidth=1.2, label=f'Median: {data.median():.1f}')
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_distributions.png'))
plt.close()


# ═══════════════════════════════════════════════
# PLOT 3: Ablation Zone vs Input Parameters
# ═══════════════════════════════════════════════
print("3/6  Generating ablation zone scatter plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Ablation Zone Dimensions vs Input Parameters', fontsize=15, fontweight='bold')

# Diameter vs Power
ax = axes[0, 0]
mask_d = df_filtered['effective_diameter_mm'].notna()
scatter = ax.scatter(df_filtered.loc[mask_d, 'power_watts'],
                     df_filtered.loc[mask_d, 'effective_diameter_mm'],
                     c=df_filtered.loc[mask_d, 'time_minutes'], cmap='viridis',
                     alpha=0.7, edgecolors='gray', s=50)
ax.set_xlabel('Power (W)')
ax.set_ylabel('Effective Diameter (mm)')
ax.set_title('Diameter vs Power (colored by Time)')
plt.colorbar(scatter, ax=ax, label='Time (min)')

# Diameter vs Time
ax = axes[0, 1]
scatter2 = ax.scatter(df_filtered.loc[mask_d, 'time_minutes'],
                      df_filtered.loc[mask_d, 'effective_diameter_mm'],
                      c=df_filtered.loc[mask_d, 'power_watts'], cmap='plasma',
                      alpha=0.7, edgecolors='gray', s=50)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Effective Diameter (mm)')
ax.set_title('Diameter vs Time (colored by Power)')
plt.colorbar(scatter2, ax=ax, label='Power (W)')

# Length vs Power
ax = axes[1, 0]
mask_l = df_filtered['length_mm'].notna()
scatter3 = ax.scatter(df_filtered.loc[mask_l, 'power_watts'],
                      df_filtered.loc[mask_l, 'length_mm'],
                      c=df_filtered.loc[mask_l, 'time_minutes'], cmap='viridis',
                      alpha=0.7, edgecolors='gray', s=50)
ax.set_xlabel('Power (W)')
ax.set_ylabel('Length (mm)')
ax.set_title('Length vs Power (colored by Time)')
plt.colorbar(scatter3, ax=ax, label='Time (min)')

# Length vs Time
ax = axes[1, 1]
scatter4 = ax.scatter(df_filtered.loc[mask_l, 'time_minutes'],
                      df_filtered.loc[mask_l, 'length_mm'],
                      c=df_filtered.loc[mask_l, 'power_watts'], cmap='plasma',
                      alpha=0.7, edgecolors='gray', s=50)
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Length (mm)')
ax.set_title('Length vs Time (colored by Power)')
plt.colorbar(scatter4, ax=ax, label='Power (W)')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'ablation_vs_inputs.png'))
plt.close()


# ═══════════════════════════════════════════════
# PLOT 4: Antenna Category Distribution
# ═══════════════════════════════════════════════
print("4/6  Generating antenna distribution chart...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Antenna Type Analysis', fontsize=15, fontweight='bold')

# Count plot
ax = axes[0]
cat_counts = df['antenna_category'].value_counts()
bars = ax.barh(cat_counts.index, cat_counts.values, color=sns.color_palette('Set2', len(cat_counts)))
ax.set_xlabel('Number of Samples')
ax.set_title('Samples per Antenna Category')
ax.invert_yaxis()
for bar, count in zip(bars, cat_counts.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(count), va='center', fontsize=9)

# Box plot: diameter by antenna type
ax = axes[1]
top_cats = cat_counts[cat_counts >= 5].index.tolist()
df_top = df_filtered[df_filtered['antenna_category'].isin(top_cats) & df_filtered['effective_diameter_mm'].notna()]
if len(df_top) > 0:
    order = df_top.groupby('antenna_category')['effective_diameter_mm'].median().sort_values(ascending=False).index
    sns.boxplot(data=df_top, y='antenna_category', x='effective_diameter_mm',
                order=order, palette='Set2', ax=ax)
    ax.set_xlabel('Effective Diameter (mm)')
    ax.set_ylabel('')
    ax.set_title('Ablation Diameter by Antenna Type\n(categories with ≥5 samples)')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'antenna_distribution.png'))
plt.close()


# ═══════════════════════════════════════════════
# PLOT 5: Missing Values Summary
# ═══════════════════════════════════════════════
print("5/6  Generating missing values chart...")

fig, ax = plt.subplots(figsize=(12, 6))
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=True)
missing_pct = (missing / len(df) * 100)

colors = ['#2ecc71' if p < 30 else '#f39c12' if p < 60 else '#e74c3c' for p in missing_pct]
bars = ax.barh(missing.index, missing_pct.values, color=colors, edgecolor='white')
ax.set_xlabel('Missing (%)')
ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax.legend()

for bar, pct, count in zip(bars, missing_pct.values, missing.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}% ({count})', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'missing_values.png'))
plt.close()


# ═══════════════════════════════════════════════
# PLOT 6: Experimental vs Simulated Comparison
# ═══════════════════════════════════════════════
print("6/6  Generating data source comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Experimental vs Simulated Data Comparison', fontsize=15, fontweight='bold')

compare_features = [
    ('effective_diameter_mm', 'Effective Diameter (mm)'),
    ('length_mm', 'Length (mm)'),
    ('power_watts', 'Power (W)')
]

for ax, (feat, label) in zip(axes, compare_features):
    for src, color in [('Experimental', '#3498db'), ('Simulated', '#e74c3c')]:
        data = df_filtered[df_filtered['data_source'] == src][feat].dropna()
        if len(data) > 0:
            ax.hist(data, bins=20, alpha=0.6, label=f'{src} (n={len(data)})',
                    color=color, edgecolor='white')
    ax.set_xlabel(label)
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'data_source_comparison.png'))
plt.close()


# ═══════════════════════════════════════════════
# PRINT SUMMARY STATISTICS
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("EDA SUMMARY STATISTICS")
print(f"{'='*60}")

print(f"\nTotal samples: {len(df)}")
print(f"  Experimental: {len(df[df['data_source']=='Experimental'])}")
print(f"  Simulated:    {len(df[df['data_source']=='Simulated'])}")

print(f"\nTarget variable availability:")
print(f"  effective_diameter_mm: {df['effective_diameter_mm'].notna().sum()} ({df['effective_diameter_mm'].notna().mean()*100:.1f}%)")
print(f"  length_mm:             {df['length_mm'].notna().sum()} ({df['length_mm'].notna().mean()*100:.1f}%)")

print(f"\nKey correlations with effective_diameter_mm (filtered ≤200W):")
target_corr = corr_df.corr()['effective_diameter_mm'].drop('effective_diameter_mm').sort_values(key=abs, ascending=False)
for feat, val in target_corr.items():
    print(f"  {feat:<30s}  r = {val:+.3f}")

print(f"\n✅ All 6 plots saved to: {PLOTS_DIR}/")
print("Done!")

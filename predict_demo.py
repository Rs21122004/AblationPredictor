"""
Phase 5: Prediction Demo
Honours Project - Ablation Zone Prediction

A simple script to demonstrate the trained models.
Input: power, time, antenna type → Output: predicted ablation zone dimensions.
"""

import pickle
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load preprocessed data and trained models
with open(os.path.join(SCRIPT_DIR, 'preprocessed_data.pkl'), 'rb') as f:
    preproc = pickle.load(f)

with open(os.path.join(SCRIPT_DIR, 'training_results.pkl'), 'rb') as f:
    results = pickle.load(f)

le = preproc['label_encoder']
scaler_diam = preproc['scaler_diam']
scaler_len = preproc['scaler_len']
feature_names = preproc['feature_names']

# Best models
best_diam_name = max(results['diameter'], key=lambda m: results['diameter'][m]['test_r2'])
best_len_name = max(results['length'], key=lambda m: results['length'][m]['test_r2'])
best_diam_model = results['diameter'][best_diam_name]['model']
best_diam_scaled = results['diameter'][best_diam_name]['use_scaled']
best_len_model = results['length'][best_len_name]['model']
best_len_scaled = results['length'][best_len_name]['use_scaled']

antenna_categories = list(le.classes_)


def predict_ablation(power_watts, time_minutes, antenna_category='Other', is_simulated=0):
    """Predict ablation zone dimensions for given parameters."""

    # Encode antenna
    if antenna_category in antenna_categories:
        antenna_code = le.transform([antenna_category])[0]
    else:
        antenna_code = le.transform(['Other'])[0]

    # Compute engineered features
    energy = power_watts * time_minutes * 60
    pt_product = power_watts * time_minutes
    log_power = np.log1p(power_watts)
    log_time = np.log1p(time_minutes)
    log_energy = np.log1p(energy)
    sqrt_time = np.sqrt(max(time_minutes, 0))

    # Feature vector (same order as training)
    features = np.array([[
        power_watts, time_minutes, energy, pt_product,
        log_power, log_time, log_energy, sqrt_time,
        is_simulated, antenna_code
    ]])

    # Predict diameter
    if best_diam_scaled:
        features_diam = scaler_diam.transform(features)
    else:
        features_diam = features
    pred_diameter = best_diam_model.predict(features_diam)[0]

    # Predict length
    if best_len_scaled:
        features_len = scaler_len.transform(features)
    else:
        features_len = features
    pred_length = best_len_model.predict(features_len)[0]

    # Estimated volume (ellipsoid approximation)
    pred_volume = (4/3) * np.pi * (pred_length/2) * (pred_diameter/2)**2

    return {
        'diameter_mm': round(pred_diameter, 2),
        'length_mm': round(pred_length, 2),
        'estimated_volume_mm3': round(pred_volume, 2),
        'sphericity': round(min(pred_diameter / pred_length, 1.0), 3),
    }


# ═══════════════════════════════════════════════
# INTERACTIVE DEMO
# ═══════════════════════════════════════════════

print("="*60)
print("ABLATION ZONE PREDICTION DEMO")
print("="*60)
print(f"\nBest models used:")
print(f"  Diameter: {best_diam_name} (R² = {results['diameter'][best_diam_name]['test_r2']:.4f})")
print(f"  Length:   {best_len_name} (R² = {results['length'][best_len_name]['test_r2']:.4f})")
print(f"\nAvailable antenna categories:")
for i, cat in enumerate(antenna_categories):
    print(f"  [{i}] {cat}")

# Demo predictions
print(f"\n{'='*60}")
print("EXAMPLE PREDICTIONS")
print(f"{'='*60}")

demo_cases = [
    (50, 5, 'Dual Slot'),
    (50, 10, 'Dual Slot'),
    (100, 5, 'Monopole'),
    (100, 10, 'Monopole'),
    (40, 3, 'Other'),
    (80, 10, 'Triaxial'),
    (20, 5, 'Dipole'),
    (30, 10, 'Cooled Antenna'),
]

print(f"\n{'Power(W)':>10s} {'Time(min)':>10s} {'Antenna':<18s} {'Diam(mm)':>10s} {'Length(mm)':>11s} {'Vol(mm³)':>12s} {'Sphericity':>11s}")
print("─" * 85)

for power, time, antenna in demo_cases:
    pred = predict_ablation(power, time, antenna)
    print(f"{power:>10d} {time:>10d} {antenna:<18s} {pred['diameter_mm']:>10.2f} {pred['length_mm']:>11.2f} {pred['estimated_volume_mm3']:>12.1f} {pred['sphericity']:>11.3f}")


# ═══════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("INTERACTIVE MODE")
print("Type 'quit' to exit")
print(f"{'='*60}")

while True:
    try:
        print()
        power = input("Enter Power (W): ").strip()
        if power.lower() == 'quit':
            break
        power = float(power)

        time = float(input("Enter Time (minutes): ").strip())

        print(f"Antenna categories: {', '.join(f'[{i}]{c}' for i, c in enumerate(antenna_categories))}")
        ant_input = input("Enter antenna category number or name [default: Other]: ").strip()
        if ant_input == '' or ant_input.lower() == 'other':
            antenna = 'Other'
        elif ant_input.isdigit() and int(ant_input) < len(antenna_categories):
            antenna = antenna_categories[int(ant_input)]
        else:
            antenna = ant_input

        pred = predict_ablation(power, time, antenna)
        print(f"\n  📊 PREDICTION:")
        print(f"     Effective Diameter: {pred['diameter_mm']:.2f} mm")
        print(f"     Length:             {pred['length_mm']:.2f} mm")
        print(f"     Est. Volume:        {pred['estimated_volume_mm3']:.1f} mm³")
        print(f"     Sphericity:         {pred['sphericity']:.3f}")

    except (ValueError, EOFError):
        print("  Invalid input. Try again or type 'quit'.")
        break
    except KeyboardInterrupt:
        break

print("\nDone!")

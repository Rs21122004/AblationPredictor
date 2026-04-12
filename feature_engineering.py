"""
Feature Engineering Script for Ablation Zone Prediction
Honours Project - Microwave Ablation Dataset

This script:
1. Reads both raw CSV datasets (Experimental & Simulated)
2. Parses messy text fields into clean numerical features
3. Engineers additional features (energy, power-time interaction, etc.)
4. Outputs two clean, feature-engineered CSV files + a combined dataset
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import re
import os

# ─────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

raw_exp = pd.read_csv(os.path.join(SCRIPT_DIR, "Ablation Zone Model - Experimental data.csv"))
raw_sim = pd.read_csv(os.path.join(SCRIPT_DIR, "Ablation Zone Model2 - Simulated data.csv"))

print(f"Experimental data: {len(raw_exp)} rows, {list(raw_exp.columns)}")
print(f"Simulated data:    {len(raw_sim)} rows, {list(raw_sim.columns)}")


# ─────────────────────────────────────────────
# 2. HELPER FUNCTIONS FOR PARSING
# ─────────────────────────────────────────────

def parse_power(val):
    """Extract power in Watts from strings like '50W', '50 W', '25 kW'."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    # Handle kW
    kw_match = re.search(r'([\d.]+)\s*kW', val, re.IGNORECASE)
    if kw_match:
        return float(kw_match.group(1)) * 1000
    # Handle W
    w_match = re.search(r'([\d.]+)\s*W', val, re.IGNORECASE)
    if w_match:
        return float(w_match.group(1))
    # Try just a number
    num_match = re.search(r'([\d.]+)', val)
    if num_match:
        return float(num_match.group(1))
    return np.nan


def parse_time(val):
    """Extract time in minutes from strings like '2 minutes', '5 minutes', '0.5 minutes'."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    # Match number followed by 'minute(s)' or 'min'
    match = re.search(r'([\d.]+)\s*min', val, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Try just a number
    num_match = re.search(r'([\d.]+)', val)
    if num_match:
        return float(num_match.group(1))
    return np.nan


def parse_temperature(val):
    """Extract temperature in °C from strings like '105°C', '>100 °C', 'NA'."""
    if pd.isna(val) or str(val).strip().upper() == 'NA':
        return np.nan
    val = str(val).strip()
    # Remove > or < signs
    val = val.replace('>', '').replace('<', '').strip()
    # Extract number before °C or C
    match = re.search(r'([\d.]+)\s*°?\s*C', val, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Try just a number
    num_match = re.search(r'([\d.]+)', val)
    if num_match:
        return float(num_match.group(1))
    return np.nan


def extract_antenna_type(ref_text):
    """Extract antenna type from the reference/paper description."""
    if pd.isna(ref_text):
        return 'Unknown'
    ref_text = str(ref_text)

    # Look for "Antenna Type:" or "Antenna type:" pattern
    match = re.search(r'Antenna\s*[Tt]ype\s*:\s*(.+?)(?:\n|$|\r|Antenna\s*dim)', ref_text, re.IGNORECASE)
    if match:
        antenna = match.group(1).strip()
        # Clean up trailing whitespace and special chars
        antenna = re.sub(r'\s+', ' ', antenna).strip()
        # Remove trailing parenthetical details for cleaner grouping
        return antenna

    # Fallback: try to identify from common keywords in the paper title
    text_lower = ref_text.lower()
    if 'dual slot' in text_lower or 'dual-slot' in text_lower:
        return 'Dual Slot'
    elif 'monopole' in text_lower:
        return 'Monopole'
    elif 'triaxial' in text_lower:
        return 'Triaxial'
    elif 'tri-slot' in text_lower or 'tri slot' in text_lower:
        return 'Tri-Slot'
    elif 'single slot' in text_lower or 'single-slot' in text_lower:
        return 'Single Slot'
    elif 'dipole' in text_lower:
        return 'Dipole'
    elif 'helical' in text_lower:
        return 'Helical Dipole'
    elif 'cooled' in text_lower:
        return 'Cooled Antenna'
    elif 'sliding choke' in text_lower:
        return 'Sliding Choke'
    elif 'triple' in text_lower:
        return 'Triple Antenna'
    elif 'floating sleeve' in text_lower or 'floating-sleeve' in text_lower:
        return 'Floating Sleeve'
    elif 'slot' in text_lower:
        return 'Slot Antenna'
    else:
        return 'Other'


def extract_paper_year(ref_text):
    """Extract publication year from reference text."""
    if pd.isna(ref_text):
        return np.nan
    # Find 4-digit year (2000-2029)
    matches = re.findall(r'(20[0-2]\d)', str(ref_text))
    if matches:
        return int(matches[-1])  # Take the last match (usually the year)
    return np.nan


def parse_ablation_dimensions(val):
    """
    Parse the ablation zone parameters field into separate dimensions.
    Returns dict with keys: length_mm, width_mm, diameter_mm, depth_mm, volume_mm3, aspect_ratio
    """
    result = {
        'length_mm': np.nan,
        'width_mm': np.nan,
        'diameter_mm': np.nan,
        'depth_mm': np.nan,
        'volume_mm3': np.nan,
        'aspect_ratio': np.nan,
    }

    if pd.isna(val) or str(val).strip().upper() == 'NA':
        return result

    val = str(val).strip()

    # --- Length ---
    length_match = re.search(r'[Ll]ength\s*[:=]\s*([\d.]+)\s*(?:±\s*[\d.]+\s*)?mm', val)
    if length_match:
        result['length_mm'] = float(length_match.group(1))

    # --- Width ---
    width_match = re.search(r'[Ww]idth(?:\s+d\d)?\s*[:=]\s*([\d.]+)\s*(?:±\s*[\d.]+\s*)?mm', val)
    if width_match:
        result['width_mm'] = float(width_match.group(1))

    # --- Diameter ---
    # Handle various diameter labels
    diam_match = re.search(r'(?:[Dd]iameter|[Dd]iameter\d?|[Ll]ateral\s+diameter|[Ll]ongitudinal\s+[Dd]iameter|[Mm]ax\s+transverse|[Mm]in\s+[Aa]xial)\s*[:=]\s*([\d.]+)\s*(?:±\s*[\d.]+\s*)?(?:mm)?', val)
    if diam_match:
        result['diameter_mm'] = float(diam_match.group(1))
    else:
        # Simple "diameter = XX mm" pattern
        diam_match2 = re.search(r'diameter\s*[:=]\s*([\d.]+)\s*(?:±\s*[\d.]+)?\s*mm', val, re.IGNORECASE)
        if diam_match2:
            result['diameter_mm'] = float(diam_match2.group(1))

    # --- Depth ---
    depth_match = re.search(r'[Dd]epth\s*[:=]\s*([\d.]+)\s*(?:±\s*[\d.]+\s*)?mm', val)
    if depth_match:
        result['depth_mm'] = float(depth_match.group(1))

    # --- Height (treat as length if no length found) ---
    if np.isnan(result['length_mm']):
        height_match = re.search(r'[Hh]eight\s*[:=]\s*([\d.]+)\s*(?:±\s*[\d.]+\s*)?mm', val)
        if height_match:
            result['length_mm'] = float(height_match.group(1))

    # --- Volume ---
    vol_match = re.search(r'([\d,]+)\s*mm[³3²]', val)
    if vol_match:
        result['volume_mm3'] = float(vol_match.group(1).replace(',', ''))
    else:
        vol_match2 = re.search(r'volume\s*[:=]\s*([\d,]+)\s*mm', val, re.IGNORECASE)
        if vol_match2:
            result['volume_mm3'] = float(vol_match2.group(1).replace(',', ''))

    # --- Aspect Ratio ---
    ar_match = re.search(r'[Aa]spect\s*ratio\s*[:=]\s*([\d.]+)', val)
    if ar_match:
        result['aspect_ratio'] = float(ar_match.group(1))

    # --- Radius to diameter ---
    radius_match = re.search(r'radius\s*[:=]?\s*([\d.]+)\s*mm', val, re.IGNORECASE)
    if radius_match and np.isnan(result['diameter_mm']):
        result['diameter_mm'] = float(radius_match.group(1)) * 2

    # --- Sphere of radius ---
    sphere_match = re.search(r'[Ss]phere\s+of\s+radius\s+([\d.]+)\s*mm', val)
    if sphere_match and np.isnan(result['diameter_mm']):
        result['diameter_mm'] = float(sphere_match.group(1)) * 2

    # --- If only a bare number with mm (no label), assign to diameter ---
    if all(np.isnan(v) for v in result.values()):
        bare_match = re.search(r'^([\d.]+)\s*(?:±\s*[\d.]+\s*)?mm', val.strip())
        if bare_match:
            result['diameter_mm'] = float(bare_match.group(1))
        # Also try "XX mm" at end
        bare_match2 = re.search(r'([\d.]+)\s*mm\s*$', val.strip())
        if bare_match2 and np.isnan(result['diameter_mm']):
            result['diameter_mm'] = float(bare_match2.group(1))

    # --- Handle "width = XX mm, depth = YY mm" pattern ---
    wd_match = re.search(r'width\s*=\s*([\d.]+)\s*mm.*?depth\s*=\s*([\d.]+)\s*mm', val, re.IGNORECASE)
    if wd_match:
        result['width_mm'] = float(wd_match.group(1))
        result['depth_mm'] = float(wd_match.group(2))

    return result


def categorize_antenna(antenna_str):
    """Group antenna types into broader categories for better ML performance."""
    if pd.isna(antenna_str):
        return 'Other'

    a = str(antenna_str).lower().strip()

    if 'dual slot' in a or 'dual-slot' in a or 'double slot' in a:
        return 'Dual Slot'
    elif 'single slot' in a or 'single-slot' in a:
        return 'Single Slot'
    elif 'tri-slot' in a or 'tri slot' in a or 'triple' in a:
        return 'Multi Slot/Triple'
    elif 'coaxial half-slot' in a or 'half-slot' in a:
        return 'Coaxial Half-Slot'
    elif 'monopole' in a:
        return 'Monopole'
    elif 'dipole' in a and 'helical' in a:
        return 'Helical Dipole'
    elif 'dipole' in a:
        return 'Dipole'
    elif 'triaxial' in a:
        return 'Triaxial'
    elif 'floating' in a and 'sleeve' in a:
        return 'Floating Sleeve'
    elif 'sicl' in a or 'substrate' in a:
        return 'SICL'
    elif 'cooled' in a or 'water' in a:
        return 'Cooled Antenna'
    elif 'sliding' in a and 'choke' in a:
        return 'Sliding Choke'
    elif 'slot' in a:
        return 'Slot Antenna'
    elif 'meandered' in a or 'flexible' in a:
        return 'Flexible Antenna'
    elif 'directional' in a:
        return 'Directional'
    elif 'mrsa' in a:
        return 'MRSA'
    elif 'omnidirectional' in a:
        return 'Omnidirectional'
    else:
        return 'Other'


# ─────────────────────────────────────────────
# 3. PROCESS EXPERIMENTAL DATA
# ─────────────────────────────────────────────

def process_dataset(df, dataset_label):
    """Process a raw dataset into clean features."""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_label}")
    print(f"{'='*60}")

    records = []

    for idx, row in df.iterrows():
        ref = row.get('PAPERS & REFERENCES', '')
        power_raw = row.get('INPUT POWER', np.nan)
        time_raw = row.get('TIME', np.nan)
        temp_raw = row.get('OUTPUT TEMPERATURE', np.nan)
        ablation_raw = row.get('ABLATION ZONE PARAMETERS', np.nan)

        # Parse each field
        power = parse_power(power_raw)
        time = parse_time(time_raw)
        temperature = parse_temperature(temp_raw)
        antenna_type = extract_antenna_type(ref)
        paper_year = extract_paper_year(ref)
        dims = parse_ablation_dimensions(ablation_raw)

        record = {
            'paper_reference': str(ref).split('\n')[0].strip()[:100] if pd.notna(ref) else '',  # type: ignore
            'paper_year': paper_year,
            'power_watts': power,
            'time_minutes': time,
            'temperature_celsius': temperature,
            'antenna_type_raw': antenna_type,
            'antenna_category': categorize_antenna(antenna_type),
            'data_source': dataset_label,
            **dims
        }
        records.append(record)

    result = pd.DataFrame(records)

    # ─── Engineered Features ───
    # Energy delivered (Power × Time)
    result['energy_joules'] = result['power_watts'] * result['time_minutes'] * 60

    # Power-time interaction
    result['power_time_product'] = result['power_watts'] * result['time_minutes']

    # Log-transformed features (useful for non-linear relationships)
    result['log_power'] = np.log1p(result['power_watts'])
    result['log_time'] = np.log1p(result['time_minutes'])
    result['log_energy'] = np.log1p(result['energy_joules'])

    # Square root of time (ablation growth often follows sqrt relationship)
    result['sqrt_time'] = np.sqrt(result['time_minutes'].clip(lower=0))

    # Unified size metric: use diameter if available, else width
    result['effective_diameter_mm'] = result['diameter_mm'].fillna(result['width_mm'])

    # Ablation volume estimate (if length and diameter/width available)
    has_length = result['length_mm'].notna()
    has_diam = result['effective_diameter_mm'].notna()
    result['estimated_volume_mm3'] = np.nan
    mask = has_length & has_diam
    # Approximate as ellipsoid: V = (4/3) * π * (L/2) * (D/2)^2
    result.loc[mask, 'estimated_volume_mm3'] = (
        (4 / 3) * np.pi *
        (result.loc[mask, 'length_mm'] / 2) *
        (result.loc[mask, 'effective_diameter_mm'] / 2) ** 2
    )
    # Use actual volume if available
    result.loc[result['volume_mm3'].notna(), 'estimated_volume_mm3'] = result.loc[result['volume_mm3'].notna(), 'volume_mm3']

    # Sphericity index (how close to a sphere: 1.0 = perfect sphere)
    result['sphericity_index'] = np.nan
    mask2 = has_length & has_diam
    result.loc[mask2, 'sphericity_index'] = (
        result.loc[mask2, 'effective_diameter_mm'] / result.loc[mask2, 'length_mm']
    ).clip(upper=1.0)

    # Temperature available flag
    result['has_temperature'] = result['temperature_celsius'].notna().astype(int)

    return result


# ─────────────────────────────────────────────
# 4. RUN PROCESSING
# ─────────────────────────────────────────────

df_exp = process_dataset(raw_exp, 'Experimental')
df_sim = process_dataset(raw_sim, 'Simulated')

# Combined dataset
df_combined = pd.concat([df_exp, df_sim], ignore_index=True)

# Add binary flag for data source
df_combined['is_simulated'] = (df_combined['data_source'] == 'Simulated').astype(int)  # type: ignore


# ─────────────────────────────────────────────
# 5. ONE-HOT ENCODE ANTENNA CATEGORIES
# ─────────────────────────────────────────────

# Create one-hot encoded version
antenna_dummies = pd.get_dummies(df_combined['antenna_category'], prefix='antenna')
df_combined_encoded = pd.concat([df_combined, antenna_dummies], axis=1)


# ─────────────────────────────────────────────
# 6. SAVE OUTPUTS
# ─────────────────────────────────────────────

output_dir = SCRIPT_DIR

# Save individual processed datasets
df_exp.to_csv(os.path.join(output_dir, 'experimental_data_engineered.csv'), index=False)
df_sim.to_csv(os.path.join(output_dir, 'simulated_data_engineered.csv'), index=False)

# Save combined dataset (without one-hot encoding)
df_combined.to_csv(os.path.join(output_dir, 'combined_data_engineered.csv'), index=False)

# Save combined dataset WITH one-hot encoding (ML-ready)
df_combined_encoded.to_csv(os.path.join(output_dir, 'combined_data_ml_ready.csv'), index=False)


# ─────────────────────────────────────────────
# 7. PRINT SUMMARY REPORT
# ─────────────────────────────────────────────

print(f"\n{'='*60}")
print("FEATURE ENGINEERING SUMMARY")
print(f"{'='*60}")
print(f"\nExperimental samples: {len(df_exp)}")
print(f"Simulated samples:   {len(df_sim)}")
print(f"Combined samples:    {len(df_combined)}")

print(f"\n--- Columns ({len(df_combined.columns)}) ---")
for col in df_combined.columns:
    non_null = df_combined[col].notna().sum()
    pct = non_null / len(df_combined) * 100
    print(f"  {col:<30s}  {non_null:>4d}/{len(df_combined)}  ({pct:5.1f}%)")

print(f"\n--- Antenna Category Distribution ---")
print(df_combined['antenna_category'].value_counts().to_string())

print(f"\n--- Numerical Feature Statistics ---")
numeric_cols = [
    'power_watts', 'time_minutes', 'temperature_celsius',
    'length_mm', 'width_mm', 'diameter_mm', 'depth_mm',
    'effective_diameter_mm', 'estimated_volume_mm3',
    'energy_joules', 'power_time_product', 'sphericity_index'
]
for col in numeric_cols:
    if col in df_combined.columns:
        s = df_combined[col].dropna()
        if len(s) > 0:
            print(f"\n  {col}:")
            print(f"    count={len(s)}, min={s.min():.2f}, max={s.max():.2f}, "
                  f"mean={s.mean():.2f}, std={s.std():.2f}")

print(f"\n--- Files Saved ---")
print(f"  1. experimental_data_engineered.csv  ({len(df_exp)} rows)")
print(f"  2. simulated_data_engineered.csv     ({len(df_sim)} rows)")
print(f"  3. combined_data_engineered.csv      ({len(df_combined)} rows)")
print(f"  4. combined_data_ml_ready.csv        ({len(df_combined_encoded)} rows, with one-hot encoding)")
print(f"\nDone!")

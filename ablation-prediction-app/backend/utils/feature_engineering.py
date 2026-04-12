"""
Feature engineering utilities for the Ablation Zone Prediction System.

Replicates the exact feature transformations used during model training
(from feature_engineering.py and data_preprocessing.py) to ensure
prediction-time features match training-time features exactly.

Feature order: [power_watts, time_minutes, energy_joules, power_time_product,
                log_power, log_time, log_energy, sqrt_time, is_simulated, antenna_encoded]
"""

import numpy as np
from typing import Tuple


def compute_derived_features(
    power_watts: float,
    time_minutes: float,
    antenna_code: int,
    is_simulated: int = 0
) -> np.ndarray:
    """
    Compute the full feature vector from raw input parameters.
    
    This mirrors the feature engineering pipeline used in training:
    - energy_joules = power × time × 60
    - power_time_product = power × time
    - log_power = log1p(power)
    - log_time = log1p(time)
    - log_energy = log1p(energy)
    - sqrt_time = sqrt(time)
    
    Args:
        power_watts: Input power in Watts
        time_minutes: Treatment duration in minutes
        antenna_code: Integer-encoded antenna category
        is_simulated: 0 for experimental, 1 for simulated (default: 0)
    
    Returns:
        1D numpy array with 10 features in the correct order
    """
    energy_joules = power_watts * time_minutes * 60
    power_time_product = power_watts * time_minutes
    log_power = np.log1p(power_watts)
    log_time = np.log1p(time_minutes)
    log_energy = np.log1p(energy_joules)
    sqrt_time = np.sqrt(max(time_minutes, 0))

    return np.array([
        power_watts,
        time_minutes,
        energy_joules,
        power_time_product,
        log_power,
        log_time,
        log_energy,
        sqrt_time,
        is_simulated,
        antenna_code
    ])


def compute_ablation_volume(diameter_mm: float, length_mm: float) -> float:
    """
    Estimate ablation zone volume using ellipsoid approximation.
    V = (4/3) × π × (L/2) × (D/2)²
    
    This matches the volume calculation used throughout the project.
    """
    return (4 / 3) * np.pi * (length_mm / 2) * (diameter_mm / 2) ** 2


def compute_sphericity(diameter_mm: float, length_mm: float) -> float:
    """
    Compute sphericity index (how close to a sphere).
    Sphericity = min(diameter / length, 1.0)
    1.0 = perfect sphere
    """
    if length_mm <= 0:
        return 0.0
    return min(diameter_mm / length_mm, 1.0)

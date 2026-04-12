"""
Prediction Service — Core inference logic.

Orchestrates feature engineering, model selection, scaling, and prediction.
Supports single predictions, model comparison, and batch predictions.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from services.model_loader import (
    get_model,
    get_best_model_name,
    get_all_model_names,
    get_label_encoder,
    get_scaler,
    get_model_metrics,
    get_antenna_categories,
)
from utils.feature_engineering import (
    compute_derived_features,
    compute_ablation_volume,
    compute_sphericity,
)

logger = logging.getLogger(__name__)


def _encode_antenna(antenna_type: str) -> int:
    """Encode antenna type string to integer using the fitted LabelEncoder."""
    le = get_label_encoder()
    categories = get_antenna_categories()

    if antenna_type in categories:
        return int(le.transform([antenna_type])[0])
    else:
        logger.warning(
            f"Unknown antenna type '{antenna_type}', defaulting to 'Other'"
        )
        return int(le.transform(["Other"])[0])


def _predict_single_target(
    features: np.ndarray,
    target: str,
    model_name: Optional[str] = None,
) -> Tuple[float, str, dict]:
    """
    Run prediction for a single target (diameter or length).
    
    Args:
        features: 1D feature array (10 features)
        target: 'diameter' or 'length'
        model_name: Specific model name, or None for best model
    
    Returns:
        Tuple of (predicted_value, model_name_used, model_metrics)
    """
    if model_name is None:
        model_name = get_best_model_name(target)

    model_info = get_model(target, model_name)
    model = model_info["model"]
    use_scaled = model_info["use_scaled"]

    # Reshape to 2D for sklearn
    X = features.reshape(1, -1)

    # Apply scaling if the model requires it
    if use_scaled:
        scaler_key = "diam" if target == "diameter" else "len"
        scaler = get_scaler(scaler_key)
        X = scaler.transform(X)

    prediction = float(model.predict(X)[0])
    metrics = get_model_metrics(target, model_name)

    return prediction, model_name, metrics


def predict(
    power: float,
    time: float,
    antenna_type: str = "Other",
    model_name: Optional[str] = None,
) -> dict:
    """
    Generate ablation zone predictions for given treatment parameters.
    
    Uses the best (or specified) model for each target. Returns predicted
    diameter, length, volume, and sphericity along with model metadata.
    
    Args:
        power: Input power in Watts
        time: Treatment duration in minutes
        antenna_type: Antenna category string
        model_name: Optional specific model name (applied to both targets)
    
    Returns:
        Dict with prediction results, model info, and uncertainty estimates
    """
    antenna_code = _encode_antenna(antenna_type)
    features = compute_derived_features(power, time, antenna_code)

    # Predict diameter
    pred_diam, diam_model_name, diam_metrics = _predict_single_target(
        features, "diameter", model_name
    )

    # Predict length
    pred_len, len_model_name, len_metrics = _predict_single_target(
        features, "length", model_name
    )

    # Ensure non-negative predictions
    pred_diam = max(pred_diam, 0.1)
    pred_len = max(pred_len, 0.1)

    # Compute derived outputs
    volume = compute_ablation_volume(pred_diam, pred_len)
    sphericity = compute_sphericity(pred_diam, pred_len)

    # Compute uncertainty via ensemble spread
    uncertainty = _compute_uncertainty(features)

    return {
        "prediction": {
            "diameter_mm": round(pred_diam, 2),
            "length_mm": round(pred_len, 2),
            "estimated_volume_mm3": round(volume, 2),
            "sphericity": round(sphericity, 3),
        },
        "diameter_model": diam_metrics,
        "length_model": len_metrics,
        "input_parameters": {
            "power_watts": power,
            "time_minutes": time,
            "antenna_type": antenna_type,
        },
        "uncertainty": uncertainty,
    }


def predict_compare(
    power: float,
    time: float,
    antenna_type: str = "Other",
) -> dict:
    """
    Generate predictions from ALL available models for comparison.
    
    Returns individual results per model plus ensemble statistics.
    """
    antenna_code = _encode_antenna(antenna_type)
    features = compute_derived_features(power, time, antenna_code)

    results = []
    all_diams = []
    all_lens = []

    # Get all unique model names (same set for diameter and length)
    model_names = get_all_model_names("diameter")

    for name in model_names:
        try:
            pred_diam, _, diam_metrics = _predict_single_target(
                features, "diameter", name
            )
            pred_len, _, len_metrics = _predict_single_target(
                features, "length", name
            )

            pred_diam = max(pred_diam, 0.1)
            pred_len = max(pred_len, 0.1)
            volume = compute_ablation_volume(pred_diam, pred_len)
            sphericity = compute_sphericity(pred_diam, pred_len)

            results.append({
                "model_name": name,
                "prediction": {
                    "diameter_mm": round(pred_diam, 2),
                    "length_mm": round(pred_len, 2),
                    "estimated_volume_mm3": round(volume, 2),
                    "sphericity": round(sphericity, 3),
                },
                "metrics": {
                    **diam_metrics,
                    "target": "diameter",
                    "length_r2": len_metrics["test_r2"],
                },
            })

            all_diams.append(pred_diam)
            all_lens.append(pred_len)

        except Exception as e:
            logger.error(f"Error predicting with model '{name}': {e}")
            continue

    # Ensemble statistics (mean ± std across all models)
    if all_diams and all_lens:
        ens_diam = float(np.mean(all_diams))
        ens_len = float(np.mean(all_lens))
        ens_volume = compute_ablation_volume(ens_diam, ens_len)
        ens_sphericity = compute_sphericity(ens_diam, ens_len)

        ensemble_prediction = {
            "diameter_mm": round(ens_diam, 2),
            "length_mm": round(ens_len, 2),
            "estimated_volume_mm3": round(ens_volume, 2),
            "sphericity": round(ens_sphericity, 3),
        }
        ensemble_std = {
            "diameter_std": round(float(np.std(all_diams)), 2),
            "length_std": round(float(np.std(all_lens)), 2),
        }
    else:
        ensemble_prediction = {
            "diameter_mm": 0, "length_mm": 0,
            "estimated_volume_mm3": 0, "sphericity": 0
        }
        ensemble_std = {"diameter_std": 0, "length_std": 0}

    return {
        "results": results,
        "ensemble_prediction": ensemble_prediction,
        "ensemble_std": ensemble_std,
        "input_parameters": {
            "power_watts": power,
            "time_minutes": time,
            "antenna_type": antenna_type,
        },
    }


def predict_batch(rows: List[dict]) -> dict:
    """
    Run predictions for a batch of input rows.
    
    Args:
        rows: List of dicts with 'power', 'time', 'antenna_type' keys
    
    Returns:
        Dict with predictions list, success/failure counts
    """
    predictions = []
    successful = 0
    failed = 0

    for i, row in enumerate(rows):
        try:
            power = float(row.get("power", 0))
            time = float(row.get("time", 0))
            antenna = str(row.get("antenna_type", "Other"))

            if power <= 0 or time <= 0:
                raise ValueError("Power and time must be positive")

            result = predict(power, time, antenna)
            predictions.append({
                "row": i + 1,
                "status": "success",
                "input": {"power": power, "time": time, "antenna_type": antenna},
                **result["prediction"],
            })
            successful += 1

        except Exception as e:
            predictions.append({
                "row": i + 1,
                "status": "error",
                "error": str(e),
                "input": row,
            })
            failed += 1

    return {
        "predictions": predictions,
        "total_rows": len(rows),
        "successful": successful,
        "failed": failed,
    }


def _compute_uncertainty(features: np.ndarray) -> dict:
    """
    Compute prediction uncertainty as the standard deviation
    across all available models (ensemble disagreement).
    
    Higher std → more uncertainty → models disagree more.
    """
    diams = []
    lens = []

    for name in get_all_model_names("diameter"):
        try:
            pred, _, _ = _predict_single_target(features, "diameter", name)
            diams.append(pred)
        except Exception:
            pass

    for name in get_all_model_names("length"):
        try:
            pred, _, _ = _predict_single_target(features, "length", name)
            lens.append(pred)
        except Exception:
            pass

    return {
        "diameter_std_mm": round(float(np.std(diams)), 2) if diams else None,
        "length_std_mm": round(float(np.std(lens)), 2) if lens else None,
        "models_used": len(diams),
    }

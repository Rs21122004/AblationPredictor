"""
Model Loader Service — Loads and manages trained ML models.

Handles loading of .pkl files at startup, provides a registry of all
available models, and exposes scalers/encoders for inference.
"""

import pickle
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ─── Global model registry (populated at startup) ───
_registry: Dict[str, Any] = {}
_is_loaded: bool = False


def get_models_dir() -> str:
    """Get the path to the models directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def load_models() -> None:
    """
    Load all trained models and preprocessing artifacts from .pkl files.
    Called once at application startup.
    
    Loads:
    - training_results.pkl: 6 models × 2 targets (diameter, length)
    - preprocessed_data.pkl: scalers, label encoder, feature names
    """
    global _registry, _is_loaded

    models_dir = get_models_dir()
    training_path = os.path.join(models_dir, "training_results.pkl")
    preproc_path = os.path.join(models_dir, "preprocessed_data.pkl")

    if not os.path.exists(training_path):
        raise FileNotFoundError(f"Training results not found: {training_path}")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessed data not found: {preproc_path}")

    # Load training results (all models + metrics)
    with open(training_path, "rb") as f:
        training_results = pickle.load(f)

    # Load preprocessing artifacts (scalers, encoder, feature names)
    with open(preproc_path, "rb") as f:
        preproc_data = pickle.load(f)

    _registry = {
        "training_results": training_results,
        "label_encoder": preproc_data["label_encoder"],
        "scaler_diam": preproc_data["scaler_diam"],
        "scaler_len": preproc_data["scaler_len"],
        "feature_names": preproc_data["feature_names"],
        "antenna_categories": list(preproc_data["label_encoder"].classes_),
    }

    _is_loaded = True
    model_count = len(training_results.get("diameter", {})) + len(
        training_results.get("length", {})
    )
    logger.info(f"Loaded {model_count} models from {models_dir}")
    logger.info(f"Features: {_registry['feature_names']}")
    logger.info(f"Antenna categories: {_registry['antenna_categories']}")


def is_loaded() -> bool:
    """Check if models have been loaded."""
    return _is_loaded


def get_registry() -> Dict[str, Any]:
    """Get the full model registry."""
    if not _is_loaded:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    return _registry


def get_training_results() -> Dict[str, Any]:
    """Get the training results dictionary."""
    return get_registry()["training_results"]


def get_label_encoder():
    """Get the fitted LabelEncoder for antenna categories."""
    return get_registry()["label_encoder"]


def get_scaler(target: str):
    """Get the fitted StandardScaler for a target ('diam' or 'len')."""
    key = f"scaler_{target}"
    return get_registry()[key]


def get_feature_names():
    """Get the list of feature names in correct order."""
    return get_registry()["feature_names"]


def get_antenna_categories():
    """Get the list of valid antenna category strings."""
    return get_registry()["antenna_categories"]


def get_model(target: str, model_name: str):
    """
    Get a specific trained model.
    
    Args:
        target: 'diameter' or 'length'
        model_name: Name of the model (e.g., 'Random Forest')
    
    Returns:
        Dict with 'model', 'best_params', 'test_r2', etc.
    """
    results = get_training_results()
    if target not in results:
        raise ValueError(f"Unknown target: {target}. Use 'diameter' or 'length'.")
    if model_name not in results[target]:
        available = list(results[target].keys())
        raise ValueError(
            f"Unknown model '{model_name}' for target '{target}'. "
            f"Available: {available}"
        )
    return results[target][model_name]


def get_best_model_name(target: str) -> str:
    """Get the name of the best-performing model for a target (by test R²)."""
    results = get_training_results()
    if target not in results:
        raise ValueError(f"Unknown target: {target}")
    return max(results[target], key=lambda m: results[target][m]["test_r2"])


def get_all_model_names(target: str):
    """Get all model names for a given target."""
    results = get_training_results()
    if target not in results:
        raise ValueError(f"Unknown target: {target}")
    return list(results[target].keys())


def get_model_metrics(target: str, model_name: str) -> dict:
    """Get performance metrics for a specific model."""
    info = get_model(target, model_name)
    return {
        "name": model_name,
        "target": target,
        "test_r2": round(info["test_r2"], 4),
        "test_mae": round(info["test_mae"], 2),
        "test_rmse": round(info["test_rmse"], 2),
        "test_mape": round(info["test_mape"], 1),
    }

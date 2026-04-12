"""
Health route — Service status and model information endpoints.
"""

from fastapi import APIRouter
from services import model_loader
from schemas.prediction import HealthResponse, ModelListResponse, ModelInfo

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Service health check.
    Returns model loading status and available model information.
    """
    loaded = model_loader.is_loaded()

    if not loaded:
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
            available_models=[],
            model_count=0,
            feature_count=0,
            antenna_types=[],
        )

    results = model_loader.get_training_results()
    all_models = list(results.get("diameter", {}).keys())

    return HealthResponse(
        status="healthy",
        models_loaded=True,
        available_models=all_models,
        model_count=len(all_models),
        feature_count=len(model_loader.get_feature_names()),
        antenna_types=model_loader.get_antenna_categories(),
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """
    List all available models with their performance metrics.
    Returns separate listings for diameter and length prediction models.
    """
    results = model_loader.get_training_results()

    diameter_models = []
    for name in results.get("diameter", {}):
        metrics = model_loader.get_model_metrics("diameter", name)
        diameter_models.append(ModelInfo(**metrics))

    length_models = []
    for name in results.get("length", {}):
        metrics = model_loader.get_model_metrics("length", name)
        length_models.append(ModelInfo(**metrics))

    # Sort by R² descending
    diameter_models.sort(key=lambda m: m.test_r2, reverse=True)
    length_models.sort(key=lambda m: m.test_r2, reverse=True)

    return ModelListResponse(
        diameter_models=diameter_models,
        length_models=length_models,
        best_diameter_model=model_loader.get_best_model_name("diameter"),
        best_length_model=model_loader.get_best_model_name("length"),
    )

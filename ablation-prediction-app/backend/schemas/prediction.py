"""
Pydantic schemas for prediction request/response validation.
Ensures strict input validation and clean API contracts.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


# ─── Available antenna types (from trained LabelEncoder) ───
class AntennaType(str, Enum):
    COAXIAL_HALF_SLOT = "Coaxial Half-Slot"
    COOLED_ANTENNA = "Cooled Antenna"
    DIPOLE = "Dipole"
    DIRECTIONAL = "Directional"
    DUAL_SLOT = "Dual Slot"
    FLOATING_SLEEVE = "Floating Sleeve"
    HELICAL_DIPOLE = "Helical Dipole"
    MRSA = "MRSA"
    MONOPOLE = "Monopole"
    MULTI_SLOT_TRIPLE = "Multi Slot/Triple"
    OTHER = "Other"
    SINGLE_SLOT = "Single Slot"
    SLIDING_CHOKE = "Sliding Choke"
    SLOT_ANTENNA = "Slot Antenna"
    TRIAXIAL = "Triaxial"


# ─── Request Schemas ───

class PredictionRequest(BaseModel):
    """Input parameters for ablation zone prediction."""
    power: float = Field(
        ..., gt=0, le=200,
        description="Input power in Watts (1–200W)"
    )
    time: float = Field(
        ..., gt=0, le=60,
        description="Treatment time in minutes (0–60 min)"
    )
    antenna_type: AntennaType = Field(
        default=AntennaType.OTHER,
        description="Type of microwave antenna used"
    )
    # Optional informational fields (not used by models directly)
    frequency: Optional[float] = Field(
        None, description="Operating frequency in GHz (informational only)"
    )
    tissue_type: Optional[str] = Field(
        None, description="Target tissue type (informational only)"
    )
    model_name: Optional[str] = Field(
        None, description="Specific model to use. If null, uses the best model."
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "power": 50,
            "time": 5,
            "antenna_type": "Dual Slot",
            "frequency": 2.45,
            "tissue_type": "liver"
        }]
    }}


class BatchPredictionRow(BaseModel):
    """A single row in a batch prediction (used internally after CSV parsing)."""
    power: float = Field(..., gt=0, le=200)
    time: float = Field(..., gt=0, le=60)
    antenna_type: str = Field(default="Other")


# ─── Response Schemas ───

class PredictionResult(BaseModel):
    """Single prediction output."""
    diameter_mm: float = Field(..., description="Predicted effective diameter (mm)")
    length_mm: float = Field(..., description="Predicted ablation length (mm)")
    estimated_volume_mm3: float = Field(..., description="Estimated ablation volume (mm³)")
    sphericity: float = Field(..., description="Sphericity index (0–1, 1=perfect sphere)")


class ModelInfo(BaseModel):
    """Metadata about a model used for prediction."""
    name: str
    target: str
    test_r2: float
    test_mae: float
    test_rmse: float
    test_mape: float


class PredictionResponse(BaseModel):
    """Full response for a single prediction request."""
    prediction: PredictionResult
    diameter_model: ModelInfo
    length_model: ModelInfo
    input_parameters: dict
    uncertainty: Optional[dict] = Field(
        None, description="Prediction uncertainty from multi-model ensemble"
    )


class ModelComparisonResult(BaseModel):
    """Prediction result from a single model (used in comparison endpoint)."""
    model_name: str
    prediction: PredictionResult
    metrics: ModelInfo


class ComparisonResponse(BaseModel):
    """Response for model comparison endpoint."""
    results: List[ModelComparisonResult]
    ensemble_prediction: PredictionResult
    ensemble_std: dict
    input_parameters: dict


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction endpoint."""
    predictions: List[dict]
    total_rows: int
    successful: int
    failed: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    available_models: List[str]
    model_count: int
    feature_count: int
    antenna_types: List[str]


class ModelListResponse(BaseModel):
    """Detailed model listing response."""
    diameter_models: List[ModelInfo]
    length_models: List[ModelInfo]
    best_diameter_model: str
    best_length_model: str

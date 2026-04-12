"""
Prediction routes — Single, comparison, and batch prediction endpoints.
"""

import csv
import io
from fastapi import APIRouter, HTTPException, UploadFile, File
from schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    ComparisonResponse,
    BatchPredictionResponse,
)
from services import prediction_service

router = APIRouter(prefix="/api", tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single ablation zone prediction.
    
    Accepts treatment parameters and returns predicted ablation dimensions
    using the best (or specified) model.
    
    **Example request:**
    ```json
    {
        "power": 50,
        "time": 5,
        "antenna_type": "Dual Slot"
    }
    ```
    """
    try:
        result = prediction_service.predict(
            power=request.power,
            time=request.time,
            antenna_type=request.antenna_type.value,
            model_name=request.model_name,
        )
        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/compare", response_model=ComparisonResponse)
async def predict_compare(request: PredictionRequest):
    """
    Compare predictions across all available models.
    
    Returns individual predictions from each model plus ensemble
    statistics (mean and standard deviation).
    """
    try:
        result = prediction_service.predict_compare(
            power=request.power,
            time=request.time,
            antenna_type=request.antenna_type.value,
        )
        return ComparisonResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """
    Batch prediction from CSV upload.
    
    CSV must contain columns: `power`, `time`, and optionally `antenna_type`.
    Returns predictions for each row along with success/failure counts.
    
    **Example CSV:**
    ```
    power,time,antenna_type
    50,5,Dual Slot
    100,10,Monopole
    ```
    """
    # Validate file type
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV file (.csv extension)"
        )

    try:
        # Read and parse CSV
        content = await file.read()
        text = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))

        rows = []
        for row in reader:
            rows.append(row)

        if not rows:
            raise HTTPException(
                status_code=400,
                detail="CSV file is empty or has no data rows"
            )

        if len(rows) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Batch size limited to 1000 rows"
            )

        # Validate required columns
        required_cols = {"power", "time"}
        available_cols = set(rows[0].keys())
        missing = required_cols - available_cols
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}. "
                       f"Found: {available_cols}"
            )

        # Run batch prediction
        result = prediction_service.predict_batch(rows)
        return BatchPredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

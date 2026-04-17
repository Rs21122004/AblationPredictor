"""Prediction routes with persistence, auth-aware logging, and batch jobs."""

import csv
import io
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user, get_optional_user
from database.connection import get_db
from database.models import BatchJob, BatchJobStatus, PredictionLog, User
from schemas.prediction import (
    BatchPredictionResponse,
    ComparisonResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
    PredictionRequest,
    PredictionResponse,
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/secure", response_model=PredictionResponse)
async def predict_and_log(
    request: PredictionRequest,
    user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    started_at = time.perf_counter()
    result = prediction_service.predict(
        power=request.power,
        time=request.time,
        antenna_type=request.antenna_type.value,
        model_name=request.model_name,
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    log = PredictionLog(
        user_id=user.id if user else None,
        power=request.power,
        time=request.time,
        antenna_type=request.antenna_type.value,
        predicted_diameter=result["prediction"]["diameter_mm"],
        predicted_length=result["prediction"]["length_mm"],
        predicted_volume=result["prediction"]["estimated_volume_mm3"],
        model_used=result["diameter_model"]["name"],
        response_time_ms=elapsed_ms,
    )
    db.add(log)
    await db.commit()
    return PredictionResponse(**result)


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

        result = prediction_service.predict_batch(rows)
        return BatchPredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/batch/start")
async def start_batch_job(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _ = user
    content = await file.read()
    text = content.decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    job = BatchJob(status=BatchJobStatus.PROCESSING, total_rows=len(rows), input_file_url=file.filename)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    try:
        result = prediction_service.predict_batch(rows)
        job.successful = result["successful"]
        job.failed = result["failed"]
        job.status = BatchJobStatus.COMPLETED
    except Exception:
        job.status = BatchJobStatus.FAILED
    job.completed_at = datetime.now(timezone.utc)
    await db.commit()
    return {"job_id": job.id, "status": job.status}


@router.get("/batch/{job_id}")
async def get_batch_job(job_id: int, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    _ = user
    job = (await db.execute(select(BatchJob).where(BatchJob.id == job_id))).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "total_rows": job.total_rows,
        "successful": job.successful,
        "failed": job.failed,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


@router.get("/predictions/history", response_model=PredictionHistoryResponse)
async def prediction_history(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    offset = (page - 1) * page_size
    total = (
        await db.execute(select(func.count(PredictionLog.id)).where(PredictionLog.user_id == user.id))
    ).scalar_one()
    logs = (
        await db.execute(
            select(PredictionLog)
            .where(PredictionLog.user_id == user.id)
            .order_by(PredictionLog.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
    ).scalars().all()
    items = [
        PredictionHistoryItem(
            id=log.id,
            created_at=log.created_at.isoformat(),
            power=log.power,
            time=log.time,
            antenna_type=log.antenna_type,
            predicted_diameter=log.predicted_diameter,
            predicted_length=log.predicted_length,
            predicted_volume=log.predicted_volume,
            model_used=log.model_used,
            response_time_ms=log.response_time_ms,
        )
        for log in logs
    ]
    return PredictionHistoryResponse(page=page, page_size=page_size, total=total, items=items)

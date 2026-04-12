"""
Ablation Zone Prediction System — FastAPI Application

A clinical decision-support tool that serves trained ML models for
predicting tumor ablation zone dimensions from treatment parameters.

Models trained on microwave ablation experimental + simulated data.
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.health import router as health_router
from routes.predict import router as predict_router
from services.model_loader import load_models

# ─── Logging configuration ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Application lifespan (startup/shutdown) ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models at startup, cleanup on shutdown."""
    logger.info("Starting Ablation Zone Prediction API...")
    try:
        load_models()
        logger.info("All models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    yield
    logger.info("Shutting down Ablation Zone Prediction API.")


# ─── Create FastAPI app ───
app = FastAPI(
    title="Ablation Zone Prediction API",
    description=(
        "Clinical decision-support API for predicting tumor ablation zone "
        "dimensions (diameter, length, volume) from microwave ablation "
        "treatment parameters using trained ML models."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS middleware ───
# Allow frontend origins (dev + production)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register routers ───
app.include_router(health_router)
app.include_router(predict_router)


@app.get("/", tags=["Root"])
async def root():
    """API root — redirect to docs."""
    return {
        "message": "Ablation Zone Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }

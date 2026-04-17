"""
Ablation Zone Prediction System — FastAPI Application

A clinical decision-support tool that serves trained ML models for
predicting tumor ablation zone dimensions from treatment parameters.

Models trained on microwave ablation experimental + simulated data.
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse

from database.connection import engine
from database.models import Base
from middleware.metrics import MetricsMiddleware, metrics_response
from middleware.rate_limiter import limiter
from middleware.request_logger import RequestLoggerMiddleware
from routes.auth import router as auth_router
from routes.health import router as health_router
from routes.predict import router as predict_router
from routes.websocket import router as websocket_router
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
        # async with engine.begin() as conn:
        #     await conn.run_sync(Base.metadata.create_all)
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
app.state.limiter = limiter

# ─── CORS middleware ───
# Allow frontend origins (dev + production)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestLoggerMiddleware)
app.add_middleware(MetricsMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

# ─── Register routers ───
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(auth_router)
app.include_router(websocket_router)


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return metrics_response()


@app.get("/", tags=["Root"])
async def root():
    """API root — redirect to docs."""
    return {
        "message": "Ablation Zone Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }

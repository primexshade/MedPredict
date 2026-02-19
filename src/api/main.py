"""
src/api/main.py — FastAPI application entry point.

Configures middleware, mounts all routers, and sets up the lifespan
context for ML model loading at startup.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.routers import auth, patients, predict, analytics, reports
from src.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan: runs at startup and shutdown.
    Preloads ML models into memory to avoid cold-start latency on first request.
    """
    logger.info("event", message="Starting Disease Prediction API", env=settings.environment)

    # Preload prediction engine (lazy model loading from MLflow registry)
    from src.api.routers.predict import prediction_engine
    await prediction_engine.preload_models()

    logger.info("event", message="All models loaded — API ready")
    yield
    logger.info("event", message="Shutting down API")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Disease Prediction & Health Risk Analysis API",
        description=(
            "Production-grade API for multi-disease risk prediction, "
            "patient clustering, association rule mining, and explainable AI outputs."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in settings.cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Routers ───────────────────────────────────────────────────────────────
    API_PREFIX = "/api/v1"
    app.include_router(auth.router,      prefix=f"{API_PREFIX}/auth",      tags=["Authentication"])
    app.include_router(patients.router,  prefix=f"{API_PREFIX}/patients",  tags=["Patients"])
    app.include_router(predict.router,   prefix=f"{API_PREFIX}/predict",   tags=["Prediction"])
    app.include_router(analytics.router, prefix=f"{API_PREFIX}/analytics", tags=["Analytics"])
    app.include_router(reports.router,   prefix=f"{API_PREFIX}/reports",   tags=["Reports"])

    # ── Health Check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check() -> dict:
        return {"status": "healthy", "version": "1.0.0", "environment": settings.environment}

    return app


app = create_app()

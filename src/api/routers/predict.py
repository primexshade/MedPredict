"""
src/api/routers/predict.py — Prediction API endpoints.

Implements disease-specific prediction routes with:
- Redis caching (cache key = hash of disease + input features)
- SHAP explanations on every prediction
- Risk scoring with composite formula
- Async batch prediction via Celery
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.deps import get_current_user, get_redis
from src.config import get_settings
from src.scoring.risk_scorer import RiskScorer

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# ─── Request / Response Schemas ──────────────────────────────────────────────

class HeartDiseaseInput(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Patient age in years")
    sex: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dL)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar >120mg/dL")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: float = Field(..., ge=40, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression (exercise vs rest)")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=3, description="Major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=2, description="Thalassemia (0=normal, 1=fixed, 2=reversible)")

    patient_id: str | None = Field(None, description="Optional patient ID for history lookup")


class DiabetesInput(BaseModel):
    pregnancies: float = Field(..., ge=0, le=20)
    glucose: float = Field(..., ge=0, le=400, description="Plasma glucose (mg/dL)")
    bloodpressure: float = Field(..., ge=0, le=200, description="Diastolic BP (mmHg)")
    skinthickness: float = Field(..., ge=0, le=100, description="Triceps skinfold (mm)")
    insulin: float = Field(..., ge=0, le=900, description="2-hour serum insulin (μU/mL)")
    bmi: float = Field(..., ge=0, le=80, description="Body Mass Index (kg/m²)")
    diabetespedigreefunction: float = Field(..., ge=0, le=3)
    age: float = Field(..., ge=1, le=120)
    patient_id: str | None = None


class FeatureContributionOut(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str
    rank: int


class PredictionResponse(BaseModel):
    patient_id: str | None
    disease: str
    risk_score: float = Field(..., description="Composite risk score ∈ [0, 1]")
    calibrated_probability: float
    risk_category: str
    confidence_interval: tuple[float, float]
    velocity: float | None = Field(None, description="Risk change since last visit")
    top_features: list[FeatureContributionOut]
    plain_english_summary: str
    clinical_action: dict[str, str]
    model_version: str
    cached: bool = False


# ─── Prediction Engine (singleton) ────────────────────────────────────────────

class PredictionEngine:
    """
    Singleton that manages loaded ML models and explainers.
    Models are loaded once at startup from the MLflow registry.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._explainers: dict[str, Any] = {}
        self._feature_names: dict[str, list[str]] = {}

    async def preload_models(self) -> None:
        """Preload all models from MLflow registry at startup."""
        import mlflow.sklearn

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        for disease, model_name in settings.registered_models.items():
            try:
                model_uri = f"models:/{model_name}/Staging"
                model = mlflow.sklearn.load_model(model_uri)
                self._models[disease] = model
                logger.info("Loaded model for disease: %s", disease)
            except Exception as e:
                logger.warning("Could not load model for %s: %s", disease, e)

    def get_model(self, disease: str) -> Any:
        model = self._models.get(disease)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model for '{disease}' is not loaded. Training may be required.",
            )
        return model

    def predict(self, disease: str, df: pd.DataFrame) -> tuple[float, list]:
        """Returns (calibrated_probability, shap_contributions)."""
        model = self.get_model(disease)
        prob = float(model.predict_proba(df)[:, 1][0])

        # SHAP explanations (best-effort — non-blocking if explainer not loaded)
        contributions = []
        explainer = self._explainers.get(disease)
        if explainer and len(df.columns) == len(self._feature_names.get(disease, [])):
            try:
                result = explainer.explain(df)
                contributions = result.contributions
            except Exception as e:
                logger.warning("SHAP explanation failed for %s: %s", disease, e)

        return prob, contributions


prediction_engine = PredictionEngine()


# ─── Cache Helper ─────────────────────────────────────────────────────────────

def _cache_key(disease: str, payload: dict) -> str:
    payload_str = json.dumps({k: v for k, v in sorted(payload.items()) if k != "patient_id"})
    return f"predict:{disease}:{hashlib.sha256(payload_str.encode()).hexdigest()[:16]}"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/heart", response_model=PredictionResponse)
@limiter.limit(settings.rate_limit_prediction)
async def predict_heart_disease(
    request: Request,
    payload: HeartDiseaseInput,
    current_user: Annotated[Any, Depends(get_current_user)],
    redis=Depends(get_redis),
) -> PredictionResponse:
    """
    Predict heart disease risk for a patient.

    Returns calibrated probability, risk category (LOW/MODERATE/HIGH/CRITICAL),
    95% confidence interval, and top SHAP feature contributions.
    """
    return await _run_prediction("heart", payload.model_dump(), redis)


@router.post("/diabetes", response_model=PredictionResponse)
@limiter.limit(settings.rate_limit_prediction)
async def predict_diabetes(
    request: Request,
    payload: DiabetesInput,
    current_user: Annotated[Any, Depends(get_current_user)],
    redis=Depends(get_redis),
) -> PredictionResponse:
    """Predict Type 2 diabetes risk."""
    return await _run_prediction("diabetes", payload.model_dump(), redis)


async def _run_prediction(
    disease: str,
    data: dict,
    redis: Any,
) -> PredictionResponse:
    """Core prediction logic — shared across all disease endpoints."""

    # 1. Cache check
    cache_key = _cache_key(disease, data)
    cached = await redis.get(cache_key)
    if cached:
        result = PredictionResponse(**json.loads(cached))
        result.cached = True
        return result

    # 2. Feature DataFrame
    patient_id = data.pop("patient_id", None)
    df = pd.DataFrame([data])

    # 3. Model inference
    prob, contributions = prediction_engine.predict(disease, df)

    # 4. Risk scoring
    scorer = RiskScorer(disease)
    risk = scorer.compute(calibrated_prob=prob)

    # 5. Build response
    response = PredictionResponse(
        patient_id=patient_id,
        disease=disease,
        risk_score=risk.composite_score,
        calibrated_probability=risk.calibrated_probability,
        risk_category=risk.risk_category.value,
        confidence_interval=risk.confidence_interval,
        velocity=risk.velocity,
        top_features=[
            FeatureContributionOut(
                feature=c.feature,
                value=c.value,
                shap_value=c.shap_value,
                direction=c.direction,
                rank=c.rank,
            )
            for c in contributions
        ],
        plain_english_summary=(
            f"Predicted {disease.replace('_', ' ')} risk: "
            f"{prob:.0%} ({risk.risk_category.value})"
        ),
        clinical_action=risk.clinical_action,
        model_version=f"{disease}_v1.0",
        cached=False,
    )

    # 6. Cache for TTL seconds
    await redis.setex(
        cache_key,
        settings.prediction_cache_ttl_seconds,
        response.model_dump_json(),
    )

    return response

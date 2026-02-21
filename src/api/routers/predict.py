"""
src/api/routers/predict.py — Prediction API endpoints.

Implements disease-specific prediction routes with:
- MLflow local file-store model loading (no running server required)
- SHAP TreeExplainer explanations on every prediction
- Composite risk scoring with calibration
- Redis caching (with graceful fallback when Redis unavailable)
- Async-safe design with background thread offloading for CPU work
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.api.deps import get_current_user, get_redis
from src.config import get_settings
from src.features.engineering import apply_feature_engineering
from src.features.pipeline import FEATURE_CONFIG
from src.scoring.risk_scorer import RiskScorer

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# Thread pool for CPU-bound inference
_executor = ThreadPoolExecutor(max_workers=4)


# ─── Request Schemas ──────────────────────────────────────────────────────────

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
    patient_id: str | None = Field(None, description="Optional patient ID")


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


class CancerInput(BaseModel):
    """Wisconsin Diagnostic Breast Cancer — 10 mean features (simplified)."""
    radius_mean: float = Field(..., ge=0)
    texture_mean: float = Field(..., ge=0)
    perimeter_mean: float = Field(..., ge=0)
    area_mean: float = Field(..., ge=0)
    smoothness_mean: float = Field(..., ge=0, le=1)
    compactness_mean: float = Field(..., ge=0, le=2)
    concavity_mean: float = Field(..., ge=0, le=2)
    concave_points_mean: float = Field(..., ge=0, le=1)
    symmetry_mean: float = Field(..., ge=0, le=1)
    fractal_dimension_mean: float = Field(..., ge=0, le=1)
    # SE features
    radius_se: float = Field(0.0, ge=0)
    texture_se: float = Field(0.0, ge=0)
    perimeter_se: float = Field(0.0, ge=0)
    area_se: float = Field(0.0, ge=0)
    smoothness_se: float = Field(0.0, ge=0)
    compactness_se: float = Field(0.0, ge=0)
    concavity_se: float = Field(0.0, ge=0)
    concave_points_se: float = Field(0.0, ge=0)
    symmetry_se: float = Field(0.0, ge=0)
    fractal_dimension_se: float = Field(0.0, ge=0)
    # Worst features
    radius_worst: float = Field(0.0, ge=0)
    texture_worst: float = Field(0.0, ge=0)
    perimeter_worst: float = Field(0.0, ge=0)
    area_worst: float = Field(0.0, ge=0)
    smoothness_worst: float = Field(0.0, ge=0)
    compactness_worst: float = Field(0.0, ge=0)
    concavity_worst: float = Field(0.0, ge=0)
    concave_points_worst: float = Field(0.0, ge=0)
    symmetry_worst: float = Field(0.0, ge=0)
    fractal_dimension_worst: float = Field(0.0, ge=0)
    patient_id: str | None = None


class KidneyInput(BaseModel):
    """Chronic Kidney Disease — core clinical features."""
    age: float = Field(..., ge=0, le=120)
    bp: float = Field(..., ge=0, le=250, description="Blood pressure (mmHg)")
    sg: float = Field(1.020, ge=1.000, le=1.030, description="Specific gravity of urine")
    al: float = Field(0.0, ge=0, le=5, description="Albumin (0-5 scale)")
    su: float = Field(0.0, ge=0, le=5, description="Sugar (0-5 scale)")
    bgr: float = Field(..., ge=0, le=500, description="Blood glucose random (mg/dL)")
    bu: float = Field(..., ge=0, le=200, description="Blood urea (mg/dL)")
    sc: float = Field(..., ge=0, le=20, description="Serum creatinine (mg/dL)")
    sod: float = Field(137.0, ge=0, le=200, description="Sodium (mEq/L)")
    pot: float = Field(4.5, ge=0, le=15, description="Potassium (mEq/L)")
    hemo: float = Field(..., ge=0, le=20, description="Hemoglobin (g/dL)")
    pcv: float = Field(44.0, ge=0, le=60, description="Packed cell volume (%)")
    wc: float = Field(7800.0, ge=0, le=30000, description="WBC count (cells/cumm)")
    rc: float = Field(5.2, ge=0, le=10, description="RBC count (millions/cmm)")
    htn: int = Field(0, ge=0, le=1, description="Hypertension (0=No, 1=Yes)")
    dm: int = Field(0, ge=0, le=1, description="Diabetes mellitus (0=No, 1=Yes)")
    cad: int = Field(0, ge=0, le=1, description="Coronary artery disease (0=No, 1=Yes)")
    appet: int = Field(1, ge=0, le=1, description="Appetite (0=Poor, 1=Good)")
    pe: int = Field(0, ge=0, le=1, description="Pedal edema (0=No, 1=Yes)")
    ane: int = Field(0, ge=0, le=1, description="Anemia (0=No, 1=Yes)")
    patient_id: str | None = None


# ─── Response Schemas ─────────────────────────────────────────────────────────

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
    velocity: float | None = None
    top_features: list[FeatureContributionOut]
    plain_english_summary: str
    clinical_action: dict[str, str]
    model_version: str
    cached: bool = False


# ─── Prediction Engine ────────────────────────────────────────────────────────

class PredictionEngine:
    """
    Singleton that manages loaded ML models and SHAP explainers.
    Models are loaded at startup from the local MLflow file store.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._explainers: dict[str, Any] = {}
        self._feature_names: dict[str, list[str]] = {}
        self._background_data: dict[str, pd.DataFrame] = {}

    def _load_models_sync(self) -> None:
        """Synchronous model loading — runs in a thread at startup."""
        import mlflow
        import mlflow.sklearn

        tracking_uri = settings.mlflow_tracking_uri
        # Prefer local file store if configured
        if tracking_uri.startswith("http://mlflow") or tracking_uri == "http://localhost:5000":
            import os
            project_root = str(__file__).split("/src/")[0]
            local_uri = f"file://{project_root}/mlruns"
            tracking_uri = local_uri
            logger.info("MLflow server not configured — falling back to: %s", local_uri)

        mlflow.set_tracking_uri(tracking_uri)

        for disease, model_name in settings.registered_models.items():
            for alias in ["@latest", "/1", "/2", "/3"]:
                try:
                    model_uri = f"models:/{model_name}{alias}"
                    model = mlflow.sklearn.load_model(model_uri)
                    self._models[disease] = model
                    logger.info("✓ Loaded %s model from %s", disease, model_uri)
                    break
                except Exception as exc:
                    logger.debug("Could not load %s%s: %s", model_name, alias, exc)
            else:
                logger.warning("✗ No model found for disease: %s", disease)

        # Load training data for SHAP background (small sample)
        self._setup_shap_explainers()

    def _setup_shap_explainers(self) -> None:
        """Initialize SHAP explainers from background data sampled from training sets."""
        try:
            import shap
            from src.data.load import load_heart, load_diabetes, load_cancer, load_kidney
            from src.features.engineering import apply_feature_engineering

            loaders = {
                "heart": load_heart,
                "diabetes": load_diabetes,
                "cancer": load_cancer,
                "kidney": load_kidney,
            }

            for disease, loader in loaders.items():
                if disease not in self._models:
                    continue
                try:
                    df = loader()
                    df = apply_feature_engineering(disease, df)
                    target_col = FEATURE_CONFIG[disease]["target"]
                    X = df.drop(columns=[target_col], errors="ignore")

                    # Use 50-row background for efficiency
                    background = X.sample(
                        n=min(50, len(X)), random_state=42
                    ).reset_index(drop=True)

                    model = self._models[disease]

                    # SHAP TreeExplainer works on the inner estimator
                    inner = self._extract_inner_model(model)
                    if inner is not None:
                        # Transform background through the preprocessor steps only
                        proc = self._extract_preprocessor(model)
                        if proc is not None:
                            bg_transformed = proc.transform(background)
                            feature_names = self._get_feature_names(proc, background)
                            bg_df = pd.DataFrame(bg_transformed, columns=feature_names)
                            explainer = shap.TreeExplainer(
                                inner,
                                data=bg_df,
                                feature_perturbation="tree_path_dependent",
                            )
                            self._explainers[disease] = explainer
                            self._feature_names[disease] = feature_names
                            self._background_data[disease] = bg_df
                            logger.info("✓ SHAP explainer ready for %s (%d features)", disease, len(feature_names))
                except Exception as exc:
                    logger.warning("Could not setup SHAP for %s: %s", disease, exc)
        except Exception as exc:
            logger.warning("SHAP setup failed (non-fatal): %s", exc)

    def _extract_inner_model(self, pipeline: Any) -> Any:
        """Extract the classifier from the imblearn pipeline."""
        try:
            if hasattr(pipeline, "named_estimators_"):
                # CalibratedClassifierCV — get one of the calibrated classifiers
                cal = pipeline.calibrated_classifiers_[0]
                inner = cal.estimator
            elif hasattr(pipeline, "named_steps"):
                inner = pipeline.named_steps.get("classifier")
            else:
                # CalibratedClassifierCV wrapping a pipeline
                cal = pipeline.calibrated_classifiers_[0]
                inner_pipeline = cal.estimator
                if hasattr(inner_pipeline, "named_steps"):
                    inner = inner_pipeline.named_steps.get("classifier")
                else:
                    inner = inner_pipeline
            return inner
        except Exception as exc:
            logger.debug("Could not extract inner model: %s", exc)
            return None

    def _extract_preprocessor(self, pipeline: Any) -> Any:
        """Extract the preprocessor from the imblearn pipeline."""
        try:
            if hasattr(pipeline, "calibrated_classifiers_"):
                cal = pipeline.calibrated_classifiers_[0]
                inner_pipeline = cal.estimator
                if hasattr(inner_pipeline, "named_steps"):
                    return inner_pipeline.named_steps.get("preprocessor")
            elif hasattr(pipeline, "named_steps"):
                return pipeline.named_steps.get("preprocessor")
            return None
        except Exception:
            return None

    def _get_feature_names(self, preprocessor: Any, X: pd.DataFrame) -> list[str]:
        """Get output feature names from a fitted ColumnTransformer."""
        try:
            names = []
            for name, trans, cols in preprocessor.transformers_:
                if name == "remainder":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    feature_names = trans.get_feature_names_out(cols)
                    names.extend(feature_names)
                else:
                    names.extend(cols if isinstance(cols, list) else [str(cols)])
            return names if names else list(range(preprocessor.transform(X).shape[1]))
        except Exception:
            # Fallback: use positional indices
            try:
                n = preprocessor.transform(X[:1]).shape[1]
                return [f"feature_{i}" for i in range(n)]
            except Exception:
                return []

    async def preload_models(self) -> None:
        """Preload all models at startup (runs sync code in a thread)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._load_models_sync)

    def get_model(self, disease: str) -> Any:
        model = self._models.get(disease)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model for '{disease}' is not loaded. Run training first.",
            )
        return model

    def _predict_sync(self, disease: str, df: pd.DataFrame) -> tuple[float, list]:
        """Synchronous model inference + SHAP (runs in thread pool)."""
        model = self.get_model(disease)

        # Apply feature engineering before sending to pipeline
        df_engineered = apply_feature_engineering(disease, df)
        target_col = FEATURE_CONFIG[disease]["target"]
        df_engineered = df_engineered.drop(columns=[target_col], errors="ignore")

        prob = float(model.predict_proba(df_engineered)[:, 1][0])

        # SHAP contributions
        contributions = []
        explainer = self._explainers.get(disease)
        feature_names = self._feature_names.get(disease, [])

        if explainer and feature_names:
            try:
                preprocessor = self._extract_preprocessor(model)
                if preprocessor is not None:
                    X_transformed = preprocessor.transform(df_engineered)
                    X_df = pd.DataFrame(X_transformed, columns=feature_names)
                    shap_values = explainer.shap_values(X_df)

                    if isinstance(shap_values, list):
                        shap_row = shap_values[1][0]
                    elif hasattr(shap_values, "values"):
                        sv = shap_values.values
                        shap_row = sv[0, :, 1] if sv.ndim == 3 else sv[0]
                    else:
                        shap_row = shap_values[0]

                    ranked = np.argsort(np.abs(shap_row))[::-1][:5]
                    for rank, idx in enumerate(ranked, start=1):
                        val = float(shap_row[idx])
                        feat = feature_names[idx] if idx < len(feature_names) else f"f_{idx}"
                        feat_val = float(X_df.iloc[0, idx]) if idx < X_df.shape[1] else 0.0
                        contributions.append({
                            "feature": feat,
                            "value": feat_val,
                            "shap_value": val,
                            "direction": (
                                "increases_risk" if val > 0.001 else
                                "decreases_risk" if val < -0.001 else
                                "neutral"
                            ),
                            "rank": rank,
                        })
            except Exception as exc:
                logger.warning("SHAP inference failed for %s: %s", disease, exc)

        return prob, contributions

    async def predict(self, disease: str, df: pd.DataFrame) -> tuple[float, list]:
        """Offload CPU-bound inference to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            partial(self._predict_sync, disease, df),
        )


prediction_engine = PredictionEngine()


# ─── Cache Helper ─────────────────────────────────────────────────────────────

def _cache_key(disease: str, payload: dict) -> str:
    payload_str = json.dumps({k: v for k, v in sorted(payload.items()) if k != "patient_id"})
    return f"predict:{disease}:{hashlib.sha256(payload_str.encode()).hexdigest()[:16]}"


async def _cache_get(redis: Any, key: str) -> str | None:
    """Redis get with graceful fallback."""
    try:
        return await redis.get(key)
    except Exception:
        return None


async def _cache_set(redis: Any, key: str, ttl: int, value: str) -> None:
    """Redis set with graceful fallback."""
    try:
        await redis.setex(key, ttl, value)
    except Exception:
        pass


# ─── Core Prediction Logic ────────────────────────────────────────────────────

async def _run_prediction(
    disease: str,
    data: dict,
    redis: Any,
) -> PredictionResponse:
    """Core prediction logic — shared across all disease endpoints."""

    # 1. Cache check
    cache_key = _cache_key(disease, data)
    cached_val = await _cache_get(redis, cache_key)
    if cached_val:
        result = PredictionResponse(**json.loads(cached_val))
        result.cached = True
        return result

    # 2. Feature DataFrame
    patient_id = data.pop("patient_id", None)
    df = pd.DataFrame([data])

    # 3. Model inference (in thread pool)
    prob, contributions = await prediction_engine.predict(disease, df)

    # 4. Risk scoring
    scorer = RiskScorer(disease)
    risk = scorer.compute(calibrated_prob=prob)

    # 5. Plain-english summary
    if contributions:
        drivers = [c["feature"].replace("_", " ") for c in contributions if c["direction"] == "increases_risk"][:3]
        protective = [c["feature"].replace("_", " ") for c in contributions if c["direction"] == "decreases_risk"][:2]
        summary = (
            f"Predicted {disease} risk: {prob:.0%} ({risk.risk_category.value}). "
        )
        if drivers:
            summary += f"Primary risk drivers: {', '.join(drivers)}. "
        if protective:
            summary += f"Protective factors: {', '.join(protective)}."
    else:
        summary = f"Predicted {disease.replace('_', ' ')} risk: {prob:.0%} ({risk.risk_category.value})."

    # 6. Build response
    response = PredictionResponse(
        patient_id=patient_id,
        disease=disease,
        risk_score=risk.composite_score,
        calibrated_probability=prob,
        risk_category=risk.risk_category.value,
        confidence_interval=risk.confidence_interval,
        velocity=risk.velocity,
        top_features=[FeatureContributionOut(**c) for c in contributions],
        plain_english_summary=summary,
        clinical_action=risk.clinical_action,
        model_version=f"{disease}_v1.0",
        cached=False,
    )

    # 7. Cache result
    await _cache_set(redis, cache_key, settings.prediction_cache_ttl_seconds, response.model_dump_json())

    return response


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/heart", response_model=PredictionResponse, summary="Heart disease risk prediction")
async def predict_heart_disease(
    payload: HeartDiseaseInput,
    current_user: Annotated[Any, Depends(get_current_user)],
    redis=Depends(get_redis),
) -> PredictionResponse:
    """
    Predict heart disease risk.

    Returns calibrated probability, composite risk score (LOW→CRITICAL),
    95% confidence interval, and SHAP feature contributions.
    """
    return await _run_prediction("heart", payload.model_dump(), redis)


@router.post("/diabetes", response_model=PredictionResponse, summary="Type 2 diabetes risk prediction")
async def predict_diabetes(
    payload: DiabetesInput,
    current_user: Annotated[Any, Depends(get_current_user)],
    redis=Depends(get_redis),
) -> PredictionResponse:
    """Predict Type 2 diabetes risk."""
    return await _run_prediction("diabetes", payload.model_dump(), redis)


@router.post("/cancer", response_model=PredictionResponse, summary="Breast cancer risk prediction")
async def predict_cancer(
    payload: CancerInput,
    current_user: Annotated[Any, Depends(get_current_user)],
    redis=Depends(get_redis),
) -> PredictionResponse:
    """
    Predict breast cancer risk (Wisconsin WDBC features).

    Returns malignancy probability with SHAP contributions showing
    which morphological features most influence the prediction.
    """
    return await _run_prediction("cancer", payload.model_dump(), redis)


@router.post("/kidney", response_model=PredictionResponse, summary="Chronic kidney disease risk prediction")
async def predict_kidney(
    payload: KidneyInput,
    current_user: Annotated[Any, Depends(get_current_user)],
    redis=Depends(get_redis),
) -> PredictionResponse:
    """Predict chronic kidney disease risk from clinical lab values."""
    return await _run_prediction("kidney", payload.model_dump(), redis)

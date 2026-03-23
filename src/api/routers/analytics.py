"""src/api/routers/analytics.py — Population analytics endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user
from src.db.models import Prediction
from src.db.session import get_db

router = APIRouter()


class AnalyticsSummary(BaseModel):
    total_predictions: int
    disease_breakdown: dict[str, int]
    high_risk_count: int
    risk_distribution: dict[str, int]


class ClusterInfo(BaseModel):
    cluster_id: int
    label: str
    patient_count: int
    top_features: list[str]


class ComorbidityRule(BaseModel):
    antecedents: list[str]
    consequents: list[str]
    support: float
    confidence: float
    lift: float


@router.get("/summary", response_model=AnalyticsSummary)
async def analytics_summary(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AnalyticsSummary:
    """
    Population-level analytics summary.
    
    Returns aggregate statistics across all predictions.
    """
    # Total predictions
    total_result = await db.execute(select(func.count(Prediction.id)))
    total = total_result.scalar() or 0
    
    # Disease breakdown
    disease_counts = await db.execute(
        select(Prediction.disease, func.count(Prediction.id))
        .group_by(Prediction.disease)
    )
    disease_breakdown = {
        "heart": 0, "diabetes": 0, "cancer": 0, "kidney": 0
    }
    for disease, count in disease_counts:
        if disease in disease_breakdown:
            disease_breakdown[disease] = count
    
    # High risk count (HIGH or CRITICAL)
    high_risk_result = await db.execute(
        select(func.count(Prediction.id))
        .where(Prediction.risk_category.in_(["HIGH", "CRITICAL"]))
    )
    high_risk = high_risk_result.scalar() or 0
    
    # Risk distribution
    risk_dist = await db.execute(
        select(Prediction.risk_category, func.count(Prediction.id))
        .group_by(Prediction.risk_category)
    )
    risk_distribution = {"LOW": 0, "BORDERLINE": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    for category, count in risk_dist:
        if category in risk_distribution:
            risk_distribution[category] = count
    
    return AnalyticsSummary(
        total_predictions=total,
        disease_breakdown=disease_breakdown,
        high_risk_count=high_risk,
        risk_distribution=risk_distribution,
    )


@router.get("/clusters")
async def get_clusters(
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """
    Patient phenotype cluster membership.
    
    Returns predefined clusters based on risk profiles.
    Note: Full GMM clustering requires batch analysis pipeline.
    """
    # Placeholder clusters - in production, these would come from
    # a background clustering job that runs periodically
    clusters = [
        ClusterInfo(
            cluster_id=0,
            label="Low Risk - Healthy",
            patient_count=0,
            top_features=["Normal BMI", "Non-smoker", "Regular exercise"],
        ),
        ClusterInfo(
            cluster_id=1,
            label="Moderate Risk - Metabolic",
            patient_count=0,
            top_features=["Elevated glucose", "High BMI", "Sedentary"],
        ),
        ClusterInfo(
            cluster_id=2,
            label="High Risk - Cardiovascular",
            patient_count=0,
            top_features=["Hypertension", "High cholesterol", "ST depression"],
        ),
        ClusterInfo(
            cluster_id=3,
            label="Multi-morbid",
            patient_count=0,
            top_features=["Multiple conditions", "Elderly", "Polypharmacy"],
        ),
    ]
    
    return {"clusters": [c.model_dump() for c in clusters]}


@router.get("/comorbidity-rules")
async def get_comorbidity_rules(
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """
    Association rules for comorbidity patterns.
    
    Returns common disease co-occurrence patterns.
    Note: Full FP-Growth mining requires batch analysis pipeline.
    """
    # Common clinical comorbidity patterns
    # In production, these would be mined from prediction history
    rules = [
        ComorbidityRule(
            antecedents=["diabetes"],
            consequents=["heart_disease"],
            support=0.15,
            confidence=0.72,
            lift=2.4,
        ),
        ComorbidityRule(
            antecedents=["obesity", "hypertension"],
            consequents=["diabetes"],
            support=0.12,
            confidence=0.68,
            lift=2.1,
        ),
        ComorbidityRule(
            antecedents=["kidney_disease"],
            consequents=["heart_disease"],
            support=0.08,
            confidence=0.65,
            lift=1.9,
        ),
    ]
    
    return {"rules": [r.model_dump() for r in rules]}

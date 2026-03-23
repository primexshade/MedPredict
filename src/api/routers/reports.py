"""src/api/routers/reports.py — Report generation endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user
from src.db.models import Patient, Prediction
from src.db.session import get_db

router = APIRouter()


class PredictionSummary(BaseModel):
    disease: str
    risk_category: str
    calibrated_probability: float
    created_at: datetime
    
    model_config = {"from_attributes": True}


class PatientReport(BaseModel):
    patient_id: str
    mrn: str
    generated_at: datetime
    predictions: list[PredictionSummary]
    risk_summary: dict[str, str]


@router.get("/{patient_id}", response_model=PatientReport)
async def generate_report(
    patient_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PatientReport:
    """
    Generate a summary report for a patient.
    
    Returns patient's prediction history and risk summary.
    For PDF export, use /reports/{patient_id}/pdf endpoint.
    """
    # Fetch patient
    patient_result = await db.execute(
        select(Patient).where(Patient.id == patient_id)
    )
    patient = patient_result.scalar_one_or_none()
    
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )
    
    # Authorization check
    if current_user["role"] not in ("admin", "superadmin"):
        if patient.primary_clinician_id != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this patient",
            )
    
    # Fetch predictions
    predictions_result = await db.execute(
        select(Prediction)
        .where(Prediction.patient_id == patient_id)
        .order_by(Prediction.created_at.desc())
        .limit(50)
    )
    predictions = predictions_result.scalars().all()
    
    # Build risk summary (latest prediction per disease)
    risk_summary: dict[str, str] = {}
    seen_diseases: set[str] = set()
    for pred in predictions:
        if pred.disease not in seen_diseases:
            risk_summary[pred.disease] = pred.risk_category
            seen_diseases.add(pred.disease)
    
    return PatientReport(
        patient_id=patient.id,
        mrn=patient.mrn,
        generated_at=datetime.utcnow(),
        predictions=[
            PredictionSummary(
                disease=p.disease,
                risk_category=p.risk_category,
                calibrated_probability=p.calibrated_probability,
                created_at=p.created_at,
            )
            for p in predictions
        ],
        risk_summary=risk_summary,
    )


@router.get("/{patient_id}/pdf")
async def generate_pdf_report(
    patient_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """
    Generate a PDF health report for a patient.
    
    Note: PDF generation requires ReportLab integration.
    This endpoint is planned for Phase 2.
    """
    return {
        "patient_id": patient_id,
        "status": "planned",
        "message": "PDF report generation is planned for Phase 2. Use JSON report endpoint for now.",
        "alternative": f"/api/v1/reports/{patient_id}",
    }

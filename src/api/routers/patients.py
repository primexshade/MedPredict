"""src/api/routers/patients.py — Patient management endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, require_permission
from src.db.models import Patient
from src.db.session import get_db

router = APIRouter()


class PatientOut(BaseModel):
    id: str
    mrn: str
    sex: str | None = None
    
    model_config = {"from_attributes": True}


class PatientCreate(BaseModel):
    mrn: str
    sex: str | None = None


@router.get("/", response_model=list[PatientOut])
async def list_patients(
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[PatientOut]:
    """
    List all patients accessible to the current user.
    
    Clinicians see only their assigned patients.
    Admins see all patients.
    """
    query = select(Patient)
    
    # RBAC: Non-admin users only see their assigned patients
    if current_user["role"] not in ("admin", "superadmin"):
        query = query.where(Patient.primary_clinician_id == current_user["user_id"])
    
    result = await db.execute(query.order_by(Patient.created_at.desc()).limit(100))
    patients = result.scalars().all()
    
    return [PatientOut.model_validate(p) for p in patients]


@router.get("/{patient_id}", response_model=PatientOut)
async def get_patient(
    patient_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PatientOut:
    """
    Get a patient by ID.
    
    Enforces authorization: clinicians can only access their assigned patients.
    """
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    
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
    
    return PatientOut.model_validate(patient)


@router.post("/", response_model=PatientOut, status_code=status.HTTP_201_CREATED)
async def create_patient(
    payload: PatientCreate,
    current_user: Annotated[dict, Depends(require_permission("write"))],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PatientOut:
    """
    Create a new patient record.
    
    Requires 'write' permission. Assigns the creating clinician as primary.
    """
    # Check for duplicate MRN
    existing = await db.execute(select(Patient).where(Patient.mrn == payload.mrn))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Patient with this MRN already exists",
        )
    
    patient = Patient(
        mrn=payload.mrn,
        sex=payload.sex,
        primary_clinician_id=current_user["user_id"],
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    
    return PatientOut.model_validate(patient)

"""src/api/routers/patients.py — Patient management endpoints (stub)."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()


class PatientOut(BaseModel):
    id: str
    mrn: str


@router.get("/", response_model=list[PatientOut])
async def list_patients() -> list[PatientOut]:
    """List all patients. TODO: real DB query."""
    return []


@router.get("/{patient_id}", response_model=PatientOut)
async def get_patient(patient_id: str) -> PatientOut:
    """Get a patient by ID. TODO: real DB query."""
    return PatientOut(id=patient_id, mrn="MRN-STUB")

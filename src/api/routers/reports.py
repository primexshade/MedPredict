"""src/api/routers/reports.py — PDF report generation endpoints (stub)."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/{patient_id}")
async def generate_report(patient_id: str) -> dict:
    """Generate PDF health report for a patient. TODO: ReportLab integration."""
    return {
        "patient_id": patient_id,
        "status": "not_implemented",
        "message": "PDF report generation is planned for Phase 2.",
    }

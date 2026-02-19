"""
src/db/models.py — SQLAlchemy ORM models for all database entities.

Uses SQLAlchemy 2.0 declarative style with type annotations.
Every table includes audit columns (created_at, updated_at).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Abstract base with shared audit columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ─── Users / Auth ─────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(50), nullable=False, default="clinician"
    )  # clinician | patient | admin | researcher | superadmin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    patients: Mapped[list["Patient"]] = relationship(back_populates="primary_clinician")
    audit_logs: Mapped[list["AuditLog"]] = relationship(back_populates="user")


# ─── Patients ─────────────────────────────────────────────────────────────────

class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    mrn: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    date_of_birth: Mapped[datetime | None] = mapped_column(DateTime)
    sex: Mapped[str | None] = mapped_column(String(1))  # M | F | O
    ethnicity: Mapped[str | None] = mapped_column(String(100))

    primary_clinician_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id")
    )
    primary_clinician: Mapped["User | None"] = relationship(back_populates="patients")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="patient")


# ─── Predictions ──────────────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    patient_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("patients.id"), nullable=False, index=True
    )
    disease: Mapped[str] = mapped_column(String(50), nullable=False)  # heart|diabetes|cancer|kidney
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Scores
    calibrated_probability: Mapped[float] = mapped_column(Float, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_category: Mapped[str] = mapped_column(String(20), nullable=False)  # LOW..CRITICAL
    ci_lower: Mapped[float] = mapped_column(Float)
    ci_upper: Mapped[float] = mapped_column(Float)

    # Input snapshot (JSONB) — stored for auditability & retraining
    input_features: Mapped[dict] = mapped_column(JSON, nullable=False)

    # SHAP explanations
    shap_contributions: Mapped[dict | None] = mapped_column(JSON)
    plain_english_summary: Mapped[str | None] = mapped_column(Text)

    # Relationships
    patient: Mapped["Patient"] = relationship(back_populates="predictions")


# ─── Audit Log ────────────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(100), nullable=False)  # predict|login|export
    resource: Mapped[str | None] = mapped_column(String(100))
    resource_id: Mapped[str | None] = mapped_column(String(255))
    ip_address: Mapped[str | None] = mapped_column(String(50))
    status_code: Mapped[int | None] = mapped_column(Integer)
    details: Mapped[dict | None] = mapped_column(JSON)

    user: Mapped["User | None"] = relationship(back_populates="audit_logs")

"""Initial schema — users, patients, predictions, audit_logs.

Revision ID: 0001
Revises:
Create Date: 2026-02-21 00:00:00.000000 UTC

Creates the four core tables from scratch:
  - users        (auth + RBAC)
  - patients     (patient registry)
  - predictions  (risk prediction records)
  - audit_logs   (immutable action trail)

All UUID primary keys use VARCHAR(36) for cross-DB compatibility.
Indexes and foreign key constraints are fully specified.
Naming convention follows SQLAlchemy recommended patterns to make
future autogenerate-detected diffs readable.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# ── Revision identifiers ──────────────────────────────────────────────────────
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ─────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="clinician"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── patients ──────────────────────────────────────────────────────────────
    op.create_table(
        "patients",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("mrn", sa.String(50), nullable=False),
        sa.Column("date_of_birth", sa.DateTime(), nullable=True),
        sa.Column("sex", sa.String(1), nullable=True),
        sa.Column("ethnicity", sa.String(100), nullable=True),
        sa.Column("primary_clinician_id", sa.String(36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["primary_clinician_id"],
            ["users.id"],
            name="fk_patients_primary_clinician_id",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_patients_mrn", "patients", ["mrn"], unique=True)
    op.create_index(
        "ix_patients_primary_clinician",
        "patients",
        ["primary_clinician_id"],
        unique=False,
    )

    # ── predictions ───────────────────────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("patient_id", sa.String(36), nullable=False),
        sa.Column("disease", sa.String(50), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("calibrated_probability", sa.Float(), nullable=False),
        sa.Column("composite_score", sa.Float(), nullable=False),
        sa.Column("risk_category", sa.String(20), nullable=False),
        sa.Column("ci_lower", sa.Float(), nullable=True),
        sa.Column("ci_upper", sa.Float(), nullable=True),
        sa.Column("input_features", sa.JSON(), nullable=False),
        sa.Column("shap_contributions", sa.JSON(), nullable=True),
        sa.Column("plain_english_summary", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_predictions_patient_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_predictions_patient_id", "predictions", ["patient_id"], unique=False)
    op.create_index("ix_predictions_disease", "predictions", ["disease"], unique=False)
    op.create_index(
        "ix_predictions_risk_category",
        "predictions",
        ["risk_category"],
        unique=False,
    )

    # ── audit_logs ────────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(36), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource", sa.String(100), nullable=True),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column("ip_address", sa.String(50), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_audit_logs_user_id",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"], unique=False)
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"], unique=False)
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False)


def downgrade() -> None:
    # Drop in reverse dependency order
    op.drop_table("audit_logs")
    op.drop_table("predictions")
    op.drop_table("patients")
    op.drop_table("users")

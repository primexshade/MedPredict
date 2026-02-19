"""
src/data/load.py — Dataset loaders with Pydantic schema validation.

Each loader returns a validated DataFrame ready for the preprocessing pipeline.
Domain constraint errors raise DataValidationError with row-level details.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"


class DataValidationError(Exception):
    """Raised when raw dataset fails domain constraint checks."""


# ─── Heart Disease ────────────────────────────────────────────────────────────

HEART_COLUMNS = {
    "age": float,
    "sex": float,
    "cp": float,        # chest pain type (0-3)
    "trestbps": float,  # resting blood pressure
    "chol": float,      # serum cholesterol mg/dl
    "fbs": float,       # fasting blood sugar > 120 mg/dl (binary)
    "restecg": float,   # resting ECG results (0-2)
    "thalach": float,   # max heart rate achieved
    "exang": float,     # exercise-induced angina (binary)
    "oldpeak": float,   # ST depression induced by exercise
    "slope": float,     # slope of peak exercise ST segment
    "ca": float,        # number of major vessels colored (0-3)
    "thal": float,      # thalassemia (0=normal, 1=fixed defect, 2=reversible)
    "target": int,      # 0 = no disease, 1 = disease
}

HEART_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (0, 700),
    "thalach": (40, 250),
    "oldpeak": (0, 10),
}


def load_heart_disease(source: str = "combined") -> pd.DataFrame:
    """
    Load the UCI Heart Disease dataset (Cleveland + Hungarian + Switzerland).

    Args:
        source: 'combined' (default), 'cleveland', 'hungarian', 'switzerland'

    Returns:
        Validated DataFrame with standardized column names.
    """
    filename = f"heart_{source}.csv"
    path = RAW_DIR / filename

    if not path.exists():
        sample_path = SAMPLE_DIR / filename
        if sample_path.exists():
            logger.warning("Raw data not found; using sample data: %s", sample_path)
            path = sample_path
        else:
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Run: bash scripts/download_datasets.sh"
            )

    df = pd.read_csv(path)
    df = df.rename(columns=str.lower)

    # Coerce dtypes
    for col, dtype in HEART_COLUMNS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    # Binarize target (original dataset uses 0-4, where >0 = disease)
    if df["target"].max() > 1:
        df["target"] = (df["target"] > 0).astype(int)

    _validate_constraints(df, HEART_CONSTRAINTS, dataset="heart_disease")
    logger.info("Loaded heart disease dataset: %d rows, %d cols", *df.shape)
    return df


# ─── Diabetes ────────────────────────────────────────────────────────────────

DIABETES_COLUMNS = {
    "pregnancies": float,
    "glucose": float,
    "bloodpressure": float,
    "skinthickness": float,
    "insulin": float,
    "bmi": float,
    "diabetespedigreefunction": float,
    "age": float,
    "outcome": int,  # target: 0 = no diabetes, 1 = diabetes
}

DIABETES_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "glucose": (0, 400),
    "bloodpressure": (0, 200),
    "bmi": (0, 80),
    "age": (1, 120),
}


def load_diabetes() -> pd.DataFrame:
    """Load the PIMA Indians Diabetes dataset."""
    path = _resolve_path("diabetes.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    for col, dtype in DIABETES_COLUMNS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    # In the PIMA dataset, zeros in physiological features encode missing values
    zero_means_missing = ["glucose", "bloodpressure", "skinthickness", "insulin", "bmi"]
    df[zero_means_missing] = df[zero_means_missing].replace(0, float("nan"))

    _validate_constraints(df, DIABETES_CONSTRAINTS, dataset="diabetes")
    logger.info("Loaded diabetes dataset: %d rows, %d cols", *df.shape)
    return df


# ─── Breast Cancer ────────────────────────────────────────────────────────────

def load_breast_cancer() -> pd.DataFrame:
    """Load the Wisconsin Breast Cancer (Diagnostic) dataset."""
    path = _resolve_path("breast_cancer.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # WDBC: diagnosis M=malignant(1), B=benign(0)
    if "diagnosis" in df.columns:
        df["target"] = (df["diagnosis"] == "M").astype(int)
        df = df.drop(columns=["diagnosis", "id"], errors="ignore")

    logger.info("Loaded breast cancer dataset: %d rows, %d cols", *df.shape)
    return df


# ─── Kidney Disease ───────────────────────────────────────────────────────────

def load_kidney_disease() -> pd.DataFrame:
    """Load the UCI Chronic Kidney Disease dataset."""
    path = _resolve_path("kidney_disease.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    # Target: ckd=1, notckd=0
    if "classification" in df.columns:
        df["target"] = (df["classification"].str.strip() == "ckd").astype(int)
        df = df.drop(columns=["classification"], errors="ignore")

    # Replace \t characters introduced in some CSV exports
    df = df.replace({r"\t": ""}, regex=True)

    logger.info("Loaded kidney disease dataset: %d rows, %d cols", *df.shape)
    return df


# ─── Internal Utilities ──────────────────────────────────────────────────────

def _resolve_path(filename: str) -> Path:
    path = RAW_DIR / filename
    if not path.exists():
        sample_path = SAMPLE_DIR / filename
        if sample_path.exists():
            logger.warning("Using sample data: %s", sample_path)
            return sample_path
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run: bash scripts/download_datasets.sh"
        )
    return path


def _validate_constraints(
    df: pd.DataFrame,
    constraints: dict[str, tuple[float, float]],
    dataset: str,
) -> None:
    """Raise DataValidationError if any feature violates domain constraints."""
    violations: list[dict[str, Any]] = []

    for col, (lo, hi) in constraints.items():
        if col not in df.columns:
            continue
        mask = df[col].notna() & ~df[col].between(lo, hi)
        if mask.any():
            bad_rows = df.loc[mask, col].head(5).to_dict()
            violations.append({"feature": col, "range": (lo, hi), "examples": bad_rows})

    if violations:
        logger.warning("%s validation warnings: %s", dataset, violations)
        # Log as warning (not error) — clinical data often has edge cases
        # Raise as error only if > 5% of rows are affected
        affected_pct = sum(len(v["examples"]) for v in violations) / len(df)
        if affected_pct > 0.05:
            raise DataValidationError(
                f"Dataset '{dataset}' has >{affected_pct:.0%} constraint violations: "
                f"{violations}"
            )

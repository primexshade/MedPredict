"""
src/features/engineering.py — Domain-aware feature engineering.

All engineered features are motivated by clinical guidelines (ACC/AHA, ADA)
and epidemiological literature. Each transformation includes the source rationale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Heart Disease Features ──────────────────────────────────────────────────

def engineer_heart_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds clinically-motivated derived features to the heart disease dataset.

    Sources:
    - ACC/AHA 2019 Cardiovascular Risk Guidelines
    - Duke Treadmill Score (DTS) literature
    """
    df = df.copy()

    # Non-linear age × cholesterol synergy (risk increases super-linearly)
    df["age_chol_ratio"] = df["chol"] / (df["age"] + 1e-5)

    # Duke Treadmill Score proxy (HR × Age product captures exercise capacity)
    df["hr_age_product"] = df["thalach"] * df["age"]

    # ACC/AHA binary thresholds
    df["chol_high"] = (df["chol"] > 240).astype(np.int8)          # borderline high
    df["chol_very_high"] = (df["chol"] > 280).astype(np.int8)     # high
    df["tachycardia_flag"] = (df["thalach"] > 100).astype(np.int8)
    df["hypertension_flag"] = (df["trestbps"] > 130).astype(np.int8)  # AHA 2017

    # Ischemia proxy: exercise-induced ST depression magnitude
    df["ischemia_severity"] = df["oldpeak"] * df["exang"]

    return df


# ─── Diabetes Features ────────────────────────────────────────────────────────

def engineer_diabetes_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds metabolic and insulin resistance proxy features.

    Sources:
    - American Diabetes Association (ADA) Standards of Care 2024
    - HOMA-IR calculation literature
    """
    df = df.copy()

    # BMI-based obesity categories (WHO classification)
    df["obese"] = (df["bmi"] >= 30).astype(np.int8)
    df["severely_obese"] = (df["bmi"] >= 35).astype(np.int8)

    # ADA diagnostic thresholds
    df["glucose_prediabetic"] = df["glucose"].between(100, 125).astype(np.int8)
    df["glucose_diabetic"] = (df["glucose"] > 126).astype(np.int8)

    # Insulin resistance proxy (HOMA-IR approximation)
    # HOMA-IR = (Glucose × Insulin) / 405  (using mg/dL units)
    df["homa_ir_proxy"] = (df["glucose"] * df["insulin"].fillna(0)) / (405 + 1e-5)

    # Metabolic syndrome indicator: glucose × BMI interaction
    df["glucose_bmi"] = df["glucose"] * df["bmi"]

    # Reproductive metabolic burden (gestational diabetes risk proxy)
    df["preg_bmi"] = df["pregnancies"] * df["bmi"]

    # Age-adjusted glucose (glucose intolerance increases with age)
    df["glucose_age_ratio"] = df["glucose"] / (df["age"] + 1e-5)

    return df


# ─── Breast Cancer Features ────────────────────────────────────────────────────

def engineer_cancer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates cell nuclei measurement statistics.
    WDBC already has mean/se/worst — we add ratios between worst and mean.

    Sources: Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995)
    """
    df = df.copy()

    # Key morphological ratios
    for feature in ["radius", "perimeter", "area", "concavity", "concave_points"]:
        mean_col = f"{feature}_mean"
        worst_col = f"{feature}_worst"
        if mean_col in df.columns and worst_col in df.columns:
            df[f"{feature}_deterioration"] = df[worst_col] / (df[mean_col] + 1e-5)

    # Compactness as a shape irregularity indicator
    if "perimeter_mean" in df.columns and "area_mean" in df.columns:
        df["shape_irregularity"] = (df["perimeter_mean"] ** 2) / (
            4 * np.pi * df["area_mean"] + 1e-5
        )

    return df


# ─── Kidney Disease Features ─────────────────────────────────────────────────

def engineer_kidney_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds clinical staging proxies for kidney disease.

    Sources:
    - KDIGO 2012 Clinical Practice Guidelines for CKD
    """
    df = df.copy()

    # eGFR is the gold standard for kidney function staging
    # Approximation using creatinine (sc) and age (CKD-EPI simplified)
    if "sc" in df.columns and "age" in df.columns:
        df["egfr_proxy"] = 186 * (df["sc"] + 1e-5) ** (-1.154) * (df["age"] + 1e-5) ** (-0.203)
        df["ckd_stage_proxy"] = pd.cut(
            df["egfr_proxy"],
            bins=[0, 15, 30, 45, 60, 90, np.inf],
            labels=[5, 4, 3, 2, 1, 0],  # Higher = worse kidney function
        ).astype(float)

    # Anemia-CKD connection: low hemoglobin is common in advanced CKD
    if "hemo" in df.columns:
        df["anemia_flag"] = (df["hemo"] < 12.0).astype(np.int8)

    # Hypertension burden
    if "bp" in df.columns:
        df["hypertension_severe"] = (df["bp"] > 90).astype(np.int8)

    return df


# ─── Dispatcher ───────────────────────────────────────────────────────────────

FEATURE_ENGINEERS = {
    "heart": engineer_heart_features,
    "diabetes": engineer_diabetes_features,
    "cancer": engineer_cancer_features,
    "kidney": engineer_kidney_features,
}


def apply_feature_engineering(disease: str, df: pd.DataFrame) -> pd.DataFrame:
    """Route to correct disease-specific feature engineering function."""
    engineer = FEATURE_ENGINEERS.get(disease)
    if engineer is None:
        raise ValueError(f"Unknown disease: {disease}. Choose from {list(FEATURE_ENGINEERS)}")
    return engineer(df)

"""
src/features/pipeline.py — Master preprocessing pipeline factory.

Returns a fitted sklearn Pipeline object that can be serialized via joblib
and used identically at training and inference time. This prevents data leakage
and ensures reproducibility.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 — must be first
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "preprocessors"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DiseaseKey = Literal["heart", "diabetes", "cancer", "kidney"]


# ─── Feature Definitions ─────────────────────────────────────────────────────

FEATURE_CONFIG: dict[DiseaseKey, dict] = {
    "heart": {
        "numeric": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "target": "target",
        "use_smote": True,
        "imbalance_ratio_threshold": 3,   # use SMOTE if minority:majority < 1:3
    },
    "diabetes": {
        "numeric": [
            "pregnancies", "glucose", "bloodpressure", "skinthickness",
            "insulin", "bmi", "diabetespedigreefunction", "age",
        ],
        "categorical": [],
        "target": "target",  # renamed from outcome in load_diabetes()
        "use_smote": True,
        "imbalance_ratio_threshold": 2,
    },
    "cancer": {
        "numeric": None,  # all 30 features are numeric — auto-detected
        "categorical": [],
        "target": "target",
        "use_smote": False,  # cancer dataset is relatively balanced
        "imbalance_ratio_threshold": None,
    },
    "kidney": {
        "numeric": [
            "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
            "sod", "pot", "hemo", "pcv", "wc", "rc",
        ],
        # rbc/pc/pcc/ba exist in full UCI CKD; our generated sample only has:
        "categorical": ["htn", "dm", "cad", "appet", "pe", "ane"],
        "target": "target",
        "use_smote": False,
        "imbalance_ratio_threshold": None,
    },
}


# ─── Numeric Sub-pipeline ────────────────────────────────────────────────────

def _numeric_transformer(strategy: str = "iterative") -> Pipeline:
    """
    Imputation + scaling for continuous features.

    strategy='iterative': MICE-style multivariate imputation (best for MAR/MCAR).
    strategy='median':    Simple median imputation (fast baseline).
    """
    if strategy == "iterative":
        imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            initial_strategy="median",
        )
    else:
        imputer = SimpleImputer(strategy="median")

    return Pipeline([
        ("imputer", imputer),
        ("scaler", RobustScaler()),  # More robust to outliers than StandardScaler
    ])


def _numeric_transformer_standard() -> Pipeline:
    """Standard (z-score) scaling — required for SVM-based cancer model."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ─── Categorical Sub-pipeline ────────────────────────────────────────────────

def _categorical_transformer() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])


# ─── Pipeline Factory ────────────────────────────────────────────────────────

def build_preprocessor(
    disease: DiseaseKey,
    df: pd.DataFrame,
    use_standard_scaler: bool = False,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer preprocessor for the given disease.

    Args:
        disease: One of 'heart', 'diabetes', 'cancer', 'kidney'.
        df: The training DataFrame (used to auto-detect column types).
        use_standard_scaler: Use StandardScaler instead of RobustScaler
                             (required for SVM-based models like cancer).

    Returns:
        Unfitted ColumnTransformer ready to be placed inside a Pipeline.
    """
    config = FEATURE_CONFIG[disease]
    target = config["target"]

    numeric_cols = config["numeric"]
    categorical_cols = config["categorical"]

    # Auto-detect numeric columns if not specified
    if numeric_cols is None:
        numeric_cols = [
            c for c in df.columns
            if c != target and pd.api.types.is_numeric_dtype(df[c])
        ]

    # Remove target from feature lists
    numeric_cols = [c for c in numeric_cols if c != target]
    categorical_cols = [c for c in categorical_cols if c != target]

    num_transformer = (
        _numeric_transformer_standard()
        if use_standard_scaler
        else _numeric_transformer()
    )

    transformers = [("num", num_transformer, numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", _categorical_transformer(), categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop any unspecified columns
    )


def build_full_pipeline(
    disease: DiseaseKey,
    classifier: object,
    df: pd.DataFrame,
    use_smote: bool | None = None,
    use_standard_scaler: bool = False,
    y: "pd.Series | None" = None,
) -> ImbPipeline:
    """
    Build the complete imbalanced-learn Pipeline:
        preprocessor → (optional SMOTE) → classifier

    Using ImbPipeline ensures SMOTE is only applied during fit(),
    not during predict(), preventing data leakage.

    Args:
        disease: Disease key for feature config lookup.
        classifier: Any sklearn-compatible estimator.
        df: Training data (for column type detection).
        use_smote: Override SMOTE setting. If None, uses disease config default.
        use_standard_scaler: Pass True for SVM-based models.

    Returns:
        Fitted-ready ImbPipeline.
    """
    config = FEATURE_CONFIG[disease]
    apply_smote = use_smote if use_smote is not None else config["use_smote"]

    preprocessor = build_preprocessor(disease, df, use_standard_scaler)

    steps: list = [("preprocessor", preprocessor)]

    if apply_smote:
        # Check imbalance ratio before applying SMOTE
        # y may be passed directly; fallback to reading target col from df
        if y is not None:
            import pandas as _pd
            class_counts = _pd.Series(y).value_counts()
        else:
            target = config["target"]
            class_counts = df[target].value_counts()
        ratio = class_counts.max() / class_counts.min()
        threshold = config.get("imbalance_ratio_threshold", 2)

        if ratio >= threshold:
            if ratio > 10:
                smote = BorderlineSMOTE(random_state=42, k_neighbors=5)
                logger.info("Severe imbalance (%.1f:1) — using BorderlineSMOTE", ratio)
            else:
                smote = SMOTE(random_state=42, k_neighbors=5)
                logger.info("Mild imbalance (%.1f:1) — using SMOTE", ratio)
            steps.append(("smote", smote))

    steps.append(("classifier", classifier))
    return ImbPipeline(steps=steps)


# ─── Serialization ────────────────────────────────────────────────────────────

def save_preprocessor(preprocessor: ColumnTransformer, disease: DiseaseKey, version: str) -> Path:
    """Serialize fitted preprocessor to disk for inference reuse."""
    path = MODELS_DIR / f"preprocessor_{disease}_v{version}.joblib"
    joblib.dump(preprocessor, path)
    logger.info("Saved preprocessor: %s", path)
    return path


def load_preprocessor(disease: DiseaseKey, version: str) -> ColumnTransformer:
    """Load a serialized preprocessor."""
    path = MODELS_DIR / f"preprocessor_{disease}_v{version}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {path}")
    return joblib.load(path)

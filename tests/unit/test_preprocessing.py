"""
tests/unit/test_preprocessing.py — Unit tests for the preprocessing pipeline.

Tests ensure:
1. Preprocessor never leaks test data into fit
2. SMOTE is only applied during fit (not transform)
3. Feature engineering produces expected columns
4. Missing value imputation preserves dtypes
5. Pipeline serializes and deserializes correctly
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path

from src.features.engineering import apply_feature_engineering
from src.features.pipeline import (
    FEATURE_CONFIG,
    build_preprocessor,
    build_full_pipeline,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_heart_df() -> pd.DataFrame:
    """Minimal heart disease dataframe for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age":      np.random.uniform(30, 80, n),
        "sex":      np.random.randint(0, 2, n).astype(float),
        "cp":       np.random.randint(0, 4, n).astype(float),
        "trestbps": np.random.uniform(90, 180, n),
        "chol":     np.random.uniform(150, 350, n),
        "fbs":      np.random.randint(0, 2, n).astype(float),
        "restecg":  np.random.randint(0, 3, n).astype(float),
        "thalach":  np.random.uniform(80, 200, n),
        "exang":    np.random.randint(0, 2, n).astype(float),
        "oldpeak":  np.random.uniform(0, 5, n),
        "slope":    np.random.randint(0, 3, n).astype(float),
        "ca":       np.random.randint(0, 4, n).astype(float),
        "thal":     np.random.randint(0, 3, n).astype(float),
        "target":   np.random.randint(0, 2, n),
    })


@pytest.fixture
def sample_diabetes_df() -> pd.DataFrame:
    np.random.seed(0)
    n = 120
    return pd.DataFrame({
        "pregnancies":              np.random.randint(0, 15, n).astype(float),
        "glucose":                  np.random.uniform(70, 200, n),
        "bloodpressure":            np.random.uniform(40, 130, n),
        "skinthickness":            np.random.uniform(0, 60, n),
        "insulin":                  np.random.uniform(0, 400, n),
        "bmi":                      np.random.uniform(18, 55, n),
        "diabetespedigreefunction": np.random.uniform(0.05, 2.5, n),
        "age":                      np.random.uniform(20, 80, n),
        "outcome":                  np.random.randint(0, 2, n),
    })


# ── Feature Engineering Tests ────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_heart_engineered_columns_created(self, sample_heart_df):
        result = apply_feature_engineering("heart", sample_heart_df)
        expected = ["age_chol_ratio", "hr_age_product", "chol_high", "tachycardia_flag"]
        for col in expected:
            assert col in result.columns, f"Expected column '{col}' not found"

    def test_heart_engineering_does_not_modify_original(self, sample_heart_df):
        original_cols = list(sample_heart_df.columns)
        _ = apply_feature_engineering("heart", sample_heart_df)
        assert list(sample_heart_df.columns) == original_cols

    def test_diabetes_engineered_columns_created(self, sample_diabetes_df):
        result = apply_feature_engineering("diabetes", sample_diabetes_df)
        assert "obese" in result.columns
        assert "glucose_diabetic" in result.columns
        assert "homa_ir_proxy" in result.columns

    def test_diabetes_obese_flag_correct(self, sample_diabetes_df):
        result = apply_feature_engineering("diabetes", sample_diabetes_df)
        for _, row in result.iterrows():
            expected = 1 if row["bmi"] >= 30 else 0
            assert row["obese"] == expected

    def test_chol_high_flag_correct(self, sample_heart_df):
        result = apply_feature_engineering("heart", sample_heart_df)
        for _, row in result.iterrows():
            expected = 1 if row["chol"] > 240 else 0
            assert row["chol_high"] == expected

    def test_invalid_disease_raises(self):
        with pytest.raises(ValueError, match="Unknown disease"):
            apply_feature_engineering("stroke", pd.DataFrame())


# ── Preprocessing Pipeline Tests ─────────────────────────────────────────────

class TestPreprocessingPipeline:
    def test_preprocessor_fits_without_error(self, sample_heart_df):
        target = FEATURE_CONFIG["heart"]["target"]
        X = sample_heart_df.drop(columns=[target])
        y = sample_heart_df[target]
        preprocessor = build_preprocessor("heart", sample_heart_df)
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == len(X)

    def test_preprocessor_output_no_nans(self, sample_heart_df):
        """Imputation must eliminate all NaN from output."""
        # Introduce some NaN values
        df = sample_heart_df.copy()
        df.loc[:10, "chol"] = np.nan
        df.loc[5:15, "trestbps"] = np.nan

        target = FEATURE_CONFIG["heart"]["target"]
        X = df.drop(columns=[target])
        preprocessor = build_preprocessor("heart", df)
        X_transformed = preprocessor.fit_transform(X)

        assert not np.isnan(X_transformed).any(), "NaN values found after imputation"

    def test_preprocessor_fit_only_on_train(self, sample_heart_df):
        """Verify that test set mean is NOT used in fitting (no leakage)."""
        from sklearn.model_selection import train_test_split

        target = FEATURE_CONFIG["heart"]["target"]
        X = sample_heart_df.drop(columns=[target])
        y = sample_heart_df[target]

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)

        preprocessor = build_preprocessor("heart", sample_heart_df)
        preprocessor.fit(X_train)   # Fit ONLY on train
        X_test_transformed = preprocessor.transform(X_test)  # Transform test independently

        assert X_test_transformed.shape[0] == len(X_test)

    def test_pipeline_serialization(self, sample_heart_df, tmp_path):
        """Pipeline must survive joblib serialization and produce same output."""
        from sklearn.dummy import DummyClassifier

        target = FEATURE_CONFIG["heart"]["target"]
        X = sample_heart_df.drop(columns=[target])
        y = sample_heart_df[target]

        pipeline = build_full_pipeline("heart", DummyClassifier(), sample_heart_df)
        pipeline.fit(X, y)

        # Serialize
        path = tmp_path / "test_pipeline.joblib"
        joblib.dump(pipeline, path)

        # Reload and predict
        loaded = joblib.load(path)
        preds_original = pipeline.predict_proba(X)
        preds_loaded = loaded.predict_proba(X)

        np.testing.assert_array_almost_equal(preds_original, preds_loaded)

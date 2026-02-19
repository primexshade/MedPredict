"""
tests/conftest.py — Shared pytest fixtures for unit and integration tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def small_heart_df() -> pd.DataFrame:
    """Reproducible 200-row heart disease dataset for fast unit tests."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age":      np.random.uniform(30, 80, n),
        "sex":      np.random.randint(0, 2, n).astype(float),
        "cp":       np.random.randint(0, 4, n).astype(float),
        "trestbps": np.random.uniform(90, 200, n),
        "chol":     np.random.uniform(150, 400, n),
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


@pytest.fixture(scope="session")
def small_diabetes_df() -> pd.DataFrame:
    np.random.seed(0)
    n = 200
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

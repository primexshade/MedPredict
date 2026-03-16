"""
scripts/augment_data.py — Realistic synthetic data augmentation for all 4 disease datasets.

Strategy: Per-class multivariate Gaussian sampling.
  - Fit a mean vector and covariance matrix on each class independently from the real data.
  - Draw synthetic samples from that distribution.
  - Clip to domain-valid ranges (same constraints used by the data loaders).
  - Discretise binary/integer columns to preserve their categorical nature.
  - Append synthetic rows to the existing CSVs (originals are backed up first).

Usage:
    python scripts/augment_data.py               # augment all diseases
    python scripts/augment_data.py --disease heart  # one disease only
    python scripts/augment_data.py --n-factor 1.5   # target 1.5× current size
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"

# ─── Per-disease configs ────────────────────────────────────────────────────────

DISEASE_CONFIGS = {
    "heart": {
        "file":   "heart_kaggle.csv",
        "target": "target",
        # Columns that must be clipped to integer {0,1} after sampling
        "binary_cols": ["sex", "fbs", "exang"],
        # Columns clipped to specific integer ranges
        "int_range_cols": {
            "cp":      (0, 3),
            "restecg": (0, 2),
            "slope":   (0, 2),
            "ca":      (0, 4),
            "thal":    (0, 3),
        },
        # Continuous value clips (min, max)
        "clips": {
            "age":      (20, 90),
            "trestbps": (80, 220),
            "chol":     (100, 600),
            "thalach":  (60, 210),
            "oldpeak":  (0.0, 8.0),
        },
        "target_n": 2000,   # approximate final row count
    },
    "diabetes": {
        "file":   "diabetes.csv",
        "target": "Outcome",
        "binary_cols": [],
        "int_range_cols": {
            "Pregnancies": (0, 17),
        },
        "clips": {
            "Glucose":                  (44, 400),
            "BloodPressure":            (24, 180),
            "SkinThickness":            (0,  99),
            "Insulin":                  (0, 900),
            "BMI":                      (10, 70),
            "DiabetesPedigreeFunction": (0.05, 2.5),
            "Age":                      (21, 85),
        },
        "target_n": 1500,
    },
    "cancer": {
        "file":   "cancer_kaggle.csv",
        "target": "diagnosis",
        "binary_cols": [],
        "int_range_cols": {},
        "clips": {},   # All continuous — rely on non-negativity only
        "non_negative_cols": True,   # All morphological features must be ≥ 0
        "target_n": 1100,
    },
    "kidney": {
        "file":   "kidney_disease.csv",
        "target": "classification",
        # Categorical columns handled separately (sampled from observed freq)
        "categorical_cols": ["htn", "dm", "cad", "appet", "pe", "ane"],
        "binary_cols": [],
        "int_range_cols": {},
        "clips": {
            "age":  (2, 90),
            "bp":   (50, 180),
            "sg":   (1.005, 1.025),
            "al":   (0, 5),
            "su":   (0, 5),
            "bgr":  (22, 500),
            "bu":   (1.5, 391),
            "sc":   (0.4, 76),
            "sod":  (111, 163),
            "pot":  (2.5, 47),
            "hemo": (3.1, 17.8),
            "pcv":  (9, 54),
            "wc":   (2200, 26400),
            "rc":   (2.1, 8),
        },
        "target_n": 850,
    },
}


# ─── Core augmentation logic ────────────────────────────────────────────────────

def _sample_class(
    class_df: pd.DataFrame,
    n_synthetic: int,
    numeric_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample n_synthetic rows from a multivariate Gaussian fit to class_df[numeric_cols].
    Returns a DataFrame with only numeric_cols.
    """
    # Drop columns that are entirely NaN (e.g. Unnamed: 32 in cancer_kaggle.csv)
    valid_cols = [c for c in numeric_cols if class_df[c].notna().any()]
    data = class_df[valid_cols].dropna().values.astype(float)
    if len(data) < 2:
        logger.warning("Too few rows to fit Gaussian after dropping NaN columns; skipping.")
        return pd.DataFrame(columns=numeric_cols)

    mean = data.mean(axis=0)
    cov = np.cov(data, rowvar=False)

    # Regularise covariance for numerical stability
    cov += np.eye(len(mean)) * 1e-6

    samples = rng.multivariate_normal(mean, cov, size=n_synthetic)
    synth = pd.DataFrame(samples, columns=valid_cols)
    # Add back any columns that were all-NaN (preserve schema)
    for c in numeric_cols:
        if c not in synth.columns:
            synth[c] = np.nan
    return synth[numeric_cols]


def _post_process(
    synth_df: pd.DataFrame,
    config: dict,
    class_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply clipping, discretisation, and categorical resampling."""
    # Clip continuous features
    for col, (lo, hi) in config.get("clips", {}).items():
        if col in synth_df.columns:
            synth_df[col] = synth_df[col].clip(lo, hi)

    # Non-negative constraint for all-numeric datasets (cancer)
    if config.get("non_negative_cols"):
        synth_df = synth_df.clip(lower=0)

    # Binarise binary columns
    for col in config.get("binary_cols", []):
        if col in synth_df.columns:
            synth_df[col] = (synth_df[col] >= 0.5).astype(float)

    # Integer-range columns (e.g., cp ∈ {0,1,2,3})
    for col, (lo, hi) in config.get("int_range_cols", {}).items():
        if col in synth_df.columns:
            synth_df[col] = synth_df[col].round().clip(lo, hi).astype(float)

    # Categorical columns — resample from observed proportions in this class
    for col in config.get("categorical_cols", []):
        if col in class_df.columns:
            value_counts = class_df[col].value_counts(normalize=True)
            synth_df[col] = rng.choice(
                value_counts.index.tolist(),
                p=value_counts.values,
                size=len(synth_df),
            )

    return synth_df


def augment_disease(disease: str, n_factor: float, rng: np.random.Generator) -> None:
    """
    Load the existing dataset for `disease`, augment to reach ~target_n rows,
    and overwrite the CSV (backing up the original first).
    """
    config = DISEASE_CONFIGS[disease]
    path = RAW_DIR / config["file"]

    if not path.exists():
        logger.warning("File not found, skipping %s: %s", disease, path)
        return

    # ── 1. Load existing data ──────────────────────────────────────────────────
    df = pd.read_csv(path)
    original_count = len(df)

    # Determine target size
    target_n = int(original_count * n_factor)
    if target_n <= original_count:
        logger.info("%-8s: already at target size (%d rows). Skipping.", disease, original_count)
        return

    n_to_add = target_n - original_count
    logger.info(
        "%-8s: %d existing → %d target (+%d synthetic rows)",
        disease, original_count, target_n, n_to_add,
    )

    # ── 2. Backup original ────────────────────────────────────────────────────
    backup = path.with_suffix(".csv.bak")
    if not backup.exists():
        shutil.copy(path, backup)
        logger.info("  Backed up to %s", backup.name)

    # ── 3. Identify columns ───────────────────────────────────────────────────
    target_col = config["target"]

    # Drop unnamed/artifact columns before processing (e.g. Unnamed: 32 in cancer)
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        logger.info("  Dropped artifact columns: %s", unnamed_cols)

    # All categorical columns for kidney (not numeric)
    exclude_cols = {target_col} | set(config.get("categorical_cols", []))
    numeric_cols = [c for c in df.columns if c not in exclude_cols]

    # For cancer, also drop the id column
    if disease == "cancer":
        numeric_cols = [c for c in numeric_cols if c.lower() != "id"]

    # ── 4. Per-class sampling ─────────────────────────────────────────────────
    classes = df[target_col].unique()
    class_counts = df[target_col].value_counts()
    synthetic_parts = []
    remaining = n_to_add

    for i, cls in enumerate(classes):
        class_df = df[df[target_col] == cls]
        # Distribute proportionally; last class gets whatever is left
        if i < len(classes) - 1:
            n_cls = int(n_to_add * (len(class_df) / original_count))
        else:
            n_cls = remaining
        remaining -= n_cls
        if n_cls < 1:
            continue

        synth = _sample_class(class_df, n_cls, numeric_cols, rng)
        if synth.empty:
            continue

        synth = _post_process(synth, config, class_df, rng)
        synth[target_col] = cls

        # Make sure column order matches original
        extra = [c for c in df.columns if c not in synth.columns]
        for c in extra:
            synth[c] = np.nan
        synth = synth[df.columns]

        synthetic_parts.append(synth)
        logger.info("  class=%s  generated %d rows", cls, len(synth))

    if not synthetic_parts:
        logger.error("  No synthetic rows generated for %s!", disease)
        return

    # ── 5. Append & save ──────────────────────────────────────────────────────
    augmented = pd.concat([df, *synthetic_parts], ignore_index=True)
    augmented.to_csv(path, index=False)
    logger.info(
        "  ✓ %s saved: %d rows (+%d added)",
        path.name, len(augmented), len(augmented) - original_count,
    )


# ─── CLI entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Augment disease prediction datasets.")
    parser.add_argument(
        "--disease",
        default="all",
        choices=["all", *DISEASE_CONFIGS.keys()],
        help="Which disease dataset(s) to augment.",
    )
    parser.add_argument(
        "--n-factor",
        type=float,
        default=None,
        help="Multiplier for existing dataset size (e.g. 2.0 = double). "
             "Overrides per-disease target_n if set.",
    )
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    diseases = list(DISEASE_CONFIGS.keys()) if args.disease == "all" else [args.disease]

    # Override n_factor per disease if provided via CLI
    if args.n_factor is not None:
        for d in diseases:
            # We'll pass n_factor directly to augment_disease
            pass

    for d in diseases:
        n_factor = args.n_factor
        if n_factor is None:
            # Use per-disease target_n to compute effective factor
            path = RAW_DIR / DISEASE_CONFIGS[d]["file"]
            if path.exists():
                current = len(pd.read_csv(path))
                target_n = DISEASE_CONFIGS[d]["target_n"]
                n_factor = target_n / current
            else:
                n_factor = 2.0
        augment_disease(d, n_factor, rng)

    logger.info("\nDone! Augmented files are in: %s", RAW_DIR)


if __name__ == "__main__":
    main()

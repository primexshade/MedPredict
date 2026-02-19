"""
src/models/train.py — Model training orchestration with Optuna and MLflow.

Implements nested cross-validation for unbiased performance estimation
and Bayesian hyperparameter optimization (Optuna) for all disease models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.config import get_settings
from src.features.engineering import apply_feature_engineering
from src.features.pipeline import FEATURE_CONFIG, build_full_pipeline
from src.models.evaluate import compute_metrics

logger = logging.getLogger(__name__)
settings = get_settings()

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TrainingResult:
    run_id: str
    model_version: str
    metrics: dict[str, float]
    best_params: dict[str, Any]
    cv_mean: float
    cv_std: float


# ─── Hyperparameter Search Spaces ────────────────────────────────────────────

def _xgboost_params(trial: optuna.Trial, scale_pos_weight: float) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 3),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }


def _lgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "class_weight": "balanced",
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }


MODEL_BUILDERS = {
    "heart": lambda trial, spw: XGBClassifier(**_xgboost_params(trial, spw)),
    "diabetes": lambda trial, _: LGBMClassifier(**_lgbm_params(trial)),
    "cancer": lambda trial, _: XGBClassifier(**_xgboost_params(trial, 1.0)),
    "kidney": lambda trial, spw: XGBClassifier(**_xgboost_params(trial, spw)),
}


# ─── Optuna Objective ─────────────────────────────────────────────────────────

def _make_objective(
    disease: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
) -> Any:
    """Returns an Optuna objective function for the given disease."""

    def objective(trial: optuna.Trial) -> float:
        model = MODEL_BUILDERS[disease](trial, scale_pos_weight)
        pipeline = build_full_pipeline(disease, model, X_train)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for fold_train_idx, fold_val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[fold_train_idx], X_train.iloc[fold_val_idx]
            y_tr, y_val = y_train.iloc[fold_train_idx], y_train.iloc[fold_val_idx]

            pipeline.fit(X_tr, y_tr)
            y_proba = pipeline.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, y_proba))

            # Prune unpromising trials early
            trial.report(np.mean(scores), step=len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    return objective


# ─── Main Training Function ──────────────────────────────────────────────────

def train_disease_model(
    disease: str,
    df: pd.DataFrame,
    n_trials: int = 80,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> TrainingResult:
    """
    Full training pipeline:
    1. Feature engineering
    2. Stratified train/val/test split
    3. Optuna hyperparameter tuning (inner 3-fold CV on train set)
    4. Nested CV for unbiased performance estimation
    5. Final model training on train+val
    6. Calibration on val set
    7. MLflow logging and model registration

    Args:
        disease: 'heart' | 'diabetes' | 'cancer' | 'kidney'
        df: Raw loaded DataFrame.
        n_trials: Number of Optuna trials.
        test_size: Fraction held out as final test set (never touched during tuning).
        val_size: Fraction used for calibration (from non-test portion).

    Returns:
        TrainingResult with run_id, metrics, and best params.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"disease-prediction-{disease}")

    # 1. Feature engineering
    df = apply_feature_engineering(disease, df)

    target_col = FEATURE_CONFIG[disease]["target"]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # 2. Stratified split: train(70%) / val(15%) / test(15%)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_fraction, stratify=y_tv, random_state=42
    )

    logger.info(
        "Disease: %s | Train: %d | Val: %d | Test: %d",
        disease, len(X_train), len(X_val), len(X_test),
    )

    # Class imbalance ratio for scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    # 3. Outer CV for performance estimation (nested CV)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_scores: list[float] = []

    for fold_i, (tr_idx, val_idx) in enumerate(outer_cv.split(X_tv, y_tv)):
        X_outer_tr = X_tv.iloc[tr_idx]
        y_outer_tr = y_tv.iloc[tr_idx]
        X_outer_val = X_tv.iloc[val_idx]
        y_outer_val = y_tv.iloc[val_idx]

        # Inner Optuna tuning on outer training fold
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            _make_objective(disease, X_outer_tr, y_outer_tr, scale_pos_weight),
            n_trials=n_trials // 5,  # Distribute trials across folds
            show_progress_bar=False,
        )

        best_model = MODEL_BUILDERS[disease](
            optuna.trial.FixedTrial(study.best_params), scale_pos_weight
        )
        pipeline = build_full_pipeline(disease, best_model, X_outer_tr)
        pipeline.fit(X_outer_tr, y_outer_tr)
        y_proba = pipeline.predict_proba(X_outer_val)[:, 1]
        fold_auc_pr = average_precision_score(y_outer_val, y_proba)
        outer_scores.append(fold_auc_pr)
        logger.info("  Outer fold %d AUC-PR: %.4f", fold_i + 1, fold_auc_pr)

    cv_mean = float(np.mean(outer_scores))
    cv_std = float(np.std(outer_scores))
    logger.info("Nested CV AUC-PR: %.4f ± %.4f", cv_mean, cv_std)

    # 4. Final tuning on full train+val data
    study_final = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study_final.optimize(
        _make_objective(disease, X_train, y_train, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best_params = study_final.best_params
    logger.info("Best Optuna params: %s", best_params)

    # 5. Train final model + calibrate on val set
    final_model = MODEL_BUILDERS[disease](
        optuna.trial.FixedTrial(best_params), scale_pos_weight
    )
    pipeline = build_full_pipeline(disease, final_model, X_train)
    pipeline.fit(X_train, y_train)

    # Calibrate probabilities using isotonic regression on validation set
    # Note: CalibratedClassifierCV wraps the whole pipeline
    calibrated = CalibratedClassifierCV(
        estimator=pipeline,
        method="isotonic",
        cv="prefit",
    )
    calibrated.fit(X_val, y_val)

    # 6. Final evaluation on held-out test set
    y_proba_test = calibrated.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)
    test_metrics = compute_metrics(y_test.values, y_proba_test, y_pred_test)

    logger.info("Test set metrics: %s", test_metrics)

    # 7. MLflow logging
    with mlflow.start_run() as run:
        mlflow.log_params({**best_params, "disease": disease, "n_trials": n_trials})
        mlflow.log_metrics({**test_metrics, "cv_auc_pr_mean": cv_mean, "cv_auc_pr_std": cv_std})

        mlflow.sklearn.log_model(
            calibrated,
            artifact_path="model",
            registered_model_name=settings.registered_models.get(disease, f"dp-{disease}"),
        )

        run_id = run.info.run_id

    return TrainingResult(
        run_id=run_id,
        model_version=f"{disease}_v1.0",
        metrics=test_metrics,
        best_params=best_params,
        cv_mean=cv_mean,
        cv_std=cv_std,
    )

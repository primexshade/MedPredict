"""
src/models/evaluate.py — Healthcare-specific model evaluation metrics.

Implements the full metrics suite required for clinical ML validation:
AUC-PR (primary), AUC-ROC, sensitivity, specificity, PPV, NPV, MCC,
Brier Score, ECE (expected calibration error), and per-subgroup fairness.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute the full clinical evaluation metrics suite.

    Args:
        y_true:  Ground truth binary labels (0/1).
        y_proba: Predicted probabilities for positive class.
        y_pred:  Binary predictions at chosen threshold.

    Returns:
        Dictionary of metric name → value.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / max(tp + fn, 1)          # Recall / True Positive Rate
    specificity = tn / max(tn + fp, 1)          # True Negative Rate
    ppv = tp / max(tp + fp, 1)                  # Precision / Positive Predictive Value
    npv = tn / max(tn + fn, 1)                  # Negative Predictive Value

    ece = _expected_calibration_error(y_true, y_proba, n_bins=10)

    return {
        # ── Discrimination ────────────────────────────────────────────────────
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        # ── At-threshold classification ───────────────────────────────────────
        "sensitivity": float(sensitivity),      # Most critical: don't miss sick patients
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(2 * ppv * sensitivity / max(ppv + sensitivity, 1e-5)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        # ── Calibration ───────────────────────────────────────────────────────
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "ece": float(ece),
        # ── Raw counts (for audit) ────────────────────────────────────────────
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_samples": int(len(y_true)),
        "prevalence": float(y_true.mean()),
    }


def _expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual outcome frequencies.
    ECE = Σ (|Bm|/n) × |acc(Bm) − conf(Bm)|

    A well-calibrated model has ECE < 0.05.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        in_bin = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = y_proba[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)

    return ece


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "youden",
    min_sensitivity: float = 0.85,
) -> float:
    """
    Find the classification threshold that optimizes the chosen strategy.

    Args:
        strategy:
            'youden'      — Maximizes Youden's J = Sensitivity + Specificity - 1
                           Best for balanced clinical importance of errors.
            'min_sens'    — Minimum sensitivity subject to Specificity ≥ 0.75
                           Best for screening (don't miss sick patients).
            'f1'          — Maximizes F1 score.
        min_sensitivity: Used only with 'min_sens' strategy.

    Returns:
        Optimal threshold value ∈ [0, 1].
    """
    from sklearn.metrics import roc_curve, precision_recall_curve

    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr  # Youden's J
        return float(thresholds[np.argmax(j_scores)])

    if strategy == "min_sens":
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        specificity = 1 - fpr
        valid = specificity >= (1 - (1 - min_sensitivity))
        if not valid.any():
            return 0.5
        # Among thresholds with sufficient sensitivity, maximize specificity
        return float(thresholds[valid][np.argmax(specificity[valid])])

    if strategy == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        return float(thresholds[np.argmax(f1_scores[:-1])])

    raise ValueError(f"Unknown threshold strategy: {strategy}")


def evaluate_fairness(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    sensitive_feature: np.ndarray,
    group_names: dict[int, str] | None = None,
) -> dict[str, dict]:
    """
    Compute per-subgroup metrics for fairness auditing.

    Args:
        sensitive_feature: Array of group labels (e.g., 0=female, 1=male).
        group_names: Optional mapping of group label → name.

    Returns:
        Dict with 'by_group', 'max_sensitivity_gap', 'equalized_odds_diff'.
    """
    groups = np.unique(sensitive_feature)
    by_group: dict = {}

    for g in groups:
        mask = sensitive_feature == g
        name = (group_names or {}).get(int(g), str(g))
        by_group[name] = compute_metrics(y_true[mask], y_proba[mask], y_pred[mask])

    sensitivities = [v["sensitivity"] for v in by_group.values()]
    specificities = [v["specificity"] for v in by_group.values()]

    return {
        "by_group": by_group,
        "max_sensitivity_gap": float(max(sensitivities) - min(sensitivities)),
        "max_specificity_gap": float(max(specificities) - min(specificities)),
        "equalized_odds_diff": float(
            max(
                max(sensitivities) - min(sensitivities),
                max(specificities) - min(specificities),
            )
        ),
    }

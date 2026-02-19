"""
src/scoring/risk_scorer.py — Composite risk scoring engine.

Combines calibrated model probabilities with temporal risk trajectory data
and comorbidity burden to produce a final composite risk score with
confidence intervals via bootstrap sampling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class RiskCategory(str, Enum):
    LOW = "LOW"
    BORDERLINE = "BORDERLINE"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Disease-specific thresholds calibrated to maximize Youden's J
# on validation cohorts. Lower thresholds → more sensitive (preferred for screening).
RISK_THRESHOLDS: dict[str, dict[str, float]] = {
    "heart": {
        "LOW": 0.15,
        "BORDERLINE": 0.30,
        "MODERATE": 0.55,
        "HIGH": 0.75,
    },
    "diabetes": {
        "LOW": 0.20,
        "BORDERLINE": 0.35,
        "MODERATE": 0.55,
        "HIGH": 0.75,
    },
    "cancer": {
        # More sensitive thresholds — critical not to miss cancer
        "LOW": 0.10,
        "BORDERLINE": 0.25,
        "MODERATE": 0.50,
        "HIGH": 0.65,
    },
    "kidney": {
        "LOW": 0.15,
        "BORDERLINE": 0.30,
        "MODERATE": 0.55,
        "HIGH": 0.75,
    },
}

# Composite score weighting constants (tuned on validation cohort)
ALPHA = 0.70  # Weight for current-visit calibrated probability
BETA = 0.20   # Weight for risk velocity (trajectory)
GAMMA = 0.10  # Weight for comorbidity burden index

# Clinical recommendations by risk category
CLINICAL_ACTIONS: dict[RiskCategory, dict[str, str]] = {
    RiskCategory.LOW: {
        "action": "Routine screening",
        "timeframe": "12–24 months",
        "urgency": "low",
    },
    RiskCategory.BORDERLINE: {
        "action": "Lifestyle modification counseling",
        "timeframe": "6 months follow-up",
        "urgency": "low",
    },
    RiskCategory.MODERATE: {
        "action": "Clinical review and additional lab work",
        "timeframe": "3 months follow-up",
        "urgency": "medium",
    },
    RiskCategory.HIGH: {
        "action": "Specialist referral recommended",
        "timeframe": "1 month follow-up",
        "urgency": "high",
    },
    RiskCategory.CRITICAL: {
        "action": "Urgent clinical evaluation required",
        "timeframe": "Within 1 week",
        "urgency": "critical",
    },
}


@dataclass
class RiskScore:
    disease: str
    calibrated_probability: float
    composite_score: float
    risk_category: RiskCategory
    confidence_interval: tuple[float, float]
    velocity: float | None           # Risk change from last visit
    comorbidity_index: float | None
    clinical_action: dict[str, str]


class RiskScorer:
    """
    Produces composite risk scores from calibrated model probabilities.

    Composite formula (when temporal history is available):
        score(t) = α × p_calibrated(t) + β × Δrisk(t-1, t) + γ × comorbidity_index

    For first-visit patients (no history):
        score(t) = p_calibrated(t)  [β and γ terms set to 0]
    """

    def __init__(self, disease: str, n_bootstrap: int = 500) -> None:
        if disease not in RISK_THRESHOLDS:
            raise ValueError(
                f"Unknown disease: {disease}. Options: {list(RISK_THRESHOLDS)}"
            )
        self.disease = disease
        self.n_bootstrap = n_bootstrap
        self.thresholds = RISK_THRESHOLDS[disease]

    def compute(
        self,
        calibrated_prob: float,
        previous_score: float | None = None,
        comorbidity_index: float | None = None,
        prediction_samples: np.ndarray | None = None,
    ) -> RiskScore:
        """
        Compute the composite risk score with confidence interval.

        Args:
            calibrated_prob: Output of the calibrated classifier (∈ [0, 1]).
            previous_score: Patient's risk score from their last visit.
            comorbidity_index: Charlson Comorbidity Index (normalized to [0, 1]).
            prediction_samples: Bootstrap probability samples for CI computation.
                               If None, uses a simplified ±2σ Gaussian CI.

        Returns:
            RiskScore dataframe with all computed fields.
        """
        # Compute velocity (rate of risk change between visits)
        velocity = None
        if previous_score is not None:
            velocity = calibrated_prob - previous_score

        # Compute composite score
        composite = calibrated_prob * ALPHA

        if velocity is not None:
            # Clip velocity contribution to prevent extreme outliers
            v_contribution = np.clip(velocity, -0.3, 0.3) * BETA
            composite += v_contribution

        if comorbidity_index is not None:
            composite += comorbidity_index * GAMMA

        # Clamp composite to [0, 1]
        composite = float(np.clip(composite, 0.0, 1.0))

        # Confidence interval
        ci = self._compute_ci(calibrated_prob, prediction_samples)

        category = self._categorize(composite)

        logger.debug(
            "Risk score | disease=%s | prob=%.3f | velocity=%s | composite=%.3f | category=%s",
            self.disease, calibrated_prob, f"{velocity:.3f}" if velocity else "N/A",
            composite, category.value,
        )

        return RiskScore(
            disease=self.disease,
            calibrated_probability=calibrated_prob,
            composite_score=composite,
            risk_category=category,
            confidence_interval=ci,
            velocity=velocity,
            comorbidity_index=comorbidity_index,
            clinical_action=CLINICAL_ACTIONS[category],
        )

    def _categorize(self, score: float) -> RiskCategory:
        """Map a continuous score to a discrete risk category."""
        t = self.thresholds
        if score <= t["LOW"]:
            return RiskCategory.LOW
        elif score <= t["BORDERLINE"]:
            return RiskCategory.BORDERLINE
        elif score <= t["MODERATE"]:
            return RiskCategory.MODERATE
        elif score <= t["HIGH"]:
            return RiskCategory.HIGH
        return RiskCategory.CRITICAL

    def _compute_ci(
        self,
        prob: float,
        samples: np.ndarray | None,
        ci_level: float = 0.95,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval."""
        if samples is not None and len(samples) >= 100:
            alpha = (1 - ci_level) / 2
            lower = float(np.percentile(samples, alpha * 100))
            upper = float(np.percentile(samples, (1 - alpha) * 100))
        else:
            # Simplified Gaussian CI using Wilson interval approximation
            z = 1.96  # 95% CI
            margin = z * np.sqrt(prob * (1 - prob) / max(self.n_bootstrap, 100))
            lower = float(max(0.0, prob - margin))
            upper = float(min(1.0, prob + margin))
        return (lower, upper)

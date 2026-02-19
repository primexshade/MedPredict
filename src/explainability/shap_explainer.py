"""
src/explainability/shap_explainer.py — SHAP-based model explainability.

Returns per-prediction feature contributions with theoretical guarantees
(Shapley values from cooperative game theory). TreeExplainer is O(TLD)
efficient — suitable for production inference latency requirements.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    feature: str
    value: float
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk" | "neutral"
    rank: int


@dataclass
class ExplanationResult:
    base_value: float
    predicted_probability: float
    contributions: list[FeatureContribution]
    plain_english_summary: str


class DiseaseExplainer:
    """
    SHAP TreeExplainer for tree-based disease prediction models.

    Usage:
        explainer = DiseaseExplainer(model, X_background, feature_names)
        result = explainer.explain(X_patient)
    """

    def __init__(
        self,
        model: object,
        X_background: pd.DataFrame,
        feature_names: list[str],
        disease: str = "heart",
    ) -> None:
        self.feature_names = feature_names
        self.disease = disease

        # TreeExplainer: works natively with XGBoost, LightGBM, RandomForest
        # "interventional" perturbation: causal interpretation aligned with
        # realistic feature distributions from background data
        self.explainer = shap.TreeExplainer(
            model,
            data=X_background,
            feature_perturbation="interventional",
        )
        logger.info("Initialized SHAP TreeExplainer for disease: %s", disease)

    def explain(self, X: pd.DataFrame, top_k: int = 5) -> ExplanationResult:
        """
        Compute SHAP values for a single patient prediction.

        Args:
            X: Single-row DataFrame of preprocessed features.
            top_k: Number of top contributing features to return.

        Returns:
            ExplanationResult with ranked feature contributions.
        """
        if len(X) != 1:
            raise ValueError("explain() expects a single-row DataFrame. Use explain_batch() for multiple.")

        shap_values = self.explainer.shap_values(X)

        # Handle models that return list of arrays (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class SHAP values

        base_value = float(self.explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1])  # Positive class base value

        shap_row = shap_values[0]
        predicted_prob = base_value + float(shap_row.sum())
        # Clamp to valid probability range
        predicted_prob = float(np.clip(predicted_prob, 0.0, 1.0))

        # Rank by absolute SHAP magnitude
        ranked_indices = np.argsort(np.abs(shap_row))[::-1][:top_k]

        contributions = []
        for rank, idx in enumerate(ranked_indices, start=1):
            shap_val = float(shap_row[idx])
            feat_name = self.feature_names[idx]
            feat_value = float(X.iloc[0, idx])

            if abs(shap_val) < 0.001:
                direction = "neutral"
            elif shap_val > 0:
                direction = "increases_risk"
            else:
                direction = "decreases_risk"

            contributions.append(FeatureContribution(
                feature=feat_name,
                value=feat_value,
                shap_value=shap_val,
                direction=direction,
                rank=rank,
            ))

        summary = self._generate_plain_english(predicted_prob, contributions)

        return ExplanationResult(
            base_value=base_value,
            predicted_probability=predicted_prob,
            contributions=contributions,
            plain_english_summary=summary,
        )

    def explain_batch(self, X: pd.DataFrame) -> list[list[FeatureContribution]]:
        """Compute SHAP values for a batch of patients (for population analysis)."""
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return shap_values

    def _generate_plain_english(
        self,
        prob: float,
        contributions: list[FeatureContribution],
    ) -> str:
        """Generate a clinician-readable explanation from SHAP contributions."""
        risk_label = (
            "LOW" if prob < 0.21 else
            "BORDERLINE" if prob < 0.41 else
            "MODERATE" if prob < 0.66 else
            "HIGH" if prob < 0.81 else
            "CRITICAL"
        )

        risk_drivers = [c for c in contributions if c.direction == "increases_risk"]
        protective = [c for c in contributions if c.direction == "decreases_risk"]

        primary_driver = risk_drivers[0].feature.replace("_", " ") if risk_drivers else "unknown"

        summary = (
            f"Predicted {self.disease.replace('_', ' ')} risk: {prob:.0%} ({risk_label}). "
        )

        if risk_drivers:
            driver_list = ", ".join(
                f"{c.feature.replace('_', ' ')} ({c.value:.1f})"
                for c in risk_drivers[:3]
            )
            summary += f"Primary risk drivers: {driver_list}. "

        if protective:
            prot_list = ", ".join(
                c.feature.replace("_", " ") for c in protective[:2]
            )
            summary += f"Protective factors: {prot_list}."

        return summary

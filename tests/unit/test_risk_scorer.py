"""
tests/unit/test_risk_scorer.py — Unit tests for the composite risk scoring engine.
"""
from __future__ import annotations

import pytest
from src.scoring.risk_scorer import RiskCategory, RiskScorer


class TestRiskScorer:
    @pytest.fixture
    def scorer_heart(self) -> RiskScorer:
        return RiskScorer("heart")

    def test_low_probability_gives_low_category(self, scorer_heart):
        result = scorer_heart.compute(calibrated_prob=0.05)
        assert result.risk_category == RiskCategory.LOW

    def test_high_probability_no_velocity_gives_high_category(self, scorer_heart):
        # composite = 0.95 * 0.70 = 0.665 → HIGH (0.55–0.75)
        result = scorer_heart.compute(calibrated_prob=0.95)
        assert result.risk_category == RiskCategory.HIGH

    def test_critical_with_velocity_boost(self, scorer_heart):
        # composite = 0.90 * 0.70 + 0.25 * 0.20 = 0.63 + 0.05 = 0.68 → still HIGH
        # To reach CRITICAL (>0.75): prob=0.95, previous=0.60 → velocity=0.35 (capped to 0.30)
        # composite = 0.95*0.70 + 0.30*0.20 = 0.665 + 0.06 = 0.725 still HIGH
        # Actually: prob=0.95, prev=0.50 → velocity=0.45, clipped=0.30
        # composite = 0.665 + 0.060 = 0.725 → HIGH
        # With comorbidity: 0.725 + 0.10*1.0 = 0.825 → CRITICAL
        result = scorer_heart.compute(
            calibrated_prob=0.95,
            previous_score=0.50,
            comorbidity_index=1.0,
        )
        assert result.risk_category == RiskCategory.CRITICAL

    def test_composite_score_clamped_to_unit_interval(self, scorer_heart):
        result = scorer_heart.compute(calibrated_prob=1.0, comorbidity_index=1.0)
        assert 0.0 <= result.composite_score <= 1.0

    def test_velocity_computed_correctly(self, scorer_heart):
        result = scorer_heart.compute(calibrated_prob=0.6, previous_score=0.4)
        assert abs(result.velocity - 0.2) < 1e-5

    def test_no_previous_score_velocity_is_none(self, scorer_heart):
        result = scorer_heart.compute(calibrated_prob=0.5)
        assert result.velocity is None

    def test_confidence_interval_valid(self, scorer_heart):
        result = scorer_heart.compute(calibrated_prob=0.5)
        lo, hi = result.confidence_interval
        assert 0.0 <= lo <= hi <= 1.0

    def test_clinical_action_returned(self, scorer_heart):
        result = scorer_heart.compute(calibrated_prob=0.9)
        assert "action" in result.clinical_action
        assert "urgency" in result.clinical_action

    def test_invalid_disease_raises(self):
        with pytest.raises(ValueError, match="Unknown disease"):
            RiskScorer("flu")

    def test_all_diseases_supported(self):
        for disease in ["heart", "diabetes", "cancer", "kidney"]:
            scorer = RiskScorer(disease)
            result = scorer.compute(calibrated_prob=0.5)
            assert result.disease == disease

    @pytest.mark.parametrize("prob,expected_category", [
        (0.05, RiskCategory.LOW),           # composite = 0.05*0.70 = 0.035 < 0.15 → LOW
        (0.25, RiskCategory.BORDERLINE),    # composite = 0.25*0.70 = 0.175, 0.15–0.30 → BORDERLINE
        (0.55, RiskCategory.MODERATE),      # composite = 0.55*0.70 = 0.385, 0.30–0.55 → MODERATE
        (0.90, RiskCategory.HIGH),          # composite = 0.90*0.70 = 0.63, 0.55–0.75 → HIGH
    ])
    def test_category_thresholds(self, scorer_heart, prob, expected_category):
        result = scorer_heart.compute(calibrated_prob=prob)
        assert result.risk_category == expected_category

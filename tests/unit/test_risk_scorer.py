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

    def test_critical_probability_gives_critical_category(self, scorer_heart):
        result = scorer_heart.compute(calibrated_prob=0.95)
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
        (0.05, RiskCategory.LOW),
        (0.20, RiskCategory.BORDERLINE),
        (0.45, RiskCategory.MODERATE),
        (0.80, RiskCategory.CRITICAL),
    ])
    def test_category_thresholds(self, scorer_heart, prob, expected_category):
        result = scorer_heart.compute(calibrated_prob=prob)
        assert result.risk_category == expected_category

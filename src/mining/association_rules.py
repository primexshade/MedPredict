"""
src/mining/association_rules.py — FP-Growth association rule mining
for comorbidity pattern discovery.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

logger = logging.getLogger(__name__)


@dataclass
class MiningConfig:
    min_support: float = 0.05      # ≥5% of patients have this pattern
    min_confidence: float = 0.60   # 60% conditional probability
    min_lift: float = 1.5          # 50% more likely than chance alone
    max_antecedent_length: int = 3 # Limit rule complexity


def mine_comorbidity_rules(
    patient_condition_matrix: pd.DataFrame,
    config: MiningConfig | None = None,
) -> pd.DataFrame:
    """
    Discover comorbidity association rules using FP-Growth.

    FP-Growth advantage over Apriori:
    - Compresses the transaction database into a prefix tree (FP-tree)
    - Only two database scans (vs. exponential candidate generation in Apriori)
    - Typical 100x speedup on clinical datasets

    Args:
        patient_condition_matrix: Binary DataFrame (patients × conditions).
            Each cell is 1 if the patient has that condition, 0 otherwise.
            Example columns: diabetes, hypertension, obesity, CKD, stroke

        config: Mining hyperparameters (thresholds for support/confidence/lift).

    Returns:
        DataFrame of association rules with support, confidence, lift,
        sorted by lift descending.

    Example output:
        antecedents                consequents      support  confidence  lift
        {diabetes, hypertension}   {CKD}            0.082    0.723       4.21
        {obesity, sedentary}       {diabetes}        0.119    0.651       3.87
        {smoking, hypertension}    {stroke}          0.063    0.581       5.12
    """
    cfg = config or MiningConfig()

    # FP-Growth: mine frequent itemsets
    logger.info(
        "Mining frequent itemsets | min_support=%.3f | patients=%d | conditions=%d",
        cfg.min_support, len(patient_condition_matrix), len(patient_condition_matrix.columns),
    )

    frequent_itemsets = fpgrowth(
        patient_condition_matrix,
        min_support=cfg.min_support,
        use_colnames=True,
        max_len=cfg.max_antecedent_length + 1,  # +1 for the consequent
    )

    if frequent_itemsets.empty:
        logger.warning("No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    logger.info("Found %d frequent itemsets", len(frequent_itemsets))

    # Generate rules from frequent itemsets
    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=cfg.min_lift,
    )

    # Filter by confidence and rule complexity
    rules = rules[
        (rules["confidence"] >= cfg.min_confidence)
        & (rules["antecedents"].apply(len) <= cfg.max_antecedent_length)
        & (rules["consequents"].apply(len) == 1)  # Single consequent only
    ].copy()

    # Add clinical interpretability columns
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: " + ".join(sorted(x))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: " → ".join(sorted(x))
    )
    rules["rule_str"] = rules["antecedents_str"] + " ⟹ " + rules["consequents_str"]

    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    logger.info(
        "Found %d rules after filtering | top lift: %.2f",
        len(rules), rules["lift"].iloc[0] if len(rules) > 0 else 0,
    )
    return rules


def build_patient_condition_matrix(
    predictions_df: pd.DataFrame,
    threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Build a binary patient-condition matrix from prediction history.

    Args:
        predictions_df: DataFrame with columns [patient_id, disease, risk_score].
        threshold: Minimum risk score to flag a condition as present.

    Returns:
        Pivoted binary matrix: rows=patients, cols=diseases (1=high risk).
    """
    flagged = predictions_df[predictions_df["risk_score"] >= threshold].copy()
    matrix = (
        flagged.pivot_table(
            index="patient_id",
            columns="disease",
            values="risk_score",
            aggfunc="max",
        )
        .notna()
        .astype(int)
    )
    return matrix

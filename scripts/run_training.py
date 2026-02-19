"""
scripts/run_training.py — CLI entry point for model training.

Usage:
    python scripts/run_training.py --disease all --trials 100
    python scripts/run_training.py --disease heart --trials 80
    python scripts/run_training.py --disease diabetes --trials 60
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_breast_cancer, load_diabetes, load_heart_disease, load_kidney_disease
from src.models.train import train_disease_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


LOADERS = {
    "heart":    load_heart_disease,
    "diabetes": load_diabetes,
    "cancer":   load_breast_cancer,
    "kidney":   load_kidney_disease,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train disease prediction models.")
    parser.add_argument(
        "--disease",
        type=str,
        default="all",
        choices=["all", *LOADERS.keys()],
        help="Which disease model(s) to train.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=80,
        help="Number of Optuna hyperparameter trials.",
    )
    args = parser.parse_args()

    diseases = list(LOADERS.keys()) if args.disease == "all" else [args.disease]

    logger.info("Starting training pipeline | diseases=%s | trials=%d", diseases, args.trials)

    results = {}
    for disease in diseases:
        logger.info("=" * 60)
        logger.info("Training: %s", disease.upper())
        logger.info("=" * 60)

        df = LOADERS[disease]()
        result = train_disease_model(disease, df, n_trials=args.trials)

        results[disease] = result
        logger.info(
            "✓ %s | AUC-PR: %.4f | AUC-ROC: %.4f | Run: %s",
            disease,
            result.metrics.get("auc_pr", 0),
            result.metrics.get("auc_roc", 0),
            result.run_id,
        )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE — SUMMARY")
    logger.info("=" * 60)
    for disease, r in results.items():
        logger.info(
            "  %-10s  AUC-PR=%.4f  AUC-ROC=%.4f  Sensitivity=%.4f",
            disease,
            r.metrics.get("auc_pr", 0),
            r.metrics.get("auc_roc", 0),
            r.metrics.get("sensitivity", 0),
        )


if __name__ == "__main__":
    main()

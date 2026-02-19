"""
src/mining/clustering.py — Gaussian Mixture Model patient clustering
for population segmentation and phenotyping.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    labels: np.ndarray           # Cluster assignment per patient
    probabilities: np.ndarray    # Posterior probability per cluster
    n_clusters: int
    bic_score: float
    cluster_profiles: pd.DataFrame  # Per-cluster feature means


class PatientClusterer:
    """
    Gaussian Mixture Model clustering for unsupervised patient phenotyping.

    GMM advantages over KMeans for clinical data:
    - Soft cluster assignments (posterior probabilities, not hard labels)
    - Models elliptical cluster shapes (KMeans forces spherical)
    - Bayesian Information Criterion (BIC) for automatic K selection

    Usage:
        clusterer = PatientClusterer(max_clusters=8)
        result = clusterer.fit_predict(X_df)
    """

    def __init__(self, max_clusters: int = 8, random_state: int = 42) -> None:
        self.max_clusters = max_clusters
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model: GaussianMixture | None = None
        self.best_k: int | None = None

    def fit_predict(self, X: pd.DataFrame) -> ClusteringResult:
        """
        Fit GMM with BIC-optimal K and return cluster assignments.

        Args:
            X: Numeric feature DataFrame (after preprocessing).

        Returns:
            ClusteringResult with labels, posterior probabilities, BIC, profiles.
        """
        X_scaled = self._scaler.fit_transform(X.select_dtypes(include=np.number))

        # BIC-based automatic K selection
        best_k, best_bic = 2, np.inf
        for k in range(2, self.max_clusters + 1):
            gmm = GaussianMixture(n_components=k, random_state=self.random_state, n_init=3)
            gmm.fit(X_scaled)
            bic = gmm.bic(X_scaled)
            logger.debug("K=%d  BIC=%.2f", k, bic)
            if bic < best_bic:
                best_bic = bic
                best_k = k

        logger.info("Optimal clusters: K=%d (BIC=%.2f)", best_k, best_bic)
        self.best_k = best_k

        self._model = GaussianMixture(
            n_components=best_k, random_state=self.random_state, n_init=5
        )
        self._model.fit(X_scaled)

        labels = self._model.predict(X_scaled)
        probabilities = self._model.predict_proba(X_scaled)

        # Cluster profiles: mean feature values per cluster
        X_profiled = X.copy()
        X_profiled["cluster"] = labels
        profiles = X_profiled.groupby("cluster").mean().round(3)

        return ClusteringResult(
            labels=labels,
            probabilities=probabilities,
            n_clusters=best_k,
            bic_score=best_bic,
            cluster_profiles=profiles,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Assign new patients to existing clusters."""
        if self._model is None:
            raise RuntimeError("Call fit_predict() before predict().")
        X_scaled = self._scaler.transform(X.select_dtypes(include=np.number))
        return self._model.predict(X_scaled)

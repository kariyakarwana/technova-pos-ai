"""Evaluation ranking metrics for TechNova AI Recommendation System."""

from __future__ import annotations

import numpy as np

from ..domain.models import RecommendationMetrics


def _calculate_dcg(recs: list[str], ground_truth: set[str], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for idx, item in enumerate(recs[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(idx + 2)
    return dcg


def _calculate_idcg(ground_truth_size: int, k: int) -> float:
    """Computes Ideal Discounted Cumulative Gain at rank K."""
    idcg = 0.0
    m = min(ground_truth_size, k)
    for idx in range(m):
        idcg += 1.0 / np.log2(idx + 2)
    return idcg


__all__ = [
    "RecommendationMetrics",
    "_calculate_dcg",
    "_calculate_idcg",
]

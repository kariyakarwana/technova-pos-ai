"""Evaluation and validation layer for TechNova AI Recommendation System."""

from __future__ import annotations

from ..data.transactions import ChronologicalSplitter
from .evaluator import (
    OfflineEvaluator,
    RecommendationValidationReport,
    validate_recommendation_dataset,
)
from .metrics import (
    RecommendationMetrics,
    _calculate_dcg,
    _calculate_idcg,
)

__all__ = [
    "ChronologicalSplitter",
    "OfflineEvaluator",
    "RecommendationValidationReport",
    "validate_recommendation_dataset",
    "RecommendationMetrics",
    "_calculate_dcg",
    "_calculate_idcg",
]

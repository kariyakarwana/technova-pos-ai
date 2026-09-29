"""Domain layer for TechNova AI Recommendation System."""

from __future__ import annotations

from .contracts import CandidateGenerator, RecommendationPredictor, Recommender
from .enums import RecommendationContext, RecommendationReasonCode
from .models import (
    AssociationRule,
    CatalogProduct,
    RecommendationMetrics,
    RecommendationValidationReport,
    ScoredRecommendation,
)

__all__ = [
    "RecommendationContext",
    "RecommendationReasonCode",
    "ScoredRecommendation",
    "RecommendationMetrics",
    "CatalogProduct",
    "AssociationRule",
    "RecommendationValidationReport",
    "Recommender",
    "CandidateGenerator",
    "RecommendationPredictor",
]

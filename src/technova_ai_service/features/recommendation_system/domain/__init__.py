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
    "AssociationRule",
    "CandidateGenerator",
    "CatalogProduct",
    "RecommendationContext",
    "RecommendationMetrics",
    "RecommendationPredictor",
    "RecommendationReasonCode",
    "RecommendationValidationReport",
    "Recommender",
    "ScoredRecommendation",
]

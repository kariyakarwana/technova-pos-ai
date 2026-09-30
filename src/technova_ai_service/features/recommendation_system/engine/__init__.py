"""Recommendation Engine layer for TechNova AI Recommendation System."""

from __future__ import annotations

from .association import AssociationRule, FPGrowthModel
from .candidate_generation import CandidateGenerationEngine
from .compatibility import CompatibilityEngine
from .personalization import PersonalizedRecommender
from .popularity import BranchPopularityEngine
from .ranking import (
    DEFAULT_CONTEXT_WEIGHTS,
    EXPLANATION_TEMPLATES,
    HybridRecommendationEngine,
    ScoredRecommendation,
)
from .similarity import ContentSimilarityModel, ItemSimilarityModel
from .trending import TrendingEngine

__all__ = [
    "DEFAULT_CONTEXT_WEIGHTS",
    "EXPLANATION_TEMPLATES",
    "AssociationRule",
    "BranchPopularityEngine",
    "CandidateGenerationEngine",
    "CompatibilityEngine",
    "ContentSimilarityModel",
    "FPGrowthModel",
    "HybridRecommendationEngine",
    "ItemSimilarityModel",
    "PersonalizedRecommender",
    "ScoredRecommendation",
    "TrendingEngine",
]

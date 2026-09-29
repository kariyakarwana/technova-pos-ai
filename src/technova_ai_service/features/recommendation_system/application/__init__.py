"""Application layer for TechNova AI Recommendation System."""

from __future__ import annotations

from .service import (
    RecommendationService,
    create_recommendations,
    get_recommendation_service,
    warmup_recommendation_model,
)

__all__ = [
    "RecommendationService",
    "create_recommendations",
    "get_recommendation_service",
    "warmup_recommendation_model",
]

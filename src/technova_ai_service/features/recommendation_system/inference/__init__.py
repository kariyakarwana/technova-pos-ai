"""Inference layer for TechNova AI Recommendation System."""

from __future__ import annotations

from .artifact_registry import ArtifactRegistry, get_default_paths
from .model_loader import (
    RecommendationBundle,
    clear_recommendation_bundle_cache,
    load_recommendation_bundle,
)
from .predictor import RecommendationPredictor

__all__ = [
    "ArtifactRegistry",
    "RecommendationBundle",
    "RecommendationPredictor",
    "clear_recommendation_bundle_cache",
    "get_default_paths",
    "load_recommendation_bundle",
]

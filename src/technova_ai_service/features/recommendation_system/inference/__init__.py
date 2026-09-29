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
    "get_default_paths",
    "RecommendationBundle",
    "load_recommendation_bundle",
    "clear_recommendation_bundle_cache",
    "RecommendationPredictor",
]

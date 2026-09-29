"""Domain enumerations for the TechNova AI Recommendation System."""

from __future__ import annotations

from enum import Enum


class RecommendationContext(str, Enum):
    """Operational retail POS context triggering a recommendation request."""

    CUSTOMER = "CUSTOMER"
    PRODUCT = "PRODUCT"
    BRANCH = "BRANCH"
    TRENDING = "TRENDING"
    COLD_START = "COLD_START"
    CART_READY = "CART_READY"
    POPULAR = "POPULAR"


class RecommendationReasonCode(str, Enum):
    """Machine-readable, deterministic explanation code for recommendations."""

    FREQUENTLY_BOUGHT_TOGETHER = "FREQUENTLY_BOUGHT_TOGETHER"
    BRANCH_POPULAR = "BRANCH_POPULAR"
    CUSTOMER_HISTORY_AFFINITY = "CUSTOMER_HISTORY_AFFINITY"
    SIMILAR_PRODUCT = "SIMILAR_PRODUCT"
    TRENDING_ACCELERATION = "TRENDING_ACCELERATION"
    COMPATIBLE_ACCESSORY = "COMPATIBLE_ACCESSORY"
    POPULAR_FALLBACK = "POPULAR_FALLBACK"

"""Domain contracts and interfaces for the TechNova AI Recommendation System."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import ScoredRecommendation


@runtime_checkable
class Recommender(Protocol):
    """Protocol for recommendation sub-engines and models."""

    def recommend(self, *args: Any, **kwargs: Any) -> list[tuple[str, float]] | list[ScoredRecommendation]:
        """Generates recommendations."""
        ...


@runtime_checkable
class CandidateGenerator(Protocol):
    """Protocol for generating recommendation candidate item sets."""

    def generate_candidates(
        self,
        context: str,
        query_products: list[str],
        customer_id: str | None = None,
        branch_id: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Extracts candidate item IDs and associated signals."""
        ...


@runtime_checkable
class RecommendationPredictor(Protocol):
    """Protocol for inference runtime recommendation predictor."""

    def predict(
        self,
        context: str,
        customer_id: str | None = None,
        product_id: str | None = None,
        cart_product_ids: list[str] | None = None,
        branch_id: str | None = None,
        top_n: int = 5,
        enforce_stock: bool = True,
        organization_id: str | None = None,
    ) -> list[ScoredRecommendation]:
        """Executes candidate scoring and returns ranked recommendations."""
        ...

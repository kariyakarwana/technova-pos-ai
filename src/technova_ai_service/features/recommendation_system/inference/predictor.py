"""Inference predictor for TechNova Recommendation System."""

from __future__ import annotations

from ..domain.models import ScoredRecommendation
from .artifact_registry import get_default_paths
from .model_loader import (
    RecommendationBundle,
    clear_recommendation_bundle_cache,
    load_recommendation_bundle,
)


class RecommendationPredictor:
    """Predictor wrapping the loaded recommendation bundle for runtime inference."""

    def __init__(self, bundle: RecommendationBundle | None = None) -> None:
        self._bundle = bundle

    @property
    def bundle(self) -> RecommendationBundle:
        if self._bundle is None:
            self._bundle = load_recommendation_bundle()
        return self._bundle

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
        """Executes candidate retrieval, hybrid multi-signal scoring, and stock-aware ranking."""
        return self.bundle.hybrid_engine.recommend(
            context=context,
            customer_id=customer_id,
            product_id=product_id,
            cart_product_ids=cart_product_ids,
            branch_id=branch_id,
            top_n=top_n,
            enforce_stock=enforce_stock,
            organization_id=organization_id,
        )


__all__ = [
    "RecommendationBundle",
    "RecommendationPredictor",
    "clear_recommendation_bundle_cache",
    "get_default_paths",
    "load_recommendation_bundle",
]

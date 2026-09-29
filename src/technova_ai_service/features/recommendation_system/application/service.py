"""Recommendation Service orchestration layer for TechNova AI Recommendation Engine."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

from ..api.schemas import (
    RecommendationModelMetadata,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedProduct,
)
from ..inference.model_loader import (
    RecommendationBundle,
    load_recommendation_bundle,
)

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service handling validation, model invocation, stock-aware filtering, and response mapping."""

    def __init__(self, bundle: RecommendationBundle | None = None) -> None:
        self._bundle = bundle

    @property
    def bundle(self) -> RecommendationBundle:
        if self._bundle is None:
            self._bundle = load_recommendation_bundle()
        return self._bundle

    def generate_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """Executes full recommendation pipeline for the given request."""
        bundle = self.bundle
        engine = bundle.hybrid_engine
        df_catalog = bundle.df_catalog
        df_inventory = bundle.df_inventory

        # 1. Tenant Isolation Verification
        catalog_org_id = (
            str(df_catalog["organization_id"].iloc[0])
            if not df_catalog.empty
            else "org_technova_default"
        )
        if request.organization_id != catalog_org_id:
            logger.warning(
                "Tenant isolation mismatch: request organization_id '%s' != catalog '%s'",
                request.organization_id,
                catalog_org_id,
            )
            raise ValueError(
                f"Invalid organization_id '{request.organization_id}'. Cross-tenant access is prohibited."
            )

        # 2. Branch ID validation if provided
        valid_branches = set(df_inventory["branch_id"])
        if request.branch_id and request.branch_id not in valid_branches:
            raise ValueError(f"Branch '{request.branch_id}' not found in organization inventory.")

        # Build runtime product context for content-based matching when target product is unseen in model catalog
        runtime_context: dict[str, Any] = {}
        if request.category:
            runtime_context["category_level_1"] = request.category
        if request.product_name:
            runtime_context["name"] = request.product_name
        if request.brand:
            runtime_context["brand"] = request.brand
        if request.price is not None:
            runtime_context["current_unit_price"] = request.price

        # 3. Invoke Hybrid Recommendation Engine
        scored_recs = engine.recommend(
            context=request.context.value,
            customer_id=request.customer_id,
            product_id=request.product_id,
            cart_product_ids=request.product_ids,
            branch_id=request.branch_id,
            top_n=request.top_n,
            enforce_stock=True,
            organization_id=request.organization_id,
            runtime_context=runtime_context if runtime_context else None,
        )

        # 5. Enrich recommendations with Catalog details
        catalog_lookup = df_catalog.set_index("product_id").to_dict(orient="index")

        recommended_products: list[RecommendedProduct] = []
        for r in scored_recs:
            prod_info = catalog_lookup.get(r.product_id, {})
            recommended_products.append(
                RecommendedProduct(
                    product_id=r.product_id,
                    name=prod_info.get("name"),
                    category=prod_info.get("category_level_1"),
                    brand=prod_info.get("brand"),
                    price=float(prod_info.get("current_unit_price", 0.0)),
                    score=r.score,
                    reason_code=RecommendationReasonCode(r.reason_code),
                    reason=r.reason,
                    stock_quantity=r.stock_quantity,
                    is_available=r.is_available,
                )
            )

        # 6. Build Provenance & Metadata
        training_meta = bundle.metadata.get("training_metadata", {})
        date_range = training_meta.get("date_range", {})

        algorithms_used = [
            "FP-Growth Association Mining",
            "Collaborative Item-to-Item Similarity",
            "Content-Based TF-IDF Vectorization",
            "Tag-Driven Technical Compatibility",
            "Personalized Customer History Affinity",
            "Dual-Window Trending Velocity",
            "Bayesian Smoothed Branch Popularity",
        ]

        model_meta = RecommendationModelMetadata(
            model_name="TechNova Hybrid Product Recommendation Engine",
            version="1.0.0",
            algorithms_used=algorithms_used,
            context=request.context.value,
            total_candidates_scored=len(scored_recs),
            training_date_range={
                "train_start": str(date_range.get("train_start", "2014-06-02")),
                "train_end": str(date_range.get("train_end", "2014-11-15")),
            },
        )

        return RecommendationResponse(
            context=request.context,
            organization_id=request.organization_id,
            branch_id=request.branch_id,
            recommendations=recommended_products,
            generated_at=datetime.now(timezone.utc).isoformat(),
            model_metadata=model_meta,
        )


_DEFAULT_SERVICE: RecommendationService | None = None


def get_recommendation_service() -> RecommendationService:
    """Returns singleton recommendation service instance."""
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = RecommendationService()
    return _DEFAULT_SERVICE


def create_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """Convenience helper to generate recommendations."""
    service = get_recommendation_service()
    return service.generate_recommendations(request)


def warmup_recommendation_model() -> None:
    """Preloads recommendation model artifacts into memory during application startup."""
    load_recommendation_bundle()
    logger.info("Recommendation model bundle preloaded and verified.")
def create_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """Convenience helper to generate recommendations."""
    service = get_recommendation_service()
    return service.generate_recommendations(request)


def warmup_recommendation_model() -> None:
    """Preloads recommendation model artifacts into memory during application startup."""
    load_recommendation_bundle()
    logger.info("Recommendation model bundle preloaded and verified.")

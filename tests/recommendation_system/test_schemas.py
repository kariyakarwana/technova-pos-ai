"""Unit tests for recommendation system Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from technova_ai_service.features.recommendation_system.api.schemas import (
    RecommendationContext,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendedProduct,
)


def test_valid_product_context_request() -> None:
    req = RecommendationRequest(
        organization_id="org_test",
        context=RecommendationContext.PRODUCT,
        product_id="PROD-001",
        top_n=5,
    )
    assert req.context == RecommendationContext.PRODUCT
    assert req.product_id == "PROD-001"
    assert req.top_n == 5


def test_product_context_missing_product_id() -> None:
    with pytest.raises(ValidationError) as exc:
        RecommendationRequest(
            organization_id="org_test",
            context=RecommendationContext.PRODUCT,
            product_id=None,
        )
    assert "product_id is required" in str(exc.value)


def test_cart_ready_context_request() -> None:
    req = RecommendationRequest(
        organization_id="org_test",
        context=RecommendationContext.CART_READY,
        product_ids=["PROD-001", "PROD-002"],
    )
    assert req.context == RecommendationContext.CART_READY
    assert len(req.product_ids) == 2


def test_cart_ready_missing_product_ids() -> None:
    with pytest.raises(ValidationError) as exc:
        RecommendationRequest(
            organization_id="org_test",
            context=RecommendationContext.CART_READY,
            product_ids=[],
        )
    assert "product_ids list with at least 1 item is required" in str(exc.value)


def test_branch_context_missing_branch_id() -> None:
    with pytest.raises(ValidationError) as exc:
        RecommendationRequest(
            organization_id="org_test",
            context=RecommendationContext.BRANCH,
            branch_id=None,
        )
    assert "branch_id is required" in str(exc.value)


def test_top_n_bounds() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(
            organization_id="org_test",
            context=RecommendationContext.COLD_START,
            top_n=0,  # below min 1
        )
    with pytest.raises(ValidationError):
        RecommendationRequest(
            organization_id="org_test",
            context=RecommendationContext.COLD_START,
            top_n=25,  # above max 20
        )


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError) as exc:
        RecommendationRequest(
            organization_id="org_test",
            context=RecommendationContext.COLD_START,
            unexpected_field="disallowed",  # type: ignore
        )
    assert "extra_forbidden" in str(exc.value) or "Extra inputs are not permitted" in str(exc.value)


def test_recommended_product_schema() -> None:
    prod = RecommendedProduct(
        product_id="PROD-001",
        name="Test Item",
        category="Tech",
        brand="BrandX",
        price=99.99,
        score=0.88,
        reason_code=RecommendationReasonCode.FREQUENTLY_BOUGHT_TOGETHER,
        reason="Frequently purchased together.",
        stock_quantity=15,
        is_available=True,
    )
    assert prod.score == 0.88
    assert prod.reason_code == RecommendationReasonCode.FREQUENTLY_BOUGHT_TOGETHER
    assert prod.stock_quantity == 15

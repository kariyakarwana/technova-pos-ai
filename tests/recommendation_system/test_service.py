"""Unit tests for recommendation service logic and tenant/stock validation."""

from __future__ import annotations

import pytest

from technova_ai_service.features.recommendation_system.api.schemas import (
    RecommendationContext,
    RecommendationRequest,
    RecommendationResponse,
)
from technova_ai_service.features.recommendation_system.application.service import (
    RecommendationService,
    get_recommendation_service,
)


@pytest.fixture
def rec_service() -> RecommendationService:
    return get_recommendation_service()


def test_service_customer_context(rec_service: RecommendationService) -> None:
    req = RecommendationRequest(
        organization_id="org_technova_default",
        context=RecommendationContext.CUSTOMER,
        customer_id="CUST-0001",
        branch_id="BRANCH-001",
        top_n=5,
    )
    resp = rec_service.generate_recommendations(req)

    assert isinstance(resp, RecommendationResponse)
    assert resp.context == RecommendationContext.CUSTOMER
    assert len(resp.recommendations) <= 5
    assert len(resp.recommendations) > 0

    # Ensure no duplicates
    rec_ids = [r.product_id for r in resp.recommendations]
    assert len(rec_ids) == len(set(rec_ids))

    # Check score descending order
    scores = [r.score for r in resp.recommendations]
    assert scores == sorted(scores, reverse=True)


def test_service_product_context(rec_service: RecommendationService) -> None:
    req = RecommendationRequest(
        organization_id="org_technova_default",
        context=RecommendationContext.PRODUCT,
        product_id="PROD-COMP-001",
        branch_id="BRANCH-001",
        top_n=5,
    )
    resp = rec_service.generate_recommendations(req)

    assert resp.context == RecommendationContext.PRODUCT
    assert len(resp.recommendations) > 0
    # Seed product PROD-COMP-001 should not be recommended to itself
    assert "PROD-COMP-001" not in [r.product_id for r in resp.recommendations]


def test_service_branch_context(rec_service: RecommendationService) -> None:
    req = RecommendationRequest(
        organization_id="org_technova_default",
        context=RecommendationContext.BRANCH,
        branch_id="BRANCH-001",
        top_n=5,
    )
    resp = rec_service.generate_recommendations(req)
    assert resp.context == RecommendationContext.BRANCH
    assert len(resp.recommendations) > 0


def test_service_trending_context(rec_service: RecommendationService) -> None:
    req = RecommendationRequest(
        organization_id="org_technova_default",
        context=RecommendationContext.TRENDING,
        branch_id="BRANCH-001",
        top_n=5,
    )
    resp = rec_service.generate_recommendations(req)
    assert resp.context == RecommendationContext.TRENDING
    assert len(resp.recommendations) > 0


def test_service_cold_start_context(rec_service: RecommendationService) -> None:
    req = RecommendationRequest(
        organization_id="org_technova_default",
        context=RecommendationContext.COLD_START,
        branch_id="BRANCH-001",
        top_n=5,
    )
    resp = rec_service.generate_recommendations(req)
    assert resp.context == RecommendationContext.COLD_START
    assert len(resp.recommendations) > 0


def test_service_cart_ready_context(rec_service: RecommendationService) -> None:
    req = RecommendationRequest(
        organization_id="org_technova_default",
        context=RecommendationContext.CART_READY,
        product_ids=["PROD-COMP-001", "PROD-PERI-001"],
        branch_id="BRANCH-001",
        top_n=5,
    )
    resp = rec_service.generate_recommendations(req)
    assert resp.context == RecommendationContext.CART_READY
    assert len(resp.recommendations) > 0


def test_service_stock_filtering_removes_inactive(rec_service: RecommendationService) -> None:
    req = RecommendationRequest(
        organization_id="org_technova_default",
        context=RecommendationContext.BRANCH,
        branch_id="BRANCH-001",
        top_n=20,
    )
    resp = rec_service.generate_recommendations(req)

    # Inactive SKUs must never be returned
    rec_ids = [r.product_id for r in resp.recommendations]
    assert "PROD-DISC-001" not in rec_ids
    assert "PROD-DISC-002" not in rec_ids

    # All returned recommendations must have stock > 0 and is_available == True
    for r in resp.recommendations:
        assert r.is_available is True
        assert r.stock_quantity > 0


def test_service_artifact_is_tenant_neutral(rec_service: RecommendationService) -> None:
    req = RecommendationRequest(
        organization_id="org_unauthorized_other",
        context=RecommendationContext.COLD_START,
    )
    response = rec_service.generate_recommendations(req)
    assert response.organization_id == "org_unauthorized_other"
    assert isinstance(response.recommendations, list)


def test_service_unseen_product_id_honest_unavailable_or_content_based(
    rec_service: RecommendationService,
) -> None:
    # 1. Unseen product without runtime content returns honest unavailable result (empty recommendations list, no error)
    req_no_content = RecommendationRequest(
        organization_id="org_technova_default",
        context=RecommendationContext.PRODUCT,
        product_id="PROD-POW-071",
    )
    res_no_content = rec_service.generate_recommendations(req_no_content)
    assert res_no_content.recommendations == []
    assert res_no_content.context == RecommendationContext.PRODUCT

    # 2. Unseen product with runtime context uses content similarity capability
    req_with_content = RecommendationRequest(
        organization_id="org_technova_default",
        context=RecommendationContext.PRODUCT,
        product_id="PROD-POW-071",
        category="Computers & Electronics",
        brand="TechNova",
        price=1200.0,
    )
    res_with_content = rec_service.generate_recommendations(req_with_content)
    assert isinstance(res_with_content.recommendations, list)
    # If content matches exist, they should have SIMILAR_PRODUCT reason code
    for rec in res_with_content.recommendations:
        assert rec.reason_code.value in ("SIMILAR_PRODUCT", "COMPATIBLE_ACCESSORY")


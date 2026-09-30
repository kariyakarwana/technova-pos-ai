"""HTTP Integration tests for recommendation system FastAPI router."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from technova_ai_service.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_http_recommend_product_context_success(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "PRODUCT",
        "product_id": "PROD-COMP-001",
        "branch_id": "BRANCH-001",
        "top_n": 5,
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["context"] == "PRODUCT"
    assert data["organization_id"] == "org_technova_default"
    assert len(data["recommendations"]) <= 5
    assert len(data["recommendations"]) > 0

    first = data["recommendations"][0]
    assert "product_id" in first
    assert "score" in first
    assert "reason_code" in first
    assert "reason" in first
    assert "stock_quantity" in first
    assert "is_available" in first
    assert first["is_available"] is True


def test_http_recommend_customer_context_success(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "CUSTOMER",
        "customer_id": "CUST-0001",
        "branch_id": "BRANCH-001",
        "top_n": 5,
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["context"] == "CUSTOMER"
    assert len(data["recommendations"]) > 0


def test_http_recommend_trending_context_success(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "TRENDING",
        "branch_id": "BRANCH-001",
        "top_n": 5,
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["context"] == "TRENDING"


def test_http_recommend_cold_start_context_success(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "COLD_START",
        "branch_id": "BRANCH-001",
        "top_n": 5,
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["context"] == "COLD_START"


def test_http_recommend_cart_ready_context_success(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "CART_READY",
        "product_ids": ["PROD-COMP-001", "PROD-PERI-001"],
        "branch_id": "BRANCH-001",
        "top_n": 5,
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["context"] == "CART_READY"


def test_http_recommend_invalid_context_returns_422(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "INVALID_CONTEXT_XYZ",
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 422


def test_http_recommend_missing_product_id_returns_422(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "PRODUCT",
        # missing product_id
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 422


def test_http_recommend_unexpected_field_returns_422(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "COLD_START",
        "hacker_field": "injected",
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 422


def test_http_recommend_unseen_product_does_not_return_400(client: TestClient) -> None:
    # A real product such as PROD-POW-071 or PROD-SMA-085 from the tenant DB must not return 400
    payload = {
        "organization_id": "org_technova_default",
        "context": "PRODUCT",
        "product_id": "PROD-POW-071",
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["context"] == "PRODUCT"
    assert data["recommendations"] == []

    # With runtime product context, content-based recommendations are returned
    payload_with_content = {
        "organization_id": "org_technova_default",
        "context": "PRODUCT",
        "product_id": "PROD-POW-071",
        "category": "Computers & Electronics",
        "product_name": "Power Bank Pro",
        "brand": "TechNova",
        "price": 120.0,
    }
    res_content = client.post("/v1/recommendations/recommend", json=payload_with_content)
    assert res_content.status_code == 200
    data_content = res_content.json()
    assert isinstance(data_content["recommendations"], list)


def test_http_recommend_accepts_authenticated_tenant_context(client: TestClient) -> None:
    payload = {
        "organization_id": "org_alien_intruder",
        "context": "COLD_START",
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 200
    assert response.json()["organization_id"] == "org_alien_intruder"


def test_http_recommend_artifacts_missing_returns_503(client: TestClient) -> None:
    with patch(
        "technova_ai_service.features.recommendation_system.api.router.create_recommendations",
        side_effect=FileNotFoundError("Mocked missing artifact"),
    ):
        payload = {
            "organization_id": "org_technova_default",
            "context": "COLD_START",
        }
        response = client.post("/v1/recommendations/recommend", json=payload)
        assert response.status_code == 503
        assert "Recommendation service unavailable" in response.json()["detail"]


def test_http_recommend_customer_no_history_returns_400(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "CUSTOMER",
        "customer_id": "CUST-NO-HISTORY",
        "branch_id": "BRANCH-001",
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 400
    assert "Insufficient historical data" in response.json()["detail"]


def test_http_recommend_trending_live_branch_uses_global_model_fallback(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "TRENDING",
        "branch_id": "BRANCH-999-NO-SALES",
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 200
    assert response.json()["branch_id"] == "BRANCH-999-NO-SALES"


def test_http_recommend_popular_alias_success(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova_default",
        "context": "POPULAR",
        "branch_id": "BRANCH-001",
        "top_n": 5,
    }
    response = client.post("/v1/recommendations/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) > 0

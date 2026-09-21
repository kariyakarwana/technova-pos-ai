from fastapi.testclient import TestClient

from technova_ai_service.main import app


def test_health_check() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ai_routes_are_registered() -> None:
    paths = set(app.openapi()["paths"])
    assert "/v1/inventory/forecast" in paths
    assert "/v1/inventory/forecast/multi-horizon" in paths
    assert "/v1/pricing/recommend" in paths
    assert "/v1/loyalty/recommendations" in paths

"""Unit and integration tests for TechNova Sales Forecasting FastAPI feature."""

from datetime import date
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from technova_ai_service.features.sales_forecasting.schemas import (
    DailyRevenuePoint,
    FutureCalendarPoint,
    SalesForecastRequest,
    SalesForecastResponse,
)
from technova_ai_service.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_sales_forecast_route_exists() -> None:
    paths = set(app.openapi()["paths"])
    assert "/v1/sales-forecast/forecast" in paths


@pytest.mark.parametrize("horizon", [1, 7, 14, 30])
def test_valid_horizons_accepted(horizon: int) -> None:
    request = SalesForecastRequest(
        organization_id="org_test_123",
        branch_id="br_test_456",
        forecast_horizon=horizon,
        recent_daily_revenue=[
            DailyRevenuePoint(date=date(2026, 9, 1), revenue=15000.0, is_operating_day=1),
            DailyRevenuePoint(date=date(2026, 9, 2), revenue=12000.0, is_operating_day=1),
        ],
        future_calendar=[
            FutureCalendarPoint(date=date(2026, 9, 3), is_operating_day=1, active_discounts=2),
            FutureCalendarPoint(date=date(2026, 9, 4), is_operating_day=1, active_discounts=0),
        ],
    )
    assert request.forecast_horizon == horizon
    assert request.organization_id == "org_test_123"
    assert request.branch_id == "br_test_456"
    assert len(request.recent_daily_revenue) == 2
    assert len(request.future_calendar) == 2


@pytest.mark.parametrize("invalid_horizon", [-1, 0, 2, 5, 10, 28, 45, 90])
def test_invalid_horizon_rejected(invalid_horizon: int) -> None:
    with pytest.raises(ValidationError):
        SalesForecastRequest(
            organization_id="org_test_123",
            branch_id="br_test_456",
            forecast_horizon=invalid_horizon,  # type: ignore[arg-type]
        )


def test_invalid_negative_revenue_rejected() -> None:
    with pytest.raises(ValidationError):
        DailyRevenuePoint(date=date(2026, 9, 1), revenue=-50.0, is_operating_day=1)


def test_empty_tenant_identifiers_rejected() -> None:
    with pytest.raises(ValidationError):
        SalesForecastRequest(
            organization_id="",
            branch_id="br_1",
            forecast_horizon=7,
        )

    with pytest.raises(ValidationError):
        SalesForecastRequest(
            organization_id="   ",
            branch_id="br_1",
            forecast_horizon=7,
        )

    with pytest.raises(ValidationError):
        SalesForecastRequest(
            organization_id="org_1",
            branch_id="",
            forecast_horizon=7,
        )


def test_non_chronological_future_dates_rejected() -> None:
    with pytest.raises(ValidationError):
        SalesForecastRequest(
            organization_id="org_1",
            branch_id="br_1",
            forecast_horizon=7,
            future_calendar=[
                FutureCalendarPoint(date=date(2026, 9, 5), is_operating_day=1, active_discounts=0),
                FutureCalendarPoint(date=date(2026, 9, 4), is_operating_day=1, active_discounts=0),
            ],
        )


def test_api_endpoint_rejects_invalid_payload(client: TestClient) -> None:
    payload = {
        "organization_id": "org_1",
        "branch_id": "br_1",
        "forecast_horizon": 5,
    }
    response = client.post("/v1/sales-forecast/forecast", json=payload)
    assert response.status_code == 422


def test_api_endpoint_rejects_negative_revenue(client: TestClient) -> None:
    payload = {
        "organization_id": "org_1",
        "branch_id": "br_1",
        "forecast_horizon": 7,
        "recent_daily_revenue": [
            {"date": "2026-09-01", "revenue": -100.0, "is_operating_day": 1}
        ],
    }
    response = client.post("/v1/sales-forecast/forecast", json=payload)
    assert response.status_code == 422


def test_missing_model_returns_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from technova_ai_service.config import get_settings

    monkeypatch.setattr(get_settings(), "artifact_dir", tmp_path / "nonexistent")

    payload = {
        "organization_id": "org_technova",
        "branch_id": "br_colombo_main",
        "forecast_horizon": 7,
        "recent_daily_revenue": [
            {"date": "2026-09-01", "revenue": 1000.0, "is_operating_day": 1}
        ],
        "future_calendar": [
            {"date": "2026-09-02", "is_operating_day": 1, "active_discounts": 0}
        ],
    }
    response = client.post("/v1/sales-forecast/forecast", json=payload)
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "Model artifact not found" in data["detail"]


def test_insufficient_historical_data_returns_400(client: TestClient) -> None:
    payload = {
        "organization_id": "org_technova",
        "branch_id": "br_colombo_main",
        "forecast_horizon": 7,
        "recent_daily_revenue": [],
        "future_calendar": [
            {"date": "2026-09-02", "is_operating_day": 1, "active_discounts": 0}
        ],
    }
    response = client.post("/v1/sales-forecast/forecast", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Insufficient historical data"



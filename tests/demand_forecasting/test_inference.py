from datetime import date, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from technova_ai_service.features.demand_forecasting.inference import (
    load_demand_forecast_bundle,
    predict_from_feature_dataframe,
)
from technova_ai_service.features.demand_forecasting.schemas import (
    MODEL_FEATURE_COLUMNS,
    DailyForecastContext,
    DemandForecastRequest,
    DemandForecastResponse,
)
from technova_ai_service.features.demand_forecasting.service import (
    create_demand_forecast,
    warmup_demand_model,
)
from technova_ai_service.main import app

client = TestClient(app)


def test_model_artifact_loading() -> None:
    """Verify production artifact loads safely, is cached, and validates bundle structure."""
    bundle = load_demand_forecast_bundle()

    assert isinstance(bundle, dict)
    assert "model" in bundle
    assert "feature_columns" in bundle
    assert "model_type" in bundle
    assert "target" in bundle
    assert bundle["model_type"] == "lightgbm"
    assert bundle["target"] == "units_sold"
    assert hasattr(bundle["model"], "predict")

    # Verify exact 11 features
    assert bundle["feature_columns"] == MODEL_FEATURE_COLUMNS
    assert len(bundle["feature_columns"]) == 11

    # Verify caching: subsequent call returns identical object in memory
    bundle_cached = load_demand_forecast_bundle()
    assert bundle is bundle_cached

    # Verify missing artifact raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_demand_forecast_bundle("artifacts/demand_forecasting/non_existent.joblib")


def test_warmup_demand_model() -> None:
    """Verify service warmup function returns valid cached bundle."""
    bundle = warmup_demand_model()
    assert isinstance(bundle, dict)
    assert bundle["target"] == "units_sold"


def test_valid_forecast_request() -> None:
    """Verify POST /v1/demand-forecast/forecast returns 200 OK with all required fields."""
    payload = {
        "product_id": "SKU_001",
        "store_id": "1",
        "category": "Beverages",
        "base_unit_price": 1.20,
        "unit_price": 1.20,
        "store_type": "a",
        "assortment": "a",
        "promo2": 0,
        "horizon": 7,
    }

    res = client.post("/v1/demand-forecast/forecast", json=payload)
    assert res.status_code == 200

    data = res.json()
    # Verify minimum required fields from requirement 9
    assert data["product_id"] == "SKU_001"
    assert data["store_id"] == "1"
    assert "forecast_date" in data
    assert "predicted_units" in data
    assert data["horizon"] == 7

    # Verify predictions array
    assert "predictions" in data
    assert len(data["predictions"]) == 7
    assert data["predicted_units"] > 0.0
    assert data["total_predicted_units"] == pytest.approx(data["predicted_units"])

    # Verify model metadata
    assert data["model_metadata"]["model_type"] == "lightgbm"
    assert data["model_metadata"]["target"] == "units_sold"
    assert data["model_metadata"]["feature_count"] == 11
    assert data["model_metadata"]["feature_columns"] == MODEL_FEATURE_COLUMNS


def test_response_schema_completeness() -> None:
    """Verify that every daily forecast point contains the required fields."""
    request = DemandForecastRequest(
        product_id="SKU_002",
        store_id="10",
        category="Packaged Foods",
        base_unit_price=2.10,
        horizon=7,
    )
    response = create_demand_forecast(request)

    assert isinstance(response, DemandForecastResponse)
    assert response.product_id == "SKU_002"
    assert response.store_id == "10"
    assert response.horizon == 7
    assert len(response.predictions) == 7

    for point in response.predictions:
        assert point.product_id == "SKU_002"
        assert point.store_id == "10"
        assert point.horizon == 7
        assert isinstance(point.forecast_date, date)
        assert point.predicted_units >= 0.0
        assert point.day_of_week in range(7)
        assert point.is_open in (0, 1)
        assert point.is_promo in (0, 1)
        assert point.unit_price > 0.0


def test_missing_required_fields() -> None:
    """Verify 422 Unprocessable Entity when required fields are missing or empty."""
    valid_base = {
        "product_id": "SKU_001",
        "store_id": "1",
        "category": "Beverages",
        "base_unit_price": 1.20,
    }

    # 1. Missing product_id
    bad_req = dict(valid_base)
    del bad_req["product_id"]
    res = client.post("/v1/demand-forecast/forecast", json=bad_req)
    assert res.status_code == 422

    # 2. Empty product_id
    bad_req = dict(valid_base, product_id="   ")
    res = client.post("/v1/demand-forecast/forecast", json=bad_req)
    assert res.status_code == 422

    # 3. Missing store_id
    bad_req = dict(valid_base)
    del bad_req["store_id"]
    res = client.post("/v1/demand-forecast/forecast", json=bad_req)
    assert res.status_code == 422

    # 4. Missing category
    bad_req = dict(valid_base)
    del bad_req["category"]
    res = client.post("/v1/demand-forecast/forecast", json=bad_req)
    assert res.status_code == 422

    # 5. Missing base_unit_price
    bad_req = dict(valid_base)
    del bad_req["base_unit_price"]
    res = client.post("/v1/demand-forecast/forecast", json=bad_req)
    assert res.status_code == 422

    # 6. Negative base_unit_price
    bad_req = dict(valid_base, base_unit_price=-5.0)
    res = client.post("/v1/demand-forecast/forecast", json=bad_req)
    assert res.status_code == 422


def test_invalid_categorical_values() -> None:
    """Verify 422 Unprocessable Entity on invalid categorical inputs."""
    valid_base = {
        "product_id": "SKU_001",
        "store_id": "1",
        "category": "Beverages",
        "base_unit_price": 1.20,
    }

    # 1. Empty or whitespace category
    res1 = client.post(
        "/v1/demand-forecast/forecast",
        json=dict(valid_base, category="   "),
    )
    assert res1.status_code == 422
    assert "category cannot be empty" in res1.text

    # 2. Invalid store_type
    res2 = client.post(
        "/v1/demand-forecast/forecast",
        json=dict(valid_base, store_type="z"),
    )
    assert res2.status_code == 422
    assert "Invalid store_type" in res2.text

    # 3. Invalid assortment
    res3 = client.post(
        "/v1/demand-forecast/forecast",
        json=dict(valid_base, assortment="x"),
    )
    assert res3.status_code == 422
    assert "Invalid assortment" in res3.text

    # 4. Invalid state_holiday in daily_contexts
    res4 = client.post(
        "/v1/demand-forecast/forecast",
        json=dict(
            valid_base,
            daily_contexts=[{"state_holiday": "Easter_Invalid"}],
        ),
    )
    assert res4.status_code == 422
    assert "Invalid state_holiday" in res4.text


def test_arbitrary_tenant_defined_categories_accepted() -> None:
    """Verify that arbitrary tenant-defined categories are accepted without validation errors.

    Tests multi-tenant SaaS categories including:
    - Power Banks
    - Laptops
    - Gaming Gear
    - Computer Accessories
    - Storage Devices
    - Cross-tenant categories (Medicine, Furniture, Dairy)
    """
    tenant_categories = [
        ("SKU_PB_01", "Power Banks", 25.00),
        ("SKU_LAP_01", "Laptops", 1200.00),
        ("SKU_GG_01", "Gaming Gear", 150.00),
        ("SKU_CA_01", "Computer Accessories", 35.00),
        ("SKU_SD_01", "Storage Devices", 80.00),
        ("SKU_MED_01", "Medicine", 15.00),
        ("SKU_FURN_01", "Furniture", 450.00),
        ("SKU_DAIRY_01", "Dairy", 4.50),
    ]

    for sku, cat, price in tenant_categories:
        payload = {
            "product_id": sku,
            "store_id": "1",
            "category": cat,
            "base_unit_price": price,
            "horizon": 7,
        }
        res = client.post("/v1/demand-forecast/forecast", json=payload)
        assert res.status_code == 200, f"Failed for category {cat}: {res.text}"
        data = res.json()
        assert data["product_id"] == sku
        assert data["horizon"] == 7
        assert len(data["predictions"]) == 7
        assert all(p["predicted_units"] >= 0.0 for p in data["predictions"])


def test_cold_start_insufficient_historical_data() -> None:
    """Verify that when no historical data exists, the API rejects with 'Insufficient historical data'."""
    payload = {
        "product_id": "SKU_NEW_COLD",
        "store_id": "1",
        "category": "Power Banks",
        "base_unit_price": 45.00,
        "has_historical_data": False,
    }
    res = client.post("/v1/demand-forecast/forecast", json=payload)
    assert res.status_code == 400
    assert "Insufficient historical data" in res.json()["detail"]


def test_customers_feature_strictly_forbidden() -> None:
    """Verify that 'customers' is rejected and cannot be passed as a model feature."""
    valid_base = {
        "product_id": "SKU_001",
        "store_id": "1",
        "category": "Beverages",
        "base_unit_price": 1.20,
    }

    # Attempt to inject customers at request level
    res1 = client.post(
        "/v1/demand-forecast/forecast",
        json=dict(valid_base, customers=500),
    )
    assert res1.status_code == 422
    assert "customers" in res1.text.lower()

    # Attempt to inject customers inside daily_contexts
    res2 = client.post(
        "/v1/demand-forecast/forecast",
        json=dict(
            valid_base,
            daily_contexts=[{"is_open": 1, "customers": 350}],
        ),
    )
    assert res2.status_code == 422
    assert "customers" in res2.text.lower()


def test_model_inference_promotional_lift() -> None:
    """Verify that promotional campaign and discounted price trigger expected demand lift."""
    bundle = load_demand_forecast_bundle()

    # Baseline: no promo, normal price
    df_base = pd.DataFrame([{
        "category": "Beverages",
        "unit_price": 1.20,
        "base_unit_price": 1.20,
        "is_open": 1,
        "is_promo": 0,
        "promo2": 0,
        "day_of_week": 2,
        "state_holiday": "0",
        "school_holiday": 0,
        "store_type": "a",
        "assortment": "a",
    }])
    pred_base = predict_from_feature_dataframe(df_base, bundle=bundle)[0]

    # Promotional: active promo, 20% discount
    df_promo = pd.DataFrame([{
        "category": "Beverages",
        "unit_price": 0.96,
        "base_unit_price": 1.20,
        "is_open": 1,
        "is_promo": 1,
        "promo2": 0,
        "day_of_week": 2,
        "state_holiday": "0",
        "school_holiday": 0,
        "store_type": "a",
        "assortment": "a",
    }])
    pred_promo = predict_from_feature_dataframe(df_promo, bundle=bundle)[0]

    assert pred_base > 0.0
    assert pred_promo > pred_base
    # Lift should be notable (at least 20% lift)
    lift = (pred_promo - pred_base) / pred_base
    assert lift >= 0.20


def test_closed_day_strictly_zero() -> None:
    """Verify that is_open == 0 guarantees predicted_units == 0.0."""
    start = date(2026, 10, 1)
    daily_contexts = [
        DailyForecastContext(forecast_date=start + timedelta(days=0), is_open=1),
        DailyForecastContext(forecast_date=start + timedelta(days=1), is_open=0),
        DailyForecastContext(forecast_date=start + timedelta(days=2), is_open=1),
    ]

    request = DemandForecastRequest(
        product_id="SKU_003",
        store_id="1",
        category="Personal Care",
        base_unit_price=1.95,
        horizon=7,
        forecast_date=start,
        daily_contexts=daily_contexts,
    )

    response = create_demand_forecast(request)
    assert response.predictions[0].is_open == 1
    assert response.predictions[0].predicted_units > 0.0

    # Day 1 is explicitly closed
    assert response.predictions[1].is_open == 0
    assert response.predictions[1].predicted_units == 0.0


def test_horizons_7_14_30() -> None:
    """Verify horizons 1, 7, 14, and 30 are supported, and unsupported horizons are rejected."""
    for h in (1, 7, 14, 30):
        req = DemandForecastRequest(
            product_id="SKU_005",
            store_id="1",
            category="Household Cleaning",
            base_unit_price=5.50,
            horizon=h,
        )
        resp = create_demand_forecast(req)
        assert resp.horizon == h
        assert len(resp.predictions) == h

    # Unsupported horizons
    for bad_h in (0, 5, 10, 20, 45, -1):
        payload = {
            "product_id": "SKU_005",
            "store_id": "1",
            "category": "Household Cleaning",
            "base_unit_price": 5.50,
            "horizon": bad_h,
        }
        res = client.post("/v1/demand-forecast/forecast", json=payload)
        assert res.status_code == 422


def test_deterministic_prediction() -> None:
    """Verify identical requests produce exact identical predictions."""
    payload = {
        "product_id": "SKU_001",
        "store_id": "1",
        "category": "Beverages",
        "base_unit_price": 1.20,
        "horizon": 7,
    }

    res1 = client.post("/v1/demand-forecast/forecast", json=payload)
    res2 = client.post("/v1/demand-forecast/forecast", json=payload)

    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res1.json() == res2.json()


def test_openapi_route_registration() -> None:
    """Verify route is properly registered in OpenAPI specification."""
    res = client.get("/openapi.json")
    assert res.status_code == 200
    paths = res.json()["paths"]
    assert "/v1/demand-forecast/forecast" in paths
    assert "post" in paths["v1/demand-forecast/forecast" if "v1/demand-forecast/forecast" in paths else "/v1/demand-forecast/forecast"]

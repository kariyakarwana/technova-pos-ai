from pathlib import Path

import joblib
from fastapi.testclient import TestClient

from technova_ai_service.main import app


def demand_context() -> dict[str, float]:
    return {
        "demand_lag_1": 8,
        "demand_lag_7": 10,
        "demand_lag_14": 9,
        "rolling_mean_7": 9.5,
        "rolling_mean_28": 8.7,
        "rolling_std_7": 2.1,
        "price_index_28": 1.0,
        "orders_lag_1": 5,
        "customers_lag_1": 5,
    }


def main() -> None:
    client = TestClient(app)
    stock_response = client.post(
        "/v1/inventory/forecast",
        json={
            "product_id": "TEST-SKU",
            "forecast_date": "2026-09-18",
            "current_stock": 40,
            "unit_price": 150,
            **demand_context(),
        },
    )
    stock_response.raise_for_status()

    pricing_response = client.post(
        "/v1/pricing/recommend",
        json={
            "product_id": "TEST-SKU",
            "pricing_date": "2026-09-18",
            "current_price": 150,
            "unit_cost": 100,
            **demand_context(),
        },
    )
    pricing_response.raise_for_status()

    loyalty_bundle = joblib.load(Path("artifacts/loyalty_recommendations/model.joblib"))
    customer_id = loyalty_bundle["customer_ids"][0]
    loyalty_response = client.post(
        "/v1/loyalty/recommendations",
        json={"customer_id": customer_id, "top_k": 5},
    )
    loyalty_response.raise_for_status()

    print("Inventory:", stock_response.json())
    print("Pricing:", pricing_response.json())
    print("Loyalty:", loyalty_response.json())


if __name__ == "__main__":
    main()

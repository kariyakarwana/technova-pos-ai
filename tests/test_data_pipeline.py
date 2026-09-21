from datetime import datetime, timedelta

import pandas as pd

from technova_ai_service.data.online_retail import clean_transactions
from technova_ai_service.features.stock_intelligence.features import build_stock_features


def test_clean_transactions_removes_returns_and_invalid_values() -> None:
    frame = pd.DataFrame(
        {
            "invoice_no": ["100", "C101", "102", "103"],
            "product_id": ["A", "A", "B", "C"],
            "description": ["A", "A", "B", "C"],
            "quantity": [2, 1, -1, 1],
            "invoice_date": [datetime(2026, 1, 1)] * 4,
            "unit_price": [10.0, 10.0, 5.0, 0.0],
            "customer_id": ["1", "1", "2", "3"],
            "country": ["LK"] * 4,
        }
    )
    cleaned = clean_transactions(frame)
    assert cleaned["invoice_no"].tolist() == ["100"]
    assert cleaned["revenue"].tolist() == [20.0]


def test_stock_features_use_prior_demand_only() -> None:
    start = datetime(2026, 1, 1)
    panel = pd.DataFrame(
        {
            "date": [start + timedelta(days=index) for index in range(40)],
            "product_id": ["SKU-1"] * 40,
            "quantity": list(range(40)),
            "unit_price": [100.0] * 40,
            "orders": [1.0] * 40,
            "customers": [1.0] * 40,
            "description": ["Product"] * 40,
            "revenue": [0.0] * 40,
        }
    )
    features = build_stock_features(panel)
    row = features.loc[features["quantity"].eq(28)].iloc[0]
    assert row["demand_lag_1"] == 27
    assert row["demand_lag_7"] == 21
    assert row["rolling_mean_7"] == 24

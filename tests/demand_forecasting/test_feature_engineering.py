from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from technova_ai_service.features.demand_forecasting.features import (
    build_demand_features,
    extract_feature_vector,
)
from technova_ai_service.features.demand_forecasting.holidays import (
    KNOWN_POYA_DATES,
    get_sri_lankan_holiday_flags,
)
from technova_ai_service.features.demand_forecasting.panel import (
    build_daily_demand_panel,
)
from technova_ai_service.features.demand_forecasting.schemas import (
    DEMAND_FEATURE_COLUMNS,
    DEMAND_LAG_HORIZONS,
    ROLLING_WINDOWS,
)


def _generate_synthetic_panel(days: int = 60, seed: int = 42) -> pd.DataFrame:
    start = date(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(days)]
    quantities = [float(x) for x in range(days)]

    records = [
        {
            "organization_id": "org-1",
            "branch_id": "branch-1",
            "product_id": "sku-1",
            "date": dt,
            "quantity": qty,
            "unit_price": 100.0,
            "cost_price": 60.0,
            "selling_price": 100.0,
            "reorder_level": 5.0,
            "has_scheduled_discount": int(i % 10 == 0),
            "discount_rate": 0.15 if i % 10 == 0 else 0.0,
        }
        for i, (dt, qty) in enumerate(zip(dates, quantities, strict=True))
    ]
    return build_daily_demand_panel(records)


def test_lag_correctness() -> None:
    panel = _generate_synthetic_panel(days=60)
    features = build_demand_features(panel, drop_incomplete_lags=True)

    # For an arithmetic sequence quantity = 0, 1, 2, ...
    # At index row corresponding to day 35, quantity should be 35
    row_35 = features.loc[features["quantity"] == 35.0].iloc[0]

    for lag in DEMAND_LAG_HORIZONS:
        expected = 35.0 - lag
        assert row_35[f"demand_lag_{lag}"] == expected


def test_rolling_statistics_correctness() -> None:
    panel = _generate_synthetic_panel(days=60)
    features = build_demand_features(panel, drop_incomplete_lags=True)

    row_40 = features.loc[features["quantity"] == 40.0].iloc[0]

    # Prior 7 days: [33, 34, 35, 36, 37, 38, 39]
    prior_7 = list(range(33, 40))
    expected_mean_7 = float(np.mean(prior_7))
    expected_std_7 = float(np.std(prior_7, ddof=1))
    expected_max_7 = float(max(prior_7))

    assert row_40["rolling_mean_7"] == pytest.approx(expected_mean_7)
    assert row_40["rolling_std_7"] == pytest.approx(expected_std_7)
    assert row_40["rolling_max_7"] == expected_max_7

    # Prior 14 days: [26..39]
    prior_14 = list(range(26, 40))
    assert row_40["rolling_mean_14"] == pytest.approx(float(np.mean(prior_14)))
    assert row_40["rolling_max_14"] == float(max(prior_14))


def test_strict_temporal_leakage_prevention() -> None:
    """Modifying quantity at day T must NOT change any feature computed for day T."""
    panel_a = _generate_synthetic_panel(days=45, seed=1)
    panel_b = panel_a.copy()

    start = date(2026, 1, 1)
    target_day = start + timedelta(days=34)

    # In panel_b, spike the demand on target_day from normal to 999.0
    panel_b.loc[panel_b["date"] == target_day, "quantity"] = 999.0

    features_a = build_demand_features(panel_a, drop_incomplete_lags=False)
    features_b = build_demand_features(panel_b, drop_incomplete_lags=False)

    row_a = features_a.loc[features_a["date"] == target_day].iloc[0]
    row_b = features_b.loc[features_b["date"] == target_day].iloc[0]

    # Check every lag and rolling feature
    for lag in DEMAND_LAG_HORIZONS:
        assert row_a[f"demand_lag_{lag}"] == row_b[f"demand_lag_{lag}"]

    for window in ROLLING_WINDOWS:
        assert row_a[f"rolling_mean_{window}"] == row_b[f"rolling_mean_{window}"]
        assert row_a[f"rolling_std_{window}"] == row_b[f"rolling_std_{window}"]
        assert row_a[f"rolling_max_{window}"] == row_b[f"rolling_max_{window}"]


def test_forbidden_columns_leakage_exception() -> None:
    panel = _generate_synthetic_panel(days=35)
    panel["discount_application"] = 100.0  # Forbidden actual redemption column

    with pytest.raises(ValueError, match="Data leakage detected"):
        build_demand_features(panel)


def test_price_index_calculation() -> None:
    panel = _generate_synthetic_panel(days=40)
    start = date(2026, 1, 1)
    spike_day = start + timedelta(days=34)
    # Give product a steady price of 100.0, then change to 120.0 at day 35
    panel.loc[panel["date"] >= spike_day, "unit_price"] = 120.0

    features = build_demand_features(panel, drop_incomplete_lags=False)

    day_before = start + timedelta(days=33)
    row_before = features.loc[features["date"] == day_before].iloc[0]
    assert row_before["price_index_28"] == pytest.approx(1.0)

    # On spike_day, prior 28 days median was 100, current price is 120 -> price_index = 1.2
    row_spike = features.loc[features["date"] == spike_day].iloc[0]
    assert row_spike["price_index_28"] == pytest.approx(1.2)


def test_sri_lankan_holiday_features() -> None:
    # 1. Independence Day: Feb 4
    flags_independence = get_sri_lankan_holiday_flags(date(2026, 2, 4))
    assert flags_independence["is_public_holiday"] == 1
    assert flags_independence["is_mercantile_holiday"] == 1

    # 2. Sinhala & Tamil New Year: April 14
    flags_new_year = get_sri_lankan_holiday_flags(date(2026, 4, 14))
    assert flags_new_year["is_public_holiday"] == 1
    assert flags_new_year["is_festive_peak"] == 1

    # 3. Known Poya Day: May 1, 2026 (Vesak Poya)
    assert date(2026, 5, 1) in KNOWN_POYA_DATES
    flags_vesak = get_sri_lankan_holiday_flags(date(2026, 5, 1))
    assert flags_vesak["is_poya_day"] == 1
    assert flags_vesak["is_public_holiday"] == 1
    assert flags_vesak["is_festive_peak"] == 1

    # 4. Standard business day (e.g., Feb 10, 2026)
    flags_standard = get_sri_lankan_holiday_flags(date(2026, 2, 10))
    assert flags_standard["is_public_holiday"] == 0
    assert flags_standard["is_poya_day"] == 0


def test_extract_feature_vector_contains_all_columns() -> None:
    panel = _generate_synthetic_panel(days=45)
    features = build_demand_features(panel, drop_incomplete_lags=True)

    row = features.iloc[0]
    vector = extract_feature_vector(row)

    assert set(vector.keys()) == set(DEMAND_FEATURE_COLUMNS)
    assert all(not pd.isna(val) for val in vector.values())

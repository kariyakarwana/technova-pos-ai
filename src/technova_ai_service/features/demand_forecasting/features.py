from typing import Any

import numpy as np
import pandas as pd

from technova_ai_service.features.demand_forecasting.holidays import (
    get_sri_lankan_holiday_flags,
)
from technova_ai_service.features.demand_forecasting.schemas import (
    DEMAND_FEATURE_COLUMNS,
    DEMAND_LAG_HORIZONS,
    ROLLING_WINDOWS,
)

FORBIDDEN_LEAKAGE_COLUMNS: set[str] = {
    "discount_application",
    "discount_applications",
    "discountapplication",
    "actual_discount",
    "actual_discount_amount",
    "sale_item_discount_total",
    "future_sale",
    "future_quantity",
}


def build_demand_features(
    panel: pd.DataFrame,
    *,
    drop_incomplete_lags: bool = True,
    custom_holiday_dates: dict[str, set[Any]] | None = None,
) -> pd.DataFrame:
    """Extract model-ready feature rows from a daily Product x Branch x Date demand panel.

    Guarantees:
    - Every lag and rolling feature is shifted by at least 1 day.
    - Zero future leakage: only historical sales (<= t-1) are used for autoregressive features.
    - Zero future discount leakage: relies solely on scheduled promotion flags, never actual redemptions.
    - Multi-tenant isolation: grouped strictly by (organization_id, branch_id, product_id).
    """
    # 1. Leakage Guardrail: check for forbidden columns
    lower_cols = {col.lower() for col in panel.columns}
    for forbidden in FORBIDDEN_LEAKAGE_COLUMNS:
        if forbidden in lower_cols:
            raise ValueError(
                f"Data leakage detected! Forbidden column '{forbidden}' present in demand features input."
            )

    frame = panel.copy()

    # Ensure chronological order within each entity group
    frame["date_dt"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(
        ["organization_id", "branch_id", "product_id", "date_dt"]
    ).reset_index(drop=True)

    grouped = frame.groupby(
        ["organization_id", "branch_id", "product_id"],
        observed=True,
    )

    # 2. Demand Lags: strictly shifted by k days
    for lag in DEMAND_LAG_HORIZONS:
        frame[f"demand_lag_{lag}"] = grouped["quantity"].shift(lag)

    # 3. Rolling Statistics: strictly shifted by 1 day (prior days only)
    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = grouped["quantity"].transform(
            lambda s, window=window: s.shift(1).rolling(window, min_periods=1).mean()
        )
        frame[f"rolling_std_{window}"] = grouped["quantity"].transform(
            lambda s, window=window: s.shift(1).rolling(window, min_periods=2).std().fillna(0.0)
        )
        frame[f"rolling_max_{window}"] = grouped["quantity"].transform(
            lambda s, window=window: s.shift(1).rolling(window, min_periods=1).max()
        )

    # 4. Price Features & Price Index 28:
    # 28-day historical rolling median price (shifted 1 day)
    rolling_median_price = grouped["unit_price"].transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).median()
    )
    frame["price_index_28"] = (
        frame["unit_price"] / rolling_median_price.replace(0, np.nan)
    ).fillna(1.0).clip(lower=0.1, upper=10.0)

    # Margin Rate
    selling = frame["selling_price"].clip(lower=0.01)
    frame["margin_rate"] = ((frame["selling_price"] - frame["cost_price"]) / selling).clip(
        lower=-1.0, upper=1.0
    )

    # 5. Scheduled Promotion Features
    frame["has_scheduled_discount"] = frame["has_scheduled_discount"].fillna(0).astype(int)
    frame["discount_rate"] = frame["discount_rate"].fillna(0.0).clip(lower=0.0, upper=1.0)

    # 6. Calendar Signals
    frame["day_of_week"] = frame["date_dt"].dt.dayofweek
    frame["day_of_month"] = frame["date_dt"].dt.day
    frame["month"] = frame["date_dt"].dt.month
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame["is_month_start"] = frame["date_dt"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["date_dt"].dt.is_month_end.astype(int)
    frame["is_payday_window"] = (
        frame["day_of_month"].isin(list(range(25, 32)) + [1, 2, 3]).astype(int)
    )

    # 7. Sri Lankan Holiday & Festive Calendar Features
    custom_poyas = custom_holiday_dates.get("poya") if custom_holiday_dates else None
    custom_public = custom_holiday_dates.get("public") if custom_holiday_dates else None
    custom_mercantile = custom_holiday_dates.get("mercantile") if custom_holiday_dates else None

    holiday_records = [
        get_sri_lankan_holiday_flags(
            row_date,
            custom_poya_dates=custom_poyas,
            custom_public_holidays=custom_public,
            custom_mercantile_holidays=custom_mercantile,
        )
        for row_date in frame["date"]
    ]
    holiday_df = pd.DataFrame(holiday_records, index=frame.index)
    frame["is_public_holiday"] = holiday_df["is_public_holiday"]
    frame["is_mercantile_holiday"] = holiday_df["is_mercantile_holiday"]
    frame["is_poya_day"] = holiday_df["is_poya_day"]
    frame["is_festive_peak"] = holiday_df["is_festive_peak"]

    # 8. Operating Context Indicators
    frame["is_in_stock"] = frame["is_in_stock"].fillna(1).astype(int)
    frame["is_branch_open"] = frame["is_branch_open"].fillna(1).astype(int)

    # Clean intermediate columns
    frame = frame.drop(columns=["date_dt"])

    # 9. Handle Incomplete Lags (e.g., initial 28 days of history)
    if drop_incomplete_lags:
        lag_cols = [f"demand_lag_{lag}" for lag in DEMAND_LAG_HORIZONS]
        frame = frame.dropna(subset=lag_cols).reset_index(drop=True)

    return frame


def extract_feature_vector(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Validate and extract the exact model-ready feature dictionary."""
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    missing = [col for col in DEMAND_FEATURE_COLUMNS if col not in data]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    return {col: data[col] for col in DEMAND_FEATURE_COLUMNS}

import pandas as pd

STOCK_FEATURE_COLUMNS = [
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "unit_price",
    "price_index_28",
    "orders_lag_1",
    "customers_lag_1",
    "day_of_week",
    "month",
    "is_weekend",
]


def build_stock_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.sort_values(["product_id", "date"]).copy()
    grouped = frame.groupby("product_id", observed=True)

    for lag in (1, 7, 14):
        frame[f"demand_lag_{lag}"] = grouped["quantity"].shift(lag)

    frame["rolling_mean_7"] = grouped["quantity"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=3).mean()
    )
    frame["rolling_mean_28"] = grouped["quantity"].transform(
        lambda values: values.shift(1).rolling(28, min_periods=7).mean()
    )
    frame["rolling_std_7"] = grouped["quantity"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=3).std()
    )
    rolling_price = grouped["unit_price"].transform(
        lambda values: values.shift(1).rolling(28, min_periods=7).median()
    )
    frame["price_index_28"] = frame["unit_price"] / rolling_price.clip(lower=0.01)
    frame["orders_lag_1"] = grouped["orders"].shift(1)
    frame["customers_lag_1"] = grouped["customers"].shift(1)
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame[STOCK_FEATURE_COLUMNS] = frame[STOCK_FEATURE_COLUMNS].replace(
        [float("inf"), float("-inf")], pd.NA
    )
    return frame.dropna(subset=STOCK_FEATURE_COLUMNS).reset_index(drop=True)

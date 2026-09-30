from __future__ import annotations

import datetime
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from technova_ai_service.modeling.artifacts import load_artifact


@lru_cache(maxsize=4)
def load_forecast_bundle(path: str) -> dict[str, Any]:
    return load_artifact(Path(path))


def predict_sales_forecast(
    *,
    artifact_path: Path,
    organization_id: str,
    branch_id: str,
    forecast_horizon: int,
    recent_daily_revenue: list[Any],
    future_calendar: list[Any],
) -> list[dict[str, Any]]:
    """Generate multi-day sales forecast using trained TechNova XGBoost model.

    Guarantees:
    - Loads model bundle using existing artifact loading utilities.
    - Fails with FileNotFoundError if artifact is missing.
    - Zero future leakage: forecasts are generated autoregressively.
    - Closed / non-operating days strictly receive 0.0 revenue.
    - Predictions are strictly >= 0.0.
    - Feature column order exactly matches bundle['feature_columns'].
    """
    bundle = load_forecast_bundle(str(artifact_path))
    model = bundle["model"]
    feature_columns: list[str] = bundle["feature_columns"]

    # 1. Parse and sort recent daily revenue history
    parsed_history: list[dict[str, Any]] = []
    for item in recent_daily_revenue:
        d = item.date if isinstance(item.date, datetime.date) else datetime.date.fromisoformat(str(item.date))
        rev = float(item.revenue)
        op = int(getattr(item, "is_operating_day", 1))
        parsed_history.append({"date": d, "revenue": rev, "is_operating_day": op})

    parsed_history.sort(key=lambda x: x["date"])

    # Require real historical data: no synthetic fallbacks
    if not parsed_history or not any(p["revenue"] > 0 for p in parsed_history):
        raise ValueError("Insufficient historical data")

    history_revenues = [p["revenue"] for p in parsed_history]
    branch_inception = parsed_history[0]["date"]

    # Pad history to at least 28 entries using the earliest available revenue
    while len(history_revenues) < 28:
        history_revenues.insert(0, history_revenues[0])

    # 2. Parse future calendar points
    horizon_calendar = future_calendar[:forecast_horizon]
    predictions: list[dict[str, Any]] = []

    for item in horizon_calendar:
        cur_date = item.date if isinstance(item.date, datetime.date) else datetime.date.fromisoformat(str(item.date))
        is_operating = int(getattr(item, "is_operating_day", 1))
        active_discounts = int(getattr(item, "active_discounts", 0))

        if is_operating == 0:
            pred_revenue = 0.0
        else:
            # Calendar features
            dow = cur_date.weekday()
            day_of_month = cur_date.day
            month = cur_date.month
            week_of_year = cur_date.isocalendar()[1]
            is_weekend = 1 if dow >= 5 else 0
            is_month_start = 1 if day_of_month == 1 else 0
            days_in_month = pd.Period(cur_date.strftime("%Y-%m")).days_in_month
            is_month_end = 1 if day_of_month == days_in_month else 0

            day_of_week_sin = round(math.sin((2 * math.pi * dow) / 7), 4)
            day_of_week_cos = round(math.cos((2 * math.pi * dow) / 7), 4)
            month_sin = round(math.sin((2 * math.pi * (month - 1)) / 12), 4)
            month_cos = round(math.cos((2 * math.pi * (month - 1)) / 12), 4)

            # Lags (strictly prior to cur_date)
            rev_lag_1 = history_revenues[-1]
            rev_lag_7 = history_revenues[-7]
            rev_lag_14 = history_revenues[-14]
            rev_lag_28 = history_revenues[-28]

            # Rolling stats (strictly prior to cur_date)
            w7 = history_revenues[-7:]
            rolling_mean_7 = float(np.mean(w7))
            rolling_std_7 = float(np.std(w7, ddof=1)) if len(w7) > 1 else 0.0

            w14 = history_revenues[-14:]
            rolling_mean_14 = float(np.mean(w14))

            w28 = history_revenues[-28:]
            rolling_mean_28 = float(np.mean(w28))
            rolling_std_28 = float(np.std(w28, ddof=1)) if len(w28) > 1 else 0.0

            prev_w7 = history_revenues[-14:-7]
            prev_mean_7 = float(np.mean(prev_w7))
            weekly_momentum_ratio = (
                round(rolling_mean_7 / prev_mean_7, 4) if prev_mean_7 > 0 else 1.0
            )

            # Profile features
            branch_age_days = (cur_date - branch_inception).days + 180
            branch_historical_avg_sales = float(np.mean(history_revenues))

            row_features = {
                "day_of_week": dow,
                "day_of_month": day_of_month,
                "month": month,
                "week_of_year": int(week_of_year),
                "is_weekend": is_weekend,
                "is_month_start": is_month_start,
                "is_month_end": is_month_end,
                "day_of_week_sin": day_of_week_sin,
                "day_of_week_cos": day_of_week_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "revenue_lag_1": rev_lag_1,
                "revenue_lag_7": rev_lag_7,
                "revenue_lag_14": rev_lag_14,
                "revenue_lag_28": rev_lag_28,
                "rolling_mean_7": round(rolling_mean_7, 2),
                "rolling_mean_14": round(rolling_mean_14, 2),
                "rolling_mean_28": round(rolling_mean_28, 2),
                "rolling_std_7": round(rolling_std_7, 2),
                "rolling_std_28": round(rolling_std_28, 2),
                "weekly_momentum_ratio": weekly_momentum_ratio,
                "branch_age_days": branch_age_days,
                "branch_historical_avg_sales": round(branch_historical_avg_sales, 2),
                "is_operating_day": is_operating,
                "active_discount_count": active_discounts,
            }

            row_df = pd.DataFrame([row_features])[feature_columns]
            pred = float(model.predict(row_df)[0])
            pred_revenue = max(0.0, round(pred, 2))

        predictions.append(
            {
                "date": cur_date,
                "predicted_revenue": pred_revenue,
            }
        )
        # Update autoregressive history pool
        history_revenues.append(pred_revenue)

    return predictions

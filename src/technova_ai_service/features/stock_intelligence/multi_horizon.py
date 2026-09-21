from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from datetime import date, timedelta
from statistics import NormalDist
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from technova_ai_service.features.stock_intelligence.inference import load_stock_bundle


class DailyPoint(TypedDict):
    date: date
    predicted_demand: float


def _pad_history(values: list[float], minimum: int = 28) -> list[float]:
    history = [float(value) for value in values]
    if len(history) >= minimum:
        return history
    repeats = (minimum + len(history) - 1) // len(history)
    return (history * repeats)[-minimum:]


def _data_readiness(history_days: int) -> dict[str, Any]:
    if history_days < 28:
        return {
            "history_days": history_days,
            "maturity": "cold_start",
            "confidence": "low",
            "production_ready": False,
            "recommended_action": "Collect at least 28 daily observations and require review.",
        }
    if history_days < 90:
        return {
            "history_days": history_days,
            "maturity": "limited_history",
            "confidence": "low",
            "production_ready": False,
            "recommended_action": "Refresh daily and keep purchase approval manual.",
        }
    if history_days < 365:
        return {
            "history_days": history_days,
            "maturity": "developing",
            "confidence": "medium",
            "production_ready": False,
            "recommended_action": "Continue collecting local seasonal and promotion history.",
        }
    return {
        "history_days": history_days,
        "maturity": "annual_history_available",
        "confidence": "medium",
        "production_ready": False,
        "recommended_action": "Retrain and backtest on TechNova data before production approval.",
    }


def _direct_horizon_forecasts(
    recent_daily_demand: list[float], service_level: float
) -> list[dict[str, float | int | str]]:
    window = np.asarray(recent_daily_demand[-28:], dtype=float)
    mean_demand = float(window.mean())
    standard_deviation = float(window.std(ddof=1)) if len(window) > 1 else 0.0
    z_score = NormalDist().inv_cdf(service_level)
    minimum_interval_rate = 0.30 if len(recent_daily_demand) < 90 else 0.20
    forecasts: list[dict[str, float | int | str]] = []
    for horizon in (7, 14, 30):
        point = mean_demand * horizon
        statistical_width = z_score * standard_deviation * np.sqrt(horizon)
        width = max(float(statistical_width), point * minimum_interval_rate)
        forecasts.append(
            {
                "horizon_days": horizon,
                "predicted_demand": point,
                "lower_bound": max(point - width, 0.0),
                "upper_bound": point + width,
                "interval_level": service_level,
                "method": "recent_demand_direct_forecast",
            }
        )
    return forecasts


def _scale_slice(values: list[float], start: int, end: int, target: float) -> None:
    end = min(end, len(values))
    if start >= end:
        return
    current = float(sum(values[start:end]))
    if current <= 0:
        replacement = target / (end - start)
        values[start:end] = [replacement] * (end - start)
        return
    scale = target / current
    values[start:end] = [value * scale for value in values[start:end]]


def _calibrate_daily_predictions(
    predictions: list[float], direct: list[dict[str, float | int | str]]
) -> list[float]:
    calibrated = list(predictions)
    totals = {int(item["horizon_days"]): float(item["predicted_demand"]) for item in direct}
    _scale_slice(calibrated, 0, 7, totals[7])
    _scale_slice(calibrated, 7, 14, max(totals[14] - totals[7], 0.0))
    _scale_slice(calibrated, 14, 30, max(totals[30] - totals[14], 0.0))

    daily_target = totals[30] / 30
    for start in range(30, len(calibrated), 30):
        end = min(start + 30, len(calibrated))
        _scale_slice(calibrated, start, end, daily_target * (end - start))
    return calibrated


def _feature_row(
    *,
    forecast_date: date,
    history: list[float],
    unit_price: float,
    price_index_28: float,
    orders_lag_1: float,
    customers_lag_1: float,
) -> dict[str, float]:
    recent_7 = np.asarray(history[-7:], dtype=float)
    recent_28 = np.asarray(history[-28:], dtype=float)
    return {
        "demand_lag_1": history[-1],
        "demand_lag_7": history[-7],
        "demand_lag_14": history[-14],
        "rolling_mean_7": float(recent_7.mean()),
        "rolling_mean_28": float(recent_28.mean()),
        "rolling_std_7": float(recent_7.std(ddof=1)),
        "unit_price": unit_price,
        "price_index_28": price_index_28,
        "orders_lag_1": orders_lag_1,
        "customers_lag_1": customers_lag_1,
        "day_of_week": float(forecast_date.weekday()),
        "month": float(forecast_date.month),
        "is_weekend": float(forecast_date.weekday() >= 5),
    }


def _aggregate(
    daily: list[DailyPoint],
    key_builder: Callable[[date], tuple[str, date, date]],
) -> list[dict[str, Any]]:
    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for point in daily:
        point_date = point["date"]
        label, period_start, period_end = key_builder(point_date)
        group = groups.setdefault(
            label,
            {
                "label": label,
                "start_date": max(period_start, daily[0]["date"]),
                "end_date": min(period_end, daily[-1]["date"]),
                "values": [],
            },
        )
        group["values"].append(float(point["predicted_demand"]))

    result: list[dict[str, Any]] = []
    for group in groups.values():
        values = group.pop("values")
        result.append(
            {
                **group,
                "total_demand": float(sum(values)),
                "average_daily_demand": float(np.mean(values)),
            }
        )
    return result


def _week_key(value: date) -> tuple[str, date, date]:
    iso_year, iso_week, _ = value.isocalendar()
    start = value - timedelta(days=value.weekday())
    return f"{iso_year}-W{iso_week:02d}", start, start + timedelta(days=6)


def _month_key(value: date) -> tuple[str, date, date]:
    start = value.replace(day=1)
    next_month = date(value.year + (value.month == 12), value.month % 12 + 1, 1)
    return value.strftime("%Y-%m"), start, next_month - timedelta(days=1)


def _season_key(value: date) -> tuple[str, date, date]:
    quarter = (value.month - 1) // 3 + 1
    start_month = (quarter - 1) * 3 + 1
    start = date(value.year, start_month, 1)
    if quarter == 4:
        end = date(value.year, 12, 31)
    else:
        end = date(value.year, start_month + 3, 1) - timedelta(days=1)
    names = {1: "Jan-Mar", 2: "Apr-Jun", 3: "Jul-Sep", 4: "Oct-Dec"}
    return f"{value.year}-Q{quarter} ({names[quarter]})", start, end


def _year_key(value: date) -> tuple[str, date, date]:
    return str(value.year), date(value.year, 1, 1), date(value.year, 12, 31)


def forecast_multiple_horizons(
    *,
    artifact_path: str,
    start_date: date,
    forecast_days: int,
    recent_daily_demand: list[float],
    current_stock: float,
    lead_time_days: int,
    review_period_days: int,
    service_level: float,
    unit_price: float,
    price_index_28: float,
    orders_lag_1: float,
    customers_lag_1: float,
) -> dict[str, Any]:
    bundle = load_stock_bundle(artifact_path)
    columns: list[str] = bundle["feature_columns"]
    history_days = len(recent_daily_demand)
    history = _pad_history(recent_daily_demand)
    daily: list[DailyPoint] = []

    for offset in range(forecast_days):
        current_date = start_date + timedelta(days=offset)
        features = _feature_row(
            forecast_date=current_date,
            history=history,
            unit_price=unit_price,
            price_index_28=price_index_28,
            orders_lag_1=orders_lag_1,
            customers_lag_1=customers_lag_1,
        )
        row = pd.DataFrame([{column: features[column] for column in columns}])
        prediction = max(float(np.expm1(bundle["model"].predict(row)[0])), 0.0)
        daily.append({"date": current_date, "predicted_demand": prediction})
        history.append(prediction)

    raw_predictions = [float(point["predicted_demand"]) for point in daily]
    direct_horizons = _direct_horizon_forecasts(recent_daily_demand, service_level)
    predictions = _calibrate_daily_predictions(raw_predictions, direct_horizons)
    for point, prediction in zip(daily, predictions, strict=True):
        point["predicted_demand"] = prediction
    recent_std = float(np.std(recent_daily_demand[-7:], ddof=1))
    uncertainty = max(recent_std, float(np.mean(predictions[:7])) * 0.10)
    safety_stock = (
        NormalDist().inv_cdf(service_level) * uncertainty * np.sqrt(lead_time_days)
    )
    lead_time_demand = float(sum(predictions[:lead_time_days]))
    target_days = lead_time_days + review_period_days
    target_stock = float(sum(predictions[:target_days]) + safety_stock)
    reorder_point = lead_time_demand + safety_stock
    recommended_order = max(target_stock - current_stock, 0.0)
    stockout_risk = 1.0 if current_stock <= 0 else min(reorder_point / current_stock, 1.0)
    if current_stock <= reorder_point:
        status = "reorder_now"
    elif current_stock <= target_stock:
        status = "monitor"
    else:
        status = "healthy"

    peak_index = int(np.argmax(predictions))
    return {
        "summary": {
            "forecast_days": forecast_days,
            "total_predicted_demand": float(sum(predictions)),
            "average_daily_demand": float(np.mean(predictions)),
            "next_7_days_demand": float(sum(predictions[:7])),
            "next_30_days_demand": float(sum(predictions[:30])),
            "peak_demand_date": daily[peak_index]["date"],
            "peak_daily_demand": predictions[peak_index],
        },
        "stock_plan": {
            "current_stock": current_stock,
            "lead_time_demand": lead_time_demand,
            "safety_stock": float(safety_stock),
            "reorder_point": float(reorder_point),
            "target_stock": target_stock,
            "recommended_order_quantity": float(recommended_order),
            "stockout_risk": float(stockout_risk),
            "status": status,
        },
        "direct_horizons": direct_horizons,
        "data_readiness": _data_readiness(history_days),
        "daily": daily,
        "weekly": _aggregate(daily, _week_key),
        "monthly": _aggregate(daily, _month_key),
        "seasonal": _aggregate(daily, _season_key),
        "yearly": _aggregate(daily, _year_key),
        "notes": [
            "Weekly, monthly, seasonal and yearly totals are aggregated from the daily forecast.",
            (
                "Seasonal periods are calendar quarters; named Sri Lankan retail events require "
                "event features and TechNova history."
            ),
            (
                "Long-range forecasts are planning estimates and should be refreshed as actual "
                "sales arrive."
            ),
            (
                "The 7, 14 and 30 day totals use a low-data direct forecast and calibrate the ML "
                "daily curve to prevent recursive demand collapse."
            ),
            "Uncertainty ranges are statistical planning bands, not yet empirically calibrated.",
            "Production readiness remains false until TechNova data is retrained and backtested.",
        ],
    }

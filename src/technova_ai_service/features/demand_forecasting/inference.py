from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from technova_ai_service.config import get_settings
from technova_ai_service.features.demand_forecasting.schemas import (
    DailyUnitForecastPoint,
    DemandForecastModelInfo,
    DemandForecastRequest,
)

DEFAULT_DEMAND_ARTIFACT_PATH = (
    Path("artifacts") / "demand_forecasting" / "model.joblib"
)

CATEGORY_ORDER: list[str] = [
    "Beverages",
    "Health",
    "Household Cleaning",
    "Packaged Foods",
    "Personal Care",
]
STATE_HOLIDAY_ORDER: list[str] = ["0", "a", "b", "c"]
STORE_TYPE_ORDER: list[str] = ["a", "b", "c", "d"]
ASSORTMENT_ORDER: list[str] = ["a", "b", "c"]


@lru_cache(maxsize=4)
def load_demand_forecast_bundle(path_str: str | None = None) -> dict[str, Any]:
    """Load and cache the trained Demand Forecasting model artifact.

    Validates artifact structure:
    - Must contain 'model', 'feature_columns', 'model_type'
    - 'model' must provide a callable predict method
    - 'feature_columns' must be a non-empty sequence
    """
    if path_str is not None:
        model_path = Path(path_str)
    else:
        model_path = get_settings().artifact_dir / "demand_forecasting" / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Demand forecasting model artifact not found at {model_path}."
        )

    bundle: dict[str, Any] = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Invalid artifact format in {model_path}: expected dictionary bundle.")

    required_keys = ("model", "feature_columns", "model_type")
    for key in required_keys:
        if key not in bundle:
            raise ValueError(f"Invalid model artifact structure in {model_path}: missing '{key}'.")

    if not hasattr(bundle["model"], "predict"):
        raise ValueError(f"Model object in {model_path} does not implement a 'predict' method.")

    if not bundle["feature_columns"] or not isinstance(bundle["feature_columns"], list):
        raise ValueError(f"Invalid feature_columns in {model_path}: expected non-empty list of column names.")

    return bundle


def resolve_categorical_levels(bundle: dict[str, Any] | None, column: str, fallback_levels: list[str]) -> list[str]:
    """Retrieve known training categorical levels from model artifact bundle or booster metadata."""
    if bundle is not None:
        if "categorical_categories" in bundle and column in bundle["categorical_categories"]:
            return list(bundle["categorical_categories"][column])
        model = bundle.get("model")
        booster = getattr(model, "booster_", None)
        if booster is not None:
            try:
                dump = booster.dump_model()
                pandas_cat = dump.get("pandas_categorical", [])
                feature_names = dump.get("feature_names", [])
                if column in feature_names:
                    idx = feature_names.index(column)
                    if idx < len(pandas_cat) and pandas_cat[idx]:
                        return list(pandas_cat[idx])
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
    return fallback_levels


def safe_categorical_series(series: pd.Series, categories: list[str]) -> pd.Series:
    """Encode a series into pandas Categorical, safely mapping unseen/unknown categories to NaN (code -1)

    without triggering warnings or inventing fake categories.
    """
    cat_set = set(categories)
    s_str = series.astype(str)
    masked = s_str.where(s_str.isin(cat_set), other=np.nan)
    cat_dtype = pd.CategoricalDtype(categories=categories)
    return pd.Series(pd.Categorical(masked, dtype=cat_dtype), index=series.index)


def predict_from_feature_dataframe(
    df: pd.DataFrame,
    bundle: dict[str, Any] | None = None,
    artifact_path: str | Path | None = None,
) -> np.ndarray:
    """Predict physical demand units from a feature DataFrame matching model features.

    Guarantees:
    - Validates presence of all required feature columns
    - Re-indexes to exact model training column order
    - Encodes categoricals preserving training categories while safely handling unseen tenant categories
    - Inverts log1p transformation via expm1
    - Clips all predictions to non-negative (>= 0.0)
    - Enforces closed-day zero demand (is_open == 0 -> predicted_units = 0.0)
    """
    if bundle is None:
        p_str = str(artifact_path) if artifact_path else None
        bundle = load_demand_forecast_bundle(p_str)

    feature_columns: list[str] = bundle["feature_columns"]
    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    df_eval = df[feature_columns].copy()


    # Preserve exact categorical values from training
    if "category" in df_eval.columns:
        cat_order = resolve_categorical_levels(bundle, "category", CATEGORY_ORDER)
        # Production-safe categorical representation:
        # Known categories map to their index; arbitrary tenant-defined categories (e.g. Power Banks,
        # Laptops, Gaming Gear) not seen in the baseline training vocabulary map safely to NaN/unknown.
        # The model's defined missing/unknown category handling handles them without crashing or inventing categories.
        df_eval["category"] = safe_categorical_series(df_eval["category"], cat_order)
    if "state_holiday" in df_eval.columns:
        state_order = resolve_categorical_levels(bundle, "state_holiday", STATE_HOLIDAY_ORDER)
        df_eval["state_holiday"] = safe_categorical_series(df_eval["state_holiday"], state_order)
    if "store_type" in df_eval.columns:
        store_order = resolve_categorical_levels(bundle, "store_type", STORE_TYPE_ORDER)
        df_eval["store_type"] = safe_categorical_series(df_eval["store_type"], store_order)
    if "assortment" in df_eval.columns:
        assort_order = resolve_categorical_levels(bundle, "assortment", ASSORTMENT_ORDER)
        df_eval["assortment"] = safe_categorical_series(df_eval["assortment"], assort_order)

    raw_pred = bundle["model"].predict(df_eval)
    pred_units = np.expm1(raw_pred).clip(min=0.0)

    # Closed-day guarantee
    if "is_open" in df_eval.columns:
        is_closed = df_eval["is_open"].to_numpy() == 0
        pred_units[is_closed] = 0.0

    return pred_units


def predict_demand_forecast(
    request: DemandForecastRequest,
    artifact_path: str | Path | None = None,
) -> tuple[list[DailyUnitForecastPoint], DemandForecastModelInfo]:
    """Execute multi-day demand forecasting inference for a SKU at a store.

    Evaluates the 11 trained features without requiring historical lags or rolling inputs.
    """
    p_str = str(artifact_path) if artifact_path else None
    bundle = load_demand_forecast_bundle(p_str)
    feature_columns: list[str] = bundle["feature_columns"]

    model_info = DemandForecastModelInfo(
        model_type=str(bundle.get("model_type", "lightgbm")),
        version=str(bundle.get("version", "1.0.0")),
        target=str(bundle.get("target", "units_sold")),
        feature_columns=list(feature_columns),
        feature_count=len(feature_columns),
    )

    start_date = request.forecast_date or (datetime.now(UTC).date() + timedelta(days=1))
    horizon = request.horizon

    contexts = request.daily_contexts or []
    date_to_ctx = {c.forecast_date: c for c in contexts if c.forecast_date is not None}

    feature_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for step in range(horizon):
        cur_date = start_date + timedelta(days=step)
        dow = cur_date.weekday()

        ctx = date_to_ctx.get(cur_date)
        if ctx is None and step < len(contexts):
            ctx = contexts[step]

        if ctx is not None:
            is_open = ctx.is_open
            is_promo = ctx.is_promo
            unit_price = ctx.unit_price or request.unit_price or request.base_unit_price
            state_holiday = str(ctx.state_holiday)
            school_holiday = ctx.school_holiday
        else:
            is_open = 0 if dow == 6 else 1
            is_promo = 0
            unit_price = request.unit_price or request.base_unit_price
            state_holiday = "0"
            school_holiday = 0

        feature_rows.append({
            "category": request.category,
            "unit_price": float(unit_price),
            "base_unit_price": float(request.base_unit_price),
            "is_open": int(is_open),
            "is_promo": int(is_promo),
            "promo2": int(request.promo2),
            "day_of_week": int(dow),
            "state_holiday": state_holiday,
            "school_holiday": int(school_holiday),
            "store_type": str(request.store_type),
            "assortment": str(request.assortment),
        })

        meta_rows.append({
            "product_id": request.product_id,
            "store_id": request.store_id,
            "forecast_date": cur_date,
            "horizon": horizon,
            "day_of_week": dow,
            "is_open": is_open,
            "is_promo": is_promo,
            "unit_price": unit_price,
        })

    df = pd.DataFrame(feature_rows)
    pred_units = predict_from_feature_dataframe(df, bundle=bundle)

    points: list[DailyUnitForecastPoint] = []
    for meta, pred in zip(meta_rows, pred_units, strict=True):
        final_units = 0.0 if meta["is_open"] == 0 else max(0.0, round(float(pred), 2))
        points.append(
            DailyUnitForecastPoint(
                product_id=meta["product_id"],
                store_id=meta["store_id"],
                forecast_date=meta["forecast_date"],
                predicted_units=final_units,
                horizon=meta["horizon"],
                day_of_week=meta["day_of_week"],
                is_open=meta["is_open"],
                is_promo=meta["is_promo"],
                unit_price=round(float(meta["unit_price"]), 2),
            )
        )

    return points, model_info


from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from technova_ai_service.modeling.artifacts import load_artifact


@lru_cache(maxsize=4)
def load_stock_bundle(path: str) -> dict[str, Any]:
    return load_artifact(Path(path))


def forecast_stock(
    *,
    artifact_path: Path,
    feature_values: dict[str, float],
    current_stock: float,
    lead_time_days: int,
    review_period_days: int,
    service_level: float,
) -> dict[str, float | str]:
    bundle = load_stock_bundle(str(artifact_path))
    columns: list[str] = bundle["feature_columns"]
    missing = [column for column in columns if column not in feature_values]
    if missing:
        raise ValueError(f"Missing stock features: {', '.join(missing)}")

    row = pd.DataFrame([{column: feature_values[column] for column in columns}])
    predicted_daily_demand = max(float(np.expm1(bundle["model"].predict(row)[0])), 0.0)
    uncertainty = max(feature_values["rolling_std_7"], predicted_daily_demand * 0.10)
    z_score = NormalDist().inv_cdf(service_level)
    safety_stock = z_score * uncertainty * np.sqrt(max(lead_time_days, 1))
    lead_time_demand = predicted_daily_demand * lead_time_days
    reorder_point = lead_time_demand + safety_stock
    target_stock = predicted_daily_demand * (lead_time_days + review_period_days) + safety_stock
    recommended_order = max(target_stock - current_stock, 0.0)
    stockout_risk = 1.0 if current_stock <= 0 else min(reorder_point / current_stock, 1.0)

    if current_stock <= reorder_point:
        status = "reorder_now"
    elif current_stock <= target_stock:
        status = "monitor"
    else:
        status = "healthy"

    return {
        "predicted_daily_demand": predicted_daily_demand,
        "lead_time_demand": float(lead_time_demand),
        "safety_stock": float(safety_stock),
        "reorder_point": float(reorder_point),
        "recommended_order_quantity": float(recommended_order),
        "stockout_risk": float(stockout_risk),
        "status": status,
    }

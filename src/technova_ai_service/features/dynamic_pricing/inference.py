from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from technova_ai_service.modeling.artifacts import load_artifact


@lru_cache(maxsize=4)
def load_pricing_bundle(path: str) -> dict[str, Any]:
    return load_artifact(Path(path))


def recommend_price(
    *,
    artifact_path: Path,
    feature_values: dict[str, float],
    current_price: float,
    unit_cost: float,
    min_margin_rate: float,
    max_change_rate: float,
    candidate_steps: int,
) -> dict[str, float | str]:
    bundle = load_pricing_bundle(str(artifact_path))
    columns: list[str] = bundle["feature_columns"]
    floor_price = unit_cost * (1.0 + min_margin_rate)
    lower = max(current_price * (1.0 - max_change_rate), floor_price)
    upper = max(current_price * (1.0 + max_change_rate), lower)
    candidates = np.linspace(lower, upper, candidate_steps)

    historical_reference_price = current_price / max(feature_values["price_index_28"], 0.01)
    rows: list[dict[str, float]] = []
    for candidate in candidates:
        row = dict(feature_values)
        row["unit_price"] = float(candidate)
        row["price_index_28"] = float(candidate / max(historical_reference_price, 0.01))
        rows.append({column: row[column] for column in columns})

    demand = np.expm1(bundle["model"].predict(pd.DataFrame(rows))).clip(min=0.0)
    profit = (candidates - unit_cost) * demand
    best_index = int(np.argmax(profit))
    recommended = float(candidates[best_index])
    expected_demand = float(demand[best_index])
    expected_profit = float(profit[best_index])
    change_rate = (recommended - current_price) / current_price

    if abs(change_rate) < 0.01:
        decision = "keep"
    elif change_rate > 0:
        decision = "increase"
    else:
        decision = "decrease"

    return {
        "recommended_price": recommended,
        "expected_daily_demand": expected_demand,
        "expected_daily_profit": expected_profit,
        "price_change_rate": float(change_rate),
        "decision": decision,
        "minimum_allowed_price": float(floor_price),
        "maximum_allowed_price": float(upper),
    }

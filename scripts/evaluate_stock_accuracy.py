import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from technova_ai_service.data.online_retail import (
    build_daily_product_panel,
    load_transactions,
)
from technova_ai_service.features.stock_intelligence.features import (
    STOCK_FEATURE_COLUMNS,
    build_stock_features,
)
from technova_ai_service.modeling.metrics import regression_metrics


def _direct_horizon_metrics(
    panel: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: int,
) -> dict[str, float | int]:
    actual: list[float] = []
    predicted: list[float] = []
    final_origin = end - pd.Timedelta(days=horizon - 1)
    for _, product in panel.groupby("product_id", observed=True):
        product = product.sort_values("date").reset_index(drop=True)
        quantities = product["quantity"].to_numpy(dtype=float)
        for index, forecast_date in enumerate(product["date"]):
            if forecast_date < start or forecast_date > final_origin or index < 28:
                continue
            history = quantities[index - 28 : index]
            actual.append(float(quantities[index : index + horizon].sum()))
            predicted.append(float(history.mean() * horizon))

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    metrics: dict[str, float | int] = regression_metrics(actual_array, predicted_array)
    metrics.update(
        {
            "origins": len(actual),
            "bias": float(np.mean(predicted_array - actual_array)),
            "actual_total": float(actual_array.sum()),
            "predicted_total": float(predicted_array.sum()),
        }
    )
    return metrics


def evaluate(input_path: Path, artifact_path: Path, top_products: int) -> dict[str, Any]:
    transactions = load_transactions(input_path)
    panel = build_daily_product_panel(transactions, top_products=top_products)
    dataset = build_stock_features(panel)
    bundle = joblib.load(artifact_path)
    metadata_path = artifact_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test_start = pd.Timestamp(metadata["train_end"]) + pd.Timedelta(days=1)
    test_end = pd.Timestamp(metadata["test_end"])
    test = dataset.loc[
        dataset["date"].between(test_start, test_end, inclusive="both")
    ].copy()
    actual = test["quantity"].to_numpy(dtype=float)
    predicted = np.expm1(bundle["model"].predict(test[STOCK_FEATURE_COLUMNS])).clip(min=0)
    baseline = test["demand_lag_7"].to_numpy(dtype=float).clip(min=0)

    report: dict[str, Any] = {
        "evaluation_window": {
            "start": str(test_start.date()),
            "end": str(test_end.date()),
            "products": top_products,
            "observations": len(test),
        },
        "next_day_model": regression_metrics(actual, predicted.astype(float)),
        "lag_7_baseline": regression_metrics(actual, baseline),
        "daily_refreshed_direct_horizons": {},
        "limitations": [
            "Development data is UCI Online Retail and not TechNova transaction history.",
            "Seasonal and yearly accuracy require multiple years of local history.",
            "Forecasts must remain advisory until TechNova backtests pass approval thresholds.",
        ],
    }
    for horizon in (7, 14, 30):
        report["daily_refreshed_direct_horizons"][str(horizon)] = (
            _direct_horizon_metrics(
                panel,
                start=test_start,
                end=test_end,
                horizon=horizon,
            )
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stock forecasting accuracy.")
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/online_retail.csv")
    )
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/stock_intelligence/model.joblib")
    )
    parser.add_argument("--top-products", type=int, default=200)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = evaluate(arguments.input, arguments.artifact, arguments.top_products)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

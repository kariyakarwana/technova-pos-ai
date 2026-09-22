from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from technova_ai_service.data.online_retail import (
    build_daily_product_panel,
    chronological_split,
    load_transactions,
)
from technova_ai_service.features.stock_intelligence.features import (
    STOCK_FEATURE_COLUMNS,
    build_stock_features,
)
from technova_ai_service.modeling.artifacts import save_artifact
from technova_ai_service.modeling.metrics import regression_metrics


def train_stock_model(
    input_path: Path,
    artifact_directory: Path,
    *,
    top_products: int = 200,
) -> dict[str, Any]:
    transactions = load_transactions(input_path)
    panel = build_daily_product_panel(transactions, top_products=top_products)
    dataset = build_stock_features(panel)
    train, validation, test = chronological_split(dataset)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    fit_frame = pd.concat([train, validation], ignore_index=True)
    model.fit(fit_frame[STOCK_FEATURE_COLUMNS], np.log1p(fit_frame["quantity"]))
    predictions = np.expm1(model.predict(test[STOCK_FEATURE_COLUMNS])).clip(min=0.0)
    metrics = regression_metrics(test["quantity"].to_numpy(dtype=float), predictions.astype(float))
    baseline = test["demand_lag_7"].to_numpy(dtype=float).clip(min=0.0)
    baseline_metrics = regression_metrics(test["quantity"].to_numpy(dtype=float), baseline)
    metrics.update({f"seasonal_baseline_{key}": value for key, value in baseline_metrics.items()})

    model_path = save_artifact(
        artifact_directory,
        bundle={
            "model": model,
            "feature_columns": STOCK_FEATURE_COLUMNS,
            "residual_std": float(np.std(predictions - test["quantity"].to_numpy(dtype=float))),
        },
        metrics=metrics,
        metadata={
            "model_type": "HistGradientBoostingRegressor",
            "target": "next-day product demand",
            "top_products": top_products,
            "training_rows": len(fit_frame),
            "test_rows": len(test),
            "train_start": str(fit_frame["date"].min()),
            "train_end": str(fit_frame["date"].max()),
            "test_end": str(test["date"].max()),
        },
    )
    return {"artifact": str(model_path), "metrics": metrics}

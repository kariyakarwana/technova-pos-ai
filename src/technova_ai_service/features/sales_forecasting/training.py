from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from technova_ai_service.modeling.artifacts import save_artifact
from technova_ai_service.modeling.metrics import calculate_forecasting_metrics

logger = logging.getLogger(__name__)

TECHNOVA_SALES_FEATURE_COLUMNS = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "revenue_lag_1",
    "revenue_lag_7",
    "revenue_lag_14",
    "revenue_lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "weekly_momentum_ratio",
    "branch_age_days",
    "branch_historical_avg_sales",
    "is_operating_day",
    "active_discount_count",
]

FORBIDDEN_LEAKAGE_COLUMNS = {
    "Store",
    "store_id",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo",
    "Customers",
    "StateHoliday",
    "SchoolHoliday",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
    "Open",
}


def validate_training_dataset(df: pd.DataFrame) -> None:
    """Validate that the dataset satisfies all TechNova-native ML training constraints."""
    # 1. Target check
    if "daily_revenue" not in df.columns:
        raise ValueError("Missing required target column 'daily_revenue'.")
    if df["daily_revenue"].isna().any():
        raise ValueError("Target 'daily_revenue' contains NaN values.")
    if (df["daily_revenue"] < 0).any():
        raise ValueError("Target 'daily_revenue' contains negative values.")

    # 2. TechNova metadata check
    for col in ("branch_id", "date"):
        if col not in df.columns:
            raise ValueError(f"Missing required identifier column '{col}'.")

    # 3. Forbidden leakage / Rossmann feature check
    present_forbidden = FORBIDDEN_LEAKAGE_COLUMNS.intersection(df.columns)
    if present_forbidden:
        raise ValueError(
            f"Forbidden legacy/Rossmann features detected in training dataset: {sorted(present_forbidden)}"
        )

    # 4. Required feature columns check
    missing_features = [
        col for col in TECHNOVA_SALES_FEATURE_COLUMNS if col not in df.columns
    ]
    if missing_features:
        raise ValueError(
            f"Training dataset is missing required features: {missing_features}"
        )

    # 5. Null values in feature set check
    null_counts = df[TECHNOVA_SALES_FEATURE_COLUMNS].isna().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        raise ValueError(
            f"Features contain unexpected NaN values: {null_cols.to_dict()}"
        )


def train_sales_forecasting_model(
    input_path: Path,
    artifact_directory: Path,
    *,
    test_days: int = 30,
    val_days: int = 30,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the TechNova daily revenue XGBoost forecasting model.

    Guarantees:
    - Pure TechNova-native Branch x Date daily revenue dataset.
    - Zero Rossmann / forbidden feature leakage.
    - Target is strictly daily_revenue.
    - Chronological, leakage-safe train/validation/holdout split.
    - CPU-compatible XGBoost Regressor with histogram tree method.
    - Metrics calculated on untouched holdout: WAPE, MASE, MAE, RMSE.
    - Model artifact and feature metadata persisted at artifact_directory.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {input_path}")

    # 1. Load dataset
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["date", "branch_id"]).reset_index(drop=True)

    # 2. Validate dataset
    validate_training_dataset(df)

    # 3. Chronological Time-Series Split
    unique_dates = sorted(df["date"].unique())
    total_dates = len(unique_dates)

    if total_dates < (test_days + val_days + 30):
        raise ValueError(
            f"Insufficient history: {total_dates} dates found, need at least {test_days + val_days + 30}."
        )

    test_dates = set(unique_dates[-test_days:])
    val_dates = set(unique_dates[-test_days - val_days : -test_days])
    train_dates = set(unique_dates[: -test_days - val_days])

    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    # Verify strict non-overlapping temporal ordering
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()

    X_train = train_df[TECHNOVA_SALES_FEATURE_COLUMNS]
    y_train = train_df["daily_revenue"].to_numpy(dtype=float)

    X_val = val_df[TECHNOVA_SALES_FEATURE_COLUMNS]
    y_val = val_df["daily_revenue"].to_numpy(dtype=float)

    X_test = test_df[TECHNOVA_SALES_FEATURE_COLUMNS]
    y_test = test_df["daily_revenue"].to_numpy(dtype=float)

    # 4. Model hyperparameters (CPU hist method)
    xgb_params: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "max_depth": 6,
        "min_child_weight": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": 42,
        "eval_metric": "mae",
    }
    if params:
        xgb_params.update(params)

    # 5. Train with early stopping on validation split
    early_stop_rounds = 30
    eval_model = xgb.XGBRegressor(
        **xgb_params, early_stopping_rounds=early_stop_rounds
    )
    eval_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iteration = (
        int(eval_model.best_iteration)
        if eval_model.best_iteration is not None
        else int(xgb_params["n_estimators"])
    )

    # 6. Evaluate on holdout test set
    test_preds = np.clip(eval_model.predict(X_test), a_min=0.0, a_max=None)
    metrics = calculate_forecasting_metrics(
        y_true=y_test,
        y_pred=test_preds,
        y_train_seasonal=pd.concat([train_df["daily_revenue"], val_df["daily_revenue"]]),
        seasonality=7,
    )

    # 7. Fit final production model on train + validation up to optimal iteration
    final_params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = max(best_iteration + 1, 50)
    final_model = xgb.XGBRegressor(**final_params)

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])
    final_model.fit(X_train_val, y_train_val, verbose=False)

    # 8. Persist artifact bundle and metadata
    metadata: dict[str, Any] = {
        "model_name": "technova_daily_revenue_xgboost",
        "model_type": "XGBRegressor",
        "framework": "xgboost",
        "target": "daily_revenue",
        "feature_columns": TECHNOVA_SALES_FEATURE_COLUMNS,
        "feature_count": len(TECHNOVA_SALES_FEATURE_COLUMNS),
        "total_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_start_date": train_df["date"].min().strftime("%Y-%m-%d"),
        "train_end_date": train_df["date"].max().strftime("%Y-%m-%d"),
        "val_start_date": val_df["date"].min().strftime("%Y-%m-%d"),
        "val_end_date": val_df["date"].max().strftime("%Y-%m-%d"),
        "test_start_date": test_df["date"].min().strftime("%Y-%m-%d"),
        "test_end_date": test_df["date"].max().strftime("%Y-%m-%d"),
        "best_iteration": int(best_iteration),
        "hyperparameters": final_params,
    }

    bundle: dict[str, Any] = {
        "model": final_model,
        "feature_columns": TECHNOVA_SALES_FEATURE_COLUMNS,
        "metadata": metadata,
    }

    model_path = save_artifact(
        directory=artifact_directory,
        bundle=bundle,
        metrics=metrics,
        metadata=metadata,
    )

    return {
        "artifact_path": str(model_path),
        "metrics": metrics,
        "metadata": metadata,
        "training_row_count": len(train_df),
        "validation_row_count": len(val_df),
        "test_row_count": len(test_df),
        "feature_count": len(TECHNOVA_SALES_FEATURE_COLUMNS),
        "feature_names": TECHNOVA_SALES_FEATURE_COLUMNS,
        "train_date_range": (metadata["train_start_date"], metadata["train_end_date"]),
        "val_date_range": (metadata["val_start_date"], metadata["val_end_date"]),
        "test_date_range": (metadata["test_start_date"], metadata["test_end_date"]),
    }


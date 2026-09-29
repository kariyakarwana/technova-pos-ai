from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb

from technova_ai_service.modeling.metrics import regression_metrics


ModelType = Literal["lightgbm", "xgboost", "hist_gradient_boosting"]

# Default feature columns used when evaluating datasets with legacy schema (excluding identifiers and target)
MODEL_FEATURE_COLUMNS: list[str] = [
    "is_open",
    "is_closed",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_3",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_21",
    "demand_lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_max_7",
    "rolling_max_14",
    "rolling_max_28",
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "state_holiday",
    "school_holiday",
    "promo",
    "promo2",
    "has_active_promo2",
    "store_type",
    "assortment",
    "competition_distance",
    "has_competition",
    "competition_open_months",
]

HORIZON_DAYS: tuple[int, ...] = (7, 14, 30)


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Partition the dataset into strict chronological Train, Validation, and Test sets."""
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)
    n_train = int(n_dates * train_ratio)
    n_val = int(n_dates * val_ratio)

    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train : n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val :])

    train_df = df[df["date"].isin(train_dates)]
    val_df = df[df["date"].isin(val_dates)]
    test_df = df[df["date"].isin(test_dates)]

    split_info = {
        "total_dates": n_dates,
        "train_dates_count": len(train_dates),
        "val_dates_count": len(val_dates),
        "test_dates_count": len(test_dates),
        "train_range": [min(train_dates), max(train_dates)],
        "val_range": [min(val_dates), max(val_dates)],
        "test_range": [min(test_dates), max(test_dates)],
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }

    return train_df, val_df, test_df, split_info


def calculate_metrics_with_fva(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute WAPE, MAE, RMSE, and FVA against baseline."""
    reg = regression_metrics(actual, predicted)
    res = {
        "wape": round(reg["wape"], 4),
        "mae": round(reg["mae"], 2),
        "rmse": round(reg["rmse"], 2),
    }
    if baseline is not None:
        base_reg = regression_metrics(actual, baseline)
        if base_reg["mae"] > 0:
            fva = 1.0 - (reg["mae"] / base_reg["mae"])
            res["fva"] = round(float(fva), 4)
            res["baseline_mae"] = round(base_reg["mae"], 2)
            res["baseline_wape"] = round(base_reg["wape"], 4)
        else:
            res["fva"] = 0.0
    return res


def create_model_instance(model_type: ModelType, random_state: int = 42) -> Any:
    """Instantiate a regressor for demand forecasting."""
    if model_type == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=8,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    elif model_type == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def predict_demand(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict non-negative demand from features, enforcing closed-day zero demand."""
    raw_log = model.predict(X)
    pred = np.expm1(raw_log).clip(min=0.0)
    # Closed days cannot produce sales
    if "is_open" in X.columns:
        pred[X["is_open"].to_numpy() == 0] = 0.0
    return pred


def resolve_target_and_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Determine target column and feature columns dynamically based on dataset schema."""
    if "units_sold" in df.columns:
        target_col = "units_sold"
        if feature_columns is not None:
            features = feature_columns
        else:
            # Exclude non-feature identifiers, target, and future leakage columns
            excluded = {
                "date",
                "store_id",
                "product_id",
                "product_name",
                "customers",  # future footfall is unknown at forecasting horizon
                "units_sold",
                "log_units_sold",
                "demand",
                "log_demand",
                "organization_id",
            }
            features = [c for c in df.columns if c not in excluded]
    else:
        target_col = "demand"
        features = feature_columns or MODEL_FEATURE_COLUMNS

    for f in features:
        if f not in df.columns:
            raise ValueError(f"Feature column '{f}' not found in dataset.")

    return target_col, features


def evaluate_horizon_subsets(
    model: Any,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    horizons: tuple[int, ...] = HORIZON_DAYS,
    target_col: str = "demand",
) -> dict[str, dict[str, float]]:
    """Evaluate model performance across horizons (first 7, 14, 30 days of test period)."""
    test_dates = sorted(test_df["date"].unique())
    results: dict[str, dict[str, float]] = {}

    for h in horizons:
        h_dates = set(test_dates[:h])
        h_df = test_df[test_df["date"].isin(h_dates)]

        X_h = h_df[feature_cols]
        y_h = h_df[target_col].to_numpy(dtype=float)
        # Baseline: demand_lag_7 with closed-day masking if available
        if "demand_lag_7" in h_df.columns:
            baseline_h = h_df["demand_lag_7"].to_numpy(dtype=float).clip(min=0.0)
            baseline_h[h_df["is_open"].to_numpy() == 0] = 0.0
        elif "units_sold_lag_7" in h_df.columns:
            baseline_h = h_df["units_sold_lag_7"].to_numpy(dtype=float).clip(min=0.0)
            baseline_h[h_df["is_open"].to_numpy() == 0] = 0.0
        else:
            baseline_h = None

        pred_h = predict_demand(model, X_h)
        metrics_h = calculate_metrics_with_fva(y_h, pred_h, baseline=baseline_h)
        results[f"horizon_{h}d"] = metrics_h

    # Overall test set metrics
    X_all = test_df[feature_cols]
    y_all = test_df[target_col].to_numpy(dtype=float)
    if "demand_lag_7" in test_df.columns:
        baseline_all = test_df["demand_lag_7"].to_numpy(dtype=float).clip(min=0.0)
        baseline_all[test_df["is_open"].to_numpy() == 0] = 0.0
    elif "units_sold_lag_7" in test_df.columns:
        baseline_all = test_df["units_sold_lag_7"].to_numpy(dtype=float).clip(min=0.0)
        baseline_all[test_df["is_open"].to_numpy() == 0] = 0.0
    else:
        baseline_all = None

    pred_all = predict_demand(model, X_all)
    results["overall_test"] = calculate_metrics_with_fva(y_all, pred_all, baseline=baseline_all)

    return results


def train_and_benchmark(
    dataset_path: str | Path,
    output_dir: str | Path = "artifacts/demand_forecasting",
    *,
    feature_columns: list[str] | None = None,
    candidate_models: tuple[ModelType, ...] = ("lightgbm", "xgboost", "hist_gradient_boosting"),
) -> dict[str, Any]:
    """Train and benchmark candidate models, select best model, and persist production artifact."""
    d_path = Path(dataset_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed dataset from {d_path}...")
    import pyarrow.parquet as pq

    schema_names = pq.ParquetFile(d_path).schema.names
    dummy_df = pd.DataFrame(columns=schema_names)
    target_col, features = resolve_target_and_features(dummy_df, feature_columns)
    print(f"Dataset Target: '{target_col}' | Feature Count: {len(features)}")

    load_cols = list(dict.fromkeys(["date", "is_open", target_col] + features))
    df = pd.read_parquet(d_path, columns=load_cols)

    # Ensure categorical string columns are cast to category dtype for model compatibility
    for col in features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")

    # Calculate dataset hash for provenance
    with open(d_path, "rb") as f:
        dataset_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()[:16]

    print("Splitting dataset chronologically (70% Train, 15% Val, 15% Test)...")
    train_df, val_df, test_df, split_info = split_chronological(df)
    del df

    # Train only on open operating days (standard practice to avoid regression distortion)
    train_open = train_df[train_df["is_open"] == 1]
    X_train = train_open[features]
    if "log_demand" in train_open.columns:
        y_train = train_open["log_demand"]
    elif "log_units_sold" in train_open.columns:
        y_train = train_open["log_units_sold"]
    else:
        y_train = np.log1p(train_open[target_col].to_numpy(dtype=float))

    X_val = val_df[features]
    y_val = val_df[target_col].to_numpy(dtype=float)
    if "demand_lag_7" in val_df.columns:
        baseline_val = val_df["demand_lag_7"].to_numpy(dtype=float).clip(min=0.0)
        baseline_val[val_df["is_open"].to_numpy() == 0] = 0.0
    elif "units_sold_lag_7" in val_df.columns:
        baseline_val = val_df["units_sold_lag_7"].to_numpy(dtype=float).clip(min=0.0)
        baseline_val[val_df["is_open"].to_numpy() == 0] = 0.0
    else:
        baseline_val = None

    comparison_results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    print(f"\nBenchmarking {len(candidate_models)} candidate models using VALIDATION metrics only...")

    for model_name in candidate_models:
        print(f"\n--- Training {model_name.upper()} ---")
        model = create_model_instance(model_name)

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)
        trained_models[model_name] = model

        # Validation set performance (strictly used for candidate comparison & selection)
        pred_val = predict_demand(model, X_val)
        val_metrics = calculate_metrics_with_fva(y_val, pred_val, baseline=baseline_val)

        model_summary = {
            "model": model_name,
            "training_time_seconds": train_time,
            "validation_metrics": val_metrics,
        }
        comparison_results.append(model_summary)
        print(f"  Training Time: {train_time}s")
        print(f"  Validation WAPE: {val_metrics['wape']} | MAE: {val_metrics['mae']} | RMSE: {val_metrics['rmse']} | FVA: {val_metrics.get('fva', 0.0):.2%}")

    # 1. & 2. Select the production model using VALIDATION WAPE ONLY (test set remains untouched)
    best_candidate = min(
        comparison_results,
        key=lambda r: r["validation_metrics"]["wape"],
    )
    selected_model_name = best_candidate["model"]
    selected_model = trained_models[selected_model_name]
    print(f"\nSELECTED MODEL: {selected_model_name.upper()} (Selected strictly by lowest Validation WAPE: {best_candidate['validation_metrics']['wape']})")

    # 3. After selecting the model, evaluate that single selected model ONCE on the holdout test set
    print(f"\nEvaluating selected model {selected_model_name.upper()} once on holdout test set (7d, 14d, 30d, overall)...")
    final_test_metrics = evaluate_horizon_subsets(
        selected_model, test_df, features, target_col=target_col
    )
    for h_name, h_met in final_test_metrics.items():
        print(f"  Holdout {h_name}: WAPE={h_met['wape']} | MAE={h_met['mae']} | RMSE={h_met['rmse']} | FVA={h_met.get('fva', 0.0):.2%}")

    # Save model artifact
    model_artifact_path = out_dir / "model.joblib"
    categorical_categories: dict[str, list[str]] = {}
    for col in features:
        if hasattr(X_train[col], "cat"):
            categorical_categories[col] = list(X_train[col].cat.categories)

    bundle = {
        "model": selected_model,
        "feature_columns": features,
        "categorical_categories": categorical_categories,
        "model_type": selected_model_name,
        "target": target_col,
        "version": "1.0.0",
    }
    print(f"Saving model artifact to {model_artifact_path}...")
    joblib.dump(bundle, model_artifact_path)

    # Save metadata JSON with explicit separation of validation selection and holdout test metrics
    metadata_path = out_dir / "model_metadata.json"
    metadata = {
        "model_type": selected_model_name,
        "selected_at_epoch": int(time.time()),
        "dataset_hash": dataset_hash,
        "dataset_path": str(d_path),
        "split_boundaries": split_info,
        "feature_columns": features,
        "feature_count": len(features),
        "horizons_evaluated": list(HORIZON_DAYS),
        "selection_criteria": "validation_wape_only",
        "validation_model_selection": comparison_results,
        "selected_model_validation_metrics": best_candidate["validation_metrics"],
        "final_holdout_test_metrics": final_test_metrics,
        "artifact_file": "model.joblib",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {metadata_path}...")

    return {
        "selected_model": selected_model_name,
        "artifact_path": str(model_artifact_path),
        "metadata_path": str(metadata_path),
        "validation_model_selection": comparison_results,
        "comparison": comparison_results,
        "selected_validation_metrics": best_candidate["validation_metrics"],
        "final_holdout_test_metrics": final_test_metrics,
        "split_info": split_info,
    }

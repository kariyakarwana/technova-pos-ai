from datetime import date, timedelta
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from technova_ai_service.features.demand_forecasting.training import (
    MODEL_FEATURE_COLUMNS,
    create_model_instance,
    predict_demand,
    split_chronological,
    train_and_benchmark,
)


def _create_mini_training_dataset(days: int = 50) -> pd.DataFrame:
    """Create a self-contained mini dataset with all required feature columns for fast unit tests."""
    start = date(2015, 1, 1)
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

    rows = []
    for store_id in (1, 2):
        for i, dt in enumerate(dates):
            is_open = 0 if i % 7 == 6 else 1
            sales = 0.0 if is_open == 0 else float(store_id * 500 + i * 15)

            row = {
                "organization_id": "test_org",
                "store_id": str(store_id),
                "date": dt,
                "demand": sales,
                "log_demand": np.log1p(sales),
                "is_open": is_open,
                "is_closed": 1 - is_open,
                "day_of_week": i % 7,
                "day_of_month": (i % 28) + 1,
                "month": 1,
                "year": 2015,
                "is_weekend": 1 if (i % 7) >= 5 else 0,
                "is_month_start": 1 if i == 0 else 0,
                "is_month_end": 0,
                "promo": 1 if i % 4 == 0 else 0,
                "promo2": 1,
                "state_holiday": 0,
                "school_holiday": 0,
                "has_active_promo2": 1,
                "store_type": 0,
                "assortment": 0,
                "competition_distance": 500.0,
                "has_competition": 1,
                "competition_open_months": 24,
            }

            # Add lags and rolling stats
            for lag in (1, 2, 3, 7, 14, 21, 28):
                row[f"demand_lag_{lag}"] = max(sales - lag * 10, 0.0)

            for w in (7, 14, 28):
                row[f"rolling_mean_{w}"] = sales
                row[f"rolling_std_{w}"] = 5.0
                row[f"rolling_max_{w}"] = sales + 10.0

            rows.append(row)

    return pd.DataFrame(rows)


def test_chronological_split_boundaries() -> None:
    df = _create_mini_training_dataset(days=40)
    train_df, val_df, test_df, split_info = split_chronological(
        df, train_ratio=0.70, val_ratio=0.15
    )

    # 40 unique dates: 70% = 28 dates, 15% = 6 dates, 15% = 6 dates
    assert split_info["total_dates"] == 40
    assert split_info["train_dates_count"] == 28
    assert split_info["val_dates_count"] == 6
    assert split_info["test_dates_count"] == 6

    # Verify strict chronological separation: max(train) < min(val) < min(test)
    assert max(train_df["date"]) < min(val_df["date"])
    assert max(val_df["date"]) < min(test_df["date"])

    # No row overlap
    all_rows = len(train_df) + len(val_df) + len(test_df)
    assert all_rows == len(df)


def test_no_target_leakage_in_features() -> None:
    """Verify target columns ('demand', 'log_demand', 'Sales') are strictly excluded from features."""
    assert "demand" not in MODEL_FEATURE_COLUMNS
    assert "log_demand" not in MODEL_FEATURE_COLUMNS
    assert "Sales" not in MODEL_FEATURE_COLUMNS
    assert "sales" not in MODEL_FEATURE_COLUMNS


def test_prediction_shape_and_non_negative() -> None:
    df = _create_mini_training_dataset(days=35)
    features = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]

    X = df[features]
    y = df["log_demand"]

    model = create_model_instance("lightgbm", random_state=42)
    model.fit(X, y)

    predictions = predict_demand(model, X)

    # Output shape matches input rows
    assert len(predictions) == len(df)
    # Non-negative output guarantee
    assert (predictions >= 0.0).all()
    # Closed-day zero demand guarantee
    closed_mask = df["is_open"].to_numpy() == 0
    assert (predictions[closed_mask] == 0.0).all()


def test_deterministic_training() -> None:
    df = _create_mini_training_dataset(days=35)
    features = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]

    X = df[features]
    y = df["log_demand"]

    m1 = create_model_instance("lightgbm", random_state=123)
    m1.fit(X, y)
    p1 = predict_demand(m1, X)

    m2 = create_model_instance("lightgbm", random_state=123)
    m2.fit(X, y)
    p2 = predict_demand(m2, X)

    np.testing.assert_allclose(p1, p2, rtol=1e-5)


def test_feature_column_consistency() -> None:
    model = create_model_instance("lightgbm")
    df = _create_mini_training_dataset(days=20)
    features = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]

    X = df[features]
    y = df["log_demand"]
    model.fit(X, y)

    # Missing a column must raise an error during inference
    X_missing = X.drop(columns=[features[0]])
    with pytest.raises(Exception):
        model.predict(X_missing)


def test_train_benchmark_and_artifact_loading(tmp_path: Path) -> None:
    df = _create_mini_training_dataset(days=40)
    parquet_path = tmp_path / "test_demand.parquet"
    df.to_parquet(parquet_path, index=False)

    artifact_dir = tmp_path / "artifacts"

    results = train_and_benchmark(
        dataset_path=parquet_path,
        output_dir=artifact_dir,
        candidate_models=("lightgbm", "hist_gradient_boosting"),
    )

    # Verify return dictionary structure
    assert results["selected_model"] in ("lightgbm", "hist_gradient_boosting")
    assert len(results["validation_model_selection"]) == 2

    # Verify model selection was chosen strictly by minimum validation WAPE
    min_val_cand = min(
        results["validation_model_selection"],
        key=lambda c: c["validation_metrics"]["wape"],
    )
    assert results["selected_model"] == min_val_cand["model"]

    # Verify candidates in validation_model_selection do NOT contain test metrics
    for candidate in results["validation_model_selection"]:
        assert "validation_metrics" in candidate
        assert "test_horizons" not in candidate

    # Verify holdout test metrics exist only for the final selected model
    assert "final_holdout_test_metrics" in results
    assert "horizon_7d" in results["final_holdout_test_metrics"]
    assert "horizon_14d" in results["final_holdout_test_metrics"]
    assert "horizon_30d" in results["final_holdout_test_metrics"]
    assert "overall_test" in results["final_holdout_test_metrics"]

    # Verify model.joblib artifact loading
    model_file = Path(results["artifact_path"])
    assert model_file.exists()

    bundle = joblib.load(model_file)
    assert "model" in bundle
    assert "feature_columns" in bundle
    assert "model_type" in bundle
    assert bundle["target"] == "demand"

    # Verify model_metadata.json
    metadata_file = Path(results["metadata_path"])
    assert metadata_file.exists()

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert "model_type" in metadata
    assert "dataset_hash" in metadata
    assert "split_boundaries" in metadata
    assert "horizons_evaluated" in metadata
    assert metadata["horizons_evaluated"] == [7, 14, 30]
    assert metadata["selection_criteria"] == "validation_wape_only"
    assert "validation_model_selection" in metadata
    assert "final_holdout_test_metrics" in metadata


def test_model_selection_strict_validation_isolation(tmp_path: Path) -> None:
    """Explicitly verify that candidate comparison uses validation metrics only and test set is never evaluated during candidate selection."""
    df = _create_mini_training_dataset(days=40)
    parquet_path = tmp_path / "test_demand_val.parquet"
    df.to_parquet(parquet_path, index=False)

    artifact_dir = tmp_path / "artifacts_val"

    results = train_and_benchmark(
        dataset_path=parquet_path,
        output_dir=artifact_dir,
        candidate_models=("lightgbm", "hist_gradient_boosting"),
    )

    metadata_file = Path(results["metadata_path"])
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    # Check that candidate comparison list has no holdout test metrics
    for comp in metadata["validation_model_selection"]:
        assert "validation_metrics" in comp
        assert "wape" in comp["validation_metrics"]
        assert "test_horizons" not in comp
        assert "test_metrics" not in comp

    # Selected model must match lowest validation WAPE candidate
    candidates = metadata["validation_model_selection"]
    lowest_wape_model = min(candidates, key=lambda x: x["validation_metrics"]["wape"])["model"]
    assert metadata["model_type"] == lowest_wape_model

    # Final holdout test metrics are stored once for the selected model
    assert "final_holdout_test_metrics" in metadata
    test_metrics = metadata["final_holdout_test_metrics"]
    for horizon_key in ("horizon_7d", "horizon_14d", "horizon_30d", "overall_test"):
        assert horizon_key in test_metrics
        assert "wape" in test_metrics[horizon_key]
        assert "fva" in test_metrics[horizon_key]


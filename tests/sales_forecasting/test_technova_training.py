"""Unit tests for Phase 5.3.4 TechNova Sales Forecasting Model Training and Inference."""

import datetime
from pathlib import Path

import pandas as pd
import pytest

from technova_ai_service.features.sales_forecasting.inference import (
    predict_sales_forecast,
)
from technova_ai_service.features.sales_forecasting.schemas import (
    DailyRevenuePoint,
    FutureCalendarPoint,
)
from technova_ai_service.features.sales_forecasting.training import (
    TECHNOVA_SALES_FEATURE_COLUMNS,
    train_sales_forecasting_model,
    validate_training_dataset,
)
from technova_ai_service.modeling.artifacts import load_artifact


@pytest.fixture
def synthetic_technova_dataset(tmp_path: Path) -> Path:
    """Generate a minimal valid TechNova-native dataset for rapid unit testing."""
    records: list[dict[str, object]] = []
    base_date = datetime.date(2025, 1, 1)

    for i in range(120):  # 120 days
        cur_date = base_date + datetime.timedelta(days=i)
        dow = cur_date.weekday()
        for branch_id in ["br_colombo_01", "br_kandy_02"]:
            rec = {
                "organization_id": "org_technova_test",
                "branch_id": branch_id,
                "branch_code": "COL-01" if branch_id == "br_colombo_01" else "KAN-02",
                "branch_name": "Branch " + branch_id,
                "date": cur_date.isoformat(),
                "daily_revenue": float(50000.0 + 1000 * dow + 50 * (i % 7)),
                "day_of_week": dow,
                "day_of_month": cur_date.day,
                "month": cur_date.month,
                "week_of_year": int(cur_date.isocalendar()[1]),
                "is_weekend": 1 if dow >= 5 else 0,
                "is_month_start": 1 if cur_date.day == 1 else 0,
                "is_month_end": 0,
                "day_of_week_sin": 0.5,
                "day_of_week_cos": 0.5,
                "month_sin": 0.1,
                "month_cos": 0.9,
                "revenue_lag_1": 50000.0,
                "revenue_lag_7": 49000.0,
                "revenue_lag_14": 48000.0,
                "revenue_lag_28": 47000.0,
                "rolling_mean_7": 49500.0,
                "rolling_mean_14": 49000.0,
                "rolling_mean_28": 48500.0,
                "rolling_std_7": 500.0,
                "rolling_std_28": 600.0,
                "weekly_momentum_ratio": 1.01,
                "branch_age_days": i + 100,
                "branch_historical_avg_sales": 50000.0,
                "is_operating_day": 1,
                "active_discount_count": 1,
            }
            records.append(rec)

    df = pd.DataFrame(records)
    parquet_path = tmp_path / "synthetic_technova_sales.parquet"
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def test_validate_training_dataset_success(synthetic_technova_dataset: Path) -> None:
    df = pd.read_parquet(synthetic_technova_dataset)
    validate_training_dataset(df)


def test_validate_training_dataset_rejects_missing_target(synthetic_technova_dataset: Path) -> None:
    df = pd.read_parquet(synthetic_technova_dataset).drop(columns=["daily_revenue"])
    with pytest.raises(ValueError, match="daily_revenue"):
        validate_training_dataset(df)


def test_validate_training_dataset_rejects_negative_revenue(synthetic_technova_dataset: Path) -> None:
    df = pd.read_parquet(synthetic_technova_dataset)
    df.loc[0, "daily_revenue"] = -100.0
    with pytest.raises(ValueError, match="negative"):
        validate_training_dataset(df)


@pytest.mark.parametrize("forbidden_col", ["Store", "Customers", "Promo", "StoreType", "Assortment"])
def test_validate_training_dataset_rejects_rossmann_columns(
    synthetic_technova_dataset: Path, forbidden_col: str
) -> None:
    df = pd.read_parquet(synthetic_technova_dataset)
    df[forbidden_col] = 1
    with pytest.raises(ValueError, match="Forbidden legacy/Rossmann features"):
        validate_training_dataset(df)


def test_train_sales_forecasting_model_end_to_end(
    synthetic_technova_dataset: Path, tmp_path: Path
) -> None:
    artifact_dir = tmp_path / "artifacts"
    results = train_sales_forecasting_model(
        input_path=synthetic_technova_dataset,
        artifact_directory=artifact_dir,
        test_days=10,
        val_days=10,
        params={"n_estimators": 20, "max_depth": 3},
    )

    assert results["training_row_count"] > 0
    assert results["validation_row_count"] > 0
    assert results["test_row_count"] > 0
    assert results["feature_count"] == 25
    assert len(results["feature_names"]) == 25

    for metric in ("wape", "mase", "mae", "rmse"):
        assert metric in results["metrics"]
        assert results["metrics"][metric] is not None
        assert results["metrics"][metric] >= 0.0

    # Verify model artifact exists and can be loaded
    model_path = Path(results["artifact_path"])
    assert model_path.exists()
    bundle = load_artifact(model_path)
    assert "model" in bundle
    assert "feature_columns" in bundle
    assert bundle["feature_columns"] == TECHNOVA_SALES_FEATURE_COLUMNS


def test_inference_closed_day_yields_zero_revenue(
    synthetic_technova_dataset: Path, tmp_path: Path
) -> None:
    artifact_dir = tmp_path / "artifacts"
    results = train_sales_forecasting_model(
        input_path=synthetic_technova_dataset,
        artifact_directory=artifact_dir,
        test_days=10,
        val_days=10,
        params={"n_estimators": 10, "max_depth": 2},
    )
    model_path = Path(results["artifact_path"])

    recent_rev = [
        DailyRevenuePoint(
            date=datetime.date(2025, 5, 1) + datetime.timedelta(days=i),
            revenue=50000.0,
            is_operating_day=1,
        )
        for i in range(30)
    ]
    future_cal = [
        FutureCalendarPoint(date=datetime.date(2025, 6, 1), is_operating_day=1, active_discounts=0),
        FutureCalendarPoint(date=datetime.date(2025, 6, 2), is_operating_day=0, active_discounts=0),  # CLOSED
        FutureCalendarPoint(date=datetime.date(2025, 6, 3), is_operating_day=1, active_discounts=1),
    ]

    preds = predict_sales_forecast(
        artifact_path=model_path,
        organization_id="org_test",
        branch_id="br_test",
        forecast_horizon=3,
        recent_daily_revenue=recent_rev,
        future_calendar=future_cal,
    )

    assert len(preds) == 3
    assert preds[0]["predicted_revenue"] > 0.0
    assert preds[1]["predicted_revenue"] == 0.0  # Closed day must be strictly 0.0
    assert preds[2]["predicted_revenue"] > 0.0

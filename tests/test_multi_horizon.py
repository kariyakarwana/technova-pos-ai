import pytest
from pydantic import ValidationError

from technova_ai_service.features.stock_intelligence.multi_horizon import (
    _calibrate_daily_predictions,
    _data_readiness,
    _direct_horizon_forecasts,
)
from technova_ai_service.features.stock_intelligence.schemas import (
    MultiHorizonForecastRequest,
)


def test_direct_horizons_and_daily_calibration_are_consistent() -> None:
    history = [float(value) for value in range(1, 29)]
    direct = _direct_horizon_forecasts(history, 0.95)
    calibrated = _calibrate_daily_predictions([1.0] * 60, direct)
    totals = {int(item["horizon_days"]): item["predicted_demand"] for item in direct}

    assert sum(calibrated[:7]) == pytest.approx(totals[7])
    assert sum(calibrated[:14]) == pytest.approx(totals[14])
    assert sum(calibrated[:30]) == pytest.approx(totals[30])


def test_short_history_is_reported_as_cold_start() -> None:
    readiness = _data_readiness(14)
    assert readiness["maturity"] == "cold_start"
    assert readiness["production_ready"] is False


def test_multi_horizon_requires_enough_days_for_stock_plan() -> None:
    with pytest.raises(ValidationError):
        MultiHorizonForecastRequest.model_validate(
            {
                "product_id": "SKU-1",
                "start_date": "2026-09-22",
                "forecast_days": 30,
                "recent_daily_demand": [1.0] * 7,
                "current_stock": 10,
                "lead_time_days": 20,
                "review_period_days": 20,
                "unit_price": 100,
                "orders_lag_1": 1,
                "customers_lag_1": 1,
            }
        )

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from technova_ai_service.config import get_settings
from technova_ai_service.features.demand_forecasting.inference import (
    load_demand_forecast_bundle,
    predict_demand_forecast,
)
from technova_ai_service.features.demand_forecasting.schemas import (
    DemandForecastRequest,
    DemandForecastResponse,
)

_bundle_cache: dict[str, Any] | None = None


def warmup_demand_model(artifact_path: Path | None = None) -> dict[str, Any]:
    """Warmup and cache the demand forecasting model bundle once at startup/service initialization."""
    global _bundle_cache
    model_path = (
        artifact_path
        if artifact_path is not None
        else get_settings().artifact_dir / "demand_forecasting" / "model.joblib"
    )
    _bundle_cache = load_demand_forecast_bundle(str(model_path))
    return _bundle_cache


def create_demand_forecast(
    request: DemandForecastRequest,
    artifact_path: Path | None = None,
) -> DemandForecastResponse:
    """Service orchestrator for Demand Forecasting.

    Validates incoming request, calls inference module, computes aggregated demand,
    and returns a structured DemandForecastResponse.
    """
    if request.has_historical_data is False:
        raise ValueError("Insufficient historical data")

    model_path = (
        artifact_path
        if artifact_path is not None
        else get_settings().artifact_dir / "demand_forecasting" / "model.joblib"
    )

    predictions, model_info = predict_demand_forecast(
        request=request,
        artifact_path=model_path,
    )

    total_units = round(sum(p.predicted_units for p in predictions), 2)
    start_date = (
        predictions[0].forecast_date
        if predictions
        else (request.forecast_date or dt.datetime.now(dt.UTC).date())
    )

    return DemandForecastResponse(
        product_id=request.product_id,
        store_id=request.store_id,
        forecast_date=start_date,
        predicted_units=total_units,
        horizon=request.horizon,
        predictions=predictions,
        total_predicted_units=total_units,
        model_metadata=model_info,
    )


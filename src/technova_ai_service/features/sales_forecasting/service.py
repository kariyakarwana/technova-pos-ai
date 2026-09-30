from technova_ai_service.config import get_settings
from technova_ai_service.features.sales_forecasting.inference import (
    predict_sales_forecast,
)
from technova_ai_service.features.sales_forecasting.schemas import (
    SalesForecastRequest,
    SalesForecastResponse,
)


def create_sales_forecast(
    request: SalesForecastRequest,
) -> SalesForecastResponse:
    # Validate request data before attempting to load a model artifact so callers
    # receive a useful client error even when a deployment is still warming up.
    if not request.recent_daily_revenue or not any(
        point.revenue > 0 for point in request.recent_daily_revenue
    ):
        raise ValueError("Insufficient historical data")

    artifact = get_settings().artifact_dir / "sales_forecasting" / "model.joblib"
    predictions = predict_sales_forecast(
        artifact_path=artifact,
        organization_id=request.organization_id,
        branch_id=request.branch_id,
        forecast_horizon=request.forecast_horizon,
        recent_daily_revenue=request.recent_daily_revenue,
        future_calendar=request.future_calendar,
    )
    return SalesForecastResponse(
        organization_id=request.organization_id,
        branch_id=request.branch_id,
        forecast_horizon=request.forecast_horizon,
        predictions=predictions,
    )

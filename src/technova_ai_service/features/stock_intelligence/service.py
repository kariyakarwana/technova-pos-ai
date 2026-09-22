from technova_ai_service.config import get_settings
from technova_ai_service.features.stock_intelligence.inference import forecast_stock
from technova_ai_service.features.stock_intelligence.multi_horizon import (
    forecast_multiple_horizons,
)
from technova_ai_service.features.stock_intelligence.schemas import (
    MultiHorizonForecastRequest,
    MultiHorizonForecastResponse,
    StockForecastRequest,
    StockForecastResponse,
)


def create_stock_forecast(request: StockForecastRequest) -> StockForecastResponse:
    artifact = get_settings().artifact_dir / "stock_intelligence" / "model.joblib"
    result = forecast_stock(
        artifact_path=artifact,
        feature_values=request.model_features(),
        current_stock=request.current_stock,
        lead_time_days=request.lead_time_days,
        review_period_days=request.review_period_days,
        service_level=request.service_level,
    )
    return StockForecastResponse.model_validate({"product_id": request.product_id, **result})


def create_multi_horizon_forecast(
    request: MultiHorizonForecastRequest,
) -> MultiHorizonForecastResponse:
    artifact = get_settings().artifact_dir / "stock_intelligence" / "model.joblib"
    result = forecast_multiple_horizons(
        artifact_path=str(artifact),
        start_date=request.start_date,
        forecast_days=request.forecast_days,
        recent_daily_demand=request.recent_daily_demand,
        current_stock=request.current_stock,
        lead_time_days=request.lead_time_days,
        review_period_days=request.review_period_days,
        service_level=request.service_level,
        unit_price=request.unit_price,
        price_index_28=request.price_index_28,
        orders_lag_1=request.orders_lag_1,
        customers_lag_1=request.customers_lag_1,
    )
    return MultiHorizonForecastResponse.model_validate(
        {"product_id": request.product_id, **result}
    )

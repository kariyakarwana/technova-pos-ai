from fastapi import APIRouter, HTTPException

from technova_ai_service.features.stock_intelligence.schemas import (
    MultiHorizonForecastRequest,
    MultiHorizonForecastResponse,
    StockForecastRequest,
    StockForecastResponse,
)
from technova_ai_service.features.stock_intelligence.service import (
    create_multi_horizon_forecast,
    create_stock_forecast,
)

router = APIRouter(prefix="/v1/inventory", tags=["inventory intelligence"])


@router.post("/forecast", response_model=StockForecastResponse)
def forecast(request: StockForecastRequest) -> StockForecastResponse:
    try:
        return create_stock_forecast(request)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/forecast/multi-horizon", response_model=MultiHorizonForecastResponse)
def multi_horizon_forecast(
    request: MultiHorizonForecastRequest,
) -> MultiHorizonForecastResponse:
    try:
        return create_multi_horizon_forecast(request)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

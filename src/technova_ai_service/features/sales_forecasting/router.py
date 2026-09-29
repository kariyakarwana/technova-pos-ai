from fastapi import APIRouter, HTTPException

from technova_ai_service.features.sales_forecasting.schemas import (
    SalesForecastRequest,
    SalesForecastResponse,
)
from technova_ai_service.features.sales_forecasting.service import (
    create_sales_forecast,
)

router = APIRouter(prefix="/v1/sales-forecast", tags=["sales forecasting"])


@router.post("/forecast", response_model=SalesForecastResponse)
def forecast(request: SalesForecastRequest) -> SalesForecastResponse:
    try:
        return create_sales_forecast(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

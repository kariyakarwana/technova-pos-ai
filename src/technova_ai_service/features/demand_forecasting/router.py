from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from technova_ai_service.features.demand_forecasting.schemas import (
    DemandForecastRequest,
    DemandForecastResponse,
)
from technova_ai_service.features.demand_forecasting.service import (
    create_demand_forecast,
)

router = APIRouter(prefix="/v1/demand-forecast", tags=["demand forecasting"])


@router.post(
    "/forecast",
    response_model=DemandForecastResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict multi-day unit demand for a SKU at a store/branch",
)
def forecast(request: DemandForecastRequest) -> DemandForecastResponse:
    """Generate multi-day physical unit demand predictions using trained production LightGBM model.

    Returns:
    - product_id
    - store_id
    - forecast_date
    - predicted_units
    - horizon
    - predictions (list of daily forecast points)
    - total_predicted_units
    - model_metadata (version, model type, target=units_sold, features)
    """
    try:
        return create_demand_forecast(request)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demand forecasting inference failed: {error}",
        ) from error

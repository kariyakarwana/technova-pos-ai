from contextlib import asynccontextmanager
import logging

import uvicorn
from fastapi import FastAPI

from technova_ai_service.features.demand_forecasting.router import (
    router as demand_forecast_router,
)
from technova_ai_service.features.demand_forecasting.service import warmup_demand_model
from technova_ai_service.features.dynamic_pricing.router import router as pricing_router
from technova_ai_service.features.loyalty_recommendations.router import router as loyalty_router
from technova_ai_service.features.recommendation_system.api.router import (
    router as recommendation_router,
)
from technova_ai_service.features.recommendation_system.application.service import (
    warmup_recommendation_model,
)
from technova_ai_service.features.sales_forecasting.router import router as sales_forecast_router
from technova_ai_service.features.stock_intelligence.router import router as stock_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at application startup / service initialization
    try:
        warmup_demand_model()
        logger.info("Demand forecasting model loaded and cached at startup.")
    except Exception as exc:
        logger.warning("Demand forecasting model preload failed or deferred: %s", exc)
    try:
        warmup_recommendation_model()
        logger.info("Recommendation model loaded and cached at startup.")
    except Exception as exc:
        logger.warning("Recommendation model preload failed or deferred: %s", exc)
    yield


app = FastAPI(
    title="TechNova AI Service",
    version="0.1.0",
    description="Inventory, pricing, sales forecasting and loyalty intelligence for TechNova POS.",
    lifespan=lifespan,
)

app.include_router(stock_router)
app.include_router(pricing_router)
app.include_router(loyalty_router)
app.include_router(sales_forecast_router)
app.include_router(demand_forecast_router)
app.include_router(recommendation_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


def run() -> None:
    uvicorn.run("technova_ai_service.main:app", host="0.0.0.0", port=8000)
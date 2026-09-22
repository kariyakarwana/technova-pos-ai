import uvicorn
from fastapi import FastAPI

from technova_ai_service.features.dynamic_pricing.router import router as pricing_router
from technova_ai_service.features.loyalty_recommendations.router import router as loyalty_router
from technova_ai_service.features.stock_intelligence.router import router as stock_router

app = FastAPI(
    title="TechNova AI Service",
    version="0.1.0",
    description="Inventory, pricing and loyalty intelligence for TechNova POS.",
)

app.include_router(stock_router)
app.include_router(pricing_router)
app.include_router(loyalty_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


def run() -> None:
    uvicorn.run("technova_ai_service.main:app", host="0.0.0.0", port=8000)

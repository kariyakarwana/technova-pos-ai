from fastapi import APIRouter, HTTPException

from technova_ai_service.features.dynamic_pricing.schemas import (
    PricingRecommendationRequest,
    PricingRecommendationResponse,
)
from technova_ai_service.features.dynamic_pricing.service import (
    create_pricing_recommendation,
)

router = APIRouter(prefix="/v1/pricing", tags=["dynamic pricing"])


@router.post("/recommend", response_model=PricingRecommendationResponse)
def recommend(request: PricingRecommendationRequest) -> PricingRecommendationResponse:
    try:
        return create_pricing_recommendation(request)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

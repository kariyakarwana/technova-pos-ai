from fastapi import APIRouter, HTTPException

from technova_ai_service.features.loyalty_recommendations.schemas import (
    LoyaltyRecommendationRequest,
    LoyaltyRecommendationResponse,
)
from technova_ai_service.features.loyalty_recommendations.service import (
    create_loyalty_recommendations,
)

router = APIRouter(prefix="/v1/loyalty", tags=["loyalty recommendations"])


@router.post("/recommendations", response_model=LoyaltyRecommendationResponse)
def recommendations(request: LoyaltyRecommendationRequest) -> LoyaltyRecommendationResponse:
    try:
        return create_loyalty_recommendations(request)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

from technova_ai_service.config import get_settings
from technova_ai_service.features.loyalty_recommendations.inference import recommend_products
from technova_ai_service.features.loyalty_recommendations.schemas import (
    LoyaltyRecommendationRequest,
    LoyaltyRecommendationResponse,
)


def create_loyalty_recommendations(
    request: LoyaltyRecommendationRequest,
) -> LoyaltyRecommendationResponse:
    artifact = get_settings().artifact_dir / "loyalty_recommendations" / "model.joblib"
    result = recommend_products(
        artifact_path=artifact,
        customer_id=request.customer_id,
        top_k=request.top_k,
    )
    return LoyaltyRecommendationResponse.model_validate(result)

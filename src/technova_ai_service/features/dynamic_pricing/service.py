from technova_ai_service.config import get_settings
from technova_ai_service.features.dynamic_pricing.inference import recommend_price
from technova_ai_service.features.dynamic_pricing.schemas import (
    PricingRecommendationRequest,
    PricingRecommendationResponse,
)


def create_pricing_recommendation(
    request: PricingRecommendationRequest,
) -> PricingRecommendationResponse:
    artifact = get_settings().artifact_dir / "dynamic_pricing" / "model.joblib"
    result = recommend_price(
        artifact_path=artifact,
        feature_values=request.model_features(),
        current_price=request.current_price,
        unit_cost=request.unit_cost,
        min_margin_rate=request.min_margin_rate,
        max_change_rate=request.max_change_rate,
        candidate_steps=request.candidate_steps,
    )
    return PricingRecommendationResponse.model_validate(
        {
            "product_id": request.product_id,
            "current_price": request.current_price,
            **result,
        }
    )

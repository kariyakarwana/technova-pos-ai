from pydantic import BaseModel, Field


class LoyaltyRecommendationRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


class LoyaltyProfile(BaseModel):
    segment: str
    loyalty_score: float
    recency_days: int
    order_count: int
    total_spend: float
    units_purchased: float


class ProductRecommendation(BaseModel):
    product_id: str
    description: str
    score: float
    reason: str


class LoyaltyRecommendationResponse(BaseModel):
    customer_id: str
    profile: LoyaltyProfile
    recommendations: list[ProductRecommendation]

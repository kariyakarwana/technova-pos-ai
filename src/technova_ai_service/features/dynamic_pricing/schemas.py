from datetime import date

from pydantic import BaseModel, Field, model_validator


class PricingRecommendationRequest(BaseModel):
    product_id: str
    pricing_date: date
    current_price: float = Field(gt=0)
    unit_cost: float = Field(gt=0)
    min_margin_rate: float = Field(default=0.10, ge=0, le=1)
    max_change_rate: float = Field(default=0.15, gt=0, le=0.50)
    candidate_steps: int = Field(default=13, ge=5, le=101)
    demand_lag_1: float = Field(ge=0)
    demand_lag_7: float = Field(ge=0)
    demand_lag_14: float = Field(ge=0)
    rolling_mean_7: float = Field(ge=0)
    rolling_mean_28: float = Field(ge=0)
    rolling_std_7: float = Field(ge=0)
    price_index_28: float = Field(gt=0)
    orders_lag_1: float = Field(ge=0)
    customers_lag_1: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_margin(self) -> "PricingRecommendationRequest":
        if self.current_price <= self.unit_cost:
            raise ValueError("current_price must be greater than unit_cost")
        return self

    def model_features(self) -> dict[str, float]:
        return {
            "demand_lag_1": self.demand_lag_1,
            "demand_lag_7": self.demand_lag_7,
            "demand_lag_14": self.demand_lag_14,
            "rolling_mean_7": self.rolling_mean_7,
            "rolling_mean_28": self.rolling_mean_28,
            "rolling_std_7": self.rolling_std_7,
            "unit_price": self.current_price,
            "price_index_28": self.price_index_28,
            "orders_lag_1": self.orders_lag_1,
            "customers_lag_1": self.customers_lag_1,
            "day_of_week": float(self.pricing_date.weekday()),
            "month": float(self.pricing_date.month),
            "is_weekend": float(self.pricing_date.weekday() >= 5),
        }


class PricingRecommendationResponse(BaseModel):
    product_id: str
    current_price: float
    recommended_price: float
    expected_daily_demand: float
    expected_daily_profit: float
    price_change_rate: float
    decision: str
    minimum_allowed_price: float
    maximum_allowed_price: float

import pytest
from pydantic import ValidationError

from technova_ai_service.features.dynamic_pricing.schemas import (
    PricingRecommendationRequest,
)


def test_pricing_rejects_price_below_cost() -> None:
    with pytest.raises(ValidationError):
        PricingRecommendationRequest(
            product_id="SKU-1",
            pricing_date="2026-09-18",
            current_price=90,
            unit_cost=100,
            demand_lag_1=1,
            demand_lag_7=1,
            demand_lag_14=1,
            rolling_mean_7=1,
            rolling_mean_28=1,
            rolling_std_7=1,
            price_index_28=1,
            orders_lag_1=1,
            customers_lag_1=1,
        )

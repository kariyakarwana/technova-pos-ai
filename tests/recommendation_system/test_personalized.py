"""Unit tests for personalized customer recommendation."""

from __future__ import annotations

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.engine.personalization import (
    PersonalizedRecommender,
)
from technova_ai_service.features.recommendation_system.engine.similarity import (
    ItemSimilarityModel,
)


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_catalog = pd.DataFrame([
        {"product_id": "P_TECH_1", "category_level_1": "Electronics", "brand": "BrandX", "is_active": True},
        {"product_id": "P_TECH_2", "category_level_1": "Electronics", "brand": "BrandX", "is_active": True},
        {"product_id": "P_FOOD_1", "category_level_1": "Groceries", "brand": "BrandY", "is_active": True},
    ])
    df_customers = pd.DataFrame([
        {
            "customer_id": "C_TECH",
            "preferred_categories": ["Electronics"],
            "affinity_brands": ["BrandX"],
        },
        {
            "customer_id": "C_EMPTY",
            "preferred_categories": ["Groceries"],
            "affinity_brands": ["BrandY"],
        },
    ])
    df_baskets = pd.DataFrame([
        {"transaction_id": "T1", "customer_id": "C_TECH", "product_id": "P_TECH_1", "quantity": 1},
    ])
    return df_catalog, df_customers, df_baskets


def test_personalized_recommender(sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    df_catalog, df_customers, df_baskets = sample_data
    item_sim = ItemSimilarityModel().fit(df_baskets)
    rec = PersonalizedRecommender(item_sim_model=item_sim).fit(df_baskets, df_customers, df_catalog)

    # For C_TECH, P_TECH_1 already purchased -> should recommend P_TECH_2 (not purchased, matches category & brand)
    recs = rec.recommend("C_TECH", top_k=5, allow_repurchases=False)
    rec_ids = [r[0] for r in recs]
    assert "P_TECH_1" not in rec_ids  # Repurchase excluded
    assert "P_TECH_2" in rec_ids

    # For C_EMPTY with no purchase history -> fallback to preferred category/brand
    recs_empty = rec.recommend("C_EMPTY", top_k=5)
    rec_empty_ids = [r[0] for r in recs_empty]
    assert "P_FOOD_1" in rec_empty_ids

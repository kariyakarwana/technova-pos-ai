"""Comprehensive verification test for CUSTOMER, TRENDING, and POPULAR contexts."""

import pytest
import pandas as pd
from technova_ai_service.features.recommendation_system.engine.ranking import HybridRecommendationEngine
from technova_ai_service.features.recommendation_system.engine.personalization import PersonalizedRecommender
from technova_ai_service.features.recommendation_system.engine.trending import TrendingEngine
from technova_ai_service.features.recommendation_system.engine.popularity import BranchPopularityEngine
from technova_ai_service.features.recommendation_system.engine.similarity import ItemSimilarityModel


def test_customer_context_with_and_without_history():
    # Setup data
    df_catalog = pd.DataFrame([
        {"product_id": "P1", "organization_id": "org_test", "is_active": True, "category_level_1": "Tech", "brand": "BrandA"},
        {"product_id": "P2", "organization_id": "org_test", "is_active": True, "category_level_1": "Tech", "brand": "BrandA"},
    ])
    df_inventory = pd.DataFrame([
        {"branch_id": "B1", "product_id": "P1", "stock_quantity": 10, "is_available": True},
        {"branch_id": "B1", "product_id": "P2", "stock_quantity": 10, "is_available": True},
    ])
    df_customers = pd.DataFrame([
        {"customer_id": "C_WITH_HISTORY", "preferred_categories": ["Tech"], "affinity_brands": ["BrandA"]},
        {"customer_id": "C_NO_HISTORY", "preferred_categories": ["Tech"], "affinity_brands": ["BrandA"]},
    ])
    df_baskets = pd.DataFrame([
        {"transaction_id": "T1", "customer_id": "C_WITH_HISTORY", "product_id": "P1", "quantity": 1, "store_id": "B1"},
    ])

    pers = PersonalizedRecommender().fit(df_baskets, df_customers, df_catalog)
    engine = HybridRecommendationEngine(personalized_rec=pers)
    engine.set_catalog_and_inventory(df_catalog, df_inventory)

    # 1. No customer purchase history -> ValueError("Insufficient historical data")
    with pytest.raises(ValueError, match="Insufficient historical data"):
        engine.recommend(context="CUSTOMER", customer_id="C_NO_HISTORY", branch_id="B1")

    # Unknown customer -> ValueError("Insufficient historical data")
    with pytest.raises(ValueError, match="Insufficient historical data"):
        engine.recommend(context="CUSTOMER", customer_id="C_UNKNOWN", branch_id="B1")

    # 2. Real customer purchase history -> existing recommendation logic
    recs = engine.recommend(context="CUSTOMER", customer_id="C_WITH_HISTORY", branch_id="B1")
    assert len(recs) > 0
    assert recs[0].product_id in ["P1", "P2"]


def test_trending_context_with_and_without_history():
    df_catalog = pd.DataFrame([
        {"product_id": "P_TREND", "organization_id": "org_test", "is_active": True},
    ])
    df_inventory = pd.DataFrame([
        {"branch_id": "B_ACTIVE", "product_id": "P_TREND", "stock_quantity": 10, "is_available": True},
        {"branch_id": "B_EMPTY", "product_id": "P_TREND", "stock_quantity": 10, "is_available": True},
    ])
    df_baskets = pd.DataFrame([
        {"transaction_id": "T1", "store_id": "B_ACTIVE", "product_id": "P_TREND", "quantity": 10, "transaction_date": "2026-01-01"},
    ])

    trending = TrendingEngine().fit(df_baskets, reference_date="2026-01-01")
    engine = HybridRecommendationEngine(trending_engine=trending)
    engine.set_catalog_and_inventory(df_catalog, df_inventory)

    # 1. Branch with no sales history -> ValueError("Insufficient historical data")
    with pytest.raises(ValueError, match="Insufficient historical data"):
        engine.recommend(context="TRENDING", branch_id="B_EMPTY")

    # 2. Real sales history -> existing trending logic
    recs = engine.recommend(context="TRENDING", branch_id="B_ACTIVE")
    assert len(recs) > 0
    assert recs[0].product_id == "P_TREND"


def test_popular_context_with_and_without_history():
    df_catalog = pd.DataFrame([
        {"product_id": "P_POP", "organization_id": "org_test", "is_active": True},
    ])
    df_inventory = pd.DataFrame([
        {"branch_id": "B1", "product_id": "P_POP", "stock_quantity": 10, "is_available": True},
    ])

    # Case A: Empty baskets (no real organization sales history)
    df_empty_baskets = pd.DataFrame(columns=["transaction_id", "store_id", "product_id", "quantity", "organization_id"])
    pop_empty = BranchPopularityEngine().fit(df_empty_baskets)
    engine_empty = HybridRecommendationEngine(branch_pop_engine=pop_empty)
    engine_empty.set_catalog_and_inventory(df_catalog, df_inventory)

    # Context COLD_START / POPULAR with no org history -> ValueError("Insufficient historical data")
    with pytest.raises(ValueError, match="Insufficient historical data"):
        engine_empty.recommend(context="COLD_START", branch_id="B1")
    with pytest.raises(ValueError, match="Insufficient historical data"):
        engine_empty.recommend(context="POPULAR", branch_id="B1")

    # Case B: Real sales history -> existing popularity logic
    df_baskets = pd.DataFrame([
        {"transaction_id": "T1", "store_id": "B1", "product_id": "P_POP", "quantity": 5, "organization_id": "org_test"},
    ])
    pop_real = BranchPopularityEngine().fit(df_baskets)
    engine_real = HybridRecommendationEngine(branch_pop_engine=pop_real)
    engine_real.set_catalog_and_inventory(df_catalog, df_inventory)

    recs_cold = engine_real.recommend(context="COLD_START", branch_id="B1")
    assert len(recs_cold) > 0
    assert recs_cold[0].product_id == "P_POP"

    recs_pop = engine_real.recommend(context="POPULAR", branch_id="B1")
    assert len(recs_pop) > 0
    assert recs_pop[0].product_id == "P_POP"

"""Unit tests for hybrid ranking, stock-aware filtering, and explainability."""

from __future__ import annotations

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.engine.association import (
    FPGrowthModel,
)
from technova_ai_service.features.recommendation_system.engine.ranking import (
    HybridRecommendationEngine,
    ScoredRecommendation,
)


@pytest.fixture
def sample_setup() -> tuple[HybridRecommendationEngine, pd.DataFrame, pd.DataFrame]:
    df_catalog = pd.DataFrame([
        {"product_id": "P_IN_STOCK", "organization_id": "org_t", "is_active": True},
        {"product_id": "P_OUT_OF_STOCK", "organization_id": "org_t", "is_active": True},
        {"product_id": "P_INACTIVE", "organization_id": "org_t", "is_active": False},
    ])
    df_inventory = pd.DataFrame([
        {"branch_id": "B1", "product_id": "P_IN_STOCK", "stock_quantity": 10, "is_available": True},
        {"branch_id": "B1", "product_id": "P_OUT_OF_STOCK", "stock_quantity": 0, "is_available": False},
        {"branch_id": "B1", "product_id": "P_INACTIVE", "stock_quantity": 0, "is_available": False},
    ])
    df_baskets = pd.DataFrame([
        {"transaction_id": "T1", "product_id": "P_QUERY"},
        {"transaction_id": "T1", "product_id": "P_IN_STOCK"},
        {"transaction_id": "T2", "product_id": "P_QUERY"},
        {"transaction_id": "T2", "product_id": "P_OUT_OF_STOCK"},
        {"transaction_id": "T3", "product_id": "P_QUERY"},
        {"transaction_id": "T3", "product_id": "P_IN_STOCK"},
    ])

    fp_model = FPGrowthModel(min_support=0.2, min_confidence=0.2).fit(df_baskets)
    engine = HybridRecommendationEngine(fp_growth_model=fp_model)
    engine.set_catalog_and_inventory(df_catalog, df_inventory)

    return engine, df_catalog, df_inventory


def test_hybrid_stock_filtering_and_explainability(
    sample_setup: tuple[HybridRecommendationEngine, pd.DataFrame, pd.DataFrame]
) -> None:
    engine, df_catalog, df_inventory = sample_setup

    # Query with enforce_stock=True
    recs = engine.recommend(
        context="PRODUCT",
        product_id="P_QUERY",
        branch_id="B1",
        top_n=5,
        enforce_stock=True,
    )

    rec_ids = [r.product_id for r in recs]
    assert "P_IN_STOCK" in rec_ids
    assert "P_OUT_OF_STOCK" not in rec_ids  # Out-of-stock pruned
    assert "P_INACTIVE" not in rec_ids      # Inactive pruned

    # Verify explainability
    top_rec = recs[0]
    assert top_rec.reason_code == "FREQUENTLY_BOUGHT_TOGETHER"
    assert "Frequently purchased" in top_rec.reason
    assert top_rec.stock_quantity == 10
    assert top_rec.is_available is True


def test_tenant_isolation(
    sample_setup: tuple[HybridRecommendationEngine, pd.DataFrame, pd.DataFrame]
) -> None:
    engine, _, _ = sample_setup

    # Request from different tenant should return empty
    recs_alien = engine.recommend(
        context="PRODUCT",
        product_id="P_QUERY",
        organization_id="org_other_company",
    )
    assert len(recs_alien) == 0


def test_customer_context_insufficient_historical_data(
    sample_setup: tuple[HybridRecommendationEngine, pd.DataFrame, pd.DataFrame]
) -> None:
    engine, _, _ = sample_setup

    # Customer with no purchase history should raise ValueError('Insufficient historical data')
    with pytest.raises(ValueError, match="Insufficient historical data"):
        engine.recommend(
            context="CUSTOMER",
            customer_id="C_UNKNOWN",
            branch_id="B1",
        )


def test_trending_context_insufficient_historical_data(
    sample_setup: tuple[HybridRecommendationEngine, pd.DataFrame, pd.DataFrame]
) -> None:
    engine, _, _ = sample_setup

    # No trending model configured or no sales history for branch
    with pytest.raises(ValueError, match="Insufficient historical data"):
        engine.recommend(
            context="TRENDING",
            branch_id="B_NO_SALES",
        )


def test_cold_start_context_insufficient_historical_data(
    sample_setup: tuple[HybridRecommendationEngine, pd.DataFrame, pd.DataFrame]
) -> None:
    engine, _, _ = sample_setup

    # No branch popularity / org popularity configured
    with pytest.raises(ValueError, match="Insufficient historical data"):
        engine.recommend(
            context="COLD_START",
            branch_id="B1",
        )

"""Unit tests for offline evaluation and chronological splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.evaluation import (
    ChronologicalSplitter,
    OfflineEvaluator,
    RecommendationMetrics,
)


@pytest.fixture
def sample_temporal_baskets() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_catalog = pd.DataFrame([
        {"product_id": "P1", "category_level_1": "Tech", "is_active": True},
        {"product_id": "P2", "category_level_1": "Tech", "is_active": True},
        {"product_id": "P3", "category_level_1": "Food", "is_active": True},
    ])
    rows = [
        # Train period (Aug)
        {"transaction_id": "T1", "store_id": "B1", "customer_id": "C1", "product_id": "P1", "transaction_date": "2014-08-01"},
        {"transaction_id": "T1", "store_id": "B1", "customer_id": "C1", "product_id": "P2", "transaction_date": "2014-08-01"},
        # Val period (Nov)
        {"transaction_id": "T2", "store_id": "B1", "customer_id": "C1", "product_id": "P1", "transaction_date": "2014-11-20"},
        # Test period (Dec) - multi-item basket
        {"transaction_id": "T3", "store_id": "B1", "customer_id": "C2", "product_id": "P1", "transaction_date": "2014-12-10"},
        {"transaction_id": "T3", "store_id": "B1", "customer_id": "C2", "product_id": "P2", "transaction_date": "2014-12-10"},
    ]
    return df_catalog, pd.DataFrame(rows)


def test_chronological_splitter(sample_temporal_baskets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, df_baskets = sample_temporal_baskets
    splitter = ChronologicalSplitter(val_start_date="2014-11-01", test_start_date="2014-12-01")
    df_train, df_val, df_test = splitter.split(df_baskets)

    assert len(df_train) == 2
    assert len(df_val) == 1
    assert len(df_test) == 2
    assert df_train["transaction_date"].max() < df_val["transaction_date"].min()
    assert df_val["transaction_date"].max() < df_test["transaction_date"].min()


def test_offline_evaluator(sample_temporal_baskets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_catalog, df_baskets = sample_temporal_baskets
    evaluator = OfflineEvaluator(df_catalog=df_catalog, sample_eval_baskets=10)

    # Test dummy recommender that predicts the complementary item in basket
    def dummy_recommender(seed_item: str, customer_id: str | None, branch_id: str, top_k: int) -> list[str]:
        return ["P1" if seed_item == "P2" else "P2", "P3"][:top_k]

    metrics = evaluator.evaluate_model(dummy_recommender, df_baskets[df_baskets["transaction_id"] == "T3"], "Dummy")
    assert isinstance(metrics, RecommendationMetrics)
    assert metrics.hit_rate_at_5 == 1.0
    assert metrics.precision_at_5 == 0.2  # 1 hit out of 5 slots
    assert metrics.recall_at_5 == 1.0     # 1 hit out of 1 ground truth item
    assert metrics.ndcg_at_5 > 0.0
    assert metrics.catalog_coverage > 0.0

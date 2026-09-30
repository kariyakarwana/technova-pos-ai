"""Unit tests for trending velocity recommender."""

from __future__ import annotations

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.engine.trending import (
    TrendingEngine,
)


@pytest.fixture
def sample_temporal_baskets() -> pd.DataFrame:
    rows = [
        # P_SURGE: massive volume in last 3 days
        {"transaction_id": "T1", "store_id": "B1", "product_id": "P_SURGE", "quantity": 10, "transaction_date": "2014-11-14"},
        {"transaction_id": "T2", "store_id": "B1", "product_id": "P_SURGE", "quantity": 10, "transaction_date": "2014-11-15"},
        # P_STABLE: steady throughout month
        {"transaction_id": "T3", "store_id": "B1", "product_id": "P_STABLE", "quantity": 2, "transaction_date": "2014-10-25"},
        {"transaction_id": "T4", "store_id": "B1", "product_id": "P_STABLE", "quantity": 2, "transaction_date": "2014-11-05"},
        {"transaction_id": "T5", "store_id": "B1", "product_id": "P_STABLE", "quantity": 2, "transaction_date": "2014-11-15"},
    ]
    return pd.DataFrame(rows)


def test_trending_engine_velocity(sample_temporal_baskets: pd.DataFrame) -> None:
    engine = TrendingEngine(short_window_days=7, long_window_days=30)
    engine.fit(sample_temporal_baskets, reference_date="2014-11-15")

    trending_recs = engine.get_trending_products(top_k=5)
    assert len(trending_recs) == 2
    top_id, top_score = trending_recs[0]
    assert top_id == "P_SURGE"  # Surging item has higher velocity
    assert top_score >= trending_recs[1][1]

"""Unit tests for branch popularity engine with Bayesian smoothing."""

from __future__ import annotations

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.engine.popularity import (
    BranchPopularityEngine,
)


@pytest.fixture
def sample_branch_baskets() -> pd.DataFrame:
    rows = [
        {"organization_id": "org_1", "store_id": "B_HIGH", "product_id": "P_LOCAL_FAV", "quantity": 10},
        {"organization_id": "org_1", "store_id": "B_HIGH", "product_id": "P_GLOBAL", "quantity": 5},
        # B_LOW has only 1 sale of rare item, smoothed toward P_GLOBAL
        {"organization_id": "org_1", "store_id": "B_LOW", "product_id": "P_RARE", "quantity": 1},
        # Global support
        {"organization_id": "org_1", "store_id": "B_OTHER", "product_id": "P_GLOBAL", "quantity": 50},
    ]
    return pd.DataFrame(rows)


def test_branch_popularity_bayesian_smoothing(sample_branch_baskets: pd.DataFrame) -> None:
    engine = BranchPopularityEngine(smoothing_weight=2.0)
    engine.fit(sample_branch_baskets)

    # For B_HIGH, P_LOCAL_FAV has high volume and should rank top
    high_recs = engine.get_popular_products(branch_id="B_HIGH", top_k=2)
    assert high_recs[0][0] == "P_LOCAL_FAV"

    # For B_LOW with small sample, P_GLOBAL should shrink upward due to global popularity
    low_recs = engine.get_popular_products(branch_id="B_LOW", top_k=2)
    assert "P_GLOBAL" in [r[0] for r in low_recs]

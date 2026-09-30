"""Unit tests for Item-to-Item co-occurrence similarity recommender."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.engine.similarity import (
    ItemSimilarityModel,
)


@pytest.fixture
def sample_baskets() -> pd.DataFrame:
    rows = [
        {"transaction_id": "T1", "product_id": "P1"},
        {"transaction_id": "T1", "product_id": "P2"},
        {"transaction_id": "T2", "product_id": "P1"},
        {"transaction_id": "T2", "product_id": "P2"},
        {"transaction_id": "T3", "product_id": "P1"},
        {"transaction_id": "T3", "product_id": "P3"},
        {"transaction_id": "T4", "product_id": "P2"},
        {"transaction_id": "T4", "product_id": "P3"},
    ]
    return pd.DataFrame(rows)


def test_item_similarity_fit_and_recommend(sample_baskets: pd.DataFrame) -> None:
    model = ItemSimilarityModel(min_co_occurrences=1)
    model.fit(sample_baskets)

    recs = model.recommend("P1", top_k=5)
    assert len(recs) > 0
    rec_ids = [r[0] for r in recs]
    assert "P1" not in rec_ids  # Should not recommend self
    assert "P2" in rec_ids

    # Test exclude_ids
    recs_ex = model.recommend("P1", top_k=5, exclude_ids={"P2"})
    rec_ids_ex = [r[0] for r in recs_ex]
    assert "P2" not in rec_ids_ex


def test_item_similarity_save_load(sample_baskets: pd.DataFrame, tmp_path: Path) -> None:
    model = ItemSimilarityModel(min_co_occurrences=1)
    model.fit(sample_baskets)

    path = tmp_path / "item_sim.joblib"
    model.save(path)
    assert path.exists()

    loaded = ItemSimilarityModel.load(path)
    assert loaded.recommend("P1") == model.recommend("P1")

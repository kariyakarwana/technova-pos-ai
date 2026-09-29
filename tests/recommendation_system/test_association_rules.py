"""Unit tests for FP-Growth association rule mining."""

from __future__ import annotations

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.engine.association import (
    AssociationRule,
    FPGrowthModel,
)


@pytest.fixture
def sample_baskets() -> pd.DataFrame:
    # 6 transactions with strong A -> B association:
    rows = [
        {"transaction_id": "T1", "product_id": "A"},
        {"transaction_id": "T1", "product_id": "B"},
        {"transaction_id": "T1", "product_id": "C"},
        {"transaction_id": "T2", "product_id": "A"},
        {"transaction_id": "T2", "product_id": "B"},
        {"transaction_id": "T3", "product_id": "A"},
        {"transaction_id": "T3", "product_id": "B"},
        {"transaction_id": "T3", "product_id": "D"},
        {"transaction_id": "T4", "product_id": "B"},
        {"transaction_id": "T4", "product_id": "C"},
        {"transaction_id": "T5", "product_id": "A"},
        {"transaction_id": "T5", "product_id": "D"},
        {"transaction_id": "T6", "product_id": "A"},
        {"transaction_id": "T6", "product_id": "B"},
    ]
    return pd.DataFrame(rows)


def test_fp_growth_mining(sample_baskets: pd.DataFrame) -> None:
    model = FPGrowthModel(min_support=0.25, min_confidence=0.4, min_lift=0.8)
    model.fit(sample_baskets)

    assert len(model.frequent_itemsets) > 0
    assert len(model.rules) > 0

    recs = model.recommend(["A"], top_k=5)
    assert len(recs) > 0
    rec_items = [r[0] for r in recs]
    assert "B" in rec_items


def test_fp_growth_save_load(sample_baskets: pd.DataFrame, tmp_path: pytest.TempPathFactory) -> None:
    model = FPGrowthModel(min_support=0.3, min_confidence=0.4)
    model.fit(sample_baskets)

    save_file = tmp_path / "fp_rules.joblib"
    model.save(save_file)
    assert save_file.exists()

    loaded = FPGrowthModel.load(save_file)
    assert len(loaded.rules) == len(model.rules)
    assert loaded.recommend(["A"]) == model.recommend(["A"])

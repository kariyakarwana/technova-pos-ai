"""Unit tests for content-based similarity engine."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.engine.similarity import (
    ContentSimilarityModel,
)


@pytest.fixture
def sample_catalog() -> pd.DataFrame:
    rows = [
        {
            "product_id": "P_LAPTOP_1",
            "category_level_1": "Electronics",
            "category_level_2": "Computers",
            "category_level_3": "Laptops",
            "brand": "TechNova",
            "current_unit_price": 999.0,
            "compatibility_tags": ["laptop", "usb_c"],
            "specifications_json": json.dumps({"ram_gb": 16, "ports": ["USB-C"]}),
        },
        {
            "product_id": "P_LAPTOP_2",
            "category_level_1": "Electronics",
            "category_level_2": "Computers",
            "category_level_3": "Laptops",
            "brand": "TechNova",
            "current_unit_price": 1099.0,
            "compatibility_tags": ["laptop", "usb_c"],
            "specifications_json": json.dumps({"ram_gb": 32, "ports": ["USB-C"]}),
        },
        {
            "product_id": "P_COFFEE",
            "category_level_1": "Supermarket",
            "category_level_2": "Beverages",
            "category_level_3": "Coffee",
            "brand": "Heritage",
            "current_unit_price": 12.0,
            "compatibility_tags": ["coffee", "staple"],
            "specifications_json": json.dumps({"weight_g": 500}),
        },
    ]
    return pd.DataFrame(rows)


def test_content_similarity_ranking(sample_catalog: pd.DataFrame) -> None:
    model = ContentSimilarityModel()
    model.fit(sample_catalog)

    # Similar to P_LAPTOP_1 should be P_LAPTOP_2 (high score) and not P_COFFEE
    sims = model.get_similar_products("P_LAPTOP_1", top_k=2)
    assert len(sims) == 2
    top_match, top_score = sims[0]
    second_match, second_score = sims[1]

    assert top_match == "P_LAPTOP_2"
    assert top_score > second_score
    assert second_match == "P_COFFEE"


def test_content_similarity_save_load(sample_catalog: pd.DataFrame, tmp_path: Path) -> None:
    model = ContentSimilarityModel()
    model.fit(sample_catalog)

    path = tmp_path / "content_sim.joblib"
    model.save(path)
    assert path.exists()

    loaded = ContentSimilarityModel.load(path)
    assert loaded.get_similar_products("P_LAPTOP_1") == model.get_similar_products("P_LAPTOP_1")

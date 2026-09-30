"""Unit and integration tests for the recommendation dataset generator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.data.catalog import (
    build_generic_retail_catalog,
)
from technova_ai_service.features.recommendation_system.training.pipeline import (
    RecommendationDatasetGenerator,
)


def test_build_generic_retail_catalog() -> None:
    catalog = build_generic_retail_catalog(organization_id="org_test")
    assert len(catalog) >= 30

    product_ids = [p.product_id for p in catalog]
    assert len(product_ids) == len(set(product_ids))  # unique IDs

    # Check multiple domains
    domains = {p.domain for p in catalog}
    assert "computers" in domains
    assert "phones" in domains
    assert "supermarket" in domains

    # Check inactive products exist for filtering validation
    inactive_items = [p for p in catalog if not p.is_active]
    assert len(inactive_items) >= 2

    # Check schema fields
    for p in catalog:
        assert p.organization_id == "org_test"
        assert p.base_unit_price > 0
        assert p.current_unit_price > 0
        assert p.cost_price > 0
        assert isinstance(p.compatibility_tags, list)
        specs = json.loads(p.specifications_json)
        assert isinstance(specs, dict)


def test_recommendation_dataset_generator_e2e(tmp_path: Path) -> None:
    rossmann_train = Path("data/raw/rossmann/train.csv")
    rossmann_store = Path("data/raw/rossmann/store.csv")

    if not rossmann_train.exists() or not rossmann_store.exists():
        pytest.skip("Rossmann raw files not available")

    out_dir = tmp_path / "rec_data"
    generator = RecommendationDatasetGenerator(
        rossmann_train_path=rossmann_train,
        rossmann_store_path=rossmann_store,
        output_dir=out_dir,
        organization_id="org_test_e2e",
        n_customers=50,
        selected_stores=[1, 2],
        date_start="2014-06-01",
        date_end="2014-06-30",
        random_seed=123,
    )

    result = generator.generate()

    # Check generated files
    assert (out_dir / "catalog_products.parquet").exists()
    assert (out_dir / "customer_profiles.parquet").exists()
    assert (out_dir / "transaction_baskets.parquet").exists()
    assert (out_dir / "branch_inventory.parquet").exists()
    assert (out_dir / "dataset_summary.json").exists()
    assert (out_dir / "dataset_metadata.json").exists()

    df_baskets = pd.read_parquet(out_dir / "transaction_baskets.parquet")
    assert not df_baskets.empty
    assert (df_baskets["quantity"] > 0).all()
    assert (df_baskets["unit_price"] > 0).all()
    assert (df_baskets["line_total"] > 0).all()

    df_inventory = pd.read_parquet(out_dir / "branch_inventory.parquet")
    assert not df_inventory.empty
    assert (df_inventory["stock_quantity"] >= 0).all()
    # Ensure inventory has both in-stock and out-of-stock items
    assert (df_inventory["stock_quantity"] == 0).any()
    assert (df_inventory["stock_quantity"] > 0).any()

    # Check summary metrics
    summary = result["summary"]
    assert summary["entity_counts"]["unique_branches"] == 2
    assert summary["entity_counts"]["total_transactions"] > 0

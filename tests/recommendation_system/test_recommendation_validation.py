"""Tests for recommendation dataset validation suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.evaluation.evaluator import (
    validate_recommendation_dataset,
)
from technova_ai_service.features.recommendation_system.training.pipeline import (
    RecommendationDatasetGenerator,
)


@pytest.fixture(scope="module")
def generated_dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("rec_val_test")
    rossmann_train = Path("data/raw/rossmann/train.csv")
    rossmann_store = Path("data/raw/rossmann/store.csv")

    if not rossmann_train.exists() or not rossmann_store.exists():
        pytest.skip("Rossmann raw files not available")

    generator = RecommendationDatasetGenerator(
        rossmann_train_path=rossmann_train,
        rossmann_store_path=rossmann_store,
        output_dir=tmp_dir,
        organization_id="org_test_val",
        n_customers=60,
        selected_stores=[1, 2, 3],
        date_start="2014-06-01",
        date_end="2014-08-31",
        random_seed=999,
    )
    generator.generate()
    return tmp_dir


def test_validation_passes_on_valid_dataset(generated_dataset_path: Path) -> None:
    report = validate_recommendation_dataset(generated_dataset_path)
    assert report.is_valid is True
    assert report.checks_passed == report.total_checks
    assert report.total_checks >= 15

    # Check metrics payload
    assert report.metrics["unique_products"] >= 30
    assert report.metrics["total_transactions"] > 0
    assert report.metrics["inventory_in_stock_rate"] > 0.5


def test_validation_detects_corrupt_quantities(generated_dataset_path: Path, tmp_path: Path) -> None:
    # Copy dataset to tmp and corrupt
    corrupt_dir = tmp_path / "corrupt_qty"
    corrupt_dir.mkdir()
    for f in generated_dataset_path.glob("*.parquet"):
        df = pd.read_parquet(f)
        if f.name == "transaction_baskets.parquet":
            df.loc[0, "quantity"] = -5  # Negative quantity!
        df.to_parquet(corrupt_dir / f.name, index=False)

    report = validate_recommendation_dataset(corrupt_dir)
    assert report.is_valid is False
    assert report.validation_results["quantities_valid"]["passed"] is False


def test_validation_detects_inactive_sold(generated_dataset_path: Path, tmp_path: Path) -> None:
    corrupt_dir = tmp_path / "corrupt_inactive"
    corrupt_dir.mkdir()
    for f in generated_dataset_path.glob("*.parquet"):
        df = pd.read_parquet(f)
        if f.name == "transaction_baskets.parquet":
            # Assign an inactive SKU to first line item
            df.loc[0, "product_id"] = "PROD-DISC-001"
        df.to_parquet(corrupt_dir / f.name, index=False)

    report = validate_recommendation_dataset(corrupt_dir)
    assert report.is_valid is False
    assert report.validation_results["no_inactive_products_sold"]["passed"] is False


def test_validation_detects_tenant_mismatch(generated_dataset_path: Path, tmp_path: Path) -> None:
    corrupt_dir = tmp_path / "corrupt_tenant"
    corrupt_dir.mkdir()
    for f in generated_dataset_path.glob("*.parquet"):
        df = pd.read_parquet(f)
        if f.name == "customer_profiles.parquet":
            df.loc[0, "organization_id"] = "org_alien_tenant"  # Cross-tenant violation
        df.to_parquet(corrupt_dir / f.name, index=False)

    report = validate_recommendation_dataset(corrupt_dir)
    assert report.is_valid is False
    assert report.validation_results["tenant_consistency"]["passed"] is False

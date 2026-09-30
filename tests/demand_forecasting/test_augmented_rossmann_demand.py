from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from technova_ai_service.features.demand_forecasting.augmented_generator import (
    generate_augmented_rossmann_dataset,
)
from technova_ai_service.features.demand_forecasting.synthetic_catalog import (
    APPROVED_CATEGORIES,
    build_synthetic_product_catalog,
    catalog_to_dataframe,
)
from technova_ai_service.features.demand_forecasting.validation import (
    run_all_validation_gates,
    validate_gate_1_physical_integrity,
    validate_gate_2_behavioral_correlation,
    validate_gate_3_promotional_effects,
    validate_gate_4_catalog_velocity,
    validate_gate_5_panel_feasibility,
)


def test_product_catalog_structure_and_pareto_weights() -> None:
    catalog = build_synthetic_product_catalog(n_products=50)
    assert len(catalog) == 50

    df = catalog_to_dataframe(catalog)
    # Check 5 approved categories
    assert set(df["category"].unique()) == set(APPROVED_CATEGORIES)
    # Check exactly 10 SKUs per category
    category_counts = df["category"].value_counts().to_dict()
    for cat in APPROVED_CATEGORIES:
        assert category_counts[cat] == 10, f"Category {cat} does not have 10 SKUs"

    # Check weights sum to 1.0
    assert abs(df["popularity_weight"].sum() - 1.0) < 1e-5

    # Check Pareto / Zipf top 20% (10 SKUs) volume share between 70% and 82%
    top_10_share = df["popularity_weight"].iloc[:10].sum()
    assert 0.70 <= top_10_share <= 0.82, f"Top 10 share {top_10_share} out of 70-82% range"

    # Check prices
    assert (df["base_unit_price"] > 0).all()
    assert (df["dispersion_param"] > 0).all()


@pytest.fixture(scope="module")
def sample_generated_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, dict]:
    tmp_path = tmp_path_factory.mktemp("augmented_sample")
    train_path = Path("data/raw/rossmann/train.csv")
    store_path = Path("data/raw/rossmann/store.csv")
    out_parquet = tmp_path / "sample_augmented.parquet"
    out_meta = tmp_path / "sample_metadata.json"

    # Generate small sample across 5 stores for rapid testing
    meta = generate_augmented_rossmann_dataset(
        train_path=train_path,
        store_path=store_path,
        output_parquet_path=out_parquet,
        output_metadata_path=out_meta,
        batch_store_size=5,
        max_stores=5,
        random_seed=42,
    )

    import pyarrow.parquet as pq

    table = pq.read_table(out_parquet)
    sample_df = table.to_pandas()
    return sample_df, meta


def test_gate_1_physical_integrity(sample_generated_data: tuple[pd.DataFrame, dict]) -> None:
    df, _ = sample_generated_data
    g1 = validate_gate_1_physical_integrity(df)
    assert g1["passed"] is True
    assert g1["min_units"] >= 0
    assert g1["is_integer_type"] is True
    assert g1["non_zero_closed_units"] == 0
    assert g1["closed_day_compliance"] == 1.0


def test_gate_2_behavioral_correlation_with_customers(sample_generated_data: tuple[pd.DataFrame, dict]) -> None:
    df, _ = sample_generated_data
    g2 = validate_gate_2_behavioral_correlation(df, min_correlation=0.85)
    assert g2["passed"] is True
    assert g2["correlation_r"] >= 0.85


def test_gate_3_promotional_effects(sample_generated_data: tuple[pd.DataFrame, dict]) -> None:
    df, _ = sample_generated_data
    g3 = validate_gate_3_promotional_effects(df)
    assert g3["passed"] is True
    assert g3["promo_mean_units"] > g3["non_promo_mean_units"]
    assert g3["sample_high_elasticity_lift"] > g3["sample_staple_elasticity_lift"]


def test_gate_4_catalog_velocity_and_intermittency(sample_generated_data: tuple[pd.DataFrame, dict]) -> None:
    df, _ = sample_generated_data
    g4 = validate_gate_4_catalog_velocity(df)
    assert g4["passed"] is True
    assert 0.70 <= g4["top_10_share"] <= 0.82
    assert g4["long_tail_zero_demand_pct"] > 0.20


def test_gate_5_panel_feasibility_and_no_duplicates(sample_generated_data: tuple[pd.DataFrame, dict]) -> None:
    df, _ = sample_generated_data
    g5 = validate_gate_5_panel_feasibility(df)
    assert g5["passed"] is True
    assert g5["duplicate_keys"] == 0
    assert g5["lags_computable"] is True
    assert g5["rolling_computable"] is True


def test_all_validation_gates_pass(sample_generated_data: tuple[pd.DataFrame, dict]) -> None:
    df, _ = sample_generated_data
    report = run_all_validation_gates(df)
    assert report.all_gates_passed is True


def test_metadata_provenance(sample_generated_data: tuple[pd.DataFrame, dict]) -> None:
    _, meta = sample_generated_data
    assert meta["provenance_type"] == "augmented_behavioral_synthetic_demand"
    assert meta["grain"] == "Product (SKU) x Branch (Store) x Date -> Physical Units Sold"
    assert meta["n_products"] == 50
    assert len(meta["real_rossmann_fields"]) > 5
    assert len(meta["synthetic_product_fields"]) >= 4

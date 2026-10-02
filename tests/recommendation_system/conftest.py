"""Self-contained recommendation artifacts for the test suite."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.data.catalog import (
    build_generic_retail_catalog,
)
from technova_ai_service.features.recommendation_system.inference import model_loader
from technova_ai_service.features.recommendation_system.training.train import (
    train_recommendation_models,
)


def _write_test_tables(data_dir: Path) -> None:
    organization_id = "org_technova_default"
    catalog = build_generic_retail_catalog(organization_id=organization_id)
    catalog_frame = pd.DataFrame(product.to_dict() for product in catalog)
    catalog_frame.to_parquet(data_dir / "catalog_products.parquet", index=False)

    customer_rows = []
    for index in range(1, 13):
        customer_rows.append(
            {
                "customer_id": f"CUST-{index:04d}",
                "organization_id": organization_id,
                "customer_number": f"CN-{index:04d}",
                "segment": "LOYAL" if index <= 4 else "OCCASIONAL",
                "archetype": "TECH_ENTHUSIAST",
                "price_sensitivity": 0.25,
                "preferred_branch_id": "BRANCH-001",
                "preferred_categories": ["Computers & Electronics"],
                "affinity_brands": ["TechNova", "ApexGear"],
                "total_transactions": 10,
                "lifetime_spend": 2500.0,
                "avg_basket_size": 3.0,
                "last_purchase_date": date(2014, 11, 1),
            }
        )
    pd.DataFrame(customer_rows).to_parquet(
        data_dir / "customer_profiles.parquet", index=False
    )

    basket_patterns = [
        ["PROD-COMP-001", "PROD-PERI-001", "PROD-CASE-001"],
        ["PROD-COMP-002", "PROD-PERI-002", "PROD-PERI-004"],
        ["PROD-PHON-001", "PROD-MACC-001", "PROD-MACC-002"],
        ["PROD-GROC-001", "PROD-GROC-009", "PROD-GROC-006"],
    ]
    price_by_product = catalog_frame.set_index("product_id")[
        "current_unit_price"
    ].to_dict()
    basket_rows = []
    start = datetime(2014, 6, 1, 9, 0, 0, tzinfo=UTC)
    for transaction_index in range(80):
        transaction_id = f"TXN-TEST-{transaction_index:04d}"
        timestamp = start + timedelta(days=transaction_index, minutes=transaction_index)
        products = basket_patterns[transaction_index % len(basket_patterns)]
        for line_index, product_id in enumerate(products, start=1):
            price = float(price_by_product[product_id])
            basket_rows.append(
                {
                    "transaction_id": transaction_id,
                    "organization_id": organization_id,
                    "store_id": "BRANCH-001",
                    "customer_id": f"CUST-{transaction_index % 12 + 1:04d}",
                    "transaction_date": timestamp.date(),
                    "timestamp": timestamp,
                    "line_item_id": line_index,
                    "product_id": product_id,
                    "quantity": 1,
                    "unit_price": price,
                    "line_total": price,
                    "is_promo_applied": transaction_index % 5 == 0,
                }
            )
    pd.DataFrame(basket_rows).to_parquet(
        data_dir / "transaction_baskets.parquet", index=False
    )

    inventory_rows = []
    for branch_id in ("BRANCH-001", "BRANCH-002"):
        for product in catalog:
            stock_quantity = 12 if product.is_active else 0
            inventory_rows.append(
                {
                    "branch_id": branch_id,
                    "product_id": product.product_id,
                    "organization_id": organization_id,
                    "stock_quantity": stock_quantity,
                    "reorder_level": 3,
                    "is_available": product.is_active and stock_quantity > 0,
                }
            )
    pd.DataFrame(inventory_rows).to_parquet(
        data_dir / "branch_inventory.parquet", index=False
    )


@pytest.fixture(scope="session", autouse=True)
def isolated_recommendation_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Build a deterministic bundle instead of relying on ignored local artifacts."""
    root = tmp_path_factory.mktemp("recommendation_bundle")
    data_dir = root / "data"
    artifacts_dir = root / "artifacts"
    data_dir.mkdir()
    artifacts_dir.mkdir()

    _write_test_tables(data_dir)
    train_recommendation_models(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        verbose=False,
    )

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        model_loader,
        "get_default_paths",
        lambda: (artifacts_dir, data_dir),
    )
    model_loader.clear_recommendation_bundle_cache()
    yield
    model_loader.clear_recommendation_bundle_cache()
    patcher.undo()

from __future__ import annotations

import json
from pathlib import Path
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from technova_ai_service.features.demand_forecasting.synthetic_catalog import (
    ProductCatalogItem,
    build_synthetic_product_catalog,
)


AUGMENTED_SCHEMA = pa.schema(
    [
        ("date", pa.string()),
        ("store_id", pa.int32()),
        ("product_id", pa.string()),
        ("product_name", pa.string()),
        ("category", pa.string()),
        ("units_sold", pa.int32()),
        ("unit_price", pa.float32()),
        ("base_unit_price", pa.float32()),
        ("is_open", pa.int8()),
        ("customers", pa.int32()),
        ("is_promo", pa.int8()),
        ("promo2", pa.int8()),
        ("day_of_week", pa.int8()),
        ("state_holiday", pa.string()),
        ("school_holiday", pa.int8()),
        ("store_type", pa.string()),
        ("assortment", pa.string()),
    ]
)


def generate_augmented_rossmann_dataset(
    train_path: str | Path,
    store_path: str | Path,
    output_parquet_path: str | Path,
    output_metadata_path: str | Path,
    *,
    batch_store_size: int = 50,
    random_seed: int = 42,
    base_purchase_rate: float = 0.35,
    max_stores: int | None = None,
) -> dict[str, Any]:
    """Generate the full Product x Store x Date Augmented Rossmann Unit-Demand dataset.

    Memory-Safe Architecture:
    - Iterates over stores in discrete batches (e.g. 50 stores per batch).
    - Writes directly to Parquet using PyArrow ParquetWriter.
    - Zero data leakage, deterministic pseudo-random sampling.
    """
    start_time = time.time()
    tracemalloc.start()

    train_p = Path(train_path)
    store_p = Path(store_path)
    out_parquet_p = Path(output_parquet_path)
    out_meta_p = Path(output_metadata_path)

    out_parquet_p.parent.mkdir(parents=True, exist_ok=True)
    out_meta_p.parent.mkdir(parents=True, exist_ok=True)

    if not train_p.exists():
        raise FileNotFoundError(f"Rossmann train dataset not found at: {train_p}")
    if not store_p.exists():
        raise FileNotFoundError(f"Rossmann store dataset not found at: {store_p}")

    # Build 50-SKU catalog
    catalog: list[ProductCatalogItem] = build_synthetic_product_catalog(n_products=50)

    # Load store metadata
    store_df = pd.read_csv(store_p)
    store_meta = store_df[["Store", "StoreType", "Assortment", "Promo2"]].copy()
    store_meta["StoreType"] = store_meta["StoreType"].fillna("a").astype(str)
    store_meta["Assortment"] = store_meta["Assortment"].fillna("a").astype(str)
    store_meta["Promo2"] = pd.to_numeric(store_meta["Promo2"], errors="coerce").fillna(0).astype(int)

    # Load Rossmann train records
    # Select only required columns for memory efficiency
    train_df = pd.read_csv(
        train_p,
        usecols=[
            "Store",
            "DayOfWeek",
            "Date",
            "Customers",
            "Open",
            "Promo",
            "StateHoliday",
            "SchoolHoliday",
        ],
        low_memory=False,
    )
    train_df["Date"] = train_df["Date"].astype(str)
    train_df["Open"] = pd.to_numeric(train_df["Open"], errors="coerce").fillna(1).astype(int)
    train_df["Customers"] = pd.to_numeric(train_df["Customers"], errors="coerce").fillna(0).astype(int)
    train_df["Promo"] = pd.to_numeric(train_df["Promo"], errors="coerce").fillna(0).astype(int)
    train_df["SchoolHoliday"] = pd.to_numeric(train_df["SchoolHoliday"], errors="coerce").fillna(0).astype(int)
    train_df["StateHoliday"] = train_df["StateHoliday"].astype(str).replace({"0.0": "0", "0": "0"}).fillna("0")

    # Set up deterministic PRNG
    rng = np.random.default_rng(random_seed)

    unique_stores = np.sort(train_df["Store"].unique())
    if max_stores is not None and max_stores > 0:
        unique_stores = unique_stores[:max_stores]
    n_stores = len(unique_stores)
    total_dates = train_df["Date"].nunique()
    date_min = train_df["Date"].min()
    date_max = train_df["Date"].max()

    # Pre-index train by store
    writer = pq.ParquetWriter(out_parquet_p, AUGMENTED_SCHEMA, compression="snappy")
    total_rows_written = 0

    try:
        for i in range(0, n_stores, batch_store_size):
            batch_stores = unique_stores[i : i + batch_store_size]
            batch_train = train_df[train_df["Store"].isin(batch_stores)].copy()
            merged = batch_train.merge(store_meta, on="Store", how="left")

            n_rows = len(merged)
            if n_rows == 0:
                continue

            open_mask = (merged["Open"].to_numpy() == 1) & (merged["Customers"].to_numpy() > 0)
            customers = merged["Customers"].to_numpy()
            day_of_week = merged["DayOfWeek"].to_numpy()
            promo = merged["Promo"].to_numpy()
            school_hol = merged["SchoolHoliday"].to_numpy()

            # Generate demand and assemble batch across all 50 SKUs
            batch_dfs: list[pd.DataFrame] = []
            for item in catalog:
                lam = np.zeros(n_rows, dtype=np.float64)
                if open_mask.any():
                    c_open = customers[open_mask]
                    dow_open = day_of_week[open_mask]
                    p_open = promo[open_mask]
                    sh_open = school_hol[open_mask]

                    mult = (
                        (1.0 + item.promo_elasticity * p_open)
                        * (1.0 + item.weekend_lift * (dow_open == 6))
                        * (1.0 + item.school_holiday_lift * (sh_open == 1))
                    )
                    lam[open_mask] = (
                        c_open
                        * base_purchase_rate
                        * item.popularity_weight
                        * item.basket_depth
                        * mult
                    )

                units = np.zeros(n_rows, dtype=np.int32)
                if open_mask.any():
                    l_open = lam[open_mask]
                    if item.rank <= 5:
                        units[open_mask] = rng.poisson(l_open).astype(np.int32)
                    else:
                        disp = item.dispersion_param
                        p_param = disp / (disp + l_open)
                        units[open_mask] = rng.negative_binomial(disp, p_param).astype(np.int32)

                # Pricing calculation: promotional discount on promo days
                discount = min(0.25, 0.15 * (item.promo_elasticity / 0.7))
                unit_price = np.where(
                    merged["Promo"].to_numpy() == 1,
                    np.round(item.base_unit_price * (1.0 - discount), 2),
                    item.base_unit_price,
                ).astype(np.float32)

                item_df = pd.DataFrame(
                    {
                        "date": merged["Date"].to_numpy(),
                        "store_id": merged["Store"].to_numpy().astype(np.int32),
                        "product_id": item.product_id,
                        "product_name": item.product_name,
                        "category": item.category,
                        "units_sold": units,
                        "unit_price": unit_price,
                        "base_unit_price": np.float32(item.base_unit_price),
                        "is_open": merged["Open"].to_numpy().astype(np.int8),
                        "customers": merged["Customers"].to_numpy().astype(np.int32),
                        "is_promo": merged["Promo"].to_numpy().astype(np.int8),
                        "promo2": merged["Promo2"].to_numpy().astype(np.int8),
                        "day_of_week": merged["DayOfWeek"].to_numpy().astype(np.int8),
                        "state_holiday": merged["StateHoliday"].to_numpy().astype(str),
                        "school_holiday": merged["SchoolHoliday"].to_numpy().astype(np.int8),
                        "store_type": merged["StoreType"].to_numpy().astype(str),
                        "assortment": merged["Assortment"].to_numpy().astype(str),
                    }
                )
                batch_dfs.append(item_df)

            combined_batch = pd.concat(batch_dfs, ignore_index=True)
            batch_table = pa.Table.from_pandas(combined_batch, schema=AUGMENTED_SCHEMA, preserve_index=False)
            writer.write_table(batch_table)
            total_rows_written += len(combined_batch)

    finally:
        writer.close()

    elapsed_time = time.time() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    file_size_bytes = out_parquet_p.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    peak_memory_mb = peak_memory / (1024 * 1024)

    # Save explicit metadata manifest
    metadata = {
        "dataset_name": "augmented_rossmann_unit_demand",
        "provenance_type": "augmented_behavioral_synthetic_demand",
        "grain": "Product (SKU) x Branch (Store) x Date -> Physical Units Sold",
        "generated_row_count": total_rows_written,
        "n_products": len(catalog),
        "n_stores": int(n_stores),
        "n_dates": int(total_dates),
        "date_range": {
            "start": str(date_min),
            "end": str(date_max),
        },
        "performance": {
            "generation_time_seconds": round(elapsed_time, 2),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "output_file_size_mb": round(file_size_mb, 2),
        },
        "real_rossmann_fields": [
            "store_id (from Rossmann Store)",
            "date (from Rossmann Date)",
            "day_of_week (from Rossmann DayOfWeek)",
            "is_open (from Rossmann Open)",
            "customers (from Rossmann Customers footfall)",
            "is_promo (from Rossmann Promo)",
            "promo2 (from Rossmann Promo2)",
            "state_holiday (from Rossmann StateHoliday)",
            "school_holiday (from Rossmann SchoolHoliday)",
            "store_type (from Rossmann StoreType)",
            "assortment (from Rossmann Assortment)",
        ],
        "synthetic_product_fields": [
            "product_id (50 deterministic retail SKUs)",
            "product_name (merchandise description)",
            "category (Beverages, Packaged Foods, Personal Care, Household Cleaning, Health)",
            "base_unit_price (catalog standard retail price)",
            "unit_price (discounted price on promotional days)",
        ],
        "synthetic_units_sold": {
            "target": "units_sold",
            "type": "discrete non-negative integer (int32)",
            "mechanism": "Coupled Negative Binomial / Poisson process anchored to Rossmann customer footfall and calendar/promo elasticities",
            "closed_store_guarantee": "units_sold == 0 whenever is_open == 0 or customers == 0",
        },
        "disclaimer": (
            "This dataset is an augmented behavioral demand dataset developed to provide genuine "
            "unit-count multi-series characteristics while preserving real-world store footfall, "
            "operating schedules, and promotional rhythms from the Rossmann benchmark. It is not raw "
            "POS barcode scanner data."
        ),
    }

    with open(out_meta_p, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata

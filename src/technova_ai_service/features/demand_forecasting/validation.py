from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from technova_ai_service.features.demand_forecasting.synthetic_catalog import (
    ProductCatalogItem,
    build_synthetic_product_catalog,
)


@dataclass
class ValidationReport:
    gate_1_physical_integrity: dict[str, Any]
    gate_2_behavioral_correlation: dict[str, Any]
    gate_3_promotional_effects: dict[str, Any]
    gate_4_catalog_velocity: dict[str, Any]
    gate_5_panel_feasibility: dict[str, Any]
    all_gates_passed: bool


def validate_gate_1_physical_integrity(df: pd.DataFrame) -> dict[str, Any]:
    """Gate 1: Verify non-negative integer units and closed store zero demand."""
    min_units = int(df["units_sold"].min())
    is_integer = bool(np.issubdtype(df["units_sold"].dtype, np.integer))

    closed_rows = df[df["is_open"] == 0]
    closed_count = len(closed_rows)
    non_zero_closed = int((closed_rows["units_sold"] != 0).sum()) if closed_count > 0 else 0
    closed_compliance = 1.0 if closed_count == 0 else float((closed_count - non_zero_closed) / closed_count)

    passed = (min_units >= 0) and is_integer and (non_zero_closed == 0)
    return {
        "passed": passed,
        "min_units": min_units,
        "is_integer_type": is_integer,
        "closed_store_rows": closed_count,
        "non_zero_closed_units": non_zero_closed,
        "closed_day_compliance": closed_compliance,
    }


def validate_gate_2_behavioral_correlation(
    df: pd.DataFrame,
    *,
    min_correlation: float = 0.85,
) -> dict[str, Any]:
    """Gate 2: Aggregate daily store units must strongly correlate with Rossmann Customers (r > 0.85)."""
    # Group by store and date
    store_daily = (
        df.groupby(["store_id", "date"], observed=True)
        .agg(
            total_units=("units_sold", "sum"),
            customers=("customers", "first"),
            is_open=("is_open", "first"),
            day_of_week=("day_of_week", "first"),
        )
        .reset_index()
    )

    open_daily = store_daily[store_daily["is_open"] == 1]
    if len(open_daily) < 10:
        return {"passed": False, "reason": "Insufficient open daily records"}

    corr = float(np.corrcoef(open_daily["customers"], open_daily["total_units"])[0, 1])

    # Weekday average profiles
    weekday_profile = (
        open_daily.groupby("day_of_week", observed=True)[["total_units", "customers"]]
        .mean()
        .round(1)
        .to_dict(orient="index")
    )

    passed = bool(corr >= min_correlation)
    return {
        "passed": passed,
        "correlation_r": round(corr, 4),
        "target_threshold": min_correlation,
        "weekday_profile": weekday_profile,
    }


def validate_gate_3_promotional_effects(df: pd.DataFrame) -> dict[str, Any]:
    """Gate 3: Promotion days have higher mean demand than non-promotion days."""
    open_df = df[df["is_open"] == 1]
    if len(open_df) == 0:
        return {"passed": False, "reason": "No open rows"}

    promo_mean = float(open_df[open_df["is_promo"] == 1]["units_sold"].mean())
    non_promo_mean = float(open_df[open_df["is_promo"] == 0]["units_sold"].mean())
    lift_ratio = float(promo_mean / max(non_promo_mean, 1e-6))

    # Check high-elasticity vs low-elasticity SKU lift
    sku_promo = (
        open_df.groupby(["product_id", "is_promo"], observed=True)["units_sold"]
        .mean()
        .unstack(fill_value=0.0)
    )
    sku_promo["lift"] = sku_promo[1] / sku_promo[0].clip(lower=0.01)

    # SKU-003 (Cola, high promo elasticity) vs SKU-002 (Bread, staple)
    cola_lift = float(sku_promo.loc["SKU-003", "lift"]) if "SKU-003" in sku_promo.index else 1.0
    bread_lift = float(sku_promo.loc["SKU-002", "lift"]) if "SKU-002" in sku_promo.index else 1.0

    passed = (promo_mean > non_promo_mean) and (cola_lift > bread_lift)
    return {
        "passed": passed,
        "promo_mean_units": round(promo_mean, 2),
        "non_promo_mean_units": round(non_promo_mean, 2),
        "aggregate_promo_lift": round(lift_ratio, 2),
        "sample_high_elasticity_lift": round(cola_lift, 2),
        "sample_staple_elasticity_lift": round(bread_lift, 2),
    }


def validate_gate_4_catalog_velocity(df: pd.DataFrame) -> dict[str, Any]:
    """Gate 4: Top 20% SKUs account for ~70-80% volume; long-tail shows >20% zero-demand days on open days."""
    total_volume = int(df["units_sold"].sum())
    if total_volume == 0:
        return {"passed": False, "reason": "Total volume is zero"}

    top10_skus = [f"SKU-{i:03d}" for i in range(1, 11)]
    top10_volume = int(df[df["product_id"].isin(top10_skus)]["units_sold"].sum())
    top10_share = float(top10_volume / total_volume)

    open_df = df[df["is_open"] == 1]
    long_tail_skus = [f"SKU-{i:03d}" for i in range(36, 51)]
    long_tail_open = open_df[open_df["product_id"].isin(long_tail_skus)]
    long_tail_zero_pct = float((long_tail_open["units_sold"] == 0).mean()) if len(long_tail_open) > 0 else 0.0

    # Top staple zero pct (should be near 0%)
    top_staple_open = open_df[open_df["product_id"] == "SKU-001"]
    top_staple_zero_pct = float((top_staple_open["units_sold"] == 0).mean()) if len(top_staple_open) > 0 else 0.0

    passed = (0.70 <= top10_share <= 0.82) and (long_tail_zero_pct > 0.20)
    return {
        "passed": passed,
        "top_10_share": round(top10_share, 4),
        "top_10_target_range": [0.70, 0.82],
        "long_tail_zero_demand_pct": round(long_tail_zero_pct, 4),
        "long_tail_target_min": 0.20,
        "top_staple_zero_demand_pct": round(top_staple_zero_pct, 4),
    }


def validate_gate_5_panel_feasibility(df: pd.DataFrame) -> dict[str, Any]:
    """Gate 5: Dense panel, no duplicate keys, chronological split possible, lags/rolling possible."""
    n_rows = len(df)
    n_dups = int(df.duplicated(subset=["product_id", "store_id", "date"]).sum())

    dates = pd.to_datetime(df["date"].drop_duplicates().sort_values())
    n_dates = len(dates)
    stores = df["store_id"].unique()
    products = df["product_id"].unique()

    # Verify lag-1, lag-7, rolling-7 feasibility on a single series
    sample_series = (
        df[(df["store_id"] == stores[0]) & (df["product_id"] == products[0])]
        .sort_values("date")
        .reset_index(drop=True)
    )
    sample_series["lag_1"] = sample_series["units_sold"].shift(1)
    sample_series["lag_7"] = sample_series["units_sold"].shift(7)
    sample_series["rolling_7"] = sample_series["units_sold"].shift(1).rolling(7).mean()

    lags_computable = bool(sample_series["lag_7"].iloc[10:].notna().all())
    rolling_computable = bool(sample_series["rolling_7"].iloc[10:].notna().all())

    # Temporal split boundaries (70/15/15)
    train_idx = int(n_dates * 0.70)
    val_idx = int(n_dates * 0.85)
    split_info = {
        "total_dates": n_dates,
        "train_dates_count": train_idx,
        "val_dates_count": val_idx - train_idx,
        "test_dates_count": n_dates - val_idx,
        "train_start": str(dates.iloc[0].date()),
        "train_end": str(dates.iloc[train_idx - 1].date()),
        "test_end": str(dates.iloc[-1].date()),
    }

    passed = (n_dups == 0) and lags_computable and rolling_computable and (n_dates >= 100)
    return {
        "passed": passed,
        "total_rows": n_rows,
        "duplicate_keys": n_dups,
        "lags_computable": lags_computable,
        "rolling_computable": rolling_computable,
        "chronological_split": split_info,
    }


def run_all_validation_gates(
    data: pd.DataFrame | str | Path,
    *,
    sample_stores: int | None = None,
) -> ValidationReport:
    """Run all 5 automated validation gates and return a structured report."""
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Parquet dataset not found at: {path}")

        # If sample_stores is provided, read a subset of stores for rapid validation
        if sample_stores is not None:
            # Read first sample_stores
            dataset = pq.ParquetDataset(path)
            # Read metadata to inspect stores
            table = dataset.read(columns=["date", "store_id", "product_id", "units_sold", "customers", "is_open", "is_promo", "day_of_week"])
            df = table.to_pandas()
            stores = df["store_id"].unique()[:sample_stores]
            df = df[df["store_id"].isin(stores)].copy()
        else:
            table = pq.read_table(path)
            df = table.to_pandas()
    else:
        df = data

    g1 = validate_gate_1_physical_integrity(df)
    g2 = validate_gate_2_behavioral_correlation(df)
    g3 = validate_gate_3_promotional_effects(df)
    g4 = validate_gate_4_catalog_velocity(df)
    g5 = validate_gate_5_panel_feasibility(df)

    all_passed = bool(
        g1["passed"] and g2["passed"] and g3["passed"] and g4["passed"] and g5["passed"]
    )

    return ValidationReport(
        gate_1_physical_integrity=g1,
        gate_2_behavioral_correlation=g2,
        gate_3_promotional_effects=g3,
        gate_4_catalog_velocity=g4,
        gate_5_panel_feasibility=g5,
        all_gates_passed=all_passed,
    )

"""Branch inventory data utilities for TechNova AI Recommendation System."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_inventory_dataframe(inventory_path: str | Path) -> pd.DataFrame:
    """Loads branch inventory parquet table into a pandas DataFrame."""
    return pd.read_parquet(inventory_path)


def build_inventory_lookup(
    df_inventory: pd.DataFrame,
) -> dict[tuple[str, str], tuple[int, bool]]:
    """Builds fast lookup map: (branch_id, product_id) -> (stock_quantity, is_available)."""
    lookup: dict[tuple[str, str], tuple[int, bool]] = {}
    for _, row in df_inventory.iterrows():
        b_id = str(row["branch_id"])
        p_id = str(row["product_id"])
        stock = int(row.get("stock_quantity", 0))
        is_avail = bool(row.get("is_available", False))
        lookup[(b_id, p_id)] = (stock, is_avail)
    return lookup

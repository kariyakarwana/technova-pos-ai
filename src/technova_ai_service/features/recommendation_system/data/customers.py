"""Customer data utilities for TechNova AI Recommendation System."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_customers_dataframe(customers_path: str | Path) -> pd.DataFrame:
    """Loads customer profiles parquet table into a pandas DataFrame."""
    return pd.read_parquet(customers_path)

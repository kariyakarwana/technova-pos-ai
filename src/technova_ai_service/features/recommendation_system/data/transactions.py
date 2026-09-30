"""Transaction and basket data utilities for TechNova AI Recommendation System."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ChronologicalSplitter:
    """Partitions transactions chronologically to prevent temporal data leakage."""

    def __init__(
        self,
        val_start_date: str = "2014-11-16",
        test_start_date: str = "2014-12-08",
    ) -> None:
        self.val_start_date = pd.to_datetime(val_start_date)
        self.test_start_date = pd.to_datetime(test_start_date)

    def split(
        self,
        df_baskets: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits transaction baskets into train, val, and test partitions."""
        df = df_baskets.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        train_df = df[df["transaction_date"] < self.val_start_date].copy()
        val_df = df[
            (df["transaction_date"] >= self.val_start_date)
            & (df["transaction_date"] < self.test_start_date)
        ].copy()
        test_df = df[df["transaction_date"] >= self.test_start_date].copy()

        return train_df, val_df, test_df


def load_baskets_dataframe(baskets_path: str | Path) -> pd.DataFrame:
    """Loads transaction baskets parquet table into a pandas DataFrame."""
    return pd.read_parquet(baskets_path)

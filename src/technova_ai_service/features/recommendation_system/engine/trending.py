"""Trending Velocity Recommender for TechNova AI Recommendation Engine."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class TrendingEngine:
    """Computes product sales momentum and acceleration using dual-window exponential velocity."""

    def __init__(
        self,
        short_window_days: int = 7,
        long_window_days: int = 30,
    ) -> None:
        self.short_window_days = short_window_days
        self.long_window_days = long_window_days
        self.global_trending_scores: dict[str, float] = {}
        self.branch_trending_scores: dict[str, dict[str, float]] = {}
        self.reference_date: pd.Timestamp | None = None

    def fit(
        self,
        df_baskets: pd.DataFrame,
        reference_date: str | pd.Timestamp | None = None,
    ) -> TrendingEngine:
        """Calculates trending acceleration relative to reference_date (zero future data leakage)."""
        df = df_baskets.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        if reference_date is None:
            self.reference_date = df["transaction_date"].max()
        else:
            self.reference_date = pd.to_datetime(reference_date)

        # Strictly filter out future transactions
        df = df[df["transaction_date"] <= self.reference_date]

        short_cutoff = self.reference_date - timedelta(days=self.short_window_days)
        long_cutoff = self.reference_date - timedelta(days=self.long_window_days)

        df_short = df[df["transaction_date"] > short_cutoff]
        df_long = df[df["transaction_date"] > long_cutoff]

        # 1. Global Trending Velocity
        short_sales = df_short.groupby("product_id")["quantity"].sum()
        long_sales = df_long.groupby("product_id")["quantity"].sum()

        all_products = set(df["product_id"].unique())
        raw_velocity: dict[str, float] = {}

        for p_id in all_products:
            s_short = float(short_sales.get(p_id, 0)) / max(1, self.short_window_days)
            s_long = float(long_sales.get(p_id, 0)) / max(1, self.long_window_days)
            # Velocity ratio with Laplace smoothing
            velocity = (s_short + 0.1) / (s_long + 0.1)
            raw_velocity[p_id] = velocity

        # Normalize to [0, 1]
        if raw_velocity:
            min_v = min(raw_velocity.values())
            max_v = max(raw_velocity.values())
            denom = max(1e-5, max_v - min_v)
            self.global_trending_scores = {
                p: round(float((v - min_v) / denom), 4)
                for p, v in raw_velocity.items()
            }
        else:
            self.global_trending_scores = {}

        # 2. Branch-specific Trending
        self.branch_trending_scores.clear()
        branches = df["store_id"].unique()
        for b_id in branches:
            df_b_short = df_short[df_short["store_id"] == b_id]
            df_b_long = df_long[df_long["store_id"] == b_id]

            b_short_sales = df_b_short.groupby("product_id")["quantity"].sum()
            b_long_sales = df_b_long.groupby("product_id")["quantity"].sum()

            b_raw_velocity: dict[str, float] = {}
            for p_id in all_products:
                bs_short = float(b_short_sales.get(p_id, 0)) / max(1, self.short_window_days)
                bs_long = float(b_long_sales.get(p_id, 0)) / max(1, self.long_window_days)
                b_velocity = (bs_short + 0.1) / (bs_long + 0.1)
                b_raw_velocity[p_id] = b_velocity

            if b_raw_velocity:
                b_min = min(b_raw_velocity.values())
                b_max = max(b_raw_velocity.values())
                b_denom = max(1e-5, b_max - b_min)
                self.branch_trending_scores[b_id] = {
                    p: round(float((v - b_min) / b_denom), 4)
                    for p, v in b_raw_velocity.items()
                }

        return self

    def get_trending_products(
        self,
        top_k: int = 10,
        branch_id: str | None = None,
        allow_global_fallback: bool = True,
    ) -> list[tuple[str, float]]:
        """Returns top-K trending products globally or for a specific branch."""
        if branch_id and branch_id in self.branch_trending_scores:
            scores = self.branch_trending_scores[branch_id]
        elif branch_id and not allow_global_fallback:
            scores = {}
        else:
            scores = self.global_trending_scores

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

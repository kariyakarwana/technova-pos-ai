"""Branch Popularity Engine with Bayesian Smoothing toward Organization Baseline."""

from __future__ import annotations

import pandas as pd


class BranchPopularityEngine:
    """Computes branch-level product popularity with empirical Bayesian smoothing."""

    def __init__(self, smoothing_weight: float = 25.0) -> None:
        self.smoothing_weight = smoothing_weight
        self.org_popularity: dict[str, float] = {}
        self.branch_popularity: dict[str, dict[str, float]] = {}
        self.organization_id: str = ""

    def fit(
        self,
        df_baskets: pd.DataFrame,
        smoothing_weight: float | None = None,
    ) -> BranchPopularityEngine:
        """Calculates Bayesian smoothed popularity probabilities per branch and organization."""
        if smoothing_weight is not None:
            self.smoothing_weight = smoothing_weight

        self.organization_id = str(df_baskets["organization_id"].iloc[0]) if not df_baskets.empty else ""

        # Global Organization Item Counts
        if "quantity" in df_baskets.columns:
            org_item_counts = df_baskets.groupby("product_id")["quantity"].sum().to_dict()
        else:
            org_item_counts = df_baskets["product_id"].value_counts().to_dict()
        total_org_items = sum(org_item_counts.values())

        if total_org_items > 0:
            self.org_popularity = {
                p: count / total_org_items
                for p, count in org_item_counts.items()
            }
        else:
            self.org_popularity = {}

        # Branch Item Counts
        self.branch_popularity.clear()
        branches = df_baskets["store_id"].unique()

        all_products = list(self.org_popularity.keys())

        for b_id in branches:
            df_b = df_baskets[df_baskets["store_id"] == b_id]
            if "quantity" in df_b.columns:
                b_item_counts = df_b.groupby("product_id")["quantity"].sum().to_dict()
            else:
                b_item_counts = df_b["product_id"].value_counts().to_dict()
            total_b_items = sum(b_item_counts.values())

            smoothed_scores: dict[str, float] = {}
            for p in all_products:
                n_bi = b_item_counts.get(p, 0)
                p_org = self.org_popularity.get(p, 0.0)

                # Bayesian shrinkage formula
                smoothed_prob = (n_bi + self.smoothing_weight * p_org) / (total_b_items + self.smoothing_weight)
                smoothed_scores[p] = float(smoothed_prob)

            # Normalize to [0, 1] relative to max score
            max_s = max(smoothed_scores.values()) if smoothed_scores else 1.0
            self.branch_popularity[b_id] = {
                p: round(float(s / max_s), 4)
                for p, s in smoothed_scores.items()
            }

        return self

    def get_popular_products(
        self,
        branch_id: str | None = None,
        top_k: int = 10,
        allow_org_fallback: bool = True,
    ) -> list[tuple[str, float]]:
        """Returns top-K popular products for a branch (or organization global fallback if allowed)."""
        if branch_id and branch_id in self.branch_popularity:
            scores = self.branch_popularity[branch_id]
        elif allow_org_fallback:
            # Fallback to org popularity normalized to [0, 1]
            max_org = max(self.org_popularity.values()) if self.org_popularity else 1.0
            scores = {
                p: round(float(val / max_org), 4)
                for p, val in self.org_popularity.items()
            }
        else:
            scores = {}

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

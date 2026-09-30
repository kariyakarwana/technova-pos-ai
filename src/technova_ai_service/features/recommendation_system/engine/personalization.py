"""Personalized Customer Recommender for TechNova AI Recommendation Engine."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from .similarity import ItemSimilarityModel


class PersonalizedRecommender:
    """Generates customer-level personalized recommendations leveraging transaction history,

    collaborative item-similarity expansion, and profile brand/category affinities.
    """

    def __init__(self, item_sim_model: ItemSimilarityModel | None = None) -> None:
        self.item_sim_model = item_sim_model
        # Customer history: customer_id -> dict(product_id -> purchase_count)
        self.cust_purchased_products: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Customer profiles: customer_id -> dict
        self.cust_profiles: dict[str, dict[str, Any]] = {}
        # Catalog metadata for category/brand matching
        self.prod_to_cat: dict[str, str] = {}
        self.prod_to_brand: dict[str, str] = {}
        self.active_prod_ids: set[str] = set()

    def fit(
        self,
        df_baskets: pd.DataFrame,
        df_customers: pd.DataFrame,
        df_catalog: pd.DataFrame,
        item_sim_model: ItemSimilarityModel | None = None,
    ) -> PersonalizedRecommender:
        """Indexes customer purchase history and profile preferences."""
        if item_sim_model is not None:
            self.item_sim_model = item_sim_model

        # Index catalog products
        self.prod_to_cat = df_catalog.set_index("product_id")["category_level_1"].to_dict()
        self.prod_to_brand = df_catalog.set_index("product_id")["brand"].to_dict()
        self.active_prod_ids = set(df_catalog[df_catalog["is_active"]]["product_id"])

        # Index customer profiles
        self.cust_profiles = df_customers.set_index("customer_id").to_dict(orient="index")

        # Index purchase history
        self.cust_purchased_products.clear()
        valid_cust_baskets = df_baskets[df_baskets["customer_id"].notna()]
        for _, row in valid_cust_baskets.iterrows():
            c_id = str(row["customer_id"])
            p_id = str(row["product_id"])
            qty = int(row.get("quantity", 1))
            self.cust_purchased_products[c_id][p_id] += qty

        return self

    def recommend(
        self,
        customer_id: str,
        top_k: int = 10,
        allow_repurchases: bool = False,
    ) -> list[tuple[str, float]]:
        """Generates ranked personalized product candidates with affinity scores in [0, 1]."""
        profile = self.cust_profiles.get(customer_id, {})
        history = self.cust_purchased_products.get(customer_id, {})

        pref_cats_val = profile.get("preferred_categories")
        pref_cats = set(pref_cats_val) if pref_cats_val is not None and len(pref_cats_val) > 0 else set()

        aff_brands_val = profile.get("affinity_brands")
        aff_brands = set(aff_brands_val) if aff_brands_val is not None and len(aff_brands_val) > 0 else set()
        purchased_ids = set(history.keys())

        candidate_scores: dict[str, float] = defaultdict(float)

        # 1. Collaborative Expansion from Purchased Items
        if history and self.item_sim_model is not None:
            total_purchases = sum(history.values())
            for prod_id, count in history.items():
                weight = count / max(1, total_purchases)
                # Query item similarity neighbors
                neighbors = self.item_sim_model.recommend(prod_id, top_k=8)
                for neighbor_id, sim_score in neighbors:
                    if neighbor_id not in self.active_prod_ids:
                        continue
                    if not allow_repurchases and neighbor_id in purchased_ids:
                        continue
                    candidate_scores[neighbor_id] += float(weight * sim_score * 2.0)

        # 2. Category & Brand Affinity Boosts
        for prod_id in self.active_prod_ids:
            if not allow_repurchases and prod_id in purchased_ids:
                continue

            boost = 0.0
            cat = self.prod_to_cat.get(prod_id, "")
            brand = self.prod_to_brand.get(prod_id, "")

            if cat in pref_cats:
                boost += 0.5
            if brand in aff_brands:
                boost += 0.5

            if boost > 0:
                candidate_scores[prod_id] += boost

        if not candidate_scores:
            return []

        # Normalize scores to [0, 1]
        max_score = max(candidate_scores.values())
        ranked = [
            (p_id, round(float(raw / max_score), 4))
            for p_id, raw in candidate_scores.items()
        ]
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

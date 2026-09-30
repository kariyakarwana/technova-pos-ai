"""Hybrid Scoring, Stock-Aware Policy Filtering, and Explainability Re-Ranker."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from ..domain.models import ScoredRecommendation
from .association import FPGrowthModel
from .compatibility import CompatibilityEngine
from .personalization import PersonalizedRecommender
from .popularity import BranchPopularityEngine
from .similarity import (
    ContentSimilarityModel,
    ItemSimilarityModel,
    _extract_product_document,
)
from .trending import TrendingEngine

# Default transparent weights per recommendation context
DEFAULT_CONTEXT_WEIGHTS: dict[str, dict[str, float]] = {
    "PRODUCT": {
        "association": 0.40,
        "similarity": 0.25,
        "compatibility": 0.20,
        "branch": 0.05,
        "trending": 0.05,
        "personalized": 0.05,
    },
    "CART_READY": {
        "association": 0.45,
        "compatibility": 0.25,
        "similarity": 0.15,
        "branch": 0.05,
        "trending": 0.05,
        "personalized": 0.05,
    },
    "CUSTOMER": {
        "personalized": 0.45,
        "association": 0.20,
        "branch": 0.15,
        "trending": 0.10,
        "similarity": 0.10,
        "compatibility": 0.00,
    },
    "BRANCH": {
        "branch": 0.50,
        "trending": 0.25,
        "similarity": 0.15,
        "association": 0.10,
        "compatibility": 0.00,
        "personalized": 0.00,
    },
    "TRENDING": {
        "trending": 0.60,
        "branch": 0.20,
        "similarity": 0.10,
        "association": 0.10,
        "compatibility": 0.00,
        "personalized": 0.00,
    },
    "COLD_START": {
        "branch": 0.45,
        "trending": 0.35,
        "similarity": 0.20,
        "association": 0.00,
        "compatibility": 0.00,
        "personalized": 0.00,
    },
}

EXPLANATION_TEMPLATES: dict[str, tuple[str, str]] = {
    "association": (
        "FREQUENTLY_BOUGHT_TOGETHER",
        "Frequently purchased together with items in your selection.",
    ),
    "compatibility": (
        "COMPATIBLE_ACCESSORY",
        "Engineered for verified compatibility with your selected device.",
    ),
    "personalized": (
        "CUSTOMER_HISTORY_AFFINITY",
        "Recommended based on your purchase history and brand preferences.",
    ),
    "similarity": (
        "SIMILAR_PRODUCT",
        "Similar alternative matching specifications, category, and price tier.",
    ),
    "trending": (
        "TRENDING_ACCELERATION",
        "Currently trending with surging sales velocity across stores.",
    ),
    "branch": (
        "BRANCH_POPULAR",
        "Top customer choice and high-demand product at this branch.",
    ),
    "fallback": (
        "POPULAR_FALLBACK",
        "Popular staple across our retail catalog.",
    ),
}


class HybridRecommendationEngine:
    """Orchestrates candidate generation, multi-signal scoring, stock-aware filtering,

    and transparent explainability across all retail contexts.
    """

    def __init__(
        self,
        fp_growth_model: FPGrowthModel | None = None,
        item_sim_model: ItemSimilarityModel | None = None,
        content_sim_model: ContentSimilarityModel | None = None,
        compat_engine: CompatibilityEngine | None = None,
        personalized_rec: PersonalizedRecommender | None = None,
        trending_engine: TrendingEngine | None = None,
        branch_pop_engine: BranchPopularityEngine | None = None,
        context_weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.fp_growth = fp_growth_model
        self.item_sim = item_sim_model
        self.content_sim = content_sim_model
        self.compat = compat_engine
        self.personalized = personalized_rec
        self.trending = trending_engine
        self.branch_pop = branch_pop_engine
        self.context_weights = context_weights or DEFAULT_CONTEXT_WEIGHTS

        # Inventory and catalog index
        self.catalog_df: pd.DataFrame | None = None
        self.inventory_df: pd.DataFrame | None = None
        self.active_prod_ids: set[str] = set()
        self.org_id: str = "org_technova_default"
        # Fast branch inventory index: (branch_id, product_id) -> (stock_qty, is_avail)
        self.inventory_lookup: dict[tuple[str, str], tuple[int, bool]] = {}

    def set_catalog_and_inventory(
        self,
        df_catalog: pd.DataFrame,
        df_inventory: pd.DataFrame,
    ) -> None:
        """Loads and indexes catalog and branch inventory data for real-time stock-aware filtering."""
        self.catalog_df = df_catalog.copy()
        self.inventory_df = df_inventory.copy()
        self.org_id = str(df_catalog["organization_id"].iloc[0]) if not df_catalog.empty else "org_technova_default"

        # Active products
        self.active_prod_ids = set(df_catalog[df_catalog["is_active"]]["product_id"])

        # Populate inventory lookup
        self.inventory_lookup.clear()
        for _, row in df_inventory.iterrows():
            b_id = str(row["branch_id"])
            p_id = str(row["product_id"])
            stock = int(row.get("stock_quantity", 0))
            is_avail = bool(row.get("is_available", False))
            self.inventory_lookup[(b_id, p_id)] = (stock, is_avail)

    def recommend(
        self,
        context: str,
        customer_id: str | None = None,
        product_id: str | None = None,
        cart_product_ids: list[str] | None = None,
        branch_id: str | None = None,
        top_n: int = 10,
        enforce_stock: bool = True,
        organization_id: str | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> list[ScoredRecommendation]:
        """Generates ranked, stock-filtered, explainable recommendations."""
        if context == "POPULAR":
            context = "COLD_START"

        req_org = organization_id or self.org_id
        if req_org != self.org_id:
            # Multi-tenant isolation guard: cross-tenant access returns empty
            return []

        # Context-specific historical data sufficiency validations (no synthetic/default fallbacks)
        if context == "CUSTOMER":
            if not customer_id or self.personalized is None or not self.personalized.cust_purchased_products.get(customer_id):
                raise ValueError("Insufficient historical data")

        elif context == "BRANCH":
            if not branch_id or self.branch_pop is None or branch_id not in self.branch_pop.branch_popularity:
                raise ValueError("Insufficient historical data")

        elif context == "TRENDING":
            if self.trending is None:
                raise ValueError("Insufficient historical data")
            if branch_id:
                if branch_id not in self.trending.branch_trending_scores or not self.trending.branch_trending_scores[branch_id]:
                    raise ValueError("Insufficient historical data")
            else:
                if not self.trending.global_trending_scores:
                    raise ValueError("Insufficient historical data")

        elif context == "COLD_START" and (
            self.branch_pop is None or not self.branch_pop.org_popularity
        ):
            raise ValueError("Insufficient historical data")

        weights = self.context_weights.get(context, self.context_weights["COLD_START"])
        candidate_signals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Input query products
        query_prods = list(cart_product_ids or [])
        if product_id and product_id not in query_prods:
            query_prods.append(product_id)
        query_set = set(query_prods)

        # -------------------------------------------------------------
        # 1. Association Rules Signal (FP-Growth)
        # -------------------------------------------------------------
        if self.fp_growth is not None and query_prods and weights.get("association", 0) > 0:
            for cand_id, score in self.fp_growth.recommend(query_prods, top_k=25):
                candidate_signals[cand_id]["association"] = score

        # -------------------------------------------------------------
        # 2. Compatibility Signal
        # -------------------------------------------------------------
        if self.compat is not None and query_prods and weights.get("compatibility", 0) > 0:
            for q_id in query_prods:
                for cand_id, score in self.compat.find_compatible_products(q_id, top_k=15, exclude_ids=query_set):
                    candidate_signals[cand_id]["compatibility"] = max(candidate_signals[cand_id]["compatibility"], score)

        # -------------------------------------------------------------
        # 3. Item & Content Similarity Signal
        # -------------------------------------------------------------
        if weights.get("similarity", 0) > 0 and query_prods:
            runtime_doc = _extract_product_document(runtime_context) if runtime_context else None
            for q_id in query_prods:
                if self.item_sim is not None:
                    for cand_id, score in self.item_sim.recommend(q_id, top_k=15, exclude_ids=query_set):
                        candidate_signals[cand_id]["similarity"] = max(
                            candidate_signals[cand_id]["similarity"], score
                        )
                if self.content_sim is not None:
                    for cand_id, score in self.content_sim.get_similar_products(
                        q_id, top_k=15, exclude_ids=query_set, runtime_document=runtime_doc
                    ):
                        candidate_signals[cand_id]["similarity"] = max(
                            candidate_signals[cand_id]["similarity"], score
                        )

        # -------------------------------------------------------------
        # 4. Personalized Signal
        # -------------------------------------------------------------
        if self.personalized is not None and customer_id and weights.get("personalized", 0) > 0:
            for cand_id, score in self.personalized.recommend(customer_id, top_k=25):
                if cand_id not in query_set:
                    candidate_signals[cand_id]["personalized"] = score

        # -------------------------------------------------------------
        # 5. Trending Signal
        # -------------------------------------------------------------
        if self.trending is not None and weights.get("trending", 0) > 0:
            allow_global = (context != "TRENDING" and context != "BRANCH")
            for cand_id, score in self.trending.get_trending_products(
                top_k=25, branch_id=branch_id, allow_global_fallback=allow_global
            ):
                if cand_id not in query_set:
                    candidate_signals[cand_id]["trending"] = score

        # -------------------------------------------------------------
        # 6. Branch Popularity Signal
        # -------------------------------------------------------------
        if self.branch_pop is not None and weights.get("branch", 0) > 0:
            allow_fallback = (context != "BRANCH" and context != "TRENDING")
            for cand_id, score in self.branch_pop.get_popular_products(
                branch_id=branch_id, top_k=25, allow_org_fallback=allow_fallback
            ):
                if cand_id not in query_set:
                    candidate_signals[cand_id]["branch"] = score

        # In PRODUCT and CART_READY contexts, recommendations must be grounded in product relationships
        # (association rules, compatibility, or item/content similarity).
        # Unrelated branch/trending items must not be invented as recommendations for an unseen product.
        if context in ("PRODUCT", "CART_READY"):
            product_grounded = {
                cand_id: sigs
                for cand_id, sigs in candidate_signals.items()
                if sigs.get("association", 0) > 0
                or sigs.get("compatibility", 0) > 0
                or sigs.get("similarity", 0) > 0
            }
            candidate_signals = defaultdict(lambda: defaultdict(float), product_grounded)

        # If candidates are empty, raise Insufficient historical data for strict contexts instead of falling back
        if not candidate_signals:
            if context in ("BRANCH", "CUSTOMER", "TRENDING", "COLD_START"):
                raise ValueError("Insufficient historical data")
            if context in ("PRODUCT", "CART_READY"):
                return []
            if self.branch_pop is not None:
                for cand_id, score in self.branch_pop.get_popular_products(branch_id=branch_id, top_k=top_n * 2):
                    candidate_signals[cand_id]["fallback"] = score

        # -------------------------------------------------------------
        # 7. Multi-Signal Scoring & Policy Filtering
        # -------------------------------------------------------------
        ranked_items: list[ScoredRecommendation] = []

        for p_id, signals in candidate_signals.items():
            # A. Policy: Product Active?
            if p_id not in self.active_prod_ids:
                continue

            # B. Policy: Stock Availability Check
            stock_qty = 999
            is_avail = True
            if branch_id:
                inv_entry = self.inventory_lookup.get((branch_id, p_id))
                if inv_entry is not None:
                    stock_qty, is_avail = inv_entry
                else:
                    stock_qty, is_avail = 0, False

            if enforce_stock and (not is_avail or stock_qty <= 0):
                continue

            # C. Weighted Hybrid Score Formulation
            weighted_score = 0.0
            highest_signal_key = "fallback"
            highest_signal_contrib = -1.0

            for sig_key, sig_val in signals.items():
                w = weights.get(sig_key, 0.1 if sig_key == "fallback" else 0.0)
                contrib = w * sig_val
                weighted_score += contrib
                if contrib > highest_signal_contrib and sig_val > 0:
                    highest_signal_contrib = contrib
                    highest_signal_key = sig_key

            # Product-based answers may use trend/popularity as secondary ranking
            # signals, but their user-facing explanation must remain grounded in
            # the queried product relationship.
            if context in ("PRODUCT", "CART_READY"):
                grounded_keys = ("association", "compatibility", "similarity")
                highest_signal_key = max(
                    (key for key in grounded_keys if signals.get(key, 0) > 0),
                    key=lambda key: weights.get(key, 0) * signals[key],
                    default="similarity",
                )

            # D. Determine Explainability Code & Text
            code, reason_txt = EXPLANATION_TEMPLATES.get(
                highest_signal_key, EXPLANATION_TEMPLATES["fallback"]
            )

            # Fallback if weighted_score is very small
            final_score = min(1.0, max(0.01, round(weighted_score, 4)))

            ranked_items.append(
                ScoredRecommendation(
                    product_id=p_id,
                    score=final_score,
                    reason_code=code,
                    reason=reason_txt,
                    stock_quantity=stock_qty,
                    is_available=is_avail,
                    signal_breakdown=dict(signals),
                )
            )

        # Sort descending by score
        ranked_items.sort(key=lambda r: r.score, reverse=True)
        return ranked_items[:top_n]

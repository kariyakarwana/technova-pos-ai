"""Multi-signal candidate generation engine for recommendation retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .association import FPGrowthModel
from .compatibility import CompatibilityEngine
from .personalization import PersonalizedRecommender
from .popularity import BranchPopularityEngine
from .similarity import ContentSimilarityModel, ItemSimilarityModel
from .trending import TrendingEngine


class CandidateGenerationEngine:
    """Retrieves and aggregates candidate recommendation items across algorithmic signals."""

    def __init__(
        self,
        fp_growth_model: FPGrowthModel | None = None,
        item_sim_model: ItemSimilarityModel | None = None,
        content_sim_model: ContentSimilarityModel | None = None,
        compat_engine: CompatibilityEngine | None = None,
        personalized_rec: PersonalizedRecommender | None = None,
        trending_engine: TrendingEngine | None = None,
        branch_pop_engine: BranchPopularityEngine | None = None,
    ) -> None:
        self.fp_growth = fp_growth_model
        self.item_sim = item_sim_model
        self.content_sim = content_sim_model
        self.compat = compat_engine
        self.personalized = personalized_rec
        self.trending = trending_engine
        self.branch_pop = branch_pop_engine

    def generate_candidates(
        self,
        context: str,
        weights: dict[str, float],
        query_products: list[str],
        customer_id: str | None = None,
        branch_id: str | None = None,
        top_n: int = 10,
    ) -> dict[str, dict[str, float]]:
        """Collects candidate products and raw signal affinities for the given context."""
        candidate_signals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        query_set = set(query_products)

        # 1. Association Rules Signal (FP-Growth)
        if self.fp_growth is not None and query_products and weights.get("association", 0) > 0:
            for cand_id, score in self.fp_growth.recommend(query_products, top_k=25):
                candidate_signals[cand_id]["association"] = score

        # 2. Compatibility Signal
        if self.compat is not None and query_products and weights.get("compatibility", 0) > 0:
            for q_id in query_products:
                for cand_id, score in self.compat.find_compatible_products(q_id, top_k=15, exclude_ids=query_set):
                    if score > candidate_signals[cand_id]["compatibility"]:
                        candidate_signals[cand_id]["compatibility"] = score

        # 3. Item & Content Similarity Signal
        if weights.get("similarity", 0) > 0 and query_products:
            for q_id in query_products:
                if self.item_sim is not None:
                    for cand_id, score in self.item_sim.recommend(q_id, top_k=15, exclude_ids=query_set):
                        candidate_signals[cand_id]["similarity"] = max(
                            candidate_signals[cand_id]["similarity"], score
                        )
                if self.content_sim is not None:
                    for cand_id, score in self.content_sim.get_similar_products(q_id, top_k=15, exclude_ids=query_set):
                        candidate_signals[cand_id]["similarity"] = max(
                            candidate_signals[cand_id]["similarity"], score
                        )

        # 4. Personalized Signal
        if self.personalized is not None and customer_id and weights.get("personalized", 0) > 0:
            for cand_id, score in self.personalized.recommend(customer_id, top_k=25):
                if cand_id not in query_set:
                    candidate_signals[cand_id]["personalized"] = score

        # 5. Trending Signal
        if self.trending is not None and weights.get("trending", 0) > 0:
            for cand_id, score in self.trending.get_trending_products(top_k=25, branch_id=branch_id):
                if cand_id not in query_set:
                    candidate_signals[cand_id]["trending"] = score

        # 6. Branch Popularity Signal
        if self.branch_pop is not None and weights.get("branch", 0) > 0:
            for cand_id, score in self.branch_pop.get_popular_products(branch_id=branch_id, top_k=25):
                if cand_id not in query_set:
                    candidate_signals[cand_id]["branch"] = score

        # Fallback if empty (prevent fallback for strict contexts)
        if not candidate_signals:
            if context in ("BRANCH", "CUSTOMER", "TRENDING", "COLD_START", "POPULAR"):
                raise ValueError("Insufficient historical data")
            if context in ("PRODUCT", "CART_READY"):
                return {}
            if self.branch_pop is not None:
                for cand_id, score in self.branch_pop.get_popular_products(branch_id=branch_id, top_k=top_n * 2):
                    candidate_signals[cand_id]["fallback"] = score

        return candidate_signals

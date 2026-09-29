"""Generic Tag-Driven Compatibility Recommender for TechNova POS."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class CompatibilityEngine:
    """Computes technical and operational compatibility between catalog items using tag intersection.

    Fully generic: matches items sharing domain compatibility tokens (e.g., socket types,
    connector interfaces, peripheral form factors, accessory standards) without hardcoding product IDs.
    """

    def __init__(self) -> None:
        self.catalog_df: pd.DataFrame | None = None
        self.prod_tags: dict[str, set[str]] = {}
        self.prod_categories: dict[str, str] = {}
        self.tag_to_products: dict[str, set[str]] = defaultdict(set)

    def fit(self, df_catalog: pd.DataFrame) -> CompatibilityEngine:
        """Indexes compatibility tags and categories across active catalog products."""
        self.catalog_df = df_catalog.copy()
        self.prod_tags.clear()
        self.prod_categories.clear()
        self.tag_to_products.clear()

        for _, row in self.catalog_df.iterrows():
            p_id = str(row["product_id"])
            cat = str(row.get("category_level_1", ""))
            tags = row.get("compatibility_tags")

            tag_set: set[str] = set()
            if isinstance(tags, (list, np.ndarray)):
                tag_set = {str(t).lower().strip() for t in tags if str(t).strip()}

            self.prod_tags[p_id] = tag_set
            self.prod_categories[p_id] = cat

            for tag in tag_set:
                self.tag_to_products[tag].add(p_id)

        return self

    def find_compatible_products(
        self,
        product_id: str,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Identifies compatible items that share specific technical interfaces or complement tags.

        Prioritizes complementary items (cross-subcategory or accessory-tagged items) that share
        concrete technical tokens (e.g., 'socket_lga1700', 'magsafe', 'usb_c').
        """
        if product_id not in self.prod_tags:
            return []

        source_tags = self.prod_tags[product_id]
        if not source_tags:
            return []

        excluded = exclude_ids or set()
        excluded_with_self = excluded | {product_id}

        candidate_scores: dict[str, float] = defaultdict(float)

        for tag in source_tags:
            # Skip universal generic tags from dominating specialized compatibility
            weight = 1.0
            if tag in {"staple", "beverage", "supermarket", "general"}:
                continue
            elif tag in {"usb_c", "bluetooth", "wireless"}:
                weight = 0.5  # common standard
            elif tag.startswith("socket_") or tag in {"magsafe", "ddr5", "hdmi21", "nvme"}:
                weight = 2.0  # high-specificity interface match

            matching_prods = self.tag_to_products.get(tag, set())
            for candidate_id in matching_prods:
                if candidate_id in excluded_with_self:
                    continue
                candidate_scores[candidate_id] += weight

        if not candidate_scores:
            return []

        # Normalize scores to [0, 1]
        max_score = max(candidate_scores.values())
        normalized_results = [
            (p_id, round(float(raw_score / max_score), 4))
            for p_id, raw_score in candidate_scores.items()
        ]

        # Sort descending by score
        normalized_results.sort(key=lambda kv: kv[1], reverse=True)
        return normalized_results[:top_k]

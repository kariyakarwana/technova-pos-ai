"""Similarity recommendation engines (Item Collaborative and Content TF-IDF)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ItemSimilarityModel:
    """Computes collaborative item-to-item cosine similarity from transaction co-occurrences."""

    def __init__(self, min_co_occurrences: int = 2) -> None:
        self.min_co_occurrences = min_co_occurrences
        self.item_counts: dict[str, int] = {}
        # Precomputed top-K similarity index: item_id -> list of (similar_item_id, similarity_score)
        self.similarity_index: dict[str, list[tuple[str, float]]] = {}
        self.unique_items: list[str] = []

    def fit(self, df_baskets: pd.DataFrame) -> ItemSimilarityModel:
        """Fits item co-occurrence similarity matrix from line-item transaction DataFrame."""
        # Step 1: Count single item frequencies
        item_counts: dict[str, int] = defaultdict(int)
        co_counts: dict[tuple[str, str], int] = defaultdict(int)

        baskets_grouped = (
            df_baskets.groupby("transaction_id")["product_id"]
            .apply(lambda s: sorted(set(s)))
        )

        for items in baskets_grouped:
            for item in items:
                item_counts[item] += 1
            if len(items) >= 2:
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        pair = (items[i], items[j])
                        co_counts[pair] += 1

        self.item_counts = dict(item_counts)
        self.unique_items = sorted(item_counts.keys())

        # Step 2: Calculate cosine similarity
        sim_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)

        for (item_a, item_b), count in co_counts.items():
            if count < self.min_co_occurrences:
                continue
            denom = np.sqrt(item_counts[item_a] * item_counts[item_b])
            if denom <= 0:
                continue
            cosine_sim = float(count / denom)

            sim_scores[item_a].append((item_b, cosine_sim))
            sim_scores[item_b].append((item_a, cosine_sim))

        # Sort each item's similar neighbors by descending similarity
        self.similarity_index = {
            item: sorted(neighbors, key=lambda kv: kv[1], reverse=True)
            for item, neighbors in sim_scores.items()
        }
        return self

    def recommend(
        self,
        item_id: str,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieves top-K similar items for a query product ID based on co-occurrence cosine similarity."""
        neighbors = self.similarity_index.get(item_id, [])
        excluded = exclude_ids or set()
        excluded_with_self = excluded | {item_id}

        filtered = [
            (target_id, score)
            for target_id, score in neighbors
            if target_id not in excluded_with_self
        ]
        return filtered[:top_k]

    def save(self, path: str | Path) -> None:
        """Serializes trained model to joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> ItemSimilarityModel:
        """Loads trained model from joblib."""
        return joblib.load(path)


def _extract_product_document(row: pd.Series | dict[str, Any]) -> str:
    """Creates a unified text token document from structured and dynamic product metadata."""
    tokens: list[str] = []

    # Category hierarchy tokens
    cat1 = row.get("category_level_1") or row.get("category")
    if pd.notna(cat1) and cat1:
        tokens.append(f"cat1_{str(cat1).replace(' ', '_').lower()}")
    cat2 = row.get("category_level_2")
    if pd.notna(cat2) and cat2:
        tokens.append(f"cat2_{str(cat2).replace(' ', '_').lower()}")
    cat3 = row.get("category_level_3")
    if pd.notna(cat3) and cat3:
        tokens.append(f"cat3_{str(cat3).replace(' ', '_').lower()}")

    # Brand token
    brand = row.get("brand")
    if pd.notna(brand) and brand:
        tokens.append(f"brand_{str(brand).replace(' ', '_').lower()}")

    # Price tier token
    price_val = row.get("current_unit_price", row.get("base_unit_price", row.get("price", 0.0)))
    try:
        price = float(price_val or 0.0)
    except (ValueError, TypeError):
        price = 0.0
    if price < 15.0:
        price_tier = "tier_budget"
    elif price < 75.0:
        price_tier = "tier_everyday"
    elif price < 300.0:
        price_tier = "tier_midrange"
    elif price < 900.0:
        price_tier = "tier_premium"
    else:
        price_tier = "tier_flagship"
    tokens.append(f"price_{price_tier}")

    # Compatibility tags
    tags = row.get("compatibility_tags")
    if isinstance(tags, (list, np.ndarray)):
        for tag in tags:
            tokens.append(f"tag_{str(tag).lower()}")

    # Name tokens
    name = row.get("name") or row.get("product_name")
    if pd.notna(name) and name:
        for word in str(name).split():
            clean_word = word.strip().lower()
            if len(clean_word) > 2:
                tokens.append(clean_word)

    # Dynamic specifications JSON
    specs_raw = row.get("specifications_json")
    if specs_raw and isinstance(specs_raw, str):
        try:
            specs_dict = json.loads(specs_raw)
            if isinstance(specs_dict, dict):
                for k, v in specs_dict.items():
                    if isinstance(v, list):
                        for sub_v in v:
                            tokens.append(f"spec_{k}_{str(sub_v).replace(' ', '_').lower()}")
                    elif isinstance(v, (str, int, float, bool)):
                        tokens.append(f"spec_{k}_{str(v).replace(' ', '_').lower()}")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return " ".join(tokens)


class ContentSimilarityModel:
    """Content-based recommender indexing product metadata and computing cosine similarity."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w\.-]+\b", sublinear_tf=True)
        self.product_ids: list[str] = []
        self.prod_to_idx: dict[str, int] = {}
        self.similarity_matrix: np.ndarray | None = None
        self.catalog_df: pd.DataFrame | None = None
        self._catalog_matrix: Any = None

    def fit(self, df_catalog: pd.DataFrame) -> ContentSimilarityModel:
        """Fits TF-IDF vectorizer and calculates dense cosine similarity matrix across catalog."""
        self.catalog_df = df_catalog.copy().reset_index(drop=True)
        self.product_ids = self.catalog_df["product_id"].tolist()
        self.prod_to_idx = {p_id: idx for idx, p_id in enumerate(self.product_ids)}

        # Build documents
        documents = [
            _extract_product_document(row)
            for _, row in self.catalog_df.iterrows()
        ]

        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self._catalog_matrix = tfidf_matrix
        return self

    def get_similar_products(
        self,
        product_id: str,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
        runtime_document: str | None = None,
    ) -> list[tuple[str, float]]:
        """Returns top-K content-similar products for a given product ID or runtime document."""
        scores: np.ndarray | None = None
        if product_id in self.prod_to_idx and self.similarity_matrix is not None:
            idx = self.prod_to_idx[product_id]
            scores = self.similarity_matrix[idx]
        elif runtime_document and self.vectorizer is not None and self.catalog_df is not None:
            try:
                if self._catalog_matrix is None:
                    cat_docs = [_extract_product_document(r) for _, r in self.catalog_df.iterrows()]
                    self._catalog_matrix = self.vectorizer.transform(cat_docs)
                q_vec = self.vectorizer.transform([runtime_document])
                scores = cosine_similarity(q_vec, self._catalog_matrix)[0]
            except (AttributeError, TypeError, ValueError):
                scores = None

        if scores is None or len(scores) == 0:
            return []

        excluded = exclude_ids or set()
        excluded_with_self = excluded | {product_id}

        # Sort indices by descending similarity
        ranked_indices = np.argsort(-scores)

        results: list[tuple[str, float]] = []
        for other_idx in ranked_indices:
            other_id = self.product_ids[other_idx]
            if other_id in excluded_with_self:
                continue
            sim_score = float(scores[other_idx])
            if runtime_document is not None and sim_score <= 0.0:
                continue
            results.append((other_id, sim_score))
            if len(results) >= top_k:
                break

        return results

    def save(self, path: str | Path) -> None:
        """Serializes trained model to joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> ContentSimilarityModel:
        """Loads trained model from joblib."""
        return joblib.load(path)

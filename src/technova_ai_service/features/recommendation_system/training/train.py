"""Recommendation model training orchestration and persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import joblib
import pandas as pd

from ..data.transactions import ChronologicalSplitter
from ..engine.association import FPGrowthModel
from ..engine.compatibility import CompatibilityEngine
from ..engine.personalization import PersonalizedRecommender
from ..engine.popularity import BranchPopularityEngine
from ..engine.ranking import DEFAULT_CONTEXT_WEIGHTS
from ..engine.similarity import ContentSimilarityModel, ItemSimilarityModel
from ..engine.trending import TrendingEngine


def train_recommendation_models(
    data_dir: str | Path = "data/processed/recommendation_system",
    artifacts_dir: str | Path = "artifacts/recommendation_system",
    verbose: bool = True,
) -> dict[str, Any]:
    """Trains and serializes all recommendation sub-models from processed parquet tables.

    Produces:
    - association_rules.joblib (FP-Growth frequent itemsets and rules)
    - item_similarity.joblib (collaborative co-occurrence cosine similarity)
    - content_vectorizer.joblib (TF-IDF metadata similarity)
    - branch_popularity.joblib (empirical Bayesian smoothed store popularity)
    - recommendation_metadata.json (components metadata, context weights, top rules)
    """
    d_dir = Path(data_dir)
    a_dir = Path(artifacts_dir)
    a_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Parquet Data
    df_catalog = pd.read_parquet(d_dir / "catalog_products.parquet")
    df_customers = pd.read_parquet(d_dir / "customer_profiles.parquet")
    df_baskets = pd.read_parquet(d_dir / "transaction_baskets.parquet")
    df_inventory = pd.read_parquet(d_dir / "branch_inventory.parquet")

    # 2. Chronological Split
    splitter = ChronologicalSplitter(val_start_date="2014-11-16", test_start_date="2014-12-08")
    df_train, df_val, df_test = splitter.split(df_baskets)

    t0 = time.time()

    # 3. Train FP-Growth Association Rule Miner
    fp_model = FPGrowthModel(min_support=0.005, min_confidence=0.15, min_lift=1.0)
    fp_model.fit(df_train)

    # 4. Train Collaborative Item-to-Item Similarity
    item_sim = ItemSimilarityModel(min_co_occurrences=2)
    item_sim.fit(df_train)

    # 5. Train Content-based Metadata Vectorizer
    content_sim = ContentSimilarityModel()
    content_sim.fit(df_catalog)

    # 6. Index Compatibility Engine
    compat = CompatibilityEngine()
    compat.fit(df_catalog)

    # 7. Index Customer Personalized History
    personalized = PersonalizedRecommender(item_sim_model=item_sim)
    personalized.fit(df_train, df_customers, df_catalog)

    # 8. Train Trending Engine
    trending = TrendingEngine(short_window_days=7, long_window_days=30)
    trending.fit(df_train, reference_date="2014-11-15")

    # 9. Train Branch Popularity Engine (Bayesian Smoothing)
    branch_pop = BranchPopularityEngine(smoothing_weight=20.0)
    branch_pop.fit(df_train)

    t_train = time.time() - t0

    # 10. Save Artifacts
    fp_path = a_dir / "association_rules.joblib"
    item_sim_path = a_dir / "item_similarity.joblib"
    content_path = a_dir / "content_vectorizer.joblib"
    branch_pop_path = a_dir / "branch_popularity.joblib"

    fp_model.save(fp_path)
    item_sim.save(item_sim_path)
    content_sim.save(content_path)
    joblib.dump(branch_pop, branch_pop_path)

    top_rules_sample = [r.to_dict() for r in fp_model.rules[:8]]

    metadata = {
        "model_name": "TechNova Hybrid Product Recommendation Engine",
        "version": "1.0.0",
        "training_metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_time_seconds": round(t_train, 2),
            "date_range": {
                "train_start": str(df_train["transaction_date"].min()),
                "train_end": str(df_train["transaction_date"].max()),
                "val_start": "2014-11-16",
                "test_start": "2014-12-08",
            },
            "training_transactions": int(df_train["transaction_id"].nunique()),
            "training_line_items": len(df_train),
        },
        "model_components": {
            "fp_growth": {
                "min_support": 0.005,
                "min_confidence": 0.15,
                "min_lift": 1.0,
                "rule_count": len(fp_model.rules),
                "frequent_itemsets": len(fp_model.frequent_itemsets),
            },
            "item_similarity": {
                "indexed_items": len(item_sim.similarity_index),
            },
            "content_similarity": {
                "vectorizer": "TfidfVectorizer(sublinear_tf=True)",
                "catalog_items": len(content_sim.product_ids),
            },
            "compatibility": {
                "unique_tags": len(compat.tag_to_products),
            },
            "trending": {
                "short_window_days": 7,
                "long_window_days": 30,
            },
            "branch_popularity": {
                "bayesian_smoothing_weight": 20.0,
                "branches_count": len(branch_pop.branch_popularity),
            },
        },
        "context_weights": DEFAULT_CONTEXT_WEIGHTS,
        "sample_top_rules": top_rules_sample,
    }

    metadata_path = a_dir / "recommendation_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    if verbose:
        print(f"  [OK] Association Rules: {len(fp_model.rules)} rules ({fp_path.name})")
        print(f"  [OK] Item Similarity: {len(item_sim.similarity_index)} items ({item_sim_path.name})")
        print(f"  [OK] Content Vectorizer: {len(content_sim.product_ids)} items ({content_path.name})")
        print(f"  [OK] Branch Popularity: {len(branch_pop.branch_popularity)} branches ({branch_pop_path.name})")
        print(f"  [OK] Recommendation Metadata: {metadata_path.name}")

    return metadata

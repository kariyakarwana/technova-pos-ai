#!/usr/bin/env python
"""Offline Evaluation and Baseline Benchmarking for TechNova Recommendation System."""

from __future__ import annotations

import pprint
import sys
from pathlib import Path

import joblib
import pandas as pd

from technova_ai_service.features.recommendation_system.engine.association import FPGrowthModel
from technova_ai_service.features.recommendation_system.engine.popularity import BranchPopularityEngine
from technova_ai_service.features.recommendation_system.engine.compatibility import CompatibilityEngine
from technova_ai_service.features.recommendation_system.engine.similarity import (
    ContentSimilarityModel,
    ItemSimilarityModel,
)
from technova_ai_service.features.recommendation_system.evaluation import (
    ChronologicalSplitter,
    OfflineEvaluator,
    RecommendationMetrics,
)
from technova_ai_service.features.recommendation_system.engine.personalization import PersonalizedRecommender
from technova_ai_service.features.recommendation_system.engine.ranking import HybridRecommendationEngine
from technova_ai_service.features.recommendation_system.engine.trending import TrendingEngine


def main() -> None:
    data_dir = Path("data/processed/recommendation_system")
    artifacts_dir = Path("artifacts/recommendation_system")

    print("=================================================================")
    print("Offline Chronological Recommendation Evaluation & Benchmarking")
    print(f"Data Source: {data_dir}")
    print(f"Artifacts Source: {artifacts_dir}")
    print("=================================================================")

    # 1. Load Parquet Data
    df_catalog = pd.read_parquet(data_dir / "catalog_products.parquet")
    df_customers = pd.read_parquet(data_dir / "customer_profiles.parquet")
    df_baskets = pd.read_parquet(data_dir / "transaction_baskets.parquet")
    df_inventory = pd.read_parquet(data_dir / "branch_inventory.parquet")

    # 2. Chronological Split
    splitter = ChronologicalSplitter(val_start_date="2014-11-16", test_start_date="2014-12-08")
    df_train, df_val, df_test = splitter.split(df_baskets)

    print(f"Test Split Date Window: {df_test['transaction_date'].min().date()} to {df_test['transaction_date'].max().date()}")
    print(f"Test Transactions: {df_test['transaction_id'].nunique():,}, Line Items: {len(df_test):,}")

    # 3. Load or Build Trained Models (fitted strictly on df_train)
    print("\nLoading models fitted on training split...")
    fp_model = FPGrowthModel.load(artifacts_dir / "association_rules.joblib")
    item_sim = ItemSimilarityModel.load(artifacts_dir / "item_similarity.joblib")
    content_sim = ContentSimilarityModel.load(artifacts_dir / "content_vectorizer.joblib")
    branch_pop = joblib.load(artifacts_dir / "branch_popularity.joblib")

    compat = CompatibilityEngine().fit(df_catalog)
    personalized = PersonalizedRecommender(item_sim_model=item_sim).fit(df_train, df_customers, df_catalog)
    trending = TrendingEngine().fit(df_train, reference_date="2014-11-15")

    hybrid_engine = HybridRecommendationEngine(
        fp_growth_model=fp_model,
        item_sim_model=item_sim,
        content_sim_model=content_sim,
        compat_engine=compat,
        personalized_rec=personalized,
        trending_engine=trending,
        branch_pop_engine=branch_pop,
    )
    hybrid_engine.set_catalog_and_inventory(df_catalog, df_inventory)

    # 4. Define Baseline Predictors
    # Baseline 1: Global Popularity
    global_top_items = [p for p, _ in branch_pop.get_popular_products(branch_id=None, top_k=20)]

    def predict_global_popularity(seed_item: str, customer_id: str | None, branch_id: str, top_k: int) -> list[str]:
        return [it for it in global_top_items if it != seed_item][:top_k]

    # Baseline 2: Branch Popularity
    def predict_branch_popularity(seed_item: str, customer_id: str | None, branch_id: str, top_k: int) -> list[str]:
        items = [p for p, _ in branch_pop.get_popular_products(branch_id=branch_id, top_k=top_k + 1)]
        return [it for it in items if it != seed_item][:top_k]

    # Baseline 3: Trending Velocity
    trending_top_items = [p for p, _ in trending.get_trending_products(top_k=20)]

    def predict_trending(seed_item: str, customer_id: str | None, branch_id: str, top_k: int) -> list[str]:
        return [it for it in trending_top_items if it != seed_item][:top_k]

    # Model: Hybrid Recommendation Engine (Context = PRODUCT with customer/branch info)
    def predict_hybrid(seed_item: str, customer_id: str | None, branch_id: str, top_k: int) -> list[str]:
        recs = hybrid_engine.recommend(
            context="PRODUCT",
            product_id=seed_item,
            customer_id=customer_id,
            branch_id=branch_id,
            top_n=top_k,
            enforce_stock=True,
        )
        return [r.product_id for r in recs]

    # 5. Run Evaluations
    evaluator = OfflineEvaluator(df_catalog=df_catalog, sample_eval_baskets=1500)

    print("\n--- Evaluating Models on Holdout Test Split ---")
    print("Evaluating Baseline 1: Global Popularity...")
    m_global = evaluator.evaluate_model(predict_global_popularity, df_test, "Global Popularity")

    print("Evaluating Baseline 2: Branch Popularity...")
    m_branch = evaluator.evaluate_model(predict_branch_popularity, df_test, "Branch Popularity")

    print("Evaluating Baseline 3: Trending Velocity...")
    m_trending = evaluator.evaluate_model(predict_trending, df_test, "Trending Velocity")

    print("Evaluating TechNova Hybrid Recommendation Engine...")
    m_hybrid = evaluator.evaluate_model(predict_hybrid, df_test, "Hybrid Engine")

    # 6. Display Comparative Table
    print("\n" + "=" * 90)
    print(f"{'Evaluation Metric':<24} | {'Global Pop':<12} | {'Branch Pop':<12} | {'Trending':<12} | {'Hybrid (Ours)':<15}")
    print("=" * 90)
    metrics_keys = [
        ("Precision@5", "precision_at_5"),
        ("Precision@10", "precision_at_10"),
        ("Recall@5", "recall_at_5"),
        ("Recall@10", "recall_at_10"),
        ("Hit Rate@5", "hit_rate_at_5"),
        ("Hit Rate@10", "hit_rate_at_10"),
        ("NDCG@5", "ndcg_at_5"),
        ("NDCG@10", "ndcg_at_10"),
        ("Catalog Coverage", "catalog_coverage"),
        ("Diversity Score", "diversity_score"),
    ]

    for label, attr in metrics_keys:
        val_g = getattr(m_global, attr)
        val_b = getattr(m_branch, attr)
        val_t = getattr(m_trending, attr)
        val_h = getattr(m_hybrid, attr)
        print(f"{label:<24} | {val_g:<12.4f} | {val_b:<12.4f} | {val_t:<12.4f} | {val_h:<15.4f}")
    print("=" * 90)

    print("\nOffline evaluation successfully completed!")


if __name__ == "__main__":
    main()

"""Cached artifact loading and engine initialization for recommendation inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ..data.transactions import ChronologicalSplitter
from ..engine.association import FPGrowthModel
from ..engine.compatibility import CompatibilityEngine
from ..engine.personalization import PersonalizedRecommender
from ..engine.popularity import BranchPopularityEngine
from ..engine.ranking import (
    DEFAULT_CONTEXT_WEIGHTS,
    HybridRecommendationEngine,
)
from ..engine.similarity import ContentSimilarityModel, ItemSimilarityModel
from ..engine.trending import TrendingEngine
from .artifact_registry import ArtifactRegistry, get_default_paths

# Register legacy module paths in sys.modules for joblib/pickle backward compatibility
# when deserializing pre-existing artifacts without touching artifact files on disk.
import sys
from ..engine import (
    association as _engine_association,
    compatibility as _engine_compatibility,
    personalization as _engine_personalization,
    popularity as _engine_popularity,
    ranking as _engine_ranking,
    similarity as _engine_similarity,
    trending as _engine_trending,
)

_LEGACY_MODULE_ALIASES = {
    "technova_ai_service.features.recommendation_system.association_rules": _engine_association,
    "technova_ai_service.features.recommendation_system.branch_popularity": _engine_popularity,
    "technova_ai_service.features.recommendation_system.compatibility": _engine_compatibility,
    "technova_ai_service.features.recommendation_system.content_similarity": _engine_similarity,
    "technova_ai_service.features.recommendation_system.item_similarity": _engine_similarity,
    "technova_ai_service.features.recommendation_system.personalized": _engine_personalization,
    "technova_ai_service.features.recommendation_system.ranking": _engine_ranking,
    "technova_ai_service.features.recommendation_system.trending": _engine_trending,
}
for _legacy_name, _module_obj in _LEGACY_MODULE_ALIASES.items():
    sys.modules.setdefault(_legacy_name, _module_obj)



@dataclass
class RecommendationBundle:
    """In-memory bundle holding all loaded recommendation models, tables, and metadata."""

    fp_model: FPGrowthModel
    item_sim: ItemSimilarityModel
    content_sim: ContentSimilarityModel
    branch_pop: BranchPopularityEngine
    compat: CompatibilityEngine
    personalized: PersonalizedRecommender
    trending: TrendingEngine
    hybrid_engine: HybridRecommendationEngine
    df_catalog: pd.DataFrame
    df_inventory: pd.DataFrame
    metadata: dict[str, Any]


_CACHED_BUNDLE: RecommendationBundle | None = None


def load_recommendation_bundle(
    artifacts_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    force_reload: bool = False,
) -> RecommendationBundle:
    """Loads and caches recommendation model artifacts, catalog, and inventory.

    Fails with FileNotFoundError if required artifacts are missing.
    """
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is not None and not force_reload:
        return _CACHED_BUNDLE

    default_art, default_dat = get_default_paths()
    a_dir = Path(artifacts_dir) if artifacts_dir else default_art
    d_dir = Path(data_dir) if data_dir else default_dat

    registry = ArtifactRegistry(artifacts_dir=a_dir, data_dir=d_dir)

    for p in registry.get_all_required_paths():
        if not p.exists():
            raise FileNotFoundError(
                f"Recommendation system required artifact or data file missing: {p}. "
                f"Ensure Phase 2 dataset and Phase 3 training scripts have been executed."
            )

    # Load artifacts
    fp_model = FPGrowthModel.load(registry.fp_growth_path)
    item_sim = ItemSimilarityModel.load(registry.item_similarity_path)
    content_sim = ContentSimilarityModel.load(registry.content_vectorizer_path)
    branch_pop = joblib.load(registry.branch_popularity_path)

    with open(registry.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load data tables
    df_catalog = pd.read_parquet(registry.catalog_path)
    df_customers = pd.read_parquet(registry.customer_path)
    df_baskets = pd.read_parquet(registry.baskets_path)
    df_inventory = pd.read_parquet(registry.inventory_path)

    # Re-link train-split components (to guarantee zero future leakage)
    splitter = ChronologicalSplitter(val_start_date="2014-11-16", test_start_date="2014-12-08")
    df_train, _, _ = splitter.split(df_baskets)

    compat = CompatibilityEngine().fit(df_catalog)
    personalized = PersonalizedRecommender(item_sim_model=item_sim).fit(
        df_train, df_customers, df_catalog
    )
    trending = TrendingEngine().fit(df_train, reference_date="2014-11-15")

    # Build hybrid ranking orchestrator
    hybrid_engine = HybridRecommendationEngine(
        fp_growth_model=fp_model,
        item_sim_model=item_sim,
        content_sim_model=content_sim,
        compat_engine=compat,
        personalized_rec=personalized,
        trending_engine=trending,
        branch_pop_engine=branch_pop,
        context_weights=DEFAULT_CONTEXT_WEIGHTS,
    )
    hybrid_engine.set_catalog_and_inventory(df_catalog, df_inventory)

    _CACHED_BUNDLE = RecommendationBundle(
        fp_model=fp_model,
        item_sim=item_sim,
        content_sim=content_sim,
        branch_pop=branch_pop,
        compat=compat,
        personalized=personalized,
        trending=trending,
        hybrid_engine=hybrid_engine,
        df_catalog=df_catalog,
        df_inventory=df_inventory,
        metadata=metadata,
    )
    return _CACHED_BUNDLE


def clear_recommendation_bundle_cache() -> None:
    """Clears the in-memory cached recommendation bundle (useful for testing)."""
    global _CACHED_BUNDLE
    _CACHED_BUNDLE = None

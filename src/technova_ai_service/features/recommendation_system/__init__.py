"""Recommendation System feature package for TechNova POS.

Architected into clean layers:
- domain: Enums, value objects, and contracts
- data: Data ingestion, catalog, transactions, and inventory
- engine: Algorithmic recommendation sub-engines and ranking
- inference: Artifact registry, model loader, and inference bundle
- application: Service orchestration, policy enforcement, and provenance
- api: FastAPI schemas and routing endpoints
- training: Dataset generation and model training pipelines
- evaluation: Offline metrics and dataset validation
"""

from __future__ import annotations

from typing import Any

from .api.router import router as recommendation_router
from .api.schemas import (
    RecommendationContext,
    RecommendationModelMetadata,
    RecommendationReasonCode,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedProduct,
)
from .application.service import (
    RecommendationService,
    create_recommendations,
    get_recommendation_service,
    warmup_recommendation_model,
)
from .data.catalog import (
    COMPLEMENTARY_AFFINITY_RULES,
    CatalogProduct,
    build_generic_retail_catalog,
)
from .domain.contracts import (
    CandidateGenerator,
    Recommender,
)
from .domain.contracts import (
    RecommendationPredictor as IRecommendationPredictor,
)
from .domain.models import ScoredRecommendation
from .engine.association import AssociationRule, FPGrowthModel
from .engine.compatibility import CompatibilityEngine
from .engine.personalization import PersonalizedRecommender
from .engine.popularity import BranchPopularityEngine
from .engine.ranking import (
    DEFAULT_CONTEXT_WEIGHTS,
    EXPLANATION_TEMPLATES,
    HybridRecommendationEngine,
)
from .engine.similarity import ContentSimilarityModel, ItemSimilarityModel
from .engine.trending import TrendingEngine
from .inference.artifact_registry import get_default_paths
from .inference.model_loader import (
    RecommendationBundle,
    clear_recommendation_bundle_cache,
    load_recommendation_bundle,
)
from .inference.predictor import RecommendationPredictor

# Lazy loading for training and evaluation to prevent importing during runtime startup
_LAZY_EXPORTS = {
    "ChronologicalSplitter": ("technova_ai_service.features.recommendation_system.data.transactions", "ChronologicalSplitter"),
    "OfflineEvaluator": ("technova_ai_service.features.recommendation_system.evaluation.evaluator", "OfflineEvaluator"),
    "RecommendationMetrics": ("technova_ai_service.features.recommendation_system.evaluation.metrics", "RecommendationMetrics"),
    "validate_recommendation_dataset": ("technova_ai_service.features.recommendation_system.evaluation.evaluator", "validate_recommendation_dataset"),
    "RecommendationValidationReport": ("technova_ai_service.features.recommendation_system.evaluation.evaluator", "RecommendationValidationReport"),
    "RecommendationDatasetGenerator": ("technova_ai_service.features.recommendation_system.training.pipeline", "RecommendationDatasetGenerator"),
    "generate_recommendation_dataset": ("technova_ai_service.features.recommendation_system.training.pipeline", "generate_recommendation_dataset"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "COMPLEMENTARY_AFFINITY_RULES",
    "DEFAULT_CONTEXT_WEIGHTS",
    "EXPLANATION_TEMPLATES",
    "AssociationRule",
    "BranchPopularityEngine",
    "CandidateGenerator",
    "CatalogProduct",
    "ChronologicalSplitter",
    "CompatibilityEngine",
    "ContentSimilarityModel",
    "FPGrowthModel",
    "HybridRecommendationEngine",
    "IRecommendationPredictor",
    "ItemSimilarityModel",
    "OfflineEvaluator",
    "PersonalizedRecommender",
    "RecommendationBundle",
    "RecommendationContext",
    "RecommendationDatasetGenerator",
    "RecommendationMetrics",
    "RecommendationModelMetadata",
    "RecommendationPredictor",
    "RecommendationReasonCode",
    "RecommendationRequest",
    "RecommendationResponse",
    "RecommendationService",
    "RecommendationValidationReport",
    "RecommendedProduct",
    "Recommender",
    "ScoredRecommendation",
    "TrendingEngine",
    "build_generic_retail_catalog",
    "clear_recommendation_bundle_cache",
    "create_recommendations",
    "generate_recommendation_dataset",
    "get_default_paths",
    "get_recommendation_service",
    "load_recommendation_bundle",
    "recommendation_router",
    "validate_recommendation_dataset",
    "warmup_recommendation_model",
]

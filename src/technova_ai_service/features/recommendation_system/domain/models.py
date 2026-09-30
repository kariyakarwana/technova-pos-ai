"""Domain entities and value objects for the TechNova AI Recommendation System."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoredRecommendation:
    """Core domain model representing a scored, ranked recommendation candidate."""

    product_id: str
    score: float
    reason_code: str
    reason: str
    stock_quantity: int
    is_available: bool
    signal_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecommendationMetrics:
    """Offline ranking and evaluation metrics."""

    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    hit_rate_at_5: float
    hit_rate_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    catalog_coverage: float
    diversity_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "Precision@5": round(self.precision_at_5, 4),
            "Precision@10": round(self.precision_at_10, 4),
            "Recall@5": round(self.recall_at_5, 4),
            "Recall@10": round(self.recall_at_10, 4),
            "Hit_Rate@5": round(self.hit_rate_at_5, 4),
            "Hit_Rate@10": round(self.hit_rate_at_10, 4),
            "NDCG@5": round(self.ndcg_at_5, 4),
            "NDCG@10": round(self.ndcg_at_10, 4),
            "Catalog_Coverage": round(self.catalog_coverage, 4),
            "Diversity": round(self.diversity_score, 4),
        }


@dataclass(frozen=True)
class CatalogProduct:
    """Domain model for a product in the retail catalog."""

    product_id: str
    organization_id: str
    sku: str
    name: str
    category_level_1: str
    category_level_2: str
    category_level_3: str
    brand: str
    cost_price: float
    base_unit_price: float
    current_unit_price: float
    specifications_json: str
    compatibility_tags: list[str]
    is_active: bool
    popularity_weight: float = 1.0
    domain: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssociationRule:
    """Frequent itemset association rule."""

    antecedent: frozenset[str]
    consequent: frozenset[str]
    support: float
    confidence: float
    lift: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "antecedent": list(self.antecedent),
            "consequent": list(self.consequent),
            "support": round(self.support, 5),
            "confidence": round(self.confidence, 4),
            "lift": round(self.lift, 4),
            "count": self.count,
        }


@dataclass
class RecommendationValidationReport:
    """Dataset integrity and signal verification report."""

    is_valid: bool
    checks_passed: int
    total_checks: int
    validation_results: dict[str, dict[str, Any]]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

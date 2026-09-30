"""Artifact registry and canonical path resolution for recommendation models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_default_paths() -> tuple[Path, Path]:
    """Returns canonical paths for recommendation artifacts and processed datasets."""
    artifacts_dir = Path("artifacts/recommendation_system")
    data_dir = Path("data/processed/recommendation_system")
    return artifacts_dir, data_dir


@dataclass(frozen=True)
class ArtifactRegistry:
    """Registry maintaining verified file locations for recommendation artifacts and data tables."""

    artifacts_dir: Path
    data_dir: Path

    @classmethod
    def default(cls) -> ArtifactRegistry:
        a_dir, d_dir = get_default_paths()
        return cls(artifacts_dir=a_dir, data_dir=d_dir)

    @property
    def fp_growth_path(self) -> Path:
        return self.artifacts_dir / "association_rules.joblib"

    @property
    def item_similarity_path(self) -> Path:
        return self.artifacts_dir / "item_similarity.joblib"

    @property
    def content_vectorizer_path(self) -> Path:
        return self.artifacts_dir / "content_vectorizer.joblib"

    @property
    def branch_popularity_path(self) -> Path:
        return self.artifacts_dir / "branch_popularity.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / "recommendation_metadata.json"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog_products.parquet"

    @property
    def customer_path(self) -> Path:
        return self.data_dir / "customer_profiles.parquet"

    @property
    def baskets_path(self) -> Path:
        return self.data_dir / "transaction_baskets.parquet"

    @property
    def inventory_path(self) -> Path:
        return self.data_dir / "branch_inventory.parquet"

    def get_all_required_paths(self) -> list[Path]:
        return [
            self.fp_growth_path,
            self.item_similarity_path,
            self.content_vectorizer_path,
            self.branch_popularity_path,
            self.metadata_path,
            self.catalog_path,
            self.customer_path,
            self.baskets_path,
            self.inventory_path,
        ]

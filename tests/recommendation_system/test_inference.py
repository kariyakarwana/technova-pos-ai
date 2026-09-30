"""Unit tests for recommendation bundle loading and artifact cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from technova_ai_service.features.recommendation_system.inference import (
    clear_recommendation_bundle_cache,
    load_recommendation_bundle,
)


def test_load_recommendation_bundle_success() -> None:
    clear_recommendation_bundle_cache()
    bundle = load_recommendation_bundle()

    assert bundle is not None
    assert bundle.fp_model is not None
    assert bundle.item_sim is not None
    assert bundle.content_sim is not None
    assert bundle.branch_pop is not None
    assert len(bundle.df_catalog) > 0
    assert len(bundle.df_inventory) > 0

    # Test caching (returns same instance)
    bundle_cached = load_recommendation_bundle()
    assert bundle_cached is bundle


def test_load_recommendation_bundle_missing_file(tmp_path: Path) -> None:
    clear_recommendation_bundle_cache()
    # Query non-existent directory -> FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc:
        load_recommendation_bundle(artifacts_dir=tmp_path / "missing_dir")
    assert "missing" in str(exc.value).lower()

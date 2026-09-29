"""Data management layer for TechNova AI Recommendation System."""

from __future__ import annotations

from .catalog import (
    COMPLEMENTARY_AFFINITY_RULES,
    CatalogProduct,
    build_generic_retail_catalog,
    load_catalog_dataframe,
)
from .customers import load_customers_dataframe
from .inventory import build_inventory_lookup, load_inventory_dataframe
from .transactions import ChronologicalSplitter, load_baskets_dataframe

__all__ = [
    "CatalogProduct",
    "COMPLEMENTARY_AFFINITY_RULES",
    "build_generic_retail_catalog",
    "load_catalog_dataframe",
    "ChronologicalSplitter",
    "load_baskets_dataframe",
    "load_customers_dataframe",
    "load_inventory_dataframe",
    "build_inventory_lookup",
]

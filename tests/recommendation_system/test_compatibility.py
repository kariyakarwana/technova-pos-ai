"""Unit tests for generic compatibility engine."""

from __future__ import annotations

import pandas as pd
import pytest

from technova_ai_service.features.recommendation_system.engine.compatibility import (
    CompatibilityEngine,
)


@pytest.fixture
def sample_catalog() -> pd.DataFrame:
    rows = [
        {
            "product_id": "CPU_1700",
            "category_level_1": "Components",
            "compatibility_tags": ["cpu", "socket_lga1700", "ddr5"],
        },
        {
            "product_id": "MOBO_1700",
            "category_level_1": "Components",
            "compatibility_tags": ["motherboard", "socket_lga1700", "ddr5", "atx"],
        },
        {
            "product_id": "RAM_DDR5",
            "category_level_1": "Components",
            "compatibility_tags": ["ram", "ddr5"],
        },
        {
            "product_id": "PHONE_15",
            "category_level_1": "Phones",
            "compatibility_tags": ["phone", "magsafe", "usb_c"],
        },
        {
            "product_id": "PHONE_CASE_MAG",
            "category_level_1": "Accessories",
            "compatibility_tags": ["case", "magsafe"],
        },
    ]
    return pd.DataFrame(rows)


def test_compatibility_engine(sample_catalog: pd.DataFrame) -> None:
    engine = CompatibilityEngine().fit(sample_catalog)

    # CPU_1700 matches MOBO_1700 and RAM_DDR5 on socket_lga1700 / ddr5
    compat_cpu = engine.find_compatible_products("CPU_1700", top_k=5)
    matched_ids = [c[0] for c in compat_cpu]

    assert "MOBO_1700" in matched_ids
    assert "RAM_DDR5" in matched_ids
    assert "PHONE_CASE_MAG" not in matched_ids

    # Phone matches MagSafe case
    compat_phone = engine.find_compatible_products("PHONE_15", top_k=5)
    matched_phone_ids = [c[0] for c in compat_phone]
    assert "PHONE_CASE_MAG" in matched_phone_ids

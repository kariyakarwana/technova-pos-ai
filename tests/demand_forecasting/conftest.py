"""Self-contained demand-forecast artifact for inference tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from technova_ai_service.config import get_settings
from technova_ai_service.features.demand_forecasting import inference, service
from technova_ai_service.features.demand_forecasting.schemas import (
    MODEL_FEATURE_COLUMNS,
)


def _build_demand_artifact(path: Path) -> None:
    categories = [
        "Beverages",
        "Health",
        "Household Cleaning",
        "Packaged Foods",
        "Personal Care",
    ]
    state_holidays = ["0", "a", "b", "c"]
    store_types = ["a", "c", "d"]
    assortments = ["a", "c"]

    rows: list[dict[str, object]] = []
    targets: list[float] = []
    for index in range(600):
        category = categories[index % len(categories)]
        is_promo = index % 2
        base_price = 10.0 + float(index % 7)
        unit_price = base_price * (0.8 if is_promo else 1.0)
        is_open = 0 if index % 31 == 0 else 1
        units = (
            0.0
            if is_open == 0
            else 8.0
            + float(index % len(categories))
            + float(is_promo) * 9.0
            + ((base_price - unit_price) / base_price) * 20.0
        )
        rows.append(
            {
                "category": category,
                "unit_price": unit_price,
                "base_unit_price": base_price,
                "is_open": is_open,
                "is_promo": is_promo,
                "promo2": index % 3 == 0,
                "day_of_week": index % 7,
                "state_holiday": state_holidays[index % len(state_holidays)],
                "school_holiday": index % 11 == 0,
                "store_type": store_types[index % len(store_types)],
                "assortment": assortments[index % len(assortments)],
            }
        )
        targets.append(float(np.log1p(units)))

    frame = pd.DataFrame(rows)
    categorical_categories = {
        "category": categories,
        "state_holiday": state_holidays,
        "store_type": store_types,
        "assortment": assortments,
    }
    for column, levels in categorical_categories.items():
        frame[column] = pd.Categorical(frame[column], categories=levels)

    model = LGBMRegressor(
        n_estimators=80,
        learning_rate=0.08,
        max_depth=5,
        random_state=42,
        verbosity=-1,
    )
    model.fit(frame[MODEL_FEATURE_COLUMNS], np.asarray(targets))

    path.parent.mkdir(parents=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "categorical_categories": categorical_categories,
            "model_type": "lightgbm",
            "target": "units_sold",
            "version": "test-fixture",
        },
        path,
    )


@pytest.fixture(scope="session", autouse=True)
def isolated_demand_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Point demand inference at a deterministic temporary LightGBM model."""
    artifact_root = tmp_path_factory.mktemp("demand_artifacts")
    model_path = artifact_root / "demand_forecasting" / "model.joblib"
    _build_demand_artifact(model_path)

    settings = get_settings()
    patcher = pytest.MonkeyPatch()
    patcher.setattr(settings, "artifact_dir", artifact_root)
    inference.load_demand_forecast_bundle.cache_clear()
    service._bundle_cache = None
    yield
    inference.load_demand_forecast_bundle.cache_clear()
    service._bundle_cache = None
    patcher.undo()

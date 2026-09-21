import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def save_artifact(
    directory: Path,
    *,
    bundle: dict[str, Any],
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.joblib"
    joblib.dump(bundle, model_path)
    enriched_metadata = {
        **metadata,
        "trained_at": datetime.now(UTC).isoformat(),
        "artifact": model_path.name,
    }
    (directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (directory / "metadata.json").write_text(
        json.dumps(enriched_metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return model_path


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run scripts/train_all.py first."
        )
    bundle: dict[str, Any] = joblib.load(path)
    return bundle

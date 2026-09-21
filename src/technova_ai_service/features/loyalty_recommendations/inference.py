from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from technova_ai_service.modeling.artifacts import load_artifact


@lru_cache(maxsize=4)
def load_loyalty_bundle(path: str) -> dict[str, Any]:
    return load_artifact(Path(path))


def recommend_products(*, artifact_path: Path, customer_id: str, top_k: int) -> dict[str, Any]:
    bundle = load_loyalty_bundle(str(artifact_path))
    product_ids: list[str] = bundle["product_ids"]
    customer_index: dict[str, int] = bundle["customer_index"]
    popularity: np.ndarray = bundle["popularity"]

    if customer_id in customer_index:
        customer_row = customer_index[customer_id]
        scores = (
            bundle["customer_factors"][customer_row] @ bundle["item_factors"].T + 0.10 * popularity
        )
        scores[bundle["seen_interactions"].getrow(customer_row).indices] = -np.inf
        strategy = "personalized_purchase_history"
    else:
        scores = popularity.copy()
        strategy = "popular_products_for_new_customer"

    available = min(top_k, len(product_ids))
    candidate_indices = np.argpartition(scores, -available)[-available:]
    ordered = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    finite_scores = scores[np.isfinite(scores)]
    score_min = float(finite_scores.min()) if finite_scores.size else 0.0
    score_range = max(float(finite_scores.max()) - score_min, 1e-9) if finite_scores.size else 1.0
    descriptions: dict[str, str] = bundle["descriptions"]
    recommendations = [
        {
            "product_id": product_ids[int(index)],
            "description": descriptions.get(product_ids[int(index)], "Unknown product"),
            "score": float((scores[int(index)] - score_min) / score_range),
            "reason": strategy,
        }
        for index in ordered
        if np.isfinite(scores[int(index)])
    ]
    profile = bundle["profiles"].get(
        customer_id,
        {
            "segment": "new_customer",
            "loyalty_score": 0.0,
            "recency_days": 0,
            "order_count": 0,
            "total_spend": 0.0,
            "units_purchased": 0.0,
        },
    )
    return {"customer_id": customer_id, "profile": profile, "recommendations": recommendations}

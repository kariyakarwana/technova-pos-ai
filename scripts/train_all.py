import argparse
import json
from pathlib import Path
from typing import Any

from technova_ai_service.features.dynamic_pricing.training import train_pricing_model
from technova_ai_service.features.loyalty_recommendations.training import train_loyalty_model
from technova_ai_service.features.stock_intelligence.training import train_stock_model


def train_all(
    input_path: Path,
    artifact_root: Path,
    *,
    top_products: int,
    recommendation_products: int,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_path}. Run scripts/download_online_retail.py first."
        )
    results = {
        "stock_intelligence": train_stock_model(
            input_path,
            artifact_root / "stock_intelligence",
            top_products=top_products,
        ),
        "dynamic_pricing": train_pricing_model(
            input_path,
            artifact_root / "dynamic_pricing",
            top_products=top_products,
        ),
        "loyalty_recommendations": train_loyalty_model(
            input_path,
            artifact_root / "loyalty_recommendations",
            max_products=recommendation_products,
        ),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all TechNova retail AI models.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/online_retail.csv"),
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--top-products", type=int, default=200)
    parser.add_argument("--recommendation-products", type=int, default=1000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller product sets for a fast local validation run.",
    )
    arguments = parser.parse_args()
    top_products = min(arguments.top_products, 50) if arguments.quick else arguments.top_products
    recommendation_products = (
        min(arguments.recommendation_products, 250)
        if arguments.quick
        else arguments.recommendation_products
    )
    results = train_all(
        arguments.input,
        arguments.artifact_root,
        top_products=top_products,
        recommendation_products=recommendation_products,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

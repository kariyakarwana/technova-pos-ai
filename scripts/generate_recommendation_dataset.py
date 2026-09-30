#!/usr/bin/env python
"""Generate the Augmented Behavioral Synthetic Recommendation Dataset for TechNova POS."""

from __future__ import annotations

import pprint
import time
from pathlib import Path

from technova_ai_service.features.recommendation_system.evaluation.evaluator import (
    validate_recommendation_dataset,
)
from technova_ai_service.features.recommendation_system.training.pipeline import (
    generate_recommendation_dataset,
)


def main() -> None:
    train_path = Path("data/raw/rossmann/train.csv")
    store_path = Path("data/raw/rossmann/store.csv")
    output_dir = Path("data/processed/recommendation_system")

    print("=================================================================")
    print("TechNova AI Recommendation Engine — Dataset Generation")
    print("Augmented Behavioral Synthetic Recommendation Dataset")
    print(f"Rossmann Train Source: {train_path}")
    print(f"Rossmann Store Source: {store_path}")
    print(f"Output Directory: {output_dir}")
    print("=================================================================")

    t0 = time.time()
    result = generate_recommendation_dataset(
        rossmann_train_path=train_path,
        rossmann_store_path=store_path,
        output_dir=output_dir,
        organization_id="org_technova_default",
        n_customers=500,
        selected_stores=[1, 2, 3, 4, 7, 8, 9, 10, 11, 13],
        date_start="2014-06-01",
        date_end="2014-12-31",
        random_seed=42,
    )
    t_gen = time.time() - t0

    print(f"\nGeneration completed in {t_gen:.2f} seconds.")
    print("\nParquet Output Files:")
    for name, path_str in result["paths"].items():
        size_kb = Path(path_str).stat().st_size / 1024
        print(f"  - {name}: {path_str} ({size_kb:.1f} KB)")

    print("\n--- Summary Metrics ---")
    pprint.pprint(result["summary"])

    print("\n--- Running Automated Validation Gates ---")
    val_report = validate_recommendation_dataset(output_dir)
    print(f"Validation Result: {'PASSED' if val_report.is_valid else 'FAILED'}")
    print(f"Checks Passed: {val_report.checks_passed} / {val_report.total_checks}")

    if not val_report.is_valid:
        print("\nFailed Checks:")
        for check_name, check_data in val_report.validation_results.items():
            if not check_data.get("passed", False):
                print(f"  [X] {check_name}: {check_data}")
        raise RuntimeError("Validation gates failed for generated recommendation dataset!")

    print("\nDataset generation and validation successful!")


if __name__ == "__main__":
    main()

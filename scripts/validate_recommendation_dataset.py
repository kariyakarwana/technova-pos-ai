#!/usr/bin/env python
"""Validate the Augmented Behavioral Synthetic Recommendation Dataset for TechNova POS."""

from __future__ import annotations

import pprint
import sys
from pathlib import Path

from technova_ai_service.features.recommendation_system.evaluation.evaluator import (
    validate_recommendation_dataset,
)


def main() -> None:
    dataset_dir = Path("data/processed/recommendation_system")

    print("=================================================================")
    print("Validating TechNova AI Recommendation Dataset...")
    print(f"Dataset Path: {dataset_dir}")
    print("=================================================================")

    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist. Run generation script first.")
        sys.exit(1)

    report = validate_recommendation_dataset(dataset_dir)

    print(f"\nOverall Validation Status: {'PASSED' if report.is_valid else 'FAILED'}")
    print(f"Checks Passed: {report.checks_passed} / {report.total_checks}\n")

    print("--- Detailed Check Results ---")
    for check_name, check_data in report.validation_results.items():
        status = "[PASS]" if check_data.get("passed", False) else "[FAIL]"
        print(f"  {status} {check_name}")

    print("\n--- Key Dataset Metrics ---")
    pprint.pprint(report.metrics)

    if not report.is_valid:
        print("\nFailed Check Details:")
        for check_name, check_data in report.validation_results.items():
            if not check_data.get("passed", False):
                print(f"  FAILED: {check_name} -> {check_data}")
        sys.exit(1)


if __name__ == "__main__":
    main()

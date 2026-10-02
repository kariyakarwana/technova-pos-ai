#!/usr/bin/env python
from __future__ import annotations

import json
import time
from pathlib import Path

from technova_ai_service.features.demand_forecasting.augmented_generator import (
    generate_augmented_rossmann_dataset,
)
from technova_ai_service.features.demand_forecasting.validation import (
    run_all_validation_gates,
)


def main() -> None:
    train_path = Path("data/raw/rossmann/train.csv")
    store_path = Path("data/raw/rossmann/store.csv")
    output_parquet = Path("data/processed/demand_forecasting/augmented_rossmann_unit_demand.parquet")
    output_metadata = Path("data/processed/demand_forecasting/augmented_rossmann_unit_demand_metadata.json")

    print("=================================================================")
    print("Starting Augmented Rossmann Unit-Demand Dataset Generation...")
    print(f"Train Source: {train_path}")
    print(f"Store Source: {store_path}")
    print(f"Output Parquet: {output_parquet}")
    print(f"Output Metadata: {output_metadata}")
    print("=================================================================")

    metadata = generate_augmented_rossmann_dataset(
        train_path=train_path,
        store_path=store_path,
        output_parquet_path=output_parquet,
        output_metadata_path=output_metadata,
        batch_store_size=50,
        random_seed=42,
    )
    print("\n--- Dataset Generation Complete ---")
    print(f"Rows Written: {metadata['generated_row_count']:,}")
    print(f"Generation Time: {metadata['performance']['generation_time_seconds']}s")
    print(f"Peak Memory: {metadata['performance']['peak_memory_mb']} MB")
    print(f"File Size: {metadata['performance']['output_file_size_mb']} MB")

    print("\n--- Running Automated Validation Gates ---")
    t_val_start = time.time()
    # Validate on sample of 25 diverse stores for high statistical power & speed
    report = run_all_validation_gates(output_parquet, sample_stores=25)
    t_val = time.time() - t_val_start
    print(f"Validation completed in {t_val:.2f}s")
    print(f"All Gates Passed: {report.all_gates_passed}")

    print("\nGate 1 (Physical Integrity):", report.gate_1_physical_integrity)
    print("\nGate 2 (Behavioral Correlation):", {k: v for k, v in report.gate_2_behavioral_correlation.items() if k != "weekday_profile"})
    print("Weekday profile sample:", report.gate_2_behavioral_correlation.get("weekday_profile"))
    print("\nGate 3 (Promotional Effects):", report.gate_3_promotional_effects)
    print("\nGate 4 (Catalog Velocity):", report.gate_4_catalog_velocity)
    print("\nGate 5 (Panel Feasibility):", report.gate_5_panel_feasibility)

    # Update metadata with validation results
    validation_summary = {
        "all_gates_passed": report.all_gates_passed,
        "gate_1_physical_integrity": report.gate_1_physical_integrity,
        "gate_2_behavioral_correlation": report.gate_2_behavioral_correlation,
        "gate_3_promotional_effects": report.gate_3_promotional_effects,
        "gate_4_catalog_velocity": report.gate_4_catalog_velocity,
        "gate_5_panel_feasibility": report.gate_5_panel_feasibility,
    }
    metadata["validation_report"] = validation_summary

    with open(output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nMetadata updated with validation results.")
    print("=================================================================")


if __name__ == "__main__":
    main()

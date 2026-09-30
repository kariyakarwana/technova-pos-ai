#!/usr/bin/env python
"""Master Production ML Training Orchestrator for TechNova POS AI.

Single source of truth for training and evaluating active machine learning systems:
1. Sales Forecasting (Rossmann / TechNova-native XGBoost)
2. Demand Forecasting (Augmented Rossmann Unit-Demand LightGBM)
3. Recommendation System (FP-Growth, Item/Content Similarity, Bayesian Branch Popularity)

Usage:
    uv run python scripts/train_all.py
    uv run python scripts/train_all.py --only sales
    uv run python scripts/train_all.py --only demand
    uv run python scripts/train_all.py --only recommendation
    uv run python scripts/train_all.py --validate-only
    uv run python scripts/train_all.py --evaluate-only
    uv run python scripts/train_all.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pyarrow.parquet as pq

# Ensure stdout handles UTF-8 cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Project root directory reference
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SystemStatus:
    name: str
    status: str  # SUCCESS, FAILED, SKIPPED, PASS
    duration_seconds: float = 0.0
    dataset_path: str | None = None
    dataset_info: dict[str, Any] = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


# ----------------------------------------------------------------------
# 1. Environment & Prerequisite Validation
# ----------------------------------------------------------------------

def validate_prerequisites() -> tuple[bool, dict[str, Any]]:
    """Validates runtime environment, required packages, and directory write permissions."""
    errors: list[str] = []
    env_info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }

    # Package verification
    required_packages = ["pandas", "numpy", "pyarrow", "joblib", "sklearn", "lightgbm", "xgboost"]
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f"Required package '{pkg}' is not installed.")

    # Artifact directory structure verification / creation
    artifact_dirs = [
        PROJECT_ROOT / "artifacts" / "sales_forecasting",
        PROJECT_ROOT / "artifacts" / "demand_forecasting",
        PROJECT_ROOT / "artifacts" / "recommendation_system",
    ]
    for ad in artifact_dirs:
        try:
            ad.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = ad / ".write_test"
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            errors.append(f"Directory {ad} is not writable: {e}")

    is_valid = len(errors) == 0
    env_info["status"] = "PASS" if is_valid else "FAILED"
    if errors:
        env_info["errors"] = errors

    return is_valid, env_info


# ----------------------------------------------------------------------
# 2. Dataset Preparation & Validation Helpers
# ----------------------------------------------------------------------

def check_raw_rossmann_dataset() -> tuple[bool, dict[str, Any]]:
    """Verifies that the raw Rossmann Store Sales dataset is available."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "rossmann"
    required_files = ["train.csv", "store.csv", "test.csv"]
    missing = [f for f in required_files if not (raw_dir / f).exists()]

    if missing:
        error_msg = (
            "Rossmann dataset is missing.\n\n"
            "Expected:\n"
            "  data/raw/rossmann/train.csv\n"
            "  data/raw/rossmann/store.csv\n"
            "  data/raw/rossmann/test.csv\n\n"
            "Run:\n"
            "  uv run python scripts/download_rossmann.py"
        )
        return False, {"exists": False, "missing": missing, "error": error_msg}

    return True, {"exists": True, "path": str(raw_dir.relative_to(PROJECT_ROOT))}


def prepare_and_validate_sales_data(auto_prepare: bool = True) -> tuple[bool, dict[str, Any]]:
    """Checks or generates the Sales Forecasting dataset."""
    data_dir = PROJECT_ROOT / "data" / "interim"
    dataset_path = data_dir / "technova_sales_forecasting_dataset.parquet"

    if not dataset_path.exists() and auto_prepare:
        print("  Generating Sales Forecasting dataset from TechNova branch profiles...")
        from technova_ai_service.features.sales_forecasting.dataset import (
            build_and_save_technova_dataset,
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        build_and_save_technova_dataset(data_dir)

    if not dataset_path.exists():
        return False, {"path": str(dataset_path), "exists": False, "error": "Dataset missing"}

    pf = pq.ParquetFile(dataset_path)
    info = {
        "path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "row_count": pf.metadata.num_rows,
        "column_count": len(pf.schema.names),
        "columns": pf.schema.names,
        "size_bytes": dataset_path.stat().st_size,
    }
    return True, info


def prepare_and_validate_demand_data(auto_prepare: bool = True) -> tuple[bool, dict[str, Any]]:
    """Checks or generates the Augmented Rossmann Unit-Demand dataset."""
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "demand_forecasting"
        / "augmented_rossmann_unit_demand.parquet"
    )
    meta_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"

    # A Parquet writer can leave a valid-looking partial file after an interrupted
    # run. The metadata manifest is written only after generation succeeds, so
    # require both files before reusing the dataset.
    if (not dataset_path.exists() or not meta_path.exists()) and auto_prepare:
        raw_ok, raw_info = check_raw_rossmann_dataset()
        if not raw_ok:
            return False, {
                "path": str(dataset_path),
                "exists": False,
                "error": raw_info["error"],
            }

        from technova_ai_service.features.demand_forecasting.augmented_generator import (
            generate_augmented_rossmann_dataset,
        )
        print("  Generating Augmented Rossmann Unit-Demand dataset (this may take several minutes)...")
        raw_train = PROJECT_ROOT / "data" / "raw" / "rossmann" / "train.csv"
        raw_store = PROJECT_ROOT / "data" / "raw" / "rossmann" / "store.csv"
        configured_max_stores = int(
            os.environ.get("TECHNOVA_DEMAND_TRAINING_MAX_STORES", "25")
        )
        max_stores = configured_max_stores if configured_max_stores > 0 else None
        sample_description = str(max_stores) if max_stores is not None else "all"
        print(f"  Using {sample_description} Rossmann stores for demand training.")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        generate_augmented_rossmann_dataset(
            train_path=raw_train,
            store_path=raw_store,
            output_parquet_path=dataset_path,
            output_metadata_path=meta_path,
            batch_store_size=25,
            random_seed=42,
            max_stores=max_stores,
        )

    if not dataset_path.exists():
        return False, {
            "path": str(dataset_path.relative_to(PROJECT_ROOT) if dataset_path.is_relative_to(PROJECT_ROOT) else dataset_path),
            "exists": False,
            "error": "Dataset missing at data/processed/demand_forecasting/augmented_rossmann_unit_demand.parquet",
        }

    # Lightweight metadata verification via PyArrow without reading 50M rows into memory
    pf = pq.ParquetFile(dataset_path)
    total_rows = pf.metadata.num_rows
    n_products = 50
    n_stores = 1115
    date_range = "2013-01-01 to 2015-07-31"

    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            n_products = meta.get("n_products", n_products)
            n_stores = meta.get("n_stores", n_stores)
            dr = meta.get("date_range", {})
            if isinstance(dr, dict):
                date_range = f"{dr.get('start', 'N/A')} to {dr.get('end', 'N/A')}"

    info = {
        "path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "row_count": total_rows,
        "product_count": n_products,
        "store_count": n_stores,
        "date_range": date_range,
        "size_bytes": dataset_path.stat().st_size,
    }
    return True, info


def prepare_and_validate_recommendation_data(auto_prepare: bool = True) -> tuple[bool, dict[str, Any]]:
    """Checks or generates the recommendation system datasets and runs validation gates."""
    rec_dir = PROJECT_ROOT / "data" / "processed" / "recommendation_system"
    required_files = [
        "catalog_products.parquet",
        "customer_profiles.parquet",
        "transaction_baskets.parquet",
        "branch_inventory.parquet",
    ]

    missing = [f for f in required_files if not (rec_dir / f).exists()]

    if missing and auto_prepare:
        raw_ok, raw_info = check_raw_rossmann_dataset()
        if not raw_ok:
            return False, {"error": raw_info["error"]}

        from technova_ai_service.features.recommendation_system.training.pipeline import (
            generate_recommendation_dataset,
        )
        print(f"  Missing recommendation datasets ({missing}). Invoking generation pipeline...")
        raw_train = PROJECT_ROOT / "data" / "raw" / "rossmann" / "train.csv"
        raw_store = PROJECT_ROOT / "data" / "raw" / "rossmann" / "store.csv"
        rec_dir.mkdir(parents=True, exist_ok=True)
        generate_recommendation_dataset(
            rossmann_train_path=raw_train,
            rossmann_store_path=raw_store,
            output_dir=rec_dir,
        )

    missing_after = [f for f in required_files if not (rec_dir / f).exists()]
    if missing_after:
        return False, {"error": f"Required dataset files missing: {missing_after}"}

    # Run existing validation gate
    from technova_ai_service.features.recommendation_system.evaluation.evaluator import (
        validate_recommendation_dataset,
    )
    report = validate_recommendation_dataset(rec_dir)
    if not report.is_valid:
        return False, {"error": f"Recommendation dataset validation failed: {report.checks_failed}"}

    info = {
        "path": str(rec_dir.relative_to(PROJECT_ROOT)),
        "exists": True,
        "files": {
            f: {
                "rows": pq.ParquetFile(rec_dir / f).metadata.num_rows,
                "size_bytes": (rec_dir / f).stat().st_size,
            }
            for f in required_files
        },
        "validation": {
            "is_valid": report.is_valid,
            "checks_passed": report.checks_passed,
            "total_checks": report.total_checks,
        },
    }
    return True, info


# ----------------------------------------------------------------------
# 3. Direct ML Training & Orchestration Runners
# ----------------------------------------------------------------------

def run_sales_forecasting(eval_only: bool = False) -> SystemStatus:
    """Orchestrates Sales Forecasting dataset preparation, training, evaluation, and artifact loading."""
    t0 = time.time()
    system_status = SystemStatus(name="Sales Forecasting", status="FAILED")

    try:
        # 1. Dataset verification / preparation
        data_ok, data_info = prepare_and_validate_sales_data(auto_prepare=not eval_only)
        if not data_ok:
            raise RuntimeError(f"Sales dataset validation failed: {data_info.get('error')}")
        system_status.dataset_path = data_info["path"]
        system_status.dataset_info = data_info
        print("[OK] Dataset prepared and verified")

        art_dir = PROJECT_ROOT / "artifacts" / "sales_forecasting"
        model_file = art_dir / "model.joblib"
        meta_file = art_dir / "metadata.json"
        metrics_file = art_dir / "metrics.json"

        # 2. Training (if not eval_only)
        if not eval_only:
            from technova_ai_service.features.sales_forecasting.training import (
                train_sales_forecasting_model,
            )
            dataset_path = PROJECT_ROOT / "data" / "interim" / "technova_sales_forecasting_dataset.parquet"
            train_sales_forecasting_model(
                input_path=dataset_path,
                artifact_directory=art_dir,
                test_days=30,
                val_days=30,
            )
            print("[OK] Model trained")

        # 3. Artifact verification
        if not model_file.exists():
            raise FileNotFoundError(f"Trained model artifact not found at {model_file}")

        from technova_ai_service.modeling.artifacts import load_artifact
        bundle = load_artifact(model_file)
        if "model" not in bundle or "feature_columns" not in bundle:
            raise ValueError(f"Sales model artifact corrupted at {model_file}")
        if len(bundle["feature_columns"]) != 25:
            raise ValueError(f"Expected 25 feature columns, got {len(bundle['feature_columns'])}")

        system_status.artifact_paths = [
            str(p.relative_to(PROJECT_ROOT)) for p in [model_file, meta_file, metrics_file] if p.exists()
        ]
        print("[OK] Artifact verified")

        # 4. Evaluation collection
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file, encoding="utf-8") as f:
                metrics = json.load(f)
        system_status.metrics = metrics
        print("[OK] Evaluation completed")

        system_status.status = "SUCCESS"

    except Exception as e:  # noqa: BLE001 - isolate one failed training subsystem
        system_status.status = "FAILED"
        system_status.error_message = str(e)
        print(f"[FAIL] Sales Forecasting FAILED: {e}")

    system_status.duration_seconds = round(time.time() - t0, 2)
    return system_status


def run_demand_forecasting(eval_only: bool = False) -> SystemStatus:
    """Orchestrates Demand Forecasting dataset verification, training, evaluation, and artifact loading."""
    t0 = time.time()
    system_status = SystemStatus(name="Demand Forecasting", status="FAILED")

    try:
        # 1. Dataset verification / preparation
        data_ok, data_info = prepare_and_validate_demand_data(auto_prepare=not eval_only)
        if not data_ok:
            raise RuntimeError(f"Demand dataset verification failed: {data_info.get('error')}")
        system_status.dataset_path = data_info["path"]
        system_status.dataset_info = data_info
        print("[OK] Dataset ready")
        print("[OK] Validation passed")

        art_dir = PROJECT_ROOT / "artifacts" / "demand_forecasting"
        model_file = art_dir / "model.joblib"
        meta_file = art_dir / "model_metadata.json"

        # 2. Training (if not eval_only)
        if not eval_only:
            from technova_ai_service.features.demand_forecasting.training import (
                train_and_benchmark,
            )
            dataset_path = (
                PROJECT_ROOT
                / "data"
                / "processed"
                / "demand_forecasting"
                / "augmented_rossmann_unit_demand.parquet"
            )
            train_and_benchmark(
                dataset_path=dataset_path,
                output_dir=art_dir,
                candidate_models=("lightgbm",),
            )
            print("[OK] Model trained")

        # 3. Artifact verification
        if not model_file.exists():
            raise FileNotFoundError(f"Trained demand model artifact not found at {model_file}")

        bundle = joblib.load(model_file)
        if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
            raise ValueError(f"Demand model artifact missing expected keys at {model_file}")

        system_status.artifact_paths = [
            str(p.relative_to(PROJECT_ROOT)) for p in [model_file, meta_file] if p.exists()
        ]
        print("[OK] Artifact verified")

        # 4. Evaluation collection
        metrics = {}
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
                metrics = {
                    "model_type": meta.get("model_type"),
                    "selected_validation_metrics": meta.get("selected_model_validation_metrics", {}),
                    "final_holdout_test_metrics": meta.get("final_holdout_test_metrics", {}),
                }
        system_status.metrics = metrics
        print("[OK] Evaluation completed")

        system_status.status = "SUCCESS"

    except Exception as e:  # noqa: BLE001 - isolate one failed training subsystem
        system_status.status = "FAILED"
        system_status.error_message = str(e)
        print(f"[FAIL] Demand Forecasting FAILED: {e}")

    system_status.duration_seconds = round(time.time() - t0, 2)
    return system_status


def run_recommendation_system(eval_only: bool = False) -> SystemStatus:
    """Orchestrates Recommendation System dataset validation, training, evaluation, and artifact loading."""
    t0 = time.time()
    system_status = SystemStatus(name="Recommendation System", status="FAILED")

    try:
        # 1. Dataset verification / preparation
        data_ok, data_info = prepare_and_validate_recommendation_data(auto_prepare=not eval_only)
        if not data_ok:
            raise RuntimeError(f"Recommendation dataset validation failed: {data_info.get('error')}")
        system_status.dataset_path = data_info["path"]
        system_status.dataset_info = data_info
        print("[OK] Dataset ready")
        print("[OK] Validation passed")

        art_dir = PROJECT_ROOT / "artifacts" / "recommendation_system"
        rec_dir = PROJECT_ROOT / "data" / "processed" / "recommendation_system"

        # 2. Training (if not eval_only)
        if not eval_only:
            from technova_ai_service.features.recommendation_system.training.train import (
                train_recommendation_models,
            )
            train_recommendation_models(data_dir=rec_dir, artifacts_dir=art_dir, verbose=False)
            print("[OK] Models trained")

        # 3. Artifact verification
        required_artifacts = [
            art_dir / "association_rules.joblib",
            art_dir / "item_similarity.joblib",
            art_dir / "content_vectorizer.joblib",
            art_dir / "branch_popularity.joblib",
            art_dir / "recommendation_metadata.json",
        ]

        for p in required_artifacts:
            if not p.exists():
                raise FileNotFoundError(f"Recommendation artifact missing: {p}")

        from technova_ai_service.features.recommendation_system.inference.model_loader import (
            load_recommendation_bundle,
        )
        bundle = load_recommendation_bundle(artifacts_dir=art_dir)
        if bundle.fp_model is None or bundle.item_sim is None:
            raise ValueError("Recommendation bundle loading failed.")

        system_status.artifact_paths = [str(p.relative_to(PROJECT_ROOT)) for p in required_artifacts]
        print("[OK] Artifacts verified")

        # 4. Evaluation execution
        eval_script = PROJECT_ROOT / "scripts" / "evaluate_recommendation_system.py"
        res_eval = subprocess.run(
            [sys.executable, str(eval_script)], cwd=str(PROJECT_ROOT), check=False
        )
        if res_eval.returncode != 0:
            raise RuntimeError("Recommendation evaluation script failed with non-zero exit code.")

        meta_path = art_dir / "recommendation_metadata.json"
        metrics: dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                rec_meta = json.load(f)
                metrics["training_transactions"] = rec_meta.get("training_metadata", {}).get(
                    "training_transactions"
                )
                metrics["training_line_items"] = rec_meta.get("training_metadata", {}).get(
                    "training_line_items"
                )
                metrics["model_components"] = rec_meta.get("model_components", {})
        system_status.metrics = metrics
        print("[OK] Evaluation completed")

        system_status.status = "SUCCESS"

    except Exception as e:  # noqa: BLE001 - isolate one failed training subsystem
        system_status.status = "FAILED"
        system_status.error_message = str(e)
        print(f"[FAIL] Recommendation System FAILED: {e}")

    system_status.duration_seconds = round(time.time() - t0, 2)
    return system_status


# ----------------------------------------------------------------------
# 4. Manifest Generation
# ----------------------------------------------------------------------

def write_training_manifest(
    statuses: dict[str, SystemStatus],
    total_duration_seconds: float,
    overall_status: str,
    pipeline_mode: str,
) -> Path:
    """Generates artifacts/training_manifest.json containing purely non-sensitive metadata."""
    manifest_path = PROJECT_ROOT / "artifacts" / "training_manifest.json"

    manifest_data: dict[str, Any] = {
        "pipeline": "technova-pos-ai",
        "version": "1.0.0",
        "pipeline_mode": pipeline_mode,
        "trained_at": datetime.now(UTC).isoformat(),
        "total_duration_seconds": round(total_duration_seconds, 2),
        "status": overall_status,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "systems": {},
    }

    for key, s in statuses.items():
        manifest_data["systems"][key] = {
            "system_name": s.name,
            "status": s.status,
            "duration_seconds": s.duration_seconds,
            "dataset_path": s.dataset_path,
            "dataset_info": s.dataset_info,
            "artifact_paths": s.artifact_paths,
            "metrics": s.metrics,
            "error": s.error_message,
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, default=str)

    return manifest_path


# ----------------------------------------------------------------------
# 5. Master Orchestrator Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TechNova POS AI -- Master ML Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/train_all.py                  # Train and evaluate all active systems
  uv run python scripts/train_all.py --only sales     # Train only Sales Forecasting
  uv run python scripts/train_all.py --only demand    # Train only Demand Forecasting
  uv run python scripts/train_all.py --only recommendation # Train only Recommendation System
  uv run python scripts/train_all.py --validate-only  # Validate environments, datasets, and existing artifacts
  uv run python scripts/train_all.py --evaluate-only  # Run evaluation on existing artifacts without retraining
        """,
    )
    parser.add_argument(
        "--only",
        choices=["sales", "demand", "recommendation"],
        help="Restricts pipeline execution to a single specific ML system.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validates prerequisites, datasets, and existing artifacts without training.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Runs evaluation and verifies existing artifacts without training models.",
    )

    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("TechNova POS AI -- ML Training Pipeline")
    print("=" * 60)

    # 1. Environment & Prerequisite Validation
    env_ok, env_info = validate_prerequisites()
    if not env_ok:
        print("\nPrerequisites validation FAILED:")
        for err in env_info.get("errors", []):
            print(f"  - {err}")
        sys.exit(1)

    # 2. Validation-only mode
    if args.validate_only:
        print("\n[VALIDATION ONLY MODE]")
        print("Checking datasets and existing artifacts without model training...\n")

        v_raw_ok, v_raw_info = check_raw_rossmann_dataset()
        v_sales_ok, v_sales_info = prepare_and_validate_sales_data(auto_prepare=False)
        v_demand_ok, v_demand_info = prepare_and_validate_demand_data(auto_prepare=False)
        v_rec_ok, v_rec_info = prepare_and_validate_recommendation_data(auto_prepare=False)

        print("-" * 60)
        print("Dataset Validation Status:")
        print(f"  Raw Rossmann Dataset:    {'PASS' if v_raw_ok else 'MISSING'}")
        if v_raw_ok:
            print(f"    Path: {v_raw_info['path']}")
        else:
            print(f"    {v_raw_info.get('error')}")

        print(f"  Sales Forecasting:       {'PASS' if v_sales_ok else 'FAIL'}")
        if v_sales_ok:
            print(f"    Path: {v_sales_info['path']} ({v_sales_info['row_count']:,} rows)")
        else:
            print(f"    Error: {v_sales_info.get('error')}")

        print(f"  Demand Forecasting:      {'PASS' if v_demand_ok else 'FAIL'}")
        if v_demand_ok:
            print(f"    Path: {v_demand_info['path']} ({v_demand_info['row_count']:,} rows, {v_demand_info['product_count']} products)")
        else:
            print(f"    Error: {v_demand_info.get('error')}")

        print(f"  Recommendation System:   {'PASS' if v_rec_ok else 'FAIL'}")
        if v_rec_ok:
            print(f"    Path: {v_rec_info['path']} (4 tables, {v_rec_info['validation']['checks_passed']}/{v_rec_info['validation']['total_checks']} checks passed)")
        else:
            print(f"    Error: {v_rec_info.get('error')}")

        print("-" * 60)
        print("Artifact Validation Status:")
        sales_art_ok = (PROJECT_ROOT / "artifacts" / "sales_forecasting" / "model.joblib").exists()
        demand_art_ok = (PROJECT_ROOT / "artifacts" / "demand_forecasting" / "model.joblib").exists()
        rec_art_ok = (PROJECT_ROOT / "artifacts" / "recommendation_system" / "association_rules.joblib").exists()

        print(f"  Sales Artifacts:         {'PASS' if sales_art_ok else 'MISSING'}")
        print(f"  Demand Artifacts:        {'PASS' if demand_art_ok else 'MISSING'}")
        print(f"  Recommendation Artifacts:{'PASS' if rec_art_ok else 'MISSING'}")
        print("-" * 60)

        all_valid = v_sales_ok and v_demand_ok and v_rec_ok and sales_art_ok and demand_art_ok and rec_art_ok
        print(f"\nOverall Validation: {'SUCCESS' if all_valid else 'FAILED'}")
        sys.exit(0 if all_valid else 1)

    # 3. Execution list determination
    systems_to_run = ["sales", "demand", "recommendation"]
    if args.only:
        systems_to_run = [args.only]

    statuses: dict[str, SystemStatus] = {}
    total_systems = len(systems_to_run)

    for idx, sys_key in enumerate(systems_to_run, 1):
        if sys_key == "sales":
            print(f"\n[{idx}/{total_systems}] Sales Forecasting")
            print("-" * 60)
            status = run_sales_forecasting(eval_only=args.evaluate_only)
            statuses["sales_forecasting"] = status
            if status.status == "FAILED":
                break

        elif sys_key == "demand":
            print(f"\n[{idx}/{total_systems}] Demand Forecasting")
            print("-" * 60)
            status = run_demand_forecasting(eval_only=args.evaluate_only)
            statuses["demand_forecasting"] = status
            if status.status == "FAILED":
                break

        elif sys_key == "recommendation":
            print(f"\n[{idx}/{total_systems}] Recommendation System")
            print("-" * 60)
            status = run_recommendation_system(eval_only=args.evaluate_only)
            statuses["recommendation_system"] = status
            if status.status == "FAILED":
                break

    t_total = time.time() - t_start
    overall_status = "SUCCESS" if all(s.status == "SUCCESS" for s in statuses.values()) else "FAILED"

    # 4. Generate Training Manifest
    pipeline_mode = "evaluate_only" if args.evaluate_only else "train"
    manifest_file = write_training_manifest(
        statuses=statuses,
        total_duration_seconds=t_total,
        overall_status=overall_status,
        pipeline_mode=pipeline_mode,
    )

    # 5. Training Summary Table
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for sys_key, s in statuses.items():
        dur_str = f"({s.duration_seconds:.1f}s)"
        print(f"{s.name:<25} {s.status:<10} {dur_str:>10}")

    mins, secs = divmod(int(t_total), 60)
    print(f"\nTotal Duration: {mins}m {secs:02d}s")
    print(f"Manifest written to: {manifest_file.relative_to(PROJECT_ROOT)}")

    if overall_status == "SUCCESS":
        print("\nTraining completed successfully.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\nTraining FAILED. See error output above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

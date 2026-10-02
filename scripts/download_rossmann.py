#!/usr/bin/env python
"""Professional Rossmann Store Sales dataset download and setup utility.

Downloads the official Kaggle Rossmann Store Sales competition dataset,
extracts and validates the raw CSV files for TechNova POS AI ML pipelines.

Usage:
    uv run python scripts/download_rossmann.py
    uv run python scripts/download_rossmann.py --output-dir data/raw/rossmann
    uv run python scripts/download_rossmann.py --force
    uv run python scripts/download_rossmann.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

# Ensure stdout handles UTF-8 cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Official Kaggle Competition Identifier
KAGGLE_COMPETITION = "rossmann-store-sales"

# Expected Schema Columns
EXPECTED_COLUMNS = {
    "train.csv": [
        "Store",
        "DayOfWeek",
        "Date",
        "Sales",
        "Customers",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
    ],
    "store.csv": [
        "Store",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
    ],
    "test.csv": [
        "Store",
        "DayOfWeek",
        "Date",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
    ],
}


def get_kaggle_credentials() -> tuple[str | None, str | None]:
    """Retrieves Kaggle credentials from environment variables or ~/.kaggle/kaggle.json.

    Returns:
        tuple[username, key] or (None, None) if not configured.
        Never logs or exposes credentials.
    """
    # 1. Environment variables
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username.strip(), key.strip()

    # 2. Local config file ~/.kaggle/kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                u = data.get("username")
                k = data.get("key")
                if u and k:
                    return str(u).strip(), str(k).strip()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None, None

    return None, None


def download_from_kaggle(destination_dir: Path) -> None:
    """Downloads and extracts the Rossmann Store Sales dataset using official Kaggle mechanisms."""
    username, key = get_kaggle_credentials()

    # Check if kaggle CLI is available
    has_kaggle_cli = shutil.which("kaggle") is not None

    if not has_kaggle_cli and not (username and key):
        print("\nERROR: Kaggle credentials not configured.", file=sys.stderr)
        print(
            "\nTo download the Rossmann dataset, please configure your Kaggle credentials via one of:",
            file=sys.stderr,
        )
        print("  1. Setting environment variables:", file=sys.stderr)
        print("     KAGGLE_USERNAME=\"your-username\"", file=sys.stderr)
        print("     KAGGLE_KEY=\"your-api-key\"", file=sys.stderr)
        print("  2. Placing kaggle.json in your home directory:", file=sys.stderr)
        print(f"     {Path.home() / '.kaggle' / 'kaggle.json'}", file=sys.stderr)
        print(
            "\nObtain your Kaggle API key at: https://www.kaggle.com/settings -> 'Create New Token'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloading official Kaggle dataset: {KAGGLE_COMPETITION}...")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        download_success = False

        # Method 1: Kaggle CLI
        if has_kaggle_cli:
            try:
                cmd = ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(temp_dir)]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    download_success = True
            except OSError:
                download_success = False

        # Method 2: Direct Official Kaggle REST API
        if not download_success and username and key:
            try:
                import base64

                api_url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{KAGGLE_COMPETITION}"
                req = urllib.request.Request(api_url)
                auth_str = f"{username}:{key}"
                auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
                req.add_header("Authorization", f"Basic {auth_b64}")

                zip_out = temp_dir / f"{KAGGLE_COMPETITION}.zip"
                with urllib.request.urlopen(req) as response, open(zip_out, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)
                download_success = True
            except OSError as e:
                print(f"ERROR: Direct Kaggle API download failed: {e}", file=sys.stderr)

        if not download_success:
            print(
                f"\nERROR: Failed to download dataset '{KAGGLE_COMPETITION}' from Kaggle.",
                file=sys.stderr,
            )
            print(
                "Verify your Kaggle account has accepted the competition rules at:\n"
                f"https://www.kaggle.com/c/{KAGGLE_COMPETITION}/rules",
                file=sys.stderr,
            )
            sys.exit(1)

        # Extract all downloaded files
        destination_dir.mkdir(parents=True, exist_ok=True)
        zip_files = list(temp_dir.glob("*.zip"))

        for zf in zip_files:
            with zipfile.ZipFile(zf, "r") as zip_ref:
                zip_ref.extractall(destination_dir)

        # Handle nested zip files if any (e.g. train.csv.zip)
        for nested_zip in destination_dir.glob("*.zip"):
            with zipfile.ZipFile(nested_zip, "r") as zip_ref:
                zip_ref.extractall(destination_dir)
            nested_zip.unlink(missing_ok=True)


def validate_rossmann_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Validates presence, readability, non-emptiness, and schema for the raw Rossmann CSV files."""
    results: dict[str, Any] = {
        "is_valid": True,
        "files": {},
        "errors": [],
    }

    required_files = ["train.csv", "store.csv", "test.csv"]
    all_check_files = ["train.csv", "store.csv", "test.csv", "sample_submission.csv"]

    # 1. Existence check
    for fname in required_files:
        fpath = dataset_dir / fname
        if not fpath.exists():
            results["is_valid"] = False
            results["errors"].append(f"Missing required file: {fname}")

    if not results["is_valid"]:
        return results

    # 2. Content, readability, and schema verification
    for fname in all_check_files:
        fpath = dataset_dir / fname
        if not fpath.exists():
            continue

        size_bytes = fpath.stat().st_size
        if size_bytes == 0:
            results["is_valid"] = False
            results["errors"].append(f"File {fname} is empty (0 bytes).")
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    results["is_valid"] = False
                    results["errors"].append(f"File {fname} has no header row.")
                    continue

                # Count rows
                row_count = sum(1 for _ in reader)

            if row_count == 0:
                results["is_valid"] = False
                results["errors"].append(f"File {fname} contains 0 data rows.")
                continue

            # Schema check
            if fname in EXPECTED_COLUMNS:
                expected = set(EXPECTED_COLUMNS[fname])
                actual = set(header)
                missing_cols = expected - actual
                if missing_cols:
                    results["is_valid"] = False
                    results["errors"].append(
                        f"File {fname} is missing expected columns: {sorted(missing_cols)}"
                    )

            results["files"][fname] = {
                "rows": row_count,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "columns": header,
            }

        except Exception as e:  # noqa: BLE001 - report every malformed dataset failure
            results["is_valid"] = False
            results["errors"].append(f"Could not read {fname}: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TechNova POS AI -- Rossmann Store Sales Dataset Downloader & Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/download_rossmann.py
  uv run python scripts/download_rossmann.py --force
  uv run python scripts/download_rossmann.py --output-dir data/raw/rossmann
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "rossmann",
        help="Target directory to place raw dataset files (default: data/raw/rossmann)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forces re-download even if dataset files already exist locally.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    print("=" * 60)
    print("TechNova POS AI -- Rossmann Dataset Setup")
    print("=" * 60)
    rel_out = output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir
    print(f"\nOutput:\n  {rel_out}/\n")

    required_files = ["train.csv", "store.csv", "test.csv"]
    all_exist = all((output_dir / f).exists() for f in required_files)

    if all_exist and not args.force:
        print("Rossmann dataset already exists.")
        print("Skipping download.\n")
    else:
        download_from_kaggle(output_dir)

    # Validate dataset
    print("Validation:")
    val_report = validate_rossmann_dataset(output_dir)

    if not val_report["is_valid"]:
        print("[FAIL] Validation failed:")
        for err in val_report["errors"]:
            print(f"  - {err}")
        print("\nDataset setup FAILED.")
        print("=" * 60)
        sys.exit(1)

    # Print summary
    for fname, info in val_report["files"].items():
        print(f"[OK] {fname} ({info['rows']:,} rows, {info['size_mb']} MB)")

    print("\n[OK] Required files present")
    print("[OK] CSV files readable")
    print("[OK] Required schema verified")

    print("\nRossmann dataset setup completed successfully.")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

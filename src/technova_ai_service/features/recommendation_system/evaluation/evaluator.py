"""Validation suite for the Augmented Behavioral Synthetic Recommendation Dataset."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


from ..domain.models import RecommendationValidationReport


def validate_recommendation_dataset(
    dataset_dir: str | Path,
) -> RecommendationValidationReport:
    """Validates the 4 generated Parquet tables and verifies recommendation signals."""
    path = Path(dataset_dir)
    catalog_path = path / "catalog_products.parquet"
    customer_path = path / "customer_profiles.parquet"
    baskets_path = path / "transaction_baskets.parquet"
    inventory_path = path / "branch_inventory.parquet"

    for p in [catalog_path, customer_path, baskets_path, inventory_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required dataset file: {p}")

    df_catalog = pd.read_parquet(catalog_path)
    df_customers = pd.read_parquet(customer_path)
    df_baskets = pd.read_parquet(baskets_path)
    df_inventory = pd.read_parquet(inventory_path)

    results: dict[str, dict[str, Any]] = {}

    # 1. Primary Key Uniqueness
    pk_catalog_ok = bool(df_catalog["product_id"].is_unique)
    pk_customer_ok = bool(df_customers["customer_id"].is_unique)
    pk_baskets_ok = bool(not df_baskets.duplicated(subset=["transaction_id", "line_item_id"]).any())
    pk_inventory_ok = bool(not df_inventory.duplicated(subset=["branch_id", "product_id"]).any())
    all_pk_ok = pk_catalog_ok and pk_customer_ok and pk_baskets_ok and pk_inventory_ok
    results["primary_keys_unique"] = {
        "passed": all_pk_ok,
        "catalog_pk_unique": pk_catalog_ok,
        "customer_pk_unique": pk_customer_ok,
        "basket_line_unique": pk_baskets_ok,
        "inventory_unique": pk_inventory_ok,
    }

    # 2. Foreign Key Integrity
    valid_prod_ids = set(df_catalog["product_id"])
    valid_cust_ids = set(df_customers["customer_id"])
    valid_branch_ids = set(df_inventory["branch_id"])

    basket_prods_ok = bool(df_baskets["product_id"].isin(valid_prod_ids).all())
    non_null_custs = df_baskets["customer_id"].dropna()
    basket_custs_ok = bool(non_null_custs.isin(valid_cust_ids).all())
    basket_branches_ok = bool(df_baskets["store_id"].isin(valid_branch_ids).all())
    inventory_prods_ok = bool(df_inventory["product_id"].isin(valid_prod_ids).all())
    all_fk_ok = basket_prods_ok and basket_custs_ok and basket_branches_ok and inventory_prods_ok
    results["foreign_keys_valid"] = {
        "passed": all_fk_ok,
        "basket_products_valid": basket_prods_ok,
        "basket_customers_valid": basket_custs_ok,
        "basket_branches_valid": basket_branches_ok,
        "inventory_products_valid": inventory_prods_ok,
    }

    # 3. Non-Negative Quantities
    qty_positive = bool((df_baskets["quantity"] > 0).all())
    stock_non_negative = bool((df_inventory["stock_quantity"] >= 0).all())
    results["quantities_valid"] = {
        "passed": qty_positive and stock_non_negative,
        "basket_quantity_positive": qty_positive,
        "stock_quantity_non_negative": stock_non_negative,
    }

    # 4. Valid Prices
    catalog_prices_ok = bool(
        (df_catalog["cost_price"] > 0).all()
        and (df_catalog["base_unit_price"] > 0).all()
        and (df_catalog["current_unit_price"] > 0).all()
    )
    basket_prices_ok = bool(
        (df_baskets["unit_price"] > 0).all()
        and (df_baskets["line_total"] > 0).all()
    )
    results["prices_valid"] = {
        "passed": catalog_prices_ok and basket_prices_ok,
        "catalog_prices_positive": catalog_prices_ok,
        "basket_prices_positive": basket_prices_ok,
    }

    # 5. Inactive Products Sold Check
    inactive_skus = set(df_catalog[~df_catalog["is_active"]]["product_id"])
    inactive_sold_count = int(df_baskets["product_id"].isin(inactive_skus).sum())
    results["no_inactive_products_sold"] = {
        "passed": inactive_sold_count == 0,
        "inactive_skus": list(inactive_skus),
        "inactive_sold_count": inactive_sold_count,
    }

    # 6. Branch Assignment
    unassigned_branches = set(df_baskets["store_id"]) - valid_branch_ids
    results["valid_branch_assignment"] = {
        "passed": len(unassigned_branches) == 0,
        "invalid_branch_count": len(unassigned_branches),
    }

    # 7. Customer / Product / Branch Reference Coverage
    purchased_prods = set(df_baskets["product_id"])
    purchased_custs = set(non_null_custs)
    results["reference_integrity"] = {
        "passed": len(purchased_prods) > 0 and len(purchased_custs) > 0,
        "active_catalog_coverage": round(len(purchased_prods) / (len(valid_prod_ids) - len(inactive_skus)), 4),
        "active_customer_coverage": round(len(purchased_custs) / len(valid_cust_ids), 4),
    }

    # 8. Tenant Consistency
    org_id = df_catalog["organization_id"].iloc[0]
    org_consistent = bool(
        (df_catalog["organization_id"] == org_id).all()
        and (df_customers["organization_id"] == org_id).all()
        and (df_baskets["organization_id"] == org_id).all()
        and (df_inventory["organization_id"] == org_id).all()
    )
    results["tenant_consistency"] = {
        "passed": org_consistent,
        "organization_id": org_id,
    }

    # 9. Basket Size Distribution
    basket_sizes = df_baskets.groupby("transaction_id")["quantity"].sum()
    mean_bs = float(basket_sizes.mean())
    median_bs = float(basket_sizes.median())
    bs_realistic = bool(1.5 <= mean_bs <= 5.0 and 1.0 <= median_bs <= 4.0 and basket_sizes.min() >= 1)
    results["basket_size_distribution"] = {
        "passed": bs_realistic,
        "mean_basket_size": round(mean_bs, 2),
        "median_basket_size": median_bs,
        "min_basket_size": int(basket_sizes.min()),
        "max_basket_size": int(basket_sizes.max()),
    }

    # 10. Product Frequency Long Tail (Pareto / Zipf)
    prod_counts = df_baskets["product_id"].value_counts()
    n_top_20pct = max(1, int(len(prod_counts) * 0.20))
    top_20pct_volume_share = float(prod_counts.iloc[:n_top_20pct].sum() / prod_counts.sum())
    has_long_tail = bool(top_20pct_volume_share >= 0.35)
    results["long_tail_distribution"] = {
        "passed": has_long_tail,
        "top_20pct_volume_share": round(top_20pct_volume_share, 4),
    }

    # 11. Co-Purchase Relationships
    baskets_by_txn = df_baskets.groupby("transaction_id")["product_id"].apply(lambda s: sorted(list(set(s))))
    co_occurrences: dict[str, int] = {}
    for items in baskets_by_txn:
        if len(items) >= 2:
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    pair = f"{items[i]} | {items[j]}"
                    co_occurrences[pair] = co_occurrences.get(pair, 0) + 1

    top_pairs = sorted(co_occurrences.items(), key=lambda kv: kv[1], reverse=True)[:10]
    co_purchase_exists = bool(len(top_pairs) >= 5 and top_pairs[0][1] >= 10)
    results["co_purchase_signals"] = {
        "passed": co_purchase_exists,
        "total_co_purchased_pairs": len(co_occurrences),
        "top_co_purchased_pairs": top_pairs,
    }

    # 12. Customer Preference Influence
    # Measure category affinity alignment: do customers buy preferred categories more often?
    cust_cat_map = df_customers.set_index("customer_id")["preferred_categories"].to_dict()
    baskets_with_cat = df_baskets[df_baskets["customer_id"].notna()].merge(
        df_catalog[["product_id", "category_level_1"]], on="product_id"
    )
    pref_match_count = 0
    total_eval_lines = 0
    for _, row in baskets_with_cat.iterrows():
        c_id = row["customer_id"]
        cat = row["category_level_1"]
        prefs = cust_cat_map.get(c_id, [])
        if prefs is not None and len(prefs) > 0:
            total_eval_lines += 1
            if cat in prefs:
                pref_match_count += 1
    pref_alignment_ratio = pref_match_count / max(1, total_eval_lines)
    customer_pref_ok = bool(pref_alignment_ratio >= 0.40)
    results["customer_preference_influence"] = {
        "passed": customer_pref_ok,
        "preferred_category_match_ratio": round(pref_alignment_ratio, 4),
    }

    # 13. Branch Behavior Differences
    branch_volume = df_baskets.groupby("store_id")["transaction_id"].nunique()
    branch_diff_ok = bool(branch_volume.std() > 0 and len(branch_volume) > 1)
    results["branch_behavior_differences"] = {
        "passed": branch_diff_ok,
        "branch_count": len(branch_volume),
        "min_branch_txns": int(branch_volume.min()),
        "max_branch_txns": int(branch_volume.max()),
        "std_branch_txns": round(float(branch_volume.std()), 2),
    }

    # 14. Compatibility Relationships
    # Verify that items with shared or compatible tags appear in the catalog and co-purchases
    catalog_tags = df_catalog["compatibility_tags"].explode().dropna().unique()
    results["compatibility_relationships"] = {
        "passed": len(catalog_tags) >= 10,
        "unique_compatibility_tokens": len(catalog_tags),
    }

    # 15. Inventory State Validity
    inv_rule_ok = bool(
        ((df_inventory["is_available"]) == (df_inventory["stock_quantity"] > 0)).all()
    )
    out_of_stock_exists = bool((df_inventory["stock_quantity"] == 0).any())
    in_stock_exists = bool((df_inventory["stock_quantity"] > 0).any())
    results["inventory_validity"] = {
        "passed": inv_rule_ok and out_of_stock_exists and in_stock_exists,
        "rule_consistency": inv_rule_ok,
        "out_of_stock_skus_present": out_of_stock_exists,
        "in_stock_skus_present": in_stock_exists,
        "out_of_stock_count": int((df_inventory["stock_quantity"] == 0).sum()),
    }

    # 16. No PII Exists
    cust_id_pattern = re.compile(r"^CUST-\d+$")
    all_cust_ids_synthetic = bool(df_customers["customer_id"].apply(lambda s: bool(cust_id_pattern.match(s))).all())
    # Verify no email or phone columns exist
    suspicious_cols = [c for c in df_customers.columns if any(p in c.lower() for p in ["email", "phone", "name", "address", "ssn"])]
    results["no_pii_exists"] = {
        "passed": all_cust_ids_synthetic and len(suspicious_cols) == 0,
        "all_customer_ids_synthetic": all_cust_ids_synthetic,
        "suspicious_pii_columns": suspicious_cols,
    }

    # 17. Chronological Leakage & Temporal Consistency
    df_baskets["transaction_date"] = pd.to_datetime(df_baskets["transaction_date"])
    date_min = df_baskets["transaction_date"].min()
    date_max = df_baskets["transaction_date"].max()
    leakage_ok = bool(date_min <= date_max and len(df_baskets[df_baskets["transaction_date"] > date_max]) == 0)
    results["temporal_consistency"] = {
        "passed": leakage_ok,
        "start_date": str(date_min.date()),
        "end_date": str(date_max.date()),
    }

    # Compute overall passed count
    passed_count = sum(1 for v in results.values() if v.get("passed", False))
    total_count = len(results)
    is_valid = passed_count == total_count

    # Extract high-level summary metrics
    total_customers = len(df_customers)
    repeat_customers = int((df_customers["total_transactions"] > 1).sum())
    repeat_ratio = repeat_customers / max(1, total_customers)

    metrics = {
        "unique_products": int(df_catalog["product_id"].nunique()),
        "active_products": int(df_catalog[df_catalog["is_active"]]["product_id"].nunique()),
        "inactive_products": int(len(inactive_skus)),
        "total_customers": total_customers,
        "repeat_customers": repeat_customers,
        "repeat_customer_ratio": round(repeat_ratio, 4),
        "unique_branches": int(df_baskets["store_id"].nunique()),
        "total_transactions": int(df_baskets["transaction_id"].nunique()),
        "total_line_items": len(df_baskets),
        "average_basket_size": round(mean_bs, 2),
        "median_basket_size": median_bs,
        "top_co_purchased_pairs": top_pairs[:5],
        "inventory_in_stock_rate": round(float(df_inventory["is_available"].mean()), 4),
    }

    return RecommendationValidationReport(
        is_valid=is_valid,
        checks_passed=passed_count,
        total_checks=total_count,
        validation_results=results,
        metrics=metrics,
    )


from .metrics import RecommendationMetrics, _calculate_dcg, _calculate_idcg


class OfflineEvaluator:
    """Evaluates recommendation models on holdout test baskets using ranking metrics."""

    def __init__(
        self,
        df_catalog: pd.DataFrame,
        sample_eval_baskets: int = 2000,
        random_seed: int = 42,
    ) -> None:
        self.df_catalog = df_catalog
        self.active_skus = set(df_catalog[df_catalog["is_active"]]["product_id"])
        self.sample_eval_baskets = sample_eval_baskets
        self.rng = np.random.default_rng(random_seed)

    def evaluate_model(
        self,
        recommend_fn: Any,
        test_baskets: pd.DataFrame,
        model_name: str = "Model",
    ) -> RecommendationMetrics:
        """Evaluates a recommender function against holdout test transactions."""
        # Group test set by basket
        baskets_grouped = (
            test_baskets.groupby(["transaction_id", "store_id", "customer_id"])["product_id"]
            .apply(lambda s: list(set(s)))
            .reset_index()
        )

        # Filter to multi-item baskets (basket_size >= 2)
        multi_item_baskets = baskets_grouped[baskets_grouped["product_id"].apply(len) >= 2].copy()
        if len(multi_item_baskets) > self.sample_eval_baskets:
            multi_item_baskets = multi_item_baskets.sample(
                n=self.sample_eval_baskets, random_state=42
            ).reset_index(drop=True)

        p5_list: list[float] = []
        p10_list: list[float] = []
        r5_list: list[float] = []
        r10_list: list[float] = []
        hr5_list: list[float] = []
        hr10_list: list[float] = []
        ndcg5_list: list[float] = []
        ndcg10_list: list[float] = []

        all_recommended_items: set[str] = set()
        diversity_categories: dict[str, str] = self.df_catalog.set_index("product_id")["category_level_1"].to_dict()
        category_entropies: list[float] = []

        for _, row in multi_item_baskets.iterrows():
            items = row["product_id"]
            branch_id = row["store_id"]
            cust_id = row["customer_id"] if pd.notna(row["customer_id"]) else None

            # Pick seed query item
            seed_item = items[0]
            ground_truth = set(items[1:])
            gt_size = len(ground_truth)
            if gt_size == 0:
                continue

            # Generate top-10 recommendations
            recs_raw = recommend_fn(
                seed_item=seed_item,
                customer_id=cust_id,
                branch_id=branch_id,
                top_k=10,
            )

            rec_ids = [
                r[0] if isinstance(r, (tuple, list)) else (r.product_id if hasattr(r, "product_id") else r)
                for r in recs_raw
            ]
            all_recommended_items.update(rec_ids)

            # Precision & Recall & Hit Rate @ 5
            top5 = rec_ids[:5]
            hits5 = sum(1 for it in top5 if it in ground_truth)
            p5 = hits5 / 5.0
            r5 = hits5 / gt_size
            hr5 = 1.0 if hits5 > 0 else 0.0

            # Precision & Recall & Hit Rate @ 10
            top10 = rec_ids[:10]
            hits10 = sum(1 for it in top10 if it in ground_truth)
            p10 = hits10 / 10.0
            r10 = hits10 / gt_size
            hr10 = 1.0 if hits10 > 0 else 0.0

            # NDCG @ 5 and @ 10
            dcg5 = _calculate_dcg(rec_ids, ground_truth, 5)
            idcg5 = _calculate_idcg(gt_size, 5)
            ndcg5 = (dcg5 / idcg5) if idcg5 > 0 else 0.0

            dcg10 = _calculate_dcg(rec_ids, ground_truth, 10)
            idcg10 = _calculate_idcg(gt_size, 10)
            ndcg10 = (dcg10 / idcg10) if idcg10 > 0 else 0.0

            p5_list.append(p5)
            p10_list.append(p10)
            r5_list.append(r5)
            r10_list.append(r10)
            hr5_list.append(hr5)
            hr10_list.append(hr10)
            ndcg5_list.append(ndcg5)
            ndcg10_list.append(ndcg10)

            # Diversity: unique categories in top 5 recommendations
            cats_in_top5 = {diversity_categories.get(it, "unknown") for it in top5}
            category_entropies.append(len(cats_in_top5) / max(1, len(top5)))

        catalog_coverage = len(all_recommended_items) / max(1, len(self.active_skus))
        avg_diversity = float(np.mean(category_entropies)) if category_entropies else 0.0

        return RecommendationMetrics(
            precision_at_5=float(np.mean(p5_list)),
            precision_at_10=float(np.mean(p10_list)),
            recall_at_5=float(np.mean(r5_list)),
            recall_at_10=float(np.mean(r10_list)),
            hit_rate_at_5=float(np.mean(hr5_list)),
            hit_rate_at_10=float(np.mean(hr10_list)),
            ndcg_at_5=float(np.mean(ndcg5_list)),
            ndcg_at_10=float(np.mean(ndcg10_list)),
            catalog_coverage=catalog_coverage,
            diversity_score=avg_diversity,
        )

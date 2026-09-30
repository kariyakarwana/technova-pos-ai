"""Augmented Behavioral Synthetic Recommendation Dataset Generator.

Combines real-world Rossmann store/footfall/temporal signals with a synthetic
multi-domain retail catalog, realistic customer archetypes, and co-purchase basket logic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.catalog import (
    COMPLEMENTARY_AFFINITY_RULES,
    CatalogProduct,
    build_generic_retail_catalog,
)


class RecommendationDatasetGenerator:
    """Generates the Augmented Behavioral Synthetic Recommendation Dataset."""

    def __init__(
        self,
        rossmann_train_path: str | Path,
        rossmann_store_path: str | Path,
        output_dir: str | Path,
        organization_id: str = "org_technova_default",
        n_customers: int = 500,
        selected_stores: list[int] | None = None,
        date_start: str = "2014-06-01",
        date_end: str = "2014-12-31",
        random_seed: int = 42,
    ) -> None:
        self.rossmann_train_path = Path(rossmann_train_path)
        self.rossmann_store_path = Path(rossmann_store_path)
        self.output_dir = Path(output_dir)
        self.organization_id = organization_id
        self.n_customers = n_customers
        self.date_start = date_start
        self.date_end = date_end
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Selected representative Rossmann stores (diverse StoreTypes: a, b, c, d)
        self.selected_stores = selected_stores or [1, 2, 3, 4, 7, 8, 9, 10, 11, 13]

    def generate(self) -> dict[str, Any]:
        """Executes full dataset generation pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build & Save Catalog
        catalog = build_generic_retail_catalog(organization_id=self.organization_id)
        df_catalog = pd.DataFrame([p.to_dict() for p in catalog])
        
        # 2. Build Customer Profiles
        df_customers = self._generate_customer_profiles(catalog)

        # 3. Load Rossmann Behavioral Signals
        df_rossmann = self._load_and_filter_rossmann()

        # 4. Generate Transaction Baskets
        df_baskets = self._generate_baskets(catalog, df_customers, df_rossmann)

        # 5. Update Customer Aggregated Metrics from Baskets
        df_customers = self._recompute_customer_metrics(df_customers, df_baskets)

        # 6. Generate Branch Inventory
        df_inventory = self._generate_branch_inventory(catalog)

        # 7. Write Parquet Files
        catalog_path = self.output_dir / "catalog_products.parquet"
        customer_path = self.output_dir / "customer_profiles.parquet"
        baskets_path = self.output_dir / "transaction_baskets.parquet"
        inventory_path = self.output_dir / "branch_inventory.parquet"

        df_catalog.to_parquet(catalog_path, index=False)
        df_customers.to_parquet(customer_path, index=False)
        df_baskets.to_parquet(baskets_path, index=False)
        df_inventory.to_parquet(inventory_path, index=False)

        # 8. Generate Summary & Metadata
        summary = self._create_summary(df_catalog, df_customers, df_baskets, df_inventory)
        metadata = self._create_metadata(summary)

        with open(self.output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        with open(self.output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        return {
            "summary": summary,
            "metadata": metadata,
            "paths": {
                "catalog": str(catalog_path),
                "customers": str(customer_path),
                "baskets": str(baskets_path),
                "inventory": str(inventory_path),
            },
        }

    def _generate_customer_profiles(self, catalog: list[CatalogProduct]) -> pd.DataFrame:
        """Generates anonymized synthetic customer personas with archetype affinities."""
        archetypes = [
            ("TECH_ENTHUSIAST", ["Computers & Electronics"], ["TechNova", "ApexGear", "SpeedCore", "Intel"], 0.2, 0.20),
            ("MOBILE_CONSUMER", ["Phones & Accessories"], ["NovaPhone", "NovaShield", "PowerVolt", "SoundPro"], 0.3, 0.25),
            ("GROCERY_FAMILY", ["Supermarket & Retail"], ["HeritageRoast", "AlpineSpring", "NutriHarvest", "MeadowFresh", "EcoCleanse"], 0.6, 0.30),
            ("OFFICE_PRO", ["Computers & Electronics", "Phones & Accessories"], ["TechNova", "LogiTechNova", "NovaCable", "UltraView"], 0.3, 0.15),
            ("GENERAL_RETAIL", ["Supermarket & Retail", "Phones & Accessories", "Computers & Electronics"], ["HeritageRoast", "NovaPhone", "TechNova"], 0.5, 0.10),
        ]

        records = []
        for i in range(1, self.n_customers + 1):
            customer_id = f"CUST-{i:04d}"
            customer_number = f"CN-{i:04d}"

            # Pick archetype
            arch_weights = [a[4] for a in archetypes]
            arch_idx = self.rng.choice(len(archetypes), p=arch_weights)
            archetype_name, pref_cats, aff_brands, base_sens, _ = archetypes[arch_idx]

            # Price sensitivity with variance
            price_sens = float(np.clip(self.rng.normal(base_sens, 0.08), 0.05, 0.95))

            # Segment
            segment_draw = self.rng.random()
            if segment_draw < 0.12:
                segment = "VIP"
            elif segment_draw < 0.45:
                segment = "REGULAR"
            elif segment_draw < 0.80:
                segment = "OCCASIONAL"
            else:
                segment = "AT_RISK"

            preferred_store_int = int(self.rng.choice(self.selected_stores))
            preferred_branch_id = f"BRANCH-{preferred_store_int:03d}"

            records.append({
                "customer_id": customer_id,
                "organization_id": self.organization_id,
                "customer_number": customer_number,
                "segment": segment,
                "archetype": archetype_name,
                "price_sensitivity": round(price_sens, 3),
                "preferred_branch_id": preferred_branch_id,
                "preferred_categories": pref_cats,
                "affinity_brands": aff_brands,
                "total_transactions": 0,
                "lifetime_spend": 0.0,
                "avg_basket_size": 0.0,
                "last_purchase_date": None,
            })

        return pd.DataFrame(records)

    def _load_and_filter_rossmann(self) -> pd.DataFrame:
        """Loads Rossmann store and train data, filtering to selected stores and dates."""
        df_stores = pd.read_csv(self.rossmann_store_path)
        df_train = pd.read_csv(self.rossmann_train_path, dtype={"StateHoliday": str})

        # Filter stores
        df_stores = df_stores[df_stores["Store"].isin(self.selected_stores)]
        df_train = df_train[df_train["Store"].isin(self.selected_stores)]

        # Filter dates & open days
        df_train["Date"] = pd.to_datetime(df_train["Date"])
        mask = (
            (df_train["Date"] >= pd.to_datetime(self.date_start))
            & (df_train["Date"] <= pd.to_datetime(self.date_end))
            & (df_train["Open"] == 1)
        )
        df_train = df_train[mask].sort_values(by=["Date", "Store"]).reset_index(drop=True)

        merged = df_train.merge(df_stores, on="Store", how="left")
        return merged

    def _generate_baskets(
        self,
        catalog: list[CatalogProduct],
        df_customers: pd.DataFrame,
        df_rossmann: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generates realistic transaction baskets driven by Rossmann footfall and temporal rhythms."""
        # Active products eligible for purchase
        active_products = [p for p in catalog if p.is_active]
        prod_map = {p.product_id: p for p in catalog}

        # Weight distribution for seed product selection
        prod_weights = np.array([p.popularity_weight for p in active_products], dtype=np.float64)
        prod_weights /= prod_weights.sum()

        cust_list = df_customers.to_dict("records")
        cust_by_store: dict[str, list[dict[str, Any]]] = {}
        for c in cust_list:
            b_id = c["preferred_branch_id"]
            cust_by_store.setdefault(b_id, []).append(c)

        basket_rows = []
        txn_counter = 1

        for _, row in df_rossmann.iterrows():
            store_id_int = int(row["Store"])
            branch_id = f"BRANCH-{store_id_int:03d}"
            date_val = row["Date"].date()
            customers_count = int(row["Customers"]) if not pd.isna(row["Customers"]) else 500
            promo_active = int(row["Promo"]) == 1

            # Determine number of baskets for this store-day
            # Scale footfall to realistic POS transaction sample (~10 to 35 baskets per day per branch)
            base_txns = max(4, int(customers_count * 0.025))
            if promo_active:
                base_txns = int(base_txns * 1.25)

            for _ in range(base_txns):
                txn_id = f"TXN-{txn_counter:06d}"
                txn_counter += 1

                # Customer assignment (65% loyalty member, 35% anonymous walk-in)
                is_loyalty = self.rng.random() < 0.65
                customer = None
                if is_loyalty:
                    store_candidates = cust_by_store.get(branch_id, cust_list)
                    if self.rng.random() < 0.70 and store_candidates:
                        customer = self.rng.choice(store_candidates)
                    else:
                        customer = self.rng.choice(cust_list)
                cust_id = customer["customer_id"] if customer else None

                # Basket size: geometric distribution (mean ~2.6, clamped between 1 and 6)
                basket_size = int(np.clip(self.rng.geometric(p=0.40), 1, 6))

                # Select Seed Product
                chosen_prod_ids: list[str] = []
                if customer:
                    # Filter products by customer's preferred category or brand
                    preferred_skus = [
                        p for p in active_products
                        if p.category_level_1 in customer["preferred_categories"]
                        or p.brand in customer["affinity_brands"]
                    ]
                    if preferred_skus and self.rng.random() < 0.75:
                        seed_weights = np.array([p.popularity_weight for p in preferred_skus], dtype=np.float64)
                        seed_weights /= seed_weights.sum()
                        seed_prod = self.rng.choice(preferred_skus, p=seed_weights)
                    else:
                        seed_prod = self.rng.choice(active_products, p=prod_weights)
                else:
                    seed_prod = self.rng.choice(active_products, p=prod_weights)

                chosen_prod_ids.append(seed_prod.product_id)

                # Generate Remaining Items (Co-Purchase / Complementary / Same-Category Logic)
                for _ in range(1, basket_size):
                    last_prod_id = chosen_prod_ids[-1]
                    rules = COMPLEMENTARY_AFFINITY_RULES.get(last_prod_id, [])

                    candidate_added = False
                    # 1. Try complementary rules with 70% probability if rules exist
                    if rules and self.rng.random() < 0.70:
                        rule_targets = [r[0] for r in rules if r[0] not in chosen_prod_ids and r[0] in prod_map and prod_map[r[0]].is_active]
                        rule_probs = [r[1] for r in rules if r[0] in rule_targets]
                        if rule_targets and sum(rule_probs) > 0:
                            rule_probs_norm = np.array(rule_probs, dtype=np.float64) / sum(rule_probs)
                            chosen_complement = self.rng.choice(rule_targets, p=rule_probs_norm)
                            chosen_prod_ids.append(chosen_complement)
                            candidate_added = True

                    # 2. Try same-category cross-sell with 20% probability
                    if not candidate_added and self.rng.random() < 0.65:
                        last_cat = prod_map[last_prod_id].category_level_1
                        cat_candidates = [
                            p.product_id for p in active_products
                            if p.category_level_1 == last_cat and p.product_id not in chosen_prod_ids
                        ]
                        if cat_candidates:
                            chosen_cat_item = self.rng.choice(cat_candidates)
                            chosen_prod_ids.append(chosen_cat_item)
                            candidate_added = True

                    # 3. Fallback: random exploratory active product
                    if not candidate_added:
                        available_skus = [p for p in active_products if p.product_id not in chosen_prod_ids]
                        if available_skus:
                            sub_weights = np.array([p.popularity_weight for p in available_skus], dtype=np.float64)
                            sub_weights /= sub_weights.sum()
                            chosen_sub = self.rng.choice(available_skus, p=sub_weights)
                            chosen_prod_ids.append(chosen_sub.product_id)

                # Timestamp generation (store open hours 09:00 - 20:30)
                random_second = int(self.rng.integers(0, 11 * 3600 + 30 * 60))
                txn_datetime = datetime.combine(date_val, time(9, 0)) + timedelta(seconds=random_second)

                # Create line items
                for line_idx, p_id in enumerate(chosen_prod_ids, start=1):
                    item = prod_map[p_id]
                    # Quantity: mostly 1 (82%), 2 (14%), 3 (4%)
                    q_draw = self.rng.random()
                    qty = 1 if q_draw < 0.82 else (2 if q_draw < 0.96 else 3)

                    # Promo pricing discount (10% off when promo active)
                    unit_price = item.current_unit_price
                    if promo_active:
                        unit_price = round(unit_price * 0.90, 2)
                    line_total = round(qty * unit_price, 2)

                    basket_rows.append({
                        "transaction_id": txn_id,
                        "organization_id": self.organization_id,
                        "store_id": branch_id,
                        "customer_id": cust_id,
                        "transaction_date": date_val,
                        "timestamp": txn_datetime,
                        "line_item_id": line_idx,
                        "product_id": p_id,
                        "quantity": qty,
                        "unit_price": unit_price,
                        "line_total": line_total,
                        "is_promo_applied": promo_active,
                    })

        return pd.DataFrame(basket_rows)

    def _recompute_customer_metrics(
        self,
        df_customers: pd.DataFrame,
        df_baskets: pd.DataFrame,
    ) -> pd.DataFrame:
        """Updates customer profiles with actual transaction counts, spend, and basket sizes."""
        cust_txns = df_baskets[df_baskets["customer_id"].notna()]
        if cust_txns.empty:
            return df_customers

        # Aggregate metrics
        txn_summary = (
            cust_txns.groupby(["customer_id", "transaction_id"])
            .agg(
                basket_items=("quantity", "sum"),
                basket_spend=("line_total", "sum"),
                txn_date=("transaction_date", "max"),
            )
            .reset_index()
        )

        cust_agg = (
            txn_summary.groupby("customer_id")
            .agg(
                total_transactions=("transaction_id", "count"),
                lifetime_spend=("basket_spend", "sum"),
                avg_basket_size=("basket_items", "mean"),
                last_purchase_date=("txn_date", "max"),
            )
            .reset_index()
        )

        df_merged = df_customers.drop(
            columns=["total_transactions", "lifetime_spend", "avg_basket_size", "last_purchase_date"]
        ).merge(cust_agg, on="customer_id", how="left")

        df_merged["total_transactions"] = df_merged["total_transactions"].fillna(0).astype(int)
        df_merged["lifetime_spend"] = df_merged["lifetime_spend"].fillna(0.0).round(2)
        df_merged["avg_basket_size"] = df_merged["avg_basket_size"].fillna(0.0).round(2)

        return df_merged

    def _generate_branch_inventory(
        self,
        catalog: list[CatalogProduct],
    ) -> pd.DataFrame:
        """Generates realistic branch-level inventory state across all branches and products."""
        records = []
        for store_int in self.selected_stores:
            branch_id = f"BRANCH-{store_int:03d}"
            for prod in catalog:
                # Inactive products always have 0 stock
                if not prod.is_active:
                    stock_qty = 0
                    is_avail = False
                    reorder = 0
                else:
                    # Intentionally designate ~8% of active SKUs out-of-stock for stock-aware validation
                    is_out_of_stock = self.rng.random() < 0.08
                    if is_out_of_stock:
                        stock_qty = 0
                        is_avail = False
                        reorder = 5
                    else:
                        # Realistic stock depth based on domain
                        if prod.domain == "computers":
                            stock_qty = int(self.rng.integers(3, 25))
                            reorder = 3
                        elif prod.domain == "phones":
                            stock_qty = int(self.rng.integers(8, 45))
                            reorder = 5
                        else:  # supermarket
                            stock_qty = int(self.rng.integers(20, 120))
                            reorder = 15
                        is_avail = True

                records.append({
                    "branch_id": branch_id,
                    "product_id": prod.product_id,
                    "organization_id": self.organization_id,
                    "stock_quantity": stock_qty,
                    "reorder_level": reorder,
                    "is_available": is_avail,
                })

        return pd.DataFrame(records)

    def _create_summary(
        self,
        df_catalog: pd.DataFrame,
        df_customers: pd.DataFrame,
        df_baskets: pd.DataFrame,
        df_inventory: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculates dataset summary metrics."""
        txn_grouped = df_baskets.groupby("transaction_id")
        basket_sizes = txn_grouped["quantity"].sum()

        top_products = df_baskets["product_id"].value_counts().head(10).to_dict()
        category_counts = (
            df_baskets.merge(df_catalog[["product_id", "category_level_1"]], on="product_id")["category_level_1"]
            .value_counts()
            .to_dict()
        )

        # Calculate top co-purchased pairs
        baskets_by_txn = (
            df_baskets.groupby("transaction_id")["product_id"]
            .apply(lambda s: sorted(set(s)))
        )
        pair_counts: dict[str, int] = {}
        for items in baskets_by_txn:
            if len(items) >= 2:
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        pair_key = f"{items[i]} | {items[j]}"
                        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        top_pairs = dict(sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:10])

        total_txns = int(df_baskets["transaction_id"].nunique())
        anonymous_txns = int(df_baskets[df_baskets["customer_id"].isna()]["transaction_id"].nunique())

        return {
            "entity_counts": {
                "unique_products": int(df_catalog["product_id"].nunique()),
                "active_products": int(df_catalog[df_catalog["is_active"]]["product_id"].nunique()),
                "inactive_products": int(df_catalog[~df_catalog["is_active"]]["product_id"].nunique()),
                "total_customers": int(df_customers["customer_id"].nunique()),
                "customers_with_purchases": int((df_customers["total_transactions"] > 0).sum()),
                "unique_branches": int(df_baskets["store_id"].nunique()),
                "total_transactions": total_txns,
                "total_line_items": len(df_baskets),
            },
            "basket_statistics": {
                "average_basket_size": round(float(basket_sizes.mean()), 2),
                "median_basket_size": float(basket_sizes.median()),
                "min_basket_size": int(basket_sizes.min()),
                "max_basket_size": int(basket_sizes.max()),
                "anonymous_transaction_ratio": round(anonymous_txns / max(1, total_txns), 4),
            },
            "inventory_statistics": {
                "total_records": len(df_inventory),
                "in_stock_records": int((df_inventory["is_available"]).sum()),
                "out_of_stock_records": int((~df_inventory["is_available"]).sum()),
                "out_of_stock_ratio": round(float((~df_inventory["is_available"]).mean()), 4),
            },
            "distributions": {
                "top_products": top_products,
                "category_sales_breakdown": category_counts,
                "top_co_purchased_pairs": top_pairs,
            },
            "temporal_range": {
                "start_date": str(df_baskets["transaction_date"].min()),
                "end_date": str(df_baskets["transaction_date"].max()),
            },
        }

    def _create_metadata(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Constructs provenance and formal metadata."""
        return {
            "dataset_name": "Augmented Behavioral Synthetic Recommendation Dataset",
            "version": "1.0.0",
            "provenance": {
                "rossmann_source_files": ["data/raw/rossmann/train.csv", "data/raw/rossmann/store.csv"],
                "rossmann_signals_used": [
                    "Store", "Date", "Customers", "Open", "Promo", "Promo2",
                    "StateHoliday", "SchoolHoliday", "StoreType", "Assortment"
                ],
                "synthetic_components": [
                    "Multi-domain catalog (Electronics, Phones, Supermarket)",
                    "Dynamic specifications JSON and compatibility tags",
                    "Customer persona archetypes with RFM loyalty distribution",
                    "Co-purchase and complementary product affinities",
                    "Branch inventory stock levels with out-of-stock test cases"
                ],
                "disclaimer": (
                    "This dataset is an Augmented Behavioral Synthetic Recommendation Dataset. "
                    "Footfall, temporal cycles, and promotional flags are anchored in real Rossmann retail patterns, "
                    "while product catalog, customer identities, and basket line items are synthetically generated. "
                    "It does not represent real TechNova customer transaction records."
                ),
            },
            "generation_parameters": {
                "organization_id": self.organization_id,
                "random_seed": self.random_seed,
                "selected_stores": self.selected_stores,
                "date_start": self.date_start,
                "date_end": self.date_end,
                "n_customers": self.n_customers,
            },
            "summary_metrics": summary,
            "generated_at": datetime.now(UTC).isoformat(),
        }


def generate_recommendation_dataset(
    rossmann_train_path: str | Path = "data/raw/rossmann/train.csv",
    rossmann_store_path: str | Path = "data/raw/rossmann/store.csv",
    output_dir: str | Path = "data/processed/recommendation_system",
    **kwargs: Any,
) -> dict[str, Any]:
    """Helper entry point for dataset generation."""
    generator = RecommendationDatasetGenerator(
        rossmann_train_path=rossmann_train_path,
        rossmann_store_path=rossmann_store_path,
        output_dir=output_dir,
        **kwargs,
    )
    return generator.generate()

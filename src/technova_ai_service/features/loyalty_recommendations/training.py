from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from technova_ai_service.data.online_retail import load_transactions
from technova_ai_service.modeling.artifacts import save_artifact


def _build_profiles(transactions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    reference_date = transactions["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    profile = (
        transactions.groupby("customer_id", observed=True)
        .agg(
            last_purchase=("invoice_date", "max"),
            order_count=("invoice_no", "nunique"),
            total_spend=("revenue", "sum"),
            units_purchased=("quantity", "sum"),
        )
        .reset_index()
    )
    profile["recency_days"] = (reference_date - profile["last_purchase"]).dt.days
    profile["frequency_rank"] = profile["order_count"].rank(pct=True)
    profile["spend_rank"] = profile["total_spend"].rank(pct=True)
    profile["recency_rank"] = 1.0 - profile["recency_days"].rank(pct=True)
    profile["loyalty_score"] = (
        0.35 * profile["frequency_rank"]
        + 0.45 * profile["spend_rank"]
        + 0.20 * profile["recency_rank"]
    ) * 100.0
    profile["segment"] = pd.cut(
        profile["loyalty_score"],
        bins=[-np.inf, 35, 60, 80, np.inf],
        labels=["new_or_at_risk", "regular", "loyal", "champion"],
    ).astype("string")
    return {
        str(row.customer_id): {
            "segment": str(row.segment),
            "loyalty_score": round(float(row.loyalty_score), 2),
            "recency_days": int(row.recency_days),
            "order_count": int(row.order_count),
            "total_spend": round(float(row.total_spend), 2),
            "units_purchased": round(float(row.units_purchased), 2),
        }
        for row in profile.itertuples(index=False)
    }


def train_loyalty_model(
    input_path: Path,
    artifact_directory: Path,
    *,
    max_products: int = 1000,
    min_customer_products: int = 3,
    components: int = 48,
) -> dict[str, Any]:
    transactions = load_transactions(input_path)
    transactions = transactions.loc[
        transactions["customer_id"].notna()
        & transactions["customer_id"].ne("<NA>")
        & transactions["customer_id"].ne("nan")
    ].copy()
    profile_transactions = transactions.copy()
    top_products = (
        transactions.groupby("product_id", observed=True)["quantity"]
        .sum()
        .nlargest(max_products)
        .index
    )
    transactions = transactions.loc[transactions["product_id"].isin(top_products)].copy()
    customer_counts = transactions.groupby("customer_id", observed=True)["product_id"].nunique()
    eligible_customers = customer_counts.loc[customer_counts.ge(min_customer_products)].index
    transactions = transactions.loc[transactions["customer_id"].isin(eligible_customers)].copy()

    last_rows = (
        transactions.sort_values("invoice_date").groupby("customer_id", observed=True).tail(1)
    )
    holdout = dict(
        zip(
            last_rows["customer_id"].astype(str),
            last_rows["product_id"].astype(str),
            strict=True,
        )
    )
    train_rows = transactions.drop(index=last_rows.index)
    interactions = (
        train_rows.groupby(["customer_id", "product_id"], observed=True)["quantity"]
        .sum()
        .reset_index()
    )

    customer_ids = sorted(interactions["customer_id"].astype(str).unique())
    product_ids = sorted(interactions["product_id"].astype(str).unique())
    customer_index = {customer_id: index for index, customer_id in enumerate(customer_ids)}
    product_index = {product_id: index for index, product_id in enumerate(product_ids)}
    rows = interactions["customer_id"].astype(str).map(customer_index).to_numpy()
    columns = interactions["product_id"].astype(str).map(product_index).to_numpy()
    values = np.log1p(interactions["quantity"].to_numpy(dtype=float))
    matrix = csr_matrix((values, (rows, columns)), shape=(len(customer_ids), len(product_ids)))

    component_count = max(2, min(components, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=component_count, n_iter=10, random_state=42)
    customer_factors = svd.fit_transform(matrix)
    item_factors = svd.components_.T
    popularity = np.asarray(matrix.sum(axis=0)).ravel()
    popularity = popularity / max(float(popularity.max()), 1.0)

    description_lookup = (
        transactions.sort_values("invoice_date")
        .groupby("product_id", observed=True)["description"]
        .last()
        .astype(str)
        .to_dict()
    )
    evaluated = 0
    hits = 0
    recommended_items: set[int] = set()
    for customer_id, held_product in list(holdout.items())[:1500]:
        if customer_id not in customer_index or held_product not in product_index:
            continue
        customer_row = customer_index[customer_id]
        scores = customer_factors[customer_row] @ item_factors.T + 0.10 * popularity
        scores[matrix.getrow(customer_row).indices] = -np.inf
        top = np.argpartition(scores, -10)[-10:]
        evaluated += 1
        hits += int(product_index[held_product] in top)
        recommended_items.update(int(value) for value in top)

    metrics = {
        "hit_rate_at_10": float(hits / max(evaluated, 1)),
        "catalog_coverage_at_10": float(len(recommended_items) / max(len(product_ids), 1)),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "evaluated_customers": float(evaluated),
    }
    model_path = save_artifact(
        artifact_directory,
        bundle={
            "customer_ids": customer_ids,
            "product_ids": product_ids,
            "customer_index": customer_index,
            "product_index": product_index,
            "customer_factors": customer_factors,
            "item_factors": item_factors,
            "popularity": popularity,
            "seen_interactions": matrix,
            "descriptions": description_lookup,
            "profiles": _build_profiles(profile_transactions),
        },
        metrics=metrics,
        metadata={
            "model_type": "TruncatedSVD hybrid recommender",
            "customers": len(customer_ids),
            "products": len(product_ids),
            "components": component_count,
            "minimum_customer_products": min_customer_products,
        },
    )
    return {"artifact": str(model_path), "metrics": metrics}

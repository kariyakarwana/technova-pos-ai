from datetime import date, datetime
from typing import Any

import pandas as pd


def _to_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    if isinstance(value, pd.Timestamp):
        return value.date()
    raise ValueError(f"Unsupported date format: {value}")


def build_daily_demand_panel(
    records: pd.DataFrame | list[dict[str, Any]],
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    closed_days: set[tuple[str, date]] | None = None,
    out_of_stock_days: set[tuple[str, str, date]] | None = None,
) -> pd.DataFrame:
    """Build a continuous, dense daily panel at the Product x Branch x Date grain.

    Guarantees:
    - Multi-tenant isolation: grouped strictly by organization_id.
    - Dense date series: every calendar date between start_date and end_date is present.
    - True zero-sales: missing sales on open, in-stock days are assigned quantity = 0.0.
    - Closed branch days: flagged with is_branch_open = 0.
    - Out of stock days: flagged with is_in_stock = 0 (censored demand).
    - Attribute preservation: product attributes (cost, selling price, category, etc.)
      are forward/backward-filled per SKU.
    """
    if isinstance(records, list):
        if not records:
            return pd.DataFrame(
                columns=[
                    "organization_id",
                    "branch_id",
                    "product_id",
                    "date",
                    "quantity",
                    "unit_price",
                    "cost_price",
                    "selling_price",
                    "reorder_level",
                    "category_id",
                    "brand_id",
                    "has_scheduled_discount",
                    "discount_rate",
                    "is_in_stock",
                    "is_branch_open",
                ]
            )
        df = pd.DataFrame(records)
    else:
        df = records.copy()

    if df.empty:
        return df

    # Normalize required identifiers
    required_cols = {"organization_id", "branch_id", "product_id", "date", "quantity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in records: {', '.join(sorted(missing))}")

    df["date"] = df["date"].apply(_to_date)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Optional columns with standard defaults
    if "unit_price" not in df.columns:
        df["unit_price"] = 0.0
    else:
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)

    if "cost_price" not in df.columns:
        df["cost_price"] = 0.0
    else:
        df["cost_price"] = pd.to_numeric(df["cost_price"], errors="coerce").fillna(0.0)

    if "selling_price" not in df.columns:
        df["selling_price"] = df["unit_price"]
    else:
        df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce").fillna(df["unit_price"])

    if "reorder_level" not in df.columns:
        df["reorder_level"] = 0.0
    else:
        df["reorder_level"] = pd.to_numeric(df["reorder_level"], errors="coerce").fillna(0.0)

    if "category_id" not in df.columns:
        df["category_id"] = None
    if "brand_id" not in df.columns:
        df["brand_id"] = None
    if "has_scheduled_discount" not in df.columns:
        df["has_scheduled_discount"] = 0
    else:
        df["has_scheduled_discount"] = df["has_scheduled_discount"].astype(int)

    if "discount_rate" not in df.columns:
        df["discount_rate"] = 0.0
    else:
        df["discount_rate"] = pd.to_numeric(df["discount_rate"], errors="coerce").fillna(0.0)

    # Aggregate multiple sales for the same (org, branch, product, date)
    agg_dict: dict[str, Any] = {
        "quantity": "sum",
        "unit_price": "median",
        "cost_price": "last",
        "selling_price": "last",
        "reorder_level": "last",
        "category_id": "first",
        "brand_id": "first",
        "has_scheduled_discount": "max",
        "discount_rate": "max",
    }
    grouped_sales = (
        df.groupby(["organization_id", "branch_id", "product_id", "date"], as_index=False)
        .agg(agg_dict)
    )

    # Determine date boundaries
    min_date = _to_date(start_date) if start_date else grouped_sales["date"].min()
    max_date = _to_date(end_date) if end_date else grouped_sales["date"].max()

    if min_date > max_date:
        raise ValueError(f"start_date ({min_date}) must not be after end_date ({max_date})")

    # Generate full date range
    all_dates = [
        date_item.date()
        for date_item in pd.date_range(min_date, max_date, freq="D")
    ]
    date_grid_df = pd.DataFrame({"date": all_dates})

    # Extract distinct series entities strictly grouped by organization_id
    entities = grouped_sales[
        ["organization_id", "branch_id", "product_id"]
    ].drop_duplicates()

    # Product-level static catalog attributes for back-filling zero days
    product_attributes = (
        grouped_sales.groupby(["organization_id", "product_id"], as_index=False)
        .agg(
            {
                "cost_price": "last",
                "selling_price": "last",
                "reorder_level": "last",
                "category_id": "first",
                "brand_id": "first",
            }
        )
    )

    # Cross join distinct entities with the dense date range
    dense_grid = entities.merge(date_grid_df, how="cross")

    # Left join observed aggregated sales onto dense grid
    panel = dense_grid.merge(
        grouped_sales,
        on=["organization_id", "branch_id", "product_id", "date"],
        how="left",
    )

    # Fill sales and promo columns
    panel["quantity"] = panel["quantity"].fillna(0.0)
    panel["has_scheduled_discount"] = panel["has_scheduled_discount"].fillna(0).astype(int)
    panel["discount_rate"] = panel["discount_rate"].fillna(0.0)

    # Fill catalog attributes from product_attributes for zero-sales days
    panel = panel.drop(
        columns=["cost_price", "selling_price", "reorder_level", "category_id", "brand_id"]
    ).merge(
        product_attributes,
        on=["organization_id", "product_id"],
        how="left",
    )

    # Unit price fallback to selling_price if no sale occurred
    panel["unit_price"] = panel["unit_price"].fillna(panel["selling_price"]).fillna(0.0)

    # Contextual flags: branch closed & out of stock
    closed_set = closed_days or set()
    oos_set = out_of_stock_days or set()

    # Vectorized / apply lookup for operating context
    panel["is_branch_open"] = [
        0 if (branch, dt) in closed_set else 1
        for branch, dt in zip(panel["branch_id"], panel["date"], strict=True)
    ]

    panel["is_in_stock"] = [
        0 if (branch, prod, dt) in oos_set else 1
        for branch, prod, dt in zip(panel["branch_id"], panel["product_id"], panel["date"], strict=True)
    ]

    # Closed branch days cannot produce realized sales
    panel.loc[panel["is_branch_open"] == 0, "quantity"] = 0.0

    # Sort strictly by organization, branch, product, and chronological date
    panel = panel.sort_values(
        ["organization_id", "branch_id", "product_id", "date"]
    ).reset_index(drop=True)

    return panel

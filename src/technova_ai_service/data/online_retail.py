from pathlib import Path

import numpy as np
import pandas as pd

NORMALIZED_COLUMNS = {
    "invoice_no",
    "product_id",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
}

UCI_COLUMN_MAP = {
    "InvoiceNo": "invoice_no",
    "StockCode": "product_id",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}


def load_transactions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, engine="openpyxl")
    elif suffix == ".csv":
        frame = pd.read_csv(
            path,
            dtype={
                "invoice_no": "string",
                "product_id": "string",
                "description": "string",
                "customer_id": "string",
                "country": "string",
            },
            low_memory=False,
        )
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported transaction file: {path}")

    frame = frame.rename(columns=UCI_COLUMN_MAP)
    missing = NORMALIZED_COLUMNS.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing transaction columns: {missing_list}")
    return clean_transactions(frame)


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.loc[:, sorted(NORMALIZED_COLUMNS)].copy()
    clean["invoice_no"] = clean["invoice_no"].astype("string").str.strip()
    clean["product_id"] = clean["product_id"].astype("string").str.strip()
    clean["description"] = clean["description"].astype("string").fillna("Unknown product")
    clean["customer_id"] = clean["customer_id"].astype("string")
    clean["country"] = clean["country"].astype("string").fillna("Unknown")
    clean["invoice_date"] = pd.to_datetime(clean["invoice_date"], errors="coerce")
    clean["quantity"] = pd.to_numeric(clean["quantity"], errors="coerce")
    clean["unit_price"] = pd.to_numeric(clean["unit_price"], errors="coerce")

    valid = (
        clean["invoice_date"].notna()
        & clean["product_id"].notna()
        & ~clean["invoice_no"].str.upper().str.startswith("C", na=False)
        & clean["quantity"].gt(0)
        & clean["unit_price"].gt(0)
    )
    clean = clean.loc[valid].copy()

    quantity_cap = float(clean["quantity"].quantile(0.995))
    price_cap = float(clean["unit_price"].quantile(0.995))
    clean = clean.loc[
        clean["quantity"].le(quantity_cap) & clean["unit_price"].le(price_cap)
    ].copy()
    clean["revenue"] = clean["quantity"] * clean["unit_price"]
    clean["date"] = clean["invoice_date"].dt.normalize()
    return clean.sort_values("invoice_date").reset_index(drop=True)


def build_daily_product_panel(
    transactions: pd.DataFrame,
    *,
    top_products: int = 200,
) -> pd.DataFrame:
    product_volume = transactions.groupby("product_id", observed=True)["quantity"].sum()
    product_ids = product_volume.nlargest(top_products).index
    selected = transactions.loc[transactions["product_id"].isin(product_ids)].copy()

    daily = (
        selected.groupby(["date", "product_id"], observed=True)
        .agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
            orders=("invoice_no", "nunique"),
            customers=("customer_id", "nunique"),
            description=("description", "last"),
        )
        .reset_index()
    )
    daily["unit_price"] = daily["revenue"] / daily["quantity"].clip(lower=1)

    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    index = pd.MultiIndex.from_product(
        [dates, product_ids.astype("string")], names=["date", "product_id"]
    )
    panel = daily.set_index(["date", "product_id"]).reindex(index).reset_index()
    panel[["quantity", "revenue", "orders", "customers"]] = panel[
        ["quantity", "revenue", "orders", "customers"]
    ].fillna(0.0)

    panel["description"] = panel.groupby("product_id", observed=True)["description"].transform(
        lambda values: values.ffill().bfill()
    )
    panel["unit_price"] = panel.groupby("product_id", observed=True)["unit_price"].transform(
        lambda values: values.ffill().bfill()
    )
    panel["unit_price"] = panel["unit_price"].fillna(
        float(selected["unit_price"].median())
    )
    return panel.sort_values(["product_id", "date"]).reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_dates = np.sort(frame["date"].dropna().unique())
    train_end = unique_dates[max(1, int(len(unique_dates) * train_fraction)) - 1]
    validation_end = unique_dates[
        max(2, int(len(unique_dates) * (train_fraction + validation_fraction))) - 1
    ]
    train = frame.loc[frame["date"].le(train_end)].copy()
    validation = frame.loc[
        frame["date"].gt(train_end) & frame["date"].le(validation_end)
    ].copy()
    test = frame.loc[frame["date"].gt(validation_end)].copy()
    return train, validation, test

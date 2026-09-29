from datetime import date, timedelta

import pandas as pd
import pytest

from technova_ai_service.features.demand_forecasting.panel import (
    build_daily_demand_panel,
)


def test_dense_date_panel_generation() -> None:
    start = date(2026, 1, 1)
    records = [
        {
            "organization_id": "org-1",
            "branch_id": "branch-A",
            "product_id": "sku-1",
            "date": start,
            "quantity": 5.0,
            "unit_price": 100.0,
        },
        {
            "organization_id": "org-1",
            "branch_id": "branch-A",
            "product_id": "sku-1",
            "date": start + timedelta(days=9),
            "quantity": 3.0,
            "unit_price": 100.0,
        },
    ]

    panel = build_daily_demand_panel(records)

    # 10 days inclusive (Jan 1 to Jan 10)
    assert len(panel) == 10
    dates = list(panel["date"])
    assert dates[0] == start
    assert dates[-1] == start + timedelta(days=9)
    for i in range(10):
        assert dates[i] == start + timedelta(days=i)


def test_zero_sales_handling() -> None:
    start = date(2026, 1, 1)
    records = [
        {
            "organization_id": "org-1",
            "branch_id": "branch-A",
            "product_id": "sku-1",
            "date": start,
            "quantity": 4.0,
            "unit_price": 50.0,
            "selling_price": 50.0,
            "cost_price": 30.0,
        },
        {
            "organization_id": "org-1",
            "branch_id": "branch-A",
            "product_id": "sku-1",
            "date": start + timedelta(days=2),
            "quantity": 6.0,
            "unit_price": 50.0,
            "selling_price": 50.0,
            "cost_price": 30.0,
        },
    ]

    panel = build_daily_demand_panel(records)

    assert len(panel) == 3
    middle_day = panel.iloc[1]
    assert middle_day["date"] == start + timedelta(days=1)
    assert middle_day["quantity"] == 0.0
    assert middle_day["is_in_stock"] == 1
    assert middle_day["is_branch_open"] == 1
    # Catalog price preserved on zero sales day
    assert middle_day["unit_price"] == 50.0
    assert middle_day["selling_price"] == 50.0
    assert middle_day["cost_price"] == 30.0


def test_out_of_stock_flag_indicates_censored_demand() -> None:
    start = date(2026, 1, 1)
    oos_date = start + timedelta(days=1)
    records = [
        {
            "organization_id": "org-1",
            "branch_id": "branch-A",
            "product_id": "sku-1",
            "date": start,
            "quantity": 2.0,
        },
        {
            "organization_id": "org-1",
            "branch_id": "branch-A",
            "product_id": "sku-1",
            "date": start + timedelta(days=2),
            "quantity": 0.0,
        },
    ]

    out_of_stock = {("branch-A", "sku-1", oos_date)}
    panel = build_daily_demand_panel(records, out_of_stock_days=out_of_stock)

    assert len(panel) == 3
    oos_row = panel.loc[panel["date"] == oos_date].iloc[0]
    assert oos_row["is_in_stock"] == 0
    assert oos_row["quantity"] == 0.0

    in_stock_row = panel.loc[panel["date"] == start].iloc[0]
    assert in_stock_row["is_in_stock"] == 1


def test_closed_day_handling() -> None:
    start = date(2026, 1, 1)
    closed_date = start + timedelta(days=1)
    records = [
        {
            "organization_id": "org-1",
            "branch_id": "branch-A",
            "product_id": "sku-1",
            "date": start,
            "quantity": 8.0,
        },
        {
            "organization_id": "org-1",
            "branch_id": "branch-A",
            "product_id": "sku-1",
            "date": closed_date,
            "quantity": 5.0,  # Simulated invalid transaction logged while closed
        },
    ]

    closed_days = {("branch-A", closed_date)}
    panel = build_daily_demand_panel(records, closed_days=closed_days)

    closed_row = panel.loc[panel["date"] == closed_date].iloc[0]
    assert closed_row["is_branch_open"] == 0
    assert closed_row["quantity"] == 0.0  # Forcefully zeroed on closed days


def test_tenant_and_branch_isolation() -> None:
    start = date(2026, 1, 1)
    records = [
        # Org 1, Branch A
        {
            "organization_id": "org-1",
            "branch_id": "branch-A",
            "product_id": "sku-1",
            "date": start,
            "quantity": 10.0,
        },
        # Org 1, Branch B
        {
            "organization_id": "org-1",
            "branch_id": "branch-B",
            "product_id": "sku-1",
            "date": start,
            "quantity": 20.0,
        },
        # Org 2, Branch A (Same branch code, different tenant)
        {
            "organization_id": "org-2",
            "branch_id": "branch-A",
            "product_id": "sku-1",
            "date": start,
            "quantity": 30.0,
        },
    ]

    panel = build_daily_demand_panel(records, end_date=start + timedelta(days=2))

    # 3 distinct series x 3 days = 9 rows total
    assert len(panel) == 9

    org1_a = panel.loc[(panel["organization_id"] == "org-1") & (panel["branch_id"] == "branch-A")]
    org1_b = panel.loc[(panel["organization_id"] == "org-1") & (panel["branch_id"] == "branch-B")]
    org2_a = panel.loc[(panel["organization_id"] == "org-2") & (panel["branch_id"] == "branch-A")]

    assert len(org1_a) == 3
    assert len(org1_b) == 3
    assert len(org2_a) == 3

    assert org1_a.iloc[0]["quantity"] == 10.0
    assert org1_b.iloc[0]["quantity"] == 20.0
    assert org2_a.iloc[0]["quantity"] == 30.0

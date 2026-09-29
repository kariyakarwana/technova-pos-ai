"""TechNova POS Sales Forecasting Dataset Generator.

Generates a realistic multi-branch, multi-year TechNova daily revenue dataset
and applies the Phase 5.3.3 feature engineering pipeline.

Guarantees:
- Strictly TechNova-native schema (Branch x Date -> daily_revenue).
- Excludes Rossmann artifacts (no StoreType, Customers, CompetitionDistance, etc.).
- Deterministic 25-feature output matching TECHNOVA_SALES_FEATURE_COLUMNS.
- Target is strictly daily_revenue.
- Pure point-in-time calculation with zero future leakage.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from technova_ai_service.features.sales_forecasting.training import (
    TECHNOVA_SALES_FEATURE_COLUMNS,
)

BRANCH_PROFILES = [
    {
        "organization_id": "org_technova_pos",
        "branch_id": "br_colombo_01",
        "branch_code": "COL-01",
        "branch_name": "Colombo Flagship",
        "base_revenue": 75000.0,
        "operates_sunday": True,
        "weekend_boost": 0.35,
    },
    {
        "organization_id": "org_technova_pos",
        "branch_id": "br_kandy_02",
        "branch_code": "KAN-02",
        "branch_name": "Kandy Central",
        "base_revenue": 48000.0,
        "operates_sunday": False,
        "weekend_boost": 0.25,
    },
    {
        "organization_id": "org_technova_pos",
        "branch_id": "br_galle_03",
        "branch_code": "GAL-03",
        "branch_name": "Galle Coastal",
        "base_revenue": 56000.0,
        "operates_sunday": True,
        "weekend_boost": 0.45,
    },
    {
        "organization_id": "org_technova_pos",
        "branch_id": "br_negombo_04",
        "branch_code": "NEG-04",
        "branch_name": "Negombo Hub",
        "base_revenue": 42000.0,
        "operates_sunday": False,
        "weekend_boost": 0.20,
    },
    {
        "organization_id": "org_technova_pos",
        "branch_id": "br_kurunegala_05",
        "branch_code": "KUR-05",
        "branch_name": "Kurunegala Branch",
        "base_revenue": 38000.0,
        "operates_sunday": False,
        "weekend_boost": 0.18,
    },
]


def generate_technova_daily_revenue_data(
    start_date: datetime.date = datetime.date(2024, 1, 1),
    end_date: datetime.date = datetime.date(2026, 8, 31),
    seed: int = 42,
) -> pd.DataFrame:
    """Generate raw Branch x Date daily revenue rows across branches."""
    rng = np.random.default_rng(seed)
    total_days = (end_date - start_date).days + 1
    dates = [start_date + datetime.timedelta(days=i) for i in range(total_days)]

    rows: list[dict[str, object]] = []

    for profile in BRANCH_PROFILES:
        base_rev = float(profile["base_revenue"])
        operates_sunday = bool(profile["operates_sunday"])
        weekend_boost = float(profile["weekend_boost"])
        branch_inception = start_date

        for cur_date in dates:
            day_of_week = cur_date.weekday()  # 0=Monday, 6=Sunday
            is_sunday = day_of_week == 6
            is_operating = 0 if (is_sunday and not operates_sunday) else 1

            if is_operating == 0:
                revenue = 0.0
                active_discounts = 0
            else:
                # Day-of-week factor
                dow_factors = [0.90, 0.95, 0.95, 1.05, 1.25, 1.30 + weekend_boost, 1.15 if operates_sunday else 0.0]
                dow_mult = dow_factors[day_of_week]

                # Month payday surge (25th to 31st)
                day_of_month = cur_date.day
                payday_mult = 1.25 if day_of_month >= 25 else (1.10 if day_of_month <= 3 else 1.0)

                # Monthly seasonality (higher in April Sinhala/Tamil New Year & Dec festive)
                month = cur_date.month
                season_mult = 1.30 if month in (4, 12) else (1.10 if month in (8, 11) else 1.0)

                # Periodic promotional campaign discount count (0 to 3)
                # Promotions cluster around weekends and month-ends
                is_promo_period = (cur_date.day % 14 in (5, 6, 7)) or (cur_date.day >= 26)
                active_discounts = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])) if is_promo_period else int(rng.choice([0, 1], p=[0.85, 0.15]))
                discount_mult = 1.0 + 0.12 * active_discounts

                # Noise
                noise = rng.normal(1.0, 0.08)
                revenue = max(0.0, base_rev * dow_mult * payday_mult * season_mult * discount_mult * noise)
                revenue = round(revenue, 2)

            rows.append({
                "organization_id": profile["organization_id"],
                "branch_id": profile["branch_id"],
                "branch_code": profile["branch_code"],
                "branch_name": profile["branch_name"],
                "date": cur_date.isoformat(),
                "daily_revenue": revenue,
                "is_operating_day": is_operating,
                "active_discount_count": active_discounts,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["branch_id", "date"]).reset_index(drop=True)
    return df


def engineer_technova_features(raw_df: pd.DataFrame, min_history_days: int = 28) -> pd.DataFrame:
    """Transform Branch x Date daily revenue rows into ML-ready feature dataset.

    Follows the exact specification of Phase 5.3.3:
    - 25 deterministic features in TECHNOVA_SALES_FEATURE_COLUMNS.
    - Zero data leakage (all lags and rollings shifted by 1 day or more).
    - Rows with insufficient history (< min_history_days) are removed.
    """
    df = raw_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["branch_id", "date"]).reset_index(drop=True)

    records: list[dict[str, object]] = []

    for (org_id, branch_id), group in df.groupby(["organization_id", "branch_id"], sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        revenues = group["daily_revenue"].to_numpy(dtype=float)
        n = len(group)
        branch_inception = group["date"].iloc[0]

        for i in range(n):
            if i < min_history_days:
                continue  # Skip warmup days to avoid null lag features

            cur_row = group.iloc[i]
            cur_date = cur_row["date"]

            # --- Calendar Features ---
            dow = cur_date.weekday()  # 0=Monday, 6=Sunday
            day_of_month = cur_date.day
            month = cur_date.month
            year = cur_date.year
            is_weekend = 1 if dow >= 5 else 0

            # ISO week
            week_of_year = cur_date.isocalendar().week

            # Month start/end
            is_month_start = 1 if day_of_month == 1 else 0
            days_in_month = pd.Period(cur_date.strftime("%Y-%m")).days_in_month
            is_month_end = 1 if day_of_month == days_in_month else 0

            # Cyclical encodings
            day_of_week_sin = round(math.sin((2 * math.pi * dow) / 7), 4)
            day_of_week_cos = round(math.cos((2 * math.pi * dow) / 7), 4)
            month_sin = round(math.sin((2 * math.pi * (month - 1)) / 12), 4)
            month_cos = round(math.cos((2 * math.pi * (month - 1)) / 12), 4)

            # --- Point-in-Time Lag Features (strictly <= i-1) ---
            rev_lag_1 = revenues[i - 1]
            rev_lag_7 = revenues[i - 7]
            rev_lag_14 = revenues[i - 14]
            rev_lag_28 = revenues[i - 28]

            # --- Point-in-Time Rolling Aggregations (strictly <= i-1) ---
            # 7-day rolling [i-7, i-1]
            w7 = revenues[i - 7 : i]
            rolling_mean_7 = float(np.mean(w7))
            rolling_std_7 = float(np.std(w7, ddof=1)) if len(w7) > 1 else 0.0

            # 14-day rolling [i-14, i-1]
            w14 = revenues[i - 14 : i]
            rolling_mean_14 = float(np.mean(w14))

            # 28-day rolling [i-28, i-1]
            w28 = revenues[i - 28 : i]
            rolling_mean_28 = float(np.mean(w28))
            rolling_std_28 = float(np.std(w28, ddof=1)) if len(w28) > 1 else 0.0

            # Weekly momentum ratio: [i-7, i-1] mean vs preceding 7 days [i-14, i-8]
            prev_w7 = revenues[i - 14 : i - 7]
            prev_mean_7 = float(np.mean(prev_w7))
            weekly_momentum_ratio = (
                round(rolling_mean_7 / prev_mean_7, 4) if prev_mean_7 > 0 else 1.0
            )

            # --- Branch Profile Features ---
            branch_age_days = (cur_date - branch_inception).days
            expanding_history = revenues[:i]
            branch_historical_avg_sales = float(np.mean(expanding_history))

            rec = {
                "organization_id": org_id,
                "branch_id": branch_id,
                "branch_code": cur_row["branch_code"],
                "branch_name": cur_row["branch_name"],
                "date": cur_date.strftime("%Y-%m-%d"),
                "daily_revenue": cur_row["daily_revenue"],
                # 25 ML Features in exact contract order
                "day_of_week": dow,
                "day_of_month": day_of_month,
                "month": month,
                "week_of_year": int(week_of_year),
                "is_weekend": is_weekend,
                "is_month_start": is_month_start,
                "is_month_end": is_month_end,
                "day_of_week_sin": day_of_week_sin,
                "day_of_week_cos": day_of_week_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "revenue_lag_1": rev_lag_1,
                "revenue_lag_7": rev_lag_7,
                "revenue_lag_14": rev_lag_14,
                "revenue_lag_28": rev_lag_28,
                "rolling_mean_7": round(rolling_mean_7, 2),
                "rolling_mean_14": round(rolling_mean_14, 2),
                "rolling_mean_28": round(rolling_mean_28, 2),
                "rolling_std_7": round(rolling_std_7, 2),
                "rolling_std_28": round(rolling_std_28, 2),
                "weekly_momentum_ratio": weekly_momentum_ratio,
                "branch_age_days": branch_age_days,
                "branch_historical_avg_sales": round(branch_historical_avg_sales, 2),
                "is_operating_day": int(cur_row["is_operating_day"]),
                "active_discount_count": int(cur_row["active_discount_count"]),
            }
            records.append(rec)

    feature_df = pd.DataFrame(records)
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    feature_df = feature_df.sort_values(["branch_id", "date"]).reset_index(drop=True)
    return feature_df


def build_and_save_technova_dataset(output_dir: Path) -> Path:
    """Build the TechNova daily revenue feature dataset and save as parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = generate_technova_daily_revenue_data()
    feature_df = engineer_technova_features(raw_df)

    parquet_path = output_dir / "technova_sales_forecasting_dataset.parquet"
    feature_df.to_parquet(parquet_path, index=False)
    csv_path = output_dir / "technova_sales_forecasting_dataset.csv"
    feature_df.to_csv(csv_path, index=False)

    return parquet_path

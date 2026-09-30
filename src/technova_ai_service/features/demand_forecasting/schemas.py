from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEMAND_LAG_HORIZONS: tuple[int, ...] = (1, 2, 3, 7, 14, 21, 28)
ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 28)

DEMAND_FEATURE_COLUMNS: list[str] = [
    # Demand Lags
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_3",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_21",
    "demand_lag_28",
    # Rolling Statistics
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_max_7",
    "rolling_max_14",
    "rolling_max_28",
    # Pricing & Valuation
    "unit_price",
    "price_index_28",
    "cost_price",
    "selling_price",
    "margin_rate",
    "reorder_level",
    # Scheduled Promotions (from DiscountRule schedule, never redemption)
    "has_scheduled_discount",
    "discount_rate",
    # Calendar Signals
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_payday_window",
    # Sri Lankan Holidays & Retail Events
    "is_public_holiday",
    "is_mercantile_holiday",
    "is_poya_day",
    "is_festive_peak",
    # Operating Context
    "is_in_stock",
    "is_branch_open",
]


class DailyDemandObservation(BaseModel):
    """Raw or aggregated daily demand observation from POS completed/partially-refunded sales."""

    model_config = ConfigDict(frozen=True)

    organization_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    date: dt.date
    quantity: float = Field(ge=0.0)
    unit_price: float = Field(default=0.0, ge=0.0)
    cost_price: float = Field(default=0.0, ge=0.0)
    selling_price: float = Field(default=0.0, ge=0.0)
    reorder_level: float = Field(default=0.0, ge=0.0)
    category_id: str | None = None
    brand_id: str | None = None
    is_in_stock: bool = True
    is_branch_open: bool = True
    has_scheduled_discount: bool = False
    discount_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    discount_type: str | None = None


class DemandForecastProductInput(BaseModel):
    """Product-level input payload for demand forecasting inference."""

    model_config = ConfigDict(frozen=True)

    product_id: str = Field(min_length=1)
    sku: str | None = None
    category_id: str | None = None
    brand_id: str | None = None
    cost_price: float = Field(default=0.0, ge=0.0)
    selling_price: float = Field(default=0.0, ge=0.0)
    reorder_level: float = Field(default=0.0, ge=0.0)
    current_stock: float = Field(default=0.0, ge=0.0)
    recent_daily_demand: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_demand_history(self) -> DemandForecastProductInput:
        if any(value < 0 for value in self.recent_daily_demand):
            raise ValueError("recent_daily_demand entries must be non-negative")
        return self


class FutureCalendarDayInput(BaseModel):
    """Calendar and planned promotion input for future prediction dates."""

    model_config = ConfigDict(frozen=True)

    date: dt.date
    day_of_week: int | None = Field(default=None, ge=0, le=6)
    is_weekend: bool | None = None
    is_public_holiday: bool | None = None
    is_mercantile_holiday: bool | None = None
    is_poya_day: bool | None = None
    is_festive_peak: bool | None = None
    has_discount: bool = False
    discount_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    discount_type: str | None = None


class DemandFeatureRow(BaseModel):
    """Model-ready feature row matching DEMAND_FEATURE_COLUMNS."""

    model_config = ConfigDict(extra="forbid")

    # Demand Lags
    demand_lag_1: float
    demand_lag_2: float
    demand_lag_3: float
    demand_lag_7: float
    demand_lag_14: float
    demand_lag_21: float
    demand_lag_28: float

    # Rolling Statistics
    rolling_mean_7: float
    rolling_mean_14: float
    rolling_mean_28: float
    rolling_std_7: float
    rolling_std_14: float
    rolling_std_28: float
    rolling_max_7: float
    rolling_max_14: float
    rolling_max_28: float

    # Pricing & Valuation
    unit_price: float
    price_index_28: float
    cost_price: float
    selling_price: float
    margin_rate: float
    reorder_level: float

    # Scheduled Promotions
    has_scheduled_discount: int
    discount_rate: float

    # Calendar Signals
    day_of_week: int
    day_of_month: int
    month: int
    is_weekend: int
    is_month_start: int
    is_month_end: int
    is_payday_window: int

    # Sri Lankan Holidays & Retail Events
    is_public_holiday: int
    is_mercantile_holiday: int
    is_poya_day: int
    is_festive_peak: int

    # Operating Context
    is_in_stock: int
    is_branch_open: int

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Phase 5A: Production Demand Forecasting Inference Schemas (SKU x Store)
# ---------------------------------------------------------------------------

VALID_CATEGORIES: tuple[str, ...] = (
    "Beverages",
    "Packaged Foods",
    "Personal Care",
    "Household Cleaning",
    "Health",
)

VALID_STORE_TYPES: tuple[str, ...] = ("a", "b", "c", "d")
VALID_ASSORTMENTS: tuple[str, ...] = ("a", "b", "c")
VALID_STATE_HOLIDAYS: tuple[str, ...] = ("0", "a", "b", "c")
VALID_HORIZONS: tuple[int, ...] = (1, 7, 14, 30)

MODEL_FEATURE_COLUMNS: list[str] = [
    "category",
    "unit_price",
    "base_unit_price",
    "is_open",
    "is_promo",
    "promo2",
    "day_of_week",
    "state_holiday",
    "school_holiday",
    "store_type",
    "assortment",
]


class DailyForecastContext(BaseModel):
    """Optional per-day operating context override for a future forecast date."""

    model_config = ConfigDict(extra="forbid")

    forecast_date: dt.date | None = Field(default=None, description="Forecast date in YYYY-MM-DD")
    is_open: int = Field(default=1, ge=0, le=1, description="1 if store is open, 0 if closed")
    is_promo: int = Field(default=0, ge=0, le=1, description="1 if promotion is active, 0 otherwise")
    unit_price: float | None = Field(default=None, gt=0.0, description="Override unit price for this day if discounted")
    state_holiday: str = Field(default="0", description="State holiday code: '0', 'a', 'b', 'c'")
    school_holiday: int = Field(default=0, ge=0, le=1, description="1 if school holiday, 0 otherwise")

    @field_validator("state_holiday", mode="before")
    @classmethod
    def validate_state_holiday(cls, v: Any) -> str:
        s = str(v)
        if s not in VALID_STATE_HOLIDAYS:
            raise ValueError(f"Invalid state_holiday: {v!r}. Must be one of {VALID_STATE_HOLIDAYS}")
        return s

    @model_validator(mode="before")
    @classmethod
    def forbid_customers(cls, data: Any) -> Any:
        if isinstance(data, dict) and "customers" in data:
            raise ValueError("Feature 'customers' is strictly forbidden.")
        return data


# Backwards compatibility alias
FutureCalendarRecord = DailyForecastContext


class DemandForecastRequest(BaseModel):
    """Inference request for SKU x Store multi-day unit demand forecasting."""

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1, description="Product / SKU identifier")
    store_id: str = Field(min_length=1, description="Store / Branch identifier")
    category: str = Field(description="Product category")
    base_unit_price: float = Field(gt=0.0, description="Regular base unit selling price")
    unit_price: float | None = Field(default=None, gt=0.0, description="Active promotional or standard unit price")
    store_type: str = Field(default="a", description="Store type: 'a', 'b', 'c', 'd'")
    assortment: str = Field(default="a", description="Store assortment level: 'a', 'b', 'c'")
    promo2: int = Field(default=0, ge=0, le=1, description="Continuous store promotion flag (0 or 1)")
    horizon: int = Field(default=7, description="Forecast horizon in days (1, 7, 14, 30)")
    forecast_date: dt.date | None = Field(default=None, description="Start date of forecast horizon (defaults to tomorrow)")
    daily_contexts: list[DailyForecastContext] | None = Field(
        default=None,
        description="Optional day-by-day operating conditions for the horizon dates",
    )
    has_historical_data: bool | None = Field(
        default=None,
        description="Optional indicator whether real historical sales exist for this product and branch",
    )

    @field_validator("product_id", "store_id")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v.strip()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("category cannot be empty or whitespace only")
        return v.strip()

    @field_validator("store_type")
    @classmethod
    def validate_store_type(cls, v: str) -> str:
        if v not in VALID_STORE_TYPES:
            raise ValueError(f"Invalid store_type: {v!r}. Allowed values are: {VALID_STORE_TYPES}")
        return v

    @field_validator("assortment")
    @classmethod
    def validate_assortment(cls, v: str) -> str:
        if v not in VALID_ASSORTMENTS:
            raise ValueError(f"Invalid assortment: {v!r}. Allowed values are: {VALID_ASSORTMENTS}")
        return v

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v not in VALID_HORIZONS:
            raise ValueError(f"Invalid horizon: {v}. Supported horizons are {VALID_HORIZONS}")
        return v

    @model_validator(mode="before")
    @classmethod
    def check_extra_and_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "customers" in data:
                raise ValueError("Feature 'customers' is strictly forbidden.")
            # Allow store_id to be passed as int, convert to str
            if "store_id" in data and isinstance(data["store_id"], int):
                data["store_id"] = str(data["store_id"])
            # Allow forecast_horizon as alias for horizon
            if "forecast_horizon" in data and "horizon" not in data:
                data["horizon"] = data.pop("forecast_horizon")
        return data


class DailyUnitForecastPoint(BaseModel):
    """Daily predicted physical demand units point."""

    product_id: str = Field(description="Product identifier")
    store_id: str = Field(description="Store identifier")
    forecast_date: dt.date = Field(description="Forecast date in YYYY-MM-DD")
    predicted_units: float = Field(ge=0.0, description="Predicted units sold (>= 0.0, 0.0 if closed)")
    horizon: int = Field(description="Forecast horizon length in days")
    day_of_week: int = Field(ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    is_open: int = Field(ge=0, le=1, description="1 if store was open, 0 if closed")
    is_promo: int = Field(ge=0, le=1, description="1 if promotion active, 0 otherwise")
    unit_price: float = Field(gt=0.0, description="Unit price evaluated")


# Backwards compatibility alias
DailyDemandForecastPoint = DailyUnitForecastPoint


class DemandForecastModelInfo(BaseModel):
    """Trained model metadata and version info."""

    model_type: str = Field(default="lightgbm", description="Model architecture type")
    target: str = Field(default="units_sold", description="Model prediction target")
    version: str = Field(default="1.0.0", description="Model version")
    feature_columns: list[str] = Field(default_factory=list, description="Exact feature columns used")
    feature_count: int = Field(default=11, description="Total feature count")


class DemandForecastResponse(BaseModel):
    """Structured response for demand forecasting."""

    product_id: str = Field(description="Product identifier")
    store_id: str = Field(description="Store identifier")
    forecast_date: dt.date = Field(description="Forecast start date")
    predicted_units: float = Field(ge=0.0, description="Predicted units (sum over horizon)")
    horizon: int = Field(description="Forecast horizon in days")
    predictions: list[DailyUnitForecastPoint] = Field(description="Daily forecast predictions")
    total_predicted_units: float = Field(ge=0.0, description="Total sum of predicted units over the horizon")
    model_metadata: DemandForecastModelInfo = Field(description="Model metadata and version")



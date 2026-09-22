from datetime import date

from pydantic import BaseModel, Field, model_validator


class StockForecastRequest(BaseModel):
    product_id: str
    forecast_date: date
    current_stock: float = Field(ge=0)
    lead_time_days: int = Field(default=7, ge=1, le=180)
    review_period_days: int = Field(default=14, ge=1, le=365)
    service_level: float = Field(default=0.95, gt=0.5, lt=0.999)
    demand_lag_1: float = Field(ge=0)
    demand_lag_7: float = Field(ge=0)
    demand_lag_14: float = Field(ge=0)
    rolling_mean_7: float = Field(ge=0)
    rolling_mean_28: float = Field(ge=0)
    rolling_std_7: float = Field(ge=0)
    unit_price: float = Field(gt=0)
    price_index_28: float = Field(gt=0)
    orders_lag_1: float = Field(ge=0)
    customers_lag_1: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_stock_context(self) -> "StockForecastRequest":
        if self.current_stock < 0:
            raise ValueError("current_stock must not be negative")
        return self

    def model_features(self) -> dict[str, float]:
        return {
            "demand_lag_1": self.demand_lag_1,
            "demand_lag_7": self.demand_lag_7,
            "demand_lag_14": self.demand_lag_14,
            "rolling_mean_7": self.rolling_mean_7,
            "rolling_mean_28": self.rolling_mean_28,
            "rolling_std_7": self.rolling_std_7,
            "unit_price": self.unit_price,
            "price_index_28": self.price_index_28,
            "orders_lag_1": self.orders_lag_1,
            "customers_lag_1": self.customers_lag_1,
            "day_of_week": float(self.forecast_date.weekday()),
            "month": float(self.forecast_date.month),
            "is_weekend": float(self.forecast_date.weekday() >= 5),
        }


class StockForecastResponse(BaseModel):
    product_id: str
    predicted_daily_demand: float
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    recommended_order_quantity: float
    stockout_risk: float
    status: str


class MultiHorizonForecastRequest(BaseModel):
    product_id: str = Field(min_length=1)
    start_date: date
    forecast_days: int = Field(default=365, ge=30, le=730)
    recent_daily_demand: list[float] = Field(min_length=7, max_length=365)
    current_stock: float = Field(ge=0)
    lead_time_days: int = Field(default=7, ge=1, le=180)
    review_period_days: int = Field(default=14, ge=1, le=365)
    service_level: float = Field(default=0.95, gt=0.5, lt=0.999)
    unit_price: float = Field(gt=0)
    price_index_28: float = Field(default=1.0, gt=0)
    orders_lag_1: float = Field(ge=0)
    customers_lag_1: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_forecast_window(self) -> "MultiHorizonForecastRequest":
        if any(value < 0 for value in self.recent_daily_demand):
            raise ValueError("recent_daily_demand values must not be negative")
        required_days = self.lead_time_days + self.review_period_days
        if self.forecast_days < required_days:
            raise ValueError(
                "forecast_days must cover lead_time_days plus review_period_days"
            )
        return self


class DailyDemandForecast(BaseModel):
    date: date
    predicted_demand: float


class AggregatedDemandForecast(BaseModel):
    label: str
    start_date: date
    end_date: date
    total_demand: float
    average_daily_demand: float


class MultiHorizonStockPlan(BaseModel):
    current_stock: float
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    target_stock: float
    recommended_order_quantity: float
    stockout_risk: float
    status: str


class MultiHorizonForecastSummary(BaseModel):
    forecast_days: int
    total_predicted_demand: float
    average_daily_demand: float
    next_7_days_demand: float
    next_30_days_demand: float
    peak_demand_date: date
    peak_daily_demand: float


class DirectHorizonForecast(BaseModel):
    horizon_days: int
    predicted_demand: float
    lower_bound: float
    upper_bound: float
    interval_level: float
    method: str


class ForecastDataReadiness(BaseModel):
    history_days: int
    maturity: str
    confidence: str
    production_ready: bool
    recommended_action: str


class MultiHorizonForecastResponse(BaseModel):
    product_id: str
    summary: MultiHorizonForecastSummary
    stock_plan: MultiHorizonStockPlan
    direct_horizons: list[DirectHorizonForecast]
    data_readiness: ForecastDataReadiness
    daily: list[DailyDemandForecast]
    weekly: list[AggregatedDemandForecast]
    monthly: list[AggregatedDemandForecast]
    seasonal: list[AggregatedDemandForecast]
    yearly: list[AggregatedDemandForecast]
    notes: list[str]

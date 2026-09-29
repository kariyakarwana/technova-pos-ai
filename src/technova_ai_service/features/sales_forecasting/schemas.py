import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DailyRevenuePoint(BaseModel):
    date: datetime.date = Field(description="Transaction date in YYYY-MM-DD")
    revenue: float = Field(ge=0, description="Observed daily revenue in LKR, must be >= 0")
    is_operating_day: int = Field(default=1, ge=0, le=1, description="1 if operating, 0 otherwise")


class FutureCalendarPoint(BaseModel):
    date: datetime.date = Field(description="Future date in YYYY-MM-DD")
    is_operating_day: int = Field(default=1, ge=0, le=1, description="1 if branch will operate, 0 otherwise")
    active_discounts: int = Field(default=0, ge=0, description="Count of active planned discounts")


class DailyForecastPrediction(BaseModel):
    date: datetime.date = Field(description="Forecast date in YYYY-MM-DD")
    predicted_revenue: float = Field(ge=0, description="Predicted daily revenue in LKR")


class SalesForecastRequest(BaseModel):
    organization_id: str = Field(min_length=1, description="Organization tenant ID")
    branch_id: str = Field(min_length=1, description="Branch ID")
    forecast_horizon: Literal[1, 7, 14, 30] = Field(
        description="Forecast horizon in days (1, 7, 14, or 30)"
    )
    recent_daily_revenue: list[DailyRevenuePoint] = Field(
        default_factory=list,
        description="Recent historical daily revenue observations",
    )
    future_calendar: list[FutureCalendarPoint] = Field(
        default_factory=list,
        description="Known-future calendar inputs for the forecast horizon",
    )

    @model_validator(mode="after")
    def validate_request(self) -> "SalesForecastRequest":
        if not self.organization_id.strip():
            raise ValueError("organization_id must not be empty or whitespace")
        if not self.branch_id.strip():
            raise ValueError("branch_id must not be empty or whitespace")

        if self.future_calendar:
            for i in range(1, len(self.future_calendar)):
                if self.future_calendar[i].date <= self.future_calendar[i - 1].date:
                    raise ValueError("future_calendar dates must be strictly chronological")

        return self


class SalesForecastResponse(BaseModel):
    organization_id: str
    branch_id: str
    forecast_horizon: int
    predictions: list[DailyForecastPrediction]

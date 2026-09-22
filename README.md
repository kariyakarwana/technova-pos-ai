# TechNova AI Service

FastAPI service and reproducible training pipelines for TechNova's retail AI features:

- Inventory intelligence and reorder recommendations
- Guardrailed dynamic-pricing recommendations
- Customer loyalty and personalized-product recommendations

The development pipeline uses the UCI Online Retail dataset. Raw datasets and trained model
artifacts are intentionally excluded from Git. Production training will use an export from the
TechNova database with the same normalized transaction schema.

## Setup

```powershell
uv sync --all-groups
uv run python scripts/download_online_retail.py
uv run python scripts/preprocess_online_retail.py
uv run python scripts/train_all.py
uv run uvicorn technova_ai_service.main:app --reload --port 8000
```

API documentation is available at `http://localhost:8000/docs`.

## Forecast APIs

- `POST /v1/inventory/forecast` returns the existing next-day demand and reorder decision.
- `POST /v1/inventory/forecast/multi-horizon` recursively forecasts daily demand and returns
  consistent daily, weekly, monthly, calendar-quarter seasonal, and yearly aggregates.

The multi-horizon request accepts at least seven actual daily demand values in chronological order.
A 365-day request is the default. With fewer than 28 observations the response is marked as a
`cold_start`; it remains non-production until a TechNova-trained artifact passes backtesting. The
returned stock plan uses supplier lead time, review period, service level, current stock, and
forecast demand to calculate its reorder point and recommended order quantity.

The seven-, fourteen-, and thirty-day totals use a conservative direct forecast based on the latest
available actual demand. These totals calibrate the ML daily curve so recursively generated values
cannot collapse over time. Each total includes an uncertainty range and the response includes data
maturity, confidence, and recommended-action fields.

Long-range forecasts are planning estimates. Refresh them daily as real sales arrive. Named retail
events such as Sinhala and Tamil New Year require explicit event features and training history from
TechNova; a calendar-quarter total alone is not an event forecast.

## Quality checks

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
uv run python scripts/evaluate_stock_accuracy.py `
  --output artifacts/stock_intelligence/accuracy-report.json
```

## Data layout

```text
data/
  raw/online_retail/Online Retail.xlsx
  processed/
artifacts/
  stock_intelligence/
  dynamic_pricing/
  loyalty_recommendations/
```

## Training principles

- Cancelled orders and non-positive quantity or price records are excluded.
- Time-based splits are used for forecasting and pricing to prevent future-data leakage.
- Forecasting features use only lagged or shifted rolling values.
- Dynamic pricing is advisory and enforces cost, margin, and price-change limits.
- Loyalty recommendations exclude products already purchased by the customer when possible.
- Metrics and model metadata are saved beside each artifact for auditability.

## Production integration

Export TechNova sales into the normalized columns documented in
`docs/technova-transaction-schema.md`, then train with `--input` pointing to that CSV or Parquet
file. Model training should run as an offline job. The FastAPI application only loads approved
artifacts for inference.

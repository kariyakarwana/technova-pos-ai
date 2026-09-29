# Phase 5.2: NestJS Backend Integration with TechNova AI Sales Forecasting

## 1. Architectural Overview

```
Frontend (React/Next.js Client)
  ↓ HTTP POST /api/v1/sales-forecast { storeId: 1, forecastHorizon: 7 }
NestJS Backend (`technova-pos-api`)
  ↓ HTTP POST http://127.0.0.1:8001/api/v1/sales-forecast { store_id: 1, forecast_horizon: 7 }
FastAPI AI Service (`technova-pos-ai`)
  ↓ Point-in-time feature extraction (30 features, strict date isolation)
Trained XGBoost Regressor (`data/interim/xgboost_sales_forecasting.json`)
```

---

## 2. Responsibilities & Separation of Concerns

1. **Frontend**:
   - Captures user intent: selected store and forecast horizon (`1`, `7`, `14`, or `30` days).
   - Sends standard camelCase payload to the NestJS API gateway.

2. **NestJS Backend (`technova-pos-api`)**:
   - Exposes `POST /api/v1/sales-forecast` (and `GET /api/v1/sales-forecast/health`).
   - Validates input using `SalesForecastRequestDto` (`class-validator` / `class-transformer`):
     - `storeId`: positive integer (`@IsInt()`, `@Min(1)`).
     - `forecastHorizon`: strictly one of `1`, `7`, `14`, `30` (`@IsIn([1, 7, 14, 30])`).
   - Rejects unwhitelisted properties (`forbidNonWhitelisted: true`).
   - Translates payload to `{ store_id, forecast_horizon }` and delegates to the AI service.
   - **Strict Data Isolation**: Does NOT contain ML model artifacts, training scripts, or feature engineering logic. Never passes `Customers` to the AI service.
   - **Resilience & Error Handling**: Maps network dropouts to `503 Service Unavailable`, timeouts to `504 Gateway Timeout`, FastAPI 404/422 to corresponding client HTTP codes, and sanitized messages without leaking Python stack traces.

3. **FastAPI AI Service (`technova-pos-ai`)**:
   - Listens on `http://127.0.0.1:8001`.
   - Loads the serialized XGBoost artifact (`data/interim/xgboost_sales_forecasting.json`) once into memory upon service startup.
   - Constructs point-in-time calendar, store, and lag/rolling features without future target leakage.
   - Never exposes or expects `Customers` as a forecasting feature.

4. **XGBoost Model**:
   - Already-trained production model (Phase 4 candidate with optimal WAPE 0.1068, MASE 0.6558 on July 2015 holdout).
   - Produces non-negative daily sales predictions.

---

## 3. Configuration & Environment Variables

| Variable | Default Value | Description |
|---|---|---|
| `AI_SERVICE_URL` | `http://127.0.0.1:8001` | Base URL of the TechNova AI FastAPI forecasting service. |
| `AI_SERVICE_TIMEOUT_MS` | `10000` | Transport timeout in milliseconds for AI HTTP requests. |

---

## 4. Endpoints

### `POST /api/v1/sales-forecast`
**Request Body**:
```json
{
  "storeId": 1,
  "forecastHorizon": 7
}
```

**Response Body**:
```json
{
  "store_id": 1,
  "forecast_horizon": 7,
  "forecast_start_date": "2015-07-01",
  "forecast_end_date": "2015-07-07",
  "predictions": [
    { "date": "2015-07-01", "predicted_sales": 5375.50 },
    { "date": "2015-07-02", "predicted_sales": 5076.87 },
    { "date": "2015-07-03", "predicted_sales": 5082.19 },
    { "date": "2015-07-04", "predicted_sales": 4188.59 },
    { "date": "2015-07-05", "predicted_sales": 0.00 },
    { "date": "2015-07-06", "predicted_sales": 5800.12 },
    { "date": "2015-07-07", "predicted_sales": 4950.30 }
  ]
}
```

### `GET /api/v1/sales-forecast/health`
**Response Body**:
```json
{
  "status": "healthy",
  "service_status": "healthy",
  "model_loaded": true,
  "model_name": "XGBoost",
  "supported_horizons": [1, 7, 14, 30]
}
```

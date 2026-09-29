# Phase 1.3: Rossmann Sales Forecasting Data Design

This document specifies the finalized data and feature design for the TechNova AI Sales Forecasting engine using the Rossmann Store Sales dataset. It translates the profiling insights discovered in `data/interim/rossmann_profile.json` into a leakage-free, production-ready forecasting specification.

---

## 1. Executive Summary & Core Definitions

| Component | Specification |
| :--- | :--- |
| **Forecast Unit** | **Store × Date → Sales** |
| **Target Variable** | **Sales** (Daily total sales per store in EUR) |
| **Forecast Level** | **Store-level daily aggregated sales** |
| **Horizons** | **1 day, 7 days, 14 days, 30 days** |
| **Splitting Strategy** | **Strict Chronological (Time-based)** |
| **Primary Metrics** | **WAPE** (Weighted Absolute Percentage Error) + **MASE** (Mean Absolute Scaled Error) |
| **Secondary Metrics** | **MAE** (Mean Absolute Error), **RMSE** (Root Mean Squared Error) |
| **Candidate Baselines** | Naive (Lag-1), Seasonal Naive (Lag-7) |
| **Candidate ML Models** | LightGBM, XGBoost |

> **Note on Granularity**: The Rossmann dataset does not contain individual transaction-level or product/SKU-level identifiers. Therefore, forecasting is strictly designed at the **store-day** level.

---

## 2. Dataset Sizing & Computational Feasibility

Based on the profiling analysis (`rossmann_profile.json`):

- **Train Rows**: 1,017,209 records
- **Unique Stores**: 1,115 stores
- **Temporal Span**: 942 consecutive calendar days globally (`2013-01-01` to `2015-07-31`)
- **Metadata Coverage**: 100% store join coverage (all 1,115 stores present in `store.csv`)

### Why Rossmann is Ideal for CPU-Based Experimentation
Compared to extreme scale transactional datasets (e.g., 100M+ raw transactions), 1,017,209 daily rows occupy under 100 MB of uncompressed memory. This provides:
1. **In-Memory Transformation**: The full dataset can be indexed, transformed, and cached directly in memory without disk spilling or distributed clusters.
2. **Rapid Iteration**: GBDT models (LightGBM / XGBoost) can train on the complete 2.5-year history in a few minutes on standard multi-core developer laptops.
3. **Reproducibility**: Fast cross-validation and feature experimentation cycles ensure robust quality gates before model deployment.

---

## 3. Forecast Horizons & Operational Objectives

The sales forecasting service targets four distinct operational horizons:

1. **1-Day Horizon ($t+1$)**: Next-day sales forecasting. Primary application: daily staffing schedules, immediate cash replenishment, and peak-hour opening preparation.
2. **7-Day Horizon ($t+1 \dots t+7$)**: Weekly forecast. Primary application: weekly roster planning, short-cycle perishables replenishment, and promotional execution.
3. **14-Day Horizon ($t+1 \dots t+14$)**: Bi-weekly forecast. Primary application: supplier order cycle synchronization and stock replenishment cadence.
4. **30-Day Horizon ($t+1 \dots t+30$)**: Monthly forecast. Primary application: store revenue budgeting, inventory working-capital planning, and macro promotions.

---

## 4. Time-Based Data Split Strategy

Random splitting (e.g., `train_test_split(shuffle=True)`) causes fatal data leakage in time series by allowing future observations to inform past predictions. Only **chronological splitting** is permitted.

### Split Schedule (Global 942-Day Range: 2013-01-01 to 2015-07-31)

```
[------------- Training Period -------------][-- Validation --][---- Test Holdout ----]
2013-01-01                         2015-05-31 2015-06-01 2015-06-30 2015-07-01 2015-07-31
               (881 Days)                          (30 Days)              (31 Days)
```

1. **Training Set**:
   - Date range: `2013-01-01` to `2015-05-31` (~881 days)
   - Used for training model parameters and calibrating feature pipelines.
2. **Validation Set**:
   - Date range: `2015-06-01` to `2015-06-30` (30 days)
   - Used for hyperparameter tuning, feature selection, early stopping, and threshold calibration.
3. **Test / Holdout Set**:
   - Date range: `2015-07-01` to `2015-07-31` (31 days)
   - Used exclusively for final performance evaluation and model card certification across all 4 forecast horizons.
4. **Unseen Out-of-Sample Period (`test.csv`)**:
   - Date range: `2015-08-01` to `2015-09-17` (48 days, 856 stores)
   - Serves as external out-of-time evaluation reflecting production inference.

---

## 5. Feature Architecture & Taxonomy

Features are partitioned into four distinct groups according to their origin and temporal availability:

### A. Calendar Features (Deterministic Future Knowledge)
Generated directly from the calendar date, available indefinitely into the future without error:
- `day_of_week`: Integer [0..6] or [1..7]
- `month`: Integer [1..12]
- `year`: Integer [2013..2015]
- `day_of_month`: Integer [1..31]
- `week_of_year`: ISO calendar week [1..53]
- `is_weekend`: Binary flag indicating Saturday or Sunday

### B. Store Metadata Features (Static / Low-Frequency Changing)
Enriched from `store.csv` using exact `Store` ID join:
- `Store`: Categorical identifier (entity embedding or label encoded)
- `StoreType`: Categorical (`a`, `b`, `c`, `d`)
- `Assortment`: Categorical (`a`, `b`, `c`)
- `CompetitionDistance`: Float (meters to nearest competitor, with missing indicator)
- `CompetitionOpenSinceMonth`: Float / Categorical month
- `CompetitionOpenSinceYear`: Float / Categorical year
- `Promo2`: Binary flag indicating continuous coupon promotion participation
- `Promo2SinceWeek`: Float
- `Promo2SinceYear`: Float
- `PromoInterval`: Categorical promo cycle string (e.g. `Jan,Apr,Jul,Oct`)

### C. Known-Future Operational Features (Exogenous Schedule)
Operational schedules scheduled in advance and available at the forecast origin $T$ for horizon $T+h$:
- `Open`: Binary flag (1 = store open, 0 = store closed)
- `Promo`: Binary flag indicating active retail discount campaign on that day
- `StateHoliday`: Categorical holiday indicator (`0`, `a`, `b`, `c`)
- `SchoolHoliday`: Binary flag indicating public school holiday

### D. Historical Sales Features (Point-in-Time Lags & Rolling Aggregations)
Engineered strictly using sales observations prior to the forecast origin $T$:
- **Lags**:
  - `lag_1`: Sales at $t-1$ (usable for 1-day ahead forecasts, or multi-step recursive models)
  - `lag_7`: Sales at $t-7$ (same day of previous week)
  - `lag_14`: Sales at $t-14$ (same day 2 weeks ago)
  - `lag_28`: Sales at $t-28$ (same day 4 weeks ago)
- **Rolling Statistics**:
  - `rolling_mean_7`: 7-day trailing average sales
  - `rolling_mean_14`: 14-day trailing average sales
  - `rolling_mean_28`: 28-day trailing average sales
  - `rolling_std_7`: 7-day trailing sales volatility
  - `rolling_std_28`: 28-day trailing sales volatility

> **Horizon-Specific Lag Rule**: For direct $h$-day horizon models where $h \ge 7$, lag features must respect the forecast origin gap (e.g. minimum lag $\ge h$, or dynamic recursive rollouts).

---

## 6. Treatment of Customers Column

The dataset profiling revealed that `Customers` has a mean of 633.1 and standard deviation of 464.4 on training records.

### Critical Leakage Rule:
**Actual contemporaneous `Customers` at day $t$ MUST NOT be used as a direct input feature for predicting `Sales` at day $t$.**

- **Rationale**: On day $t$ in the future, the actual number of customers entering the store is unknown until the business day has finished. Using future customer counts creates massive artificial leakage that cannot exist during live production inference.
- **Permissible Usage**:
  - Historical customer metrics available prior to forecast origin (e.g., `customers_lag_7`, `customers_rolling_mean_14`).
  - Optional Phase 2 architecture: A two-stage hierarchy where Customer count is forecasted first as an auxiliary target, and then passed into the sales predictor.

---

## 7. Closed Stores & Zero Sales Policy

The profiling revealed 172,871 zero-sales instances (16.99% of all rows):
- **172,817 rows**: `Open == 0` and `Sales == 0` (Store closed, zero sales expected)
- **54 rows**: `Open == 1` and `Sales == 0` (Store open, but zero sales recorded)
- **0 rows**: `Open == 0` and `Sales > 0` (No closed stores made sales)

### Strategy:
1. **Preserve Distinction**: Do not indiscriminately purge all zero-sales rows from the training pipeline. The two states represent fundamentally different commercial mechanisms:
   - `Open == 0`: Known deterministic zero. In production inference, if `Open == 0` is known from the calendar/schedule, predicted sales can be set directly to 0 by rule.
   - `Open == 1` with `Sales == 0`: Operational anomalies, extreme stock-outs, emergency closures, or data capture glitches.
2. **Modeling Treatment**:
   - Model training can be evaluated either on (a) all observations with `Open` as an explicit feature, or (b) open-day filtering (`Open == 1`) with an operational post-processing override (`if Open == 0 then Sales = 0`).
   - The final design retains both pathways for baseline comparison.

---

## 8. Missing Dates & Calendarization Strategy

Profiling showed:
- **934 stores** have complete 942-day calendar records (83.8% of stores).
- **181 stores** have dates with gaps (16.2% of stores; minimum record count is 758 days).
- **0 duplicate store-date pairs** across the dataset.

### Point-in-Time Calendarization Protocol:
1. **No Blind Forward-Filling**: Do not forward-fill or interpolate sales on missing dates.
2. **Classifying Gap Types**:
   - *Store Refurbishment / Extended Closure*: Some stores were closed for prolonged periods (months) for renovations. During these periods, sales are legitimately 0, but store closure must be documented.
   - *Late Opening*: 1 store started operations after `2013-01-01`.
3. **Lag Computation Alignment**:
   - When generating rolling and lag features, create an explicit continuous date grid per store (`pd.date_range(min_date, max_date, freq='D')`).
   - Align missing dates with explicit indicator flags so that rolling windows calculate over true elapsed calendar days rather than indexed row positions.

---

## 9. Target Transformation Policy

- **Current State**: Keep original `Sales` (in EUR) unaltered in interim pipelines.
- **No Early Transformation**: Do not prematurely apply `log1p` scaling or Box-Cox transformations during the preliminary design stage.
- **Rationale**: Retaining untransformed target values ensures that baseline models, WAPE, and MASE calculations evaluate on exact commercial monetary units.
- **Future Evaluation**: In Phase 2 model tuning, `log1p(Sales)` will be evaluated as a hyperparameter to stabilize exponential spikes on peak holiday days.

---

## 10. Strict Data Leakage Prevention Rules

| Rule | Enforcement Mechanism |
| :--- | :--- |
| **No Random Splitting** | Only chronological cutoffs (`2015-05-31` / `2015-06-30` / `2015-07-31`). |
| **No Future Sales** | All lags and rolling statistics must be strictly shifted: $X_{t} = f(\text{Sales}_{\le t-1})$. |
| **No Future Customers** | Actual contemporaneous `Customers` is excluded from feature matrices. |
| **Point-in-Time Metadata** | Dynamic store attributes (e.g. `CompetitionOpenSince`) must only reflect information known as of time $t$. |
| **Exogenous Field Verification** | `Open`, `Promo`, `StateHoliday`, `SchoolHoliday` may only be used when they represent planned operational schedules available at forecast origin. |

---

## 11. Modeling Strategy & Planned Candidates

### 1. Baselines (Benchmarks)
- **Naive Predictor**: $\hat{y}_{t} = y_{t-1}$ (Sales of yesterday).
- **Seasonal Naive Predictor**: $\hat{y}_{t} = y_{t-7}$ (Sales of the same day of the prior week).
- Any candidate ML model must statistically outperform Seasonal Naive on WAPE and MASE to be considered viable.

### 2. Machine Learning Candidates
- **LightGBM Regressor**: Primary tree-based model for tabular time series. High efficiency, native handling of categorical features (`StoreType`, `Assortment`), fast training.
- **XGBoost Regressor**: Benchmark comparison model with exact histogram depthwise tree growth.

---

## 12. Evaluation Framework & Metrics

Due to the presence of zero-sales days (16.99% of records), standard **MAPE (Mean Absolute Percentage Error)** causes division-by-zero division instability or artificial blow-ups.

### Primary Metrics:
1. **WAPE (Weighted Absolute Percentage Error)**:
   $$\text{WAPE} = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} y_i}$$
   Robust against individual small or zero actuals; gives revenue-weighted accuracy.
2. **MASE (Mean Absolute Scaled Error)**:
   $$\text{MASE} = \frac{\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|}{\frac{1}{n-7} \sum_{i=8}^{n} |y_i - y_{i-7}|}$$
   Normalizes performance against the Seasonal Naive benchmark. A MASE $< 1.0$ indicates improvement over the 7-day seasonal persistence baseline.

### Secondary Metrics:
- **MAE (Mean Absolute Error)**: Average absolute error in EUR per store-day.
- **RMSE (Root Mean Squared Error)**: Penalizes large out-of-scale forecasting errors.

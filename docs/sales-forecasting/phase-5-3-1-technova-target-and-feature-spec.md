# Phase 5.3.1: TechNova-Native Sales Forecasting Specification
## Target Definition, Feature Architecture, and Data Contract

---

## 1. Executive Summary

This document specifies the **TechNova-Native Sales Forecasting** design. While Phase 3 and Phase 4 validated modeling approaches on the benchmark Rossmann dataset (Phase 5.1/5.2 API implementation), this specification defines the transition to **direct ingestion of TechNova POS transactional data** stored in PostgreSQL via Prisma.

---

## 2. Forecasting Unit & Grain

### Candidate Evaluation:
1. **Option A: Branch × Date → Daily Revenue**
   - **Schema Support**: Direct. `Sale.total` records the exact net revenue of each completed sale associated with a `Branch.id` and finalized at `Sale.completedAt` (or `createdAt`).
   - **Characteristics**: Universally fungible across all retail categories (electronics, groceries, hardware, clothing), matches POS financial dashboard metrics, directly drives cash flow and staffing forecasts.
2. **Option B: Branch × Date → Daily Units Sold**
   - **Schema Support**: `SaleItem.quantity` exists (`Decimal(18, 3)`).
   - **Limitations at Store Level**: Summing raw physical units across diverse catalog items (e.g., summing 1 laptop + 2 apples + 5 screws = 8 "units") produces an aggregate with inconsistent unit economics and high variance across product mixes. (Unit-level forecasting is already handled by `stock_intelligence`).
3. **Option C: Branch × Date → Both Revenue and Units**
   - **Evaluation**: Adds architectural complexity and dual-objective optimization without clear operational benefits for top-level store management.

### Architectural Decision:
> **Selected Grain: Option A — Branch × Date $\rightarrow$ Daily Net Sales Revenue (LKR)**.

---

## 3. Primary Target Definition

- **Target Identifier**: `daily_revenue`
- **Unit of Measurement**: Currency units (LKR, standard TechNova POS currency).
- **Mathematical Formula**:
  $$\text{daily\_revenue}(b, d) = \sum_{s \in \mathcal{S}(b, d)} s.\text{total}$$
  where:
  - $b \in \text{Branch.id}$
  - $d = \text{date}(\operatorname{COALESCE}(s.\text{completedAt}, s.\text{createdAt}))$
  - $\mathcal{S}(b, d) = \{s \in \text{Sale} \mid s.\text{branchId} = b \land \text{date}(s) = d \land s.\text{status} \in \{\text{COMPLETED}, \text{PARTIALLY\_REFUNDED}\}\}$

### Status Handling Rules:
- `COMPLETED`: Included in full (`s.total`).
- `PARTIALLY_REFUNDED`: Included as net payable (`s.total`), reflecting the post-refund actual revenue of that sale.
- `DRAFT`: **Excluded** (uncommitted cart; no revenue realized).
- `VOIDED`: **Excluded** (aborted before transaction completion; zero realized revenue).
- `REFUNDED`: **Excluded** (fully refunded transactions contribute 0 net revenue).

---

## 4. Time Granularity & Temporal Strategy

1. **Granularity**: Daily ($T = 1\text{ day}$).
2. **Forecast Horizons**:
   $$H \in \{1, 7, 14, 30\}\text{ days}$$
   - Horizon 1: Next-day operational planning and cash drawer reconciliation.
   - Horizon 7: Next-week staffing and inventory replenishment scheduling.
   - Horizon 14: Bi-weekly reorder and working capital planning.
   - Horizon 30: Monthly financial forecasting and executive budgeting.
3. **Minimum Historical Data Requirement**:
   - **Strict Minimum**: **28 consecutive days** of historical POS operations (mathematical requirement to compute `rolling_mean_28` and `lag_28`).
   - **Recommended Baseline**: **90 days** (captures weekday/weekend intra-month seasonality).
   - **Production Full Seasonality**: $\ge \mathbf{365\text{ days}}$ (captures festive, quarterly, and annual retail peaks).
4. **Chronological Splitting Strategy** (Zero Shuffling / Zero Random Splits):
   - **Training Set**: Start of observations $T_0 \rightarrow (T_{\text{holdout}} - 60\text{ days})$
   - **Validation Set**: 30 consecutive days $[T_{\text{holdout}} - 60\text{ days}, T_{\text{holdout}} - 31\text{ days}]$
   - **Holdout Test Set**: 30 consecutive days $[T_{\text{holdout}} - 30\text{ days}, T_{\text{holdout}}]$

---

## 5. Required Features Architecture

Features are strictly categorized by origin and leakage profile:

### Category A: Calendar & Temporal Features (Deterministic, Zero Leakage)
- `day_of_week`: Integer $0$ (Monday) to $6$ (Sunday).
- `day_of_month`: Integer $1$ to $31$.
- `month`: Integer $1$ to $12$.
- `week_of_year`: Integer $1$ to $53$.
- `is_weekend`: Binary flag ($1$ if `day_of_week` $\ge 5$, else $0$).
- `is_month_start`: Binary flag ($1$ if `day_of_month` $\le 3$, else $0$).
- `is_month_end`: Binary flag ($1$ if `day_of_month` $\ge 28$, else $0$).
- `cyclical_day_sin` / `cyclical_day_cos`: $\sin(2\pi \cdot \text{day\_of\_week}/7)$, $\cos(2\pi \cdot \text{day\_of\_week}/7)$.
- `cyclical_month_sin` / `cyclical_month_cos`: $\sin(2\pi \cdot \text{month}/12)$, $\cos(2\pi \cdot \text{month}/12)$.

### Category B: Historical Sales Features (Strictly Point-in-Time $\le t-1$)
- `revenue_lag_1`: Sales revenue on day $t-1$.
- `revenue_lag_7`: Sales revenue on day $t-7$ (seasonal weekly baseline).
- `revenue_lag_14`: Sales revenue on day $t-14$.
- `revenue_lag_28`: Sales revenue on day $t-28$.
- `rolling_mean_7`: 7-day trailing average revenue over $[t-7, t-1]$.
- `rolling_mean_14`: 14-day trailing average revenue over $[t-14, t-1]$.
- `rolling_mean_28`: 28-day trailing average revenue over $[t-28, t-1]$.
- `rolling_std_7`: 7-day trailing standard deviation over $[t-7, t-1]$.
- `rolling_std_28`: 28-day trailing standard deviation over $[t-28, t-1]$.
- `rolling_min_7` / `rolling_max_7`: 7-day trailing minimum and maximum daily revenue.
- `weekly_trend_ratio`: $\text{rolling\_mean\_7} / (\text{rolling\_mean\_28} + 1e-5)$ (momentum signal).

### Category C: Branch Profile Features (Static / Slowly Changing)
- `branch_id`: Categorical string or mapped integer branch identifier.
- `branch_age_days`: $\operatorname{days}(t - \text{Branch.createdAt})$ (lifecycle maturity).
- `branch_historical_avg_sales`: Overall historical mean daily revenue (computed strictly on the training partition).

### Category D: Known-Future Operational Features (Safe at Prediction Time)
- `is_operating_day`: Binary flag indicating whether the branch is scheduled to be open on day $t$.
- `active_discount_count`: Count of active store-level `DiscountRule` records with $\text{startDate} \le t \le \text{endDate}$.

### Category E: Strictly Excluded Features (Data Leakage)
- Contemporaneous `Sale.total` or `Sale.paidTotal` on prediction day $t$.
- Contemporaneous `SaleItem.quantity` or `SaleItem.lineTotal` on prediction day $t$.
- Contemporaneous `Customers` (number of customers or transactions on prediction day $t$ cannot be known before the day occurs).
- Post-hoc status modifications (refunds processed days later backdated to original date).

---

## 6. Data Leakage Prevention Rules

1. **Information Cutoff ($t_0$)**:
   - At the moment a forecast is generated at $t_0$, only transactional data finalized prior to $t_0$ ($\le t_0 - 1$) is available.
2. **Multi-Day Recursive Forecasting**:
   - For horizon $h = 1$: Lag features use actual historical observations from database.
   - For horizons $h \in [2, H]$: Features that depend on intermediate days $t_0 \dots t_0 + h - 1$ must be evaluated using **autoregressive substitution** (feeding predicted revenue into future lags) or **direct multi-horizon models**. Future actual sales must never be peeked.
3. **Customer Count Prohibition**:
   - Customers count is contemporaneous demand, not an exogenous calendar event. It must **never** be accepted as an inference input or used as a future prediction feature.

---

## 7. Missing-Day Semantics & Calendar Reconstruction

In real POS operations, days with zero recorded sales can represent four distinct phenomena:

| Case | Scenario | Evidence in TechNova | Proper Modeling Treatment |
|---|---|---|---|
| **1. Pre-Launch** | Dates before branch was opened. | Date < $\min(\text{Sale.createdAt})$. | **Exclude completely**. Do not insert zero sales; branch did not exist. |
| **2. Scheduled Closure** | Regular closed days (e.g., Sundays, Poya days). | Consistent 0 sales on specific weekday across historical months. | Set `daily_revenue = 0.0` and set `is_operating_day = 0`. |
| **3. Open with Zero Demand** | Branch open, but no customers purchased. | Branch was marked `ACTIVE`, preceding/succeeding days active. | Set `daily_revenue = 0.0` and set `is_operating_day = 1`. |
| **4. Unsynced / Offline Data** | Network outage or terminal un-synced. | Check `OfflineSyncBatch` or consecutive abnormal gaps. | Flag as missing data; do not treat as true zero revenue if unsynced batches exist. |

---

## 8. Multi-Tenant Security & Tenant Isolation

1. **Tenant Isolation Guarantee**:
   Every database query must filter by `Branch.organizationId = :currentOrganizationId`.
2. **Cross-Tenant Contamination Prevention**:
   - Aggregations, lag computation, rolling statistics, and store feature encoding must run in complete isolation per `organizationId`.
   - Feature tables for Organization A must never be accessible, combined, or normalized with Organization B.

---

## 9. Retirement of Rossmann-Specific Features

| Rossmann Feature | Why It Must Be Retired | TechNova Replacement |
|---|---|---|
| `StoreType` (`a, b, c, d`) | Dataset-specific German drugstore typology; non-existent in TechNova schema. | Replaced by `branch_historical_avg_sales` and `branch_age_days`. |
| `Assortment` (`a, b, c`) | Rossmann retail assortment tiers (`basic`, `extra`, `extended`). | Replaced by branch active product count. |
| `CompetitionDistance` | External geospatial survey data not captured in TechNova POS. | Excluded. |
| `CompetitionOpenSince*`| Historical competitor timeline not captured in POS. | Excluded. |
| `Promo2` / `PromoInterval`| Complex supplier-funded recurring promotional calendar. | Replaced by active `DiscountRule` count. |
| `StateHoliday` / `SchoolHoliday` | German holiday schedule (`0`, `a`, `b`, `c`). | Replaced by Sri Lankan / localized retail calendar utility (`is_weekend`, `day_of_week`). |
| Integer `store_id` (1–1115) | Synthetic numeric indices. | Replaced by TechNova `Branch.code` or deterministic tenant-scoped mapping. |

---

## 10. Comprehensive Feature Specification Table

| Feature Name | Source Entity | Data Type | Available at $t_0$? | Leakage Risk? | Decision | Design Rationale |
|---|---|---|---|---|---|---|
| `branch_id` | `Branch.id` | String / Categorical | Yes | No | **Keep** | Entity identifier for store segmentation. |
| `day_of_week` | Date calendar | Int (0–6) | Yes | No | **Keep** | Essential weekly sales seasonality. |
| `day_of_month` | Date calendar | Int (1–31) | Yes | No | **Keep** | Monthly pay-cycle seasonality (salaries, month-end). |
| `month` | Date calendar | Int (1–12) | Yes | No | **Keep** | Annual macro-seasonality. |
| `is_weekend` | Date calendar | Binary (0/1) | Yes | No | **Keep** | Strongest retail traffic driver. |
| `is_month_end` | Date calendar | Binary (0/1) | Yes | No | **Keep** | Payday effect on consumer spending. |
| `revenue_lag_1` | `Sale.total` | Float ($\ge 0$) | Yes | No | **Keep** | Immediate daily auto-correlation. |
| `revenue_lag_7` | `Sale.total` | Float ($\ge 0$) | Yes | No | **Keep** | Primary seasonal reference (same day last week). |
| `revenue_lag_14`| `Sale.total` | Float ($\ge 0$) | Yes | No | **Keep** | Bi-weekly pay-cycle auto-correlation. |
| `revenue_lag_28`| `Sale.total` | Float ($\ge 0$) | Yes | No | **Keep** | 4-week monthly auto-correlation. |
| `rolling_mean_7` | `Sale.total` | Float ($\ge 0$) | Yes | No | **Keep** | Short-term smoothed sales level. |
| `rolling_mean_14`| `Sale.total` | Float ($\ge 0$) | Yes | No | **Keep** | Medium-term sales baseline. |
| `rolling_mean_28`| `Sale.total` | Float ($\ge 0$) | Yes | No | **Keep** | Monthly steady-state sales volume. |
| `rolling_std_7`  | `Sale.total` | Float ($\ge 0$) | Yes | No | **Keep** | Short-term volatility / revenue stability. |
| `rolling_std_28` | `Sale.total` | Float ($\ge 0$) | Yes | No | **Keep** | Monthly volatility metric. |
| `weekly_trend_ratio` | Derived | Float ($\ge 0$) | Yes | No | **Keep** | Normalized ratio of 7-day to 28-day moving average. |
| `is_operating_day` | Branch profile | Binary (0/1) | Yes | No | **Keep** | Disables predictions on planned closures. |
| `active_discounts` | `DiscountRule` | Int ($\ge 0$) | Yes | No | **Keep** | Promotional lift indicator. |
| `Customers` | `Sale.customerId` | Int | No | **CRITICAL** | **REMOVE** | Cannot be known in future; causes severe leakage. |
| `SaleItem.quantity` | `SaleItem` | Float | No | **CRITICAL** | **REMOVE** | Future units sold cannot be known before day ends. |
| `StoreType` | External | Categorical | No | N/A | **REMOVE** | Rossmann-only artifact. |
| `CompetitionDistance` | External | Float | No | N/A | **REMOVE** | Rossmann-only artifact. |

---

## 11. Final Model Input & Inference Contract

Future TechNova-native inference requests dispatched by NestJS to FastAPI will follow this clean, tenant-isolated contract:

```json
{
  "organization_id": "cly1234567890abcdef",
  "branch_id": "clz0987654321fedcba",
  "forecast_horizon": 7,
  "recent_daily_revenue": [
    { "date": "2026-08-26", "revenue": 45200.00, "is_operating_day": 1 },
    { "date": "2026-08-27", "revenue": 52100.50, "is_operating_day": 1 },
    "..."
  ],
  "future_calendar": [
    { "date": "2026-09-24", "is_operating_day": 1, "active_discounts": 1 },
    { "date": "2026-09-25", "is_operating_day": 1, "active_discounts": 0 },
    "..."
  ]
}
```

---

## 12. Final Architecture Specification Summary

| Dimension | Specification |
|---|---|
| **Target** | Daily Net Sales Revenue (`daily_revenue = SUM(Sale.total)`) |
| **Forecasting Grain** | Branch $\times$ Date |
| **Supported Horizons** | $1, 7, 14, 30\text{ days}$ |
| **Minimum History** | 28 days strictly required; 90 days recommended |
| **Features Included** | 10 Calendar features + 11 Historical Lag/Rolling features + 3 Branch features |
| **Strictly Excluded** | `Customers`, contemporaneous line items, Rossmann artifacts (`StoreType`, `Competition*`) |
| **Validation Strategy** | Chronological 30-day split (zero shuffling, zero future leakage) |
| **Primary Evaluation Metrics** | WAPE (Weighted Absolute Percentage Error), MASE, MAE, RMSE |

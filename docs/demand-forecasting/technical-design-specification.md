# Technical Design Specification: TechNova POS Demand Forecasting ML

**Feature Name:** Product × Branch × Date Demand Forecasting  
**Module Target:** `technova-pos-ai` (FastAPI) & `technova-pos-api` (NestJS) & `technova-pos` (Next.js)  
**Status:** DRAFT / DESIGN SPECIFICATION  
**Author:** Antigravity AI Engineering Team  
**Date:** September 2026  
**Document Version:** 1.0.0  

---

## 1. Executive Summary & Problem Formulation

### 1.1 Context & Motivation
TechNova POS currently features store-level macro Sales Forecasting ($\text{Branch} \times \text{Date} \to \text{LKR Revenue}$) powered by an XGBoost model, alongside an operational Stock Intelligence module for inventory replenishment. However, the existing Stock Intelligence module relies on a `HistGradientBoostingRegressor` trained on an external, non-native dataset (UCI Online Retail, UK e-commerce gift catalog) and generates heuristic daily curves calibrated via moving averages without branch awareness.

This specification designs a **native, multi-tenant Machine Learning Demand Forecasting engine** that operates at the true atomic grain of retail operations:
$$\mathbf{Product} \times \mathbf{Branch} \times \mathbf{Date} \longrightarrow \mathbf{Daily\ Units\ Sold}$$

### 1.2 Core Architectural Principles
1. **Separation of Forecasting and Decisioning:** Demand forecasting is the *predictive engine* ($\hat{y}_{p, b, t} \in \mathbb{R}^+$); Stock Intelligence is the *decision/policy engine* (safety stock, reorder point, purchase order generation). The forecast feeds the replenishment policy.
2. **Multi-Tenant Data Isolation:** Strict physical and logical boundary enforcement at the `Organization` and `Branch` level.
3. **High-Performance Batch Inference:** Replacing per-product N+1 HTTP calls with a single vectorized batch inference contract.
4. **Zero Temporal Leakage:** Strict shift guarantees on lag and rolling features, using only historical demand and scheduled future exogenous signals (e.g., calendar, planned discounts).
5. **CPU-First Production Deployment:** Models must train and serve with sub-150ms inference latency on standard cloud CPU environments without requiring GPU infrastructure.

---

## 2. Mathematical Formulation & Target Definition

### 2.1 Forecasting Unit (Grain)
The fundamental grain of prediction is:
$$\langle p, b, t \rangle \in \mathcal{P}_{\text{org}} \times \mathcal{B}_{\text{org}} \times \mathcal{T}$$
where:
- $p \in \mathcal{P}_{\text{org}}$: Active Product SKU within the tenant organization (`Product.id`).
- $b \in \mathcal{B}_{\text{org}}$: Active physical Branch/Store within the tenant organization (`Branch.id`).
- $t \in \mathcal{T}$: Daily calendar date at 00:00:00 UTC (`Date`).

### 2.2 Primary Target Variable ($y_{p, b, t}$)
The target variable is the **total non-refunded physical units demanded and sold** for product $p$ at branch $b$ on date $t$:
$$y_{p, b, t} = \sum_{i \in \mathcal{I}(p, b, t)} i.\text{quantity}$$

where $\mathcal{I}(p, b, t)$ is the set of all `SaleItem` records satisfying:
$$\mathcal{I}(p, b, t) = \left\{ i \in \text{SaleItem} \;\middle|\; \begin{aligned} &i.\text{productId} = p \\ &\land i.\text{sale}.\text{branchId} = b \\ &\land \operatorname{date}(i.\text{sale}.\text{completedAt} \mathbin{?} i.\text{sale}.\text{completedAt} : i.\text{sale}.\text{createdAt}) = t \\ &\land i.\text{sale}.\text{status} \in \{\text{SaleStatus.COMPLETED}, \text{SaleStatus.PARTIALLY\_REFUNDED}\} \end{aligned} \right\}$$

### 2.3 Transaction Status Handling Matrix

| Sale Status | Inclusion in Target ($y_{p, b, t}$) | Rationale |
|---|---|---|
| `COMPLETED` | **INCLUDED (Full `SaleItem.quantity`)** | Genuine, realized customer demand. |
| `PARTIALLY_REFUNDED` | **INCLUDED (Net Units Sold)** | Items that were retained represent realized demand; units returned via `ReturnItem` are subtracted from total units for that day. |
| `DRAFT` | **EXCLUDED** | Open carts/uncommitted transactions; no demand realized. |
| `VOIDED` | **EXCLUDED** | Transactions aborted before payment; zero realized demand. |
| `REFUNDED` | **EXCLUDED** | Fully unwound transactions; net zero realized customer demand. |

---

## 3. Forecast Horizons & Operational Objectives

The model produces multi-step forward predictions across three distinct operational horizons:

```
Today (T)
  ├──► H=7 Days   [T+1 ... T+7]   Operational Store Replenishment & Perishable Ordering
  ├──► H=14 Days  [T+1 ... T+14]  Bi-Weekly Supplier Purchase Orders & Working Capital
  └──► H=30 Days  [T+1 ... T+30]  Monthly Promotional & Inventory Buffer Planning
```

1. **Horizon 7 Days ($H=7$):**
   - **Operational Objective:** Short-cycle store replenishment, perishable shelf re-stocking, and emergency purchase orders.
   - **Accuracy Priority:** High precision on intra-week daily seasonality (Monday vs. Friday vs. Weekend).
2. **Horizon 14 Days ($H=14$):**
   - **Operational Objective:** Standard supplier review cycle and bi-weekly purchase order generation matching TechNova's standard 7-day lead time + 7-day buffer.
   - **Accuracy Priority:** Low cumulative bias over the 14-day window.
3. **Horizon 30 Days ($H=30$):**
   - **Operational Objective:** Monthly inventory budget allocation, warehouse-to-branch bulk transfers, and category management.
   - **Accuracy Priority:** Stable trend estimation and promotional peak capture.

---

## 4. Historical Data Requirements & Cold-Start Semantics

### 4.1 Historical Data Tiers

| Data Volume | Classification | System Capabilities | Operational Action |
|---|---|---|---|
| **$< 28$ Days** | **Cold-Start** | Model lags cannot be fully computed. Statistical baseline fallback only. | Display cold-start informational badge (`maturity: cold_start`, `confidence: low`). Reorder recommendations require manual review. |
| **$28 - 89$ Days** | **Limited History** | Full feature extraction enabled; captures weekday/weekend intra-month patterns. | ML inference active (`maturity: limited_history`, `confidence: medium`). Purchase orders require supervisory sign-off. |
| **$90 - 364$ Days** | **Developed History** | Captures multi-month trends, pay-day cycles, and local seasonal momentum. | ML inference production ready (`maturity: developed`, `confidence: high`). Automated reorder drafting enabled. |
| **$\ge 365$ Days** | **Mature History** | Full annual retail seasonality, festive peaks (Sinhala/Tamil New Year, Christmas, Ramadan), and YoY growth captured. | Full autonomous replenishment planning (`maturity: mature`, `confidence: high`). |

### 4.2 Missing-Day & Zero-Sales Semantics

A critical vulnerability in retail time-series modeling is conflating different causes of zero sales. The feature engineering pipeline must enforce strict semantics:

```
                            Daily Sales = 0
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
 Store was OPEN                                      Store was CLOSED
         │                                                   │
 ┌───────┴───────┐                                   ┌───────┴───────┐
 ▼               ▼                                   ▼               ▼
In-Stock    Out-of-Stock                         Planned          Emergency
(True Zero) (Censored Demand)                    (e.g., Poya)     (Strike/Disaster)
```

1. **True Zero Sales (Operating Day, In-Stock):**
   - **Condition:** Branch had sales transactions on date $t$, product $p$ had `StockLevel.quantityOnHand > 0`, but units sold = 0.
   - **Representation:** Target $y_{p, b, t} = 0.0$. Dense zero-filled record. Valid sample point.
2. **Censored Demand (Operating Day, Out-of-Stock):**
   - **Condition:** Branch was open, but `StockLevel.quantityOnHand == 0` for product $p$.
   - **Representation:** Target $y_{p, b, t} = 0.0$, but flagged with `is_out_of_stock = 1`. During training, censored loss weights or moving average imputations are applied so the model does not learn a false drop in customer demand caused by stockout.
3. **Store Closed Days:**
   - **Condition:** Branch had 0 transactions across all products on date $t$ (e.g., statutory non-operating holiday, Sunday closure).
   - **Representation:** Flagged with `is_branch_open = 0`. Future forecast for closed days is forced to 0 with zero variance.
4. **New Product Introduction (SKU Cold-Start):**
   - **Condition:** Product created within $< 28$ days (`Product.createdAt > t - 28d`).
   - **Representation:** Hierarchical prior fallback: borrow the historical sales velocity of the product's `Category` and `Brand` at that branch.

---

## 5. Feature Architecture & Engineering Specification

The feature matrix $\mathbf{X}_{p, b, t}$ combines 6 categories of tabular features:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FEATURE MATRIX X(p, b, t)                       │
├───────────────────┬───────────────────┬────────────────────────────────┤
│ 1. Demand Lags    │ 2. Rolling Stats  │ 3. Static Entity Attributes    │
│    lag_1, lag_2,  │    mean_7, 14, 28 │    product cost, price, brand, │
│    lag_3, lag_7,  │    std_7, 14, 28  │    category, branch tier       │
│    lag_14, 21, 28 │    max_7, 14, 28  │                                │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ 4. Price & Promo  │ 5. Calendar       │ 6. Holiday / Event             │
│    price_index_28 │    day_of_week    │    is_poya_day, mercantile,    │
│    has_discount   │    month, weekend │    public_holiday,             │
│    discount_rate  │    payday_window  │    festive_season              │
└───────────────────┴───────────────────┴────────────────────────────────┘
```

### 5.1 Demand Lags (Autoregressive Signals)
Strictly shifted by $k$ days to prevent lookahead leakage:
- `demand_lag_1`: Sales at $t-1$ (yesterday velocity).
- `demand_lag_2`: Sales at $t-2$.
- `demand_lag_3`: Sales at $t-3$.
- `demand_lag_7`: Sales at $t-7$ (same day previous week).
- `demand_lag_14`: Sales at $t-14$ (same day 2 weeks prior).
- `demand_lag_21`: Sales at $t-21$ (same day 3 weeks prior).
- `demand_lag_28`: Sales at $t-28$ (same day 4 weeks prior).

### 5.2 Rolling Window Statistics
Computed over historical sales strictly prior to cutoff $t$, shifted by 1 day:
$$\text{rolling\_stat}_W(t) = \operatorname{stat}(\{y_{p, b, t - \tau} \mid \tau \in [1, W]\})$$
- **Rolling Means:** `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`.
- **Rolling Volatility:** `rolling_std_7`, `rolling_std_14`, `rolling_std_28`.
- **Rolling Maximums (Peak Demand):** `rolling_max_7`, `rolling_max_14`, `rolling_max_28`.

### 5.3 Static & Product Catalog Attributes
- `category_id`: Integer encoded category identifier (`Category.id`).
- `brand_id`: Integer encoded brand identifier (`Brand.id`).
- `cost_price`: Base inventory unit cost (`Product.costPrice`).
- `selling_price`: Base retail selling price (`Product.sellingPrice`).
- `margin_rate`: `(selling_price - cost_price) / selling_price`.
- `reorder_level`: Catalog statutory minimum stock threshold (`Product.reorderLevel`).

### 5.4 Branch Attributes
- `branch_id`: Categorical encoding of the store branch (`Branch.id`).
- `branch_volume_tier`: Categorical tier (1 = Low, 2 = Medium, 3 = High Volume) derived from 90-day aggregate store revenue percentile.

### 5.5 Price Dynamics & Scheduled Promotions
- `price_index_28`: Current unit price divided by the 28-day historical rolling median price:
  $$\text{price\_index\_28} = \frac{\text{unit\_price}_{p, t}}{\operatorname{median}_{1 \le \tau \le 28}(\text{unit\_price}_{p, t-\tau}) + \epsilon}$$
- `has_scheduled_discount`: Binary flag (0 or 1) indicating if a `DiscountRule` is active on date $t$ for product $p$ (`startsAt <= t <= endsAt`).
- `discount_rate`: Percentage discount value applicable on date $t$.
- `discount_type`: Encoded type (`PERCENTAGE`, `FIXED_AMOUNT`, `UNIT_PRICE`).

### 5.6 Calendar & Exogenous Features
- `day_of_week`: Integer $0$ (Monday) to $6$ (Sunday).
- `day_of_month`: Integer $1$ to $31$.
- `month`: Integer $1$ to $12$.
- `is_weekend`: Binary flag ($1$ if Saturday or Sunday, else $0$).
- `is_payday_window`: Binary flag ($1$ if `day_of_month` $\in [25, 31]$ or $[1, 3]$).

### 5.7 Sri Lankan Holiday & Festive Calendar Features
Specific to TechNova POS operating in Sri Lanka:
- `is_public_holiday`: Binary flag for Sri Lankan gazetted public holidays.
- `is_mercantile_holiday`: Binary flag for commercial sector holidays.
- `is_poya_day`: Binary flag for monthly Full Moon Poya Day (strict alcohol/meat bans; grocery surges on prior day).
- `is_festive_peak`: Multi-day festive period indicator:
  - Sinhala & Tamil New Year (April 10–18).
  - Vesak Season (May).
  - Christmas & Year-End Shopping (December 15–31).
  - Ramadan & Eid shopping window.

---

## 6. Strict Data Leakage Prevention

To ensure valid generalization, the following boundary rules are strictly enforced:

```
[ Historical Observation Window: T_0 ... T-1 ] | [ Forecast Horizon: T ... T+H ]
───────────────────────────────────────────────┼──────────────────────────────
 ✅ Actual Sales (y)                           │ ❌ No Future Sales
 ✅ Historical Rolling Means/Std               │ ❌ No Future Rolling Stats on y
 ✅ Historical Prices                          │ ✅ Scheduled DiscountRule (startsAt/endsAt)
 ✅ Historical Returns/Refunds                 │ ❌ No Future DiscountApplication
                                               │ ✅ Future Calendar & Holiday Flags
```

1. **Temporal Horizon Shift:** Every rolling window calculation and lag must apply a shift of at least 1 day relative to forecast generation date $T$. When forecasting recursively for step $T + h$ ($h \ge 2$), lagged targets must use either the model's own prior predictions $\hat{y}_{T + h - 1}$ or direct multi-step models.
2. **Promotional Schedule vs. Redemption Outcome:**
   - **Allowed:** `DiscountRule.startsAt` and `DiscountRule.endsAt`. These are business decisions committed before date $t$.
   - **STRICTLY FORBIDDEN:** `DiscountApplication` or `SaleItem.discountTotal`. These reflect actual consumer redemptions and only exist *after* sales occur.
3. **Multi-Tenant Cross-Contamination:** No global features that aggregate across organizations. All product frequencies, rolling means, and rankings must be strictly grouped by `[organization_id, branch_id]`.

---

## 7. Model Candidates & Selection Trade-offs

Because TechNova POS operates in cost-effective cloud container environments (FastAPI on CPU), deep learning transformers (e.g., Temporal Fusion Transformers) are ruled out due to excessive inference latency and GPU dependencies. Four CPU-friendly architectures are evaluated:

| Model Architecture | Training Speed (100k rows) | CPU Inference Latency (Batch of 50 SKUs) | Memory Footprint | Handling of Categoricals & Zeros | Uncertainty / Quantiles |
|---|---|---|---|---|---|
| **LightGBM Regressor** *(Recommended)* | **< 3 seconds** | **< 25 ms** | **Very Low (< 80 MB)** | Native handling of categorical integer bins; optimal histogram splitting. | Supported natively via `objective="quantile"`. |
| **XGBoost Regressor** | ~8 seconds | ~45 ms | Low (~120 MB) | Requires one-hot or target encoding for categorical IDs. | Supported via custom quantile loss. |
| **HistGradientBoostingRegressor** | ~5 seconds | ~35 ms | Very Low (< 90 MB) | Native scikit-learn; no external binary dependencies. | Does not support simultaneous pinball/quantile output directly. |
| **CatBoost Regressor** | ~20 seconds | ~60 ms | Medium (~250 MB) | Best-in-class categorical encoding, but slower CPU training. | Supported natively. |

> **Design Recommendation:** **LightGBM Regressor** (or scikit-learn `HistGradientBoostingRegressor` as zero-dependency fallback) using Tweedie loss or $\log(1+y)$ objective. Model selection will be validated empirically during Phase 2 training.

---

## 8. Training & Validation Strategy

### 8.1 Chronological Split (No Random Shuffling)
Retail time-series records cannot be partitioned with uniform random splitting. A strict **chronological cutoff** is required:

```
Total Available Dataset (e.g., 365 Days)
[──────────────────────────── Train: 70% ───────────────────────────][── Val: 15% ──][── Test: 15% ──]
Day 1                                                              Day 255          Day 310         Day 365
```

- **Training Partition ($70\%$):** Days $1 \to T_{\text{train}}$ (learns tree splits).
- **Validation Partition ($15\%$):** Days $T_{\text{train}} + 1 \to T_{\text{val}}$ (early stopping and hyperparameter tuning).
- **Holdout Test Partition ($15\%$):** Days $T_{\text{val}} + 1 \to T_{\text{end}}$ (unseen evaluation benchmark).

### 8.2 Expanding-Window Cross-Validation (Rolling Origin)
For robust performance verification across different retail seasons:

```
Fold 1: [── Train: 180 Days ──][─ Val: 30 Days ─]
Fold 2: [──── Train: 210 Days ────][─ Val: 30 Days ─]
Fold 3: [────── Train: 240 Days ──────][─ Val: 30 Days ─]
Fold 4: [──────── Train: 270 Days ────────][─ Val: 30 Days ─]
```

---

## 9. Evaluation Metrics Matrix

The evaluation suite must report both aggregate accuracy and intermittent demand performance:

### 9.1 Weighted Absolute Percentage Error (WAPE)
Standard MAPE ($\frac{|y - \hat{y}|}{y}$) divides by zero on zero-demand days. WAPE is mathematically stable:
$$\text{WAPE} = \frac{\sum_{p, b, t} |y_{p, b, t} - \hat{y}_{p, b, t}|}{\sum_{p, b, t} y_{p, b, t}}$$

### 9.2 Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
$$\text{MAE} = \frac{1}{N} \sum |y - \hat{y}|, \qquad \text{RMSE} = \sqrt{\frac{1}{N} \sum (y - \hat{y})^2}$$

### 9.3 Forecast Value Added (FVA) against Seasonal Naive Baseline
Measures the percentage improvement delivered by the ML model over a simple 7-day seasonal lag baseline ($\hat{y}_{t} = y_{t-7}$):
$$\text{FVA} = 1 - \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{seasonal\_naive}}}$$
*Acceptance Criterion:* $\text{FVA} \ge 15\%$ on active SKUs.

### 9.4 Quantile / Pinball Loss (Probabilistic Evaluation)
For inventory safety buffer calculation, evaluating the $q \in \{0.50, 0.90\}$ quantiles:
$$L_q(y, \hat{y}_q) = \max\Big(q(y - \hat{y}_q), (1 - q)(\hat{y}_q - y)\Big)$$
- $\hat{y}_{0.50}$ ($P_{50}$): Median expected demand.
- $\hat{y}_{0.90}$ ($P_{90}$): Upper bound demand for $90\%$ service level protection.

---

## 10. FastAPI Service Contract

### 10.1 Endpoint Definition
- **Route:** `POST /v1/demand-forecast/forecast`
- **Tag:** `demand forecasting`
- **Location:** `technova-pos-ai/src/technova_ai_service/features/demand_forecasting/`

### 10.2 Batch Request Schema (`DemandForecastBatchRequest`)
```json
{
  "organization_id": "cuid_org_123",
  "branch_id": "cuid_branch_456",
  "forecast_horizon": 14,
  "start_date": "2026-09-25",
  "products": [
    {
      "product_id": "cuid_prod_001",
      "sku": "LAP-001",
      "category_id": "cat_electronics",
      "brand_id": "brand_asus",
      "cost_price": 180000.0,
      "selling_price": 220000.0,
      "reorder_level": 5.0,
      "current_stock": 8.0,
      "recent_daily_demand": [1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 1.0, 0.0, ... 28 values min]
    }
  ],
  "future_calendar": [
    {
      "date": "2026-09-25",
      "day_of_week": 4,
      "is_weekend": false,
      "is_holiday": false,
      "has_discount": true,
      "discount_rate": 0.10
    }
  ]
}
```

### 10.3 Batch Response Schema (`DemandForecastBatchResponse`)
```json
{
  "organization_id": "cuid_org_123",
  "branch_id": "cuid_branch_456",
  "forecast_horizon": 14,
  "generated_at": "2026-09-24T10:00:00.000Z",
  "predictions": [
    {
      "product_id": "cuid_prod_001",
      "sku": "LAP-001",
      "total_predicted_demand": 18.4,
      "average_daily_demand": 1.31,
      "peak_date": "2026-09-27",
      "peak_daily_demand": 3.2,
      "data_readiness": {
        "history_days": 90,
        "maturity": "developed",
        "confidence": "high",
        "production_ready": true
      },
      "daily": [
        {
          "date": "2026-09-25",
          "predicted_demand": 1.25,
          "lower_bound": 0.5,
          "upper_bound": 2.1
        }
      ]
    }
  ]
}
```

---

## 11. NestJS Integration Architecture

### 11.1 Extraction Service Design (Reusing Existing Patterns)
The backend leverages patterns established in `SalesDataExtractionService` to eliminate redundant SQL logic.

```
┌────────────────────────────────────────────────────────┐
│               NestJS AiIntelligenceService            │
└───────────────────────────┬────────────────────────────┘
                            │
              Calls Batched Extraction Method
                            ▼
┌────────────────────────────────────────────────────────┐
│             DemandDataExtractionService                │
│ (Follows SalesDataExtractionService Multi-Tenant Design)│
└───────────────────────────┬────────────────────────────┘
                            │
               Single Optimized Prisma Query
                            ▼
┌────────────────────────────────────────────────────────┐
│      Prisma: SaleItem JOIN Sale (status IN [...])      │
│      WHERE branchId = :branchId AND createdAt >= :T-90 │
└───────────────────────────┬────────────────────────────┘
                            │
          Constructs Dense [Product x Date] Matrix
                            │
               POST /v1/demand-forecast/forecast
                            ▼
┌────────────────────────────────────────────────────────┐
│                  FastAPI AI Service                    │
└────────────────────────────────────────────────────────┘
```

### 11.2 High-Performance Batch Extraction Query
Instead of iterating products in a loop:
```typescript
const historicalSales = await this.prisma.saleItem.groupBy({
  by: ['productId'],
  where: {
    sale: {
      branchId: targetBranchId,
      branch: { organizationId },
      status: { in: [SaleStatus.COMPLETED, SaleStatus.PARTIALLY_REFUNDED] },
      createdAt: { gte: historyStartDate },
    },
  },
  _sum: { quantity: true },
});
```
For daily granularity, NestJS queries `SaleItem` with `select: { productId: true, quantity: true, sale: { select: { createdAt: true } } }` in a single query bounded by `historyStartDate = today - 90 days`, then zero-fills the sparse date matrix in memory in $\mathcal{O}(N)$ time.

---

## 12. Stock Intelligence Integration Flow

The new Demand Forecasting model does **not** duplicate Stock Intelligence; it serves as its **foundational input**:

```
[ New Demand Forecasting ML ]
      │
      │ Output: Daily predictions { d_1, d_2, ... d_H } and uncertainty σ_d
      ▼
[ Existing Stock Intelligence Replenishment Policy ]
      │
      ├─► Lead Time Demand:   LTD = Σ_{t=1}^{L} d_t
      ├─► Safety Stock:       SS  = Z_{service_level} × σ_d × √L
      ├─► Reorder Point:      ROP = LTD + SS
      ├─► Target Stock:       S   = Σ_{t=1}^{L+R} d_t + SS
      └─► Suggested Order:    ROQ = max(S - CurrentStock, 0)
```

By supplying high-accuracy, branch-specific demand forecasts to the replenishment policy, TechNova eliminates the dependency on the generic UCI Online Retail heuristic curve while preserving the entire existing inventory decision framework.

---

## 13. Frontend UI Integration

### 13.1 Single Source of Truth in UI
- **Zero Chart Duplication:** The system will **NOT** create a second demand chart.
- **Target Component:** [ProductForecastPanel.tsx](file:///d:/Computer%20Science/Semester%2006/connex/pos/Project/technova-pos/src/components/dashboard/ai-intelligence/AIIntelligenceClientView.tsx#L141).
- **Modification Plan:**
  1. The existing Recharts line chart inside `ProductForecastPanel` will bind to `product.demand_forecast.daily` (generated by the new ML model).
  2. Add an area band showing probabilistic bounds (`[lower_bound, upper_bound]`).
  3. Display a subtle data maturity badge: `"Branch History: 90 days (Production Ready)"` or `"Cold-start forecast (< 28 days)"`.

---

## 14. Performance & Scalability Design

1. **Vectorized Matrix Inference:** The FastAPI inference handler processes all SKUs simultaneously using 2D NumPy array / Pandas DataFrame slicing, avoiding per-SKU Python loops.
2. **Inference Latency SLA:**
   - Single SKU: $< 15\text{ ms}$.
   - Batch of 50 SKUs: $< 80\text{ ms}$.
   - Batch of 200 SKUs: $< 220\text{ ms}$.
3. **Payload Optimization:** Compressed numerical arrays in JSON; timestamps truncated to `YYYY-MM-DD`.

---

## 15. Artifact Storage Structure

The model artifact is stored cleanly in the designated repository path:
```
technova-pos-ai/
  └── artifacts/
      └── demand_forecasting/
          ├── model.joblib          <-- Serialized LightGBM/GBR model bundle
          ├── metrics.json          <-- Test set evaluation benchmarks
          └── metadata.json         <-- Feature list, categorical maps, train dates
```

**Joblib Bundle Schema:**
```python
{
    "model": fitted_model_instance,
    "feature_columns": [ ... list of 28 feature names ... ],
    "categorical_features": [ "category_id", "brand_id", "branch_id", "day_of_week" ],
    "target": "daily_product_quantity",
    "version": "1.0.0",
    "trained_at": "2026-09-25T00:00:00Z"
}
```

---

## 16. Comprehensive Verification & Testing Strategy

```
┌────────────────────────────────────────────────────────┐
│                     TESTING SUITE                      │
├───────────────────┬───────────────────┬────────────────┤
│ 1. Schema Tests   │ 2. Leakage Tests  │ 3. Inference   │
│    Pydantic valid │    Strict shift   │    Batch match │
│    Bound checks   │    No future disc │    Non-negative│
├───────────────────┼───────────────────┼────────────────┤
│ 4. Unit Tests     │ 5. Cold-Start     │ 6. Integration │
│    Zero-fill logic│    Graceful fallbk│    NestJS→Fast │
│    WAPE formula   │    Maturity flags │    E2E Status  │
└───────────────────┴───────────────────┴────────────────┘
```

1. **Schema & Contract Tests:**
   - Validate that `DemandForecastBatchRequest` strictly rejects negative quantities, invalid date strings, and unsupported horizons ($H \notin \{7, 14, 30\}$).
2. **Temporal Leakage Unit Tests:**
   - Verify that shifting operations guarantee $X_t$ uses only $\{y_\tau \mid \tau \le t - 1\}$.
   - Verify that future discount inputs only reflect planned `startsAt` schedules, never actual sale redemptions.
3. **Inference Correctness Tests:**
   - Ensure all output demand predictions satisfy $\hat{y}_{p, b, t} \ge 0.0$ (no negative retail demand).
   - Ensure output array length strictly matches the requested `forecast_horizon`.
4. **Cold-Start Fallback Tests:**
   - Mock a product with 7 days of history: assert response status code 200 with `maturity: "cold_start"` and fallback statistical predictions.
5. **End-to-End Integration Tests:**
   - Call NestJS `GET /api/v1/ai-intelligence/overview` $\to$ verify response contains native product predictions generated via FastAPI batch inference.

---

## 17. Summary Checklist for Implementation Phases

- [ ] **Phase 1 (Data & Features):** Build `DemandDataExtractionService` in NestJS; create continuous daily panel builder in Python with zero-fill semantics.
- [ ] **Phase 2 (Model Training):** Train candidate models (LightGBM vs. HistGradientBoosting) on TechNova POS panel; evaluate WAPE, MAE, FVA; serialize to `artifacts/demand_forecasting/model.joblib`.
- [ ] **Phase 3 (FastAPI Service):** Implement `POST /v1/demand-forecast/forecast` batch endpoint with Pydantic validation.
- [ ] **Phase 4 (NestJS & Stock Intelligence Wiring):** Feed demand forecast into `StockIntelligenceService` replenishment formula.
- [ ] **Phase 5 (Frontend):** Update `ProductForecastPanel` in `AIIntelligenceClientView.tsx` with ML predictions and uncertainty bands.

---
*End of Technical Design Specification.*

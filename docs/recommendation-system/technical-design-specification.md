# Technical Design Specification: TechNova POS AI Recommendation Engine

**Feature Name:** Multi-Context Stock-Aware Product Recommendation Engine  
**Module Target:** `technova-pos-ai` (FastAPI), `technova-pos-api` (NestJS), `technova-pos` (Next.js)  
**Status:** DRAFT / DESIGN SPECIFICATION  
**Author:** TechNova POS AI Architecture & Engineering Team  
**Date:** September 2026  
**Document Version:** 1.0.0  

---

## 1. Overview & System Purpose

Retail Point of Sale (POS) environments require actionable, real-time product recommendations to maximize basket size, improve cross-sell/up-sell velocity, and assist sales cashiers with context-sensitive recommendations. However, traditional e-commerce recommendation systems assume infinite stock availability and uniform customer identities, failing completely in brick-and-mortar retail where:
1. **Physical inventory varies by branch:** Recommending an out-of-stock item damages cashier efficiency and customer trust.
2. **Context shifts rapidly:** A transaction may involve a known loyalty customer, an anonymous walk-in, an item currently being scanned on the POS terminal, or branch-specific inventory clearances.
3. **Retail domains vary widely:** The POS platform must serve computer shops, electronics stores, supermarkets, mobile phone retailers, and fashion boutiques without requiring domain-specific re-architecture.

The **TechNova AI Recommendation Engine** is a multi-tenant, context-aware, stock-governed hybrid recommendation system designed to generate ranked product suggestions with deterministic, transparent rationale across any retail vertical.

---

## 2. Core Goals & Capabilities

1. **Context-Driven Retrieval:** Deliver high-precision recommendations across 6 operational contexts: Customer, Product, Branch, Trending, Anonymous Cold Start, and Cart-Ready bundles.
2. **Domain-Agnostic Extensibility:** Support diverse catalog structures (e.g., PC hardware socket compatibility, phone charger pin types, supermarket grocery categories) using a generic, schema-free metadata representation.
3. **Stock & Status Governance:** Automatically prune inactive, discontinued, and out-of-stock items for the targeted branch in real-time, optionally suggesting available in-stock substitutes.
4. **Deterministic Explainability:** Provide human-readable, verifiable justifications for every recommended item (e.g., *"Frequently bought with this item"*, *"Trending at Colombo Branch"*, *"Based on your previous purchases"*).
5. **Multi-Tenant Isolation:** Enforce absolute organizational and branch tenant boundaries at the database, memory, and API layers.
6. **Sub-100ms Inference Latency:** Vectorized scoring and retrieval suitable for instant POS checkout line execution.

---

## 3. Explicit Non-Goals

1. **Autonomous Cart Modification:** The engine outputs recommendations with advisory confidence scores; it does not automatically add products to an active POS cart or alter transaction line items without cashier/user action.
2. **Dynamic Price Adjustments:** Price optimization remains the sole responsibility of the existing Dynamic Pricing module (`/v1/pricing/recommend`). Recommendations ingest current selling prices but do not adjust them.
3. **Deep Reinforcement Learning / Real-time Bandits:** Online exploration policies (multi-armed bandits) that display sub-optimal items to test cashier response are out of scope. Initial production utilizes batch-trained representations and online hybrid re-ranking.
4. **Customer PII Storage in AI Service:** The AI service (`technova-pos-ai`) operates on anonymized IDs (`customer_id`, `product_id`, `branch_id`). Customer names, emails, and phone numbers remain strictly encapsulated within the NestJS API layer (`technova-pos-api`).

---

## 4. Recommendation Contexts

The engine responds to distinct operational scenarios at the POS counter:

| Context ID | Trigger / Input | Primary Business Objective | Example Scenario |
| :--- | :--- | :--- | :--- |
| **`CUSTOMER`** | `customer_id` | Loyalty retention, repeat purchases, personalized catalog expansion. | Regular customer scans loyalty card; cashier sees complementary products tailored to their purchase history. |
| **`PRODUCT`** | `product_id` | Upselling, cross-selling, accessory discovery, alternative SKU substitution. | Cashier scans an Intel Core i7 CPU; system recommends compatible motherboards and thermal paste. |
| **`BRANCH`** | `branch_id` | Hyper-local demand matching, regional preferences, store assortment optimization. | Regional branch in Kandy features top-selling items specific to local climate and footfall patterns. |
| **`TRENDING`** | Date window / velocity | Exploiting sudden spikes in consumer interest, promotional surges, seasonal waves. | High-velocity products experiencing rapid multi-day sales acceleration across the active retail calendar. |
| **`COLD_START`** | No identifiers / new tenant | Graceful fallback for walk-in anonymous shoppers or freshly onboarded branches. | Anonymous cashier walk-in; system serves universally high-converting, high-margin, top-rated products. |
| **`CART_READY`** | List of `[product_id]` in cart | Multi-item basket completion and checkout-line impulse additions. | Customer has laptop and mouse in cart; system recommends laptop sleeve and USB-C multiport hub. |

---

## 5. Recommendation Types & Business Logic

### A. Personalized Recommendations
- **Input:** `customer_id`, `organization_id`, `branch_id`.
- **Logic:** Identifies customer affinity clusters from past transactions, predicts next-to-buy categories, and boosts items frequently reordered by similar customer segments (RFM alignment).
- **Fallback:** If customer history is `< 3` transactions, blends smoothly into `COLD_START` blended recommendations.

### B. Frequently Bought Together (FBT)
- **Input:** `product_id` (or active basket array).
- **Logic:** Evaluates pairwise and multi-item transaction co-occurrence. Calculated via Association Rule Mining (Support, Confidence, and Lift metrics) and Item-to-Item co-purchase matrices.
- **Rule:** Lift must exceed $1.2$ to eliminate spurious correlations.

### C. Similar / Related Products
- **Input:** `product_id`.
- **Logic:** Content-based cosine similarity across category hierarchy, brand tier, price band ($\pm 25\%$), and specification embeddings.
- **Rule:** If the anchor product is out-of-stock, this type acts as the **Substitute Product Generator**.

### D. Trending Products
- **Input:** `branch_id`, optional `category_id`.
- **Logic:** Short-term velocity metric:
  $$\text{Velocity Ratio} = \frac{\text{Sales}_{t-7 \to t}}{\text{Sales}_{t-28 \to t-7} \times 0.25 + \epsilon}$$
  Surfaces products where 7-day velocity outpaces the 28-day baseline by the widest margin.

### E. Branch-Aware Recommendations
- **Input:** `branch_id`.
- **Logic:** Localized popularity and conversion rates. Accounts for differing demographics between rural vs. urban stores, tourist corridors, or specialty branches.

### F. Cold Start Handling
- **New Customer:** Uses branch top-sellers filtered by current time of day / day of week.
- **New Product (Zero sales):** Content-based projection linking new product specifications to existing mature items in the same category.
- **New Branch:** Organizational top-sellers filtered by matching store format (`store_type`).

### G. Stock-Aware Recommendations
- **Mandatory Filter:** Candidate list is inner-joined against current branch inventory levels.
- **Pruning Policy:** Any product with `current_stock <= 0` or `status != 'ACTIVE'` is excluded from the primary output and routed to alternative substitution pipelines.

---

## 6. Generic Product & Attribute Model

To prevent domain lock-in (such as hardcoding PC hardware components), the engine uses a flexible two-tier attribute model:

```
┌────────────────────────────────────────────────────────┐
│                   Generic Product Model                │
├──────────────────────────┬─────────────────────────────┤
│ Core Rigid Schema        │ Dynamic Extensible Metadata │
│ (All Retail Domains)     │ (Domain-Specific JSONB)     │
├──────────────────────────┼─────────────────────────────┤
│ • product_id (string)    │ • attributes: {             │
│ • organization_id (str)  │     "socket": "LGA1700",    │
│ • sku (string)           │     "form_factor": "ATX",   │
│ • name (string)          │     "power_watts": 65,      │
│ • category_path (str[])  │     "ram_type": "DDR5",     │
│ • brand (string)         │     "screen_size": "6.1",   │
│ • base_price (decimal)   │     "weight_grams": 500,    │
│ • current_price (decimal)│     "organic": true         │
│ • status (ACTIVE/...)    │   }                         │
│ • branch_stock (dict)    │ • compatibility_keys: [...] │
└──────────────────────────┴─────────────────────────────┘
```

### Domain Mapping Examples

| Attribute Dimension | Computer Shop | Mobile Phone Shop | Supermarket | Fashion Boutique |
| :--- | :--- | :--- | :--- | :--- |
| **Category Tier 1** | Components | Devices | Groceries | Apparel |
| **Category Tier 2** | Motherboards | Smartphones | Dairy | Men's Clothing |
| **Category Tier 3** | Intel Motherboards | 5G Android | Fresh Milk | Formal Shirts |
| **Brand** | ASUS, MSI | Samsung, Apple | Anchor, Highland | Emerald |
| **Dynamic Spec 1** | `chipset: "Z790"` | `charging_port: "Type-C"` | `fat_content: "Full Cream"` | `fit: "Slim Fit"` |
| **Dynamic Spec 2** | `form_factor: "ATX"` | `screen_size: "6.7"` | `shelf_life_days: 14` | `collar_size: "16.5"` |
| **Compatibility Tag**| `["socket:LGA1700", "ram:DDR5"]` | `["case:iPhone15Pro", "watt:20W"]` | `["recipe:baking", "diet:dairy"]` | `["collection:formal2026"]` |

---

## 7. Data Strategy: Augmented Behavioral Synthetic Dataset

### 7.1 Real-World Foundation & Synthetic Augmentation
TechNova POS does not currently possess millions of historical transaction receipts with cross-item customer baskets. Following the established pattern from Demand Forecasting Phase 3:

```
┌───────────────────────────────┐     ┌───────────────────────────────────┐
│     Rossmann Retail Data      │     │  Synthetic Behavioral Generator   │
│  (Real Retail Footfall/Sales) │     │    (SKUs, Baskets, Attributes)    │
└───────────────┬───────────────┘     └─────────────────┬─────────────────┘
                │                                       │
                ▼                                       ▼
        ┌───────────────────────────────────────────────────────┐
        │  Augmented Behavioral Synthetic Recommendation Dataset│
        │  (Preserves real seasonal, store, and weekly signals; │
        │   augments realistic multi-item customer baskets)     │
        └───────────────────────────────────────────────────────┘
```

- **From Rossmann (Real signals):**
  - Store IDs and store types (`a`, `b`, `c`, `d`).
  - True chronological dates (2013–2015), day of week, seasonal cycles.
  - Real customer transaction counts per store per day (`Customers`).
  - Promotional periods (`Promo`, `Promo2`).
  - School and state holiday retail behavior.
- **From Synthetic Generator (Simulated retail behavior):**
  - Realistic customer cohorts (10,000 distinct customer profiles with varied RFM loyalty segments).
  - Cohesive product catalog (500 SKUs distributed across 5 major categories with realistic pricing and specifications).
  - Basket formation rules: Market basket distribution modeling affinity, co-purchasing probabilities, and price sensitivity.

---

## 8. Dataset Schema & Architecture

The recommendation engine dataset is partitioned into three relational tables stored as high-performance Parquet artifacts in `data/processed/recommendations/`:

### 8.1 Catalog & Product Metadata (`catalog_products.parquet`)
```sql
product_id           VARCHAR(64) PRIMARY KEY,
organization_id      VARCHAR(64) NOT NULL,
sku                  VARCHAR(64) NOT NULL,
name                 VARCHAR(255) NOT NULL,
category_level_1     VARCHAR(64) NOT NULL,
category_level_2     VARCHAR(64) NOT NULL,
category_level_3     VARCHAR(64) NOT NULL,
brand                VARCHAR(64) NOT NULL,
cost_price           DOUBLE PRECISION NOT NULL,
base_unit_price      DOUBLE PRECISION NOT NULL,
current_unit_price   DOUBLE PRECISION NOT NULL,
specifications_json  TEXT NOT NULL,          -- Serialized JSON of dynamic specs
compatibility_tags   TEXT[] NOT NULL,        -- Array of compatibility tokens
is_active            BOOLEAN NOT NULL DEFAULT TRUE
```

### 8.2 Customer Profiles (`customer_profiles.parquet`)
```sql
customer_id          VARCHAR(64) PRIMARY KEY,
organization_id      VARCHAR(64) NOT NULL,
customer_number      VARCHAR(32) NOT NULL,
segment              VARCHAR(32) NOT NULL,   -- VIP, REGULAR, AT_RISK, OCCASIONAL
total_transactions   INTEGER NOT NULL,
lifetime_spend       DOUBLE PRECISION NOT NULL,
avg_basket_size      DOUBLE PRECISION NOT NULL,
preferred_categories TEXT[] NOT NULL,
affinity_brands      TEXT[] NOT NULL,
last_purchase_date   DATE NOT NULL
```

### 8.3 Transaction Baskets (`transaction_baskets.parquet`)
```sql
transaction_id       VARCHAR(64) NOT NULL,
organization_id      VARCHAR(64) NOT NULL,
store_id             VARCHAR(64) NOT NULL,
customer_id          VARCHAR(64) NULLABLE,   -- NULL indicates anonymous walk-in
transaction_date     DATE NOT NULL,
timestamp            TIMESTAMP NOT NULL,
line_item_id         INTEGER NOT NULL,
product_id           VARCHAR(64) NOT NULL,
quantity             INTEGER NOT NULL,
unit_price           DOUBLE PRECISION NOT NULL,
line_total           DOUBLE PRECISION NOT NULL,
is_promo_applied     BOOLEAN NOT NULL,
PRIMARY KEY (transaction_id, line_item_id)
```

---

## 9. Recommendation Algorithms: Comparative Analysis

| Algorithm | Method Class | Strengths | Weaknesses | Cold Start Resilience | Computational Latency | Suitable Role in TechNova |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **FP-Growth / Association Rules** | Frequent Pattern Mining | Deterministic; highly explainable; finds true checkout co-occurrences. | Fails for rare items; ignores customer preference nuances. | Low (Needs basket history) | Offline: Medium<br>Online: **< 5ms** (Hash index) | **Primary for FBT & Cart-Ready** |
| **Item-to-Item Cosine Similarity** | Collaborative / Co-Occurrence | Symmetric item affinity; captures broader substitution patterns. | High memory footprint for massive catalogs ($O(N^2)$). | Low (Items must be purchased) | Offline: High<br>Online: **< 10ms** (Top-K index) | **Candidate Generator for Product Context** |
| **Content-Based Spec Vectorization** | TF-IDF / Embedding Cosine | Works with zero sales history; matches technical specs and brands. | Doesn't capture behavioral popularity or unexpected affinity. | **Excellent** (Uses metadata only) | Offline: Low<br>Online: **< 15ms** | **Primary for Cold-Start & Alternatives** |
| **Matrix Factorization (Implicit ALS)** | Collaborative Filtering | Uncovers latent customer-item affinities; high personalization. | "Black-box" embeddings; difficult to explain simply to a cashier. | Poor (Fails on new users/items) | Offline: High<br>Online: **< 20ms** | **Candidate Generator for Customer Context** |
| **Decayed Velocity Ranking** | Heuristic / Statistical | Surfaces fast-moving trends; extremely robust and simple. | Not personalized; ignores product complementarity. | **High** (Aggregates store volume) | Offline: Very Low<br>Online: **< 2ms** | **Primary for Trending & Fallback** |

---

## 10. Selected Multi-Stage Hybrid Architecture

TechNova POS adopts a **Two-Stage Hybrid Architecture** (Candidate Generation $\to$ Multi-Signal Re-Ranking & Stock Filtering):

```
                   Incoming Recommendation Request
         (Context: Customer, Product, Branch, or Cart)
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 STAGE 1: Candidate Generation               │
  │                     (Generates Top-50 Items)                │
  ├────────────────────────┬────────────────────────────────────┤
  │ FBT Association Rules  │ Item-to-Item Affinity Matrix       │
  │ (Frequent Co-purchase) │ (Collaborative Co-occurrence)      │
  ├────────────────────────┼────────────────────────────────────┤
  │ Content Similarity     │ Branch Velocity / Popularity       │
  │ (Specs, Brand, Price)  │ (Trending / Best Sellers)          │
  └────────────────────────┴────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         STAGE 2: Multi-Signal Scoring & Context Weighting   │
  │   S_final(i) = w1*S_fbt + w2*S_collab + w3*S_sim +          │
  │                w4*S_trend + w5*S_user + w6*S_compat         │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │       STAGE 3: Real-Time Stock & Business Rule Filtering    │
  │   • Filter: branch_stock[i] > 0                             │
  │   • Filter: status == 'ACTIVE'                              │
  │   • Substitute out-of-stock items with highest S_sim        │
  │   • Deterministic Explainability String Assignment          │
  └─────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
                   Final Top-K Ranked Response
                   (Default: Top-5 Recommendations)
```

---

## 11. Scoring Formulation & Weighting Methodology

### 11.1 Signal Normalization
All component signals are min-max normalized to $[0.0, 1.0]$:
1. **$S_{\text{FBT}}$ (Co-purchase Lift):** Normalized association confidence $\times$ normalized lift.
2. **$S_{\text{Collab}}$ (Item-to-Item Similarity):** Jaccard/Cosine co-occurrence coefficient between items.
3. **$S_{\text{Content}}$ (Specification & Category Similarity):** Weighted token and category depth overlap.
4. **$S_{\text{Customer}}$ (Customer Affinity):** Dot product of customer category spend vector and item category.
5. **$S_{\text{Trend}}$ (Branch Velocity):** Relative 7-day velocity ratio normalized across active catalog.
6. **$S_{\text{Compat}}$ (Strict Hardware/Retail Compatibility):** Binary multiplier $\{0.0, 1.0\}$ or fractional score if optional.

### 11.2 Context-Adaptive Weighting Matrix
Weights shift dynamically based on the primary request context:

| Context | $w_{\text{FBT}}$ | $w_{\text{Collab}}$ | $w_{\text{Content}}$ | $w_{\text{Customer}}$ | $w_{\text{Trend}}$ | $w_{\text{Compat}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CUSTOMER`** | 0.15 | 0.15 | 0.10 | **0.40** | 0.10 | 0.10 |
| **`PRODUCT`** | **0.35** | 0.25 | 0.20 | 0.00 | 0.10 | 0.10 |
| **`BRANCH`** | 0.10 | 0.10 | 0.05 | 0.05 | **0.50** | 0.20 |
| **`TRENDING`** | 0.05 | 0.05 | 0.05 | 0.05 | **0.80** | 0.00 |
| **`COLD_START`**| 0.00 | 0.00 | 0.30 | 0.00 | **0.60** | 0.10 |
| **`CART_READY`**| **0.45** | 0.20 | 0.15 | 0.10 | 0.05 | 0.05 |

*Note: All weight sets satisfy $\sum w_i = 1.0$. The weights are configurable via environment/config without code re-deployment.*

---

## 12. Explainability Engine

Every returned recommendation includes an explicit `reason` string and an analytical `reason_code`. The reason is assigned to the component signal contributing the highest weighted score to the final ranking:

| Dominant Signal | Assigned `reason_code` | Customer-Facing Explanation (`reason`) |
| :--- | :--- | :--- |
| **Highest $w_{\text{FBT}} \cdot S_{\text{FBT}}$** | `FREQUENTLY_BOUGHT_TOGETHER` | *"Frequently purchased together with [Anchor Product]"* |
| **Highest $w_{\text{Customer}} \cdot S_{\text{Customer}}$** | `PAST_PURCHASE_AFFINITY` | *"Based on your preferred brand and category history"* |
| **Highest $w_{\text{Content}} \cdot S_{\text{Content}}$** | `SIMILAR_PRODUCT` | *"Similar specifications to items you viewed"* |
| **Highest $w_{\text{Trend}} \cdot S_{\text{Trend}}$** | `CURRENTLY_TRENDING` | *"Trending item with high demand this week"* |
| **Highest $w_{\text{Branch}} \cdot S_{\text{Branch}}$** | `BRANCH_POPULAR` | *"Top seller at this branch"* |
| **Compatibility Match** | `VERIFIED_COMPATIBLE` | *"Verified compatible with your selected item"* |
| **Out-of-Stock Substitute** | `IN_STOCK_ALTERNATIVE` | *"In-stock alternative to [Original Item]"* |

---

## 13. Stock-Aware Filtering & Inventory Integration

Recommendations are filtered against the local branch inventory state.

### 13.1 Filter Sequence
```
Candidate Product ID
       │
       ├──► 1. Is Product ACTIVE in Organization?
       │         └── No ──► Discard
       │
       ├──► 2. Is Product Assigned to Target Branch?
       │         └── No ──► Discard
       │
       ├──► 3. Branch Current Stock > 0?
       │         ├── Yes ──► Retain candidate
       │         └── No  ──► Route to Substitute Generator
       │                          │
       │                          ▼
       │                     Find highest Content Similarity
       │                     item with Branch Stock > 0
       │
       └──► 4. Output to Final Top-K Ranked List
```

### 13.2 Integration with Existing Stock Intelligence
The engine reuses the branch inventory query contracts from `StockIntelligenceService` (`currentStock`, `reorderPoint`) to avoid maintaining redundant stock caches.

---

## 14. API Design (`technova-pos-ai`)

### 14.1 Route Definition
`POST /v1/recommendations/recommend`

### 14.2 Request Schema (`RecommendationRequest`)
```json
{
  "organization_id": "org_technova_01",
  "branch_id": "branch_colombo_main",
  "context": "PRODUCT",
  "customer_id": "cust_8923",
  "product_id": "prod_cpu_intel_14700k",
  "cart_product_ids": ["prod_cpu_intel_14700k", "prod_ram_ddr5_32gb"],
  "top_k": 5,
  "include_out_of_stock": false
}
```

### 14.3 Response Schema (`RecommendationResponse`)
```json
{
  "organization_id": "org_technova_01",
  "branch_id": "branch_colombo_main",
  "context": "PRODUCT",
  "generated_at": "2026-09-25T01:30:00Z",
  "total_candidates_evaluated": 48,
  "recommendations": [
    {
      "product_id": "prod_mb_asus_z790",
      "sku": "MB-ASUS-Z790-PLUS",
      "name": "ASUS TUF Gaming Z790-Plus WiFi",
      "category": "Motherboards",
      "brand": "ASUS",
      "current_price": 89500.0,
      "score": 0.942,
      "reason_code": "FREQUENTLY_BOUGHT_TOGETHER",
      "reason": "Frequently purchased together with Intel Core i7-14700K",
      "stock_status": "IN_STOCK",
      "available_stock": 14,
      "metadata": {
        "socket": "LGA1700",
        "form_factor": "ATX"
      }
    },
    {
      "product_id": "prod_cooler_noctua_d15",
      "sku": "FAN-NOC-NH-D15",
      "name": "Noctua NH-D15 Chromax.Black CPU Cooler",
      "category": "Cooling",
      "brand": "Noctua",
      "current_price": 38500.0,
      "score": 0.887,
      "reason_code": "VERIFIED_COMPATIBLE",
      "reason": "Verified compatible cooler for Intel Core i7-14700K",
      "stock_status": "IN_STOCK",
      "available_stock": 8,
      "metadata": {
        "socket_support": ["LGA1700", "AM5"],
        "tdp_rating": "250W"
      }
    }
  ],
  "model_metadata": {
    "engine_version": "1.0.0",
    "algorithms_used": ["FP-Growth", "Item-to-Item Cosine", "Stock-Aware Re-ranking"],
    "weights": {
      "fbt": 0.35,
      "collab": 0.25,
      "content": 0.20,
      "trend": 0.10,
      "compat": 0.10
    }
  }
}
```

---

## 15. NestJS Integration (`technova-pos-api`)

```
   Client (Web POS)
          │
          │ JWT Bearer Token (Authenticated Cashier/Manager)
          ▼
   AiIntelligenceController
   @Post('recommendations')
   @RequirePermissions('dashboard:view')
          │
          ▼
   AiIntelligenceService.getRecommendations()
          ├── 1. Resolve & enforce authenticated organizationId from req.user
          ├── 2. Validate branchId belongs to organizationId
          ├── 3. If productId provided, verify tenant ownership
          ├── 4. If customerId provided, verify tenant ownership
          ├── 5. Forward validated payload to FastAPI:
          │      POST http://127.0.0.1:8000/v1/recommendations/recommend
          ├── 6. Ingest FastAPI response
          ├── 7. Enrich response with active database stock quantities & currency formatting
          └── 8. Return mapped RecommendationResult to frontend
```

---

## 16. Frontend Design (`technova-pos`)

A dedicated tab will be added to the AI Intelligence navigation bar:

```
AI Intelligence
├── Overview
├── Sales Forecast
├── Demand Forecast
├── Recommendations [NEW TAB]
├── Inventory
├── Dynamic Pricing
├── Loyalty
└── Business Assistant
```

### UI Components for Recommendations Tab:
1. **Context Control Bar:**
   - Context Selector Buttons: `Customer` | `Product` | `Branch` | `Trending`.
   - Dynamic Target Selector:
     - When *Customer* selected: Customer search/select dropdown.
     - When *Product* selected: Product search/select dropdown.
     - When *Branch* selected: Branch selector dropdown.
     - When *Trending* selected: Horizon period (7, 14, 30 days).
2. **Recommendation Cards Grid:**
   - Ranked product cards with badge indicating `#1 Recommendation`, `#2 Recommendation`, etc.
   - Product SKU, Name, Brand, and Category.
   - Price formatted in LKR.
   - Dynamic Stock Pill (`In Stock: 14 units` in emerald or `Low Stock: 2 units` in amber).
   - Prominent **Reason Pill** with sparkler icon (e.g., *"Frequently bought with this item"*).
   - Single-click action button: `Add to Active Sale` or `Create Purchase Order` if reorder is needed.

---

## 17. Evaluation Methodology & Metrics

Evaluation occurs offline using strict chronological train/test splits to eliminate data leakage.

### 17.1 Chronological Split Strategy
- **Train Period (70%):** Baskets from Date $T_0$ to $T_{\text{train}}$.
- **Validation Period (15%):** Baskets from $T_{\text{train}}$ to $T_{\text{val}}$.
- **Test Period (15%):** Future holdout baskets from $T_{\text{val}}$ to $T_{\text{end}}$.

### 17.2 Metrics Suite
1. **Precision@K & Recall@K ($K \in \{3, 5, 10\}$):** Proportion of actual held-out items purchased by the customer/basket captured in the Top-K recommendations.
2. **Hit Rate@K:** Proportion of test baskets where at least one recommended item was genuinely purchased.
3. **Mean Average Precision (MAP@K):** Evaluates rank position quality of relevant items.
4. **Normalized Discounted Cumulative Gain (NDCG@K):** Emphasizes placing highest-relevance items at rank 1.
5. **Catalog Coverage (%):** Percentage of unique catalog SKUs recommended across the test population (guards against popularity bias collapse).
6. **Stock Availability Rate (%):** Percentage of recommended items that were physically in stock at recommendation time (Target: 100%).

---

## 18. Security & Multi-Tenant Isolation

1. **Logical Isolation:** Every query, candidate set, matrix lookup, and cached index is keyed by `organization_id`. Cross-tenant candidate pooling is physically impossible.
2. **FastAPI Loopback Defense:** The FastAPI microservice listens strictly on loopback (`127.0.0.1:8000`) and has zero public internet exposure. External access is only possible via NestJS authentication guards (`JwtAuthGuard`, `PermissionsGuard`).
3. **Input Sanitization:** Product IDs and customer IDs are validated against regex alphanumeric constraints to prevent injection in specification filters.
4. **Data Privacy:** Customer names, loyalty points, contact information, and billing details are never serialized into the AI service model.

---

## 19. Cold-Start Strategy

| Cold-Start Scenario | Primary Fallback Mechanism | Secondary Fallback Mechanism |
| :--- | :--- | :--- |
| **New Customer** (0 past purchases) | Branch-specific Top-10 Sellers (7-day velocity) | Overall Organization Trending Items |
| **New Product** (0 transactions) | Content-Based Specification Matching against existing catalog | Placement in Category New-Arrival Banners |
| **New Branch** (Fresh store opening) | Chain-wide Top Sellers matched by Store Type (`store_type`) | High-margin core catalog items |
| **Anonymous Walk-In POS Sale** | Cart-Item Association Rules (if $\ge 1$ item scanned) | Counter Impulse / Trending Bestsellers |

---

## 20. Future Extensibility

1. **POS Terminal Barcode Scanner Hook:** As cashiers scan barcodes in `/pos`, an asynchronous debounced trigger can request `CART_READY` recommendations to display impulse upsells directly beside the payment summary.
2. **Customer Mobile Loyalty App Integration:** The same `POST /v1/recommendations/recommend` endpoint can feed personalized offers into a future customer-facing e-commerce or loyalty mobile app.
3. **E-Commerce Online Storefront:** Headless commerce API sharing the identical association and stock engine.

---

## 21. Phased Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Requirements & Technical Design (Current)          │
│ • Deliverable: Technical design specification document      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Recommendation Dataset Generation & Validation     │
│ • Generate augmented Rossmann transaction basket dataset    │
│ • Build catalog with extensible specs & synthetic customers │
│ • Automated data validation gates                           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Recommendation Algorithm Development & Training    │
│ • Implement FP-Growth FBT rule miner                        │
│ • Implement Content/Spec similarity vectorizer              │
│ • Build hybrid re-ranking pipeline & offline evaluation     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: FastAPI Recommendation Microservice                │
│ • Implement schemas, router, service, warmup cache          │
│ • POST /v1/recommendations/recommend endpoint              │
│ • Automated pytest test suite                               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: NestJS API Gateway Integration                     │
│ • DTOs, controller endpoint, tenant isolation guards        │
│ • Service communication with FastAPI, Jest test suite       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: Frontend UI Implementation (AI Intelligence Tab)   │
│ • Recommendations tab in AI Intelligence navigation         │
│ • Context selectors (Customer/Product/Branch/Trending)      │
│ • Ranked product cards with reason badges & stock status    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 7: End-to-End Verification & Performance Audit        │
│ • Full E2E verification across all contexts                 │
│ • Latency benchmarking (<100ms) & zero regressions check    │
└─────────────────────────────────────────────────────────────┘
```

---

## 22. Acceptance Criteria

1. **Architecture Completeness:** Technical design covers all 6 contexts, 7 recommendation types, and generic product model without hardcoding single-domain assumptions.
2. **Domain Neutrality:** Dynamic JSON specifications allow seamless deployment to computer hardware, mobile phones, supermarkets, or apparel stores.
3. **Stock Safety:** 100% of generated recommendations in branch context have `available_stock > 0`.
4. **Explainability Compliance:** 100% of recommended items feature a verified `reason` and `reason_code` directly mapping to the primary scoring signal.
5. **Tenant Isolation:** Zero recommendation leakage between organizations.
6. **Test Coverage:** Future implementation phases must achieve $\ge 80\%$ test coverage on both FastAPI and NestJS services.

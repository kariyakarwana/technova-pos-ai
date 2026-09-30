# Phase 4.2: Model Selection and Deployment Decision Record

## 1. Purpose

This document formally records the model evaluation, selection decision, and deployment candidate for the TechNova POS Sales Forecasting engine. The evaluation compares three forecasting approaches on both the June 2015 validation window and the untouched July 2015 final holdout period using predefined, leakage-safe error metrics.

---

## 2. Temporal Evaluation Periods

The evaluation strictly maintains chronological integrity without temporal overlap:

| Period | Start Date | End Date | Calendar Days | Evaluated Observations | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation Window** | `2015-06-01` | `2015-06-30` | 30 days | 33,450 | Validation tuning, early stopping & comparative benchmarking |
| **Final Holdout Window** | `2015-07-01` | `2015-07-31` | 31 days | 34,565 | Final uncompromised generalization test |

> [!IMPORTANT]
> The final holdout period (`2015-07-01` to `2015-07-31`) was strictly isolated throughout the project lifecycle. It was never accessed for feature engineering, model training, early stopping, hyperparameter search, or preliminary model comparison.

---

## 3. Models Evaluated

1. **Seasonal Naive Baseline**:
   - Definition: $\text{prediction}(t) = \text{Sales}(t - 7\text{ days})$
   - Strictly store-isolated, 7-day cyclical weekly baseline.
   - Implementation: `src/technova_ai_service/sales_forecasting/baselines.py`.

2. **LightGBM Regressor**:
   - CPU-friendly histogram regression model with native categorical feature handling.
   - Trained on `2013-01-01` to `2015-05-31` with early stopping on June 2015 validation MAE.
   - Serialized Model Artifact: `data/interim/lightgbm_sales_forecasting.txt`.
   - Results Artifact: `data/interim/lightgbm_validation_results.json`.

3. **XGBoost Regressor**:
   - CPU-friendly histogram tree method (`tree_method="hist"`) with native categorical handling (`enable_categorical=True`).
   - Trained on `2013-01-01` to `2015-05-31` with early stopping on June 2015 validation MAE.
   - Serialized Model Artifact: `data/interim/xgboost_sales_forecasting.json`.
   - Results Artifact: `data/interim/xgboost_validation_results.json`.

---

## 4. Validation Results (June 2015)

Evaluation results on the 30-day June 2015 validation set (`2015-06-01` through `2015-06-30`, 33,450 observations) extracted from `data/interim/ml_model_comparison.json`:

| Model | WAPE | MASE | MAE (€) | RMSE (€) |
| :--- | :---: | :---: | :---: | :---: |
| **Seasonal Naive** | 0.397725 | 1.192748 | 2465.5789 | 3573.4240 |
| **LightGBM** | 0.074540 | 0.223541 | 462.0902 | 720.9805 |
| **XGBoost** | 0.075835 | 0.227425 | 470.1187 | 734.1000 |

*Both gradient boosting architectures substantially outperformed the Seasonal Naive baseline, reducing WAPE by ~81% and MAE by over €1,990 per store-day.*

---

## 5. Final Holdout Results (July 2015)

Evaluation results on the untouched 31-day July 2015 final holdout (`2015-07-01` through `2015-07-31`, 34,565 observations) extracted from `data/interim/final_holdout_results.json`:

| Model | WAPE | MASE | MAE (€) | RMSE (€) |
| :--- | :---: | :---: | :---: | :---: |
| **Seasonal Naive** | 0.291761 | 0.858800 | 1792.2009 | 2432.5229 |
| **LightGBM** | 0.082383 | 0.242495 | 506.0547 | 761.3373 |
| **XGBoost** | **0.079967** | **0.235383** | **491.2118** | **743.5149** |

---

## 6. Model Selection and Deployment Decision

### Selected Deployment Candidate: **XGBoost**

**Decision Rationale**:
- On the untouched July 2015 final holdout, **XGBoost achieved the lowest error across all four primary and secondary evaluation metrics**:
  - **WAPE**: `0.079967` (vs. LightGBM `0.082383`, Seasonal Naive `0.291761`)
  - **MASE**: `0.235383` (vs. LightGBM `0.242495`, Seasonal Naive `0.858800`)
  - **MAE**: `€491.2118` (vs. LightGBM `€506.0547`, Seasonal Naive `€1792.2009`)
  - **RMSE**: `€743.5149` (vs. LightGBM `€761.3373`, Seasonal Naive `€2432.5229`)
- Although LightGBM performed marginally better on the June validation set on which early stopping was monitored, XGBoost exhibited superior out-of-distribution generalization on the genuinely unseen July holdout, maintaining lower degradation across all metrics.
- XGBoost demonstrated greater resilience to store-level promotional and calendar transitions present in July.

---

## 7. Artifact Registry

| Component | Role | File Path | Format |
| :--- | :--- | :--- | :--- |
| **Selected Model** | Deployment Candidate | `data/interim/xgboost_sales_forecasting.json` | XGBoost JSON |
| **Alternative Model** | Evaluated Secondary | `data/interim/lightgbm_sales_forecasting.txt` | LightGBM Text |
| **Baseline Model** | Benchmark Reference | `src/technova_ai_service/sales_forecasting/baselines.py` | Python implementation |
| **Validation Comparison** | Pre-Holdout Report | `data/interim/ml_model_comparison.json` | JSON Report |
| **Final Holdout Results** | Decision Ground Truth | `data/interim/final_holdout_results.json` | JSON Report |

---

## 8. Holdout Isolation Confirmation

- **Training Isolation**: The training split strictly ended on `2015-05-31`. No July 2015 data was present in training matrices.
- **Validation Isolation**: Validation tuning and early stopping used exclusively June 2015 (`2015-06-01` to `2015-06-30`).
- **Zero Retraining**: Evaluation on July 2015 was performed strictly using pre-saved model weights without refitting or fine-tuning.
- **No Contemporaneous Features**: The `Customers` column was strictly excluded from all models.

---

## 9. Limitations & Production Considerations

1. **Single Holdout Period**:
   - The final evaluation represents one calendar month (July 2015). Seasonal dynamics in other quarters (e.g., Q4 holiday peaks) may exhibit different error characteristics.
2. **Domain Adaptation**:
   - Historical Rossmann pharmacy retail data serves as an interim proxy. When deploying to live TechNova POS enterprise clients, store assortment, product categories, and transaction velocities may diverge.
3. **Continuous Performance Monitoring**:
   - Following deployment, actual POS sales telemetry should be continuously tracked against forecasts using live WAPE and MASE tracking with alerting on distribution shifts.

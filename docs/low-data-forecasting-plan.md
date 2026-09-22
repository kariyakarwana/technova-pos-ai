# Low-data forecasting plan

TechNova currently has too little local history to approve fully automated purchasing or pricing.
The service therefore uses staged maturity instead of presenting every result as equally reliable.

## Maturity stages

| Available daily history | Service behaviour | Decision policy |
| --- | --- | --- |
| 7-27 days | Cold-start direct forecast with wide intervals | Manual review required |
| 28-89 days | Latest 28-day direct forecast and ML-shaped daily curve | Manual review required |
| 90-364 days | Developing local pattern, refreshed every day | Advisory purchase suggestions |
| 365+ days | Eligible for TechNova retraining and rolling backtests | Automation only after approval |

`production_ready` remains false until a model trained on TechNova data passes agreed accuracy,
bias, stockout and business-cost thresholds.

## Data to collect from the POS

Store one daily record for every product and branch, including zero-sale days:

- date, organization, branch, product, category and brand
- quantity sold and returned
- opening, closing, reserved and available stock
- whether the product was out of stock
- selling price, unit cost, promotion and discount
- invoice and customer counts
- supplier lead time and received purchase quantity
- local holiday or retail-event identifier

An out-of-stock day must not be treated as genuine zero demand. Keep its stockout indicator so the
training pipeline can censor or adjust the observation.

## Retraining schedule

1. Export and validate aggregated POS data every night.
2. Refresh operational forecasts daily using the latest actual demand.
3. Run accuracy evaluation weekly without replacing the approved model.
4. Retrain a candidate model monthly after sufficient data exists.
5. Promote a candidate only when it improves accuracy and business-cost measures.

## Seasonal forecasting

Calendar quarters are only summaries. Named Sinhala and Tamil New Year, Vesak, Christmas and school
seasons require explicit event fields and at least two occurrences in historical data. Until that
history exists, event uplifts should be administrator-entered rules, clearly separated from learned
predictions.

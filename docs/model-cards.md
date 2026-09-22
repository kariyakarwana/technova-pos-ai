# Model Cards

## Development dataset

The development models use the UCI Online Retail dataset, containing transactions from a
UK-based non-store retailer between December 2010 and December 2011. It is useful for validating
the pipeline but does not represent TechNova products, Sri Lankan seasonality, LKR prices, branch
behaviour or current customers. Production decisions require retraining and evaluation on approved
TechNova transaction exports.

Source: <https://archive.ics.uci.edu/dataset/352/online%2Bretail>

## Inventory intelligence

The model predicts next-day sold quantity from lagged demand, shifted rolling statistics, price and
calendar features. Training and evaluation use chronological splits. Reorder recommendations combine
predicted demand with lead time, review period and a safety-stock calculation.

Do not interpret the stockout-risk score as a calibrated probability. Treat it as a prioritization
signal and keep purchasing approval with an authorized employee.

## Dynamic pricing

The model predicts demand for candidate prices and selects the candidate with the highest expected
margin. Recommendations are bounded by minimum margin and maximum price-change rules.

The development dataset is observational, so the model does not prove that price changes caused a
demand change. Promotions, competitor prices, inventory age and legal or policy constraints must be
added before production use. The service must remain advisory until controlled experiments validate
its recommendations.

## Loyalty recommendations

The recommender combines matrix-factorization scores with product popularity and excludes previously
purchased products when possible. The profile contains recency, order count, spend and purchased-unit
features. It does not use names, email addresses, phone numbers or message content.

The model loyalty score is an analytical signal and is separate from the customer's account loyalty
points. Business rules must determine how points are earned, expired and redeemed.

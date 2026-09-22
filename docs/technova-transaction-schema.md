# TechNova Transaction Training Schema

Training accepts CSV or Parquet data using these normalized columns:

| Column | Type | Description |
| --- | --- | --- |
| `invoice_no` | string | Unique sale or invoice identifier |
| `product_id` | string | Stable TechNova product or SKU identifier |
| `description` | string | Product name used in recommendation responses |
| `quantity` | number | Positive sold quantity; returns must be separate or negative and filtered |
| `invoice_date` | timestamp | Sale timestamp in a consistent timezone |
| `unit_price` | number | Selling price before multiplying by quantity |
| `customer_id` | string/null | Stable customer identifier; null for anonymous sales |
| `country` | string | Country or replace with the TechNova branch identifier |

For production, include at least six months of history and preferably twelve months. Export only
fields approved for model training. Do not include names, email addresses, phone numbers, passwords,
or message content.

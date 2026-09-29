from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd


APPROVED_CATEGORIES: list[str] = [
    "Beverages",
    "Packaged Foods",
    "Personal Care",
    "Household Cleaning",
    "Health",
]


@dataclass(frozen=True)
class ProductCatalogItem:
    product_id: str
    product_name: str
    category: str
    base_unit_price: float
    popularity_weight: float
    basket_depth: float
    promo_elasticity: float
    weekend_lift: float
    school_holiday_lift: float
    dispersion_param: float
    rank: int


def build_synthetic_product_catalog(
    n_products: int = 50,
    *,
    zipf_alpha: float = 1.25,
) -> list[ProductCatalogItem]:
    """Build a deterministic, realistic retail product catalog with 50 SKUs.

    Properties:
    - 5 approved categories (10 SKUs each).
    - Pareto / Zipf popularity weights where top 20% accounts for ~70-80% volume.
    - Category-specific and product-specific prices, promo elasticities, and weekend lifts.
    - Negative Binomial dispersion parameters: higher for staples, lower for long-tail.
    """
    if n_products != 50:
        raise ValueError(f"Catalog specification requires exactly 50 SKUs, got {n_products}")

    # Raw item templates across the 5 categories
    # Each entry: (category, product_name, base_price, promo_elasticity, weekend_lift, holiday_lift, basket_depth, dispersion)
    sku_definitions = [
        # Top 10: Fast-Moving Staples (Ranks 1 to 10)
        ("Beverages", "Mineral Water 1.5L", 1.20, 0.25, 0.15, 0.10, 1.35, 15.0),
        ("Packaged Foods", "Whole Wheat Bread 500g", 2.10, 0.20, 0.10, 0.05, 1.15, 15.0),
        ("Beverages", "Cola Classic 2L", 2.40, 0.95, 0.45, 0.25, 1.25, 12.0),
        ("Packaged Foods", "Fresh Milk Whole 1L", 1.50, 0.18, 0.10, 0.05, 1.30, 15.0),
        ("Household Cleaning", "Toilet Paper 12 Rolls", 5.50, 0.40, 0.12, 0.05, 1.10, 12.0),
        ("Packaged Foods", "Spaghetti No. 5 500g", 1.80, 0.35, 0.15, 0.10, 1.25, 12.0),
        ("Personal Care", "Liquid Hand Soap 250ml", 1.95, 0.30, 0.10, 0.05, 1.05, 10.0),
        ("Beverages", "Orange Juice 100% 1L", 2.80, 0.70, 0.35, 0.20, 1.15, 10.0),
        ("Household Cleaning", "Dishwashing Liquid 500ml", 2.30, 0.45, 0.10, 0.05, 1.05, 10.0),
        ("Health", "Paracetamol 500mg 20s", 3.20, 0.20, 0.05, 0.02, 1.05, 10.0),

        # Ranks 11 to 35: Medium-Velocity FMCG
        ("Packaged Foods", "Basmati Rice 1kg", 3.50, 0.40, 0.12, 0.10, 1.15, 8.0),
        ("Personal Care", "Moisturizing Shampoo 400ml", 4.20, 0.55, 0.20, 0.08, 1.05, 8.0),
        ("Household Cleaning", "Laundry Detergent Liquid 1.5L", 8.90, 0.85, 0.15, 0.05, 1.05, 8.0),
        ("Beverages", "Green Tea Bags 50s", 3.10, 0.40, 0.15, 0.05, 1.05, 8.0),
        ("Health", "Vitamin C 1000mg Effervescent", 5.50, 0.75, 0.15, 0.10, 1.05, 7.0),
        ("Packaged Foods", "Crushed Tomatoes 400g", 1.40, 0.50, 0.15, 0.05, 1.30, 8.0),
        ("Personal Care", "Fluoride Toothpaste 100ml", 2.60, 0.35, 0.10, 0.05, 1.10, 8.0),
        ("Beverages", "Energy Drink 250ml", 2.20, 0.80, 0.45, 0.20, 1.20, 7.0),
        ("Health", "Multivitamin Daily 60 Tablets", 9.90, 0.65, 0.10, 0.05, 1.02, 6.0),
        ("Household Cleaning", "Surface Disinfectant Spray 750ml", 3.40, 0.50, 0.12, 0.05, 1.05, 7.0),
        ("Packaged Foods", "Rolled Oats 1kg", 2.80, 0.30, 0.10, 0.10, 1.10, 7.0),
        ("Personal Care", "Body Wash Hydrating 500ml", 3.80, 0.60, 0.25, 0.08, 1.05, 7.0),
        ("Beverages", "Instant Coffee Classic 200g", 6.50, 0.55, 0.15, 0.05, 1.02, 6.0),
        ("Health", "Ibuprofen 400mg 10s", 3.80, 0.25, 0.05, 0.02, 1.02, 6.0),
        ("Household Cleaning", "Trash Bags 50L 20s", 2.90, 0.35, 0.10, 0.05, 1.08, 7.0),
        ("Packaged Foods", "Olive Oil Extra Virgin 500ml", 7.80, 0.60, 0.15, 0.05, 1.05, 6.0),
        ("Personal Care", "Deodorant Spray 150ml", 3.50, 0.50, 0.20, 0.05, 1.05, 6.0),
        ("Beverages", "Sparkling Flavored Water 1L", 1.80, 0.65, 0.35, 0.15, 1.15, 6.0),
        ("Health", "First Aid Plaster Pack 40s", 3.10, 0.20, 0.08, 0.05, 1.02, 6.0),
        ("Household Cleaning", "Kitchen Paper Towels 2 Rolls", 2.50, 0.40, 0.12, 0.05, 1.10, 6.0),
        ("Packaged Foods", "Dark Chocolate 70% 100g", 2.40, 0.85, 0.40, 0.20, 1.15, 6.0),
        ("Personal Care", "Hair Conditioner 400ml", 4.30, 0.55, 0.20, 0.08, 1.02, 5.0),
        ("Beverages", "Oat Milk Unsweetened 1L", 2.70, 0.45, 0.20, 0.10, 1.10, 6.0),
        ("Health", "Cough Syrup Herbal 150ml", 6.80, 0.30, 0.05, 0.02, 1.02, 5.0),
        ("Household Cleaning", "Sponges Heavy Duty 3pk", 1.90, 0.35, 0.10, 0.05, 1.05, 6.0),

        # Ranks 36 to 50: Slow-Moving Long-Tail
        ("Packaged Foods", "Organic Quinoa 500g", 4.90, 0.40, 0.15, 0.05, 1.02, 4.0),
        ("Personal Care", "Facial Moisturizer SPF30 50ml", 12.50, 0.45, 0.20, 0.05, 1.01, 3.5),
        ("Beverages", "Craft Kombucha Ginger 330ml", 3.20, 0.60, 0.35, 0.15, 1.05, 4.0),
        ("Health", "Omega-3 Fish Oil 90 Softgels", 14.50, 0.50, 0.10, 0.05, 1.01, 3.5),
        ("Household Cleaning", "Fabric Softener Lavender 1L", 4.20, 0.65, 0.15, 0.05, 1.05, 4.5),
        ("Packaged Foods", "Almond Butter Crunchy 250g", 5.80, 0.45, 0.18, 0.08, 1.02, 3.5),
        ("Personal Care", "Shaving Razor 3-Blade 4pk", 6.20, 0.40, 0.15, 0.05, 1.01, 3.5),
        ("Health", "Electrolyte Hydration Sachets 10pk", 7.20, 0.55, 0.25, 0.15, 1.02, 3.5),
        ("Beverages", "Herbal Chamomile Tea 20s", 2.90, 0.35, 0.15, 0.05, 1.02, 4.0),
        ("Household Cleaning", "Glass Window Cleaner 500ml", 2.80, 0.40, 0.12, 0.05, 1.02, 4.0),
        ("Personal Care", "Cotton Wool Rounds 100pk", 2.20, 0.35, 0.12, 0.05, 1.05, 3.5),
        ("Personal Care", "Sunscreen Lotion SPF50 200ml", 11.00, 0.65, 0.35, 0.20, 1.02, 3.0),
        ("Health", "Muscle Relief Balm 50g", 5.90, 0.25, 0.08, 0.02, 1.01, 3.0),
        ("Health", "Throat Soothing Lozenges 24s", 3.50, 0.50, 0.15, 0.05, 1.05, 3.5),
        ("Household Cleaning", "Floor Cleaner Floral 1L", 3.60, 0.45, 0.15, 0.05, 1.02, 3.5),
    ]

    ranks = np.arange(1, n_products + 1)
    raw_weights = 1.0 / (ranks ** zipf_alpha)
    normalized_weights = raw_weights / raw_weights.sum()

    catalog: list[ProductCatalogItem] = []
    for rank, (cat, name, price, promo_e, wknd_l, hol_l, bdepth, disp), weight in zip(
        ranks, sku_definitions, normalized_weights, strict=True
    ):
        sku_id = f"SKU-{rank:03d}"
        catalog.append(
            ProductCatalogItem(
                product_id=sku_id,
                product_name=name,
                category=cat,
                base_unit_price=float(price),
                popularity_weight=float(weight),
                basket_depth=float(bdepth),
                promo_elasticity=float(promo_e),
                weekend_lift=float(wknd_l),
                school_holiday_lift=float(hol_l),
                dispersion_param=float(disp),
                rank=int(rank),
            )
        )

    return catalog


def catalog_to_dataframe(catalog: list[ProductCatalogItem]) -> pd.DataFrame:
    """Convert catalog list to a pandas DataFrame."""
    return pd.DataFrame([asdict(item) for item in catalog])

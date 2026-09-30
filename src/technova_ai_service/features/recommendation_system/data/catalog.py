"""Generic retail product catalog definition for TechNova AI Recommendation Engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..domain.models import CatalogProduct


def load_catalog_dataframe(catalog_path: str | Path) -> pd.DataFrame:
    """Loads catalog parquet table into a pandas DataFrame."""
    return pd.read_parquet(catalog_path)


# Domain-specific co-purchase / complementary rule definitions:
# Mapping of seed product_id -> list of (complementary_product_id, conditional_affinity_weight)
COMPLEMENTARY_AFFINITY_RULES: dict[str, list[tuple[str, float]]] = {
    # Laptops & Computers
    "PROD-COMP-001": [  # Pro Laptop 15"
        ("PROD-PERI-001", 0.70),  # Wireless Mouse
        ("PROD-CASE-001", 0.55),  # Laptop Sleeve 15.6"
        ("PROD-PERI-003", 0.45),  # USB-C Multiport Dock
        ("PROD-CABL-001", 0.35),  # HDMI Cable
    ],
    "PROD-COMP-002": [  # Gaming Laptop 16" RTX
        ("PROD-PERI-002", 0.65),  # Mechanical Gaming Keyboard
        ("PROD-PERI-001", 0.60),  # Wireless Mouse
        ("PROD-PERI-004", 0.55),  # Gaming Headset
        ("PROD-CASE-002", 0.45),  # Desk Pad Mouse Mat XL
    ],
    "PROD-COMP-003": [  # Ultrabook 13"
        ("PROD-CASE-001", 0.60),  # Laptop Sleeve
        ("PROD-PERI-003", 0.50),  # USB-C Dock
        ("PROD-PERI-001", 0.45),  # Wireless Mouse
    ],
    "PROD-DISP-001": [  # 27" 4K Monitor
        ("PROD-CABL-001", 0.70),  # HDMI Cable
        ("PROD-CABL-002", 0.55),  # USB-C to DisplayPort
        ("PROD-PERI-003", 0.40),  # USB-C Dock
    ],
    "PROD-DISP-002": [  # 24" 144Hz Monitor
        ("PROD-CABL-001", 0.65),  # HDMI Cable
        ("PROD-PERI-002", 0.40),  # Mechanical Keyboard
    ],
    # Hardware Upgrades & PC Building
    "PROD-COMP-007": [  # Intel Core i7 LGA1700 CPU
        ("PROD-COMP-008", 0.75),  # Z790 Motherboard
        ("PROD-COMP-006", 0.70),  # Thermal Paste
        ("PROD-COMP-005", 0.60),  # 16GB DDR5 RAM
        ("PROD-COMP-004", 0.50),  # 1TB NVMe SSD
    ],
    "PROD-COMP-008": [  # Z790 Motherboard
        ("PROD-COMP-007", 0.70),  # CPU
        ("PROD-COMP-005", 0.65),  # DDR5 RAM
        ("PROD-COMP-004", 0.55),  # NVMe SSD
        ("PROD-COMP-006", 0.45),  # Thermal Paste
    ],
    "PROD-COMP-004": [  # 1TB NVMe SSD
        ("PROD-COMP-005", 0.45),  # DDR5 RAM
        ("PROD-COMP-006", 0.35),  # Thermal Paste
    ],
    # Smartphones & Mobile Accessories
    "PROD-PHON-001": [  # NovaPhone 15 Pro
        ("PROD-MACC-001", 0.75),  # Clear Magnetic Case
        ("PROD-MACC-002", 0.70),  # Tempered Glass Protector
        ("PROD-MACC-004", 0.55),  # 30W GaN Fast Charger
        ("PROD-MACC-005", 0.45),  # Braided USB-C Cable
        ("PROD-MACC-007", 0.40),  # Wireless ANC Earbuds
    ],
    "PROD-PHON-002": [  # NovaPhone Lite
        ("PROD-MACC-003", 0.65),  # Armor Case
        ("PROD-MACC-002", 0.60),  # Tempered Glass Protector
        ("PROD-MACC-004", 0.45),  # 30W Fast Charger
        ("PROD-MACC-005", 0.40),  # USB-C Cable
    ],
    "PROD-MACC-006": [  # Magnetic Wireless Power Bank
        ("PROD-MACC-005", 0.55),  # USB-C Cable
        ("PROD-MACC-008", 0.40),  # Car Vent Mount
    ],
    # Supermarket & Convenience
    "PROD-GROC-001": [  # Roasted Coffee Beans 500g
        ("PROD-GROC-009", 0.65),  # Fresh Whole Milk 1L
        ("PROD-GROC-006", 0.45),  # Dark Chocolate 72%
        ("PROD-GROC-008", 0.40),  # Whole Grain Toast Bread
    ],
    "PROD-GROC-005": [  # Crunchy Oat Granola 400g
        ("PROD-GROC-009", 0.70),  # Fresh Whole Milk 1L
        ("PROD-GROC-010", 0.50),  # Organic Honey 350g
    ],
    "PROD-GROC-008": [  # Whole Grain Toast Bread
        ("PROD-GROC-010", 0.45),  # Honey
        ("PROD-GROC-009", 0.40),  # Milk
    ],
    "PROD-GROC-004": [  # Zero Sugar Cola 1.5L
        ("PROD-GROC-007", 0.60),  # Sea Salt Crisps 150g
        ("PROD-GROC-006", 0.35),  # Dark Chocolate
    ],
    "PROD-GROC-011": [  # Eco Dishwashing Liquid
        ("PROD-GROC-012", 0.65),  # Recycled Paper Towels
    ],
}


def build_generic_retail_catalog(
    organization_id: str = "org_technova_default",
) -> list[CatalogProduct]:
    """Build a multi-domain generic retail catalog with 48 representative SKUs.

    Includes:
    - Electronics & Computers (17 SKUs)
    - Phones & Accessories (13 SKUs)
    - Supermarket / Convenience (16 SKUs)
    - Inactive / Discontinued SKUs (2 SKUs, for stock & status filtering validation)
    """
    raw_catalog = [
        # =========================================================================
        # 1. COMPUTERS & HARDWARE
        # =========================================================================
        (
            "PROD-COMP-001", "SKU-COMP-001", "TechNova ProBook 15 Core i7 16GB 512GB",
            "Computers & Electronics", "Laptops", "Business Laptops", "TechNova",
            850.00, 1199.00, 1149.00,
            {"processor": "Intel Core i7-13700H", "ram_gb": 16, "storage_gb": 512, "screen_in": 15.6, "ports": ["USB-C", "HDMI", "USB-A"]},
            ["laptop", "intel", "usb_c", "hdmi", "wifi6"],
            True, 1.0, "computers",
        ),
        (
            "PROD-COMP-002", "SKU-COMP-002", "Apex Gaming Laptop 16 RTX 4070 32GB 1TB",
            "Computers & Electronics", "Laptops", "Gaming Laptops", "ApexGear",
            1350.00, 1899.00, 1849.00,
            {"processor": "AMD Ryzen 9 7945HX", "gpu": "RTX 4070 8GB", "ram_gb": 32, "storage_gb": 1024, "screen_in": 16.0, "refresh_hz": 165},
            ["laptop", "gaming", "nvidia_rtx", "high_performance"],
            True, 0.6, "computers",
        ),
        (
            "PROD-COMP-003", "SKU-COMP-003", "NovaAir Slim Ultrabook 13.3 M2-Class 8GB 256GB",
            "Computers & Electronics", "Laptops", "Ultrabooks", "NovaAir",
            680.00, 949.00, 929.00,
            {"processor": "NovaARM Octa-Core", "ram_gb": 8, "storage_gb": 256, "screen_in": 13.3, "weight_kg": 1.15},
            ["laptop", "ultrabook", "portable", "usb_c"],
            True, 0.8, "computers",
        ),
        (
            "PROD-DISP-001", "SKU-DISP-001", "UltraView 27 4K UHD IPS Designer Monitor",
            "Computers & Electronics", "Displays", "4K Monitors", "UltraView",
            260.00, 399.00, 379.00,
            {"panel": "IPS", "resolution": "3840x2160", "size_in": 27, "color_gamut": "99% DCI-P3", "inputs": ["HDMI 2.1", "DisplayPort 1.4", "USB-C 65W"]},
            ["monitor", "4k", "ips", "hdmi", "displayport", "usb_c"],
            True, 0.9, "computers",
        ),
        (
            "PROD-DISP-002", "SKU-DISP-002", "SwiftPulse 24 FHD 165Hz Fast-IPS Esports Monitor",
            "Computers & Electronics", "Displays", "Gaming Monitors", "ApexGear",
            130.00, 199.00, 189.00,
            {"panel": "Fast-IPS", "resolution": "1920x1080", "size_in": 23.8, "refresh_hz": 165, "response_ms": 1},
            ["monitor", "gaming", "165hz", "hdmi", "displayport"],
            True, 1.0, "computers",
        ),
        (
            "PROD-PERI-001", "SKU-PERI-001", "ErgoMaster Wireless Dual-Mode Optical Mouse",
            "Computers & Electronics", "Peripherals", "Mice", "LogiTechNova",
            22.00, 45.00, 39.99,
            {"connectivity": "2.4GHz + Bluetooth 5.2", "sensor_dpi": 4000, "battery_life_days": 70},
            ["mouse", "wireless", "bluetooth", "usb_receiver", "ergonomic"],
            True, 12.0, "computers",
        ),
        (
            "PROD-PERI-002", "SKU-PERI-002", "MechStrike RGB Hot-Swappable Mechanical Keyboard Red",
            "Computers & Electronics", "Peripherals", "Keyboards", "ApexGear",
            48.00, 89.00, 79.99,
            {"switch_type": "Linear Red", "layout": "Tenkeyless (TKL)", "rgb": True, "connection": "Detachable USB-C"},
            ["keyboard", "mechanical", "gaming", "rgb", "usb_c"],
            True, 3.0, "computers",
        ),
        (
            "PROD-PERI-003", "SKU-PERI-003", "NovaHub 7-in-1 Aluminium USB-C Multiport Dock",
            "Computers & Electronics", "Peripherals", "Docks & Adapters", "TechNova",
            28.00, 59.00, 52.00,
            {"ports": ["4K HDMI", "100W PD USB-C", "3x USB 3.0", "SD/MicroSD"], "chassis": "Anodized Aluminium"},
            ["dock", "usb_c", "hdmi", "power_delivery", "accessories"],
            True, 4.0, "computers",
        ),
        (
            "PROD-PERI-004", "SKU-PERI-004", "AcousticPro Wireless Active Noise Cancelling Headset",
            "Computers & Electronics", "Audio", "Headsets", "AcousticPro",
            55.00, 119.00, 109.00,
            {"type": "Over-Ear", "anc": True, "battery_hours": 35, "mic": "Detachable Boom Mic"},
            ["headset", "audio", "wireless", "anc", "bluetooth"],
            True, 1.2, "computers",
        ),
        (
            "PROD-COMP-004", "SKU-COMP-004", "HyperSpeed 1TB NVMe M.2 PCIe 4.0 Internal SSD",
            "Computers & Electronics", "Components", "Storage", "SpeedCore",
            42.00, 79.99, 74.99,
            {"interface": "PCIe 4.0 x4 M.2 2280", "read_speed_mbps": 7000, "write_speed_mbps": 6000},
            ["ssd", "nvme", "m2", "pcie4", "storage", "internal_component"],
            True, 2.5, "computers",
        ),
        (
            "PROD-COMP-005", "SKU-COMP-005", "ViperElite 16GB (2x8GB) DDR5 5600MHz CL36 RAM Kit",
            "Computers & Electronics", "Components", "Memory", "SpeedCore",
            38.00, 69.99, 64.99,
            {"type": "DDR5", "speed_mhz": 5600, "capacity_gb": 16, "form_factor": "UDIMM 288-pin"},
            ["ram", "ddr5", "desktop_memory", "internal_component"],
            True, 2.5, "computers",
        ),
        (
            "PROD-COMP-006", "SKU-COMP-006", "ArcticFreeze High-Performance Thermal Paste 4g",
            "Computers & Electronics", "Components", "Cooling & Maintenance", "ArcticFreeze",
            3.20, 11.99, 9.99,
            {"weight_g": 4, "thermal_conductivity_w_mk": 8.5, "viscosity_pa_s": 870},
            ["thermal_paste", "cooling", "maintenance", "cpu", "internal_component"],
            True, 3.0, "computers",
        ),
        (
            "PROD-COMP-007", "SKU-COMP-007", "Intel Core i7-13700K 16-Core 5.4GHz Desktop CPU",
            "Computers & Electronics", "Components", "Processors", "Intel",
            275.00, 389.00, 369.00,
            {"socket": "LGA1700", "cores": 16, "threads": 24, "boost_clock_ghz": 5.4, "tdp_w": 125},
            ["cpu", "intel", "socket_lga1700", "desktop_processor", "internal_component"],
            True, 0.7, "computers",
        ),
        (
            "PROD-COMP-008", "SKU-COMP-008", "Maximus Z790 DDR5 ATX Gaming Motherboard LGA1700",
            "Computers & Electronics", "Components", "Motherboards", "ApexGear",
            165.00, 249.00, 239.00,
            {"socket": "LGA1700", "chipset": "Intel Z790", "memory_support": "DDR5", "pcie_version": "5.0", "wifi": "Wi-Fi 6E"},
            ["motherboard", "socket_lga1700", "ddr5", "atx", "internal_component"],
            True, 0.6, "computers",
        ),
        (
            "PROD-CABL-001", "SKU-CABL-001", "Braided Ultra High Speed HDMI 2.1 Cable 2m 8K60Hz",
            "Computers & Electronics", "Cables", "Video Cables", "NovaCable",
            4.50, 16.99, 14.99,
            {"connector": "HDMI to HDMI", "length_m": 2.0, "bandwidth_gbps": 48, "braided": True},
            ["cable", "hdmi", "hdmi21", "8k", "accessories"],
            True, 9.0, "computers",
        ),
        (
            "PROD-CABL-002", "SKU-CABL-002", "USB-C to DisplayPort 1.4 Adapter Cable 1.8m",
            "Computers & Electronics", "Cables", "Video Cables", "NovaCable",
            5.80, 19.99, 17.99,
            {"connector": "USB-C to DisplayPort", "length_m": 1.8, "support": "4K144Hz / 8K60Hz"},
            ["cable", "usb_c", "displayport", "accessories"],
            True, 1.2, "computers",
        ),
        (
            "PROD-CASE-001", "SKU-CASE-001", "NovaShield Water-Resistant Padded Laptop Sleeve 15.6",
            "Computers & Electronics", "Accessories", "Bags & Sleeves", "NovaShield",
            8.00, 24.99, 21.99,
            {"compatible_size_in": 15.6, "material": "Waterproof Neoprene", "pockets": 2},
            ["sleeve", "laptop_accessory", "protection"],
            True, 4.0, "computers",
        ),
        (
            "PROD-CASE-002", "SKU-CASE-002", "PrecisionDesk Extended Anti-Fray Gaming Desk Pad XL",
            "Computers & Electronics", "Accessories", "Mouse Pads", "ApexGear",
            7.50, 22.99, 19.99,
            {"dimensions_mm": "900x400x4", "surface": "Micro-Woven Cloth", "base": "Non-Slip Rubber"},
            ["mousepad", "deskmat", "gaming", "accessories"],
            True, 1.5, "computers",
        ),

        # =========================================================================
        # 2. PHONES & MOBILE ACCESSORIES
        # =========================================================================
        (
            "PROD-PHON-001", "SKU-PHON-001", "NovaPhone 15 Pro 256GB Titanium Blue 5G",
            "Phones & Accessories", "Smartphones", "Flagship Phones", "NovaPhone",
            750.00, 1099.00, 1049.00,
            {"screen_in": 6.7, "storage_gb": 256, "connector": "USB-C", "charging_w": 27, "magsafe": True},
            ["smartphone", "novaphone", "usb_c", "5g", "magsafe", "flagship"],
            True, 1.0, "phones",
        ),
        (
            "PROD-PHON-002", "SKU-PHON-002", "NovaPhone 15 Lite 128GB Midnight Black 5G",
            "Phones & Accessories", "Smartphones", "Mid-Range Phones", "NovaPhone",
            380.00, 599.00, 569.00,
            {"screen_in": 6.1, "storage_gb": 128, "connector": "USB-C", "charging_w": 20, "magsafe": False},
            ["smartphone", "novaphone", "usb_c", "5g", "midrange"],
            True, 1.2, "phones",
        ),
        (
            "PROD-MACC-001", "SKU-MACC-001", "NovaPhone 15 Pro Crystal Clear MagSafe Hybrid Case",
            "Phones & Accessories", "Cases & Protection", "Phone Cases", "NovaShield",
            5.20, 24.99, 21.99,
            {"material": "TPU + Polycarbonate", "magsafe_compatible": True, "drop_test_m": 2.5},
            ["case", "novaphone_15_pro", "magsafe", "protection", "accessories"],
            True, 14.0, "phones",
        ),
        (
            "PROD-MACC-002", "SKU-MACC-002", "TemperedShield 9H Anti-Scratch Glass Screen Protector 2-Pack",
            "Phones & Accessories", "Cases & Protection", "Screen Protectors", "NovaShield",
            2.80, 14.99, 12.99,
            {"hardness": "9H Tempered Glass", "thickness_mm": 0.33, "pack_count": 2, "oleophobic": True},
            ["screen_protector", "protection", "accessories", "novaphone"],
            True, 15.0, "phones",
        ),
        (
            "PROD-MACC-003", "SKU-MACC-003", "ArmorGuard Rugged Dual-Layer Shockproof Case",
            "Phones & Accessories", "Cases & Protection", "Phone Cases", "ArmorGuard",
            6.50, 29.99, 26.99,
            {"military_grade_drop": "MIL-STD-810G", "kickstand": True, "raised_bezels": True},
            ["case", "rugged", "shockproof", "protection", "accessories"],
            True, 1.5, "phones",
        ),
        (
            "PROD-MACC-004", "SKU-MACC-004", "PowerVolt 30W GaN Super Fast Wall Charger USB-C",
            "Phones & Accessories", "Power & Charging", "Wall Chargers", "PowerVolt",
            7.00, 24.99, 21.99,
            {"tech": "GaN III", "output_w": 30, "port": "USB-C PD 3.0", "foldable_prongs": True},
            ["charger", "fast_charge", "usb_c", "power_delivery", "gan"],
            True, 8.0, "phones",
        ),
        (
            "PROD-MACC-005", "SKU-MACC-005", "DuraWeave Braided USB-C to USB-C 100W Cable 1.5m",
            "Phones & Accessories", "Power & Charging", "Cables", "NovaCable",
            3.00, 14.99, 12.99,
            {"wattage_max": 100, "data_rate": "USB 2.0 480Mbps", "length_m": 1.5, "durability_bends": 25000},
            ["cable", "usb_c", "100w", "charging", "accessories"],
            True, 12.0, "phones",
        ),
        (
            "PROD-MACC-006", "SKU-MACC-006", "VoltBank Mag 10000mAh Magnetic Wireless Power Bank",
            "Phones & Accessories", "Power & Charging", "Power Banks", "PowerVolt",
            16.00, 44.99, 39.99,
            {"capacity_mah": 10000, "wireless_w": 15, "wired_pd_w": 20, "magsafe": True},
            ["power_bank", "wireless_charging", "magsafe", "battery", "usb_c"],
            True, 3.5, "phones",
        ),
        (
            "PROD-MACC-007", "SKU-MACC-007", "SoundBuds Pro True Wireless Earbuds Active Noise Cancelling",
            "Phones & Accessories", "Audio", "Wireless Earbuds", "SoundPro",
            24.00, 69.99, 59.99,
            {"driver_mm": 11, "anc_db": 38, "battery_total_hours": 30, "bluetooth": "5.3", "water_resistance": "IPX5"},
            ["earbuds", "audio", "anc", "bluetooth", "wireless"],
            True, 2.0, "phones",
        ),
        (
            "PROD-MACC-008", "SKU-MACC-008", "MagMount Air Vent Magnetic In-Car Phone Mount",
            "Phones & Accessories", "Accessories", "Car Mounts", "NovaShield",
            5.00, 19.99, 16.99,
            {"mount_type": "Air Vent Twist-Lock", "magnets": "N52 Neodymium", "rotation": "360-Degree"},
            ["car_mount", "magsafe", "car_accessory", "accessories"],
            True, 2.0, "phones",
        ),

        # =========================================================================
        # 3. SUPERMARKET & FMCG
        # =========================================================================
        (
            "PROD-GROC-001", "SKU-GROC-001", "Heritage Roast Arabica Coffee Beans Medium Roast 500g",
            "Supermarket & Retail", "Beverages", "Coffee", "HeritageRoast",
            5.20, 11.99, 10.99,
            {"roast_level": "Medium", "origin": "Single Origin Colombia", "weight_g": 500, "bean_type": "100% Arabica"},
            ["coffee", "arabica", "whole_bean", "beverage", "supermarket"],
            True, 8.0, "supermarket",
        ),
        (
            "PROD-GROC-002", "SKU-GROC-002", "GreenLeaf Organic Japanese Matcha Green Tea 50 Bags",
            "Supermarket & Retail", "Beverages", "Tea", "GreenLeaf",
            3.00, 7.49, 6.99,
            {"organic_certified": True, "caffeine": "Low", "count": 50, "packaging": "Biodegradable Bags"},
            ["tea", "green_tea", "organic", "beverage", "supermarket"],
            True, 2.0, "supermarket",
        ),
        (
            "PROD-GROC-003", "SKU-GROC-003", "AlpineSpring Pure Natural Mineral Water 1.5L",
            "Supermarket & Retail", "Beverages", "Water", "AlpineSpring",
            0.40, 1.49, 1.29,
            {"source": "Alpine Artesian", "volume_l": 1.5, "ph": 7.4, "container": "100% rPET"},
            ["water", "mineral_water", "hydration", "beverage", "staple"],
            True, 25.0, "supermarket",
        ),
        (
            "PROD-GROC-004", "SKU-GROC-004", "SparkleCola Zero Sugar Refreshing Carbonated Drink 1.5L",
            "Supermarket & Retail", "Beverages", "Soft Drinks", "SparkleCola",
            0.60, 2.29, 1.99,
            {"sugar_free": True, "volume_l": 1.5, "flavor": "Classic Cola", "caffeine": "Present"},
            ["cola", "soft_drink", "zero_sugar", "soda", "beverage"],
            True, 20.0, "supermarket",
        ),
        (
            "PROD-GROC-005", "SKU-GROC-005", "NutriHarvest Crunchy Golden Honey Almond Granola 400g",
            "Supermarket & Retail", "Food & Pantry", "Breakfast Cereals", "NutriHarvest",
            2.10, 5.49, 4.99,
            {"main_ingredients": ["Rolled Oats", "Almonds", "Pure Honey"], "fiber_g_per_100": 8.5, "weight_g": 400},
            ["granola", "breakfast", "cereal", "oats", "pantry"],
            True, 2.5, "supermarket",
        ),
        (
            "PROD-GROC-006", "SKU-GROC-006", "NoirArtisan 72% Dark Chocolate Single Origin Bar 100g",
            "Supermarket & Retail", "Food & Pantry", "Confectionery", "NoirArtisan",
            1.10, 2.99, 2.79,
            {"cocoa_percentage": 72, "weight_g": 100, "vegan": True, "fair_trade": True},
            ["chocolate", "dark_chocolate", "snack", "sweet", "supermarket"],
            True, 10.0, "supermarket",
        ),
        (
            "PROD-GROC-007", "SKU-GROC-007", "CrispyFarm Sea Salt & Cracked Black Pepper Kettle Crisps 150g",
            "Supermarket & Retail", "Food & Pantry", "Snacks", "CrispyFarm",
            0.85, 2.49, 2.29,
            {"cooking_style": "Kettle Cooked", "flavor": "Sea Salt & Pepper", "gluten_free": True, "weight_g": 150},
            ["crisps", "chips", "snack", "savory", "supermarket"],
            True, 16.0, "supermarket",
        ),
        (
            "PROD-GROC-008", "SKU-GROC-008", "BakerHeritage 7-Grain Wholemeal Sliced Toast Bread 500g",
            "Supermarket & Retail", "Food & Pantry", "Bakery", "BakerHeritage",
            0.90, 2.29, 2.09,
            {"type": "Wholemeal Toast", "slices": 18, "weight_g": 500, "preservative_free": True},
            ["bread", "bakery", "wholemeal", "staple", "supermarket"],
            True, 18.0, "supermarket",
        ),
        (
            "PROD-GROC-009", "SKU-GROC-009", "MeadowFresh Grade A Pasteurised Whole Milk 1L",
            "Supermarket & Retail", "Dairy & Chilled", "Milk", "MeadowFresh",
            0.55, 1.69, 1.49,
            {"fat_content": "3.5% Whole Milk", "volume_l": 1.0, "organic": False, "pasteurised": True},
            ["milk", "dairy", "staple", "fresh", "supermarket"],
            True, 22.0, "supermarket",
        ),
        (
            "PROD-GROC-010", "SKU-GROC-010", "GoldenWild 100% Pure Raw Wildflower Honey 350g",
            "Supermarket & Retail", "Food & Pantry", "Spreads & Sweeteners", "GoldenWild",
            2.80, 6.99, 6.49,
            {"raw_unfiltered": True, "weight_g": 350, "origin": "Natural Flora", "jar": "Glass"},
            ["honey", "spread", "sweetener", "natural", "pantry"],
            True, 2.0, "supermarket",
        ),
        (
            "PROD-GROC-011", "SKU-GROC-011", "EcoCleanse Concentrated Citrus Dishwashing Liquid 500ml",
            "Supermarket & Retail", "Household & Cleaning", "Dish Care", "EcoCleanse",
            0.80, 2.49, 2.19,
            {"formula": "Plant-Based Biodegradable", "scent": "Sweet Orange", "volume_ml": 500},
            ["dishwashing", "cleaning", "household", "eco", "supermarket"],
            True, 7.0, "supermarket",
        ),
        (
            "PROD-GROC-012", "SKU-GROC-012", "SoftEarth 100% Recycled 2-Ply Kitchen Paper Towels 2-Pack",
            "Supermarket & Retail", "Household & Cleaning", "Paper Products", "SoftEarth",
            1.20, 3.29, 2.99,
            {"plies": 2, "rolls": 2, "recycled_post_consumer": "100%", "sheets_per_roll": 100},
            ["paper_towels", "household", "recycled", "cleaning", "staple"],
            True, 7.0, "supermarket",
        ),

        # =========================================================================
        # 4. INACTIVE / DISCONTINUED PRODUCTS (For Stock & Status Filtering Tests)
        # =========================================================================
        (
            "PROD-DISC-001", "SKU-DISC-001", "LegacyLink USB 2.0 4-Port External Hub (Discontinued)",
            "Computers & Electronics", "Peripherals", "Legacy Adapters", "LegacyLink",
            4.00, 9.99, 7.99,
            {"status_note": "End of Life", "connector": "USB-A 2.0"},
            ["legacy", "usb_a", "discontinued"],
            False, 0.0, "computers",  # Inactive product!
        ),
        (
            "PROD-DISC-002", "SKU-DISC-002", "RetroRam 4GB DDR3 1333MHz Desktop RAM (Discontinued)",
            "Computers & Electronics", "Components", "Memory", "RetroRam",
            6.00, 14.99, 11.99,
            {"status_note": "End of Life", "generation": "DDR3"},
            ["legacy", "ddr3", "discontinued"],
            False, 0.0, "computers",  # Inactive product!
        ),
    ]

    catalog: list[CatalogProduct] = []
    for item in raw_catalog:
        catalog.append(
            CatalogProduct(
                product_id=item[0],
                organization_id=organization_id,
                sku=item[1],
                name=item[2],
                category_level_1=item[3],
                category_level_2=item[4],
                category_level_3=item[5],
                brand=item[6],
                cost_price=item[7],
                base_unit_price=item[8],
                current_unit_price=item[9],
                specifications_json=json.dumps(item[10]),
                compatibility_tags=item[11],
                is_active=item[12],
                popularity_weight=item[13],
                domain=item[14],
            )
        )
    return catalog

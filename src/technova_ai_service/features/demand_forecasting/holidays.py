from datetime import date
from typing import TypedDict


class SriLankanHolidayFlags(TypedDict):
    is_public_holiday: int
    is_mercantile_holiday: int
    is_poya_day: int
    is_festive_peak: int


# Known Full Moon Poya Days (Gazetted Sri Lankan Poya Days 2024 - 2027)
KNOWN_POYA_DATES: set[date] = {
    # 2024
    date(2024, 1, 25),
    date(2024, 2, 23),
    date(2024, 3, 24),
    date(2024, 4, 23),
    date(2024, 5, 23),
    date(2024, 5, 24),  # Adhi Vesak
    date(2024, 6, 21),
    date(2024, 7, 20),
    date(2024, 8, 19),
    date(2024, 9, 17),
    date(2024, 10, 17),
    date(2024, 11, 15),
    date(2024, 12, 15),
    # 2025
    date(2025, 1, 13),
    date(2025, 2, 12),
    date(2025, 3, 13),
    date(2025, 4, 12),
    date(2025, 5, 12),
    date(2025, 6, 10),
    date(2025, 7, 10),
    date(2025, 8, 8),
    date(2025, 9, 7),
    date(2025, 10, 6),
    date(2025, 11, 5),
    date(2025, 12, 4),
    # 2026
    date(2026, 1, 2),
    date(2026, 2, 1),
    date(2026, 3, 3),
    date(2026, 4, 1),
    date(2026, 5, 1),
    date(2026, 5, 31),
    date(2026, 6, 29),
    date(2026, 7, 29),
    date(2026, 8, 27),
    date(2026, 9, 26),
    date(2026, 10, 25),
    date(2026, 11, 24),
    date(2026, 12, 23),
    # 2027
    date(2027, 1, 22),
    date(2027, 2, 20),
    date(2027, 3, 22),
    date(2027, 4, 20),
    date(2027, 5, 20),
    date(2027, 6, 19),
    date(2027, 7, 18),
    date(2027, 8, 17),
    date(2027, 9, 15),
    date(2027, 10, 15),
    date(2027, 11, 13),
    date(2027, 12, 13),
}

# Fixed annual statutory public & mercantile holidays in Sri Lanka
FIXED_STATUTORY_HOLIDAYS: set[tuple[int, int]] = {
    (1, 1),    # New Year's Day
    (2, 4),    # Independence Commemoration Day
    (4, 12),   # Day prior to Sinhala & Tamil New Year
    (4, 13),   # Sinhala & Tamil New Year Day
    (4, 14),   # Sinhala & Tamil New Year Day (Bank/Public)
    (5, 1),    # May Day (International Workers' Day)
    (12, 25),  # Christmas Day
}

FIXED_MERCANTILE_HOLIDAYS: set[tuple[int, int]] = {
    (2, 4),    # Independence Day
    (4, 13),   # New Year Day
    (4, 14),   # New Year Day
    (5, 1),    # May Day
    (12, 25),  # Christmas Day
}


def is_sri_lankan_festive_peak(target_date: date) -> bool:
    """Detect if the date falls in a major retail festive buying surge in Sri Lanka."""
    month = target_date.month
    day = target_date.day

    # 1. Sinhala & Tamil New Year surge (April 10 - April 18)
    if month == 4 and 10 <= day <= 18:
        return True

    # 2. Vesak Season (May month festive window)
    if month == 5 and (1 <= day <= 25 or target_date in KNOWN_POYA_DATES):
        return True

    # 3. Christmas & Year-End Shopping Peak (Dec 15 - Dec 31)
    if month == 12 and 15 <= day <= 31:
        return True

    return False


def get_sri_lankan_holiday_flags(
    target_date: date,
    *,
    custom_poya_dates: set[date] | None = None,
    custom_public_holidays: set[date] | None = None,
    custom_mercantile_holidays: set[date] | None = None,
) -> SriLankanHolidayFlags:
    """Return binary holiday and event indicator flags for any calendar date."""
    poyas = custom_poya_dates if custom_poya_dates is not None else KNOWN_POYA_DATES
    month_day = (target_date.month, target_date.day)

    is_poya = target_date in poyas

    is_public = (
        is_poya
        or (month_day in FIXED_STATUTORY_HOLIDAYS)
        or (custom_public_holidays is not None and target_date in custom_public_holidays)
    )

    is_mercantile = (
        is_poya
        or (month_day in FIXED_MERCANTILE_HOLIDAYS)
        or (custom_mercantile_holidays is not None and target_date in custom_mercantile_holidays)
    )

    is_festive = is_sri_lankan_festive_peak(target_date)

    return {
        "is_public_holiday": int(is_public),
        "is_mercantile_holiday": int(is_mercantile),
        "is_poya_day": int(is_poya),
        "is_festive_peak": int(is_festive),
    }

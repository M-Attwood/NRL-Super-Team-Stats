"""Tests for clean_data — specifically the year-aware median imputation.

The previous version computed medians across all rows including 2026
partial-season data, leaking forward-looking signal into 2024 NaN fills.
"""
import numpy as np
import pandas as pd

from model import clean_data


def test_missing_value_filled_with_year_specific_median():
    """A NaN in 2024 should be filled from 2024's medians, not from a
    blend that includes 2026 data."""
    df = pd.DataFrame([
        # 2024 group: HOK avg_minutes around 50
        {"player_name": "a", "positions": "HOK", "scrape_year": 2024, "avg_minutes": 50},
        {"player_name": "b", "positions": "HOK", "scrape_year": 2024, "avg_minutes": 50},
        {"player_name": "c", "positions": "HOK", "scrape_year": 2024, "avg_minutes": np.nan},
        # 2026 group: HOK avg_minutes around 80 (a high-minutes year)
        {"player_name": "d", "positions": "HOK", "scrape_year": 2026, "avg_minutes": 80},
        {"player_name": "e", "positions": "HOK", "scrape_year": 2026, "avg_minutes": 80},
    ])
    cleaned = clean_data(df)
    # Player c's NaN must be imputed with 2024's HOK median (50), NOT
    # the cross-year median (~62) that the buggy version would produce.
    c_minutes = cleaned.loc[cleaned["player_name"] == "c", "avg_minutes"].iloc[0]
    assert c_minutes == 50, (
        f"Year-aware imputation failed: got {c_minutes}, expected 50 "
        "(2024 HOK median). Cross-year leakage is back."
    )


def test_falls_back_to_position_median_when_no_same_year_data():
    """If a player's (year, position) combo has no other rows, fall back
    to the position median across all years rather than dropping or
    leaving NaN."""
    df = pd.DataFrame([
        # Only one HOK in 2024, with NaN — no same-year peers
        {"player_name": "lonely", "positions": "HOK", "scrape_year": 2024, "avg_minutes": np.nan},
        # Plenty of HOK data in 2025
        {"player_name": "x", "positions": "HOK", "scrape_year": 2025, "avg_minutes": 60},
        {"player_name": "y", "positions": "HOK", "scrape_year": 2025, "avg_minutes": 60},
    ])
    cleaned = clean_data(df)
    lonely = cleaned.loc[cleaned["player_name"] == "lonely", "avg_minutes"].iloc[0]
    # Same-year median is undefined (NaN within group); falls through to
    # position-only median = 60.
    assert lonely == 60

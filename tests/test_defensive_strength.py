"""Tests for compute_defensive_strength.

Why: this lookup table is the seed for NN-4 (opponent-strength feature).
Buggy aggregation here would silently corrupt the feature once fixture
scraping lands.
"""
import pandas as pd

from model import compute_defensive_strength


def test_returns_empty_when_no_vs_team():
    df = pd.DataFrame({
        "player_name": ["a", "b"],
        "positions": ["HOK", "FRF"],
        "avg_points": [50.0, 60.0],
    })
    assert compute_defensive_strength(df).empty


def test_returns_empty_when_vs_team_all_blank():
    df = pd.DataFrame({
        "player_name": ["a", "b"],
        "positions": ["HOK", "FRF"],
        "avg_points": [50.0, 60.0],
        "vs_team": ["", None],
    })
    assert compute_defensive_strength(df).empty


def test_aggregates_per_team_and_position():
    df = pd.DataFrame({
        "player_name":  ["a", "b", "c", "d"],
        "positions":    ["HOK", "HOK",   "CTW",   "CTW"],
        "avg_points":   [80.0,  60.0,    100.0,   50.0],
        "vs_team":      ["BRO", "BRO",   "BRO",   "MEL"],
    })
    out = compute_defensive_strength(df)
    # 3 unique (vs_team, primary_position) cells: (BRO,HOK), (BRO,CTW), (MEL,CTW)
    assert len(out) == 3

    # BRO conceded mean(80, 60) = 70 to HOKs across 2 samples
    bro_hok = out[(out["opponent_team"] == "BRO")
                  & (out["primary_position"] == "HOK")]
    assert len(bro_hok) == 1
    assert bro_hok.iloc[0]["expected_points_conceded"] == 70.0
    assert int(bro_hok.iloc[0]["n_samples"]) == 2


def test_uses_primary_position_only():
    """Pipe-separated positions should aggregate by the FIRST position."""
    df = pd.DataFrame({
        "player_name":  ["a", "b"],
        "positions":    ["2RF|FRF", "FRF|2RF"],
        "avg_points":   [70.0, 80.0],
        "vs_team":      ["NEW", "NEW"],
    })
    out = compute_defensive_strength(df)
    # Two distinct primary positions despite shared eligibility
    assert set(out["primary_position"]) == {"2RF", "FRF"}

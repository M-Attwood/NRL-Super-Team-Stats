"""Regression tests for the model's biggest historical bug: the target
column appearing in the feature matrix.

If `avg_points` (or any aliased form) ends up in the feature list, the
network can trivially memorise the target and report a misleadingly low
validation MAE. This test pins the constants so a future edit to
SCALE_COLS can't silently re-introduce the leak.
"""
import pandas as pd

from model import (
    SCALE_COLS, ALL_POSITIONS, ALL_TEAMS, TARGET, derive_feature_cols,
)


def test_target_column_not_in_scale_cols():
    """If `avg_points` is the target, it must NOT be a feature."""
    assert TARGET == "avg_points"
    assert TARGET not in SCALE_COLS, (
        f"TARGET column {TARGET!r} found in SCALE_COLS — this is the leak "
        "the NN review caught. The model would memorise target=feature."
    )


def test_derive_feature_cols_excludes_target():
    """derive_feature_cols must never return the target column even if
    avg_points happens to exist in the input dataframe."""
    df = pd.DataFrame({
        "avg_points": [50, 60, 70],          # the target — must be excluded
        "avg_last3":  [48, 58, 68],          # legitimate feature
        "positions": ["HOK", "FRF", "CTW"],
        "team":      ["MEL", "PTH", "BRO"],
    })
    cols = derive_feature_cols(df)
    assert TARGET not in cols
    assert "avg_last3" in cols  # sanity: legit feature still there


def test_all_teams_match_real_data_codes():
    """The hardcoded team list previously had codes ('BRI', 'CBY', ...)
    that didn't appear in the scraper's output, leaving 13/17 team
    one-hots permanently zero. Pin the canonical list here."""
    expected = {"BRO", "BUL", "CBR", "DOL", "GCT", "MEL", "MNL", "NEW",
                "NQC", "NZL", "PAR", "PTH", "SHA", "STG", "STH", "SYD", "WST"}
    assert set(ALL_TEAMS) == expected, (
        f"ALL_TEAMS drift detected. Missing: {expected - set(ALL_TEAMS)}, "
        f"extras: {set(ALL_TEAMS) - expected}"
    )
    assert len(ALL_TEAMS) == 17


def test_position_features_cover_all_positions():
    """Every position in ALL_POSITIONS must produce a stable feature
    column name (the `5/8` slash is the trap that would produce a path
    component if anyone forgets to escape it)."""
    df = pd.DataFrame({"positions": ["5/8", "HOK"]})
    cols = derive_feature_cols(df)
    for pos in ALL_POSITIONS:
        safe_col = f"pos_{pos.replace('/', '_').replace(' ', '_')}"
        assert safe_col in cols, f"missing position column for {pos}"

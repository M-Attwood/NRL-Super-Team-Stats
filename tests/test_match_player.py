"""Tests for player-name resolution.

Why this matters: the trade advisor matches user-typed names against
scraper-formatted names. A silent mismatch ("Brian To'o" not finding
"Too, Brian") means the player is excluded from the optimisation pool —
you'd ship a wrong lineup with no error.
"""
import pandas as pd

from trade_advisor import _normalise_name, match_player, resolve_names


def test_normalise_strips_straight_apostrophe():
    assert _normalise_name("Brian To'o") == _normalise_name("Brian Too")


def test_normalise_strips_curly_apostrophe():
    # The scraper drops apostrophes entirely, so the curly form must
    # collapse to the same key as the apostrophe-less form.
    assert _normalise_name("Papali’i") == _normalise_name("Papalii")


def test_normalise_strips_hawaiian_okina():
    assert _normalise_name("Olakauʻatu") == _normalise_name("Olakauatu")


def test_normalise_is_case_insensitive():
    assert _normalise_name("HARRY GRANT") == _normalise_name("harry grant")


def test_match_finds_lastname_firstname_format(tiny_pool):
    assert match_player("Harry Grant", tiny_pool) == "Grant, Harry"


def test_match_finds_apostrophe_player(tiny_pool):
    assert match_player("Brian To'o", tiny_pool) == "Too, Brian"


def test_match_handles_curly_apostrophe(tiny_pool):
    assert match_player("Isaiah Papali’i", tiny_pool) == "Papalii, Isaiah"


def test_match_returns_none_for_unknown(tiny_pool):
    assert match_player("Nobody Here", tiny_pool) is None


def test_match_does_not_collapse_distinct_surnames(tiny_pool):
    """Regression guard: fuzzy match must require exact surname.
    Without this guard, 'Keenan Going' could silently match 'Kalani Going'."""
    pool = pd.DataFrame({"player_name": ["Going, Kalani"]})
    assert match_player("Keenan Smith", pool) is None


def test_resolve_names_skips_unmatched_with_warning(caplog, tiny_pool):
    matched = resolve_names(["Harry Grant", "Definitely Not Real"], tiny_pool)
    assert matched == ["Grant, Harry"]

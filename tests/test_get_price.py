"""Tests for _get_price.

Why: silently defaulting to $0 on missing prices was a real bug — the
salary-cap math read 0 and the optimizer accepted phantom trades. The
fix logs once-per-player when the fallback fires; this test pins that
behaviour so a future change can't quietly remove the warning.
"""
import logging

import pytest

from trade_advisor import _get_price, _MISSING_PRICE_WARNED


@pytest.fixture(autouse=True)
def reset_warning_set():
    """Each test starts with a clean 'already-warned' set so we can
    observe the warning fire on the first call."""
    _MISSING_PRICE_WARNED.clear()
    yield
    _MISSING_PRICE_WARNED.clear()


def test_prefers_price_usd():
    assert _get_price({"price_usd": 500_000, "price": 999}) == 500_000


def test_falls_back_to_price():
    assert _get_price({"price": 400_000}) == 400_000


def test_returns_zero_when_both_missing(caplog):
    with caplog.at_level(logging.WARNING):
        result = _get_price({"player_name": "Ghost"})
    assert result == 0.0
    assert any("Ghost" in rec.message for rec in caplog.records)


def test_warns_only_once_per_player(caplog):
    p = {"player_name": "Repeat"}
    with caplog.at_level(logging.WARNING):
        _get_price(p)
        _get_price(p)
        _get_price(p)
    repeat_warnings = [r for r in caplog.records if "Repeat" in r.message]
    assert len(repeat_warnings) == 1


def test_handles_unparseable_price(caplog):
    with caplog.at_level(logging.WARNING):
        result = _get_price({"player_name": "Junk", "price_usd": "$not_a_number$"})
    assert result == 0.0
    assert any("Junk" in rec.message for rec in caplog.records)

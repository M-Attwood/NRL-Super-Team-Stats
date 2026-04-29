"""Tests for the trade-recommendation logic.

This is the function whose silent bug would ship a wrong lineup, so it
deserves the most scrutiny. We test the two failure modes that matter:
- the cap gate (mid-season inflated team value must not lock the loop)
- the position-quota validation (trades that produce invalid squads
  must be rejected, even if the points-gain looks great).
"""
import pandas as pd
import pytest

from trade_advisor import recommend_trades, build_squad_dicts, resolve_names


def _player(name, pos, price=300_000, pts=50.0, team="MEL"):
    return {
        "player_name": name,
        "positions": pos,
        "team": team,
        "price_usd": price,
        "price": price,
        "predicted_points": pts,
        "avg_points": pts / 26,
    }


def _full_valid_squad():
    """A valid 26-player squad (HOK=2 FRF=4 2RF=6 HFB=2 5/8=2 CTW=7 FLB=2 + 1 flex)."""
    s = []
    for pos, n in [("HOK", 2), ("FRF", 4), ("2RF", 6), ("HFB", 2),
                   ("5/8", 2), ("CTW", 7), ("FLB", 2)]:
        for i in range(n):
            s.append(_player(f"{pos}_{i}", pos, price=250_000, pts=40.0))
    s.append(_player("flex_extra", "CTW", price=250_000, pts=40.0))
    return s


def _build_pool(squad, extras):
    rows = squad + extras
    return pd.DataFrame(rows)


def test_no_trades_when_squad_already_optimal():
    squad = _full_valid_squad()
    pool = _build_pool(squad, [])
    starters = {p["player_name"] for p in squad}
    comparison = {
        "non_starters": [],
        "players_to_drop": [],
        "players_to_add": [],
    }
    trades = recommend_trades(squad, comparison, starters, pool, max_trades=2)
    assert trades == []


def test_recommends_trade_for_non_starter():
    """Non-starters score 0 — any same-position starter at the same price
    should beat them and surface as a trade."""
    squad = _full_valid_squad()
    # Replace one CTW with a "non-starter" (same position, same price).
    bench_ctw = squad[-1]  # the flex CTW
    bench_ctw["predicted_points"] = 0  # simulate "won't take field"
    upgrade = _player("upgrade_ctw", "CTW", price=250_000, pts=70.0)

    pool = _build_pool(squad, [upgrade])
    # confirmed_starters EXCLUDES bench_ctw (so it's a non-starter), INCLUDES upgrade
    starters = {p["player_name"] for p in squad if p["player_name"] != bench_ctw["player_name"]}
    starters.add(upgrade["player_name"])

    comparison = {
        "non_starters": [bench_ctw],
        "players_to_drop": [bench_ctw],
        "players_to_add": [upgrade],
    }
    trades = recommend_trades(squad, comparison, starters, pool,
                              max_trades=2, round_num=1)
    assert len(trades) == 1
    assert trades[0]["out"]["player_name"] == bench_ctw["player_name"]
    assert trades[0]["in"]["player_name"] == upgrade["player_name"]


def test_skips_trade_that_would_break_position_quota():
    """If trading OUT would leave HOK quota at 1 (need 2), the trade must
    be rejected even though it gains points."""
    squad = _full_valid_squad()
    hok = next(p for p in squad if p["positions"] == "HOK")
    # Try to swap HOK for an extra CTW — would bust HOK=2 quota
    bad_replacement = _player("extra_ctw", "CTW", price=200_000, pts=200.0)

    pool = _build_pool(squad, [bad_replacement])
    starters = {p["player_name"] for p in squad}
    starters.add(bad_replacement["player_name"])

    comparison = {
        "non_starters": [],
        "players_to_drop": [hok],
        "players_to_add": [bad_replacement],
    }
    trades = recommend_trades(squad, comparison, starters, pool,
                              max_trades=2, round_num=1)
    # No CTW->HOK position overlap, so the swap should be skipped at the
    # position-compatibility check or the quota validator.
    assert all(t["out"]["player_name"] != hok["player_name"] for t in trades)


def test_respects_explicit_salary_cap():
    """When over the default cap, only trades that stay within the
    effective cap should be returned."""
    squad = _full_valid_squad()
    # Bump squad value above default cap by inflating one player
    squad[0]["price"] = 1_500_000
    squad[0]["price_usd"] = 1_500_000

    expensive_in = _player("expensive_in", squad[0]["positions"],
                           price=2_000_000, pts=300.0)
    pool = _build_pool(squad, [expensive_in])
    starters = {p["player_name"] for p in squad}
    starters.add(expensive_in["player_name"])

    comparison = {
        "non_starters": [],
        "players_to_drop": [squad[0]],
        "players_to_add": [expensive_in],
    }

    # Default cap blocks the trade
    current_total = sum(p["price"] for p in squad)
    no_room = recommend_trades(squad, comparison, starters, pool,
                               salary_cap=current_total)
    assert no_room == []

    # Effective cap large enough → trade is allowed
    plenty = recommend_trades(squad, comparison, starters, pool,
                              salary_cap=current_total + 1_000_000)
    assert len(plenty) == 1

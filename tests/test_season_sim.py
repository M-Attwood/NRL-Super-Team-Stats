"""Tests for the season simulator's rules: lineup quotas, price history and
who counts as available for a trade."""
import pandas as pd

from season_sim import RoundData, best_lineup, horizon_value, price_table


def test_lineup_respects_active_quotas():
    # Six hookers and one fullback: only 2 HOK slots + 1 FLEX can take hookers.
    players = pd.DataFrame(
        {"positions": ["HOK"] * 6 + ["FLB"], "value": [90, 80, 70, 60, 50, 40, 10]},
        index=[f"h{i}" for i in range(6)] + ["f0"])
    chosen = set(best_lineup(players, "value"))
    assert chosen == {"h0", "h1", "h2", "f0"}     # 2 HOK + best hooker as FLEX + FLB


def test_price_table_carries_prices_through_games_missed():
    rounds = pd.DataFrame({"player_name": ["A", "A"], "year": [2026, 2026],
                           "round": [1, 3], "price": [500_000, 550_000]})
    totals = {2026: pd.DataFrame({"player_name": ["A", "B"],
                                  "start_price": [480_000, 200_000]})}
    px = price_table(rounds, totals, 2026, last_round=4)
    assert list(px.loc["A"]) == [500_000, 500_000, 550_000, 550_000]
    assert list(px.loc["B"]) == [200_000] * 4         # never played: start price


def _round(named, teams_playing):
    frame = pd.DataFrame({"team": ["MEL", "BRO", "PAR"], "named": named,
                          "value": [50.0, 50.0, 50.0]}, index=["mel", "bro", "par"])
    return RoundData(frame, set(teams_playing))


def test_horizon_value_counts_byes_and_unnamed_players():
    data = {
        1: _round([True, False, False], {"MEL", "PAR"}),   # BRO bye, PAR player not named
        2: _round([True, True, True], {"MEL", "BRO", "PAR"}),
        3: _round([True, True, True], {"MEL", "BRO"}),     # PAR bye
    }
    value = horizon_value(data, 1, "value")
    assert value["mel"] == 150          # plays all three rounds
    assert value["bro"] == 100          # bye now, back for rounds 2-3
    assert value["par"] == 0            # team played, player not named: assumed out

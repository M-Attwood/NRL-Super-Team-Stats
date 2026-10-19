"""Tests for the season simulator's rules: lineup quotas, price history,
squad valuation and who counts as available in future rounds."""
import numpy as np
import pandas as pd
import pytest

from season_sim import (IMPROVED, ORIGINAL, RoundData, SeasonInputs, best_lineup,
                        greedy_lineup_value, outlook_for, price_table)


def test_lineup_respects_active_quotas():
    # Six hookers and one fullback: only 2 HOK slots + 1 FLEX can take hookers.
    players = pd.DataFrame(
        {"positions": ["HOK"] * 6 + ["FLB"], "value": [90, 80, 70, 60, 50, 40, 10]},
        index=[f"h{i}" for i in range(6)] + ["f0"])
    chosen = set(best_lineup(players, "value"))
    assert chosen == {"h0", "h1", "h2", "f0"}     # 2 HOK + best hooker as FLEX + FLB


def test_greedy_lineup_counts_quotas_and_doubles_captain():
    values = np.array([90.0, 80.0, 70.0, 60.0, 10.0])
    eligible = [["HOK"], ["HOK"], ["HOK"], ["HOK"], ["FLB"]]
    # 2 HOK + 1 FLEX hooker + FLB = 90+80+70+10, captain doubles the 90.
    assert greedy_lineup_value(values, eligible) == 90 + 80 + 70 + 10 + 90


def test_price_table_carries_prices_through_games_missed():
    rounds = pd.DataFrame({"player_name": ["A", "A"], "year": [2026, 2026],
                           "round": [1, 3], "price": [500_000, 550_000]})
    totals = {2026: pd.DataFrame({"player_name": ["A", "B"],
                                  "start_price": [480_000, 200_000]})}
    px = price_table(rounds, totals, 2026, last_round=4)
    assert list(px.loc["A"]) == [500_000, 500_000, 550_000, 550_000]
    assert list(px.loc["B"]) == [200_000] * 4         # never played: start price


def _inputs():
    frame = pd.DataFrame({
        "team": ["MEL", "BRO", "PAR"], "named": [True, False, False],
        "team_plays": [True, False, True], "p_next": [0.9, 0.8, 0.2],
        "season_avg_value": [50.0, 50.0, 50.0], "price_change_3": [0.0, 0.0, 0.0],
    }, index=["mel", "bro", "par"])
    preds = pd.DataFrame({1: [50.0] * 3, 2: [50.0] * 3, 3: [50.0] * 3}, index=frame.index)
    rounds = {r: RoundData(frame, preds, teams) for r, teams in
              [(1, {"MEL", "PAR"}), (2, {"MEL", "BRO", "PAR"}), (3, {"MEL", "BRO"})]}
    return SeasonInputs(2026, rounds, points_per_dollar=1e-4)


def test_outlook_uses_team_lists_now_and_availability_later():
    e = outlook_for(_inputs(), 1, IMPROVED).expected
    assert list(e[1]) == [50, 0, 0]                    # only named players score now
    assert e.loc["bro", 2] == pytest.approx(40)       # back from bye: 50 x 0.8
    assert e.loc["par", 2] == pytest.approx(10)       # not named: 50 x 0.2
    assert e.loc["par", 3] == 0                       # PAR on bye in round 3


def test_original_strategy_assumes_unnamed_players_stay_out():
    e = outlook_for(_inputs(), 1, ORIGINAL).expected
    assert e.loc["bro", 2] == 50                      # bye now, counted back
    assert e.loc["par", 2] == 0                       # team played, not named: out

"""Tests for the season simulator's rules: lineup quotas, price history,
squad valuation and who counts as available in future rounds."""
import numpy as np
import pandas as pd
import pytest

from season_sim import (IMPROVED, ORIGINAL, RoundData, SeasonInputs, best_lineup,
                        greedy_lineup_value, outlook_for, price_table, spendable_trades)


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


def test_origin_weeks_use_origin_availability():
    inputs = _inputs()
    frame = inputs.rounds[1].frame.assign(p_origin=0.3, p_after_origin=0.6)
    inputs.rounds = {r: RoundData(frame, rd.preds, rd.teams_playing)
                     for r, rd in inputs.rounds.items()}
    inputs.origin_week = {2: 2, 3: 1}
    e = outlook_for(inputs, 1, IMPROVED).expected
    assert e.loc["bro", 2] == pytest.approx(15)       # Origin week: 50 x 0.3
    assert e.loc["mel", 3] == pytest.approx(30)       # the week after: 50 x 0.6


def test_original_strategy_assumes_unnamed_players_stay_out():
    e = outlook_for(_inputs(), 1, ORIGINAL).expected
    assert e.loc["bro", 2] == 50                      # bye now, counted back
    assert e.loc["par", 2] == 0                       # team played, not named: out


def test_improved_strategy_keeps_trades_for_the_run_home():
    # 46 trades + 5 boosts. Round 20 of 27: keep 1.5 x 7 rounds = 11 (rounded up).
    assert spendable_trades(IMPROVED, 20, 27, trades_used=34, boosts_left=0) == 1
    assert spendable_trades(IMPROVED, 20, 27, trades_used=36, boosts_left=0) == 0
    assert spendable_trades(IMPROVED, 27, 27, trades_used=44, boosts_left=0) == 2
    # The original strategy spends whatever is left.
    assert spendable_trades(ORIGINAL, 20, 27, trades_used=36, boosts_left=0) == 10


def test_reserve_saves_a_big_bye_rounds_full_allowance():
    byes = {12: 3, 15: 3, 18: 3}
    # Round 17: keep 3 for round 18 plus 1.5 x 9 later rounds = 16.5 -> 17.
    assert spendable_trades(IMPROVED, 17, 27, trades_used=27, boosts_left=0,
                            bye_trades=byes) == 2
    # Round 18 itself keeps only 1.5 x 9 = 13.5 -> 14, so all 3 trades can go.
    assert spendable_trades(IMPROVED, 18, 27, trades_used=29, boosts_left=0,
                            bye_trades=byes) == 3
    assert spendable_trades(ORIGINAL, 17, 27, trades_used=27, boosts_left=0,
                            bye_trades=byes) == 19

"""Tests for the season simulator's rules: lineup quotas, price history,
squad valuation and who counts as available in future rounds."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from season_sim import (IMPROVED, ORIGINAL, RoundData, SeasonInputs, best_lineup,
                        expected_rise, greedy_lineup_value, outlook_for, price_table,
                        round_score, spendable_trades)


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
    # A big bye round counts only the best few: here the top 2 plus captain.
    assert greedy_lineup_value(values, eligible, counted=2) == 90 + 80 + 90


def test_bye_rounds_count_the_best_13_with_the_captain_doubled():
    scores = pd.Series([10.0 * i for i in range(1, 19)], index=[f"p{i}" for i in range(1, 19)])
    # Normal round: all 18 plus the captain's score again.
    assert round_score(scores, "p18")["points"] == scores.sum() + 180
    # Bye round: best 13 after doubling the captain (p1 at 20 is not among them).
    bye = round_score(scores, "p1", counted=13)
    assert bye["points"] == sum(10.0 * i for i in range(6, 19))
    assert bye["captain_points"] == 0 and bye["empty_slots"] == 0
    # Only 9 fielded: every score counts and 4 of the 13 spots are empty.
    few = round_score(scores.iloc[:9], "p9", counted=13)
    assert few["points"] == scores.iloc[:9].sum() + 90 and few["empty_slots"] == 4


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
    # 46 trades, boosted ones included. Round 20 of 27: keep 1.25 x 7 = 8.75 -> 9.
    assert spendable_trades(IMPROVED, 20, 27, trades_used=36) == 1
    assert spendable_trades(IMPROVED, 20, 27, trades_used=37) == 0
    assert spendable_trades(IMPROVED, 27, 27, trades_used=44) == 2
    # The original strategy spends whatever is left.
    assert spendable_trades(ORIGINAL, 20, 27, trades_used=36) == 10


def test_reserve_can_save_a_big_bye_rounds_full_allowance():
    byes = {12: 4, 15: 4, 18: 4}
    saver = replace(IMPROVED, reserve=1.5, save_for_byes=True)
    # Round 17: keep 4 for round 18 plus 1.5 x 9 later rounds = 17.5 -> 18.
    assert spendable_trades(saver, 17, 27, trades_used=26, bye_trades=byes) == 2
    # Round 18 itself keeps only 1.5 x 9 = 13.5 -> 14, so all 4 trades can go.
    assert spendable_trades(saver, 18, 27, trades_used=28, bye_trades=byes) == 4
    # Off by default: keep 1.25 x 10 = 12.5 -> 13, bye rounds paced like any other.
    assert spendable_trades(IMPROVED, 17, 27, trades_used=26, bye_trades=byes) == 7


def test_bye_exempt_lets_a_bye_round_use_its_full_allowance():
    byes = {12: 4, 15: 4, 18: 4}
    exempt = replace(IMPROVED, bye_exempt=True)
    # Round 18: keep 1.25 x 9 = 11.25 -> 12; with 13 left only 1 could go.
    assert spendable_trades(IMPROVED, 18, 27, trades_used=33, bye_trades=byes) == 1
    assert spendable_trades(exempt, 18, 27, trades_used=33, bye_trades=byes) == 4
    assert spendable_trades(exempt, 18, 27, trades_used=44, bye_trades=byes) == 2   # never past 46
    assert spendable_trades(exempt, 17, 27, trades_used=33, bye_trades=byes) == 0   # normal round


def test_cash_cows_are_valued_by_their_whole_expected_rise():
    frame = pd.DataFrame({
        "price": [250_000.0, 250_000.0, 700_000.0], "predicted_points": [60.0, 60.0, 90.0],
        "price_change_3": [40_000.0, 40_000.0, 5_000.0], "named": [True, False, True],
    }, index=["cow", "unplayed", "premium"])
    rounds = {1: RoundData(frame, pd.DataFrame(), set())}
    inputs = SeasonInputs(2026, rounds, points_per_dollar=1e-4, price_intercept=0.0)
    off = expected_rise(inputs, 1, IMPROVED)
    assert list(off) == [40_000, 40_000, 5_000]               # off: the 3-game price model
    on = expected_rise(inputs, 1, replace(IMPROVED, cow_price=300_000))
    assert on["cow"] == 350_000             # deserves $600K (60 pts / 1e-4), costs $250K
    assert on["unplayed"] == 40_000         # no games yet: no cow bonus
    assert on["premium"] == 5_000           # over the price cap

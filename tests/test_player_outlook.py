"""Tests for the availability examples: what was known before each round
and whether the player was named."""
import pandas as pd

from player_outlook import availability_examples, debut_rate


def _rounds():
    # MEL play rounds 1-5. Player A plays 1, 2, then misses 3-4, returns in 5.
    rows = []
    for r in range(1, 6):
        rows.append({"player_name": "Team, Mate", "year": 2026, "round": r, "team": "MEL",
                     "score": 50, "mins": 80, "jersey_number": 10, "price": 400_000})
        if r in (1, 2, 5):
            rows.append({"player_name": "A, Player", "year": 2026, "round": r, "team": "MEL",
                         "score": 60, "mins": 80, "jersey_number": 3, "price": 500_000})
    df = pd.DataFrame(rows)
    for col in ["pts_base", "pts_attack", "pts_create", "pts_evade", "pts_negative",
                "base_power_pts", "stat_tackles", "stat_missed_tackles", "stat_tackle_breaks",
                "stat_line_breaks", "stat_lb_assists", "stat_offloads", "stat_tries",
                "stat_try_assists", "stat_goals", "stat_hitups_h8", "stat_hitups_hu",
                "stat_errors", "stat_penalties", "break_even", "season_price_change"]:
        df[col] = 0.0
    df["positions"] = "CTW"
    return df


def test_examples_cover_every_team_game_after_debut():
    ex = availability_examples(_rounds())
    a = ex[ex["player_name"] == "A, Player"].set_index("next_round")
    assert list(a.index) == [2, 3, 4, 5]
    assert list(a["named"]) == [True, False, False, True]


def test_missed_streak_counts_team_games_since_last_appearance():
    ex = availability_examples(_rounds())
    a = ex[ex["player_name"] == "A, Player"].set_index("next_round")
    assert a.loc[3, "missed_streak"] == 0      # played round 2
    assert a.loc[4, "missed_streak"] == 1      # missed round 3
    assert a.loc[5, "missed_streak"] == 2      # missed rounds 3 and 4
    assert a.loc[5, "games_played"] == 2       # history stops at the last game


def test_debut_rate_counts_players_still_waiting_for_a_game():
    rounds = _rounds()
    totals = {2026: pd.DataFrame({"player_name": ["Team, Mate", "A, Player", "Never, Played"]})}
    # Everyone who plays debuts in round 1, so nobody debuts in rounds 2-5.
    assert debut_rate(rounds, totals) == 0.0

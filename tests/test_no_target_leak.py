"""Leakage tests for the next-game training pairs.

The original model predicted a season's average from that same season's
score components, which made validation R² look like 0.98. These tests pin
the fix: a pair's features may only use games up to and including game k,
and its target is the score in game k+1.
"""
import pandas as pd

from model import FEATURES, TARGET, build_training_pairs, split_season


def _season(year=2025, scores=(40, 60, 80, 100), player="Test, Player"):
    rows = [{
        "player_name": player, "current_round": f"{i + 1:02d}", "team": "MEL",
        "positions": "HOK", "total_points": s, "total_minutes": 80,
        "price": 500_000, "jersey_number": 9, "vs_team": "BRO",
    } for i, s in enumerate(scores)]
    rows.append({"player_name": player, "current_round": "Totals", "team": "MEL",
                 "positions": "HOK", "total_points": sum(scores),
                 "avg_points": sum(scores) / len(scores), "games_played": len(scores),
                 "price": 500_000, "start_price": 500_000})
    return split_season(pd.DataFrame(rows), year)


def _pairs(scores):
    rounds, totals = _season(scores=scores)
    return build_training_pairs(rounds, {2025: totals}).sort_values("round")


def test_target_is_not_a_feature():
    assert TARGET not in FEATURES
    assert "score" not in FEATURES


def test_target_is_the_next_game():
    pairs = _pairs((40, 60, 80, 100))
    assert list(pairs[TARGET]) == [60, 80, 100]
    assert list(pairs["next_round"]) == [2, 3, 4]


def test_features_only_use_games_played_so_far():
    pairs = _pairs((40, 60, 80, 100))
    first = pairs.iloc[0]          # state after game 1, predicting game 2
    assert first["season_avg"] == 40
    assert first["games_played"] == 1
    second = pairs.iloc[1]         # state after games 1-2
    assert second["season_avg"] == 50
    assert second["last1"] == 60


def test_changing_a_future_score_never_changes_earlier_features():
    a = _pairs((40, 60, 80, 100))
    b = _pairs((40, 60, 80, 5))    # only the final game differs
    pd.testing.assert_frame_equal(
        a[FEATURES].iloc[:2].reset_index(drop=True),
        b[FEATURES].iloc[:2].reset_index(drop=True))


def test_five_eighth_position_is_not_split_on_slash():
    rounds, totals = _season()
    rounds["positions"] = "5/8"
    pairs = build_training_pairs(rounds, {2025: totals})
    assert (pairs["pos_5_8"] == 1).all()
    assert (pairs[["pos_HOK", "pos_FLB"]] == 0).all().all()

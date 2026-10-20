"""Leakage tests for the next-game training pairs.

The original model predicted a season's average from that same season's
score components, which made validation R² look like 0.98. These tests pin
the fix: a pair's features may only use games up to and including game k,
and its target is the score in game k+1. The one exception is game k+1's
team list (named to start or on the interchange), published before the
game: its jersey number may enter, nothing the game produced.
"""
import pandas as pd
import pytest

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


# ── The team list: the one input from game k+1, and only its jersey ─────────

def _pairs_with(jerseys, scores=(40, 60, 80, 100), mins=80):
    rounds, totals = _season(scores=scores)
    rounds["jersey_number"] = list(jerseys)
    rounds["mins"] = mins
    return build_training_pairs(rounds, {2025: totals}).sort_values("round")


def test_next_games_team_list_is_an_input():
    # Starts game 1, interchange (14) in game 2, reserve called up (18) in game 3.
    pairs = _pairs_with([9, 14, 18, 9])
    assert list(pairs["named_bench"]) == [1, 0, 0]     # predicting games 2, 3, 4
    assert list(pairs["named_start"]) == [0, 0, 1]
    assert list(pairs["last_started"]) == [1, 0, 0]    # games 1, 2, 3 as before


def test_only_the_team_list_of_the_next_game_enters():
    base = _pairs_with([9, 9, 9, 9])
    # The next game's result (score, minutes, stats) moves no input.
    rounds, totals = _season(scores=(40, 60, 80, 100))
    rounds.loc[3, ["total_points", "score", "mins", "stat_tackles"]] = [5, 5, 10, 1]
    other = build_training_pairs(rounds, {2025: totals}).sort_values("round")
    pd.testing.assert_frame_equal(base[FEATURES].reset_index(drop=True),
                                  other[FEATURES].reset_index(drop=True), check_dtype=False)
    # Its jersey moves only the team-list inputs of the pair predicting it.
    benched = _pairs_with([9, 9, 9, 16])
    changed = [c for c in FEATURES
               if not base[c].reset_index(drop=True).equals(benched[c].reset_index(drop=True))]
    assert changed == ["named_start", "named_bench"]
    assert list(benched["named_bench"]) == [0, 0, 1]


def test_prediction_frame_uses_the_published_team_list():
    from model import build_prediction_frame
    rounds, totals = _season(scores=(40, 60, 80, 100))
    totals["bye_round"] = 0
    before = build_prediction_frame(rounds, {2025: totals}, 2025, 3)
    assert before["named_start"].iloc[0] == 1          # list not out: last game's role
    listed = build_prediction_frame(rounds, {2025: totals}, 2025, 3,
                                    team_list=pd.Series({"Test, Player": 15}))
    assert listed["named_start"].iloc[0] == 0 and listed["named_bench"].iloc[0] == 1
    # Nothing else changes: the state still stops at game 2.
    same = [c for c in FEATURES if c not in ("named_start", "named_bench")]
    pd.testing.assert_frame_equal(before[same], listed[same])


def test_rookie_estimate_rewards_being_named_to_start():
    from model import fit_rookie_baseline
    rows = []
    for y in (2024, 2025):
        for i in range(30):
            starts = i % 2 == 0
            rows.append({"player_name": f"R{y}{i}", "year": y, "round": 1,
                         "price": 200_000 + 1000 * i, "jersey_number": 10 if starts else 16,
                         "score": 45.0 if starts else 25.0})
    fit = fit_rookie_baseline(pd.DataFrame(rows))
    # 2024 has no season before it in the data, so only 2025's rookies count.
    assert fit["named_start"] - fit["named_bench"] == pytest.approx(20.0)


def test_role_transitions_count_how_often_a_role_is_kept():
    from model import role_transitions
    rows = [("A", r, j) for r, j in zip(range(1, 5), (9, 9, 15, 9))]
    rows += [("B", r, 15) for r in range(1, 4)]
    rounds = pd.DataFrame(rows, columns=["player_name", "round", "jersey_number"]).assign(year=2025)
    t = role_transitions(rounds, ahead=1)[0]          # rows: start, interchange, other
    assert list(t[0]) == [0.5, 0.5, 0.0]              # A started twice: kept it once
    assert list(t[1]) == pytest.approx([1 / 3, 2 / 3, 0.0])


def test_unknown_team_lists_average_the_prediction_over_roles():
    import numpy as np
    from model import predict_over_roles

    class Fake:
        def predict(self, df):
            return (10 * df["named_start"] + 5 * df["named_bench"]).to_numpy()
    frame = pd.DataFrame({"named_start": [1.0], "named_bench": [0.0]})
    probs = np.array([[0.5, 0.25, 0.25]])             # start, interchange, other
    assert predict_over_roles(Fake(), frame, probs)[0] == pytest.approx(10 * 0.5 + 5 * 0.25)

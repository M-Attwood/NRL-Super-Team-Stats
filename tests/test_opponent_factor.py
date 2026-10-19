"""Tests for opponent_factor — how generous the next opponent has been to a
position, using only rounds before the game being predicted."""
import pandas as pd
import pytest

from model import OPP_PRIOR_GAMES, opponent_factor


def _rounds(rows):
    df = pd.DataFrame(rows, columns=["round", "vs_team", "primary_pos", "score"])
    df["year"] = 2026
    return df


ROUNDS = _rounds([
    # Round 1: BRO concede big scores to hookers, MEL concede small ones.
    (1, "BRO", "HOK", 90), (1, "MEL", "HOK", 30),
    # Round 2 data must be invisible when predicting round 2.
    (2, "BRO", "HOK", 0), (2, "MEL", "HOK", 200),
])


def _query(opp, rnd, pos="HOK"):
    return pd.DataFrame({"year": [2026], "opp": [opp], "primary_pos": [pos],
                         "next_round": [rnd]})


def test_round_one_has_no_history():
    assert opponent_factor(ROUNDS, _query("BRO", 1)).iloc[0] == 1.0


def test_generous_defence_scores_above_one_and_only_uses_earlier_rounds():
    # League HOK average after round 1 = 60; BRO conceded 90 in one game.
    expected = ((90 + OPP_PRIOR_GAMES * 60) / (1 + OPP_PRIOR_GAMES)) / 60
    assert opponent_factor(ROUNDS, _query("BRO", 2)).iloc[0] == pytest.approx(expected)
    assert opponent_factor(ROUNDS, _query("MEL", 2)).iloc[0] < 1.0


def test_unknown_opponent_or_bye_is_neutral():
    assert opponent_factor(ROUNDS, _query(None, 3)).iloc[0] == 1.0
    assert opponent_factor(ROUNDS, _query("XXX", 3)).iloc[0] == 1.0


def test_result_is_aligned_to_query_index():
    queries = pd.concat([_query("BRO", 2), _query("MEL", 2)])
    queries.index = [10, 20]
    out = opponent_factor(ROUNDS, queries)
    assert list(out.index) == [10, 20]
    assert out.loc[10] > 1.0 > out.loc[20]

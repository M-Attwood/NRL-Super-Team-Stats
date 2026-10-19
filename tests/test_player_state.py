"""Tests for player_state and build_prediction_frame — the features
describing a player *after* each game, and the frame used for live picks."""
import pandas as pd
import pytest

from model import build_prediction_frame, player_state, split_season


def _raw(player, year, scores, team="MEL", jersey=9, price=500_000):
    rows = [{"player_name": player, "current_round": f"{i + 1:02d}", "team": team,
             "positions": "HOK", "total_points": s, "total_minutes": 80,
             "price": price, "jersey_number": jersey, "vs_team": "BRO"}
            for i, s in enumerate(scores)]
    rows.append({"player_name": player, "current_round": "Totals", "team": team,
                 "positions": "HOK", "total_points": sum(scores),
                 "avg_points": sum(scores) / max(len(scores), 1),
                 "games_played": len(scores), "price": price, "start_price": price,
                 "bye_round": 12})
    return pd.DataFrame(rows)


def test_rolling_form_and_jersey_features():
    rounds, _ = split_season(_raw("A, B", 2026, [10, 20, 30, 40, 50], jersey=15), 2026)
    state = player_state(rounds).set_index("round")
    assert state.loc[5, "season_avg"] == 30
    assert state.loc[5, "last3_avg"] == 40
    assert state.loc[2, "season_std"] == pytest.approx(7.0711, rel=1e-3)
    assert state.loc[5, "last_benched"] == 1.0
    assert state.loc[5, "start_share"] == 0.0


def test_prediction_frame_uses_only_games_before_the_round():
    r25, t25 = split_season(_raw("A, B", 2025, [50, 50]), 2025)
    r26, t26 = split_season(_raw("A, B", 2026, [10, 20, 90]), 2026)
    rounds = pd.concat([r25, r26], ignore_index=True)
    frame = build_prediction_frame(rounds, {2025: t25, 2026: t26}, 2026, next_round=3)
    row = frame.iloc[0]
    assert row["season_avg"] == 15          # rounds 1-2 only; round 3's 90 unseen
    assert row["prev_season_avg"] == 50
    assert not row["is_rookie"]


def test_pre_season_falls_back_to_last_season_and_flags_rookies():
    r25, t25 = split_season(_raw("Vet, A", 2025, [70, 80]), 2025)
    r26, t26 = split_season(pd.concat([_raw("Vet, A", 2026, [60]),
                                       _raw("New, B", 2026, [])]), 2026)
    rounds = pd.concat([r25, r26], ignore_index=True)
    frame = build_prediction_frame(rounds, {2025: t25, 2026: t26}, 2026,
                                   next_round=1).set_index("player_name")
    assert frame.loc["Vet, A", "season_avg"] == 75
    assert frame.loc["Vet, A", "rounds_since_last"] == 5
    assert bool(frame.loc["New, B", "is_rookie"])

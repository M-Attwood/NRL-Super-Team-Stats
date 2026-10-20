"""Tests for the availability examples: what was known before each round,
whether the player was named, and the State of Origin features."""
import pandas as pd
import pytest

from player_outlook import (add_origin, availability_examples, debut_rate, origin_misses,
                            origin_squads, origin_weeks, team_rounds)


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


def _origin_rounds():
    # Six teams. Round 3 is an Origin week (five teams on bye), round 4 the week after.
    # A (T0) plays 1, 2, 4: sat out the Origin week after playing round 2.
    # B (T0) plays 1, 3, 4: missed round 2, an ordinary week.
    # C (T1) plays 1, 2: T1 is on bye in round 3, so C sat out round 4.
    rows = [(f"Mate, T{t}", 2026, r, f"T{t}") for t in range(6)
            for r in (1, 2, 3, 4) if r != 3 or t == 0]
    rows += [("A", 2026, r, "T0") for r in (1, 2, 4)]
    rows += [("B", 2026, r, "T0") for r in (1, 3, 4)]
    rows += [("C", 2026, r, "T1") for r in (1, 2)]
    return pd.DataFrame(rows, columns=["player_name", "year", "round", "team"])


def test_origin_weeks_come_from_the_draw():
    weeks = origin_weeks(team_rounds(_origin_rounds()))
    assert weeks.loc[2026].to_dict() == {1: 0, 2: 0, 3: 2, 4: 1}


def test_origin_misses_need_the_previous_team_game_played():
    rounds = _origin_rounds()
    tr = team_rounds(rounds)
    misses = origin_misses(rounds, tr, origin_weeks(tr))
    assert set(zip(misses["player_name"], misses["round"])) == {("A", 3), ("C", 4)}


def test_origin_features_only_count_earlier_misses():
    rounds = _origin_rounds()
    tr = team_rounds(rounds)
    weeks = origin_weeks(tr)
    q = pd.DataFrame({"player_name": ["A", "A", "A"], "year": 2026, "next_round": [2, 3, 4]})
    f = add_origin(q, origin_misses(rounds, tr, weeks), weeks)
    assert list(f["origin_week"]) == [0, 2, 1]
    assert list(f["origin_missed"]) == [0, 0, 1]
    assert f["origin_missed_last_season"].isna().all()      # 2025 not in the data


def test_origin_squads_count_only_the_two_previous_series():
    squads = origin_squads()
    assert "Cleary, Nathan" in squads[2025] and "Cleary, Nathan" not in squads[2024]
    q = pd.DataFrame({"player_name": ["Cleary, Nathan", "Nobody, Unknown"],
                      "year": 2026, "next_round": 5})
    f = add_origin(q, pd.DataFrame(columns=["player_name", "year", "round"]),
                   origin_weeks(team_rounds(_origin_rounds())))
    assert list(f["origin_squads_last2"]) == [1, 0]   # 2025 yes, 2024 no; 2026 never used


# ── The price rule ────────────────────────────────────────────────────────────

def _rule():
    from player_outlook import PriceRule
    return PriceRule(K=1000.0, mu=0.75, floor=200_000.0)


def test_price_rule_is_zero_sum_when_mu_balances_the_round():
    import numpy as np
    from player_outlook import PriceRule
    price = np.array([300_000.0, 600_000.0, 800_000.0])
    sum3 = np.array([210.0, 150.0, 270.0])        # 70, 50 and 90 a game
    mu = price.sum() / (price + 1000 * sum3).sum()
    new = PriceRule(1000.0, mu, 0.0).next_price(price, sum3)
    assert abs(new.sum() - price.sum()) <= 300   # $100 rounding per player
    assert new[0] > price[0] and new[1] < price[1]   # cheap scorer rises, dud falls


def test_break_even_holds_the_price():
    rule = _rule()
    be = rule.break_even(450_000.0, 100.0)
    assert rule.next_price(450_000.0, 100.0 + be) == 450_000.0


def test_price_never_falls_below_the_floor():
    assert _rule().next_price(205_000.0, 0.0) == 200_000.0


def test_fit_price_rule_recovers_k_and_mu():
    import numpy as np
    from player_outlook import fit_price_rule
    rng = np.random.default_rng(0)
    rule, rows = _rule(), []
    for i in range(40):
        price, scores = 300_000.0 + 10_000 * i, rng.integers(20, 90, size=6)
        for g, sc in enumerate(scores, start=1):
            if g >= 3:
                price = float(rule.next_price(price, scores[g - 3:g].sum()))
            rows.append({"player_name": f"P{i}", "year": 2026, "round": g,
                         "price": price, "score": float(sc)})
    totals = {2026: pd.DataFrame({"start_price": [200_000.0]})}
    table = fit_price_rule(pd.DataFrame(rows), totals)
    assert table["K"].to_numpy() == pytest.approx(1000.0, rel=0.01)
    assert table["mu"].to_numpy() == pytest.approx(0.75, abs=0.001)


def test_rule_known_before_a_round_uses_only_earlier_moves():
    from player_outlook import rule_before
    table = pd.DataFrame({"K": [900.0, 1000.0, 1010.0], "mu": [0.74, 0.75, 0.70]},
                         index=pd.MultiIndex.from_tuples([(2025, 3), (2026, 3), (2026, 4)],
                                                         names=["year", "round"]))
    assert rule_before(table, 2026, 4, 1.0).mu == 0.75      # round 4's own mu not yet known
    assert rule_before(table, 2026, 3, 1.0).K == 900.0      # nothing moved yet: last season
    assert rule_before(table, 2025, 3, 1.0) is None


def test_projected_price_path_waits_for_three_games_and_skips_byes():
    import numpy as np
    from player_outlook import project_prices
    rule = _rule()
    # One game played (scored 60); predicted 60 a game; bye in the second round.
    path = project_prices(rule, [300_000.0], [0.0], [60.0], [1], np.full((1, 4), 60.0),
                          np.array([[1.0, 0.0, 1.0, 0.5]]))[0]
    assert path[1] == 300_000.0                  # game 2: prices don't move yet
    assert path[2] == path[1]                    # bye: nothing changes
    third = float(rule.next_price(300_000.0, 180.0))
    assert path[3] == third                      # game 3: the rule on 60+60+60
    assert path[4] == pytest.approx(third + 0.5 * (float(rule.next_price(third, 180.0)) - third))


def test_price_examples_trade_at_the_price_after_the_last_game():
    from player_outlook import price_examples
    rows = [{"player_name": "A", "year": 2026, "round": r, "team": "MEL", "score": 50,
             "mins": 80, "jersey_number": 3, "price": 400_000 + 10_000 * r}
            for r in range(1, 6)]
    df = pd.DataFrame(rows)
    for col in ["pts_base", "pts_attack", "pts_create", "pts_evade", "pts_negative",
                "base_power_pts", "stat_tackles", "stat_missed_tackles", "stat_tackle_breaks",
                "stat_line_breaks", "stat_lb_assists", "stat_offloads", "stat_tries",
                "stat_try_assists", "stat_goals", "stat_hitups_h8", "stat_hitups_hu",
                "stat_errors", "stat_penalties", "break_even", "season_price_change"]:
        df[col] = 0.0
    ex = price_examples(df).set_index("round")
    assert list(ex.index) == [1, 2]                         # 3 more games needed
    assert ex.loc[1, "price"] == 410_000                    # after game 1
    assert ex.loc[1, "price_change"] == 30_000              # after game 4 minus after game 1

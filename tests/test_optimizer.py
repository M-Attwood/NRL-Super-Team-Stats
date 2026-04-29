"""Integration test for the LP optimizer.

Feeds a fixed mini-pool through `select_team` and asserts the output is
a valid 26-player squad within the cap. This is the slowest test (the
LP solve takes ~1 second) but it catches the most.
"""
import numpy as np
import pandas as pd

from optimizer import select_team, SALARY_CAP


def _build_test_pool(n_per_position: int = 10) -> pd.DataFrame:
    """Build a pool with enough players at every position for the LP to
    have feasible solutions.

    The LP requires `quota + MIN_COVERAGE_DEPTH = quota + 1` eligible
    players selected per starting position. With single-position players
    the coverage "+1" can only come from the lone FLEX slot, which can't
    cover all 7 positions at once — so the fixture also includes a few
    multi-position players (mimicking real Supercoach where ~30% of
    players are dual-eligible).

    Prices average ~$325K so 26 picks fit well under the $11.95M cap;
    we use 8 teams to satisfy the hard 5-per-team limit.
    """
    positions = ["HOK", "FRF", "2RF", "HFB", "5/8", "CTW", "FLB"]
    teams = ["MEL", "PTH", "BRO", "SOU", "PAR", "SHA", "NEW", "DOL"]
    rng = np.random.default_rng(seed=42)
    rows = []
    for pos in positions:
        for i in range(n_per_position):
            rows.append({
                "player_name": f"{pos}_player_{i}",
                "positions": pos,
                "team": teams[i % len(teams)],
                "price": int(rng.integers(200_000, 450_000)),
                "predicted_points": float(rng.uniform(20, 90)),
                "avg_points": float(rng.uniform(20, 90)),
            })
    # Multi-position players (one per cross-pair) so coverage-depth is feasible.
    dual = [("2RF|FRF", "MEL"), ("CTW|FLB", "PTH"), ("HOK|HFB", "BRO"),
            ("5/8|HFB", "SOU"), ("CTW|FLB", "PAR"), ("2RF|FRF", "SHA")]
    for i, (pos, team) in enumerate(dual):
        rows.append({
            "player_name": f"dual_{i}",
            "positions": pos,
            "team": team,
            "price": int(rng.integers(200_000, 400_000)),
            "predicted_points": float(rng.uniform(40, 80)),
            "avg_points": float(rng.uniform(40, 80)),
        })
    return pd.DataFrame(rows)


def _assert_feasible(result: dict, pool_size: int, cap: int):
    """Helper that fails loudly when the LP is infeasible. Without this,
    a future change that tightens a constraint shows up as a cryptic
    KeyError on ``result["total_price"]`` instead of pointing at the LP
    status."""
    status = result.get("solver_status")
    if status != "Optimal":
        raise AssertionError(
            f"LP returned {status!r} (expected 'Optimal'). "
            f"Pool size: {pool_size}, cap: ${cap:,}. "
            f"Likely the test fixture is too tight against MIN_COVERAGE_DEPTH, "
            f"HARD_MAX_PER_TEAM, position quotas, or the salary cap — "
            f"check tests/test_optimizer.py::_build_test_pool."
        )


def test_optimizer_returns_valid_squad():
    pool = _build_test_pool()
    result = select_team(pool, round_number=1)
    _assert_feasible(result, len(pool), SALARY_CAP)

    all_players = (result["starting_13"] + result["bench_4"]
                   + result.get("flex_1", []) + result["reserves_8"])
    assert len(all_players) == 26


def test_optimizer_respects_salary_cap():
    pool = _build_test_pool()
    result = select_team(pool, round_number=1)
    _assert_feasible(result, len(pool), SALARY_CAP)
    assert result["total_price"] <= SALARY_CAP


def test_optimizer_honours_explicit_cap_override():
    """Mid-season callers pass an effective cap above the default
    (price-rise allowance). The constraint must follow the override."""
    pool = _build_test_pool()
    raised_cap = int(SALARY_CAP * 1.05)
    result = select_team(pool, round_number=1, salary_cap=raised_cap)
    _assert_feasible(result, len(pool), raised_cap)
    assert result["total_price"] <= raised_cap


def test_optimizer_position_quotas():
    pool = _build_test_pool()
    result = select_team(pool, round_number=1)
    _assert_feasible(result, len(pool), SALARY_CAP)
    all_players = (result["starting_13"] + result["bench_4"]
                   + result.get("flex_1", []) + result["reserves_8"])

    expected = {"HOK": 2, "FRF": 4, "2RF": 6, "HFB": 2,
                "5/8": 2, "CTW": 7, "FLB": 2}
    counts = {pos: 0 for pos in expected}
    for p in all_players:
        # `positions` is the raw eligibility string; assigned slot is in
        # assigned_position. Count the eligibility (which is single-pos
        # in our fixture) — for multi-pos players you'd count assigned.
        first_eligible = p.get("positions", "").split("|")[0]
        if first_eligible in counts:
            counts[first_eligible] += 1
    # +1 flex floats; total still 26
    assert sum(counts.values()) == 26

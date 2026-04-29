"""Tests for the position-aware scoring-18 total.

This replaced a buggy 'top 18 by points regardless of role' sum.
The greedy must respect position quotas: 13 starting + 4 bench + 1 flex.
"""
from trade_advisor import _scoring_18_total


def _p(name, pos, pts, team="MEL"):
    return {"player_name": name, "positions": pos, "team": team,
            "predicted_points": pts}


def test_overweighted_position_does_not_inflate_total():
    """If you have 8 great CTWs but only 4 CTW slots + 1 flex,
    only 5 of them should score. The buggy version summed all 8."""
    squad = []
    # 8 CTWs at 100 pts each
    squad.extend(_p(f"ctw_{i}", "CTW", 100) for i in range(8))
    # Fill remaining 18 slots with 0-pt placeholders so they don't crowd in
    fillers = [("HOK", 2), ("FRF", 4), ("2RF", 6), ("HFB", 2),
               ("5/8", 2), ("FLB", 2)]
    for pos, n in fillers:
        squad.extend(_p(f"{pos}_{i}", pos, 0) for i in range(n))
    assert len(squad) == 26

    total = _scoring_18_total(squad)
    # 7 CTWs in active slots (4 starting + flex slot? FLEX takes the 5th best
    # remaining unplaced player). Quotas: CTW active = 4 (starting) + 0
    # (no CTW bench). So 4 CTWs in starting + 1 in flex = 5 CTWs scoring.
    # Buggy version would have summed top 18 regardless = 8 * 100 + filler = 800.
    # Correct: 5 * 100 = 500. (Other slots score 0.)
    assert total == 500.0


def test_confirmed_starter_filter_zeroes_non_starters():
    """Players not in confirmed_starters score 0 (didn't take the field)."""
    squad = [
        _p("playing", "HOK", 80),
        _p("benched", "HOK", 90),
    ]
    # Pad with zero-point fillers so we hit 26
    fillers = [("FRF", 4), ("2RF", 6), ("HFB", 2),
               ("5/8", 2), ("CTW", 8), ("FLB", 2)]
    for pos, n in fillers:
        squad.extend(_p(f"{pos}_{i}", pos, 0) for i in range(n))
    assert len(squad) == 26

    total = _scoring_18_total(squad, confirmed_starters={"playing"})
    # Only 'playing' contributes its 80. 'benched' filtered out as non-starter.
    assert total == 80.0


def test_handles_empty_squad_gracefully():
    assert _scoring_18_total([]) == 0.0

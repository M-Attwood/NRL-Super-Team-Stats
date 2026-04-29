"""Tests for position quota validation.

Why: every trade recommendation is gated by `validate_position_quotas`.
A bug here means we'd recommend trades that produce structurally invalid
squads (too many CTW, no FLB, etc.) — which Supercoach itself would
reject when you tried to save the team.
"""
import pytest

from planner import validate_position_quotas


def test_valid_squad_passes(valid_squad_26):
    assert validate_position_quotas(valid_squad_26) is True


def test_too_many_at_one_position_fails(valid_squad_26):
    # Replace one HOK with an extra CTW — now CTW=8 (max 7), HOK=1 (min 2)
    bad = list(valid_squad_26)
    for p in bad:
        if p["positions"] == "HOK":
            p["positions"] = "CTW"
            break
    assert validate_position_quotas(bad) is False


def test_short_squad_fails():
    # 25 players is not 26
    short = [{"player_name": f"p{i}", "positions": "HOK", "team": "MEL",
              "price": 200_000} for i in range(25)]
    assert validate_position_quotas(short) is False


def test_multi_position_player_can_fill_either_slot():
    """A 2RF|FRF player can satisfy either a 2RF or FRF quota slot.
    The validator must not double-count or refuse."""
    def p(name, pos):
        return {"player_name": name, "positions": pos, "team": "MEL",
                "price": 200_000}

    # Build a squad short one FRF, with a 2RF|FRF flex available
    squad = []
    quotas = [("HOK", 2), ("FRF", 3), ("2RF", 6), ("HFB", 2),
              ("5/8", 2), ("CTW", 7), ("FLB", 2), ("CTW", 1)]  # 25 + 1 flex CTW
    for pos, n in quotas:
        for i in range(n):
            squad.append(p(f"{pos}_{i}_{len(squad)}", pos))
    squad.append(p("dual_pos", "2RF|FRF"))
    assert len(squad) == 26
    assert validate_position_quotas(squad) is True

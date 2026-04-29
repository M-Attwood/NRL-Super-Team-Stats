"""Shared pytest fixtures.

`conftest.py` is auto-discovered by pytest — anything defined here is
available to every test in this directory tree without explicit imports.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable so `import trade_advisor` works
# without needing to install the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tiny_pool() -> pd.DataFrame:
    """A 4-player DataFrame mimicking the scraped data shape, including
    apostrophe-stripped names ("Too, Brian") that the matcher must reach
    when the user types "Brian To'o"."""
    return pd.DataFrame([
        {"player_name": "Grant, Harry", "team": "MEL",
         "positions": "HOK", "price": 540_000, "predicted_points": 65.0},
        {"player_name": "Too, Brian", "team": "PTH",
         "positions": "CTW", "price": 600_000, "predicted_points": 80.0},
        {"player_name": "Papalii, Isaiah", "team": "PTH",
         "positions": "2RF|FRF", "price": 631_000, "predicted_points": 72.0},
        {"player_name": "Cleary, Nathan", "team": "PTH",
         "positions": "HFB", "price": 866_000, "predicted_points": 89.0},
    ])


@pytest.fixture
def valid_squad_26() -> list[dict]:
    """A position-quota-valid 26-player squad
    (HOK=2, FRF=4, 2RF=6, HFB=2, 5/8=2, CTW=7, FLB=2, +1 flex)."""
    def p(name, pos, price=300_000, pts=50.0, team="MEL"):
        return {"player_name": name, "positions": pos, "team": team,
                "price_usd": price, "price": price, "predicted_points": pts}
    squad = []
    quotas = [("HOK", 2), ("FRF", 4), ("2RF", 6), ("HFB", 2),
              ("5/8", 2), ("CTW", 7), ("FLB", 2)]
    for pos, n in quotas:
        for i in range(n):
            squad.append(p(f"{pos}_{i}", pos))
    # Plus the flex slot (any position) — give it CTW so it's not a 5th HFB
    squad.append(p("flex_player", "CTW"))
    assert len(squad) == 26
    return squad

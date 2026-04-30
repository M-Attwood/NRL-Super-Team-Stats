"""
squad_state.py — Shared squad / season state loader.

Single source of truth for the user's actual current squad, current round,
trades remaining, and Origin-risk watchlist. Read by both the bye planner
(planner.py) and the weekly trade advisor (trade_advisor.py) so they
agree on what "the team" is.

YAML schema (data/inputs/round_{N}.yaml):

    round: 9
    squad: [<26 names>]
    starters: [<confirmed starters this round>]
    # Optional fields (defaults applied when missing):
    trades_remaining: 30          # default: TOTAL_TRADES - (round - 1) * 2
    trade_boosts_remaining: 5     # default: TRADE_BOOSTS
    salary_cap: 11950000          # default: optimizer.SALARY_CAP

Origin watchlist (data/origin_watchlist.yaml):

    origin_rounds: [14, 17, 19]
    likely_origin:
      "Cleary, Nathan": NSW
      "Munster, Cameron": QLD
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)

ORIGIN_WATCHLIST_PATH = Path("data") / "origin_watchlist.yaml"


def _round_yaml_path(round_num: int) -> Path:
    from paths import DATA_INPUTS
    return DATA_INPUTS / f"round_{round_num}.yaml"


def load_state(round_num: int, df_pred: pd.DataFrame,
               default_trades_total: int = 46,
               default_boosts: int = 5,
               default_cap: int = 11_950_000) -> dict:
    """
    Load the round YAML and hydrate squad players from df_pred.

    Returns dict with keys:
        current_round, squad (list of dicts), confirmed_starters (set),
        trades_remaining, trade_boosts_remaining, salary_cap.

    Names are matched via trade_advisor.match_player so existing fuzzy
    matching (initial+surname, comma-flip, apostrophe variants) keeps
    working without duplication.
    """
    from trade_advisor import match_player

    path = _round_yaml_path(round_num)
    if not path.exists():
        raise FileNotFoundError(
            f"No round YAML at {path}. Create it (squad + starters) before "
            f"loading state."
        )

    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_squad = doc.get("squad") or []
    raw_starters = doc.get("starters") or []

    if not raw_squad:
        raise ValueError(f"{path} has no 'squad' list")

    # Hydrate squad
    squad = []
    unmatched = []
    for name in raw_squad:
        matched = match_player(name, df_pred)
        if matched is None:
            unmatched.append(name)
            continue
        rows = df_pred[df_pred["player_name"] == matched]
        if rows.empty:
            unmatched.append(name)
            continue
        squad.append(rows.iloc[0].to_dict())

    if unmatched:
        raise ValueError(
            f"Unknown player(s) in {path}: {unmatched}. "
            f"Fix the spelling or update your squad list."
        )

    # Hydrate starters (don't error on misses — starters often have typos
    # from copy-pasting team lists; we just drop the unknowns with a warning)
    starter_names = set()
    for name in raw_starters:
        matched = match_player(name, df_pred)
        if matched is not None:
            starter_names.add(matched)

    # Defaulted fields
    trades_remaining = int(doc.get(
        "trades_remaining",
        max(0, default_trades_total - max(0, round_num - 1) * 2),
    ))
    boosts_remaining = int(doc.get("trade_boosts_remaining", default_boosts))
    salary_cap = int(doc.get("salary_cap", default_cap))

    return {
        "current_round": int(doc.get("round", round_num)),
        "squad": squad,
        "confirmed_starters": starter_names,
        "trades_remaining": trades_remaining,
        "trade_boosts_remaining": boosts_remaining,
        "salary_cap": salary_cap,
    }


# ── Origin watchlist ────────────────────────────────────────────────────────

def load_origin_watchlist(path: Path | str | None = None) -> dict:
    """
    Load the Origin watchlist. Returns:
        { "origin_rounds": set[int], "likely_origin": dict[str, str] }

    Returns empty defaults if the file is missing — Origin awareness is
    purely a flag, never blocking, so absence shouldn't fail the run.
    """
    p = Path(path) if path else ORIGIN_WATCHLIST_PATH
    if not p.exists():
        return {"origin_rounds": set(), "likely_origin": {}}

    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rounds = set(int(r) for r in (doc.get("origin_rounds") or []))
    likely = dict(doc.get("likely_origin") or {})
    return {"origin_rounds": rounds, "likely_origin": likely}


def origin_risk(player_name: str, rnd: int, watchlist: dict) -> str | None:
    """
    Return the Origin state ("NSW" / "QLD" / etc.) if this player is on
    the watchlist AND `rnd` is an Origin round. Otherwise None.

    Resolution is name-based with a single fuzzy fallback so a watchlist
    written as "Cleary, Nathan" still matches an entry stored as
    "Nathan Cleary" (and vice-versa).
    """
    if rnd not in watchlist.get("origin_rounds", set()):
        return None

    likely = watchlist.get("likely_origin", {})
    if player_name in likely:
        return likely[player_name]

    # Try comma-flip match
    if "," in player_name:
        last, first = (s.strip() for s in player_name.split(",", 1))
        flipped = f"{first} {last}"
    else:
        parts = player_name.rsplit(" ", 1)
        flipped = f"{parts[-1]}, {parts[0]}" if len(parts) == 2 else player_name

    return likely.get(flipped)

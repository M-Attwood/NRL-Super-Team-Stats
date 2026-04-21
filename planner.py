"""
planner.py — Season-long bye round planner with trade scheduling.

Simulates all 27 rounds of the NRL Supercoach season, selecting the
optimal scoring 18 each round from a 26-player squad and scheduling
trades to cover bye weeks.

Usage:
    python main.py --no-scrape --round 1 --plan
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import pulp

from optimizer import (
    select_team, INJURED_PLAYERS, SALARY_CAP, SQUAD_SIZE,
    POSITION_QUOTAS, ACTIVE_QUOTAS, STARTING_SLOTS, BENCH_SLOTS,
    _eligible_positions,
)

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")

# ── Bye round schedule (2026 season) ─────────────────────────────────────────
BYE_ROUNDS: dict[str, list[int]] = {
    "BRO": [12, 16, 19],
    "CBR": [11, 18, 26],
    "BUL": [2, 15, 18],
    "SHA": [7, 12, 17],
    "GCT": [8, 13, 18],
    "MNL": [3, 15, 22],
    "MEL": [15, 18, 24],
    "NEW": [12, 15, 27],
    "NZL": [10, 14, 18],
    "NQC": [15, 18, 25],
    "PAR": [12, 16, 20],
    "PTH": [12, 15, 19],
    "STH": [4, 13, 16],
    "STG": [9, 15, 19],
    "SYD": [5, 12, 18],
    "DOL": [6, 13, 21],
    "WST": [1, 12, 23],
}

TOTAL_ROUNDS = 27
TOTAL_TRADES = 46
SCORING_SLOTS = 18
TRADE_BOOSTS = 5

BIG_BYE_ROUNDS = {12, 15, 18}  # 7 teams on bye → 3 base trades
BASE_TRADES_NORMAL = 2
BASE_TRADES_BIG_BYE = 3

# Trade reservation: keep at least this many trades available for critical rounds
TRADE_RESERVATIONS = {12: 3, 15: 3, 18: 3}  # 9 reserved total

# ── Derived lookups ──────────────────────────────────────────────────────────
ROUND_BYES: dict[int, set[str]] = {}
for _team, _rounds in BYE_ROUNDS.items():
    for _r in _rounds:
        ROUND_BYES.setdefault(_r, set()).add(_team)


def base_trades_for_round(rnd: int) -> int:
    """Base trade limit for a round (before boosts)."""
    return BASE_TRADES_BIG_BYE if rnd in BIG_BYE_ROUNDS else BASE_TRADES_NORMAL


# ── Availability helpers ─────────────────────────────────────────────────────

def is_on_bye(team: str, rnd: int) -> bool:
    """Check if a team has a bye in this round."""
    return team in ROUND_BYES.get(rnd, set())


def is_injured(player_name: str, rnd: int) -> bool:
    """Check if a player is injured/suspended for this round."""
    for name, avail_round, _ in INJURED_PLAYERS:
        if name == player_name and rnd < avail_round:
            return True
    return False


def is_available(player_name: str, team: str, rnd: int) -> bool:
    """Return True if a player can score in this round."""
    return not is_on_bye(team, rnd) and not is_injured(player_name, rnd)


def rounds_available(player_name: str, team: str,
                     start_round: int = 1) -> int:
    """Count how many rounds a player is available from start_round to end."""
    return sum(
        1 for r in range(start_round, TOTAL_ROUNDS + 1)
        if is_available(player_name, team, r)
    )


# ── Season-adjusted initial squad ───────────────────────────────────────────

def compute_availability_adjusted_points(df: pd.DataFrame,
                                          start_round: int = 1) -> pd.Series:
    """
    Scale each player's predicted_points by their availability fraction.
    Players available more rounds get higher adjusted scores.
    """
    total_remaining = TOTAL_ROUNDS - start_round + 1
    adjusted = []
    for _, row in df.iterrows():
        team = row.get("team", "")
        name = row.get("player_name", "")
        pred = row.get("predicted_points", 0)
        avail = rounds_available(name, team, start_round)
        frac = avail / max(total_remaining, 1)
        adjusted.append(pred * frac)
    return pd.Series(adjusted, index=df.index)


def select_initial_squad(df_pred: pd.DataFrame,
                          start_round: int = 1) -> dict:
    """Select initial squad using bye-adjusted predicted_points."""
    df = df_pred.copy()
    df["predicted_points"] = compute_availability_adjusted_points(
        df, start_round=start_round
    )
    return select_team(df, round_number=start_round)


# ── Per-round lineup selection (small LP) ────────────────────────────────────

def select_scoring_18(squad: list[dict], rnd: int) -> tuple[list[dict], list[dict]]:
    """
    From a 26-player squad, pick the best 18 to score this round.
    Respects ACTIVE_QUOTAS position constraints + 1 FLEX.
    Players on bye or injured are forced into reserves.

    Returns (scoring_18, reserves_8).
    """
    available = []
    forced_reserve = []

    for p in squad:
        if not is_available(p["player_name"], p.get("team", ""), rnd):
            forced_reserve.append(p)
        else:
            available.append(p)

    if len(available) <= SCORING_SLOTS:
        # Everyone available scores; rest are reserves
        return available, forced_reserve

    # Small LP: maximize predicted_points of the 18 selected from available
    n = len(available)
    prob = pulp.LpProblem(f"Lineup_R{rnd}", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("play", range(n), cat="Binary")

    prob += pulp.lpSum(
        x[i] * available[i].get("predicted_points", 0) for i in range(n)
    )

    # Exactly 18 scoring
    prob += pulp.lpSum(x[i] for i in range(n)) == SCORING_SLOTS

    # Position assignment variables
    eligible = [_eligible_positions(p.get("positions", "")) for p in available]
    pos_assign = {}
    for i in range(n):
        pos_assign[i] = {}
        for p in eligible[i]:
            pos_assign[i][p] = pulp.LpVariable(f"pa_{rnd}_{i}_{p}", cat="Binary")
        pos_assign[i]["FLEX"] = pulp.LpVariable(f"pa_{rnd}_{i}_FLEX", cat="Binary")

    # Each selected player assigned to exactly one position slot
    for i in range(n):
        prob += pulp.lpSum(pos_assign[i].values()) == x[i]

    # Active position quotas
    for pos, quota in ACTIVE_QUOTAS.items():
        eligible_for_pos = [i for i in range(n) if pos in pos_assign[i]]
        if eligible_for_pos:
            prob += pulp.lpSum(pos_assign[i][pos] for i in eligible_for_pos) == quota

    # FLEX: exactly 1
    prob += pulp.lpSum(pos_assign[i]["FLEX"] for i in range(n)) == 1

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if prob.status != pulp.constants.LpStatusOptimal:
        # Fallback: just take top 18 by predicted_points
        available.sort(key=lambda p: p.get("predicted_points", 0), reverse=True)
        scoring = available[:SCORING_SLOTS]
        reserves = available[SCORING_SLOTS:] + forced_reserve
        return scoring, reserves

    scoring = [available[i] for i in range(n) if pulp.value(x[i]) > 0.5]
    benched = [available[i] for i in range(n) if pulp.value(x[i]) <= 0.5]
    reserves = forced_reserve + benched

    return scoring, reserves


# ── Trade evaluation ─────────────────────────────────────────────────────────

def _remaining_value(player_name: str, team: str, predicted_points: float,
                     current_round: int) -> float:
    """Sum of effective points from current_round through end of season."""
    return sum(
        predicted_points if is_available(player_name, team, r) else 0.0
        for r in range(current_round, TOTAL_ROUNDS + 1)
    )


def evaluate_trade(p_out: dict, p_in: dict, current_round: int) -> float:
    """Net remaining-season value of swapping p_out for p_in."""
    out_val = _remaining_value(
        p_out["player_name"], p_out.get("team", ""),
        p_out.get("predicted_points", 0), current_round,
    )
    in_val = _remaining_value(
        p_in["player_name"], p_in.get("team", ""),
        p_in.get("predicted_points", 0), current_round,
    )
    return in_val - out_val


def _get_price(p: dict) -> float:
    """Get player price (prefer price_usd)."""
    return float(p.get("price_usd", p.get("price", 0)) or 0)


def _build_pool_candidates(squad: list[dict], pool: pd.DataFrame,
                            current_round: int) -> list[dict]:
    """Build candidate pool: players not in squad with future availability."""
    squad_names = {p["player_name"] for p in squad}
    candidates = []
    for _, row in pool.iterrows():
        name = row.get("player_name", "")
        if name in squad_names:
            continue
        team = row.get("team", "")
        if rounds_available(name, team, current_round) > 0:
            candidates.append(row.to_dict())
    return candidates


def find_best_trades(squad: list[dict], pool: pd.DataFrame,
                     current_round: int, max_trades: int,
                     trades_remaining: int) -> list[tuple[dict, dict, float]]:
    """
    Find the best PERMANENT trades to make this round.
    Returns list of (player_out, player_in, net_value) tuples.
    """
    allowed = min(max_trades, trades_remaining)
    if allowed <= 0:
        return []

    current_salary = sum(_get_price(p) for p in squad)
    pool_candidates = _build_pool_candidates(squad, pool, current_round)

    trade_options = []
    for p_out in squad:
        out_price = _get_price(p_out)
        for p_in in pool_candidates:
            in_price = _get_price(p_in)
            if current_salary - out_price + in_price > SALARY_CAP:
                continue
            net_val = evaluate_trade(p_out, p_in, current_round)
            if net_val > 0:
                trade_options.append((p_out, p_in, net_val))

    trade_options.sort(key=lambda t: t[2], reverse=True)

    selected = []
    used_out = set()
    used_in = set()
    for p_out, p_in, net_val in trade_options:
        if len(selected) >= allowed:
            break
        out_name = p_out["player_name"]
        in_name = p_in["player_name"]
        if out_name in used_out or in_name in used_in:
            continue
        used_out.add(out_name)
        used_in.add(in_name)
        selected.append((p_out, p_in, net_val))

    return selected


# ── Bye loop trades ──────────────────────────────────────────────────────────
# A "bye loop" = trade out a bye-affected player before the bye round,
# trade them back after. Costs 2 trades, gains replacement's points for 1 round.

def find_bye_loop_trades(squad: list[dict], pool: pd.DataFrame,
                          bye_round: int, max_loops: int,
                          trades_remaining: int) -> list[dict]:
    """
    Find the best bye loop trades for an upcoming bye round.
    Each loop: trade out a bye-affected scorer, trade in a non-bye replacement.
    Scheduled trade-back happens in bye_round + 1.

    Returns list of loop dicts:
        {out: player_dict, temp_in: player_dict, gain: float}
    """
    # Each loop costs 2 trades
    max_possible = min(max_loops, trades_remaining // 2)
    if max_possible <= 0:
        return []

    bye_teams = ROUND_BYES.get(bye_round, set())
    if not bye_teams:
        return []

    current_salary = sum(_get_price(p) for p in squad)
    squad_names = {p["player_name"] for p in squad}

    # Find squad members who'll be on bye (and are actually good enough to loop)
    bye_affected = [
        p for p in squad
        if p.get("team", "") in bye_teams
        and not is_injured(p["player_name"], bye_round)
        and p.get("predicted_points", 0) > 30  # only worth looping good players
    ]

    if not bye_affected:
        return []

    # Find replacements who are NOT on bye in bye_round and NOT in squad
    pool_candidates = []
    for _, row in pool.iterrows():
        name = row.get("player_name", "")
        if name in squad_names:
            continue
        team = row.get("team", "")
        if team not in bye_teams and is_available(name, team, bye_round):
            pool_candidates.append(row.to_dict())

    # Evaluate each (bye_player_out, replacement_in) loop
    loop_options = []
    for p_out in bye_affected:
        out_price = _get_price(p_out)
        for p_in in pool_candidates:
            in_price = _get_price(p_in)
            if current_salary - out_price + in_price > SALARY_CAP:
                continue
            # Gain = replacement scores in the bye round; original would score 0
            gain = p_in.get("predicted_points", 0)
            loop_options.append({
                "out": p_out,
                "temp_in": p_in,
                "gain": gain,
            })

    loop_options.sort(key=lambda x: x["gain"], reverse=True)

    # Greedily select non-conflicting loops
    selected = []
    used_out = set()
    used_in = set()
    for loop in loop_options:
        if len(selected) >= max_possible:
            break
        out_name = loop["out"]["player_name"]
        in_name = loop["temp_in"]["player_name"]
        if out_name in used_out or in_name in used_in:
            continue
        used_out.add(out_name)
        used_in.add(in_name)
        selected.append(loop)

    return selected


def validate_position_quotas(squad: list[dict]) -> bool:
    """Check that a 26-player squad can fill all position quotas + 1 FLEX."""
    n = len(squad)
    if n < SQUAD_SIZE:
        return False

    prob = pulp.LpProblem("QuotaCheck", pulp.LpMaximize)
    x = {}
    for i in range(n):
        eligible = _eligible_positions(squad[i].get("positions", ""))
        for pos in eligible:
            x[(i, pos)] = pulp.LpVariable(f"qc_{i}_{pos}", cat="Binary")
        x[(i, "FLEX")] = pulp.LpVariable(f"qc_{i}_FLEX", cat="Binary")

    # Each player fills at most 1 slot
    for i in range(n):
        slots = [v for k, v in x.items() if k[0] == i]
        if slots:
            prob += pulp.lpSum(slots) <= 1

    # Position quotas
    for pos, quota in POSITION_QUOTAS.items():
        vars_for_pos = [x[(i, pos)] for i in range(n) if (i, pos) in x]
        if vars_for_pos:
            prob += pulp.lpSum(vars_for_pos) >= quota

    # FLEX: 1
    flex_vars = [x[(i, "FLEX")] for i in range(n) if (i, "FLEX") in x]
    if flex_vars:
        prob += pulp.lpSum(flex_vars) >= 1

    prob += pulp.lpSum(x.values())

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    return prob.status == pulp.constants.LpStatusOptimal


# ── Effective trade limit with reservations ──────────────────────────────────

def get_effective_trade_limit(rnd: int, trades_used: int) -> int:
    """
    Max trades to use this round, considering reservations for future
    big bye rounds so the greedy algorithm doesn't exhaust the budget early.
    """
    base = base_trades_for_round(rnd)
    remaining = TOTAL_TRADES - trades_used

    # How many trades are reserved for future critical rounds?
    future_reserved = sum(
        budget for future_rnd, budget in TRADE_RESERVATIONS.items()
        if future_rnd > rnd
    )

    available_now = max(0, remaining - future_reserved)
    return min(base, available_now)


# ── Trade boost allocation ───────────────────────────────────────────────────

def allocate_boosts(round_results: list[dict], boosts_remaining: int) -> dict[int, int]:
    """
    After the initial simulation, determine which rounds would benefit
    most from a +1 trade boost. Returns {round: boost_count}.

    Strategy: find rounds where the next-best available trade (the one
    that was NOT executed due to the trade limit) has the highest net value.
    """
    if boosts_remaining <= 0:
        return {}

    # Collect rounds that had beneficial trades left on the table
    boost_candidates = []
    for rr in round_results:
        # The "overflow" trades that couldn't be made due to limit
        overflow_value = rr.get("_best_rejected_trade_value", 0.0)
        if overflow_value > 0:
            boost_candidates.append((rr["round"], overflow_value))

    boost_candidates.sort(key=lambda x: x[1], reverse=True)

    boosts = {}
    for rnd, val in boost_candidates[:boosts_remaining]:
        boosts[rnd] = 1
    return boosts


# ── Main season simulation ───────────────────────────────────────────────────

def run_season_plan(df_pred: pd.DataFrame,
                     start_round: int = 1) -> dict:
    """
    Simulate the full season and produce a trade plan.

    Returns dict with:
        round_results: list of per-round dicts
        trade_log: list of all trades
        total_projected: total season scoring points
        initial_squad: the starting 26
        boosts_used: {round: count}
    """
    log.info("=" * 60)
    log.info("  SEASON BYE ROUND PLANNER — Rounds %d to %d", start_round, TOTAL_ROUNDS)
    log.info("=" * 60)

    # ── Step 1: Season-adjusted initial squad ────────────────────────────────
    log.info("Selecting bye-adjusted initial squad ...")
    result = select_initial_squad(df_pred, start_round=start_round)

    if result.get("solver_status") != "Optimal":
        log.error("Failed to select initial squad: %s", result.get("solver_status"))
        return {"round_results": [], "trade_log": [], "total_projected": 0.0}

    # Flatten squad into list of dicts with original (unadjusted) predicted_points
    squad = []
    for group in ["starting_13", "bench_4", "flex_1", "reserves_8"]:
        for p in result.get(group, []):
            # Restore original predicted_points from df_pred
            name = p["player_name"]
            orig_row = df_pred[df_pred["player_name"] == name]
            if not orig_row.empty:
                p["predicted_points"] = float(orig_row.iloc[0]["predicted_points"])
            squad.append(p)

    log.info("Initial squad: %d players, $%s",
             len(squad), f"{sum(_get_price(p) for p in squad):,.0f}")

    # ── Step 2: Round-by-round simulation ────────────────────────────────────
    trades_used = 0
    total_projected = 0.0
    round_results = []
    trade_log = []
    # Scheduled trade-backs: list of (round_to_execute, original_player, temp_player)
    pending_tradebacks = []

    for rnd in range(start_round, TOTAL_ROUNDS + 1):
        bye_teams = sorted(ROUND_BYES.get(rnd, set()))
        trade_limit = get_effective_trade_limit(rnd, trades_used)
        trades_made = []

        # ── Execute scheduled trade-backs first ──────────────────────────────
        due_tradebacks = [tb for tb in pending_tradebacks if tb[0] == rnd]
        pending_tradebacks = [tb for tb in pending_tradebacks if tb[0] != rnd]

        for _, orig_player, temp_player in due_tradebacks:
            if trades_used >= TOTAL_TRADES or len(trades_made) >= trade_limit:
                log.warning("  R%d: Cannot execute trade-back (limit reached) — "
                            "%s stays", rnd, temp_player["player_name"])
                continue
            new_squad = [p for p in squad
                         if p["player_name"] != temp_player["player_name"]]
            new_squad.append(orig_player)
            if validate_position_quotas(new_squad):
                squad = new_squad
                trades_used += 1
                entry = {
                    "round": rnd,
                    "out": temp_player["player_name"],
                    "out_team": temp_player.get("team", ""),
                    "out_price": _get_price(temp_player),
                    "in": orig_player["player_name"],
                    "in_team": orig_player.get("team", ""),
                    "in_price": _get_price(orig_player),
                    "net_value": 0,  # trade-back, restoring original
                    "salary_delta": _get_price(orig_player) - _get_price(temp_player),
                    "is_tradeback": True,
                }
                trades_made.append(entry)
                log.info("  R%d TRADE-BACK: %s (%s) OUT -> %s (%s) IN",
                         rnd, temp_player["player_name"], temp_player.get("team"),
                         orig_player["player_name"], orig_player.get("team"))

        remaining_trades_this_round = trade_limit - len(trades_made)

        # ── Bye loop trades ──────────────────────────────────────────────────
        # For the current round or next round: if it's a big bye, trade out
        # bye-affected players temporarily for non-bye replacements.
        # Check current round first (immediate coverage), then next round.
        loop_target = None
        if rnd in BIG_BYE_ROUNDS and len(ROUND_BYES.get(rnd, set())) >= 3:
            loop_target = rnd
        else:
            next_rnd = rnd + 1
            if (next_rnd in BIG_BYE_ROUNDS
                    and len(ROUND_BYES.get(next_rnd, set())) >= 3):
                loop_target = next_rnd

        loop_trades_made = 0
        if (loop_target is not None
                and remaining_trades_this_round > 0
                and TOTAL_TRADES - trades_used >= 2):  # need 2 per loop
            loops = find_bye_loop_trades(
                squad=squad,
                pool=df_pred,
                bye_round=loop_target,
                max_loops=remaining_trades_this_round,
                trades_remaining=TOTAL_TRADES - trades_used,
            )
            for loop in loops:
                if trades_used >= TOTAL_TRADES or len(trades_made) >= trade_limit:
                    break
                if TOTAL_TRADES - trades_used < 2:
                    break  # need 2 for a loop (out + back)
                p_out = loop["out"]
                p_in = loop["temp_in"]
                new_squad = [p for p in squad
                             if p["player_name"] != p_out["player_name"]]
                new_squad.append(p_in)
                if validate_position_quotas(new_squad):
                    squad = new_squad
                    trades_used += 1
                    loop_trades_made += 1
                    entry = {
                        "round": rnd,
                        "out": p_out["player_name"],
                        "out_team": p_out.get("team", ""),
                        "out_price": _get_price(p_out),
                        "in": p_in["player_name"],
                        "in_team": p_in.get("team", ""),
                        "in_price": _get_price(p_in),
                        "net_value": loop["gain"],
                        "salary_delta": _get_price(p_in) - _get_price(p_out),
                        "is_bye_loop": True,
                        "bye_round": loop_target,
                    }
                    trades_made.append(entry)
                    # Schedule trade-back for the round after the bye
                    tradeback_round = min(loop_target + 1, TOTAL_ROUNDS)
                    pending_tradebacks.append((tradeback_round, p_out, p_in))
                    log.info("  R%d BYE-LOOP: %s (%s) OUT -> %s (%s) IN "
                             "[+%.1f pts in R%d, trade-back R%d]",
                             rnd, p_out["player_name"], p_out.get("team"),
                             p_in["player_name"], p_in.get("team"),
                             loop["gain"], loop_target, tradeback_round)

        remaining_trades_this_round = trade_limit - len(trades_made)

        # ── Permanent trades ─────────────────────────────────────────────────
        if remaining_trades_this_round > 0:
            all_potential = find_best_trades(
                squad=squad,
                pool=df_pred,
                current_round=rnd,
                max_trades=remaining_trades_this_round + 1,
                trades_remaining=TOTAL_TRADES - trades_used + 1,
            )
            perm_trades = all_potential[:remaining_trades_this_round]
            best_rejected_value = (
                all_potential[remaining_trades_this_round][2]
                if len(all_potential) > remaining_trades_this_round else 0.0
            )

            for p_out, p_in, net_val in perm_trades:
                new_squad = [p for p in squad
                             if p["player_name"] != p_out["player_name"]]
                new_squad.append(p_in)
                if validate_position_quotas(new_squad):
                    squad = new_squad
                    trades_used += 1
                    entry = {
                        "round": rnd,
                        "out": p_out["player_name"],
                        "out_team": p_out.get("team", ""),
                        "out_price": _get_price(p_out),
                        "in": p_in["player_name"],
                        "in_team": p_in.get("team", ""),
                        "in_price": _get_price(p_in),
                        "net_value": net_val,
                        "salary_delta": _get_price(p_in) - _get_price(p_out),
                    }
                    trades_made.append(entry)
                    log.info("  R%d TRADE: %s (%s) OUT -> %s (%s) IN  [net=%.1f]",
                             rnd, p_out["player_name"], p_out.get("team"),
                             p_in["player_name"], p_in.get("team"), net_val)
        else:
            best_rejected_value = 0.0

        trade_log.extend(trades_made)

        # ── Select scoring 18 ───────────────────────────────────────────────
        scoring_18, reserves = select_scoring_18(squad, rnd)
        round_points = sum(p.get("predicted_points", 0) for p in scoring_18)
        total_projected += round_points

        unavailable = [
            p["player_name"] for p in squad
            if not is_available(p["player_name"], p.get("team", ""), rnd)
        ]

        round_results.append({
            "round": rnd,
            "teams_on_bye": bye_teams,
            "n_bye_teams": len(bye_teams),
            "scoring_18": scoring_18,
            "reserves": reserves,
            "unavailable": unavailable,
            "n_unavailable": len(unavailable),
            "trades_made": trades_made,
            "n_trades": len(trades_made),
            "trades_remaining": TOTAL_TRADES - trades_used,
            "projected_points": round_points,
            "squad_snapshot": [p.copy() for p in squad],
            "_best_rejected_trade_value": best_rejected_value,
        })

        bye_str = ", ".join(bye_teams) if bye_teams else "none"
        log.info("  R%02d | Bye: %-25s | Unavail: %d | Score: %6.0f | "
                 "Trades: %d | Left: %d",
                 rnd, bye_str, len(unavailable), round_points,
                 len(trades_made), TOTAL_TRADES - trades_used)

    # ── Step 3: Allocate trade boosts (post-simulation) ──────────────────────
    boosts = allocate_boosts(round_results, TRADE_BOOSTS)
    if boosts:
        log.info("\nTrade boosts recommended for: %s",
                 ", ".join(f"R{r}" for r in sorted(boosts)))

    return {
        "round_results": round_results,
        "trade_log": trade_log,
        "total_projected": total_projected,
        "initial_squad": squad,
        "boosts_used": boosts,
        "trades_used": trades_used,
    }


# ── Display ──────────────────────────────────────────────────────────────────

def print_season_summary(state: dict):
    """Pretty-print the season plan to stdout."""
    round_results = state.get("round_results", [])
    trade_log = state.get("trade_log", [])
    total = state.get("total_projected", 0)
    boosts = state.get("boosts_used", {})

    if not round_results:
        print("\nNo season plan generated.")
        return

    n_rounds = len(round_results)
    avg_pts = total / n_rounds if n_rounds else 0

    points_list = [rr["projected_points"] for rr in round_results]
    worst_rnd = round_results[np.argmin(points_list)]
    best_rnd = round_results[np.argmax(points_list)]

    print("\n" + "=" * 80)
    print(f"  NRL SUPERCOACH BYE ROUND SEASON PLAN — 2026")
    print(f"  Projected Season Total: {total:,.0f} pts  |  "
          f"Avg: {avg_pts:,.0f}/round")
    print(f"  Trades Used: {state.get('trades_used', 0)} / {TOTAL_TRADES}  |  "
          f"Boosts: {len(boosts)} / {TRADE_BOOSTS}")
    print("=" * 80)

    for rr in round_results:
        bye_str = f"{rr['n_bye_teams']} teams" if rr["n_bye_teams"] > 1 \
            else (rr["teams_on_bye"][0] if rr["teams_on_bye"] else "none")

        print(f"\n  R{rr['round']:02d} | Bye: {bye_str:<12} | "
              f"Unavail: {rr['n_unavailable']} | "
              f"Score: {rr['projected_points']:6.0f} | "
              f"Trades: {rr['n_trades']} | Left: {rr['trades_remaining']}")

        for t in rr["trades_made"]:
            tag = ""
            if t.get("is_tradeback"):
                tag = " (TRADE-BACK)"
            elif t.get("is_bye_loop"):
                tag = f" (BYE-LOOP for R{t['bye_round']})"
            print(f"       OUT: {t['out']:<22} ({t['out_team']}) -> "
                  f"IN: {t['in']:<22} ({t['in_team']})  "
                  f"[net={t['net_value']:+.1f}]{tag}")

    # Boost recommendations
    if boosts:
        print(f"\n  TRADE BOOST RECOMMENDATIONS:")
        for rnd in sorted(boosts):
            rr = next(r for r in round_results if r["round"] == rnd)
            val = rr.get("_best_rejected_trade_value", 0)
            print(f"    R{rnd}: Use boost for +1 trade (marginal value: {val:.1f})")

    print(f"\n  {'-' * 60}")
    print(f"  SEASON SUMMARY")
    print(f"  Total Projected Points:  {total:>8,.0f}")
    print(f"  Average per Round:       {avg_pts:>8,.0f}")
    print(f"  Best Round:              R{best_rnd['round']:02d} ({best_rnd['projected_points']:,.0f} pts)")
    print(f"  Worst Round:             R{worst_rnd['round']:02d} ({worst_rnd['projected_points']:,.0f} pts)")
    print(f"  Total Trades:            {state.get('trades_used', 0)} / {TOTAL_TRADES}")
    print("=" * 80 + "\n")


# ── CSV export ───────────────────────────────────────────────────────────────

def export_season_plan(state: dict):
    """Export season plan to CSV files in outputs/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    round_results = state.get("round_results", [])
    trade_log = state.get("trade_log", [])

    # ── trade_plan.csv ───────────────────────────────────────────────────────
    if trade_log:
        df_trades = pd.DataFrame(trade_log)
        trade_path = OUTPUT_DIR / "trade_plan.csv"
        df_trades.to_csv(trade_path, index=False)
        log.info("Trade plan exported -> %s", trade_path)

    # ── round_summary.csv ────────────────────────────────────────────────────
    summary_rows = []
    for rr in round_results:
        summary_rows.append({
            "round": rr["round"],
            "teams_on_bye": ",".join(rr["teams_on_bye"]),
            "n_bye_teams": rr["n_bye_teams"],
            "n_unavailable": rr["n_unavailable"],
            "scoring_total": rr["projected_points"],
            "n_trades": rr["n_trades"],
            "trades_remaining": rr["trades_remaining"],
        })
    df_summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "round_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    log.info("Round summary exported -> %s", summary_path)

    # ── season_plan.csv (1 row per player per round) ─────────────────────────
    plan_rows = []
    for rr in round_results:
        rnd = rr["round"]
        scoring_names = {p["player_name"] for p in rr["scoring_18"]}

        for p in rr["squad_snapshot"]:
            name = p["player_name"]
            team = p.get("team", "")
            plan_rows.append({
                "round": rnd,
                "player_name": name,
                "team": team,
                "positions": p.get("positions", ""),
                "predicted_points": p.get("predicted_points", 0),
                "price": _get_price(p),
                "on_bye": is_on_bye(team, rnd),
                "injured": is_injured(name, rnd),
                "available": is_available(name, team, rnd),
                "in_scoring_18": name in scoring_names,
            })

    df_plan = pd.DataFrame(plan_rows)
    plan_path = OUTPUT_DIR / "season_plan.csv"
    df_plan.to_csv(plan_path, index=False)
    log.info("Season plan exported -> %s (%d rows)", plan_path, len(df_plan))

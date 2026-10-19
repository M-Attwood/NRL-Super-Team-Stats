"""
season_sim.py — replay a finished season under Supercoach rules.

This is the game the system was built to play: pick a squad before round 1,
then trade every week. The replay only uses what was known at the time.

  Pre-season   26 players at start-of-season prices under the $11.95M cap
               (the optimizer's live season objective).
  Each round   - Prices move exactly as they did that season.
               - Trades: 2 per round (3 in the big bye rounds), 46 for the
                 season, plus 5 one-off boosts (max one a round) for an extra
                 trade. A trade must
                 fit the bank (cash left + sale price) and the position quotas.
               - Trades are ranked by predicted points gained over the next 3
                 rounds, counting byes. A player not named this round is
                 treated as injured / dropped for that window.
               - The scoring 18 (13 starters, 4 bench, 1 flex) is chosen from
                 squad players named that round by predicted points; the
                 highest prediction is captain (double points).
               - Scored with the real points from that round.

How players are valued is pluggable, so the same rules replay the
heuristics for comparison.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from model import build_prediction_frame, predict_next_round
from optimizer import ACTIVE_QUOTAS, SALARY_CAP, _eligible_positions, select_team
from planner import (TOTAL_TRADES, TRADE_BOOSTS, get_effective_trade_limit,
                     validate_position_quotas)
from scraper import load_fixtures

log = logging.getLogger(__name__)

HORIZON = 3            # rounds a trade is judged over
MIN_TRADE_GAIN = 15.0  # predicted points over the horizon to justify a trade
MIN_BOOST_GAIN = 60.0  # ... and to spend a one-off boost
CANDIDATES_PER_OUT = 5


# ── Per-round inputs (shared by every strategy) ───────────────────────────────

@dataclass
class RoundData:
    frame: pd.DataFrame          # indexed by player_name: team, positions, price,
                                 # named, score, value_<strategy> columns
    teams_playing: set[str]


def price_table(rounds: pd.DataFrame, totals: dict, year: int, last_round: int) -> pd.DataFrame:
    """Player x round price as it stood during each round. Prices only move
    after games played, so gaps are forward-filled from the start price."""
    season = rounds[rounds["year"] == year]
    px = season.pivot_table(index="player_name", columns="round", values="price", aggfunc="last")
    start = totals[year].set_index("player_name")["start_price"]
    px = px.reindex(index=start.index, columns=range(1, last_round + 1))
    px.insert(0, 0, start)
    return px.ffill(axis=1).drop(columns=0)


def prepare_rounds(rounds: pd.DataFrame, totals: dict, year: int, network,
                   price_fit: tuple[float, float]) -> dict[int, RoundData]:
    """Predictions (from games before each round only), real prices, team
    lists and real scores for every round of `year`."""
    fixtures = load_fixtures(year)
    season = rounds[rounds["year"] == year]
    last_round = int(season["round"].max())
    prices = price_table(rounds, totals, year, last_round)
    out = {}
    for rnd in range(1, last_round + 1):
        played = season[season["round"] == rnd].set_index("player_name")
        frame = build_prediction_frame(rounds, totals, year, rnd, fixtures)
        frame["price"] = frame["player_name"].map(prices[rnd]).astype(float)
        frame["price_usd"] = frame["price"]
        frame = frame[frame["price"] > 0]
        frame = predict_next_round(frame, network, price_fit=price_fit)
        frame["named"] = frame["player_name"].isin(played.index)
        frame["score"] = frame["player_name"].map(played["score"])
        frame["value_model"] = frame["predicted_points"]
        frame["value_season_avg"] = np.where(frame["is_rookie"], frame["predicted_points"],
                                             frame["season_avg"])
        frame["value_price"] = frame["price"]
        cols = ["team", "positions", "price", "price_usd", "named", "score",
                "predicted_points", "season_avg", "is_rookie",
                "value_model", "value_season_avg", "value_price"]
        rnd_fix = fixtures[fixtures["round"] == rnd]
        out[rnd] = RoundData(frame.set_index("player_name")[cols],
                             set(rnd_fix["home_team"]) | set(rnd_fix["away_team"]))
        log.info("Prepared round %d: %d players, %d named", rnd, len(frame),
                 int(frame["named"].sum()))
    return out


# ── Lineups ───────────────────────────────────────────────────────────────────

def best_lineup(players: pd.DataFrame, value_col: str) -> pd.Index:
    """The legal scoring 18 that maximises `value_col`: Supercoach's active
    quotas (2 HOK, 4 FRF, 4 2RF, 1 HFB, 1 5/8, 4 CTW, 1 FLB) plus one FLEX."""
    if players.empty:
        return players.index
    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    slots = {}
    for k, (name, row) in enumerate(players.iterrows()):
        for pos in _eligible_positions(row["positions"]) + ["FLEX"]:
            slots[name, pos] = pulp.LpVariable(f"s{k}_{pos.replace('/', '_')}", cat="Binary")
    prob += pulp.lpSum(v * float(players.at[n, value_col]) for (n, _), v in slots.items())
    for name in players.index:
        prob += pulp.lpSum(v for (n, _), v in slots.items() if n == name) <= 1
    for pos, quota in {**ACTIVE_QUOTAS, "FLEX": 1}.items():
        prob += pulp.lpSum(v for (_, p), v in slots.items() if p == pos) <= quota
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    chosen = {n for (n, _), v in slots.items() if (v.value() or 0) > 0.5}
    return players.index[players.index.isin(chosen)]


# ── Trades ────────────────────────────────────────────────────────────────────

def horizon_value(data: dict[int, RoundData], rnd: int, value_col: str) -> pd.Series:
    """Predicted points each player is expected to add over the next HORIZON
    rounds, zero for rounds their team has a bye. A player whose team played
    this round but who wasn't named is assumed out for the whole window."""
    f = data[rnd].frame
    on_bye_now = ~f["team"].isin(data[rnd].teams_playing)
    available = f["named"] | on_bye_now
    total = pd.Series(0.0, index=f.index)
    for h in range(rnd, min(rnd + HORIZON, max(data) + 1)):
        plays = f["team"].isin(data[h].teams_playing)
        now = f["named"] if h == rnd else available
        total += f[value_col].clip(lower=0) * (plays & now)
    return total


def best_trade(squad: list[str], bank: float, f: pd.DataFrame, value: pd.Series):
    """Highest-gain cap-legal, quota-legal single trade, or None."""
    pool = f.drop(index=[p for p in squad if p in f.index])
    pool_pos = pool["positions"].map(lambda s: set(_eligible_positions(s)))
    options = []
    for out in squad:
        if out not in f.index:
            continue
        out_pos = set(_eligible_positions(f.at[out, "positions"]))
        budget = bank + f.at[out, "price"]
        ok = (pool["price"] <= budget) & pool_pos.map(lambda ps: bool(ps & out_pos))
        gains = (value[pool.index[ok]] - value[out]).nlargest(CANDIDATES_PER_OUT)
        options += [(g, out, inn) for inn, g in gains.items()]
    for gain, out, inn in sorted(options, reverse=True):
        new_squad = [inn if p == out else p for p in squad]
        if validate_position_quotas([{"positions": f.at[p, "positions"]} for p in new_squad]):
            return gain, out, inn
    return None


# ── Simulation ────────────────────────────────────────────────────────────────

@dataclass
class SeasonResult:
    name: str
    rounds: pd.DataFrame                       # one row per round
    trades: pd.DataFrame                       # one row per trade
    initial_squad: list[str] = field(default_factory=list)
    final_squad: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return float(self.rounds["points"].sum())


def simulate_season(data: dict[int, RoundData], value_col: str, name: str,
                    trading: bool = True) -> SeasonResult:
    # Round-1 team lists are public before lockout, so the opening squad is
    # valued like a trade: predicted points over the first HORIZON rounds,
    # with players not named for round 1 counted as out.
    first = data[1].frame.assign(
        predicted_points=horizon_value(data, 1, value_col) / HORIZON).reset_index()
    pick = select_team(first, round_number=1, apply_injuries=False)
    squad = [p["player_name"] for g in ("starting_13", "bench_4", "flex_1", "reserves_8")
             for p in pick[g]]
    bank = SALARY_CAP - pick["total_price"]
    initial = list(squad)
    trades_used, boosts_left = 0, TRADE_BOOSTS
    round_rows, trade_rows = [], []

    for rnd, rd in data.items():
        f = rd.frame
        made = 0
        if trading and rnd > 1:
            value = horizon_value(data, rnd, value_col)
            limit = get_effective_trade_limit(rnd, trades_used)
            while trades_used < TOTAL_TRADES:
                boost = made >= limit          # at most one boost per round
                if made > limit or (boost and boosts_left == 0):
                    break
                found = best_trade(squad, bank, f, value)
                if found is None or found[0] < (MIN_BOOST_GAIN if boost else MIN_TRADE_GAIN):
                    break
                gain, out, inn = found
                bank += f.at[out, "price"] - f.at[inn, "price"]
                squad = [inn if p == out else p for p in squad]
                trade_rows.append({"round": rnd, "out": out, "in": inn,
                                   "out_price": f.at[out, "price"], "in_price": f.at[inn, "price"],
                                   "predicted_gain_3rd": round(gain, 1), "boost": boost})
                made += 1
                if boost:
                    boosts_left -= 1
                else:
                    trades_used += 1

        members = f.loc[[p for p in squad if p in f.index]]
        named = members[members["named"]]
        lineup = named.loc[best_lineup(named, value_col)]
        captain = lineup[value_col].idxmax() if not lineup.empty else None
        base = float(lineup["score"].sum())
        bonus = float(lineup.at[captain, "score"]) if captain else 0.0
        round_rows.append({
            "round": rnd, "points": base + bonus, "lineup_points": base,
            "captain": captain, "captain_points": bonus, "named": len(named),
            "trades": made, "trades_left": TOTAL_TRADES - trades_used,
            "squad_value": float(members["price"].sum()), "bank": bank,
        })

    out = SeasonResult(name, pd.DataFrame(round_rows).set_index("round"),
                       pd.DataFrame(trade_rows), initial, squad)
    log.info("%-28s %6.0f pts, %d trades + %d boosts", name, out.total,
             trades_used, TRADE_BOOSTS - boosts_left)
    return out


"""
season_sim.py — replay a finished season under Supercoach rules.

This is the game the system was built to play: pick a squad before round 1,
then trade every week. The replay only uses what was known at the time.

  Pre-season   26 players at start-of-season prices under the $11.95M cap.
  Each round   - Prices move exactly as they did that season.
               - Trades: 2 per round (3 in the big bye rounds), 46 for the
                 season, plus 5 one-off boosts (max one a round). A trade must
                 fit the bank (cash left + sale price) and the position quotas.
               - The scoring 18 (13 starters, 4 bench, 1 flex) is chosen from
                 squad players named that round by predicted points; the
                 highest prediction is captain (double points).
               - Scored with the real points from that round.

How trades are chosen is set by a Strategy:

  ORIGINAL   compares players one-for-one on predicted points over the next
             3 rounds, assumes an unnamed player stays out, fixed trade bar.
  IMPROVED   values the whole squad's best 18 (plus captain) in every
             remaining round, discounted, so depth for bye rounds counts;
             per-round predictions against each upcoming opponent; the
             availability model for who will be named; cash value of the
             bank and of expected price rises; a trade bar that rises as
             trades get scarce; no buying back a player just sold.

Strategy parameters are tuned on a development season only (see
season_review.py), never on the season being reported.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from model import (build_prediction_frame, opponent_factor, player_state,
                   predict_next_round)
from optimizer import ACTIVE_QUOTAS, SALARY_CAP, _eligible_positions, select_team
from planner import (TOTAL_TRADES, TRADE_BOOSTS, get_effective_trade_limit,
                     validate_position_quotas)
from player_outlook import (AVAIL_FEATURES, attach_history, predict_availability,
                            predict_price_change, team_rounds)
from scraper import load_fixtures

log = logging.getLogger(__name__)

LOOKAHEAD = 6            # rounds predicted individually against each opponent
SEASON_PACE = TOTAL_TRADES / 26
LINEUP_SLOTS = [(pos, n) for pos, n in ACTIVE_QUOTAS.items()] + [("FLEX", 1)]


@dataclass(frozen=True)
class Strategy:
    name: str
    value: str = "model"           # "model" predictions or "season_avg" heuristic
    horizon: int = 27              # rounds ahead a trade is judged over
    discount: float = 0.9          # weight of each further round
    lineup_aware: bool = True      # value the squad's best 18, not players one-for-one
    availability: bool = True      # use the availability model for future rounds
    cash_weight: float = 0.5       # share of cash assumed turned into points later
    trade_bar: float = 20.0        # points a trade must add at the normal trade pace
    pace_scaled: bool = True       # raise the bar when trades are scarce
    cooldown: int = 3              # rounds before a sold player can be bought back
    candidates: int = 40           # trades scored in full per decision


ORIGINAL = Strategy("Original strategy", horizon=3, discount=1.0, lineup_aware=False,
                    availability=False, cash_weight=0.0, trade_bar=15.0,
                    pace_scaled=False, cooldown=0)
IMPROVED = Strategy("Improved strategy")


# ── Per-round inputs ──────────────────────────────────────────────────────────

@dataclass
class RoundData:
    frame: pd.DataFrame          # indexed by player_name
    preds: pd.DataFrame          # player x round: model prediction for upcoming rounds
    teams_playing: set[str]


@dataclass
class SeasonInputs:
    year: int
    rounds: dict[int, RoundData]
    points_per_dollar: float     # what $1 buys in points per game (price slope)


def fixtures_for(rounds: pd.DataFrame, year: int) -> pd.DataFrame:
    """The season draw: data/raw/fixtures_<year>.csv if present, otherwise
    rebuilt from who played whom (the draw is published before the season)."""
    fixtures = load_fixtures(year)
    if not fixtures.empty:
        return fixtures
    games = rounds.loc[rounds["year"] == year, ["round", "team", "vs_team"]].dropna()
    games = games[games["team"] < games["vs_team"]].drop_duplicates()
    return games.rename(columns={"team": "home_team", "vs_team": "away_team"})


def price_table(rounds: pd.DataFrame, totals: dict, year: int, last_round: int) -> pd.DataFrame:
    """Player x round price as it stood during each round. Prices only move
    after games played, so gaps are forward-filled from the start price."""
    season = rounds[rounds["year"] == year]
    px = season.pivot_table(index="player_name", columns="round", values="price", aggfunc="last")
    start = totals[year].set_index("player_name")["start_price"]
    px = px.reindex(index=start.index, columns=range(1, last_round + 1))
    px.insert(0, 0, start)
    return px.ffill(axis=1).drop(columns=0)


def _opponents(fixtures: pd.DataFrame, rnd: int) -> dict[str, str]:
    f = fixtures[fixtures["round"] == rnd]
    return {**dict(zip(f["home_team"], f["away_team"])),
            **dict(zip(f["away_team"], f["home_team"]))}


def _availability(frame: pd.DataFrame, rounds: pd.DataFrame, year: int, rnd: int,
                  clf, team_games: pd.DataFrame, debut_rate: float) -> pd.Series:
    """P(named for the team's next game after this round), from what is
    known before this round: recent games plus this round's team list."""
    history = rounds[(rounds["year"] == year) & (rounds["round"] < rnd)]
    q = pd.DataFrame({"player_name": frame.index, "year": year, "next_round": rnd,
                      "home_team": frame["team"].values})
    q = attach_history(q, player_state(history), team_games)
    named = frame["named"].to_numpy()
    plays = frame["team_plays"].to_numpy()
    q["missed_streak"] = np.where(named, 0, q["missed_streak"] + plays).clip(0, 6)
    q["named_prev"] = named.astype(float)
    next_rounds = {}
    for team, grp in team_games[team_games["year"] == year].groupby("team"):
        later = grp.loc[grp["round"] > rnd, "round"]
        next_rounds[team] = later.min() if not later.empty else rnd + 1
    q["next_round"] = frame["team"].map(next_rounds).fillna(rnd + 1).to_numpy()
    p = predict_availability(clf, q[AVAIL_FEATURES])
    no_history = q["games_played"].isna().to_numpy()
    p = np.where(no_history, np.where(named, 0.9, debut_rate), p)
    return pd.Series(p, index=frame.index)


def prepare_season(rounds: pd.DataFrame, totals: dict, year: int, predictor,
                   price_fit: tuple[float, float], outlook) -> SeasonInputs:
    """Everything the replay needs for each round of `year`, using only
    games before that round: per-round predictions against each upcoming
    opponent, availability, expected price change, real prices and results."""
    avail_clf, price_model, debut_rate = outlook
    fixtures = fixtures_for(rounds, year)
    season = rounds[rounds["year"] == year]
    last_round = int(season["round"].max())
    prices = price_table(rounds, totals, year, last_round)
    team_games = team_rounds(rounds[rounds["year"] <= year])
    playing = {r: set(fixtures.loc[fixtures["round"] == r, "home_team"])
               | set(fixtures.loc[fixtures["round"] == r, "away_team"])
               for r in range(1, last_round + 1)}
    out = {}
    for rnd in range(1, last_round + 1):
        played = season[season["round"] == rnd].set_index("player_name")
        frame = build_prediction_frame(rounds, totals, year, rnd, fixtures)
        frame["price"] = frame["player_name"].map(prices[rnd]).astype(float)
        frame["price_usd"] = frame["price"]
        frame = frame[frame["price"] > 0].reset_index(drop=True)
        frame = predict_next_round(frame, predictor, price_fit=price_fit)

        # Predictions for each of the next LOOKAHEAD rounds, against that
        # round's opponent (opponent strength measured before this round).
        preds = {rnd: frame["predicted_points"].to_numpy()}
        for h in range(rnd + 1, min(rnd + LOOKAHEAD, last_round + 1)):
            opp = frame["team"].map(_opponents(fixtures, h))
            ctx = frame.assign(opp=opp, next_round=rnd, year=year)
            future = frame.assign(opp=opp, next_round=h,
                                  opp_factor=opponent_factor(season, ctx).to_numpy())
            p = predictor.predict(future)
            preds[h] = np.where(frame["is_rookie"], frame["predicted_points"], p).clip(min=0)
        frame = frame.set_index("player_name")
        frame["named"] = frame.index.isin(played.index)
        frame["score"] = frame.index.map(played["score"])
        frame["team_plays"] = frame["team"].isin(playing[rnd])
        frame["season_avg_value"] = np.where(frame["is_rookie"], frame["predicted_points"],
                                             frame["season_avg"])
        frame["p_next"] = _availability(frame, rounds, year, rnd, avail_clf,
                                        team_games, debut_rate)
        # Within-season price model; prices don't move before round 2.
        frame["price_change_3"] = predict_price_change(price_model, frame) if rnd > 1 else 0.0
        keep = ["team", "positions", "price", "price_usd", "named", "score", "team_plays",
                "predicted_points", "season_avg_value", "is_rookie", "p_next",
                "price_change_3"]
        out[rnd] = RoundData(frame[keep], pd.DataFrame(preds, index=frame.index),
                             playing[rnd])
        log.info("Prepared %d round %d: %d players, %d named", year, rnd, len(frame),
                 int(frame["named"].sum()))
    return SeasonInputs(year, out, points_per_dollar=price_fit[0])


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


def greedy_lineup_value(values: np.ndarray, eligible: list[list[str]]) -> float:
    """Fast approximation of a round's best 18 plus captain: fill position
    slots highest value first (how Supercoach's own emergencies work)."""
    open_slots = dict(LINEUP_SLOTS)
    total, best = 0.0, 0.0
    for i in np.argsort(-values):
        v = values[i]
        if v <= 0:
            break
        for pos in eligible[i] + ["FLEX"]:
            if open_slots.get(pos, 0) > 0:
                open_slots[pos] -= 1
                total += v
                best = max(best, v)
                break
    return total + best          # captain doubles the top scorer


# ── Valuing players and squads ────────────────────────────────────────────────

@dataclass
class Outlook:
    """Expected points for every player in every round of the horizon."""
    expected: pd.DataFrame       # player x round
    weights: np.ndarray          # discount per round
    cash_rate: float             # points per $ of cash, over the rest of the season


def outlook_for(inputs: SeasonInputs, rnd: int, strat: Strategy) -> Outlook:
    data = inputs.rounds
    last = max(data)
    rounds_ahead = list(range(rnd, min(rnd + strat.horizon, last + 1)))
    f = data[rnd].frame
    if strat.value == "model":
        base = data[rnd].preds
        fill = base.mean(axis=1)
        values = pd.DataFrame({h: base[h] if h in base else fill for h in rounds_ahead})
    else:
        values = pd.DataFrame({h: f["season_avg_value"] for h in rounds_ahead})
    still_available = f["named"] | ~f["team_plays"]
    expected = {}
    for h in rounds_ahead:
        plays = f["team"].isin(data[h].teams_playing)
        if h == rnd:
            avail = f["named"].astype(float)
        elif strat.availability:
            avail = f["p_next"]
        else:
            avail = still_available.astype(float)
        expected[h] = values[h].clip(lower=0) * plays * avail
    weights = strat.discount ** np.arange(len(rounds_ahead))
    later = strat.discount ** np.arange(1, last - rnd + 1)
    cash_rate = strat.cash_weight * inputs.points_per_dollar * float(later.sum())
    return Outlook(pd.DataFrame(expected), weights, cash_rate)


def squad_value(squad: list[str], view: Outlook, eligible: dict) -> float:
    e = view.expected.loc[squad].to_numpy()
    elig = [eligible[p] for p in squad]
    return float(sum(w * greedy_lineup_value(e[:, j], elig)
                     for j, w in enumerate(view.weights)))


# ── Trades ────────────────────────────────────────────────────────────────────

def candidate_trades(squad: list[str], bank: float, f: pd.DataFrame, view: Outlook,
                     eligible: dict, banned_in: set, locked_out: set, n: int):
    """Shortlist of cap-legal, position-compatible trades by one-for-one value."""
    solo = pd.Series(view.expected.to_numpy() @ view.weights, index=view.expected.index)
    solo += view.cash_rate * f["price_change_3"]
    pool = f.index.difference(squad).difference(list(banned_in))
    pool_price = f.loc[pool, "price"]
    pool_pos = pd.Series([set(eligible[p]) for p in pool], index=pool)
    options = []
    for out in squad:
        if out in locked_out:
            continue
        out_pos = set(eligible[out])
        out_price = f.at[out, "price"]
        ok = pool[(pool_price <= bank + out_price).to_numpy()
                  & pool_pos.map(lambda ps: bool(ps & out_pos)).to_numpy()]
        cash = view.cash_rate * (out_price - pool_price[ok])
        gains = (solo[ok] - solo[out] + cash).nlargest(6)
        options += [(g, out, inn) for inn, g in gains.items()]
    return sorted(options, reverse=True)[:n]


def best_trade(squad, bank, f, view, eligible, strat, banned_in, locked_out):
    """Highest-value legal trade as (gain, out, in), or None."""
    shortlist = candidate_trades(squad, bank, f, view, eligible, banned_in, locked_out,
                                 strat.candidates)
    if strat.lineup_aware:
        base = squad_value(squad, view, eligible)
        scored = []
        for _, out, inn in shortlist:
            new = [inn if p == out else p for p in squad]
            cash = view.cash_rate * (f.at[out, "price"] - f.at[inn, "price"]
                                     + f.at[inn, "price_change_3"]
                                     - f.at[out, "price_change_3"])
            scored.append((squad_value(new, view, eligible) - base + cash, out, inn))
        shortlist = sorted(scored, reverse=True)
    for gain, out, inn in shortlist:
        new = [inn if p == out else p for p in squad]
        if validate_position_quotas([{"positions": f.at[p, "positions"]} for p in new]):
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


def initial_squad(inputs: SeasonInputs, strat: Strategy) -> tuple[list[str], float]:
    """Pre-season pick with the optimizer's season objective. Players are
    valued by expected points per round over the strategy's horizon."""
    view = outlook_for(inputs, 1, strat)
    per_round = (view.expected.to_numpy() @ view.weights) / view.weights.sum()
    first = inputs.rounds[1].frame.assign(
        predicted_points=pd.Series(per_round, index=view.expected.index)).reset_index()
    pick = select_team(first, round_number=1, apply_injuries=False)
    squad = [p["player_name"] for g in ("starting_13", "bench_4", "flex_1", "reserves_8")
             for p in pick[g]]
    return squad, SALARY_CAP - pick["total_price"]


def simulate_season(inputs: SeasonInputs, strat: Strategy, trading: bool = True,
                    label: str | None = None) -> SeasonResult:
    data = inputs.rounds
    squad, bank = initial_squad(inputs, strat)
    eligible = {p: _eligible_positions(pos) for rd in data.values()
                for p, pos in rd.frame["positions"].items()}
    initial = list(squad)
    trades_used, boosts_left = 0, TRADE_BOOSTS
    sold_in, bought_in = {}, {}
    round_rows, trade_rows = [], []
    value_col = "predicted_points" if strat.value == "model" else "season_avg_value"

    for rnd, rd in data.items():
        f = rd.frame
        made = 0
        if trading and rnd > 1:
            view = outlook_for(inputs, rnd, strat)
            limit = get_effective_trade_limit(rnd, trades_used)
            pace = (TOTAL_TRADES - trades_used) / (max(data) - rnd + 1)
            bar = strat.trade_bar * (min(3.0, SEASON_PACE / max(pace, 1e-6))
                                     if strat.pace_scaled else 1.0)
            banned_in = {p for p, r in sold_in.items() if rnd - r < strat.cooldown}
            locked_out = {p for p, r in bought_in.items()
                          if rnd - r < strat.cooldown and f.at[p, "named"]}
            while trades_used < TOTAL_TRADES:
                boost = made >= limit                    # at most one boost per round
                if made > limit or (boost and boosts_left == 0):
                    break
                found = best_trade(squad, bank, f, view, eligible, strat,
                                   banned_in, locked_out)
                if found is None or found[0] < bar * (2 if boost else 1):
                    break
                gain, out, inn = found
                bank += f.at[out, "price"] - f.at[inn, "price"]
                squad = [inn if p == out else p for p in squad]
                sold_in[out], bought_in[inn] = rnd, rnd
                banned_in.add(out)
                trade_rows.append({"round": rnd, "out": out, "in": inn,
                                   "out_price": f.at[out, "price"], "in_price": f.at[inn, "price"],
                                   "predicted_gain": round(gain, 1), "boost": boost})
                made += 1
                if boost:
                    boosts_left -= 1
                else:
                    trades_used += 1

        members = f.loc[squad]
        named = members[members["named"]]
        lineup = named.loc[best_lineup(named, value_col)]
        captain = lineup[value_col].idxmax() if not lineup.empty else None
        base = float(lineup["score"].sum())
        bonus = float(lineup.at[captain, "score"]) if captain else 0.0
        round_rows.append({
            "round": rnd, "points": base + bonus, "lineup_points": base,
            "captain": captain, "captain_points": bonus, "named": len(named),
            "empty_slots": 18 - len(lineup), "trades": made,
            "trades_left": TOTAL_TRADES - trades_used,
            "squad_value": float(members["price"].sum()), "bank": bank,
        })

    name = label or (strat.name if trading else f"{strat.name}, no trades")
    out = SeasonResult(name, pd.DataFrame(round_rows).set_index("round"),
                       pd.DataFrame(trade_rows), initial, squad)
    log.info("%d %-40s %6.0f pts, %d trades + %d boosts, %d empty slots", inputs.year,
             name, out.total, trades_used, TRADE_BOOSTS - boosts_left,
             int(out.rounds["empty_slots"].sum()))
    return out

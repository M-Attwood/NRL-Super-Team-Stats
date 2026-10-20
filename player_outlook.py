"""
player_outlook.py — the two forecasts a coach needs besides next week's score.

  availability   Will the player be named for their team's next game? Learned
                 from every round a player's team played after their debut
                 that season: named or not, given recent games, minutes,
                 starting role, how many games in a row they have missed,
                 and State of Origin: is it an Origin week, and has the
                 player been missing Origin-week club games this season or
                 last (the trace Origin selection leaves in club data).
                 Picks in the last two Origin series (data/inputs/
                 origin_squads.yaml) are also computed but switched off
                 (USE_ORIGIN_SQUADS): they sharpened the predictions but
                 lowered season totals on 2024-25.
  price          Supercoach's own price rule, reverse-engineered from every
                 price move in the data (PriceRule): a price follows the
                 player's last three scores, and the update is zero-sum
                 across the players who played. With predicted scores it
                 projects each player's price path over the coming rounds.
                 A gradient-boosted model of the 3-game price change, which
                 the rule replaced, is kept for comparison.

The trees are trained only on seasons before the one being predicted, like
the score model; the price rule's two constants are read from each
season's own price moves as they happen.

Every per-round row's price is the price *after* that round's game.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

from model import player_state
from paths import DATA_INPUTS

log = logging.getLogger(__name__)

MAX_STREAK = 6
PRICE_HORIZON = 3      # games
ORIGIN_BYES = 5        # teams on bye in a State of Origin week (other rounds: 0-1)

AVAIL_FEATURES = ["named_prev", "missed_streak", "games_played", "season_avg",
                  "last3_avg", "mins_last1", "mins_avg", "last_started", "last_benched",
                  "start_share", "price", "next_round",
                  "origin_week", "origin_missed", "origin_missed_last_season"]
USE_ORIGIN_SQUADS = False   # tested on 2024-25: better log-loss, -353 points a season
if USE_ORIGIN_SQUADS:
    AVAIL_FEATURES.append("origin_squads_last2")
PRICE_FEATURES = ["price", "last1", "last3_avg", "season_avg", "season_std",
                  "break_even", "games_played", "mins_avg", "season_price_change"]


# ── Availability ──────────────────────────────────────────────────────────────

def team_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    """(year, team, round) for every game a team played, with the team's
    running game index (used to count rounds a player has missed)."""
    tr = rounds[["year", "team", "round"]].drop_duplicates().sort_values(["year", "team", "round"])
    tr["team_game"] = tr.groupby(["year", "team"]).cumcount()
    return tr.reset_index(drop=True)


def availability_examples(rounds: pd.DataFrame) -> pd.DataFrame:
    """One row per player per team game after the player's season debut:
    what was known before that round, and whether they were named."""
    state = player_state(rounds)
    tr = team_rounds(rounds)
    home = (rounds.groupby(["player_name", "year"])["team"]
            .agg(lambda s: s.mode().iloc[0]).rename("home_team").reset_index())
    debut = rounds.groupby(["player_name", "year"])["round"].min().rename("debut").reset_index()
    grid = home.merge(debut, on=["player_name", "year"]).merge(
        tr.rename(columns={"team": "home_team"}), on=["year", "home_team"])
    grid = grid[grid["round"] > grid["debut"]]
    named = set(zip(rounds["player_name"], rounds["year"], rounds["round"]))
    grid["named"] = [(p, y, r) in named for p, y, r in
                     zip(grid["player_name"], grid["year"], grid["round"])]
    ex = attach_history(grid.rename(columns={"round": "next_round"}), state, tr)
    weeks = origin_weeks(tr)
    return add_origin(ex, origin_misses(rounds, tr, weeks), weeks)


def origin_weeks(tr: pd.DataFrame) -> pd.Series:
    """State of Origin weeks, read from the draw: 2 for a round with at
    least ORIGIN_BYES teams on bye (the weekend cleared for Origin), 1 for
    the round after (Origin players are often rested), otherwise 0.
    Indexed by (year, round)."""
    playing = tr.groupby(["year", "round"])["team"].nunique()
    teams = tr.groupby("year")["team"].nunique()
    byes = teams.reindex(playing.index.get_level_values("year")).to_numpy() - playing.to_numpy()
    origin = set(playing.index[byes >= ORIGIN_BYES])
    after = {(y, r + 1) for y, r in origin}
    return pd.Series([2 if k in origin else 1 if k in after else 0 for k in playing.index],
                     index=playing.index)


def origin_misses(rounds: pd.DataFrame, tr: pd.DataFrame, weeks: pd.Series) -> pd.DataFrame:
    """Origin-week club games a player sat out straight after playing their
    team's previous game, as (player_name, year, round). Each is known once
    that round's team lists are out."""
    games = rounds[["player_name", "year", "round", "team"]].merge(
        tr, on=["year", "team", "round"])
    following = tr.assign(team_game=tr["team_game"] - 1).rename(columns={"round": "next"})
    games = games.merge(following, on=["year", "team", "team_game"])
    named = set(zip(rounds["player_name"], rounds["year"], rounds["round"]))
    sat_out = np.array([(p, y, r) not in named for p, y, r in
                        zip(games["player_name"], games["year"], games["next"])], dtype=bool)
    week = weeks.reindex(pd.MultiIndex.from_arrays([games["year"], games["next"]])).to_numpy()
    missed = games[sat_out & (week > 0)]
    return (missed[["player_name", "year", "next"]].rename(columns={"next": "round"})
            .reset_index(drop=True))


@lru_cache(maxsize=1)
def origin_squads() -> dict[int, frozenset]:
    """Players in each season's State of Origin squads (both states)."""
    path = DATA_INPUTS / "origin_squads.yaml"
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {int(y): frozenset(n for names in states.values() for n in names)
            for y, states in doc.items()}


def add_origin(q: pd.DataFrame, misses: pd.DataFrame, weeks: pd.Series) -> pd.DataFrame:
    """Origin features for (player_name, year, next_round) queries: the
    week type of next_round, Origin-week games missed earlier this season,
    and in all of last season (missing when that season isn't in the data),
    plus how many of the last two Origin series the player was picked for."""
    q = q.copy()
    q["origin_week"] = weeks.reindex(pd.MultiIndex.from_arrays(
        [q["year"], q["next_round"]])).fillna(0).to_numpy()
    when = {k: np.sort(g.to_numpy()) for k, g in
            misses.groupby(["player_name", "year"])["round"]}
    q["origin_missed"] = [np.searchsorted(when.get((p, y), []), r) for p, y, r in
                          zip(q["player_name"], q["year"], q["next_round"])]
    seasons = set(weeks.index.get_level_values("year"))
    q["origin_missed_last_season"] = [
        len(when.get((p, y - 1), [])) if y - 1 in seasons else np.nan
        for p, y in zip(q["player_name"], q["year"])]
    # Picked in the previous two Origin series: known before the season starts.
    squads = origin_squads()
    q["origin_squads_last2"] = [sum(p in squads.get(y - k, ()) for k in (1, 2))
                                for p, y in zip(q["player_name"], q["year"])]
    return q


def attach_history(queries: pd.DataFrame, state: pd.DataFrame,
                   tr: pd.DataFrame) -> pd.DataFrame:
    """Join each (player, year, next_round, home_team) query to the player's
    state after their last game before next_round, and count the team games
    they have missed since."""
    cols = ["player_name", "year", "round", "games_played", "season_avg", "last3_avg",
            "mins_last1", "mins_avg", "last_started", "last_benched", "start_share",
            "price", "last1", "season_std", "break_even", "season_price_change"]
    s = state[cols].rename(columns={"round": "last_round"}).sort_values("last_round")
    q = queries.copy()
    q["_order"] = np.arange(len(q))
    q = pd.merge_asof(q.sort_values("next_round"), s, left_on="next_round",
                      right_on="last_round", by=["player_name", "year"],
                      allow_exact_matches=False)
    idx = tr.set_index(["year", "team", "round"])["team_game"]
    def team_game(teams, rnds):
        return pd.Series([idx.get((y, t, r), np.nan) for y, t, r in
                          zip(q["year"], teams, rnds)], index=q.index)
    # Team games strictly between their last game and the queried round.
    last_tg = team_game(q["home_team"], q["last_round"])
    next_tg = team_game(q["home_team"], q["next_round"])
    q["missed_streak"] = (next_tg - last_tg - 1).clip(0, MAX_STREAK).fillna(MAX_STREAK)
    q["named_prev"] = (q["missed_streak"] == 0).astype(float)
    q = q.sort_values("_order").drop(columns="_order")
    q.index = queries.index
    return q


def _avail_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    # An unknown last season stays missing: the trees route it themselves.
    return frame[AVAIL_FEATURES].fillna(
        {f: 0 for f in AVAIL_FEATURES if f != "origin_missed_last_season"})


def train_availability(examples: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingClassifier
    data = examples.dropna(subset=["games_played"])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                         random_state=0)
    return clf.fit(_avail_matrix(data), data["named"].astype(int))


def predict_availability(clf, frame: pd.DataFrame) -> np.ndarray:
    return clf.predict_proba(_avail_matrix(frame))[:, 1]


# ── Price rule ────────────────────────────────────────────────────────────────

PRICE_GAMES = 3        # a price follows the player's last three scores


@dataclass(frozen=True)
class PriceRule:
    """Supercoach's price update, reverse-engineered from the per-round data:

        new price = max(floor, mu * (old price + K * sum of the last 3 scores))

    rounded to $100, after each game from a player's third of the season on
    (the first two games leave the price alone). K is fixed for a season.
    mu is set each round so that the total value of the players whose price
    moved stays the same: prices are zero-sum, and a player rises when his
    last three scores are high for his price compared with everyone else's.
    The floor is the season's lowest price."""
    K: float
    mu: float
    floor: float

    def next_price(self, price, last3_sum):
        return np.maximum(self.floor, np.round(self.mu * (price + self.K * last3_sum), -2))

    def break_even(self, price, last2_sum):
        """Score in the next game that would leave the price unchanged."""
        return price * (1 / self.mu - 1) / self.K - last2_sum


def price_moves(rounds: pd.DataFrame) -> pd.DataFrame:
    """Every game that can move a price (a player's third game of the season
    or later): the price before and after it, and the last three scores."""
    r = rounds.sort_values(["player_name", "year", "round"])
    g = r.groupby(["player_name", "year"], sort=False)
    out = r[["player_name", "year", "round", "price"]].assign(
        game=g.cumcount() + 1, price_before=g["price"].shift(1),
        last3_sum=g["score"].transform(lambda s: s.rolling(PRICE_GAMES).sum()))
    return out[out["game"] >= PRICE_GAMES].dropna(subset=["price_before", "last3_sum"])


def fit_price_rule(rounds: pd.DataFrame, totals: dict) -> pd.DataFrame:
    """K and mu for every round with price moves, by least squares on
    new price = mu K sum3 + mu price_before. Moves held at the floor are left
    out, and so are the few more than $1,000 off the first fit (a corrected
    score, a missing game). Indexed by (year, round)."""
    moves = price_moves(rounds)
    floors = {y: price_floor(t) for y, t in totals.items()}
    moves = moves[moves["price"] > moves["year"].map(floors)]
    rows = {}
    for (year, rnd), m in moves.groupby(["year", "round"]):
        X = np.column_stack([m["last3_sum"], m["price_before"]])
        y = m["price"].to_numpy()
        keep = np.ones(len(m), bool)
        for _ in range(2):
            if keep.sum() < 20:
                break
            (a, mu), *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            keep = np.abs(X @ (a, mu) - y) <= 1000
        else:
            rows[year, rnd] = {"K": a / mu, "mu": mu, "moves": int(keep.sum())}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["year", "round"]
    return table


def price_floor(season_totals: pd.DataFrame) -> float:
    """The season's minimum price (what untried players start at)."""
    start = pd.to_numeric(season_totals["start_price"], errors="coerce")
    return float(start[start > 0].min())


def rule_before(table: pd.DataFrame, year: int, rnd: int, floor: float) -> PriceRule | None:
    """The price rule as known before round `rnd` of `year`: K from this
    season's price moves so far and the latest round's mu, or, before any
    price has moved, last season's K and its first mu (None without one)."""
    years, rounds = (table.index.get_level_values(n) for n in ("year", "round"))
    now = table[(years == year) & (rounds < rnd)]
    if not now.empty:
        return PriceRule(float(now["K"].median()), float(now["mu"].iloc[-1]), floor)
    if not (years < year).any():
        return None
    last = table[years == years[years < year].max()]
    return PriceRule(float(last["K"].median()), float(last["mu"].iloc[0]), floor)


def project_prices(rule: PriceRule, price, last2, last1, games, scores, plays) -> np.ndarray:
    """Expected price path under the rule: price going into each coming round.

    price, last2, last1, games: per player, the current price, the last two
    real scores this season (0 if fewer) and games played this season.
    scores, plays: players x coming rounds, the predicted score in each and
    the chance of playing it (0 for a bye or when not named). A game moves
    the price by the rule; the path adds that move times the chance it
    happens. Returns players x (rounds + 1), starting at the current price."""
    P = np.asarray(price, float).copy()
    s2, s1 = np.asarray(last2, float).copy(), np.asarray(last1, float).copy()
    k = np.asarray(games, float).copy()
    expected = [P.copy()]
    level = P.copy()
    for j in range(scores.shape[1]):
        s, p = scores[:, j], plays[:, j]
        plays_now = p > 0
        moved = plays_now & (k + 1 >= PRICE_GAMES)
        new = np.where(moved, rule.next_price(P, s2 + s1 + s), P)
        level = level + p * (new - P)
        expected.append(level.copy())
        P = np.where(plays_now, new, P)
        s2, s1 = np.where(plays_now, s1, s2), np.where(plays_now, s, s1)
        k = k + plays_now
    return np.column_stack(expected)


def rule_reproduction(rounds: pd.DataFrame, totals: dict, year: int,
                      table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every price move of `year` with the rule's price after it, from the
    K and mu read from that round's own moves ("rule_price"), and from the
    rule as known before the round ("rule_ahead", NaN without one)."""
    table = fit_price_rule(rounds, totals) if table is None else table
    m = price_moves(rounds)
    m = m[m["year"] == year].sort_values("round", kind="stable").copy()
    floor = price_floor(totals[year])
    own = table.loc[year].reindex(m["round"])
    m["rule_price"] = np.maximum(floor, np.round(own["mu"].to_numpy() * (
        m["price_before"].to_numpy() + own["K"].to_numpy() * m["last3_sum"].to_numpy()), -2))
    m["rule_ahead"] = np.nan
    for rnd, idx in m.groupby("round").groups.items():
        rule = rule_before(table, year, rnd, floor)
        if rule is not None:
            m.loc[idx, "rule_ahead"] = rule.next_price(m.loc[idx, "price_before"],
                                                       m.loc[idx, "last3_sum"])
    return m


def evaluate_price_rule(rounds: pd.DataFrame, totals: dict, years) -> dict:
    """How exactly the rule reproduces real price moves, per season.
    explained: with the K and mu read from that round's own moves.
    one round ahead: with the rule as known before the round, given the
    real score (rounds with no earlier season to read the rule from are
    skipped)."""
    table = fit_price_rule(rounds, totals)
    out = {}
    for year in years:
        m = rule_reproduction(rounds, totals, year, table)
        a = m.dropna(subset=["rule_ahead"])
        change = a["price"] - a["price_before"]
        err = np.abs(m["rule_price"] - m["price"])
        ahead = np.abs(a["rule_ahead"] - a["price"])
        out[year] = {"moves": int(len(m)), "K": round(float(table.loc[year, "K"].median()), 1),
                     "within_100_explained": float(np.mean(err <= 100)),
                     "mae_explained": float(err.mean()),
                     "within_1000_one_round_ahead": float(np.mean(ahead <= 1000)),
                     "mae_one_round_ahead": float(ahead.mean()),
                     "r2_change_one_round_ahead": float(
                         1 - np.sum(ahead ** 2) / np.sum((change - change.mean()) ** 2))}
    return out


# ── Price change: gradient-boosted model (replaced by the rule) ────────────────

def price_examples(rounds: pd.DataFrame) -> pd.DataFrame:
    """State after game k (its price is the price after game k, the one a
    coach trades at before game k+1) and the price change over the next
    PRICE_HORIZON games, with the last two scores for the price rule."""
    state = player_state(rounds)
    g = state.groupby(["player_name", "year"], sort=False)
    state["price_change"] = g["price"].shift(-PRICE_HORIZON) - state["price"]
    state["last2"] = g["score"].shift(1).fillna(0)
    for j in range(1, PRICE_HORIZON + 1):
        state[f"score_next{j}"] = g["score"].shift(-j)
    return state.dropna(subset=["price_change"]).copy()


def train_price_model(examples: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         random_state=0).fit(
        examples[PRICE_FEATURES].fillna(0), examples["price_change"])


def predict_price_change(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(frame[PRICE_FEATURES].fillna(0))


def rule_price_change(examples: pd.DataFrame, table: pd.DataFrame, floor: float,
                      scores: np.ndarray) -> np.ndarray:
    """Price change over the next PRICE_HORIZON games projected by the price
    rule as known after game k, given a score for each of those games."""
    out = np.full(len(examples), np.nan)
    for (year, rnd), idx in examples.groupby(["year", "round"]).indices.items():
        rule = rule_before(table, year, rnd + 1, floor)
        if rule is None:
            continue
        e = examples.iloc[idx]
        path = project_prices(rule, e["price"], e["last2"], e["last1"], e["games_played"],
                              scores[idx], np.ones((len(idx), PRICE_HORIZON)))
        out[idx] = path[:, -1] - e["price"].to_numpy()
    return out


# ── Training both, honestly ───────────────────────────────────────────────────

def debut_rate(rounds: pd.DataFrame, totals: dict) -> float:
    """Chance a rostered player who hasn't played yet this season debuts in
    any given round (the availability model needs a game to work from)."""
    rates = []
    for year, t in totals.items():
        season = rounds[rounds["year"] == year]
        if season.empty:
            continue
        debuts = season.groupby("player_name")["round"].min()
        for rnd in range(2, int(season["round"].max()) + 1):
            waiting = len(t) - int((debuts < rnd).sum())
            if waiting > 0:
                rates.append(int((debuts == rnd).sum()) / waiting)
    return float(np.mean(rates)) if rates else 0.02


class OutlookModels(NamedTuple):
    availability: object          # P(named for the next team game)
    price_trees: object           # 3-game price change (replaced by the rule)
    debut_rate: float
    price_rules: pd.DataFrame     # K and mu per (year, round) from fit_price_rule


def train_outlook(rounds: pd.DataFrame, totals: dict, before_year: int) -> OutlookModels:
    """Fit the trees (and the debut rate) on seasons before `before_year`.
    The price rule's table also covers `before_year` itself: its K and mu
    are read round by round, only from moves before the round in question
    (rule_before)."""
    past = rounds[rounds["year"] < before_year]
    avail = train_availability(availability_examples(past))
    price = train_price_model(price_examples(past))
    rate = debut_rate(past, {y: t for y, t in totals.items() if y < before_year})
    rules = fit_price_rule(rounds[rounds["year"] <= before_year],
                           {y: t for y, t in totals.items() if y <= before_year})
    return OutlookModels(avail, price, rate, rules)


def evaluate_outlook(rounds: pd.DataFrame, totals: dict, year: int,
                     next_score: pd.DataFrame | None = None) -> dict:
    """Out-of-time check on `year` (trees trained on earlier years).
    `next_score` (player_name, year, round, predicted): the score model's
    prediction for the game after each game; with it, the price rule's
    3-game forecast from predicted scores is scored against the trees."""
    from sklearn.metrics import log_loss, mean_absolute_error, r2_score, roc_auc_score
    models = train_outlook(rounds, totals, year)
    season = rounds[rounds["year"] <= year]
    ex = availability_examples(season)
    ex = ex[(ex["year"] == year)].dropna(subset=["games_played"])
    p = predict_availability(models.availability, ex)
    naive = np.clip(ex["named_prev"].values, 0.02, 0.98) * 0.9 + 0.05
    origin = (ex["origin_week"] > 0).to_numpy()
    out = {
        "availability_auc": float(roc_auc_score(ex["named"], p)),
        "availability_log_loss": float(log_loss(ex["named"], p)),
        "naive_log_loss": float(log_loss(ex["named"], naive)),
        "origin_weeks_log_loss": float(log_loss(ex["named"][origin], p[origin])),
    }
    pe = price_examples(season)
    pe = pe[pe["year"] == year].reset_index(drop=True)
    if next_score is not None:
        pe = pe.merge(next_score, on=["player_name", "year", "round"])
    trees = predict_price_change(models.price_trees, pe)
    out["price_trees_r2"] = float(r2_score(pe["price_change"], trees))
    out["price_trees_mae"] = float(mean_absolute_error(pe["price_change"], trees))
    if next_score is not None:
        floor = price_floor(totals[year])
        forecasts = {
            "price_rule": np.repeat(pe[["predicted"]].to_numpy(), PRICE_HORIZON, axis=1),
            "price_rule_real_scores": pe[[f"score_next{j}" for j in
                                          range(1, PRICE_HORIZON + 1)]].to_numpy(float)}
        for name, scores in forecasts.items():
            change = rule_price_change(pe, models.price_rules, floor, scores)
            out[f"{name}_r2"] = float(r2_score(pe["price_change"], change))
            out[f"{name}_mae"] = float(mean_absolute_error(pe["price_change"], change))
    out["price_rule_reproduction"] = evaluate_price_rule(rounds, totals, [year])[year]
    log.info("Outlook models on %d: availability AUC %.3f (log-loss %.3f vs naive %.3f), "
             "3-game price change R2: trees %.3f, rule from predicted scores %s",
             year, out["availability_auc"], out["availability_log_loss"],
             out["naive_log_loss"], out["price_trees_r2"],
             f"{out['price_rule_r2']:.3f}" if "price_rule_r2" in out else "n/a")
    return out

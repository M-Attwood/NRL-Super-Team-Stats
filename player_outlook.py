"""
player_outlook.py — the two forecasts a coach needs besides next week's score.

  availability   Will the player be named for their team's next game? Learned
                 from every round a player's team played after their debut
                 that season: named or not, given recent games, minutes,
                 starting role and how many games in a row they have missed.
  price change   How much will their price move over their next three games?
                 Supercoach prices react to scores against the break-even,
                 so the move is largely predictable from recent form.

Both are gradient-boosted trees trained only on seasons before the one
being predicted, like the score model.
"""

import logging

import numpy as np
import pandas as pd

from model import player_state

log = logging.getLogger(__name__)

MAX_STREAK = 6
PRICE_HORIZON = 3      # games

AVAIL_FEATURES = ["named_prev", "missed_streak", "games_played", "season_avg",
                  "last3_avg", "mins_last1", "mins_avg", "last_started", "last_benched",
                  "start_share", "price", "next_round"]
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
    return attach_history(grid.rename(columns={"round": "next_round"}), state, tr)


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


def train_availability(examples: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingClassifier
    data = examples.dropna(subset=["games_played"])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                         random_state=0)
    return clf.fit(data[AVAIL_FEATURES].fillna(0), data["named"].astype(int))


def predict_availability(clf, frame: pd.DataFrame) -> np.ndarray:
    return clf.predict_proba(frame[AVAIL_FEATURES].fillna(0))[:, 1]


# ── Price change ──────────────────────────────────────────────────────────────

def price_examples(rounds: pd.DataFrame) -> pd.DataFrame:
    """State after game k with the price before game k+1 ("now"), and the
    price change over the next PRICE_HORIZON games."""
    state = player_state(rounds)
    g = state.groupby(["player_name", "year"], sort=False)["price"]
    state["price_now"] = g.shift(-1)
    state["price_change"] = g.shift(-1 - PRICE_HORIZON) - state["price_now"]
    out = state.dropna(subset=["price_now", "price_change"]).copy()
    out["price"] = out["price_now"]
    return out


def train_price_model(examples: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         random_state=0).fit(
        examples[PRICE_FEATURES].fillna(0), examples["price_change"])


def predict_price_change(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(frame[PRICE_FEATURES].fillna(0))


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


def train_outlook(rounds: pd.DataFrame, totals: dict, before_year: int):
    """Fit both models (and the debut rate) on seasons before `before_year`."""
    past = rounds[rounds["year"] < before_year]
    avail = train_availability(availability_examples(past))
    price = train_price_model(price_examples(past))
    rate = debut_rate(past, {y: t for y, t in totals.items() if y < before_year})
    return avail, price, rate


def evaluate_outlook(rounds: pd.DataFrame, totals: dict, year: int) -> dict:
    """Out-of-time check of both models on `year` (trained on earlier years)."""
    from sklearn.metrics import log_loss, mean_absolute_error, r2_score, roc_auc_score
    avail, price, _ = train_outlook(rounds, totals, year)
    season = rounds[rounds["year"] <= year]
    ex = availability_examples(season)
    ex = ex[(ex["year"] == year)].dropna(subset=["games_played"])
    p = predict_availability(avail, ex)
    naive = np.clip(ex["named_prev"].values, 0.02, 0.98) * 0.9 + 0.05
    pe = price_examples(season)
    pe = pe[pe["year"] == year]
    pp = predict_price_change(price, pe)
    out = {
        "availability_auc": float(roc_auc_score(ex["named"], p)),
        "availability_log_loss": float(log_loss(ex["named"], p)),
        "naive_log_loss": float(log_loss(ex["named"], naive)),
        "price_change_r2": float(r2_score(pe["price_change"], pp)),
        "price_change_mae": float(mean_absolute_error(pe["price_change"], pp)),
    }
    log.info("Outlook models on %d: availability AUC %.3f (log-loss %.3f vs naive %.3f), "
             "price-change R2 %.3f, MAE $%.0f", year, out["availability_auc"],
             out["availability_log_loss"], out["naive_log_loss"],
             out["price_change_r2"], out["price_change_mae"])
    return out

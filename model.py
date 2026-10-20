"""
model.py — Next-game Supercoach score prediction.

Every row the stats API returns is a player-round snapshot: that round's
score, minutes and stat line. The model learns

    state after a player's game k   →   score in their game k+1

where "state" only uses games 1..k (season-to-date averages, recent form,
minutes, scoring-component averages, price, last season's average, and the
upcoming opponent's points conceded to that position so far). Nothing from
game k+1 or later leaks into the features, with one deliberate exception:
the team list for game k+1, published before it is played. Whether the
player is named to start (jersey 1-13) or on the interchange (14-17) is
an input. This assumes the final team list, as the coach had it.

Evaluation is out-of-time: train on 2022-2025, test on every 2026 game, and
compare against the heuristics fantasy coaches actually use (season average,
last-3 average, price-implied average) plus a ridge regression.

Usage:
    python model.py                 # evaluate on the latest season, then train
                                    # the production model on all seasons
"""

import json
import logging

import numpy as np
import pandas as pd

from paths import MODELS_DIR

log = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "next_game_model.keras"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

ALL_POSITIONS = ["HOK", "FRF", "2RF", "HFB", "5/8", "CTW", "FLB"]
POS_COLS = [f"pos_{p.replace('/', '_')}" for p in ALL_POSITIONS]   # "5/8" → pos_5_8

# Per-game columns averaged over the season so far.
ROLLING_STATS = {
    "pts_base": "base_avg",
    "pts_attack": "attack_avg",
    "pts_create": "create_avg",
    "pts_evade": "evade_avg",
    "pts_negative": "negative_avg",
    "base_power_pts": "base_power_avg",
    "stat_tackles": "tackles_avg",
    "stat_missed_tackles": "missed_tackles_avg",
    "stat_tackle_breaks": "tackle_breaks_avg",
    "stat_line_breaks": "line_breaks_avg",
    "stat_lb_assists": "lb_assists_avg",
    "stat_offloads": "offloads_avg",
    "stat_tries": "tries_avg",
    "stat_try_assists": "try_assists_avg",
    "stat_goals": "goals_avg",
    "stat_hitups_h8": "hitups_h8_avg",
    "stat_hitups_hu": "hitups_hu_avg",
    "stat_errors": "errors_avg",
    "stat_penalties": "penalties_avg",
}

FEATURES = [
    # form
    "season_avg", "last1", "last3_avg", "last5_avg", "season_std", "games_played",
    # opportunity
    "mins_avg", "mins_last3", "mins_last1", "ppm",
    "last_started", "last_benched", "start_share",
    # this game's team list, published before it is played
    "named_start", "named_bench",
    # how the points are scored
    *ROLLING_STATS.values(),
    # market signals
    "price", "break_even", "season_price_change",
    # prior
    "prev_season_avg", "prev_season_games", "has_prev_season",
    # team and next-game context
    "team_factor", "next_round", "rounds_since_last", "opp_factor",
    *POS_COLS,
]
TARGET = "next_score"

NN_LABEL = "Wide & Deep NN"
MODEL_LABEL = "NN + ridge ensemble"      # the production predictor
ORACLE_LABEL = "Hindsight oracle"

MAX_GAP = 5            # rounds_since_last is capped (off-season ≈ long gap)
OPP_PRIOR_GAMES = 30   # shrinkage strength for the opponent-concession factor
TEAM_PRIOR_GAMES = 3   # ... and for the team-strength factor
SEED = 42


# ── Raw season data → per-round rows ──────────────────────────────────────────

def split_season(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one season's scrape into (per-round rows, season-total rows)."""
    df = df.copy()
    is_total = df["current_round"].astype(str).str.strip() == "Totals"
    # The site occasionally repeats a row (e.g. 14 duplicate season totals in
    # 2023); keep one per player, and one per player-round.
    totals = df[is_total].drop_duplicates("player_name").copy()
    rounds = df[~is_total].drop_duplicates(["player_name", "current_round"]).copy()

    rounds["year"] = year
    rounds["round"] = pd.to_numeric(rounds["current_round"], errors="coerce")
    rounds = rounds.dropna(subset=["round"])
    rounds["round"] = rounds["round"].astype(int)
    rounds["score"] = pd.to_numeric(rounds["total_points"], errors="coerce").fillna(0)
    rounds["mins"] = pd.to_numeric(rounds["total_minutes"], errors="coerce").fillna(0)
    for col in [*ROLLING_STATS, "price", "break_even", "season_price_change",
                "jersey_number"]:
        rounds[col] = pd.to_numeric(rounds.get(col), errors="coerce")
    rounds["primary_pos"] = primary_position(rounds["positions"])

    totals["year"] = year
    for col in ("avg_points", "games_played", "price", "start_price", "total_points"):
        totals[col] = pd.to_numeric(totals.get(col), errors="coerce")
    return rounds.reset_index(drop=True), totals.reset_index(drop=True)


def primary_position(positions: pd.Series) -> pd.Series:
    return positions.fillna("").astype(str).str.split("|").str[0].replace("", "UNK")


def position_one_hots(positions: pd.Series) -> pd.DataFrame:
    """One column per position; dual-position players get two 1s.
    Split on "|" only — "5/8" is a position name."""
    parts = positions.fillna("").astype(str).str.split("|")
    return pd.DataFrame(
        {col: parts.map(lambda ps, p=pos: float(p in ps))
         for pos, col in zip(ALL_POSITIONS, POS_COLS)},
        index=positions.index,
    )


# ── Feature engineering ───────────────────────────────────────────────────────

def named_role(jersey: pd.Series) -> pd.DataFrame:
    """Team-list role from a jersey number: 1-13 named to start, 14-17 on
    the interchange. Reserves called up late (18+) and unknown get neither."""
    jersey = pd.to_numeric(jersey, errors="coerce")
    return pd.DataFrame({"named_start": (jersey <= 13).astype(float),
                         "named_bench": jersey.between(14, 17).astype(float)},
                        index=jersey.index)


# The three team-list roles and their (named_start, named_bench) inputs.
ROLES = {"start": (1.0, 0.0), "interchange": (0.0, 1.0), "other": (0.0, 0.0)}


def role_index(roles: pd.DataFrame) -> np.ndarray:
    """0 = named to start, 1 = interchange, 2 = neither, per row of a
    (named_start, named_bench) frame."""
    return np.where(roles["named_start"] > 0.5, 0, np.where(roles["named_bench"] > 0.5, 1, 2))


def role_transitions(rounds: pd.DataFrame, ahead: int) -> np.ndarray:
    """P(team-list role j games later | role now) for j = 1..ahead, as an
    (ahead, 3, 3) array over ROLES. Used to predict games whose team list
    isn't out yet: a starter is still named to start 93% of the time a game
    later, an interchange player stays there about 70% of the time."""
    r = rounds.sort_values(["player_name", "year", "round"])
    role = pd.Series(role_index(named_role(r["jersey_number"])), index=r.index)
    by = role.groupby([r["player_name"], r["year"]])
    out = np.zeros((ahead, 3, 3))
    for j in range(1, ahead + 1):
        later = by.shift(-j)
        ok = later.notna()
        counts = pd.crosstab(role[ok], later[ok].astype(int)).reindex(
            index=range(3), columns=range(3), fill_value=0).to_numpy(float)
        out[j - 1] = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    return out


def predict_over_roles(predictor, frame: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    """Expected prediction when the team-list role is uncertain: the
    prediction in each role weighted by its probability (rows x ROLES)."""
    out = np.zeros(len(frame))
    for k, (start, bench) in enumerate(ROLES.values()):
        out += probs[:, k] * predictor.predict(frame.assign(named_start=start,
                                                            named_bench=bench))
    return out


def player_state(rounds: pd.DataFrame) -> pd.DataFrame:
    """For every player-round row, the player's state *after* that game,
    using only that game and earlier ones in the same season."""
    r = rounds.sort_values(["player_name", "year", "round"]).reset_index(drop=True)
    g = r.groupby(["player_name", "year"], sort=False)

    n = g.cumcount() + 1
    cum_score = g["score"].cumsum()
    cum_sq = (r["score"] ** 2).groupby([r["player_name"], r["year"]]).cumsum()
    cum_mins = g["mins"].cumsum()

    r["games_played"] = n
    r["season_avg"] = cum_score / n
    var = (cum_sq - n * r["season_avg"] ** 2) / (n - 1).where(n > 1)
    r["season_std"] = np.sqrt(var.clip(lower=0)).fillna(0)
    r["last1"] = r["score"]
    r["last3_avg"] = g["score"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    r["last5_avg"] = g["score"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    r["mins_avg"] = cum_mins / n
    r["mins_last3"] = g["mins"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    r["mins_last1"] = r["mins"]
    r["ppm"] = (cum_score / cum_mins.replace(0, np.nan)).fillna(0)
    # Jersey 1-13 = named to start, 14-17 = interchange.
    r["last_started"] = (r["jersey_number"] <= 13).astype(float)
    r["last_benched"] = r["jersey_number"].between(14, 17).astype(float)
    r["start_share"] = r["last_started"].groupby([r["player_name"], r["year"]]).cumsum() / n
    for raw, feat in ROLLING_STATS.items():
        r[feat] = r[raw].fillna(0).groupby([r["player_name"], r["year"]]).cumsum() / n
    return r


def _prev_season_lookup(totals_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """(player_name, year) → that player's average and games in year-1."""
    frames = []
    for year, t in totals_by_year.items():
        played = t[t["games_played"] > 0]
        frames.append(pd.DataFrame({
            "player_name": played["player_name"].values,
            "year": year + 1,
            "prev_season_avg": played["avg_points"].values,
            "prev_season_games": played["games_played"].values,
        }))
    if not frames:
        return pd.DataFrame(columns=["player_name", "year",
                                     "prev_season_avg", "prev_season_games"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["player_name", "year"])


def add_prev_season(df: pd.DataFrame, prev_lookup: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["prev_season_avg", "prev_season_games"], errors="ignore")
    df = df.merge(prev_lookup, on=["player_name", "year"], how="left")
    df["has_prev_season"] = df["prev_season_avg"].notna().astype(float)
    df["prev_season_avg"] = df["prev_season_avg"].fillna(0)
    df["prev_season_games"] = df["prev_season_games"].fillna(0)
    return df


def _cumulative(rounds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Running score total and game count per `keys` group, round by round."""
    agg = (rounds.groupby(keys + ["round"])["score"].agg(["sum", "count"])
           .reset_index().sort_values("round"))
    agg[["cum_sum", "cum_n"]] = agg.groupby(keys)[["sum", "count"]].cumsum()
    return agg[keys + ["round", "cum_sum", "cum_n"]]


def _as_of(queries: pd.DataFrame, table: pd.DataFrame, by: list[str],
           at: str, inclusive: bool) -> pd.DataFrame:
    """For each query, the latest `table` row with matching `by` keys at
    (inclusive=True) or strictly before (False) round `queries[at]`.
    Index-aligned to `queries`; NaN where nothing matches."""
    q = queries[by + [at]].dropna().rename(columns={at: "round"})
    t = table.copy()
    for col in by:   # merge_asof needs identical key dtypes on both sides
        dtype = "int64" if col == "year" else "string"
        q[col], t[col] = q[col].astype(dtype), t[col].astype(dtype)
    q["round"], t["round"] = q["round"].astype("int64"), t["round"].astype("int64")
    q["_idx"] = q.index
    merged = pd.merge_asof(q.sort_values("round"), t.sort_values("round"),
                           on="round", by=by, allow_exact_matches=inclusive)
    return merged.set_index("_idx").reindex(queries.index)


def _relative_rate(group: pd.DataFrame, league: pd.DataFrame, prior: int) -> pd.Series:
    """Group's per-game average shrunk toward the league's, as a ratio to it."""
    lg_avg = league["cum_sum"] / league["cum_n"]
    shrunk = ((group["cum_sum"].fillna(0) + prior * lg_avg)
              / (group["cum_n"].fillna(0) + prior))
    return (shrunk / lg_avg).where(lg_avg > 0).fillna(1.0)


def opponent_factor(rounds: pd.DataFrame, queries: pd.DataFrame) -> pd.Series:
    """How generous the upcoming opponent has been to this position.

    Average score the opponent conceded to the query's primary position in
    rounds *before* `next_round` that season, shrunk toward the league
    average and expressed as a ratio (1.1 = concedes 10% more than average).
    Unknown opponents (byes, missing fixtures, round 1) get 1.0.
    """
    if rounds.empty:
        return pd.Series(1.0, index=queries.index)
    conceded = _cumulative(rounds, ["year", "vs_team", "primary_pos"]).rename(
        columns={"vs_team": "opp"})
    league = _cumulative(rounds, ["year", "primary_pos"])
    opp = _as_of(queries, conceded, ["year", "opp", "primary_pos"], "next_round", False)
    lg = _as_of(queries, league, ["year", "primary_pos"], "next_round", False)
    return _relative_rate(opp, lg, OPP_PRIOR_GAMES)


def team_factor(rounds: pd.DataFrame, queries: pd.DataFrame) -> pd.Series:
    """Team's total Supercoach points per game up to round `as_of`, relative
    to the league — strong attacking sides lift every player's ceiling."""
    if rounds.empty:
        return pd.Series(1.0, index=queries.index)
    per_game = rounds.groupby(["year", "team", "round"])["score"].sum().reset_index()
    team = _as_of(queries, _cumulative(per_game, ["year", "team"]),
                  ["year", "team"], "as_of", True)
    league = _as_of(queries, _cumulative(per_game, ["year"]), ["year"], "as_of", True)
    return _relative_rate(team, league, TEAM_PRIOR_GAMES)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df.drop(columns=POS_COLS, errors="ignore"),
                    position_one_hots(df["positions"])], axis=1)
    df["rounds_since_last"] = df["rounds_since_last"].clip(1, MAX_GAP)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def load_seasons(seasons: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Raw season scrapes → (all per-round rows, {year: totals})."""
    rounds, totals = [], {}
    for year, df in sorted(seasons.items()):
        r, t = split_season(df, year)
        rounds.append(r)
        totals[year] = t
    return pd.concat(rounds, ignore_index=True), totals


def build_training_pairs(rounds: pd.DataFrame, totals: dict) -> pd.DataFrame:
    """(state after game k, score in game k+1) pairs for every player-season."""
    state = player_state(rounds)
    g = state.groupby(["player_name", "year"], sort=False)
    state[TARGET] = g["score"].shift(-1)
    state["next_round"] = g["round"].shift(-1)
    state["opp"] = g["vs_team"].shift(-1)
    # Game k+1's team list: only its jersey number, nothing it produced.
    state[["named_start", "named_bench"]] = named_role(g["jersey_number"].shift(-1))
    pairs = state.dropna(subset=[TARGET]).copy()
    pairs["rounds_since_last"] = pairs["next_round"] - pairs["round"]
    pairs["opp_factor"] = opponent_factor(rounds, pairs)
    pairs["team_factor"] = team_factor(rounds, pairs.assign(as_of=pairs["round"]))
    pairs = add_prev_season(pairs, _prev_season_lookup(totals))
    pairs = _finalise(pairs)
    log.info("Training pairs: %d games across seasons %s",
             len(pairs), sorted(pairs["year"].unique()))
    return pairs


STATE_FEATURES = ["season_avg", "last1", "last3_avg", "last5_avg", "season_std",
                  "games_played", "mins_avg", "mins_last3", "mins_last1", "ppm",
                  "last_started", "last_benched", "start_share",
                  *ROLLING_STATS.values(), "break_even", "season_price_change"]


def build_prediction_frame(rounds: pd.DataFrame, totals: dict, year: int,
                           next_round: int,
                           fixtures: pd.DataFrame | None = None,
                           price_col: str = "price",
                           team_list: pd.Series | None = None) -> pd.DataFrame:
    """One row per rostered player in `year`, ready to predict round `next_round`.

    Each player's state comes from their latest game before `next_round`
    this season, or from last season's final game if they haven't played
    yet (the pre-season case). Players with neither are flagged `is_rookie`.
    Only games before `next_round` are used, so any round of a finished
    season can be replayed exactly as it looked at the time. Pass
    price_col="start_price" to replay the pre-season market.

    `team_list` (jersey number by player_name) is the round's published team
    list. Players on it get its role; anyone else, or everyone when it isn't
    out yet, gets the role from their latest game as the best guess.
    """
    from scraper import get_next_opponent

    roster = totals[year][["player_name", "team", "positions", "bye_round"]].copy()
    roster["price"] = totals[year][price_col].values

    history = rounds[(rounds["year"] == year - 1)
                     | ((rounds["year"] == year) & (rounds["round"] < next_round))]
    latest = (player_state(history)
              .sort_values(["year", "round"])
              .drop_duplicates("player_name", keep="last"))
    frame = roster.merge(latest[["player_name", "year", "round", *STATE_FEATURES]],
                         on="player_name", how="left")

    in_season = frame["year"] == year
    frame["is_rookie"] = frame["year"].isna()
    frame["year"] = frame["year"].fillna(year).astype(int)
    frame["next_round"] = next_round
    frame["rounds_since_last"] = np.where(in_season, next_round - frame["round"], MAX_GAP)
    frame["avg_points"] = np.where(in_season, frame["season_avg"], 0.0)  # display only

    frame["opp"] = [get_next_opponent(t, next_round, fixtures)[0] for t in frame["team"]]
    frame["primary_pos"] = primary_position(frame["positions"])
    this_season = rounds[rounds["year"] == year]
    context = frame.assign(year=year, as_of=next_round - 1)
    frame["opp_factor"] = opponent_factor(this_season, context)
    frame["team_factor"] = team_factor(this_season, context)

    frame = add_prev_season(frame, _prev_season_lookup(totals))
    frame[["named_start", "named_bench"]] = frame[["last_started", "last_benched"]].to_numpy()
    if team_list is not None:
        listed = frame["player_name"].isin(team_list.index).to_numpy()
        role = named_role(frame["player_name"].map(team_list))
        frame.loc[listed, ["named_start", "named_bench"]] = role[listed].to_numpy()
    frame["named_listed"] = (frame["player_name"].isin(team_list.index)
                             if team_list is not None else False)
    frame = _finalise(frame)
    frame["price_usd"] = frame["price"]   # dollar price used by optimizer / planner
    return frame


# ── Baselines ─────────────────────────────────────────────────────────────────

def fit_price_baseline(pairs: pd.DataFrame) -> tuple[float, float]:
    """Linear fit of next-game score on price — Supercoach prices are set
    from expected average, so this is the market's own forecast."""
    slope, intercept = np.polyfit(pairs["price"], pairs[TARGET], 1)
    return float(slope), float(intercept)


ROOKIE_TERMS = ["intercept", "price", "named_start", "named_bench"]


def fit_rookie_baseline(rounds: pd.DataFrame) -> dict | None:
    """Score in a rookie's first game (no games last season) from his price
    and team-list role, by least squares over every season in `rounds` that
    has the season before it. The network has no history to work from for
    these players; a rookie named to start scores far more than one on the
    interchange at the same price."""
    r = rounds.sort_values(["player_name", "year", "round"])
    first = r.drop_duplicates(["player_name", "year"])
    seasons = set(r["year"])
    played_before = set(zip(r["player_name"], r["year"] + 1))
    rookie = first[[(p, y) not in played_before and y - 1 in seasons
                    for p, y in zip(first["player_name"], first["year"])]]
    if len(rookie) < 30:
        return None                    # no earlier season to tell rookies apart
    role = named_role(rookie["jersey_number"])
    X = np.column_stack([np.ones(len(rookie)), rookie["price"], role])
    coef, *_ = np.linalg.lstsq(X, rookie["score"].to_numpy(float), rcond=None)
    return dict(zip(ROOKIE_TERMS, map(float, coef)))


def rookie_estimate(frame: pd.DataFrame, rookie_fit: dict) -> np.ndarray:
    return (rookie_fit["intercept"] + rookie_fit["price"] * frame["price"]
            + rookie_fit["named_start"] * frame["named_start"]
            + rookie_fit["named_bench"] * frame["named_bench"]).to_numpy()


def baseline_predictions(df: pd.DataFrame, price_fit: tuple[float, float]) -> dict:
    slope, intercept = price_fit
    return {
        "Season average": df["season_avg"].values,
        "Last-3 average": df["last3_avg"].values,
        "Price-implied": (df["price"] * slope + intercept).values,
    }


# ── Neural network ────────────────────────────────────────────────────────────

def _build_network(X_train: np.ndarray, y_mean: float, hidden=(64, 32),
                   dropout=(0.3, 0.2), wide: bool = True, loss="mse"):
    """Wide & Deep regressor (Cheng et al., 2016).

    The wide path is a linear model on the standardised features; the deep
    path (two ReLU layers, dropout + L2) learns non-linear corrections and
    starts at zero, so training begins from the linear solution and only
    moves away where the data supports it. On this noisy target a plain MLP
    (wide=False) under-performed a linear model; this architecture matches it
    while keeping the capacity to model interactions. The keyword arguments
    exist so network_report.py can compare variants on a development season.
    """
    from tensorflow import keras

    l2 = keras.regularizers.l2(1e-4)
    norm = keras.layers.Normalization(name="standardise")
    norm.adapt(X_train)

    inputs = keras.Input(shape=(X_train.shape[1],), name="features")
    z = norm(inputs)
    if wide:     # built first: layer order fixes the seeded initial weights
        wide_out = keras.layers.Dense(1, name="wide",
                                      bias_initializer=keras.initializers.Constant(y_mean))(z)
    deep = z
    for i, (units, rate) in enumerate(zip(hidden, dropout), start=1):
        deep = keras.layers.Dense(units, activation="relu", kernel_regularizer=l2,
                                  name=f"hidden_{i}")(deep)
        deep = keras.layers.Dropout(rate, name=f"dropout_{i}")(deep)
    if wide:
        deep = keras.layers.Dense(1, name="deep", kernel_initializer="zeros")(deep)
        outputs = keras.layers.Add(name="predicted_score")([wide_out, deep])
    else:
        outputs = keras.layers.Dense(1, name="predicted_score",
                                     bias_initializer=keras.initializers.Constant(y_mean))(deep)

    model = keras.Model(inputs, outputs, name="wide_and_deep" if wide else "mlp")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["mae"])
    return model


def fit_network(pairs: pd.DataFrame, seed: int = SEED, **arch):
    """Fit the network and return (model, per-epoch training history).
    15% of *players* are held out for early stopping so the same player
    never sits in both the fit and the stopping set."""
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    players = pairs["player_name"].unique()
    rng = np.random.default_rng(seed)
    val_players = set(rng.choice(players, size=int(len(players) * 0.15), replace=False))
    is_val = pairs["player_name"].isin(val_players).values

    X = pairs[FEATURES].to_numpy(np.float32)
    y = pairs[TARGET].to_numpy(np.float32)
    model = _build_network(X[~is_val], float(y[~is_val].mean()), **arch)
    history = model.fit(
        X[~is_val], y[~is_val],
        validation_data=(X[is_val], y[is_val]),
        epochs=200, batch_size=256, verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=5, min_lr=1e-5),
        ],
    )
    return model, history.history


def train_network(pairs: pd.DataFrame, seed: int = SEED, **arch):
    return fit_network(pairs, seed, **arch)[0]


def nn_predict(model, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].to_numpy(np.float32)
    return model.predict(X, verbose=0, batch_size=2048).flatten()


class LinearModel:
    """Standardised ridge regression stored as plain numbers, so it can live
    in the JSON model card instead of a pickle."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.coef, self.intercept = np.asarray(coef), float(intercept)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].to_numpy(float)
        return ((X - self.mean) / self.scale) @ self.coef + self.intercept

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept}


def train_ridge(pairs: pd.DataFrame) -> LinearModel:
    from sklearn.linear_model import Ridge
    X = pairs[FEATURES].to_numpy(float)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    ridge = Ridge(alpha=1.0).fit((X - mean) / scale, pairs[TARGET])
    return LinearModel(mean, scale, ridge.coef_, ridge.intercept_)


class Predictor:
    """The production score model: the average of the Wide & Deep network
    and a ridge regression. Chosen because the average ranked players better
    than either alone on both development seasons (2024, 2025)."""

    def __init__(self, network, ridge: LinearModel):
        self.network, self.ridge = network, ridge

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (nn_predict(self.network, df) + self.ridge.predict(df)) / 2


def train_predictor(pairs: pd.DataFrame) -> Predictor:
    return Predictor(train_network(pairs), train_ridge(pairs))


# ── Evaluation ────────────────────────────────────────────────────────────────

def score_metrics(y_true, y_pred, rounds=None) -> dict:
    """MAE / RMSE / R², plus mean per-round Spearman rank correlation —
    for squad selection, ranking players correctly is what matters."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    }
    if rounds is not None:
        df = pd.DataFrame({"t": y_true, "p": y_pred, "r": np.asarray(rounds)})
        rho = df.groupby("r").apply(
            lambda d: d["t"].rank().corr(d["p"].rank()), include_groups=False)
        out["rank_corr"] = float(rho.mean())
    return out


def evaluate_holdout(pairs: pd.DataFrame, test_year: int):
    """Train on seasons before `test_year`, predict every `test_year` game.

    Returns ({model_name: metrics}, test frame with one prediction column
    per model, the Predictor trained without `test_year`) — the honest,
    out-of-time estimate of model quality.
    """
    train = pairs[pairs["year"] < test_year]
    test = pairs[pairs["year"] == test_year].copy()
    log.info("Holdout evaluation: train %d games (%d-%d), test %d games (%d)",
             len(train), train["year"].min(), train["year"].max(),
             len(test), test_year)

    predictor = train_predictor(train)
    nn, ridge = nn_predict(predictor.network, test), predictor.ridge.predict(test)
    preds = {MODEL_LABEL: (nn + ridge) / 2, NN_LABEL: nn, "Ridge regression": ridge}
    preds.update(baseline_predictions(test, fit_price_baseline(train)))
    # Not a real model: it knows each player's average over the very games
    # being predicted. It marks how much error is irreducible game-day noise.
    preds[ORACLE_LABEL] = test.groupby("player_name")[TARGET].transform("mean").values

    metrics = {}
    for name, p in preds.items():
        test[f"pred_{name}"] = p
        metrics[name] = score_metrics(test[TARGET], p, test["next_round"])
        log.info("  %-24s MAE %.2f  RMSE %.2f  R2 %.3f  rank-corr %.3f", name,
                 metrics[name]["mae"], metrics[name]["rmse"],
                 metrics[name]["r2"], metrics[name]["rank_corr"])
    return metrics, test, predictor


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(predictor: Predictor, card: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    predictor.network.save(MODEL_PATH)
    card = {**card, "ridge": predictor.ridge.to_dict()}
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2), encoding="utf-8")
    log.info("Model saved -> %s", MODEL_PATH)


def load_model() -> Predictor:
    from tensorflow import keras
    ridge = LinearModel(**load_model_card()["ridge"])
    return Predictor(keras.models.load_model(MODEL_PATH), ridge)


def load_model_card() -> dict:
    if MODEL_CARD_PATH.exists():
        return json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
    return {}


def train_production_model(pairs: pd.DataFrame, holdout_metrics: dict | None,
                           test_year: int | None, rookie_fit: dict | None = None):
    """Train on every season and save alongside a model card."""
    predictor = train_predictor(pairs)
    card = {
        "target": "next-game Supercoach score",
        "model": "mean of a Wide & Deep network and a ridge regression",
        "features": FEATURES,
        "train_seasons": sorted(int(y) for y in pairs["year"].unique()),
        "train_games": int(len(pairs)),
        "price_baseline": fit_price_baseline(pairs),
        "rookie_baseline": rookie_fit,
        "holdout_season": test_year,
        "holdout_metrics": holdout_metrics,
    }
    save_model(predictor, card)
    return predictor


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_next_round(frame: pd.DataFrame, model: Predictor | None = None,
                       price_fit: tuple[float, float] | None = None,
                       rookie_fit: dict | None = None) -> pd.DataFrame:
    """Add `predicted_points` to a frame from build_prediction_frame.
    Rookies (no games in this or last season) have no history for the
    models: those on the round's team list get the rookie estimate from
    price and named role (fit_rookie_baseline), the rest the price-implied
    estimate."""
    card = {} if model and price_fit else load_model_card()
    model = model or load_model()
    frame = frame.copy()
    slope, intercept = price_fit or card.get("price_baseline", (1 / 10_000, 0.0))
    rookie_fit = rookie_fit or card.get("rookie_baseline")
    preds = model.predict(frame)
    rookie = (frame["price"] * slope + intercept).to_numpy()
    if rookie_fit and "named_listed" in frame:
        rookie = np.where(frame["named_listed"], rookie_estimate(frame, rookie_fit), rookie)
    frame["predicted_points"] = np.where(frame["is_rookie"], rookie, preds)
    frame["predicted_points"] = frame["predicted_points"].clip(lower=0)
    log.info("Predicted %d players (%d rookies on the rookie estimate): mean %.1f, max %.1f",
             len(frame), int(frame["is_rookie"].sum()),
             frame["predicted_points"].mean(), frame["predicted_points"].max())
    return frame


if __name__ == "__main__":
    import argparse
    from scraper import load_season

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Evaluate and train the score model")
    parser.add_argument("--first-year", type=int, default=2022)
    parser.add_argument("--last-year", type=int, default=2026)
    args = parser.parse_args()

    years = range(args.first_year, args.last_year + 1)
    rounds, totals = load_seasons({y: load_season(y, scrape_if_missing=False) for y in years})
    pairs = build_training_pairs(rounds, totals)
    metrics, _, _ = evaluate_holdout(pairs, args.last_year)
    train_production_model(pairs, metrics, args.last_year)

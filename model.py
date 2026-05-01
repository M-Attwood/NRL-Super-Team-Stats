"""
model.py — Feature engineering, TensorFlow model training & prediction
for NRL Supercoach score prediction.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "supercoach_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.pkl"

# Pre-season heuristic: 1 predicted Supercoach point per $10K of player price.
# Calibrated against 2022-2024 prices (~$200K minimum priced player ≈ 20 pts);
# revisit if Supercoach rescales prices between seasons.
POINTS_PER_DOLLAR = 1.0 / 10_000

# ── NRL positions and teams for one-hot encoding ─────────────────────────────
ALL_POSITIONS = ["HOK", "FRF", "2RF", "HFB", "5/8", "CTW", "FLB"]

# The 17 club codes that actually appear in scraped data. The previous list
# used internal-style codes ("BRI", "CBY", "EEL", ...) that didn't match the
# scraper's output ("BRO", "BUL", "PAR", ...), which meant 13/17 team one-hots
# were silently always zero. Now sourced from the canonical bye-round table in
# planner.BYE_ROUNDS so the two stay in sync.
def _canonical_team_codes() -> list[str]:
    from planner import BYE_ROUNDS
    return sorted(BYE_ROUNDS.keys())

ALL_TEAMS = _canonical_team_codes()


# ── Position-vs-team defensive strength loader ───────────────────────────────
#
# The scraper writes data/raw/position_vs_team_<year>.csv, which has two
# sections: ranks (1..17) and PPM (points-per-minute) conceded — both keyed
# by (team, position). We parse the second (PPM) section into a lookup that
# the feature builder uses to produce `def_ppm_conceded`.
#
# Cached in module state so we don't re-parse the CSV on every prediction call.
_DEF_TABLE_CACHE: dict[int | None, dict[tuple[str, str], float]] = {}


def _parse_position_vs_team_csv(path) -> dict[tuple[str, str], float]:
    """Parse the 'BY PPM CONCEDED' section of a position_vs_team CSV.

    Returns a dict keyed by (team_code, position_name). Position names are
    HOK, FRF, 2RF, HFB, 5/8, CTW, FLB. Team codes match planner.BYE_ROUNDS
    ("BRO", "BUL", ...). Returns {} on parse failure.
    """
    import csv
    from pathlib import Path as _Path
    table: dict[tuple[str, str], float] = {}
    p = _Path(path)
    if not p.exists():
        return table
    in_ppm = False
    columns: list[str] | None = None
    try:
        with p.open("r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                first = (row[0] or "").strip()
                if "BY PPM CONCEDED" in first:
                    in_ppm = True
                    columns = None
                    continue
                if not in_ppm:
                    continue
                # Header row beneath the section marker: ",HOK,FRF,2RF,..."
                if first == "" and len(row) > 1 and any(c.strip() for c in row[1:]):
                    columns = [c.strip() for c in row[1:]]
                    continue
                if columns is None:
                    continue
                team = first.upper()
                if not team or len(team) > 4:  # team codes are 3 chars (BRO, BUL, ...)
                    continue
                for pos, val in zip(columns, row[1:]):
                    if not pos or pos.lower() == "average":
                        continue
                    try:
                        table[(team, pos)] = float(val)
                    except (ValueError, TypeError):
                        pass
    except OSError as e:
        log.warning("Could not read %s: %s", p, e)
    return table


def load_def_strength_table(year: int | None = None) -> dict[tuple[str, str], float]:
    """Load (team, position) → ppm-conceded from data/raw/position_vs_team_*.csv.

    If `year` is given, prefers that year's file; otherwise falls back to
    the most recent file matching `position_vs_team_*.csv`. Returns an
    empty dict (so callers degrade gracefully) when no data is available.
    """
    if year in _DEF_TABLE_CACHE:
        return _DEF_TABLE_CACHE[year]
    from paths import DATA_RAW
    candidates: list[Path] = []
    if year is not None:
        candidates.append(DATA_RAW / f"position_vs_team_{year}.csv")
    candidates.extend(sorted(DATA_RAW.glob("position_vs_team_*.csv"), reverse=True))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        table = _parse_position_vs_team_csv(path)
        if table:
            log.info("Loaded def-strength table: %d (team, position) cells from %s",
                     len(table), path.name)
            _DEF_TABLE_CACHE[year] = table
            return table
    log.warning("No usable position_vs_team CSV found — def_ppm_conceded "
                "feature will fall back to mean.")
    _DEF_TABLE_CACHE[year] = {}
    return {}

# Numeric columns to scale and feed into the model.
#
# CRITICAL: `avg_points` is the TARGET (TARGET = "avg_points"). It must NEVER
# appear here, otherwise the model trivially memorises target = feature and
# reports leakage-driven validation MAE that doesn't reflect any real signal.
# (This was a real bug — caught in the NN-only review and removed.)
#
# `price` was also removed earlier (NN-6 review item). Price lives separately
# as `price_usd` for cap math in the optimizer. Pushing it through
# StandardScaler added noise without signal — historical performance columns
# already encode "player quality."
SCALE_COLS = [
    "avg_last3", "avg_last5", "avg_last2",
    "avg_minutes", "points_per_minute", "break_even",
    "season_price_change", "round_price_change", "games_played",
    "coeff_variation", "pct_60plus",
    "base_ppm", "base_power_ppm", "base_power_pts",
    "avg_penalties", "avg_errors", "avg_pc_er",
    "pct_h8", "pct_tackle_break", "pct_missed_tackle", "pct_offload", "pct_base",
    "pts_base", "pts_attack", "pts_create", "pts_evade", "pts_negative",
    "stat_tries", "stat_try_assists", "stat_last_touch", "stat_goals",
    "stat_missed_goals", "stat_field_goals", "stat_missed_fg",
    "stat_tackles", "stat_missed_tackles", "stat_tackle_breaks",
    "stat_forced_dropouts", "stat_offloads", "stat_ineff_offloads",
    "stat_line_breaks", "stat_lb_assists", "stat_forty_twenty",
    "stat_kick_regather", "stat_hitups_h8", "stat_hitups_hu",
    "stat_held_goal", "stat_intercept", "stat_kicked_dead",
    "stat_penalties", "stat_errors", "stat_sinbin",
    "avg_base", "avg_scoring", "avg_create", "avg_evade",
    "avg_negative", "avg_base_power",
    "avg_r1_9", "avg_r10_18", "avg_r19_27",
    "avg_mins_last3", "avg_mins_last5",
    "form_momentum",
    "is_rookie",
    # Position-mean baseline: explicit signal of "what does an average
    # player at this position score?" so the model isn't asked to decode
    # this from one-hots alone (May 2026 — see engineer_features).
    "pos_mean_avg_points",
    # Defensive strength: points-per-minute conceded by the player's
    # opponent at their primary position. For training rows this is the
    # player's last-played opponent (vs_team in the historical row); for
    # prediction rows it's overwritten with the upcoming fixture opponent
    # via build_prediction_features (NN-4).
    "def_ppm_conceded",
    # NOTE: `is_home_next` is set by build_prediction_features when fixtures
    # are wired, but it's intentionally NOT in SCALE_COLS yet because
    # historical training rows don't have a parallel home/away signal —
    # the feature would be a constant 0 during training and add no signal.
    # Add it here once the scraper extracts historical home/away from the
    # `Venue` column in the per-round data.
]

# Target column (raw, unscaled — `_raw` variants are saved before scaling).
TARGET = "avg_points"

# NN-2: predict delta from rolling-3 average instead of absolute score.
# The model only has to learn the deviation from recent form, which has
# much lower variance than the absolute score. After prediction we add the
# rolling baseline back. Disable by setting USE_DELTA_TARGET = False (e.g.
# for a head-to-head A/B test of the two targets).
#
# Disabled (May 2026): on this dataset the delta is mostly noise — round-to-
# round Supercoach variation is dominated by matchup, minutes and luck, while
# the learnable signal lives in the *level*. avg_last3 / avg_last5 / form_momentum
# are already in SCALE_COLS, so the network can use the rolling baseline as a
# feature rather than as a target transform. Predicting absolute scores also
# avoids the cold-start collapse where rookies (avg_last3 ≈ 0) get
# delta-only predictions and need a position-median patch downstream.
USE_DELTA_TARGET = False

# NN-1: most-recent historical year is reserved as a held-out validation
# set. The model is trained on years strictly older than this, so we get
# an honest "next season" estimate instead of leaking through random
# train/test splits.
HOLDOUT_YEAR_OFFSET = 1  # holdout = max(scrape_year) - 0; train < that


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw scraped data:
    - Coerce all numeric columns to float
    - Fill missing values with position-group medians
    - Normalise positions column
    """
    df = df.copy()

    # Ensure positions column is clean
    # Note: the scraper already produces pipe-separated positions (e.g. "2RF|FRF").
    # Do NOT replace "/" here — "5/8" is a position name, not a separator.
    if "positions" in df.columns:
        df["positions"] = df["positions"].fillna("").str.strip()
    else:
        df["positions"] = ""

    # Price: strip non-numeric characters (preserve decimal point)
    if "price" in df.columns:
        df["price"] = (df["price"].astype(str)
                       .str.replace(r"[^\d.]", "", regex=True)
                       .replace("", np.nan)
                       .astype(float))

    # Coerce all numeric-looking columns to float
    skip_cols = {"player_id", "player_name", "positions", "team", "vs_team",
                 "weather", "scrape_date", "scrape_year"}
    for col in df.columns:
        if col in skip_cols:
            continue
        try:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True).replace("", np.nan),
                errors="coerce"
            )
        except (ValueError, TypeError, AttributeError) as e:
            log.debug("skipping numeric coercion for column %s: %s", col, e)

    # Fill NaN with year-aware position-group median.
    #
    # We group by (scrape_year, primary_position) so a 2024 NaN is filled
    # using only 2024 medians — never with future-season data. Without this
    # 2026 partial-season rows (still being scored) would leak forward-looking
    # signal back into the older rows that the model trains on.
    primary_pos = df["positions"].str.split("|").str[0].fillna("UNK")
    df["_primary_pos"] = primary_pos
    if "scrape_year" in df.columns:
        df["_year_key"] = pd.to_numeric(df["scrape_year"], errors="coerce")
        group_keys = ["_year_key", "_primary_pos"]
    else:
        group_keys = ["_primary_pos"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Don't impute the year column itself
    numeric_cols = [c for c in numeric_cols if c != "_year_key"]
    for col in numeric_cols:
        if df[col].isna().any():
            group_medians = df.groupby(group_keys)[col].transform("median")
            # Fall back to position-only median (same year unavailable),
            # then global median, then zero.
            pos_medians = df.groupby("_primary_pos")[col].transform("median")
            global_median = df[col].median()
            df[col] = (df[col]
                       .fillna(group_medians)
                       .fillna(pos_medians)
                       .fillna(global_median)
                       .fillna(0))
    df = df.drop(columns=["_year_key"], errors="ignore")

    df = df.drop(columns=["_primary_pos"], errors="ignore")

    # Drop rows with no player name
    if "player_name" in df.columns:
        df = df[df["player_name"].str.strip().ne("") & df["player_name"].notna()].copy()

    log.info("clean_data: %d rows, %d columns", len(df), len(df.columns))
    return df.reset_index(drop=True)


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame,
                      scaler: StandardScaler = None,
                      fit_scaler: bool = True
                      ) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Build the feature matrix used for training and prediction.

    Returns (df_with_features, fitted_scaler).
    If fit_scaler=False, the provided scaler is used without refitting.
    """
    df = df.copy()

    # ── Derived features ─────────────────────────────────────────────────────

    # Form momentum: improvement over last 5 games
    if "avg_last3" in df.columns and "avg_last5" in df.columns:
        df["form_momentum"] = df["avg_last3"].fillna(0) - df["avg_last5"].fillna(0)
    else:
        df["form_momentum"] = 0.0

    # is_rookie: explicit signal so the model can learn how to extrapolate
    # for players with no historical games_played. Previously these rows were
    # silently filtered out before training — meaning the model never saw
    # rookies and had to fall back to a post-hoc position-median patch at
    # predict time. Keeping them in (with this flag set) lets the model
    # learn an actual rookie distribution.
    if "games_played" in df.columns:
        df["is_rookie"] = (
            pd.to_numeric(df["games_played"], errors="coerce").fillna(0) == 0
        ).astype(float)
    else:
        df["is_rookie"] = 0.0

    # ── Position-mean baseline ───────────────────────────────────────────────
    # Hand the model the absolute scoring level for each position rather than
    # asking it to decode this from the position one-hots alone. Computed
    # from rows with non-zero avg_points in the *current* df, so it works at
    # both training time (historical rows, all have avg_points) and prediction
    # time (after build_prediction_features merges historical stats in,
    # rookies get position-median fills via clean_data).
    primary_pos_for_mean = df["positions"].astype(str).str.split("|").str[0].fillna("UNK")
    if "avg_points" in df.columns:
        pts_numeric = pd.to_numeric(df["avg_points"], errors="coerce").fillna(0)
        nonzero_mask = pts_numeric > 0
        if nonzero_mask.any():
            pos_means_series = (
                pts_numeric[nonzero_mask]
                .groupby(primary_pos_for_mean[nonzero_mask])
                .mean()
            )
            global_mean = float(pts_numeric[nonzero_mask].mean())
            pos_means = pos_means_series.to_dict()
        else:
            pos_means, global_mean = {}, 0.0
        df["pos_mean_avg_points"] = (
            primary_pos_for_mean.map(pos_means).fillna(global_mean)
        )
    else:
        df["pos_mean_avg_points"] = 0.0

    # ── Defensive strength of opponent ───────────────────────────────────────
    # Look up points-per-minute conceded by the player's `vs_team` at their
    # primary position. For training rows, vs_team is the player's last
    # played opponent (per the API season-total dedup). For prediction rows
    # this column is overwritten with the upcoming fixture opponent in
    # build_prediction_features, so the same feature reflects "next matchup
    # ease" at predict time.
    def_table = load_def_strength_table()
    if def_table:
        vs_team_norm = (
            df["vs_team"].astype(str).str.strip().str.upper()
            if "vs_team" in df.columns else pd.Series("", index=df.index)
        )
        keys = list(zip(vs_team_norm.values, primary_pos_for_mean.values))
        ppm_vals = np.array(
            [def_table.get(k, np.nan) for k in keys], dtype=np.float32
        )
        overall_mean = (
            float(np.nanmean(list(def_table.values()))) if def_table else 0.0
        )
        ppm_vals = np.where(np.isnan(ppm_vals), overall_mean, ppm_vals)
        df["def_ppm_conceded"] = ppm_vals
    else:
        df["def_ppm_conceded"] = 0.0

    # is_home_next: only set by build_prediction_features when fixture data
    # is wired in. Default to 0 here so the column exists during training.
    if "is_home_next" not in df.columns:
        df["is_home_next"] = 0.0

    # ── One-hot encode positions ──────────────────────────────────────────────
    for pos in ALL_POSITIONS:
        safe_col = f"pos_{pos.replace('/', '_').replace(' ', '_')}"
        df[safe_col] = df["positions"].str.contains(pos, regex=False, na=False).astype(float)

    # ── One-hot encode teams ──────────────────────────────────────────────────
    if "team" in df.columns:
        for team in ALL_TEAMS:
            df[f"team_{team}"] = (df["team"].str.upper() == team).astype(float)
    else:
        for team in ALL_TEAMS:
            df[f"team_{team}"] = 0.0

    # ── Collect feature columns ───────────────────────────────────────────────
    pos_cols = [f"pos_{p.replace('/', '_').replace(' ', '_')}" for p in ALL_POSITIONS]
    team_cols = [f"team_{t}" for t in ALL_TEAMS]

    # Only include SCALE_COLS that actually exist in df
    present_scale_cols = [c for c in SCALE_COLS if c in df.columns]

    feature_cols = present_scale_cols + pos_cols + team_cols
    # Remove duplicates while preserving order
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    # Ensure all feature columns exist and are numeric
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ── Preserve originals before scaling ────────────────────────────────────
    if "price" in df.columns:
        df["price_usd"] = df["price"].copy()
    # Save raw versions so the trainer / predictor can use unscaled values:
    #   avg_points_raw  → target column for training (NN-2 delta target uses
    #                     this minus the rolling-3 average).
    #   avg_last3_raw   → rolling baseline used by NN-2 to convert predicted
    #                     deltas back to absolute Supercoach points.
    if "avg_points" in df.columns:
        df["avg_points_raw"] = df["avg_points"].copy()
    if "avg_last3" in df.columns:
        df["avg_last3_raw"] = df["avg_last3"].copy()

    # ── Scale numeric features ────────────────────────────────────────────────
    if scaler is None:
        scaler = StandardScaler()

    X = df[feature_cols].values.astype(np.float32)

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        with open(FEATURE_COLS_PATH, "wb") as f:
            pickle.dump(feature_cols, f)
        log.info("Scaler fitted and saved → %s", SCALER_PATH)
    else:
        X_scaled = scaler.transform(X)

    # Put scaled values back (price column is now scaled — use price_usd for $ amounts)
    df[feature_cols] = X_scaled
    df["_feature_cols_marker"] = 1  # sentinel so we know features are ready

    # Defragment after the many `df["new_col"] = ...` insertions above.
    # Without this, pandas emits PerformanceWarnings and downstream
    # operations slow down substantially.
    df = df.copy()

    log.info("engineer_features: %d features, %d rows", len(feature_cols), len(df))
    return df, scaler


def _load_scaler() -> StandardScaler | None:
    """Load the saved scaler, or return None if not yet fitted."""
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            return pickle.load(f)
    return None


def derive_feature_cols(df: pd.DataFrame) -> list[str]:
    """Compute the feature column list directly from `df` and the global
    SCALE_COLS / ALL_POSITIONS / ALL_TEAMS constants.

    Use this during training/feature-engineering when you want changes to
    SCALE_COLS or ALL_TEAMS to actually take effect. (The previous behaviour
    of always reading the saved pickle first meant constant changes were
    silently ignored until the pickle was deleted.)
    """
    pos_cols = [f"pos_{p.replace('/', '_').replace(' ', '_')}" for p in ALL_POSITIONS]
    team_cols = [f"team_{t}" for t in ALL_TEAMS]
    present = [c for c in SCALE_COLS if c in df.columns]
    return present + pos_cols + team_cols


def load_saved_feature_cols() -> list[str]:
    """Read the feature column list saved alongside the trained model.
    Returns [] if no model has been trained yet."""
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH, "rb") as f:
            return pickle.load(f)
    return []


def get_feature_cols(df: pd.DataFrame = None) -> list[str]:
    """Backwards-compatible wrapper. Returns the saved list if it exists
    (the right call at *inference* time, where features must match what
    the saved model was trained on); otherwise derives from `df`.

    For training/feature-engineering callers, prefer `derive_feature_cols`
    so that SCALE_COLS edits propagate immediately."""
    saved = load_saved_feature_cols()
    if saved:
        return saved
    if df is not None:
        return derive_feature_cols(df)
    return []


# ── TensorFlow Model ──────────────────────────────────────────────────────────

def _build_model(n_features: int):
    """Build and compile the Keras regression model."""
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1),
    ], name="supercoach_predictor")

    # NN-7: Huber loss instead of MSE.
    # Supercoach scores have outliers — a player who randomly scores 150
    # one round shouldn't drag the model into chasing extreme tails.
    # Huber is quadratic for small errors (like MSE) but linear for large
    # ones (like MAE), giving the best of both.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def load_or_train_model(df: pd.DataFrame, force_retrain: bool = False,
                        holdout_year: int | None = None,
                        final_retrain: bool = False):
    """
    Load an existing saved model and fine-tune it, or train from scratch.
    Returns the trained Keras model.

    holdout_year: explicit year to reserve for validation. If None, the
    most recent year present in `df` is used (NN-1 default). Pass an
    explicit value to lock the validation set across runs — useful for
    reproducible benchmarking and for training on data older than this
    year only.

    final_retrain: when True, train the **production** model on ALL data
    (no year-aware holdout — the most recent year is included in training).
    A small 5% random validation split is kept solely so early-stopping has
    something to monitor; metrics from this run are NOT honest because the
    same player can appear in both train and val. Always run a non-final
    pass first to get an honest generalisation estimate, then a final pass
    to ship.

      Workflow:
        python model.py --retrain          # eval: holds out most recent year
        # inspect outputs/model_metrics.csv to confirm the model is useful
        python model.py --retrain --final  # production: trains on everything

    final_retrain=True implies a from-scratch train (the existing saved
    model would have been trained under a different validation regime, so
    fine-tuning across regimes is not meaningful).
    """
    import tensorflow as tf
    from tensorflow import keras

    # Training: always derive from `df` so SCALE_COLS / ALL_TEAMS edits
    # take effect immediately. Previously read the saved pickle first,
    # which meant constant changes were silently ignored.
    feature_cols = derive_feature_cols(df)
    if not feature_cols:
        raise ValueError("No feature columns found. Run engineer_features first.")

    # Filter to columns present in df
    feature_cols = [c for c in feature_cols if c in df.columns]
    n_features = len(feature_cols)

    if TARGET not in df.columns:
        log.warning("Target column '%s' not in df — cannot train", TARGET)
        return None

    X_full = df[feature_cols].fillna(0).values.astype(np.float32)
    # Use raw (unscaled) avg_points as target so predictions are in original units
    target_col = "avg_points_raw" if "avg_points_raw" in df.columns else TARGET
    y_absolute = df[target_col].fillna(0).values.astype(np.float32)

    # NN-2: build a delta target. The model learns avg_points - rolling_avg;
    # at prediction time we add the rolling avg back. If avg_last3_raw isn't
    # present (e.g. very old data), fall back to absolute target.
    if USE_DELTA_TARGET and "avg_last3_raw" in df.columns:
        baseline_raw = df["avg_last3_raw"].values.astype(np.float32)
        # Where the rolling-3 baseline is missing, fall back to the absolute
        # score so the delta is 0 for that row (i.e. neutral, no signal).
        baseline = np.where(np.isnan(baseline_raw), y_absolute, baseline_raw)
        y_full = y_absolute - baseline
        log.info("Training with DELTA target (avg_points - avg_last3)")
    else:
        baseline = np.zeros_like(y_absolute)
        y_full = y_absolute
        log.info("Training with ABSOLUTE target (avg_points)")

    # Keep all rows (including y==0 rookies) so the model learns the
    # cold-start tail. The previous behaviour silently filtered out every
    # player with no games — exactly the population we most need predictions
    # for. The `is_rookie` feature added in engineer_features lets the model
    # condition on this.
    mask = np.ones(len(y_absolute), dtype=bool)

    if len(X_full) < 20:
        log.warning("Not enough training samples (%d) — skipping model training",
                    len(X_full))
        return None

    # FINAL retrain (production): use ALL data, no year-aware holdout.
    # The eval pass with --retrain (no --final) gives the honest metric;
    # this pass ships the model that gets to use the most recent year too.
    if final_retrain:
        log.warning("FINAL retrain: training on ALL data with no year holdout. "
                    "Validation metrics from this run are NOT honest "
                    "(player-level leakage in the random split). Use the "
                    "non-final eval run for honest generalisation metrics.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.05, random_state=42
        )
        holdout_year = None  # for the metrics row
        log.info("Final-retrain split: %d train, %d val (random 5%%)",
                 len(X_train), len(X_val))
    # NN-1: year-aware holdout. The most recent historical year is reserved
    # for evaluation and never seen during training. This gives an honest
    # "next-season" estimate instead of the random-split leakage we had
    # before (where the same player could appear in train and test).
    elif "scrape_year" in df.columns:
        years = pd.to_numeric(df["scrape_year"], errors="coerce")
        years = years[mask].values
        max_year = int(np.nanmax(years))
        if holdout_year is None:
            holdout_year = max_year - (HOLDOUT_YEAR_OFFSET - 1)
        train_idx = years < holdout_year
        val_idx = years == holdout_year
        if train_idx.sum() < 20 or val_idx.sum() < 5:
            log.warning("Not enough samples for year holdout (train=%d, val=%d)"
                        " — falling back to random split",
                        int(train_idx.sum()), int(val_idx.sum()))
            X_train, X_val, y_train, y_val = train_test_split(
                X_full, y_full, test_size=0.15, random_state=42
            )
        else:
            X_train, y_train = X_full[train_idx], y_full[train_idx]
            X_val, y_val = X_full[val_idx], y_full[val_idx]
            log.info("Year-aware split: train on years <%d (%d samples), "
                     "validate on %d (%d samples)",
                     holdout_year, len(X_train), holdout_year, len(X_val))
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.15, random_state=42
        )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7,
            restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=0
        ),
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and not force_retrain and not final_retrain:
        log.info("Loading existing model from %s for fine-tuning ...", MODEL_PATH)
        candidate = keras.models.load_model(MODEL_PATH)
        # Schema guard: if SCALE_COLS or ALL_TEAMS changed since the model
        # was saved, the input dim won't match. Rather than crash mid-fit,
        # detect it and retrain from scratch.
        try:
            saved_n = candidate.input_shape[1]
        except (AttributeError, IndexError, TypeError):
            saved_n = None
        if saved_n is not None and saved_n != n_features:
            log.warning(
                "Saved model expects %d features but data now has %d — "
                "feature schema changed, retraining from scratch.",
                saved_n, n_features,
            )
            model = _build_model(n_features)
            epochs = 100
        else:
            # Recompile with a lower LR for fine-tuning. Recompiling (rather
            # than mutating optimizer.learning_rate) resets Adam's internal
            # moment estimates, which is the right move when the data
            # distribution may have shifted week-to-week.
            model = candidate
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss=keras.losses.Huber(delta=1.0),
                metrics=["mae"],
            )
            epochs = 20
    else:
        log.info("Training model from scratch (%d features, %d samples) ...",
                 n_features, len(X_train))
        model = _build_model(n_features)
        epochs = 100

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    model.save(MODEL_PATH)
    log.info("Model saved → %s", MODEL_PATH)

    # Evaluation
    y_pred = model.predict(X_val, verbose=0).flatten()
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    mae = mean_absolute_error(y_val, y_pred)
    log.info("Validation  RMSE: %.2f  MAE: %.2f", rmse, mae)

    # Persist metrics so we can track whether changes are actually helping.
    # Console-only metrics vanish; an append-only CSV gives a sparkline of
    # model performance run-to-run that you can plot or eyeball.
    _append_metrics_row(
        rmse=float(rmse), mae=float(mae),
        holdout_year=holdout_year,
        n_train=int(len(X_train)), n_val=int(len(X_val)),
        n_features=int(n_features),
        target_kind=("delta" if (USE_DELTA_TARGET
                                  and "avg_last3_raw" in df.columns) else "absolute"),
        max_epochs=int(epochs),
        mode=("final" if final_retrain else "eval"),
    )

    return model


# ── Metric tracking ──────────────────────────────────────────────────────────

METRICS_PATH = Path("outputs") / "model_metrics.csv"


def _append_metrics_row(**fields) -> None:
    """Append one row to outputs/model_metrics.csv (creates file + header
    on first run). Failure here must never break a training run, so all
    I/O errors are swallowed with a warning."""
    from datetime import datetime
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **fields}
    try:
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not METRICS_PATH.exists()
        pd.DataFrame([row]).to_csv(
            METRICS_PATH, mode="a", header=write_header, index=False,
        )
        log.info("Metrics appended → %s", METRICS_PATH)
    except OSError as e:
        log.warning("Could not write %s: %s", METRICS_PATH, e)


# ── Opponent strength (NN-4) ──────────────────────────────────────────────────

def compute_defensive_strength(df_historical: pd.DataFrame) -> pd.DataFrame:
    """Build a (team, primary_position) → mean points-conceded table.

    Uses per-round historical rows that have a non-null `vs_team` (i.e.
    "this player scored X against opponent Y"). Aggregates to give an
    expected-points-conceded estimate for each opponent at each position.

    Returns a DataFrame with columns:
        opponent_team, primary_position, expected_points_conceded, n_samples

    Higher expected_points_conceded = weaker defence vs that position
    (more attractive opponent for fantasy purposes).

    Why this isn't yet wired into the model:
      To turn this into a per-player feature we need to know who each
      player's NEXT opponent is. The scraper currently fetches season
      totals, not fixtures. Once fixture scraping is added, build a
      `next_opponent` column on the prediction frame and merge this
      table in on (next_opponent, primary_position). The hook point in
      `build_prediction_features` is marked `TODO(NN-4)`.
    """
    df = df_historical.copy()
    if "vs_team" not in df.columns or "avg_points" not in df.columns:
        log.warning("Cannot compute defensive strength: missing vs_team or avg_points")
        return pd.DataFrame()

    df = df[df["vs_team"].notna() & (df["vs_team"].astype(str).str.strip() != "")]
    if df.empty:
        return pd.DataFrame()

    df["primary_position"] = (
        df["positions"].astype(str).str.split("|").str[0].fillna("UNK")
    )
    df["avg_points"] = pd.to_numeric(df["avg_points"], errors="coerce")
    df = df[df["avg_points"].notna()]

    grouped = (
        df.groupby(["vs_team", "primary_position"])["avg_points"]
          .agg(["mean", "size"])
          .reset_index()
          .rename(columns={
              "vs_team": "opponent_team",
              "mean": "expected_points_conceded",
              "size": "n_samples",
          })
    )
    log.info("Defensive strength: %d (team, position) cells from %d samples",
             len(grouped), int(df.shape[0]))
    return grouped


# ── Historical feature builder ────────────────────────────────────────────────

def build_prediction_features(df_2026: pd.DataFrame,
                               df_historical: pd.DataFrame,
                               fixtures: pd.DataFrame | None = None,
                               current_round: int | None = None) -> pd.DataFrame:
    """
    Build a prediction-ready feature set for 2026 players by joining each
    player to their most recent historical season stats.

    - For players with 2025 data: use 2025 stats + 2026 price
    - For players with only 2024 data: use 2024 stats + 2026 price
    - For rookies (no history): fill stat columns with position-group medians

    When `fixtures` and `current_round` are supplied, this also overlays the
    upcoming opponent for each player (NN-4 fixture wiring): the `vs_team`
    column is overwritten with the next-round opponent, which downstream
    feeds into the `def_ppm_conceded` feature so it reflects the upcoming
    matchup rather than whatever historical row was carried forward.
    `is_home_next` is set on the same path. When fixtures aren't provided,
    behaviour is unchanged from before — the historical opponent is kept
    and `is_home_next` defaults to 0.

    Returns a DataFrame with the same schema as df_historical, representing
    expected 2026 performance. Used as input to the model at prediction time.
    """
    # Take the season-total row per player (most recent year, then highest
    # games_played, then highest total_points as tiebreaker).
    # Historical data may contain per-round snapshots — we need end-of-season totals
    # to match what the model was trained on.
    df_hist = df_historical.copy()
    for col in ["games_played", "total_points"]:
        if col in df_hist.columns:
            df_hist[col] = pd.to_numeric(df_hist[col], errors="coerce").fillna(0)
    sort_cols = []
    sort_asc = []
    if "scrape_year" in df_hist.columns:
        sort_cols.append("scrape_year")
        sort_asc.append(False)
    if "games_played" in df_hist.columns:
        sort_cols.append("games_played")
        sort_asc.append(False)
    if "total_points" in df_hist.columns:
        sort_cols.append("total_points")
        sort_asc.append(False)
    if sort_cols:
        df_hist = (df_hist.sort_values(sort_cols, ascending=sort_asc)
                          .drop_duplicates(subset=["player_name"], keep="first"))
    else:
        df_hist = df_hist.drop_duplicates(subset=["player_name"], keep="last")

    df_hist_reset = df_hist.copy()  # already deduped, has player_name as a column

    # Position-group medians from historical data for rookies
    primary_pos_hist = df_hist_reset["positions"].str.split("|").str[0].fillna("UNK")
    df_hist_reset["_pp"] = primary_pos_hist
    numeric_hist = df_hist_reset.select_dtypes(include=[np.number]).columns.tolist()
    pos_medians = df_hist_reset.groupby("_pp")[numeric_hist].median()
    pos_medians_overall = df_hist_reset[numeric_hist].mean()
    df_hist_reset = df_hist_reset.drop(columns=["_pp"])

    # ── Build the 2026 side: just metadata we want to overlay ────────────────
    # Vectorised version of the per-row loop. Left-merging with the history
    # gives us one row per 2026 player; rookies have NaN in historical
    # columns and get filled from position-group medians.
    meta = df_2026[["player_name"]].copy()
    meta["player_name"] = meta["player_name"].astype(str).str.strip()
    meta = meta[meta["player_name"] != ""].reset_index(drop=True)

    pos_2026  = df_2026["positions"] if "positions" in df_2026.columns else pd.Series("", index=df_2026.index)
    team_2026 = df_2026["team"]      if "team" in df_2026.columns      else pd.Series("", index=df_2026.index)
    price_pref = df_2026.get("price_usd")
    price_fallback = df_2026.get("price")
    if price_pref is None and price_fallback is None:
        price_2026 = pd.Series(np.nan, index=df_2026.index)
    elif price_pref is None:
        price_2026 = price_fallback
    elif price_fallback is None:
        price_2026 = price_pref
    else:
        price_2026 = price_pref.fillna(price_fallback)

    meta["_pos_2026"]   = pos_2026.reindex(meta.index).fillna("").astype(str)
    meta["_team_2026"]  = team_2026.reindex(meta.index).fillna("").astype(str)
    meta["_price_2026"] = pd.to_numeric(price_2026.reindex(meta.index), errors="coerce")

    # Left-merge against history; rookies get NaN for historical columns.
    merged = meta.merge(df_hist_reset, on="player_name", how="left",
                        suffixes=("", "_hist"))

    # Pick a marker column to detect rookies. Prefer "scrape_year" (always
    # present in historical rows); fall back to any non-name column.
    marker_col = "scrape_year" if "scrape_year" in df_hist_reset.columns else next(
        (c for c in df_hist_reset.columns if c != "player_name"), None
    )
    if marker_col is None:
        is_rookie = pd.Series(False, index=merged.index)
    else:
        is_rookie = merged[marker_col].isna()

    # Fill rookie rows with position-group medians
    if is_rookie.any():
        rookie_pos = (
            merged["_pos_2026"].astype(str).str.split("|").str[0]
            .replace("", "UNK")
        )
        for col in pos_medians.columns:
            median_for_row = rookie_pos.map(pos_medians[col]).fillna(
                pos_medians_overall[col]
            )
            merged.loc[is_rookie, col] = median_for_row[is_rookie].astype(
                merged[col].dtype if not merged[col].isna().all() else float,
                errors="ignore",
            )

    # Overlay 2026 metadata where it's truthy/present (matches original semantics).
    mask_pos   = merged["_pos_2026"].str.strip() != ""
    mask_team  = merged["_team_2026"].str.strip() != ""
    mask_price = merged["_price_2026"].notna()
    if "positions" in merged.columns:
        merged.loc[mask_pos, "positions"] = merged.loc[mask_pos, "_pos_2026"]
    else:
        merged["positions"] = merged["_pos_2026"]
    if "team" in merged.columns:
        merged.loc[mask_team, "team"] = merged.loc[mask_team, "_team_2026"]
    else:
        merged["team"] = merged["_team_2026"]
    if "price" in merged.columns:
        merged.loc[mask_price, "price"] = merged.loc[mask_price, "_price_2026"]
    else:
        merged["price"] = merged["_price_2026"]
    merged["scrape_year"] = 2026

    merged = merged.drop(columns=["_pos_2026", "_team_2026", "_price_2026"])

    # ── Overlay current-season form columns (NN-2 freshness fix) ─────────────
    # The delta-target predictor adds back avg_last3_raw at predict time; if
    # we left it as the historical-year value, mid-season predictions would
    # baseline off LAST year's recent form rather than this season's. Take
    # 2026 form values whenever the player has played enough games to have
    # them populated — otherwise keep the historical fallback.
    FORM_COLUMNS = (
        "avg_last3", "avg_last5", "avg_last2",
        "avg_last3_raw", "avg_points", "avg_points_raw",
        "form_momentum", "games_played",
        "avg_minutes", "avg_mins_last3", "avg_mins_last5",
        "points_per_minute", "break_even",
        "season_price_change", "round_price_change",
    )
    df_2026_indexed = df_2026.drop_duplicates(subset=["player_name"], keep="last")
    df_2026_indexed = df_2026_indexed.set_index("player_name")
    for col in FORM_COLUMNS:
        if col not in df_2026_indexed.columns:
            continue
        fresh = pd.to_numeric(
            merged["player_name"].map(df_2026_indexed[col]),
            errors="coerce",
        )
        if col not in merged.columns:
            merged[col] = fresh
        else:
            # Only overlay where the 2026 value is present AND non-zero
            # (zero is what the scraper writes pre-game, i.e. no signal yet).
            mask_fresh = fresh.notna() & (fresh != 0)
            merged.loc[mask_fresh, col] = fresh[mask_fresh]

    # ── Next-opponent overlay (NN-4 fixture wiring) ─────────────────────────
    # When a fixture frame and a current-round number are supplied, replace
    # each player's `vs_team` with their upcoming opponent and set the
    # `is_home_next` flag. `engineer_features` then turns this into the
    # `def_ppm_conceded` feature via the position_vs_team CSV. Players on a
    # bye get vs_team cleared so the feature falls back to the league mean.
    if fixtures is not None and not fixtures.empty and current_round is not None:
        from scraper import get_next_opponent
        team_series = (
            merged["team"].astype(str).str.upper().str.strip()
            if "team" in merged.columns else pd.Series("", index=merged.index)
        )
        next_opps, home_flags = [], []
        for team in team_series:
            opp, is_home = get_next_opponent(team, current_round, fixtures)
            next_opps.append(opp if opp else "")
            home_flags.append(1.0 if is_home else 0.0)
        merged["vs_team"] = next_opps
        merged["is_home_next"] = home_flags
        n_matched = sum(1 for n in next_opps if n)
        log.info("Fixture wiring: %d/%d players have a round-%d opponent",
                 n_matched, len(merged), current_round)
    else:
        # Default — column is required by engineer_features.
        if "is_home_next" not in merged.columns:
            merged["is_home_next"] = 0.0

    # TODO(NN-3): replace the per-position one-hot columns with a learned
    # position embedding. Refactor `_build_model` to use the Functional
    # API: numeric feature input + position id input → embedding(7, 4) →
    # concat → dense head. Drop pos_* columns from SCALE_COLS.
    # TODO(NN-5): switch the regression head to predict (mean, log_var)
    # and use a Gaussian NLL loss. Pass the resulting std into the
    # optimizer so it can prefer higher-variance picks for differentials.

    if merged.empty:
        log.warning("build_prediction_features: no rows produced")
        return df_2026.copy()

    n_with_hist = int((~is_rookie).sum())
    n_rookies = int(is_rookie.sum())
    log.info("build_prediction_features: %d players (%d with history, %d rookies)",
             len(merged), n_with_hist, n_rookies)
    return merged.reset_index(drop=True)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_next_round_scores(df: pd.DataFrame,
                               df_historical: pd.DataFrame = None,
                               fixtures: pd.DataFrame | None = None,
                               current_round: int | None = None) -> pd.DataFrame:
    """
    Add a 'predicted_points' column to df using the saved model.

    Pre-season path (avg_points == 0 for all 2026 players):
      - If df_historical is provided: build features from historical stats,
        run model inference, map predictions back to df by player_name.
      - Fallback: price / 10000.

    In-season path: use the model directly on df's features.
    Cold-start: zero/negative predictions → position-group median.

    fixtures / current_round (optional): when both are supplied, the
    pre-season path overlays the round-`current_round` opponent onto each
    player so the `def_ppm_conceded` feature reflects the upcoming matchup.
    Falls back silently to "no fixture wiring" if either is missing.
    """
    from tensorflow import keras

    df = df.copy()
    # Use raw (unscaled) avg_points to detect pre-season; the scaled version is never 0
    if "avg_points_raw" in df.columns:
        avg_pts = df["avg_points_raw"].fillna(0)
    else:
        avg_pts = df.get("avg_points", pd.Series(0, index=df.index)).fillna(0)
    pre_season = avg_pts.max() == 0

    # ── Pre-season with historical data ──────────────────────────────────────
    if pre_season and df_historical is not None and not df_historical.empty:
        hist_with_scores = df_historical[
            pd.to_numeric(df_historical.get("avg_points", pd.Series(0)), errors="coerce").fillna(0) > 0
        ] if "avg_points" in df_historical.columns else pd.DataFrame()

        if not hist_with_scores.empty and MODEL_PATH.exists():
            log.info("Pre-season mode: using 2024/2025 history + trained model for predictions")
            df_feat_pred = build_prediction_features(
                df, hist_with_scores,
                fixtures=fixtures, current_round=current_round,
            )
            df_feat_pred_clean = clean_data(df_feat_pred)  # type: ignore
            df_feat_pred_eng, _ = engineer_features(       # type: ignore
                df_feat_pred_clean, fit_scaler=False,
                scaler=_load_scaler()
            )
            model = keras.models.load_model(MODEL_PATH)
            feature_cols = get_feature_cols(df_feat_pred_eng)
            feature_cols = [c for c in feature_cols if c in df_feat_pred_eng.columns]
            X = df_feat_pred_eng[feature_cols].fillna(0).values.astype(np.float32)
            preds = model.predict(X, verbose=0).flatten().clip(min=0)

            # Map predictions back to the original df by player_name
            pred_map = dict(zip(df_feat_pred_eng["player_name"].values, preds))
            df["predicted_points"] = df["player_name"].map(pred_map).fillna(0)

            # Rookies (no mapping): fill with position-group median
            primary_pos = df["positions"].str.split("|").str[0].fillna("UNK")
            df["_pp"] = primary_pos
            pos_med = df.groupby("_pp")["predicted_points"].transform("median")
            global_med = df["predicted_points"].median()
            mask = df["predicted_points"] <= 0
            df.loc[mask, "predicted_points"] = pos_med[mask].fillna(global_med)
            df = df.drop(columns=["_pp"], errors="ignore")

            # ── Confidence shrinkage: pull low-data players toward position median ──
            MIN_GAMES_FULL_CONFIDENCE = 15
            hist_gp = df_historical.copy()
            hist_gp["games_played"] = pd.to_numeric(
                hist_gp["games_played"], errors="coerce"
            ).fillna(0)
            games_lookup = hist_gp.groupby("player_name")["games_played"].max()
            player_games = df["player_name"].map(games_lookup).fillna(0)

            confidence = (player_games / MIN_GAMES_FULL_CONFIDENCE).clip(0, 1)
            _pp = df["positions"].str.split("|").str[0].fillna("UNK")
            pos_median = (
                df.assign(_pp=_pp)
                .groupby("_pp")["predicted_points"]
                .transform("median")
            )
            df["predicted_points"] = (
                confidence * df["predicted_points"]
                + (1 - confidence) * pos_median
            )

            log.info("Confidence shrinkage applied (min_games=%d)", MIN_GAMES_FULL_CONFIDENCE)
            log.info("Predictions (history-based): min=%.1f  mean=%.1f  max=%.1f",
                     df["predicted_points"].min(),
                     df["predicted_points"].mean(),
                     df["predicted_points"].max())
            return df

    # ── No model yet ──────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        if pre_season and "price_usd" in df.columns:
            log.info("Pre-season fallback: using price * %.6f as performance proxy",
                     POINTS_PER_DOLLAR)
            df["predicted_points"] = (
                df["price_usd"].fillna(0) * POINTS_PER_DOLLAR
            ).clip(lower=0)
        else:
            log.warning("No saved model — using avg_points as fallback")
            df["predicted_points"] = avg_pts
        return df

    model = keras.models.load_model(MODEL_PATH)

    feature_cols = get_feature_cols(df)
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        log.warning("No feature columns available — using avg_points as fallback")
        df["predicted_points"] = df.get("avg_points", 0).fillna(0)
        return df

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    preds = model.predict(X, verbose=0).flatten()

    # NN-2: if we trained against a delta target, the model output is
    # (score - rolling_3_avg). Add the rolling baseline back to recover
    # absolute predicted points.
    if USE_DELTA_TARGET and "avg_last3_raw" in df.columns:
        baseline = df["avg_last3_raw"].fillna(0).values.astype(np.float32)
        preds = preds + baseline
    df["predicted_points"] = preds.clip(min=0)

    # Cold-start: replace zero/negative predictions with position-group median
    primary_pos = df["positions"].str.split("|").str[0].fillna("UNK")
    df["_primary_pos"] = primary_pos
    pos_medians = df.groupby("_primary_pos")["predicted_points"].transform("median")
    global_median = df["predicted_points"].median()

    cold_mask = df["predicted_points"] <= 0
    df.loc[cold_mask, "predicted_points"] = pos_medians[cold_mask].fillna(global_median)
    df = df.drop(columns=["_primary_pos"], errors="ignore")

    log.info("Predictions: min=%.1f  mean=%.1f  max=%.1f",
             df["predicted_points"].min(),
             df["predicted_points"].mean(),
             df["predicted_points"].max())
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from scraper import scrape_full, update_historical_data

    parser = argparse.ArgumentParser(description="Train / evaluate prediction model")
    parser.add_argument("--data", default="data/processed/master_historical.csv",
                        help="Path to processed CSV")
    parser.add_argument("--retrain", action="store_true", help="Force full retrain")
    parser.add_argument(
        "--final", action="store_true",
        help="Train production model on ALL data (no year holdout). "
             "Validation metrics from this run are NOT honest — run a "
             "non-final pass first to get an honest generalisation estimate. "
             "Implies --retrain (always trains from scratch).",
    )
    args = parser.parse_args()
    if args.final:
        args.retrain = True  # final mode always retrains from scratch

    data_path = Path(args.data)
    if not data_path.exists():
        log.info("No processed data found — running scraper first ...")
        df_raw = scrape_full(year=2026, save=True)
        df = update_historical_data(df_raw)
    else:
        df = pd.read_csv(data_path, low_memory=False)
        log.info("Loaded %d rows from %s", len(df), data_path)

    df_clean = clean_data(df)
    df_feat, scaler = engineer_features(df_clean, fit_scaler=True)

    model = load_or_train_model(df_feat, force_retrain=args.retrain,
                                final_retrain=args.final)

    if model is not None:
        df_pred = predict_next_round_scores(df_feat)
        print("\nTop 15 predicted scorers:")
        cols = ["player_name", "positions", "team", "price",
                "avg_points", "avg_last3", "predicted_points"]
        cols = [c for c in cols if c in df_pred.columns]
        print(df_pred.sort_values("predicted_points", ascending=False)[cols]
              .head(15).to_string(index=False))

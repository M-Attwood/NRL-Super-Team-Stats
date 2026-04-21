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

# ── NRL positions and teams for one-hot encoding ─────────────────────────────
ALL_POSITIONS = ["HOK", "FRF", "2RF", "HFB", "5/8", "CTW", "FLB"]
ALL_TEAMS = [
    "BRI", "CBY", "CRO", "DOL", "EEL", "GLD", "HUR", "MAN",
    "MEL", "NEW", "NQL", "PEN", "RDB", "SOU", "STI", "WST",
]

# Numeric columns to scale
SCALE_COLS = [
    "price", "avg_points", "avg_last3", "avg_last5", "avg_last2",
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
]

# Target column
TARGET = "avg_points"


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
        except Exception:
            pass

    # Fill NaN with position-group median
    primary_pos = df["positions"].str.split("|").str[0].fillna("UNK")
    df["_primary_pos"] = primary_pos

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            group_medians = df.groupby("_primary_pos")[col].transform("median")
            global_median = df[col].median()
            df[col] = df[col].fillna(group_medians).fillna(global_median).fillna(0)

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
    # Save raw avg_points so load_or_train_model can use unscaled targets
    if "avg_points" in df.columns:
        df["avg_points_raw"] = df["avg_points"].copy()

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

    log.info("engineer_features: %d features, %d rows", len(feature_cols), len(df))
    return df, scaler


def _load_scaler() -> StandardScaler | None:
    """Load the saved scaler, or return None if not yet fitted."""
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            return pickle.load(f)
    return None


def get_feature_cols(df: pd.DataFrame = None) -> list[str]:
    """Load saved feature column list, or derive from df."""
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH, "rb") as f:
            return pickle.load(f)
    if df is not None:
        pos_cols = [f"pos_{p.replace('/', '_').replace(' ', '_')}" for p in ALL_POSITIONS]
        team_cols = [f"team_{t}" for t in ALL_TEAMS]
        present = [c for c in SCALE_COLS if c in df.columns]
        return present + pos_cols + team_cols
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

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def load_or_train_model(df: pd.DataFrame, force_retrain: bool = False):
    """
    Load an existing saved model and fine-tune it, or train from scratch.
    Returns the trained Keras model.
    """
    import tensorflow as tf
    from tensorflow import keras

    feature_cols = get_feature_cols(df)
    if not feature_cols:
        raise ValueError("No feature columns found. Run engineer_features first.")

    # Filter to columns present in df
    feature_cols = [c for c in feature_cols if c in df.columns]
    n_features = len(feature_cols)

    if TARGET not in df.columns:
        log.warning("Target column '%s' not in df — cannot train", TARGET)
        return None

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    # Use raw (unscaled) avg_points as target so predictions are in original units
    target_col = "avg_points_raw" if "avg_points_raw" in df.columns else TARGET
    y = df[target_col].fillna(0).values.astype(np.float32)

    # Remove rows where y is 0 (players with no history)
    mask = y > 0
    X, y = X[mask], y[mask]

    if len(X) < 20:
        log.warning("Not enough training samples (%d) — skipping model training", len(X))
        return None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
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

    if MODEL_PATH.exists() and not force_retrain:
        log.info("Loading existing model from %s for fine-tuning ...", MODEL_PATH)
        model = keras.models.load_model(MODEL_PATH)
        # Lower learning rate for fine-tuning (Keras 3 compatible)
        model.optimizer.learning_rate = 0.0001
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

    return model


# ── Historical feature builder ────────────────────────────────────────────────

def build_prediction_features(df_2026: pd.DataFrame,
                               df_historical: pd.DataFrame) -> pd.DataFrame:
    """
    Build a prediction-ready feature set for 2026 players by joining each
    player to their most recent historical season stats.

    - For players with 2025 data: use 2025 stats + 2026 price
    - For players with only 2024 data: use 2024 stats + 2026 price
    - For rookies (no history): fill stat columns with position-group medians

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

    df_hist = df_hist.set_index("player_name")

    # Build output rows — one per 2026 player
    result_rows = []
    # Position-group medians from historical data for rookies
    hist_reset = df_hist.reset_index()
    primary_pos = hist_reset["positions"].str.split("|").str[0].fillna("UNK")
    hist_reset["_pp"] = primary_pos
    numeric_hist = hist_reset.select_dtypes(include=[np.number]).columns.tolist()
    pos_medians = hist_reset.groupby("_pp")[numeric_hist].median()

    for _, row_2026 in df_2026.iterrows():
        name = row_2026.get("player_name", "")
        if not name or str(name).strip() == "":
            continue

        # Use price_usd (pre-scaling raw $) if available; otherwise fall back to price
        # This handles both raw and already-engineered 2026 DataFrames.
        price_raw = row_2026.get("price_usd")
        if price_raw is None or (isinstance(price_raw, float) and np.isnan(price_raw)):
            price_raw = row_2026.get("price")
        pos_2026  = row_2026.get("positions", "")
        team_2026 = row_2026.get("team", "")

        if name in df_hist.index:
            # Start from raw historical stats; only overwrite 2026-specific metadata
            hist_row = df_hist.loc[name].to_dict()
            if pd.notna(price_raw):
                hist_row["price"] = price_raw
            hist_row["player_name"] = name
            hist_row["positions"]   = pos_2026 if pos_2026 else hist_row.get("positions", "")
            hist_row["team"]        = team_2026 if team_2026 else hist_row.get("team", "")
            hist_row["scrape_year"] = 2026
            result_rows.append(hist_row)
        else:
            # Rookie — no historical data; build row from position-group medians
            pp = str(pos_2026).split("|")[0] or "UNK"
            if pp in pos_medians.index:
                rookie_row = pos_medians.loc[pp].to_dict()
            else:
                rookie_row = pos_medians.mean().to_dict()
            if pd.notna(price_raw):
                rookie_row["price"] = price_raw
            rookie_row["player_name"] = name
            rookie_row["positions"]   = pos_2026
            rookie_row["team"]        = team_2026
            rookie_row["scrape_year"] = 2026
            result_rows.append(rookie_row)

    if not result_rows:
        log.warning("build_prediction_features: no rows produced")
        return df_2026.copy()

    df_pred = pd.DataFrame(result_rows)
    # Ensure player_name is first and is a column
    if "player_name" not in df_pred.columns and df_pred.index.name == "player_name":
        df_pred = df_pred.reset_index()

    log.info("build_prediction_features: %d players (%d with history, %d rookies)",
             len(df_pred),
             sum(p.get("player_name", "") in df_hist.index for p in result_rows),
             sum(p.get("player_name", "") not in df_hist.index for p in result_rows))
    return df_pred.reset_index(drop=True)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_next_round_scores(df: pd.DataFrame,
                               df_historical: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add a 'predicted_points' column to df using the saved model.

    Pre-season path (avg_points == 0 for all 2026 players):
      - If df_historical is provided: build features from historical stats,
        run model inference, map predictions back to df by player_name.
      - Fallback: price / 10000.

    In-season path: use the model directly on df's features.
    Cold-start: zero/negative predictions → position-group median.
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
            df_feat_pred = build_prediction_features(df, hist_with_scores)
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
            log.info("Pre-season fallback: using price / 10000 as performance proxy")
            df["predicted_points"] = (df["price_usd"].fillna(0) / 10000).clip(lower=0)
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
    args = parser.parse_args()

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

    model = load_or_train_model(df_feat, force_retrain=args.retrain)

    if model is not None:
        df_pred = predict_next_round_scores(df_feat)
        print("\nTop 15 predicted scorers:")
        cols = ["player_name", "positions", "team", "price",
                "avg_points", "avg_last3", "predicted_points"]
        cols = [c for c in cols if c in df_pred.columns]
        print(df_pred.sort_values("predicted_points", ascending=False)[cols]
              .head(15).to_string(index=False))

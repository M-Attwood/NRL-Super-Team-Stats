"""
main.py — NRL Supercoach Optimizer pipeline

Workflow each round:
  1. Scrape 2026 player data
  1b. Load or scrape 2022-2025 historical data (filtered to 2026-active players)
  2. Merge all data into master historical CSV
  3. Clean and engineer features
  4. Train model on 2022-2025 data (rows where avg_points > 0)
  5. Predict 2026 scores using historical stats as feature basis
  6. Run the PuLP optimizer
  7. Print and export the optimal squad

Usage:
    python main.py                          # standard weekly update
    python main.py --historical             # first-run: bootstrap player page history
    python main.py --no-scrape             # skip 2026 scraping, use existing data
    python main.py --rescrape-history      # force re-scrape of 2022-2025 even if cached
    python main.py --retrain               # force full model retrain from scratch
    python main.py --round 5              # manually set round number
    python main.py --year 2026            # season year (default: 2026)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Historical seasons used for model training. The 2026 (current) season is
# scraped separately and merged in. Going back to 2022 gives ~4 years of data,
# which is plenty for the position-conditioned features we build.
_HIST_START_YEAR = 2022


def _historical_years(current_season: int) -> list[int]:
    """All complete seasons before the current one, going back to _HIST_START_YEAR."""
    return list(range(_HIST_START_YEAR, current_season))


# Default historical year list, derived from today's date. Override per call
# if you need a frozen window for reproducibility.
HIST_YEARS = _historical_years(datetime.now().year)


# ── Directory setup ───────────────────────────────────────────────────────────

def create_dirs():
    from paths import (
        DATA_RAW, DATA_ROUNDS, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR,
    )
    for d in (DATA_RAW, DATA_ROUNDS, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── Round number detection ────────────────────────────────────────────────────

def detect_round_number(override: int = None) -> int:
    if override is not None:
        return override
    existing = list(Path("outputs").glob("team_round_*.csv"))
    return len(existing) + 1


# ── Pipeline steps ────────────────────────────────────────────────────────────

def run_scraper(year: int, historical: bool) -> pd.DataFrame:
    from scraper import scrape_full
    log.info("── Step 1: Scraping nrlsupercoachstats.com (year=%d) ──", year)
    df_new = scrape_full(year=year, historical=historical, save=True)
    if df_new.empty:
        log.error("Scraper returned no data — aborting pipeline")
        sys.exit(1)
    log.info("Scraped %d players", len(df_new))
    return df_new


def load_existing_data(path: str | None) -> pd.DataFrame:
    """Load a cached scrape. If path is None, falls back to the most
    recent nrl_data_*.csv in DATA_RAW so the script keeps working as the
    season progresses without manually bumping a default date."""
    from paths import DATA_RAW
    if path is None:
        candidates = sorted(DATA_RAW.glob("nrl_data_*.csv"), reverse=True)
        if not candidates:
            log.error("No cached data in %s — run without --no-scrape first",
                      DATA_RAW)
            sys.exit(1)
        p = candidates[0]
    else:
        p = Path(path)
        if not p.exists():
            log.error("Data file not found: %s — run without --no-scrape first", p)
            sys.exit(1)
    df = pd.read_csv(p, low_memory=False)
    log.info("Loaded %d rows from %s", len(df), p)
    return df


def load_or_scrape_historical(df_2026: pd.DataFrame,
                               years: list = None,
                               force_rescrape: bool = False) -> pd.DataFrame:
    """
    Load cached historical season CSVs (data/raw/nrl_data_{year}.csv).
    If a year's file is missing or force_rescrape=True, scrape it fresh.
    Only returns rows for players currently in the 2026 active roster.
    """
    from scraper import scrape_historical_seasons

    if years is None:
        years = HIST_YEARS

    active_players = df_2026["player_name"].dropna().tolist() \
        if "player_name" in df_2026.columns else []

    years_to_scrape = []
    frames = []

    from paths import DATA_RAW
    for yr in years:
        raw_path = DATA_RAW / f"nrl_data_{yr}.csv"
        if raw_path.exists() and not force_rescrape:
            log.info("── Step 1b: Loading cached %d data from %s ──", yr, raw_path)
            df_yr = pd.read_csv(raw_path, low_memory=False)
            # Cached files have per-round snapshots; keep only the
            # season-total row per player. Use total_points as tiebreaker
            # (cumulative > any single round when games_played ties).
            if "player_name" in df_yr.columns and "games_played" in df_yr.columns:
                df_yr["games_played"] = pd.to_numeric(
                    df_yr["games_played"], errors="coerce"
                ).fillna(0)
                if "total_points" in df_yr.columns:
                    df_yr["total_points"] = pd.to_numeric(
                        df_yr["total_points"], errors="coerce"
                    ).fillna(0)
                before_dedup = len(df_yr)
                sort_cols = ["games_played"] + (
                    ["total_points"] if "total_points" in df_yr.columns else []
                )
                df_yr = (df_yr.sort_values(sort_cols, ascending=False)
                              .drop_duplicates(subset=["player_name"], keep="first"))
                log.info("Year %d: %d → %d after per-round dedup",
                         yr, before_dedup, len(df_yr))
            # Re-apply 2026-active filter in case the file pre-dates roster changes
            if active_players and "player_name" in df_yr.columns:
                before = len(df_yr)
                df_yr = df_yr[df_yr["player_name"].isin(set(active_players))].copy()
                log.info("Year %d: %d → %d (2026-active players)", yr, before, len(df_yr))
            frames.append(df_yr)
        else:
            years_to_scrape.append(yr)

    if years_to_scrape:
        log.info("── Step 1b: Scraping historical seasons %s ──", years_to_scrape)
        df_scraped = scrape_historical_seasons(
            active_players=active_players,
            years=years_to_scrape,
            save=True,
        )
        if not df_scraped.empty:
            frames.append(df_scraped)

    if not frames:
        log.warning("No historical data available for years %s", years)
        return pd.DataFrame()

    df_hist = pd.concat(frames, ignore_index=True)
    log.info("Historical data loaded: %d total rows (years %s)", len(df_hist), years)
    return df_hist


def run_feature_engineering(df: pd.DataFrame) -> tuple:
    from model import clean_data, engineer_features
    log.info("── Step 3: Cleaning and engineering features ──")
    df_clean = clean_data(df)
    df_feat, scaler = engineer_features(df_clean, fit_scaler=True)
    return df_feat, scaler


def run_model(df_feat: pd.DataFrame, retrain: bool,
              holdout_year: int | None = None):
    from model import load_or_train_model
    log.info("── Step 4: Training model on historical data ──")
    model = load_or_train_model(df_feat, force_retrain=retrain,
                                holdout_year=holdout_year)
    if model is None:
        log.warning("Model training skipped (not enough data with avg_points > 0).")
    return model


def run_predictions(df_feat: pd.DataFrame,
                    df_historical: pd.DataFrame = None,
                    fixtures: pd.DataFrame | None = None,
                    current_round: int | None = None) -> pd.DataFrame:
    from model import predict_next_round_scores
    log.info("── Step 5: Predicting next-round scores ──")
    # Filter df_feat to the current season (2026) players only for the final output
    if "scrape_year" in df_feat.columns:
        df_2026_feat = df_feat[df_feat["scrape_year"] == 2026].copy()
    else:
        df_2026_feat = df_feat.copy()
    df_pred = predict_next_round_scores(
        df_2026_feat, df_historical=df_historical,
        fixtures=fixtures, current_round=current_round,
    )
    return df_pred


def load_fixtures(year: int, no_scrape: bool) -> pd.DataFrame:
    """Load this season's fixtures from disk; scrape if missing.

    Returns an empty DataFrame if neither path produces data — the model
    degrades gracefully (def_ppm_conceded falls back to last-played
    opponent / league mean).
    """
    from scraper import scrape_fixtures
    try:
        return scrape_fixtures(year=year, save=not no_scrape)
    except (RuntimeError, OSError) as e:
        log.warning("Fixture load failed: %s — continuing without fixtures.", e)
        return pd.DataFrame()


def run_optimizer(df_pred: pd.DataFrame, round_num: int) -> dict:
    from optimizer import select_team, print_team, export_team
    log.info("── Step 6: Running optimizer (Round %d) ──", round_num)
    result = select_team(df_pred, round_number=round_num)
    print_team(result)
    out_path = export_team(result, round_num)
    log.info("Squad exported → %s", out_path)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NRL Supercoach Optimizer — weekly pipeline"
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--historical", action="store_true",
                        help="Bootstrap full player page history (slow, first run only)")
    parser.add_argument("--no-scrape", action="store_true",
                        help="Skip 2026 scraping; load from --data")
    parser.add_argument("--rescrape-history", action="store_true",
                        help="Force re-scrape of 2024/2025 data even if cached")
    parser.add_argument("--retrain", action="store_true",
                        help="Force full model retrain from scratch")
    parser.add_argument("--holdout-year", type=int, default=None,
                        help="Year to reserve for validation (default: most "
                             "recent year in data). Train uses years strictly "
                             "older than this.")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--plan", action="store_true",
                        help="Run season-long bye round planner after optimizer")
    parser.add_argument("--plan-from-round", type=int, default=1,
                        help="Start season plan from this round (default: 1)")
    parser.add_argument("--use-squad-state", action="store_true",
                        help="Seed planner from data/inputs/round_{N}.yaml "
                             "(actual squad + trades remaining) instead of "
                             "building a fresh ideal squad.")
    parser.add_argument("--data", default=None,
                        help="CSV to use when --no-scrape is set "
                             "(default: most recent in data/raw/)")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  NRL SUPERCOACH OPTIMIZER  |  Season %d", args.year)
    log.info("=" * 55)

    create_dirs()
    round_num = detect_round_number(override=args.round)
    log.info("Round: %d", round_num)

    # ── Step 1: 2026 current season data ─────────────────────────────────────
    if args.no_scrape:
        df_2026 = load_existing_data(args.data)
    else:
        df_2026 = run_scraper(year=args.year, historical=args.historical)

    # ── Step 1b: 2024 + 2025 historical data (filtered to 2026 active players) ─
    df_hist_raw = load_or_scrape_historical(
        df_2026,
        years=HIST_YEARS,
        force_rescrape=args.rescrape_history,
    )

    # ── Step 2: Merge all into master historical CSV ──────────────────────────
    from scraper import update_historical_data
    if not df_hist_raw.empty:
        df_all = update_historical_data(
            pd.concat([df_2026, df_hist_raw], ignore_index=True)
        )
    else:
        df_all = update_historical_data(df_2026)
    log.info("Master historical CSV: %d total rows", len(df_all))

    # ── Step 3: Feature engineering on ALL data (for model training) ─────────
    df_feat, _ = run_feature_engineering(df_all)

    # ── Step 4: Train model on 2024+2025 rows (avg_points > 0) ───────────────
    run_model(df_feat, retrain=args.retrain, holdout_year=args.holdout_year)

    # ── Step 4b: Load fixtures (best-effort) for next-opponent feature ───────
    # build_prediction_features uses these to overlay vs_team with the
    # round-N opponent so def_ppm_conceded reflects the upcoming matchup
    # instead of whatever historical opponent the row carried.
    fixtures = load_fixtures(args.year, no_scrape=args.no_scrape)

    # ── Step 5: Predict 2026 scores using historical feature lookup ───────────
    # Pass the raw historical data (before feature engineering) so that
    # build_prediction_features can join 2025/2024 stats to 2026 players.
    df_pred = run_predictions(df_feat, df_historical=df_hist_raw,
                              fixtures=fixtures, current_round=round_num)

    # ── Step 6: Optimise and output ───────────────────────────────────────────
    result = run_optimizer(df_pred, round_num)

    if result.get("solver_status") == "Optimal":
        log.info("Pipeline complete. Squad saved for Round %d.", round_num)
    else:
        log.warning("Pipeline completed but optimizer status: %s",
                    result.get("solver_status"))

    # ── Step 7: Season planner (optional) ───────────────────────────────────
    season_state = None
    if args.plan:
        from planner import run_season_plan, print_season_summary, export_season_plan
        log.info("── Step 7: Running season-long bye round planner ──")

        squad_state = None
        origin_wl = None
        if args.use_squad_state:
            from squad_state import load_state, load_origin_watchlist
            try:
                squad_state = load_state(round_num, df_pred)
                log.info("Squad state loaded: R%d, %d players, %d trades left",
                         squad_state["current_round"], len(squad_state["squad"]),
                         squad_state["trades_remaining"])
            except Exception as e:
                log.error("Failed to load squad state for R%d: %s", round_num, e)
                log.error("Falling back to fresh-squad planner.")
            origin_wl = load_origin_watchlist()

        season_state = run_season_plan(
            df_pred,
            start_round=args.plan_from_round,
            squad_state=squad_state,
            origin_watchlist=origin_wl,
        )
        print_season_summary(season_state)
        export_season_plan(season_state)
        log.info("Season plan complete.")

    # ── Step 8: Visualisation reports ─────────────────────────────────────────
    try:
        from visualise import run_visualisation
        log.info("── Step 8: Generating visualisation reports ──")
        run_visualisation(
            df_feat=df_feat,
            df_pred=df_pred,
            round_num=round_num,
            season_state=season_state,
        )
    except Exception as e:
        log.warning("Visualisation failed (non-fatal): %s", e)


if __name__ == "__main__":
    main()

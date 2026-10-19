"""
main.py — weekly NRL Supercoach pipeline.

  1. Scrape the current season (earlier seasons load from data/raw/ cache)
  2. Build next-game training pairs; evaluate + train the model if needed
  3. Predict every player's score for the upcoming round
  4. Solve the squad linear program -> outputs/team_round_<N>.csv
  5. Optionally plan byes and trades for the rest of the season

Usage:
    python main.py                  # scrape, predict the next round, pick a squad
    python main.py --no-scrape      # use cached data
    python main.py --round 10       # replay a round using only data known before it
    python main.py --retrain        # re-evaluate and retrain the model first
    python main.py --plan           # also run the season bye/trade planner

After round 27 the season is over: run `python season_review.py` instead.
"""

import argparse
import logging
import sys

import pandas as pd

from paths import OUTPUTS_DIR

log = logging.getLogger(__name__)

FIRST_SEASON = 2022    # first season of history used for training
TOTAL_ROUNDS = 27


def load_data(year: int, no_scrape: bool = False) -> tuple[pd.DataFrame, dict]:
    """All per-round rows and season totals from FIRST_SEASON to `year`."""
    from model import load_seasons
    from scraper import load_season

    seasons = {}
    for y in range(FIRST_SEASON, year + 1):
        refresh = (y == year) and not no_scrape   # only the live season changes
        seasons[y] = load_season(y, force_scrape=refresh)
    return load_seasons(seasons)


def ensure_model(rounds: pd.DataFrame, totals: dict, year: int, retrain: bool):
    """Load the saved model, or evaluate-then-train a fresh one."""
    from model import (MODEL_PATH, build_training_pairs, evaluate_holdout,
                       load_model, train_production_model)

    if MODEL_PATH.exists() and not retrain:
        return load_model()

    pairs = build_training_pairs(rounds, totals)
    test_year = year if (pairs["year"] == year).sum() >= 500 else year - 1
    metrics, _, _ = evaluate_holdout(pairs, test_year)
    return train_production_model(pairs, metrics, test_year)


def next_round_to_predict(rounds: pd.DataFrame, year: int) -> int:
    played = rounds.loc[rounds["year"] == year, "round"]
    return int(played.max()) + 1 if not played.empty else 1


def build_predictions(year: int, round_num: int | None = None,
                      no_scrape: bool = False, retrain: bool = False
                      ) -> tuple[pd.DataFrame, int]:
    """Scrape/load data, make sure a model exists, and predict a round.
    Returns (predictions, round_num)."""
    from model import build_prediction_frame, predict_next_round
    from scraper import load_fixtures

    rounds, totals = load_data(year, no_scrape)
    if round_num is None:
        round_num = next_round_to_predict(rounds, year)
    model = ensure_model(rounds, totals, year, retrain)

    frame = build_prediction_frame(rounds, totals, year, round_num, load_fixtures(year))
    return predict_next_round(frame, model), round_num


def export_predictions(df_pred: pd.DataFrame, round_num: int):
    cols = ["player_name", "team", "positions", "price_usd", "predicted_points",
            "avg_points", "games_played", "opp", "is_rookie", "bye_round"]
    path = OUTPUTS_DIR / f"predictions_round_{round_num}.csv"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (df_pred[cols].sort_values("predicted_points", ascending=False)
     .round(1).to_csv(path, index=False))
    log.info("Predictions exported -> %s", path)


def main():
    parser = argparse.ArgumentParser(description="NRL Supercoach weekly pipeline")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int, default=None,
                        help="Round to predict (default: next unplayed round)")
    parser.add_argument("--no-scrape", action="store_true",
                        help="Use cached data in data/raw/ instead of scraping")
    parser.add_argument("--retrain", action="store_true",
                        help="Re-run the holdout evaluation and retrain the model")
    parser.add_argument("--plan", action="store_true",
                        help="Run the season-long bye/trade planner after the optimizer")
    parser.add_argument("--use-squad-state", action="store_true",
                        help="Seed the planner from data/inputs/round_<N>.yaml "
                             "(your real squad + trades left)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log.info("NRL SUPERCOACH OPTIMIZER | season %d", args.year)

    df_pred, round_num = build_predictions(args.year, args.round,
                                           args.no_scrape, args.retrain)
    if round_num > TOTAL_ROUNDS:
        log.info("Round %d is past the end of the %d season - nothing left to "
                 "pick. Run `python season_review.py` for the season retrospective.",
                 round_num, args.year)
        return
    log.info("Predicting round %d", round_num)
    export_predictions(df_pred, round_num)

    from optimizer import export_team, print_team, select_team
    result = select_team(df_pred, round_number=round_num)
    print_team(result)
    if result.get("solver_status") != "Optimal":
        log.error("Optimizer status: %s", result.get("solver_status"))
        sys.exit(1)
    export_team(result, round_num)

    from visualise import plot_squad
    plot_squad(result, round_num, OUTPUTS_DIR / f"team_round_{round_num}.png")

    if args.plan:
        from planner import export_season_plan, print_season_summary, run_season_plan
        squad_state, origin_wl = None, None
        if args.use_squad_state:
            from squad_state import load_origin_watchlist, load_state
            squad_state = load_state(round_num, df_pred)
            origin_wl = load_origin_watchlist()
        state = run_season_plan(df_pred, start_round=round_num,
                                squad_state=squad_state, origin_watchlist=origin_wl)
        print_season_summary(state)
        export_season_plan(state)

        from visualise import plot_season_plan
        plot_season_plan(state, OUTPUTS_DIR / "season_plan.png")


if __name__ == "__main__":
    main()

"""
season_review.py — the end-of-season retrospective (the project's final run).

  1. Predictor test: train on 2022-2025 only, predict every game of 2026,
     and compare with the heuristics fantasy coaches use.
  2. Predictor comparison: every round, pick a fresh 26-man squad under the
     cap with each way of valuing players and score its best 18 on real
     results. Not how the game is played; it isolates the predictor.
  3. Full-season simulations under Supercoach rules (season_sim.py):
       - development seasons 2024 and 2025, where the trading strategy was
         designed and its few settings chosen (models trained on earlier years)
       - the test season 2026, never used for tuning, played with those settings and ranked
         against real coaches (data/inputs/human_benchmarks_<year>.yaml)
  4. Players and positions: predicted vs actual per player, position
     analysis, top players, team of the season, bargains.
  5. Retrain the production model on every season, ready for next year.

Outputs: outputs/season_review/*.csv + summary.json, docs/images/*.png, models/.

Usage:
    python season_review.py             # uses cached data in data/raw/
    python season_review.py --scrape    # refresh the season from the site first
"""

import argparse
import json
import logging
from dataclasses import replace

import pandas as pd
import yaml

import visualise as viz
from main import load_data
from model import (MODEL_LABEL, ORACLE_LABEL, TARGET, build_training_pairs,
                   evaluate_holdout, fit_price_baseline, primary_position,
                   train_predictor, train_production_model)
from optimizer import POINTS_ONLY, SALARY_CAP, select_team
from paths import DATA_INPUTS, OUTPUTS_DIR, PROJECT_ROOT
from player_outlook import evaluate_outlook, train_outlook
from season_sim import IMPROVED, ORIGINAL, best_lineup, prepare_season, simulate_season

log = logging.getLogger(__name__)

REVIEW_DIR = OUTPUTS_DIR / "season_review"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
DEV_SEASONS = (2024, 2025)

VALUATIONS = {                       # label -> per-round frame column
    MODEL_LABEL: "predicted_points",
    "Season average": "season_avg_value",
    "Price only": "price",
    "Hindsight-perfect": "score",
}
IMPROVED_LABEL = "Model + improved strategy"
ORIGINAL_LABEL = "Model + original strategy"
SIMULATIONS = [                      # (label, strategy, trading?)
    (IMPROVED_LABEL, IMPROVED, True),
    (ORIGINAL_LABEL, ORIGINAL, True),
    ("Season averages + improved strategy", replace(IMPROVED, value="season_avg"), True),
    ("Model squad, no trades", IMPROVED, False),
]


def squad_table(result: dict) -> pd.DataFrame:
    rows = []
    for group, role in [("starting_13", "Starting"), ("bench_4", "Bench"),
                        ("flex_1", "Flex"), ("reserves_8", "Reserve")]:
        for p in result.get(group, []):
            rows.append({**p, "role": role})
    return pd.DataFrame(rows)


def predictor_comparison(data: dict) -> pd.DataFrame:
    """Pick a fresh 26 every round with each valuation (players named that
    round, that round's prices, points-only objective) and score its best 18
    on real results. 27 independent picks per valuation."""
    rows = {}
    for rnd, rd in data.items():
        pool = rd.frame[rd.frame["named"]].reset_index()
        rows[rnd] = {}
        for label, col in VALUATIONS.items():
            squad = squad_table(select_team(pool.assign(predicted_points=pool[col]),
                                            round_number=rnd, apply_injuries=False,
                                            weights=POINTS_ONLY, team_penalty=0.0))
            squad = squad.set_index("player_name")
            rows[rnd][label] = float(squad.loc[best_lineup(squad, "score"), "score"].sum())
    weekly = pd.DataFrame(rows).T
    weekly.index.name = "round"
    return weekly


def team_of_the_season(data: dict, totals: pd.DataFrame) -> pd.DataFrame:
    """The best legal squad at start-of-season prices, fed actual points."""
    first = data[1].frame.reset_index()
    actual = totals.set_index("player_name")
    first["season_avg"] = first["player_name"].map(actual["avg_points"]).fillna(0)
    first["start_price"] = first["price"]
    total_points = first["player_name"].map(actual["total_points"]).fillna(0)
    return squad_table(select_team(first.assign(predicted_points=total_points),
                                   round_number=1, apply_injuries=False))


def replay(rounds, totals, pairs, year, predictor=None):
    """Prepare `year` with models trained only on earlier seasons, then play
    it with every strategy in SIMULATIONS."""
    train = pairs[pairs["year"] < year]
    predictor = predictor or train_predictor(train)
    inputs = prepare_season(rounds, totals, year, predictor, fit_price_baseline(train),
                            train_outlook(rounds, totals, year))
    sims = {label: simulate_season(inputs, strat, trading, label=label)
            for label, strat, trading in SIMULATIONS}
    return inputs, sims


def run_review(year: int, scrape: bool = False) -> dict:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rounds, totals = load_data(year, no_scrape=not scrape)
    pairs = build_training_pairs(rounds, totals)
    train_pairs = pairs[pairs["year"] < year]
    season_totals = totals[year]

    # 1. Predictor test: train < year, predict every game of `year`.
    metrics, test, predictor = evaluate_holdout(pairs, year)
    outlook_metrics = evaluate_outlook(rounds, totals, year)
    pd.DataFrame(metrics).T.round(4).to_csv(REVIEW_DIR / "model_benchmark.csv")
    keep = ["player_name", "team", "positions", "next_round", "opp", TARGET,
            *[c for c in test.columns if c.startswith("pred_")]]
    test[keep].round(2).to_csv(REVIEW_DIR / f"game_predictions_{year}.csv", index=False)
    viz.plot_model_benchmark(metrics, len(test), year, IMAGES_DIR / "model_benchmark.png",
                             MODEL_LABEL, ORACLE_LABEL)
    viz.plot_error_by_round(test, IMAGES_DIR / "error_by_round.png", MODEL_LABEL)
    viz.plot_calibration(test, IMAGES_DIR / "calibration.png", MODEL_LABEL)

    # 3a. Development seasons: where the strategy was designed and tuned.
    dev = {y: replay(rounds, totals, pairs, y)[1] for y in DEV_SEASONS}

    # 3b. The test season, never used for tuning.
    inputs, sims = replay(rounds, totals, pairs, year, predictor)
    best = sims[IMPROVED_LABEL]
    pd.DataFrame({k: s.rounds["points"] for k, s in sims.items()}).to_csv(
        REVIEW_DIR / "season_simulation_scores.csv")
    best.rounds.round(1).to_csv(REVIEW_DIR / "season_simulation_rounds.csv")
    best.trades.to_csv(REVIEW_DIR / "season_simulation_trades.csv", index=False)
    final = inputs.rounds[max(inputs.rounds)].frame.loc[
        best.final_squad, ["team", "positions", "price"]].reset_index()
    final["season_avg"] = final["player_name"].map(
        season_totals.set_index("player_name")["avg_points"])
    final.round(1).to_csv(REVIEW_DIR / "season_simulation_final_squad.csv", index=False)

    main_sims = {k: sims[k].rounds for k in (IMPROVED_LABEL, ORIGINAL_LABEL,
                                             "Model squad, no trades")}
    viz.plot_season_simulation(main_sims, IMPROVED_LABEL,
                               IMAGES_DIR / "season_simulation.png", year,
                               compare=ORIGINAL_LABEL)
    viz.plot_team_value(best.rounds, IMAGES_DIR / "season_team_value.png", year, SALARY_CAP)
    by_season = {y: {k: dev[y][k].total for k in (ORIGINAL_LABEL, IMPROVED_LABEL)}
                 for y in DEV_SEASONS}
    by_season[year] = {k: sims[k].total for k in (ORIGINAL_LABEL, IMPROVED_LABEL)}
    viz.plot_strategy_improvement(by_season, ORIGINAL_LABEL, IMPROVED_LABEL, year,
                                  IMAGES_DIR / "strategy_improvement.png")

    humans = {}
    bench_path = DATA_INPUTS / f"human_benchmarks_{year}.yaml"
    if bench_path.exists():
        humans = yaml.safe_load(bench_path.read_text(encoding="utf-8"))["season_totals"]
        viz.plot_human_comparison({k: s.total for k, s in sims.items()}, humans,
                                  IMPROVED_LABEL, IMAGES_DIR / "human_comparison.png", year)

    # 2. Predictor comparison: a fresh squad every round.
    weekly = predictor_comparison(inputs.rounds)
    weekly.round(1).to_csv(REVIEW_DIR / "predictor_comparison_weekly.csv")
    viz.plot_strategy_backtest(weekly, IMAGES_DIR / "predictor_comparison.png", year,
                               highlight=MODEL_LABEL, baseline="Season average",
                               reference="Hindsight-perfect")

    # 4. Players and positions.
    viz.plot_player_predictions(test, IMAGES_DIR / "player_predicted_vs_actual.png",
                                MODEL_LABEL, year)
    viz.plot_position_analysis(test, season_totals, IMAGES_DIR / "position_analysis.png",
                               MODEL_LABEL, year)
    viz.plot_top_players(test, season_totals, IMAGES_DIR / "top_players.png", MODEL_LABEL, year)
    tots = team_of_the_season(inputs.rounds, season_totals)
    viz.plot_team_of_season(tots[tots["role"] == "Starting"], tots[tots["role"] != "Starting"],
                            IMAGES_DIR / "team_of_the_season.png", year,
                            float(tots["start_price"].sum()))
    regulars = season_totals[(season_totals["games_played"] >= 10)
                             & (season_totals["start_price"] > 0)].copy()
    regulars["season_avg"] = regulars["avg_points"]
    regulars["primary_pos"] = primary_position(regulars["positions"])
    viz.plot_value_picks(regulars, IMAGES_DIR / "value_picks.png", year)

    # Pre-season: how well did each valuation rank players' final averages?
    pre = inputs.rounds[1].frame.join(
        season_totals.set_index("player_name")[["avg_points", "games_played"]])
    pre = pre[pre["games_played"] >= 5]
    preseason_corr = {label: round(float(pre[col].corr(pre["avg_points"])), 3)
                      for label, col in [(MODEL_LABEL, "predicted_points"),
                                         ("Start price", "price"),
                                         ("Last season's average", "season_avg_value")]}

    # 5. Production model for next season, trained on every season.
    train_production_model(pairs, metrics, year)

    model_weekly = weekly[MODEL_LABEL]
    rr = best.rounds
    summary = {
        "season": year,
        "training_games": int(len(train_pairs)),
        "games_tested": int(len(test)),
        "model_benchmark": {k: {m: round(v, 4) for m, v in d.items()}
                            for k, d in metrics.items()},
        "availability_and_price_models": {k: round(v, 4) for k, v in outlook_metrics.items()},
        "predictor_comparison": {
            "season_points": {k: round(float(v)) for k, v in weekly.sum().items()},
            "rounds_won_by_model": {k: int((model_weekly > weekly[k]).sum())
                                    for k in weekly.columns if k != MODEL_LABEL},
        },
        "season_simulation": {
            "season_points": {k: round(s.total) for k, s in sims.items()},
            "average_per_round": {k: round(s.total / len(s.rounds), 1) for k, s in sims.items()},
            "empty_scoring_slots": {k: int(s.rounds["empty_slots"].sum())
                                    for k, s in sims.items()},
            "bye_rounds_12_15_18": round(float(rr.loc[[12, 15, 18], "points"].sum())),
            "trades_used": int((~best.trades["boost"]).sum()),
            "boosts_used": int(best.trades["boost"].sum()),
            "final_team_value": round(float(rr["squad_value"].iloc[-1] + rr["bank"].iloc[-1])),
        },
        "strategy_by_season": {str(y): {k: round(v) for k, v in d.items()}
                               for y, d in by_season.items()},
        "vs_real_coaches": {k: round(best.total - v) for k, v in humans.items()},
        "preseason_correlation_with_final_average": preseason_corr,
        "salary_cap": SALARY_CAP,
    }
    (REVIEW_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Season review complete -> %s, %s", REVIEW_DIR, IMAGES_DIR)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("optimizer", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="End-of-season retrospective")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--scrape", action="store_true",
                        help="Re-scrape the season before reviewing it")
    args = parser.parse_args()
    summary = run_review(args.year, args.scrape)
    print(json.dumps({k: summary[k] for k in ("strategy_by_season", "season_simulation",
                                              "vs_real_coaches")}, indent=2))

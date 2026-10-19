"""
season_review.py — the end-of-season retrospective (the project's final run).

  1. Model benchmark: train on 2022-2025 only, predict every game of 2026,
     and compare with the heuristics fantasy coaches use and a ridge model.
  2. Squad backtests: every round, pick a fresh 26-man squad under the cap
     with each strategy's player valuations and score its best 18 on real
     results; plus one set-and-forget pre-season pick per strategy.
  3. Team of the season (the hindsight-optimal squad) and the best-value buys.
  4. Retrain the production model on every season, ready for next year.

Outputs: outputs/season_review/*.csv + summary.json, docs/images/*.png, models/.

Usage:
    python season_review.py             # uses cached data in data/raw/
    python season_review.py --scrape    # refresh the season from the site first
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd
import pulp

import visualise as viz
from main import load_data
from model import (NN_LABEL, ORACLE_LABEL, TARGET, build_prediction_frame,
                   build_training_pairs, evaluate_holdout, fit_price_baseline,
                   predict_next_round, primary_position, train_production_model)
from optimizer import (ACTIVE_QUOTAS, POINTS_ONLY, SALARY_CAP, _eligible_positions,
                       select_team)
from paths import OUTPUTS_DIR, PROJECT_ROOT
from scraper import load_fixtures

log = logging.getLogger(__name__)

REVIEW_DIR = OUTPUTS_DIR / "season_review"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"


# ── Squad scoring ─────────────────────────────────────────────────────────────

def best_scoring_18(players: pd.DataFrame) -> float:
    """Best legal 18 for one round: Supercoach's active quotas (2 HOK, 4 FRF,
    4 2RF, 1 HFB, 1 5/8, 4 CTW, 1 FLB) plus one FLEX, filled from the squad
    members who played. Captaincy is ignored."""
    if players.empty:
        return 0.0
    prob = pulp.LpProblem("best_18", pulp.LpMaximize)
    slots = {}
    for i, row in players.iterrows():
        for pos in _eligible_positions(row["positions"]) + ["FLEX"]:
            slots[i, pos] = pulp.LpVariable(f"s_{i}_{pos.replace('/', '_')}", cat="Binary")
    prob += pulp.lpSum(var * players.at[i, "score"] for (i, _), var in slots.items())
    for i in players.index:
        prob += pulp.lpSum(v for (j, _), v in slots.items() if j == i) <= 1
    for pos, quota in {**ACTIVE_QUOTAS, "FLEX": 1}.items():
        prob += pulp.lpSum(v for (_, p), v in slots.items() if p == pos) <= quota
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return float(pulp.value(prob.objective) or 0.0)


def weekly_scores(squad: pd.DataFrame, rounds: pd.DataFrame, year: int) -> pd.Series:
    season = rounds[(rounds["year"] == year) & rounds["player_name"].isin(squad["player_name"])]
    positions = squad.set_index("player_name")["positions"]
    out = {}
    for rnd in range(1, int(rounds.loc[rounds["year"] == year, "round"].max()) + 1):
        played = season[season["round"] == rnd][["player_name", "score"]].copy()
        played["positions"] = played["player_name"].map(positions)
        out[rnd] = best_scoring_18(played.reset_index(drop=True))
    return pd.Series(out, name="points")


def weekly_backtest(rounds: pd.DataFrame, totals: dict, year: int, network,
                    price_fit: tuple[float, float]) -> pd.DataFrame:
    """Pick a fresh 26-man squad every round under each valuation and score
    its best 18 with that round's real results.

    Each round's pool is the players who took the field that round (team
    lists are public before lockout), priced at that round's price. The
    model only sees games before the round. 27 independent selections per
    strategy make this far less noisy than a single pre-season pick.
    """
    fixtures = load_fixtures(year)
    season = rounds[rounds["year"] == year]
    rows = {}
    for rnd in sorted(season["round"].unique()):
        played = season[season["round"] == rnd].set_index("player_name")
        frame = build_prediction_frame(rounds, totals, year, int(rnd), fixtures)
        frame = frame[frame["player_name"].isin(played.index)].copy()
        frame["price"] = frame["player_name"].map(played["price"]).astype(float)
        frame["price_usd"] = frame["price"]
        frame = predict_next_round(frame, network, price_fit=price_fit)
        frame["score"] = frame["player_name"].map(played["score"])
        season_avg = np.where(frame["is_rookie"], frame["predicted_points"], frame["season_avg"])
        valuations = {
            NN_LABEL: frame["predicted_points"],
            "Season average": pd.Series(season_avg, index=frame.index),
            "Price only": frame["price"],
            "Hindsight-perfect": frame["score"],
        }
        rows[int(rnd)] = {}
        for name, value in valuations.items():
            squad = squad_table(select_team(frame.assign(predicted_points=value),
                                            round_number=int(rnd), apply_injuries=False,
                                            weights=POINTS_ONLY, team_penalty=0.0))
            rows[int(rnd)][name] = best_scoring_18(squad[["positions", "score"]])
        log.info("Round %2d: %s", rnd, "  ".join(f"{k} {v:.0f}" for k, v in rows[int(rnd)].items()))
    weekly = pd.DataFrame(rows).T
    weekly.index.name = "round"
    return weekly


def squad_table(result: dict) -> pd.DataFrame:
    rows = []
    for group, role in [("starting_13", "Starting"), ("bench_4", "Bench"),
                        ("flex_1", "Flex"), ("reserves_8", "Reserve")]:
        for p in result.get(group, []):
            rows.append({**p, "role": role})
    return pd.DataFrame(rows)


# ── Review ────────────────────────────────────────────────────────────────────

def preseason_squads(rounds: pd.DataFrame, totals: dict, year: int, network,
                     price_fit: tuple[float, float]) -> dict:
    """One set-and-forget squad per valuation, picked before round 1 at start
    prices, scored with its best 18 every round (no trades)."""
    frame = build_prediction_frame(rounds, totals, year, next_round=1,
                                   fixtures=load_fixtures(year), price_col="start_price")
    frame = frame[frame["price"] > 0]
    frame = predict_next_round(frame, network, price_fit=price_fit)
    actual = totals[year].set_index("player_name")
    frame["start_price"] = frame["price"]
    frame["season_total"] = frame["player_name"].map(actual["total_points"]).fillna(0)
    frame["season_avg_actual"] = frame["player_name"].map(actual["avg_points"]).fillna(0)
    frame["games_actual"] = frame["player_name"].map(actual["games_played"]).fillna(0)

    last_season = np.where(frame["is_rookie"], frame["predicted_points"], frame["season_avg"])
    valuations = {
        NN_LABEL: frame["predicted_points"],
        "Last season's average": pd.Series(last_season, index=frame.index),
        "Price only": frame["price"].astype(float),
        "Hindsight-perfect": frame["season_total"],
    }
    out = {}
    for name, value in valuations.items():
        # The live season objective: its diversity/versatility terms hedge
        # the injury and bye risk a squad without trades has to carry.
        result = select_team(frame.assign(predicted_points=value), round_number=1,
                             apply_injuries=False)
        squad = squad_table(result)
        out[name] = {"squad": squad, "cost": int(result["total_price"]),
                     "points": float(weekly_scores(squad, rounds, year).sum())}
        slug = name.lower().replace("'", "").replace(" ", "_").replace("&", "and")
        squad[["role", "player_name", "assigned_position", "team", "start_price",
               "predicted_points", "season_avg_actual", "games_actual", "season_total"]
              ].round(1).to_csv(REVIEW_DIR / f"preseason_squad_{slug}.csv", index=False)
        log.info("Pre-season %-22s %6.0f pts", name, out[name]["points"])

    regulars = frame[frame["games_actual"] >= 5]
    out["_preseason_corr"] = {
        name: round(float(regulars[col].corr(regulars["season_avg_actual"])), 3)
        for name, col in [(NN_LABEL, "predicted_points"), ("Start price", "price"),
                          ("Last season's average", "season_avg")]}
    return out


def run_review(year: int, scrape: bool = False) -> dict:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rounds, totals = load_data(year, no_scrape=not scrape)
    pairs = build_training_pairs(rounds, totals)
    train_pairs = pairs[pairs["year"] < year]
    price_fit = fit_price_baseline(train_pairs)

    # 1. Out-of-time benchmark: train < year, predict every game of `year`.
    metrics, test, network = evaluate_holdout(pairs, year)
    pd.DataFrame(metrics).T.round(4).to_csv(REVIEW_DIR / "model_benchmark.csv")
    keep = ["player_name", "team", "positions", "next_round", "opp", TARGET,
            *[c for c in test.columns if c.startswith("pred_")]]
    test[keep].round(2).to_csv(REVIEW_DIR / f"game_predictions_{year}.csv", index=False)
    viz.plot_model_benchmark(metrics, len(test), year, IMAGES_DIR / "model_benchmark.png",
                             NN_LABEL, ORACLE_LABEL)
    viz.plot_error_by_round(test, IMAGES_DIR / "error_by_round.png", NN_LABEL)
    viz.plot_calibration(test, IMAGES_DIR / "calibration.png", NN_LABEL)

    # 2. Squad backtests: a fresh squad every round, and one pre-season pick.
    weekly = weekly_backtest(rounds, totals, year, network, price_fit)
    weekly.round(1).to_csv(REVIEW_DIR / "weekly_squad_backtest.csv")
    viz.plot_strategy_backtest(weekly, IMAGES_DIR / "squad_backtest.png", year,
                               highlight=NN_LABEL, baseline="Season average",
                               reference="Hindsight-perfect")
    preseason = preseason_squads(rounds, totals, year, network, price_fit)

    # 3. Team of the season and value picks.
    perfect = preseason["Hindsight-perfect"]["squad"].assign(
        season_avg=lambda d: d["season_avg_actual"])
    viz.plot_team_of_season(perfect[perfect["role"] == "Starting"],
                            perfect[perfect["role"] != "Starting"],
                            IMAGES_DIR / "team_of_the_season.png", year,
                            float(perfect["start_price"].sum()))
    regulars = totals[year][(totals[year]["games_played"] >= 10)
                            & (totals[year]["start_price"] > 0)].copy()
    regulars["season_avg"] = regulars["avg_points"]
    regulars["primary_pos"] = primary_position(regulars["positions"])
    viz.plot_value_picks(regulars, IMAGES_DIR / "value_picks.png", year)

    # 4. Production model for next season, trained on every season.
    train_production_model(pairs, metrics, year)

    nn = weekly[NN_LABEL]
    summary = {
        "season": year,
        "training_games": int(len(train_pairs)),
        "games_tested": int(len(test)),
        "model_benchmark": {k: {m: round(v, 4) for m, v in d.items()}
                            for k, d in metrics.items()},
        "weekly_squad_backtest": {
            "season_points": {k: round(float(v)) for k, v in weekly.sum().items()},
            "rounds_won_by_model": {k: int((nn > weekly[k]).sum())
                                    for k in weekly.columns if k != NN_LABEL},
            "rounds": int(len(weekly)),
        },
        "preseason_squad_backtest": {
            "season_points": {k: round(v["points"]) for k, v in preseason.items()
                              if not k.startswith("_")},
            "squad_cost": {k: v["cost"] for k, v in preseason.items() if not k.startswith("_")},
            "correlation_with_final_average": preseason["_preseason_corr"],
        },
        "salary_cap": SALARY_CAP,
    }
    (REVIEW_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Season review complete -> %s, %s", REVIEW_DIR, IMAGES_DIR)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="End-of-season retrospective")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--scrape", action="store_true",
                        help="Re-scrape the season before reviewing it")
    args = parser.parse_args()
    summary = run_review(args.year, args.scrape)
    print(json.dumps({k: summary[k] for k in ("weekly_squad_backtest",
                                              "preseason_squad_backtest")}, indent=2))

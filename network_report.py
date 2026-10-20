"""
network_report.py — a closer look at the points network.

  1. Architecture choice, on the development season: train on 2022-2024,
     predict every 2025 game. Ridge regression, a plain MLP, the Wide & Deep
     network (three seeds), deeper and Huber-loss variants, and the
     network + ridge average that became the production predictor.
  2. Training curve of the network that predicts 2026 (train 2022-2025):
     fit vs held-out-player loss per epoch, and where early stopping cut in.
  3. Permutation importance on 2026: shuffle one group of inputs at a time
     and measure how much the error rises.
     Also split by phase: the first rounds (1-4 games known) vs the rest.
  4. What the deep path adds: the size of its corrections to the linear
     (wide) path's prediction.

Outputs: outputs/network_report/*.csv + network_report.json,
         docs/images/network_*.png.

Usage:
    python network_report.py
"""

import json
import logging

import numpy as np
import pandas as pd

import visualise as viz
from main import load_data
from model import (FEATURES, POS_COLS, ROLLING_STATS, SEED, TARGET,
                   build_training_pairs, fit_network, nn_predict, score_metrics,
                   train_ridge)
from paths import OUTPUTS_DIR, PROJECT_ROOT

log = logging.getLogger(__name__)

REPORT_DIR = OUTPUTS_DIR / "network_report"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
DEV_YEAR, TEST_YEAR = 2025, 2026
EARLY_ROUNDS = 5         # rounds 2-5: only 1-4 games of this season known

FEATURE_GROUPS = {
    "Recent form": ["season_avg", "last1", "last3_avg", "last5_avg", "season_std",
                    "games_played"],
    "Minutes and role": ["mins_avg", "mins_last3", "mins_last1", "ppm", "last_started",
                         "last_benched", "start_share"],
    "How points are scored": list(ROLLING_STATS.values()),
    "Price and break-even": ["price", "break_even", "season_price_change"],
    "Last season": ["prev_season_avg", "prev_season_games", "has_prev_season"],
    "Next opponent": ["opp_factor"],
    "Own team's strength": ["team_factor"],
    "Round and days off": ["next_round", "rounds_since_last"],
    "Position": POS_COLS,
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURES)

CHOSEN = "Wide & Deep (production)"


def _variants():
    from tensorflow import keras
    return {
        "Plain MLP, no wide path": {"wide": False},
        CHOSEN: {},
        "Wide & Deep, 3 layers (128-64-32)": {"hidden": (128, 64, 32),
                                              "dropout": (0.3, 0.3, 0.2)},
        "Wide & Deep, Huber loss": {"loss": keras.losses.Huber(delta=20.0)},
    }


def trainable_params(model) -> int:
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def _metrics(test, pred) -> dict:
    m = score_metrics(test[TARGET], pred, test["next_round"])
    return {k: round(v, 3) for k, v in m.items()}


def architecture_comparison(pairs: pd.DataFrame) -> pd.DataFrame:
    """Every candidate trained on seasons before DEV_YEAR, scored on DEV_YEAR."""
    train, test = pairs[pairs["year"] < DEV_YEAR], pairs[pairs["year"] == DEV_YEAR]
    ridge = train_ridge(train).predict(test)
    rows = {"Season average (heuristic)": {**_metrics(test, test["season_avg"]), "params": 0},
            "Ridge regression": {**_metrics(test, ridge), "params": len(FEATURES) + 1}}
    for label, arch in _variants().items():
        model, hist = fit_network(train, **arch)
        nn = nn_predict(model, test)
        rows[label] = {**_metrics(test, nn), "params": trainable_params(model),
                       "epochs": len(hist["loss"])}
        if label == CHOSEN:
            rows["Wide & Deep + ridge average"] = {**_metrics(test, (nn + ridge) / 2),
                                                   "params": rows[label]["params"]
                                                   + len(FEATURES) + 1}
            seeds = [rows[label]["mae"]]
            for seed in (1, 2):
                other, _ = fit_network(train, seed=seed)
                seeds.append(round(score_metrics(test[TARGET],
                                                 nn_predict(other, test))["mae"], 3))
            rows[label]["mae_seed_range"] = [min(seeds), max(seeds)]
        log.info("  %-36s MAE %.3f", label, rows[label]["mae"])
    return pd.DataFrame(rows).T


def _shuffle_rise(predict, test: pd.DataFrame, cols: list, mask: np.ndarray,
                  rng, repeats: int) -> np.ndarray:
    """Rise in MAE over the `mask` games when `cols` are shuffled among them."""
    sub = test[mask]
    y = sub[TARGET].to_numpy()
    base = np.abs(predict(sub) - y).mean()
    rises = []
    for _ in range(repeats):
        shuffled = sub.copy()
        shuffled[cols] = sub[cols].to_numpy()[rng.permutation(len(sub))]
        rises.append(np.abs(predict(shuffled) - y).mean() - base)
    return np.array(rises)


def permutation_importance(model, ridge, test: pd.DataFrame, repeats: int = 5) -> pd.DataFrame:
    """Rise in MAE when one group of inputs is shuffled across games: over the
    whole season, and separately for the first rounds (1-4 games known)."""
    rng = np.random.default_rng(SEED)
    network = lambda df: nn_predict(model, df)          # noqa: E731
    everything = np.ones(len(test), bool)
    early = (test["next_round"] <= EARLY_ROUNDS).to_numpy()
    rows = []
    for group, cols in FEATURE_GROUPS.items():
        nn = _shuffle_rise(network, test, cols, everything, rng, repeats)
        rows.append({"group": group, "features": len(cols),
                     "nn_mae_rise": nn.mean(), "nn_min": nn.min(), "nn_max": nn.max(),
                     "ridge_mae_rise": _shuffle_rise(ridge.predict, test, cols, everything,
                                                     rng, repeats).mean(),
                     "nn_early": _shuffle_rise(network, test, cols, early, rng, repeats).mean(),
                     "nn_later": _shuffle_rise(network, test, cols, ~early, rng, repeats).mean()})
    return pd.DataFrame(rows).sort_values("nn_mae_rise", ascending=False)


def deep_path(model, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(wide, deep) contributions to each prediction; they sum to it."""
    from tensorflow import keras
    sub = keras.Model(model.inputs, [model.get_layer("wide").output,
                                     model.get_layer("deep").output])
    wide, deep = sub.predict(test[FEATURES].to_numpy(np.float32), verbose=0,
                             batch_size=2048)
    return wide.flatten(), deep.flatten()


def run_report() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rounds, totals = load_data(TEST_YEAR, no_scrape=True)
    pairs = build_training_pairs(rounds, totals)

    # 1. Architecture choice on the development season.
    log.info("Architecture comparison: train < %d, test %d", DEV_YEAR, DEV_YEAR)
    arch = architecture_comparison(pairs)
    arch.to_csv(REPORT_DIR / "architecture_comparison.csv")

    # 2. The network that predicts the test season.
    train, test = pairs[pairs["year"] < TEST_YEAR], pairs[pairs["year"] == TEST_YEAR].copy()
    model, hist = fit_network(train)
    ridge = train_ridge(train)
    history = pd.DataFrame(hist)
    history.index = history.index + 1
    history.index.name = "epoch"
    history.to_csv(REPORT_DIR / "training_history.csv")
    best_epoch = int(history["val_loss"].idxmin())

    # 3. Which inputs it relies on.
    imp = permutation_importance(model, ridge, test)
    imp.round(3).to_csv(REPORT_DIR / "permutation_importance.csv", index=False)

    # 4. What the deep path adds.
    wide, deep = deep_path(model, test)
    nn = wide + deep
    report = {
        "features": len(FEATURES),
        "trainable_parameters": trainable_params(model),
        "training_games": int(len(train)),
        "epochs_run": int(len(history)),
        "best_epoch": best_epoch,
        "best_val_rmse": round(float(np.sqrt(history["val_loss"].min())), 2),
        "test_metrics": {"network": _metrics(test, nn),
                         "ridge": _metrics(test, ridge.predict(test)),
                         "ensemble": _metrics(test, (nn + ridge.predict(test)) / 2)},
        "deep_path": {
            "mean_abs_correction": round(float(np.abs(deep).mean()), 2),
            "share_over_5_points": round(float((np.abs(deep) > 5).mean()), 3),
            "share_over_10_points": round(float((np.abs(deep) > 10).mean()), 3),
            "wide_only_mae": round(float(np.abs(wide - test[TARGET]).mean()), 3),
            "corr_network_vs_ridge": round(float(np.corrcoef(nn, ridge.predict(test))[0, 1]), 3),
        },
        "architecture_comparison_dev": json.loads(arch.to_json(orient="index")),
    }
    (REPORT_DIR / "network_report.json").write_text(json.dumps(report, indent=2),
                                                    encoding="utf-8")
    draw_charts()
    log.info("Network report -> %s", REPORT_DIR)
    return report


def draw_charts() -> None:
    """Redraw the charts from the saved outputs (no retraining)."""
    report = json.loads((REPORT_DIR / "network_report.json").read_text(encoding="utf-8"))
    history = pd.read_csv(REPORT_DIR / "training_history.csv", index_col="epoch")
    viz.plot_network_training(history, report["best_epoch"],
                              IMAGES_DIR / "network_training.png")
    imp = pd.read_csv(REPORT_DIR / "permutation_importance.csv")
    viz.plot_feature_importance(imp, IMAGES_DIR / "network_feature_importance.png", TEST_YEAR)
    viz.plot_importance_by_phase(imp, EARLY_ROUNDS, IMAGES_DIR / "network_importance_by_phase.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("matplotlib",):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    import argparse
    parser = argparse.ArgumentParser(description="A closer look at the points network")
    parser.add_argument("--charts-only", action="store_true",
                        help="Redraw the charts from outputs/network_report/")
    if parser.parse_args().charts_only:
        draw_charts()
    else:
        print(json.dumps(run_report(), indent=2))

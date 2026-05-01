"""
visualise.py — Model performance & bye analysis charts for the NRL Supercoach Optimizer.

Generates two report images:
  1. outputs/model_performance.png — 8-panel model analysis
  2. outputs/bye_analysis.png — bye schedule & season plan charts

Usage:
    python visualise.py                    # standalone
    (also called automatically from main.py pipeline)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import (
    clean_data, engineer_features, get_feature_cols,
    build_prediction_features, predict_next_round_scores,
    MODEL_PATH, SCALER_PATH,
    USE_DELTA_TARGET, HOLDOUT_YEAR_OFFSET,
)
from planner import BYE_ROUNDS, ROUND_BYES, BIG_BYE_ROUNDS, TOTAL_ROUNDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_ORDER = ["HOK", "FRF", "2RF", "HFB", "5/8", "CTW", "FLB"]
ROLE_COLOURS = {"Starting": "#2ecc71", "Bench": "#f39c12", "Flex": "#9b59b6", "Reserve": "#95a5a6"}

TEAM_NAMES = {
    "BRO": "Broncos", "CBR": "Raiders", "BUL": "Bulldogs", "SHA": "Sharks",
    "GCT": "Titans", "MNL": "Sea Eagles", "MEL": "Storm", "NEW": "Knights",
    "NZL": "Warriors", "NQC": "Cowboys", "PAR": "Eels", "PTH": "Panthers",
    "STH": "Rabbitohs", "STG": "Dragons", "SYD": "Roosters", "DOL": "Dolphins",
    "WST": "Tigers",
}


def _load_model():
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH)


def _prepare_data():
    """Load master data, engineer features, split into train/val.

    Uses the same year-aware holdout that load_or_train_model uses (NN-1):
    the most recent year is the validation set, everything older is training.
    Previously this used a random 15% split with random_state=42, which (a)
    leaked the same player into both train and val and (b) didn't match the
    split the saved model was actually evaluated on — so the metrics on the
    chart were measuring the wrong thing.
    """
    from paths import MASTER_HISTORICAL_CSV
    df_all = pd.read_csv(MASTER_HISTORICAL_CSV, low_memory=False)
    df_clean = clean_data(df_all)
    df_feat, scaler = engineer_features(df_clean, fit_scaler=True)

    feature_cols = get_feature_cols(df_feat)
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    target_col = "avg_points_raw" if "avg_points_raw" in df_feat.columns else "avg_points"
    y = df_feat[target_col].fillna(0).values.astype(np.float32)
    mask = y > 0

    df_valid = df_feat[mask].copy().reset_index(drop=True)
    X = df_valid[feature_cols].fillna(0).values.astype(np.float32)
    y = y[mask]

    # Year-aware split, mirroring load_or_train_model.
    if "scrape_year" in df_valid.columns:
        years = pd.to_numeric(df_valid["scrape_year"], errors="coerce").values
        finite = years[~np.isnan(years)]
        max_year = int(np.nanmax(years)) if finite.size else None
        holdout_year = (
            max_year - (HOLDOUT_YEAR_OFFSET - 1) if max_year is not None else None
        )
        train_idx = years < holdout_year if holdout_year is not None else None
        val_idx = years == holdout_year if holdout_year is not None else None

        if (
            holdout_year is not None
            and train_idx.sum() >= 20
            and val_idx.sum() >= 5
        ):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            df_val = df_valid.iloc[val_idx].copy()
            log.info("Year-aware val split: train<%d (%d) vs val=%d (%d)",
                     holdout_year, int(train_idx.sum()),
                     holdout_year, int(val_idx.sum()))
            return df_feat, df_val, X_train, X_val, y_train, y_val, feature_cols

        log.warning("Year-aware split insufficient (train=%s val=%s) — random fallback",
                    None if train_idx is None else int(train_idx.sum()),
                    None if val_idx is None else int(val_idx.sum()))

    # Fallback: random split (small datasets / no scrape_year column).
    idx = np.arange(len(df_valid))
    idx_train, idx_val = train_test_split(idx, test_size=0.15, random_state=42)
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]
    df_val = df_valid.iloc[idx_val].copy()

    return df_feat, df_val, X_train, X_val, y_train, y_val, feature_cols


def _val_predictions(model, X_val, df_val):
    """Run the model on X_val and return predictions in the same units as
    y_val (absolute avg_points). When the model was trained against the delta
    target (USE_DELTA_TARGET), the rolling-3 baseline is added back here so
    we're not comparing deltas to absolute scores. This mirrors what
    predict_next_round_scores does at inference time."""
    preds = model.predict(X_val, verbose=0).flatten()
    if USE_DELTA_TARGET and "avg_last3_raw" in df_val.columns:
        baseline = (
            pd.to_numeric(df_val["avg_last3_raw"], errors="coerce")
            .fillna(0).values.astype(np.float32)
        )
        preds = preds + baseline
    return preds


def _get_predictions(df_feat, df_pred=None):
    """Get or compute 2026 predictions. Returns df_pred or None."""
    if df_pred is not None and not df_pred.empty:
        return df_pred

    df_2026 = df_feat[df_feat.get("scrape_year", pd.Series()) == 2026].copy()
    if df_2026.empty:
        return None

    from paths import MASTER_HISTORICAL_CSV
    if not MASTER_HISTORICAL_CSV.exists():
        return None

    hist_raw = pd.read_csv(MASTER_HISTORICAL_CSV, low_memory=False)
    hist_with_scores = hist_raw[
        pd.to_numeric(hist_raw["avg_points"], errors="coerce").fillna(0) > 0
    ]
    return predict_next_round_scores(df_2026, df_historical=hist_with_scores)


# ── Model Performance Charts ────────────────────────────────────────────────

def plot_actual_vs_predicted(ax, model, X_val, y_val, df_val):
    """Chart 1: Actual vs Predicted scatter."""
    preds = _val_predictions(model, X_val, df_val)

    primary_pos = df_val["positions"].str.split("|").str[0].fillna("UNK")
    colours = sns.color_palette("husl", len(POSITION_ORDER))
    pos_colour = {p: colours[i] for i, p in enumerate(POSITION_ORDER)}

    for pos in POSITION_ORDER:
        mask = primary_pos.values == pos
        if mask.sum() == 0:
            continue
        ax.scatter(
            y_val[mask], preds[mask],
            label=pos, color=pos_colour[pos], alpha=0.7, s=30, edgecolors="white", linewidths=0.3,
        )

    mn, mx = min(y_val.min(), preds.min()) - 2, max(y_val.max(), preds.max()) + 2
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.2, label="Perfect")

    rmse = mean_squared_error(y_val, preds) ** 0.5
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    ax.text(
        0.05, 0.92,
        f"R$^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Actual avg_points")
    ax.set_ylabel("Predicted avg_points")
    ax.set_title("Actual vs Predicted (Validation Set)")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)


def plot_residuals(ax, model, X_val, y_val, df_val):
    """Chart 2: Residual distribution."""
    preds = _val_predictions(model, X_val, df_val)
    residuals = preds - y_val

    ax.hist(residuals, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", lw=1.2)

    mu, sigma = residuals.mean(), residuals.std()
    mae = np.abs(residuals).mean()
    within_5 = (np.abs(residuals) <= 5).sum() / len(residuals) * 100
    ax.text(
        0.95, 0.92,
        f"Mean = {mu:+.2f}\nStd = {sigma:.2f}\nMAE = {mae:.2f}\n"
        f"Within +/-5: {within_5:.0f}%",
        transform=ax.transAxes, fontsize=8,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Residual (Predicted - Actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")


def plot_per_position_error(ax, model, X_val, y_val, df_val):
    """Chart 3: Per-position MAE and RMSE."""
    preds = _val_predictions(model, X_val, df_val)
    primary_pos = df_val["positions"].str.split("|").str[0].fillna("UNK").values

    positions_present = []
    mae_vals = []
    rmse_vals = []
    counts = []

    for pos in POSITION_ORDER:
        mask = primary_pos == pos
        if mask.sum() < 3:
            continue
        positions_present.append(pos)
        mae_vals.append(mean_absolute_error(y_val[mask], preds[mask]))
        rmse_vals.append(mean_squared_error(y_val[mask], preds[mask]) ** 0.5)
        counts.append(int(mask.sum()))

    if not positions_present:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    x = np.arange(len(positions_present))
    width = 0.35

    ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#e74c3c", alpha=0.8)

    for i, count in enumerate(counts):
        ax.text(x[i], max(mae_vals[i], rmse_vals[i]) + 0.3, f"n={count}",
                ha="center", fontsize=6, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(positions_present, fontsize=8)
    ax.set_xlabel("Position")
    ax.set_ylabel("Error")
    ax.set_title("Model Error by Position")
    ax.legend(fontsize=7)


def plot_error_by_price_tier(ax, model, X_val, y_val, df_val):
    """Chart 4: Prediction error by price tier."""
    preds = _val_predictions(model, X_val, df_val)

    if "price_usd" in df_val.columns:
        prices = df_val["price_usd"].values
    elif "price" in df_val.columns:
        prices = df_val["price"].values
    else:
        ax.text(0.5, 0.5, "No price data", ha="center", va="center",
                transform=ax.transAxes)
        return

    tier_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    thresholds = [
        (0, 200000, "Budget\n(<$200K)"),
        (200000, 400000, "Mid\n($200-400K)"),
        (400000, 600000, "Premium\n($400-600K)"),
        (600000, float("inf"), "Elite\n(>$600K)"),
    ]

    tiers = []
    tier_labels = []
    active_colors = []
    for (low, high, label), color in zip(thresholds, tier_colors):
        mask = (prices >= low) & (prices < high)
        if mask.sum() >= 3:
            residuals = np.abs(preds[mask] - y_val[mask])
            tiers.append(residuals)
            tier_labels.append(f"{label}\nn={mask.sum()}")
            active_colors.append(color)

    if not tiers:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    bp = ax.boxplot(tiers, tick_labels=tier_labels, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], active_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Absolute Error")
    ax.set_title("Prediction Error by Price Tier")


def plot_position_boxplot(ax, df_feat, df_pred=None):
    """Chart 5: Prediction distribution by position (2026 players)."""
    pred_data = _get_predictions(df_feat, df_pred)
    if pred_data is None or pred_data.empty:
        ax.text(0.5, 0.5, "No 2026 data", ha="center", va="center",
                transform=ax.transAxes)
        return

    pred_data = pred_data.copy()
    pred_data["_pp"] = pred_data["positions"].str.split("|").str[0].fillna("UNK")
    plot_data = pred_data[pred_data["_pp"].isin(POSITION_ORDER)]

    colours = sns.color_palette("husl", len(POSITION_ORDER))
    bp = ax.boxplot(
        [plot_data[plot_data["_pp"] == p]["predicted_points"].dropna().values
         for p in POSITION_ORDER],
        tick_labels=POSITION_ORDER, patch_artist=True, showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    ax.set_xlabel("Position")
    ax.set_ylabel("Predicted Points")
    ax.set_title("2026 Predictions by Position")


def plot_top_scorers_by_position(ax, df_feat, df_pred=None):
    """Chart 6: Top 5 predicted scorers per position (2026 players)."""
    pred_data = _get_predictions(df_feat, df_pred)
    if pred_data is None or pred_data.empty:
        ax.text(0.5, 0.5, "No 2026 data", ha="center", va="center",
                transform=ax.transAxes)
        return

    pred_data = pred_data.copy()
    pred_data["_pos"] = pred_data["positions"].str.split("|").str[0].fillna("UNK")

    colours = sns.color_palette("husl", len(POSITION_ORDER))
    pos_colour = {p: colours[i] for i, p in enumerate(POSITION_ORDER)}

    y_labels, bar_vals, bar_colours, bar_prices = [], [], [], []

    for pos in reversed(POSITION_ORDER):
        grp = (pred_data[pred_data["_pos"] == pos]
               .nlargest(5, "predicted_points"))
        for _, row in grp.iterrows():
            short = (row["player_name"].split(",")[0]
                     if "," in row["player_name"]
                     else row["player_name"].split()[-1])
            price_k = float(row.get("price_usd", row.get("price", 0)) or 0) / 1000
            y_labels.append(f"{short:<14} ({pos})")
            bar_vals.append(row["predicted_points"])
            bar_colours.append(pos_colour[pos])
            bar_prices.append(price_k)

    y_pos = list(range(len(y_labels)))
    ax.barh(y_pos, bar_vals, color=bar_colours, alpha=0.82,
            edgecolor="white", linewidth=0.3)

    for i, (val, price) in enumerate(zip(bar_vals, bar_prices)):
        ax.text(val + 0.3, i, f"{val:.0f}  ${price:.0f}K",
                va="center", fontsize=5.5, color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xlabel("Predicted Points")
    ax.set_title("Top 5 Predicted Scorers by Position (2026)")


def plot_price_vs_predicted(ax, df_feat, df_pred=None, round_num=1):
    """Chart 7: Price vs Predicted — value map with squad status colours."""
    pred_data = _get_predictions(df_feat, df_pred)
    if pred_data is None or pred_data.empty or "price_usd" not in pred_data.columns:
        ax.text(0.5, 0.5, "No 2026 data", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Try to load confirmed starters from the latest trade advice CSV
    confirmed_starters = set()
    for rnd in range(round_num, 0, -1):
        adv_path = OUTPUT_DIR / f"trade_advice_r{rnd}.csv"
        if adv_path.exists():
            df_adv = pd.read_csv(adv_path)
            if "is_starter" in df_adv.columns:
                confirmed_starters = set(
                    df_adv.loc[df_adv["is_starter"], "player_name"]
                )
            break

    # Squad membership — prefer trade advice CSV (has your actual team)
    squad_names = set()
    for rnd in range(round_num, 0, -1):
        adv_path = OUTPUT_DIR / f"trade_advice_r{rnd}.csv"
        if adv_path.exists():
            squad_names = set(pd.read_csv(adv_path)["player_name"].values)
            break
    if not squad_names:
        squad_path = OUTPUT_DIR / f"team_round_{round_num}.csv"
        if not squad_path.exists():
            squad_path = OUTPUT_DIR / "team_round_1.csv"
        if squad_path.exists():
            squad_names = set(pd.read_csv(squad_path)["player_name"].values)

    pred = pred_data.copy()
    in_squad = pred["player_name"].isin(squad_names)

    # Background — all non-squad players
    ax.scatter(
        pred.loc[~in_squad, "price_usd"] / 1000,
        pred.loc[~in_squad, "predicted_points"],
        color="#d5d8dc", alpha=0.3, s=12, label="Not in squad", zorder=1,
    )

    # Squad players: classify by status
    for _, row in pred[in_squad].iterrows():
        nm = row["player_name"]
        is_starter = nm in confirmed_starters if confirmed_starters else True
        try:
            on_bye = (int(row.get("bye_round", -1)) == round_num
                      and not is_starter and round_num > 0)
        except (TypeError, ValueError):
            on_bye = False

        if is_starter:
            color, zorder = "#27ae60", 4
        elif on_bye:
            color, zorder = "#f39c12", 4
        else:
            color, zorder = "#e74c3c", 4

        ax.scatter(row["price_usd"] / 1000, row["predicted_points"],
                   color=color, alpha=0.92, s=55,
                   edgecolors="black", linewidths=0.5, zorder=zorder)

        short = (row["player_name"].split(",")[0]
                 if "," in row["player_name"]
                 else row["player_name"].split()[-1])
        tag = " (bye)" if on_bye else ("" if is_starter else " (out)")
        ax.annotate(
            short + tag,
            (row["price_usd"] / 1000, row["predicted_points"]),
            fontsize=5.5, xytext=(4, 3), textcoords="offset points",
        )

    ax.set_xlabel("Price ($K)")
    ax.set_ylabel("Predicted Points")
    ax.set_title(f"Price vs Predicted — Round {round_num} Squad Status")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(fc="#d5d8dc", alpha=0.5, label="Not in squad"),
        Patch(fc="#27ae60", label="Starter"),
        Patch(fc="#f39c12", label="On bye"),
        Patch(fc="#e74c3c", label="Out / not named"),
    ], fontsize=7, loc="lower right")


def plot_squad_bar(ax, round_num=1):
    """Chart 8: Selected squad horizontal bar chart."""
    squad_path = OUTPUT_DIR / f"team_round_{round_num}.csv"
    if not squad_path.exists():
        squad_path = OUTPUT_DIR / "team_round_1.csv"
    if not squad_path.exists():
        ax.text(0.5, 0.5, "No squad CSV found", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.read_csv(squad_path)
    role_order = {"Starting": 0, "Bench": 1, "Flex": 2, "Reserve": 3}
    df["_sort"] = df["role"].map(role_order).fillna(4)
    df = df.sort_values(["_sort", "predicted_points"], ascending=[False, True])

    colours = [ROLE_COLOURS.get(r, "#95a5a6") for r in df["role"]]
    labels = [f"{row['player_name']} ({row['assigned_position']})"
              for _, row in df.iterrows()]

    ax.barh(range(len(df)), df["predicted_points"], color=colours,
            edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("Predicted Points")

    n_reserves = (df["role"] == "Reserve").sum()
    if n_reserves > 0:
        ax.axhline(y=n_reserves - 0.5, color="black", linestyle="--", lw=0.8, alpha=0.6)
        ax.text(ax.get_xlim()[1] * 0.02, n_reserves - 0.7, "<- Reserves (no points)",
                fontsize=5, alpha=0.6, va="top")

    scoring_total = df[df["role"] != "Reserve"]["predicted_points"].sum()
    ax.set_title(f"Round {round_num} -- Optimal Squad "
                 f"(Scoring 18 total: {scoring_total:.0f} pts)")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=r) for r, c in ROLE_COLOURS.items()]
    ax.legend(handles=legend_elements, fontsize=6, loc="lower right")


# ── Bye Analysis Charts ─────────────────────────────────────────────────────

def plot_bye_heatmap(ax):
    """Bye schedule heatmap: teams vs rounds."""
    teams = sorted(BYE_ROUNDS.keys())
    rounds = list(range(1, TOTAL_ROUNDS + 1))

    grid = np.zeros((len(teams), len(rounds)))
    for i, team in enumerate(teams):
        for bye_rnd in BYE_ROUNDS[team]:
            if 1 <= bye_rnd <= TOTAL_ROUNDS:
                grid[i, bye_rnd - 1] = 1

    cmap = matplotlib.colors.ListedColormap(["#f0f0f0", "#e74c3c"])
    ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest")

    team_labels = [f"{t} ({TEAM_NAMES.get(t, t)})" for t in teams]
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(team_labels, fontsize=6)

    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels(rounds, fontsize=6)
    ax.set_xlabel("Round")

    # Highlight big bye rounds
    for rnd in BIG_BYE_ROUNDS:
        idx = rnd - 1
        ax.axvline(x=idx - 0.5, color="#f39c12", linestyle="--", lw=1.0, alpha=0.7)
        ax.axvline(x=idx + 0.5, color="#f39c12", linestyle="--", lw=1.0, alpha=0.7)

    # Bye count per round at top
    byes_per_round = [len(ROUND_BYES.get(r, set())) for r in rounds]
    for j, count in enumerate(byes_per_round):
        if count > 0:
            ax.text(j, -0.8, str(count), ha="center", va="center", fontsize=5,
                    fontweight="bold" if count >= 5 else "normal",
                    color="#e74c3c" if count >= 5 else "#666")

    ax.set_title("Bye Schedule (red = bye | numbers = teams on bye per round)")


def plot_round_scoring(ax, season_state=None):
    """Round-by-round projected scoring bar chart."""
    rounds = scores = n_byes = None

    if season_state is not None:
        round_results = season_state.get("round_results", [])
        if round_results:
            rounds = [rr["round"] for rr in round_results]
            scores = [rr["projected_points"] for rr in round_results]
            n_byes = [rr["n_bye_teams"] for rr in round_results]

    if rounds is None:
        summary_path = OUTPUT_DIR / "round_summary.csv"
        if summary_path.exists():
            df_sum = pd.read_csv(summary_path)
            rounds = df_sum["round"].tolist()
            scores = df_sum["scoring_total"].tolist()
            n_byes = df_sum["n_bye_teams"].tolist()
        else:
            ax.text(0.5, 0.5, "No season plan data\nRun with --plan",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            return

    colors = []
    for nb in n_byes:
        if nb >= 5:
            colors.append("#e74c3c")
        elif nb >= 3:
            colors.append("#f39c12")
        elif nb > 0:
            colors.append("#3498db")
        else:
            colors.append("#2ecc71")

    ax.bar(rounds, scores, color=colors, edgecolor="white", linewidth=0.3)
    avg_score = np.mean(scores)
    ax.axhline(y=avg_score, color="black", linestyle="--", lw=1, alpha=0.5)
    ax.text(max(rounds) + 0.5, avg_score, f"Avg: {avg_score:.0f}",
            fontsize=7, va="center")

    ax.set_xlabel("Round")
    ax.set_ylabel("Projected Points")
    ax.set_title("Round-by-Round Projected Scoring")
    ax.set_xticks(rounds)
    ax.set_xticklabels([str(r) for r in rounds], fontsize=6)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="No byes"),
        Patch(facecolor="#3498db", label="1-2 teams bye"),
        Patch(facecolor="#f39c12", label="3-4 teams bye"),
        Patch(facecolor="#e74c3c", label="5+ teams bye"),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc="lower left")


def plot_trade_timeline(ax, season_state=None):
    """Trade schedule with cumulative budget line."""
    df_trades = None

    if season_state is not None:
        trade_log = season_state.get("trade_log", [])
        if trade_log:
            df_trades = pd.DataFrame(trade_log)

    if df_trades is None:
        trade_path = OUTPUT_DIR / "trade_plan.csv"
        if trade_path.exists():
            df_trades = pd.read_csv(trade_path)

    if df_trades is None or df_trades.empty:
        ax.text(0.5, 0.5, "No trade data\nRun with --plan",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        return

    all_rounds = list(range(1, TOTAL_ROUNDS + 1))
    trades_per_round = df_trades.groupby("round").size()
    trade_counts = [int(trades_per_round.get(r, 0)) for r in all_rounds]

    # Color each round's bar by dominant trade type
    colors = []
    for r in all_rounds:
        rnd_trades = df_trades[df_trades["round"] == r]
        if rnd_trades.empty:
            colors.append("#bdc3c7")
        elif "is_tradeback" in rnd_trades.columns and rnd_trades["is_tradeback"].fillna(False).any():
            colors.append("#9b59b6")
        elif "is_bye_loop" in rnd_trades.columns and rnd_trades["is_bye_loop"].fillna(False).any():
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    ax.bar(all_rounds, trade_counts, color=colors, edgecolor="white", linewidth=0.3)

    # Cumulative trades line on secondary axis
    cumulative = np.cumsum(trade_counts)
    ax2 = ax.twinx()
    ax2.plot(all_rounds, cumulative, color="black", linewidth=1.5,
             marker=".", markersize=3)
    ax2.set_ylabel("Cumulative Trades", fontsize=8)
    ax2.set_ylim(0, 50)
    ax2.axhline(y=46, color="red", linestyle=":", lw=0.8, alpha=0.5)
    ax2.text(TOTAL_ROUNDS + 0.5, 46, "Limit: 46", fontsize=6,
             va="center", color="red")

    ax.set_xlabel("Round")
    ax.set_ylabel("Trades This Round")
    ax.set_title("Trade Schedule & Budget")
    ax.set_xticks(all_rounds)
    ax.set_xticklabels([str(r) for r in all_rounds], fontsize=6)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="Permanent"),
        Patch(facecolor="#e74c3c", label="Bye loop"),
        Patch(facecolor="#9b59b6", label="Trade-back"),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc="upper left")


def plot_squad_availability(ax, season_state=None):
    """Squad availability over the season (available vs scoring vs unavailable)."""
    rounds = available_counts = scoring_counts = None

    if season_state is not None:
        round_results = season_state.get("round_results", [])
        if round_results:
            rounds = [rr["round"] for rr in round_results]
            available_counts = [
                len(rr["squad_snapshot"]) - rr["n_unavailable"]
                for rr in round_results
            ]
            scoring_counts = [len(rr["scoring_18"]) for rr in round_results]

    if rounds is None:
        plan_path = OUTPUT_DIR / "season_plan.csv"
        if plan_path.exists():
            df_plan = pd.read_csv(plan_path)
            avail_by_round = df_plan.groupby("round")["available"].sum()
            scoring_by_round = df_plan.groupby("round")["in_scoring_18"].sum()
            rounds = avail_by_round.index.tolist()
            available_counts = avail_by_round.values.tolist()
            scoring_counts = scoring_by_round.values.tolist()
        else:
            ax.text(0.5, 0.5, "No season plan data\nRun with --plan",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            return

    ax.fill_between(rounds, 26, available_counts, color="#e74c3c", alpha=0.3,
                    label="Unavailable (bye/injured)")
    ax.fill_between(rounds, available_counts, scoring_counts, color="#f39c12",
                    alpha=0.3, label="Available (reserve)")
    ax.fill_between(rounds, scoring_counts, 0, color="#2ecc71", alpha=0.4,
                    label="Scoring 18")

    ax.plot(rounds, available_counts, color="#e74c3c", linewidth=1.5,
            marker=".", markersize=4)
    ax.plot(rounds, scoring_counts, color="#2ecc71", linewidth=1.5,
            marker=".", markersize=4)

    ax.axhline(y=18, color="black", linestyle="--", lw=0.8, alpha=0.4)
    ax.text(max(rounds) + 0.3, 18, "18 slots", fontsize=6, va="center", alpha=0.6)

    ax.set_xlabel("Round")
    ax.set_ylabel("Players")
    ax.set_ylim(0, 27)
    ax.set_title("Squad Availability Over Season")
    ax.set_xticks(rounds)
    ax.set_xticklabels([str(r) for r in rounds], fontsize=6)
    ax.legend(fontsize=6, loc="lower left")


# ── Player Analysis Charts ──────────────────────────────────────────────────

def _squad_names_for_round(round_num: int) -> set:
    """Load current squad player names from trade advice or team CSV."""
    for rnd in range(round_num, 0, -1):
        p = OUTPUT_DIR / f"trade_advice_r{rnd}.csv"
        if p.exists():
            return set(pd.read_csv(p)["player_name"].values)
    for rnd in range(round_num, 0, -1):
        p = OUTPUT_DIR / f"team_round_{rnd}.csv"
        if p.exists():
            return set(pd.read_csv(p)["player_name"].values)
    return set()


def _starters_for_round(round_num: int) -> set:
    """Load confirmed starters from trade advice CSV."""
    for rnd in range(round_num, 0, -1):
        p = OUTPUT_DIR / f"trade_advice_r{rnd}.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "is_starter" in df.columns:
                return set(df.loc[df["is_starter"], "player_name"])
    return set()


def plot_value_leaders(ax, df_pred, round_num=1):
    """
    Top 25 value players: predicted_points / (price / 100000).
    Highlights your squad players in green.
    """
    if df_pred is None or df_pred.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    squad = _squad_names_for_round(round_num)
    df = df_pred.dropna(subset=["price_usd", "predicted_points"]).copy()
    df = df[df["price_usd"] > 50_000].copy()
    df["value"] = df["predicted_points"] / (df["price_usd"] / 100_000)
    df = df.nlargest(25, "value")

    colours = ["#27ae60" if nm in squad else "#3498db"
               for nm in df["player_name"]]
    short_names = [
        (nm.split(",")[0] if "," in nm else nm.split()[-1])
        + f" ({str(row['positions']).split('|')[0]})"
        for nm, (_, row) in zip(df["player_name"], df.iterrows())
    ]

    y = list(range(len(df)))
    ax.barh(y, df["value"], color=colours, alpha=0.85,
            edgecolor="white", linewidth=0.3)
    for i, (val, pred, price) in enumerate(
        zip(df["value"], df["predicted_points"], df["price_usd"])
    ):
        ax.text(val + 0.02, i,
                f"{pred:.0f}pts  ${price/1000:.0f}K",
                va="center", fontsize=5.5)

    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=6.5)
    ax.set_xlabel("Value (pred pts per $100K)")
    ax.set_title(f"Top 25 Value Players (Green = in your squad)")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(fc="#27ae60", label="Your squad"),
        Patch(fc="#3498db", label="Available"),
    ], fontsize=7, loc="lower right")


def plot_squad_bye_schedule(ax, round_num=1):
    """
    Show which rounds each of your squad players has a bye.
    Highlights the current round.
    """
    squad = _squad_names_for_round(round_num)
    if not squad:
        ax.text(0.5, 0.5, "No squad data found", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Load player data to get bye_round per player
    from paths import DATA_RAW
    raw_files = sorted(DATA_RAW.glob("nrl_data_2026*.csv"), reverse=True)
    if not raw_files:
        ax.text(0.5, 0.5, "No raw data", ha="center", va="center",
                transform=ax.transAxes)
        return

    df_raw = pd.read_csv(raw_files[0], low_memory=False)
    df_squad = df_raw[df_raw["player_name"].isin(squad)].copy()
    df_squad = df_squad.drop_duplicates("player_name")

    # Use planner's BYE_ROUNDS as ground truth per team
    rows = []
    for _, row in df_squad.iterrows():
        team = row.get("team", "?")
        bye = row.get("bye_round", None)
        try:
            bye = int(float(bye))
        except (TypeError, ValueError):
            bye = None
        short = (row["player_name"].split(",")[0]
                 if "," in str(row["player_name"])
                 else str(row["player_name"]).split()[-1])
        rows.append({
            "name": short,
            "team": team,
            "bye_round": bye,
            "pred": float(row.get("predicted_points", 0) or 0),
        })

    rows.sort(key=lambda r: (r["bye_round"] or 99, -r["pred"]))

    y_pos = list(range(len(rows)))
    bye_rounds = [r["bye_round"] for r in rows]
    y_labels = [f"{r['name']:<14} {r['team']:<4}  bye R{r['bye_round'] or '?'}"
                for r in rows]

    colours = ["#e74c3c" if b == round_num
               else "#f39c12" if b and b <= round_num + 3
               else "#3498db"
               for b in bye_rounds]

    ax.barh(y_pos, [r["pred"] for r in rows],
            color=colours, alpha=0.8, edgecolor="white", linewidth=0.3)
    for i, r in enumerate(rows):
        ax.text(r["pred"] + 0.2, i, f"pred {r['pred']:.0f}",
                va="center", fontsize=5.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=6.5, fontfamily="monospace")
    ax.set_xlabel("Predicted Points")
    ax.set_title(f"Your Squad — Bye Schedule  (Red = bye this round R{round_num}  "
                 f"Orange = bye soon  Blue = bye later)")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(fc="#e74c3c", label=f"Bye this round (R{round_num})"),
        Patch(fc="#f39c12", label="Bye within 3 rounds"),
        Patch(fc="#3498db", label="Bye later in season"),
    ], fontsize=7, loc="lower right")


def plot_position_depth(ax, df_pred, round_num=1):
    """
    Bar chart showing predicted points for top 8 players per position.
    Squad players highlighted. Shows at a glance where you have depth.
    """
    if df_pred is None or df_pred.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    squad = _squad_names_for_round(round_num)
    df = df_pred.copy()
    df["_pos"] = df["positions"].str.split("|").str[0].fillna("UNK")

    colours = sns.color_palette("husl", len(POSITION_ORDER))
    pos_colour = {p: colours[i] for i, p in enumerate(POSITION_ORDER)}

    x_base = 0
    tick_positions, tick_labels = [], []

    for pos in POSITION_ORDER:
        grp = (df[df["_pos"] == pos]
               .dropna(subset=["predicted_points"])
               .nlargest(8, "predicted_points"))
        if grp.empty:
            continue
        xs = list(range(x_base, x_base + len(grp)))
        bar_colours = [
            pos_colour[pos] if nm in squad else (*pos_colour[pos][:3], 0.35)
            for nm in grp["player_name"]
        ]
        ax.bar(xs, grp["predicted_points"], color=bar_colours,
               edgecolor="white", linewidth=0.4)

        tick_positions.append(x_base + len(grp) / 2 - 0.5)
        tick_labels.append(pos)

        # Label top 3 in squad
        for x, (_, row) in zip(xs, grp.iterrows()):
            if row["player_name"] in squad:
                short = (row["player_name"].split(",")[0]
                         if "," in row["player_name"]
                         else row["player_name"].split()[-1])
                ax.text(x, row["predicted_points"] + 0.3, short,
                        ha="center", fontsize=4.5, rotation=75,
                        color=pos_colour[pos], fontweight="bold")

        x_base += len(grp) + 1  # gap between positions

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("Predicted Points")
    ax.set_title(f"Position Depth — Top 8 per Position  "
                 f"(Solid = in your squad  Faded = not in squad)")


def plot_avg_vs_predicted(ax, df_pred, round_num=1):
    """
    Scatter: season avg_points (x) vs predicted_points (y).
    Players above the diagonal are forecast to improve; below = regress.
    Squad highlighted.
    """
    if df_pred is None or df_pred.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    squad = _squad_names_for_round(round_num)
    df = df_pred.dropna(subset=["avg_points", "predicted_points"]).copy()
    df = df[pd.to_numeric(df["avg_points"], errors="coerce").fillna(0) > 0]
    df["avg_points"] = pd.to_numeric(df["avg_points"], errors="coerce")

    in_squad = df["player_name"].isin(squad)

    ax.scatter(df.loc[~in_squad, "avg_points"],
               df.loc[~in_squad, "predicted_points"],
               color="#d5d8dc", alpha=0.3, s=12, zorder=1,
               label="Not in squad")
    ax.scatter(df.loc[in_squad, "avg_points"],
               df.loc[in_squad, "predicted_points"],
               color="#27ae60", alpha=0.92, s=55,
               edgecolors="black", linewidths=0.5, zorder=3,
               label="Your squad")

    for _, row in df[in_squad].iterrows():
        short = (row["player_name"].split(",")[0]
                 if "," in row["player_name"]
                 else row["player_name"].split()[-1])
        ax.annotate(short,
                    (row["avg_points"], row["predicted_points"]),
                    fontsize=5.5, xytext=(4, 3), textcoords="offset points")

    mn = min(df["avg_points"].min(), df["predicted_points"].min()) - 2
    mx = max(df["avg_points"].max(), df["predicted_points"].max()) + 2
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.2, alpha=0.6,
            label="No change line")
    ax.fill_between([mn, mx], [mn, mx], mx,
                    color="#2ecc71", alpha=0.04, label="Model expects improvement")
    ax.fill_between([mn, mx], mn, [mn, mx],
                    color="#e74c3c", alpha=0.04, label="Model expects regression")

    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_xlabel("Season Avg Points (actual)")
    ax.set_ylabel("Predicted Points (model)")
    ax.set_title("Avg vs Predicted — Who the model expects to improve or regress")
    ax.legend(fontsize=6, loc="lower right")


def run_player_analysis(df_pred=None, round_num=1):
    """
    Generate outputs/player_analysis.png — 4-panel player/squad analysis.
    """
    if df_pred is None:
        from paths import DATA_RAW, MASTER_HISTORICAL_CSV
        raw_files = sorted(DATA_RAW.glob("nrl_data_2026*.csv"), reverse=True)
        if not raw_files:
            log.warning("No raw data for player analysis. Skipping.")
            return
        from model import clean_data, engineer_features
        df_raw = pd.read_csv(raw_files[0], low_memory=False)
        df_clean = clean_data(df_raw)
        df_feat, _ = engineer_features(df_clean, fit_scaler=False)
        hist = (pd.read_csv(MASTER_HISTORICAL_CSV, low_memory=False)
                if MASTER_HISTORICAL_CSV.exists() else pd.DataFrame())
        df_pred = predict_next_round_scores(df_feat, df_historical=hist)

    fig, axes = plt.subplots(2, 2, figsize=(22, 24))
    fig.suptitle(
        "NRL Supercoach — Player Analysis",
        fontsize=16, fontweight="bold", y=0.995,
    )
    plt.subplots_adjust(hspace=0.32, wspace=0.28,
                        top=0.97, bottom=0.04, left=0.14, right=0.97)

    log.info("Player analysis 1/4: value leaders ...")
    plot_value_leaders(axes[0, 0], df_pred, round_num)

    log.info("Player analysis 2/4: bye schedule ...")
    plot_squad_bye_schedule(axes[0, 1], round_num)

    log.info("Player analysis 3/4: position depth ...")
    plot_position_depth(axes[1, 0], df_pred, round_num)

    log.info("Player analysis 4/4: avg vs predicted ...")
    plot_avg_vs_predicted(axes[1, 1], df_pred, round_num)

    out_path = OUTPUT_DIR / "player_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out_path)


# ── Main entry point ────────────────────────────────────────────────────────

def run_visualisation(df_feat=None, df_pred=None, round_num=1, season_state=None):
    """
    Generate all visualisation reports.

    Args:
        df_feat: Feature-engineered DataFrame (if None, loads from master CSV)
        df_pred: 2026 predictions DataFrame (avoids re-computation)
        round_num: Current round number
        season_state: Output from run_season_plan() (optional, for bye charts)
    """
    log.info("=" * 55)
    log.info("  GENERATING VISUALISATION REPORTS")
    log.info("=" * 55)

    # Always prepare validation split for model performance charts
    log.info("Preparing validation data ...")
    df_feat_local, df_val, X_train, X_val, y_train, y_val, feature_cols = _prepare_data()

    # Use provided df_feat for 2026-specific charts if available
    df_for_pred = df_feat if df_feat is not None else df_feat_local

    log.info("Loading model ...")
    model = _load_model()

    # ── Page 1: Model Performance (4x2) ─────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(16, 28))
    fig.suptitle(
        "NRL Supercoach Optimizer --- Model Performance Report",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.30, top=0.95, bottom=0.03,
                        left=0.08, right=0.96)

    log.info("Chart 1/8: Actual vs Predicted ...")
    plot_actual_vs_predicted(axes[0, 0], model, X_val, y_val, df_val)

    log.info("Chart 2/8: Residuals ...")
    plot_residuals(axes[0, 1], model, X_val, y_val, df_val)

    log.info("Chart 3/8: Per-position error ...")
    plot_per_position_error(axes[1, 0], model, X_val, y_val, df_val)

    log.info("Chart 4/8: Error by price tier ...")
    plot_error_by_price_tier(axes[1, 1], model, X_val, y_val, df_val)

    log.info("Chart 5/8: Position box plot ...")
    plot_position_boxplot(axes[2, 0], df_for_pred, df_pred=df_pred)

    log.info("Chart 6/8: Top scorers by position ...")
    plot_top_scorers_by_position(axes[2, 1], df_for_pred, df_pred=df_pred)

    log.info("Chart 7/8: Price vs Predicted ...")
    plot_price_vs_predicted(axes[3, 0], df_for_pred, df_pred=df_pred,
                            round_num=round_num)

    log.info("Chart 8/8: Squad bar chart ...")
    plot_squad_bar(axes[3, 1], round_num=round_num)

    out_path = OUTPUT_DIR / "model_performance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", out_path)

    # ── Page 2: Bye Analysis (2x2) ──────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(20, 12))
    fig2.suptitle(
        "NRL Supercoach Optimizer --- Bye Round Analysis",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92, bottom=0.06,
                        left=0.06, right=0.96)

    log.info("Bye Chart 1/4: Bye schedule heatmap ...")
    plot_bye_heatmap(axes2[0, 0])

    log.info("Bye Chart 2/4: Round-by-round scoring ...")
    plot_round_scoring(axes2[0, 1], season_state=season_state)

    log.info("Bye Chart 3/4: Trade timeline ...")
    plot_trade_timeline(axes2[1, 0], season_state=season_state)

    log.info("Bye Chart 4/4: Squad availability ...")
    plot_squad_availability(axes2[1, 1], season_state=season_state)

    out_path2 = OUTPUT_DIR / "bye_analysis.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info("Saved -> %s", out_path2)

    # ── Page 3: Player Analysis (2x2) ───────────────────────────────────────
    log.info("Generating player analysis page ...")
    run_player_analysis(df_pred=df_pred, round_num=round_num)

    log.info("Visualisation complete.")


# ── Standalone ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_visualisation()

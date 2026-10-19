"""
visualise.py — every chart the project draws, in one consistent style.

Weekly pipeline:   plot_squad, plot_season_plan
Season review:     plot_model_benchmark, plot_error_by_round, plot_calibration,
                   plot_strategy_backtest, plot_season_simulation, plot_team_value,
                   plot_player_predictions, plot_position_analysis, plot_top_players,
                   plot_team_of_season, plot_value_picks

Style: light chart surface, hairline grids, thin marks, a fixed categorical
order (validated for colour-vision deficiency), text in ink colours only.
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

log = logging.getLogger(__name__)

# ── Design tokens ─────────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW = SERIES[:4]
DE_EMPHASIS = "#c9c8c1"

ROLE_COLOURS = {"Starting": BLUE, "Bench": AQUA, "Flex": SERIES[6], "Reserve": DE_EMPHASIS}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "text.color": INK,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
        "axes.labelcolor": INK_2, "axes.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


def _titles(fig, title: str, subtitle: str | None = None, y: float = 0.985):
    """Left-aligned headline + one-line subtitle, the chart's takeaway."""
    fig.text(0.02, y, title.replace("$", r"\$"), fontsize=15, fontweight="semibold",
             color=INK, ha="left", va="top")
    if subtitle:
        gap = 0.36 / fig.get_figheight()          # fixed 0.36in below the title
        fig.text(0.02, y - gap, subtitle.replace("$", r"\$"), fontsize=10.5,
                 color=INK_2, ha="left", va="top")


def _footnote(fig, text: str):
    fig.text(0.02, 0.012, text.replace("$", r"\$"), fontsize=8.5, color=MUTED,
             ha="left", va="bottom")


def _save(fig, path: Path, dpi: int = 160):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log.info("Chart saved -> %s", path)


def short_name(name: str) -> str:
    """'Cleary, Nathan' → 'N. Cleary'."""
    if "," in name:
        last, first = [s.strip() for s in name.split(",", 1)]
        return f"{first[:1]}. {last}" if first else last
    return name


# ── Weekly charts ─────────────────────────────────────────────────────────────

def plot_squad(result: dict, round_num: int, path: Path):
    """The optimizer's 26-man squad, predicted points by player and role."""
    apply_style()
    rows = []
    for group, role in [("starting_13", "Starting"), ("bench_4", "Bench"),
                        ("flex_1", "Flex"), ("reserves_8", "Reserve")]:
        for p in result.get(group, []):
            rows.append({"name": short_name(p["player_name"]), "role": role,
                         "pos": p.get("assigned_position", ""), "team": p.get("team", ""),
                         "pred": float(p.get("predicted_points", 0)),
                         "price": float(p.get("price_usd", p.get("price", 0)) or 0)})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df = df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 9.5))
    fig.subplots_adjust(left=0.30, right=0.93, top=0.86, bottom=0.07)
    ax.barh(df.index, df["pred"], height=0.62,
            color=[ROLE_COLOURS[r] for r in df["role"]])
    ax.set_yticks(df.index, [f"{n}  ·  {p} {t}" for n, p, t in
                             zip(df["name"], df["pos"], df["team"])], fontsize=9)
    for i, v in enumerate(df["pred"]):
        ax.text(v + 0.8, i, f"{v:.0f}", va="center", fontsize=8.5, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Predicted points")
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in ROLE_COLOURS.values()]
    ax.legend(handles, ROLE_COLOURS.keys(), loc="lower right", ncols=4,
              bbox_to_anchor=(1.0, 1.0))
    scoring = df.loc[df["role"] != "Reserve", "pred"].sum()
    _titles(fig, f"Round {round_num} squad",
            f"Scoring 18 projected at {scoring:,.0f} pts  ·  "
            f"total price ${result.get('total_price', 0) / 1e6:.2f}M")
    _save(fig, path)


def plot_season_plan(state: dict, path: Path):
    """Projected scoring-18 total per round with the bye load underneath."""
    rr = state.get("round_results", [])
    if not rr:
        return
    apply_style()
    df = pd.DataFrame({"round": [r["round"] for r in rr],
                       "points": [r["projected_points"] for r in rr],
                       "byes": [r["n_bye_teams"] for r in rr],
                       "trades": [r["n_trades"] for r in rr]})
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.09, hspace=0.12)
    ax.plot(df["round"], df["points"], color=BLUE, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_ylabel("Projected points")
    ax2.bar(df["round"], df["byes"], width=0.6, color=ORANGE)
    ax2.set_ylabel("Teams on bye")
    ax2.set_xlabel("Round")
    ax2.set_xticks(df["round"])
    _titles(fig, "Season plan: projected score by round",
            f"{state.get('trades_used', 0)} trades scheduled  ·  "
            f"{state.get('total_projected', 0):,.0f} projected points")
    _save(fig, path)


# ── Season review: model ──────────────────────────────────────────────────────

def plot_model_benchmark(metrics: dict, n_games: int, season: int, path: Path,
                         nn_label: str, oracle_label: str):
    """MAE and rank correlation per method; the oracle is a reference line."""
    apply_style()
    methods = [m for m in metrics if m != oracle_label]
    methods.sort(key=lambda m: metrics[m]["mae"], reverse=True)
    colours = [BLUE if m == nn_label else DE_EMPHASIS for m in methods]
    y = np.arange(len(methods))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.76, bottom=0.17, wspace=0.08)

    mae = [metrics[m]["mae"] for m in methods]
    ax1.barh(y, mae, height=0.55, color=colours)
    oracle = metrics[oracle_label]["mae"]
    ax1.axvline(oracle, color=INK_2, lw=1, ls=(0, (3, 3)))
    ax1.text(oracle, len(methods) - 0.45, f" hindsight oracle {oracle:.1f}",
             fontsize=8.5, color=INK_2, va="bottom")
    for i, v in enumerate(mae):
        ax1.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9.5,
                 color=INK if methods[i] == nn_label else INK_2,
                 fontweight="semibold" if methods[i] == nn_label else "normal")
    ax1.set_xlim(0, max(mae) * 1.15)
    ax1.set_xlabel("Mean absolute error, points per game  (lower is better)")
    ax1.set_yticks(y, methods)
    ax1.grid(axis="y", visible=False)

    rho = [metrics[m]["rank_corr"] for m in methods]
    ax2.barh(y, rho, height=0.55, color=colours)
    for i, v in enumerate(rho):
        ax2.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=9.5,
                 color=INK if methods[i] == nn_label else INK_2,
                 fontweight="semibold" if methods[i] == nn_label else "normal")
    ax2.set_xlim(0, max(rho) * 1.18)
    ax2.set_xlabel("Rank correlation with actual scores, per round  (higher is better)")
    ax2.grid(axis="y", visible=False)

    base = metrics["Season average"]["mae"]
    best = metrics[nn_label]["mae"]
    _titles(fig, f"Predicting every {season} game before it was played",
            f"Trained on 2022–{season - 1} only, tested on all {n_games:,} games of {season}. "
            f"{nn_label}: the lowest error of any method, {100 * (base - best) / base:.0f}% below "
            "the season-average heuristic.")
    _footnote(fig, "Hindsight oracle = each player's true 2026 average over the predicted "
                   "games; unattainable, it shows how much of the error is game-day noise.")
    _save(fig, path)


def plot_error_by_round(test: pd.DataFrame, path: Path, nn_label: str):
    """MAE per round: where does the model's edge come from?"""
    apply_style()
    series = [(nn_label, BLUE), ("Season average", ORANGE), ("Price-implied", AQUA)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.12)
    for name, colour in series:
        err = (test[f"pred_{name}"] - test["next_score"]).abs()
        by_round = err.groupby(test["next_round"]).mean()
        ax.plot(by_round.index, by_round.values, color=colour, label=name)
    ax.set_xlabel("Round of the 2026 season")
    ax.set_ylabel("MAE (points)")
    ax.set_xticks(range(2, 28))
    ax.set_xlim(1.5, 27.5)
    ax.legend(loc="upper right", ncols=3, bbox_to_anchor=(1.0, 1.1))
    _titles(fig, "The edge is biggest early in the season",
            "When a player has only played a few games, the season average is noisy; "
            "the model leans on last season and price instead.")
    _save(fig, path)


def plot_calibration(test: pd.DataFrame, path: Path, nn_label: str):
    """Predicted vs actual, binned: is a predicted 70 really worth 70?"""
    apply_style()
    pred = test[f"pred_{nn_label}"]
    bins = pd.qcut(pred, 15, duplicates="drop")
    grp = test.groupby(bins, observed=True)
    x = grp[f"pred_{nn_label}"].mean()
    y = grp["next_score"].mean()
    lo = grp["next_score"].quantile(0.25)
    hi = grp["next_score"].quantile(0.75)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.80, bottom=0.11)
    lim = (0, max(hi.max(), x.max()) * 1.05)
    ax.plot(lim, lim, color=BASELINE, lw=1)
    ax.text(lim[1] * 0.97, lim[1] * 0.90, "perfect calibration", fontsize=8.5,
            color=MUTED, ha="right", rotation=38)
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.10, lw=0)
    ax.plot(x, y, color=BLUE, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted score (15 equal-size bins)")
    ax.set_ylabel("Actual score")
    _titles(fig, "Predictions are well calibrated",
            "Mean actual score per prediction bin; band = middle 50% of outcomes.")
    _save(fig, path)


# ── Season review: squads ─────────────────────────────────────────────────────

def plot_strategy_backtest(weekly: pd.DataFrame, path: Path, season: int,
                           highlight: str, baseline: str, reference: str):
    """Weekly fresh-squad backtest: running advantage over the baseline
    heuristic (left) and season totals (right)."""
    apply_style()
    strategies = [c for c in weekly.columns if c not in (baseline, reference)]
    colours = {highlight: BLUE}
    for name, colour in zip([s for s in strategies if s != highlight], [ORANGE, AQUA]):
        colours[name] = colour

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.13, wspace=0.42)

    rounds = weekly.index
    ax.axhline(0, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    ax.text(rounds[0], 0, f"{baseline} heuristic  ", fontsize=8.5, color=INK_2,
            ha="left", va="bottom")
    for name in strategies:
        lead = (weekly[name] - weekly[baseline]).cumsum()
        ax.plot(rounds, lead, color=colours[name], lw=2.6 if name == highlight else 2)
        ax.plot(rounds[-1], lead.iloc[-1], "o", color=colours[name], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{lead.iloc[-1]:+,.0f}\n{name}", (rounds[-1], lead.iloc[-1]),
                    xytext=(9, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=INK if name == highlight else INK_2,
                    fontweight="semibold" if name == highlight else "normal")
    ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 4.5)
    ax.spines["bottom"].set_bounds(rounds[0] - 0.5, rounds[-1] + 0.5)
    ax.set_xticks(range(rounds[0], rounds[-1] + 1, 2))
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Running points ahead of {baseline.lower()}")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+,.0f}"))

    totals = weekly.sum().drop(reference).sort_values()
    best = weekly[reference].sum()
    y = np.arange(len(totals))
    ax2.barh(y, totals.values, height=0.45,
             color=[colours.get(n, DE_EMPHASIS) for n in totals.index])
    for i, (name, v) in enumerate(totals.items()):
        ax2.text(v * 0.985, i, f"{v:,.0f}", ha="right", va="center", fontsize=9.5,
                 color=SURFACE, fontweight="semibold")
    ax2.set_yticks(y, totals.index)
    ax2.set_xlim(0, totals.max() * 1.05)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("Season points (best 18 each round)")
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))

    nn = weekly[highlight]
    won = {n: int((nn > weekly[n]).sum()) for n in weekly.columns if n not in (highlight, reference)}
    beats = ", ".join(f"{n.lower()} in {w} of {len(weekly)} rounds" for n, w in won.items())
    _titles(fig, "Which predictor picks better teams?",
            f"A controlled test: every round of {season}, each method picks a brand-new squad "
            f"under the $11.95M cap. The model's squad beat {beats}.")
    _footnote(fig, "Not a real-game simulation (that allows 2 trades a round; see the season "
                   "simulation). Pool = players named that round, at that round's prices; the "
                   f"model only sees earlier games. Hindsight-perfect: {best:,.0f}.")
    _save(fig, path)


# Starting XIII as a rugby league attacking shape: (position, x, y).
_FIELD_LAYOUT = [
    ("FLB", 0.50, 0.86),
    ("CTW", 0.14, 0.69), ("CTW", 0.38, 0.69), ("CTW", 0.62, 0.69), ("CTW", 0.86, 0.69),
    ("5/8", 0.37, 0.52), ("HFB", 0.63, 0.52),
    ("2RF", 0.20, 0.35), ("2RF", 0.50, 0.35), ("2RF", 0.80, 0.35),
    ("FRF", 0.20, 0.17), ("HOK", 0.50, 0.17), ("FRF", 0.80, 0.17),
]


def plot_team_of_season(starters: pd.DataFrame, bench: pd.DataFrame, path: Path,
                        season: int, total_price: float):
    """Field graphic of the best squad money could buy, with hindsight."""
    apply_style()
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_axes([0.03, 0.03, 0.72, 0.84])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                fc="#f2f6ee", ec=GRID, lw=1))
    for yy in (0.10, 0.27, 0.44, 0.61, 0.78, 0.95):
        ax.plot([0.03, 0.97], [yy, yy], color="#dfe7d6", lw=1, zorder=0)

    def card(x, y, p):
        w, h = 0.205, 0.105
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=SURFACE, ec=GRID, lw=1, zorder=2))
        ax.add_patch(Rectangle((x - w / 2, y + h / 2 - 0.012), w, 0.012,
                               color=BLUE, zorder=3, lw=0))
        ax.text(x, y + 0.017, short_name(p["player_name"]), ha="center", va="center",
                fontsize=10, fontweight="semibold", color=INK, zorder=4)
        ax.text(x, y - 0.012, f"{p['assigned_position']}  ·  {p['team']}",
                ha="center", va="center", fontsize=8.5, color=INK_2, zorder=4)
        ax.text(x, y - 0.035, f"{p['season_avg']:.0f} avg  ·  ${p['start_price'] / 1000:.0f}K",
                ha="center", va="center", fontsize=8.5, color=INK_2, zorder=4)

    queues = {pos: [p for _, p in grp.sort_values("season_avg", ascending=False).iterrows()]
              for pos, grp in starters.groupby("assigned_position")}
    for pos, x, y in _FIELD_LAYOUT:
        if queues.get(pos):
            card(x, y, queues[pos].pop(0))

    side = fig.add_axes([0.77, 0.03, 0.21, 0.84])
    side.axis("off")
    side.text(0, 1.0, "Interchange & reserves", fontsize=11, fontweight="semibold",
              color=INK, va="top")
    for i, (_, p) in enumerate(bench.iterrows()):
        yy = 0.95 - i * 0.068
        side.add_patch(Rectangle((0, yy - 0.022), 0.025, 0.04,
                                 color=ROLE_COLOURS.get(p["role"], DE_EMPHASIS), lw=0))
        side.text(0.06, yy + 0.008, short_name(p["player_name"]), fontsize=9.5,
                  color=INK, va="center")
        side.text(0.06, yy - 0.018, f"{p['assigned_position']} · {p['team']} · "
                                    f"{p['season_avg']:.0f} avg", fontsize=8.2,
                  color=INK_2, va="center")
    side.set_ylim(0, 1)
    side.set_xlim(0, 1)

    _titles(fig, f"{season} Team of the Season, under the salary cap",
            f"The optimizer fed actual {season} points: the best legal 26 at start-of-season "
            f"prices (${total_price / 1e6:.2f}M of $11.95M).", y=0.975)
    _save(fig, path)


def plot_value_picks(players: pd.DataFrame, path: Path, season: int, n_label: int = 10):
    """Start price vs season average; the best points-per-dollar buys labelled."""
    apply_style()
    df = players.copy()
    df["value"] = df["season_avg"] / (df["start_price"] / 100_000)
    top = df.nlargest(n_label, "value")

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.11)
    ax.scatter(df["start_price"] / 1000, df["season_avg"], s=22, color=DE_EMPHASIS,
               edgecolors=SURFACE, linewidths=0.8, zorder=2)
    ax.scatter(top["start_price"] / 1000, top["season_avg"], s=60, color=BLUE,
               edgecolors=SURFACE, linewidths=2, zorder=3)
    slope, intercept = np.polyfit(df["start_price"] / 1000, df["season_avg"], 1)
    xs = np.array([df["start_price"].min(), df["start_price"].max()]) / 1000
    ax.plot(xs, slope * xs + intercept, color=BASELINE, lw=1.2, zorder=1)
    ax.text(xs[1], slope * xs[1] + intercept + 2, "what the price implied",
            fontsize=8.5, color=MUTED, ha="right")
    # Labels in a spaced column right of the cheap cluster, with leader lines.
    label_x = top["start_price"].max() / 1000 + 45
    label_y = None
    for _, r in top.sort_values("season_avg", ascending=False).iterrows():
        label_y = r["season_avg"] if label_y is None else min(r["season_avg"], label_y - 4.2)
        ax.annotate(f"{short_name(r['player_name'])} ({r['primary_pos']}, "
                    f"{r['season_avg']:.0f} avg)",
                    (r["start_price"] / 1000, r["season_avg"]), xytext=(label_x, label_y),
                    fontsize=8.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.7,
                                    shrinkA=0, shrinkB=4))
    ax.set_xlabel("Start-of-season price ($K)")
    ax.set_ylabel(f"{season} average (points per game)")
    _titles(fig, f"The {season} bargains",
            f"Players with 10+ games. Blue = the {n_label} best points-per-dollar buys at "
            "start-of-season prices, the cheap risers that fund a winning squad.")
    _save(fig, path)


# ── Season review: full-season simulation ─────────────────────────────────────

LIGHT_BLUE = "#86b6ef"
POSITIONS = ["HOK", "FRF", "2RF", "HFB", "5/8", "CTW", "FLB"]
POSITION_NAMES = {"HOK": "Hooker", "FRF": "Front row", "2RF": "Second row",
                  "HFB": "Halfback", "5/8": "Five-eighth", "CTW": "Centre / wing",
                  "FLB": "Fullback"}


def _primary(positions: pd.Series) -> pd.Series:
    return positions.fillna("").astype(str).str.split("|").str[0]


def plot_season_simulation(results: dict, highlight: str, path: Path, season: int,
                           compare: str, big_byes=(12, 15, 18)):
    """Round scores of the simulated team (left) and running totals (right)."""
    apply_style()
    main = results[highlight]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.06, right=0.8, top=0.78, bottom=0.13, wspace=0.22)

    r = main.index
    top = main["points"].max() * 1.12
    for b in big_byes:
        ax.axvspan(b - 0.5, b + 0.5, color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.text(b, top * 0.97, "bye", ha="center", va="top", fontsize=8, color=INK_2)
    ax.bar(r, main["lineup_points"], width=0.7, color=BLUE, label="Best 18 on the field")
    ax.bar(r, main["captain_points"], bottom=main["lineup_points"], width=0.7,
           color=LIGHT_BLUE, label="Captain's bonus")
    avg = main["points"].mean()
    ax.axhline(avg, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(r[-1] + 0.7, avg, f"avg\n{avg:,.0f}", fontsize=8.5, color=INK_2, va="center")
    ax.set_xticks(range(1, r[-1] + 1, 2))
    ax.set_xlim(0.4, r[-1] + 0.6)
    ax.set_xlabel("Round")
    ax.set_ylabel("Points")
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0, 1.12))

    colours = dict(zip(results, [BLUE, ORANGE, AQUA, YELLOW]))
    ends = sorted(((v["points"].sum(), k) for k, v in results.items()), reverse=True)
    for name, res in results.items():
        cum = res["points"].cumsum()
        ax2.plot(cum.index, cum.values, color=colours[name],
                 lw=2.6 if name == highlight else 2)
    gap = ends[0][0] * 0.09                # two-line labels need ~9% of the axis
    label_y = None
    for total, name in ends:
        label_y = total if label_y is None else min(total, label_y - gap)
        ax2.annotate(f"{total:,.0f}\n{name}", (r[-1], total), xytext=(r[-1] + 1.2, label_y),
                     fontsize=8.5, va="center", annotation_clip=False,
                     color=INK if name == highlight else INK_2,
                     fontweight="semibold" if name == highlight else "normal",
                     arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.7))
    ax2.set_xlim(1, r[-1])
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Season points so far")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))

    total = main["points"].sum()
    gain = total - results[compare]["points"].sum()
    _titles(fig, f"The full {season} season under Supercoach rules: {total:,.0f} points",
            f"Pre-season squad, then trades every round (46, up to 5 boosted rounds), with real "
            f"prices, team lists and byes; captain doubled. {gain:+,.0f} vs. the original trading strategy.")
    _footnote(fig, "Every decision uses only what was known before that round; points are the "
                   f"real {season} scores. Scoring 18 = 13 starters + 4 bench + 1 flex "
                   f"(best 13 count in the bye rounds).")
    _save(fig, path)


def plot_team_value(rounds: pd.DataFrame, path: Path, season: int, cap: float):
    """Squad value + cash in the bank, and trades used, round by round."""
    apply_style()
    value = (rounds["squad_value"] + rounds["bank"]) / 1e6
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(left=0.08, right=0.9, top=0.8, bottom=0.11, hspace=0.12)
    ax.axhline(cap / 1e6, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(value.index[-1] + 0.5, cap / 1e6, "starting\ncap", fontsize=8.5,
            color=INK_2, va="center")
    ax.plot(value.index, value.values, color=BLUE, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    peak = value.idxmax()
    ax.annotate(f"peak ${value.max():.2f}M", (peak, value.max()), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_ylabel("Team value ($M)")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax2.bar(rounds.index, rounds["trades"], width=0.6, color=ORANGE)
    ax2.set_ylabel("Trades")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_xlabel("Round")
    ax2.set_xticks(range(1, rounds.index[-1] + 1, 2))
    _titles(fig, f"Trading grew the team's value from ${cap / 1e6:.2f}M to ${value.iloc[-1]:.2f}M",
            "Squad value plus cash in the bank (the money available for trades). "
            + (f"All trades were used by round {rounds.index[rounds['trades_left'] == 0][0]}."
               if (rounds["trades_left"] == 0).any() else ""))
    _save(fig, path)


# ── Season review: players and positions ──────────────────────────────────────

def _player_table(test: pd.DataFrame, nn_label: str, min_games: int = 10) -> pd.DataFrame:
    df = test.assign(pos=_primary(test["positions"]))
    agg = df.groupby("player_name").agg(
        pos=("pos", "last"), team=("team", "last"), games=("next_score", "size"),
        actual=("next_score", "mean"), predicted=(f"pred_{nn_label}", "mean"))
    return agg[agg["games"] >= min_games]


def plot_player_predictions(test: pd.DataFrame, path: Path, nn_label: str, season: int):
    """Each player's average prediction vs average actual score, by position."""
    apply_style()
    players = _player_table(test, nn_label)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.08, hspace=0.3, wspace=0.08)
    lim = (0, max(players["actual"].max(), players["predicted"].max()) * 1.08)
    for ax, pos in zip(axes.flat, POSITIONS):
        sub = players[players["pos"] == pos]
        ax.plot(lim, lim, color=BASELINE, lw=1)
        ax.scatter(sub["predicted"], sub["actual"], s=22, color=BLUE, alpha=0.75,
                   edgecolors=SURFACE, linewidths=0.6)
        offsets = [(4, 3), (4, -10), (-4, 6)]
        for (name, r), (dx, dy) in zip(sub.nlargest(3, "actual").iterrows(), offsets):
            ax.annotate(short_name(name), (r["predicted"], r["actual"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=7.5, color=INK,
                        ha="left" if dx > 0 else "right")
        corr = sub["predicted"].corr(sub["actual"])
        ax.set_title(f"{POSITION_NAMES[pos]}  ·  r = {corr:.2f}", fontsize=10,
                     color=INK, loc="left")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    note = axes.flat[-1]
    note.axis("off")
    note.text(0.04, 0.9, "How to read it", fontsize=10.5, fontweight="semibold", color=INK,
              transform=note.transAxes)
    note.text(0.04, 0.8, "Each dot is a player (10+ games).\n"
                         "Across: the model's average prediction,\n"
                         "made before each game.\n"
                         "Up: what the player actually averaged.\n\n"
                         "On the line = predicted exactly.\n"
                         "Above the line = beat expectations.\n"
                         "r = correlation within the position.",
              fontsize=9, color=INK_2, va="top", transform=note.transAxes, linespacing=1.5)
    for ax in axes[1]:
        ax.set_xlabel("Predicted (average)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Actual (average)")
    corr_all = players["predicted"].corr(players["actual"])
    _titles(fig, f"Every player's {season} season: predicted vs actual",
            f"Predictions made before each game, averaged per player. Across "
            f"{len(players)} regulars the correlation is {corr_all:.2f}.")
    _save(fig, path)


def plot_position_analysis(test: pd.DataFrame, totals: pd.DataFrame, path: Path,
                           nn_label: str, season: int):
    """Scoring by position (left) and prediction error by position (right)."""
    apply_style()
    regulars = totals[totals["games_played"] >= 10]
    regulars = regulars.assign(pos=_primary(regulars["positions"]))
    df = test.assign(pos=_primary(test["positions"]))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.12, wspace=0.08)
    y = np.arange(len(POSITIONS))[::-1]
    rng = np.random.default_rng(0)
    for yy, pos in zip(y, POSITIONS):
        vals = regulars.loc[regulars["pos"] == pos, "avg_points"].values
        ax.scatter(vals, yy + rng.uniform(-0.22, 0.22, len(vals)), s=14, color=DE_EMPHASIS,
                   edgecolors=SURFACE, linewidths=0.5, zorder=2)
        ax.plot([np.median(vals)] * 2, [yy - 0.32, yy + 0.32], color=BLUE, lw=3, zorder=3)
        ax.text(vals.max() + 2, yy, f"median {np.median(vals):.0f}", fontsize=8.5,
                color=INK_2, va="center")
    ax.set_yticks(y, [POSITION_NAMES[p] for p in POSITIONS])
    ax.set_xlabel(f"{season} average per game (each dot a player, 10+ games)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=ax.get_xlim()[1] + 15)

    h = 0.36
    series = [(nn_label, f"pred_{nn_label}", BLUE),
              ("Season-average heuristic", "pred_Season average", ORANGE)]
    for off, (label, col, colour) in zip((h / 2, -h / 2), series):
        err = (df[col] - df["next_score"]).abs()
        mae = err.groupby(df["pos"]).mean()
        vals = [mae.get(p, np.nan) for p in POSITIONS]
        ax2.barh(y + off, vals, height=h * 0.9, color=colour, label=label)
        for yy, v in zip(y + off, vals):
            ax2.text(v + 0.2, yy, f"{v:.1f}", va="center", fontsize=8, color=INK_2)
    ax2.set_xlabel("Prediction error per game (MAE, points)")
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.12), ncols=2)
    _titles(fig, "Position analysis",
            "Left: how each position scored. Right: how far off the predictions were, "
            "model vs. the season-average heuristic.")
    _save(fig, path)


def plot_top_players(test: pd.DataFrame, totals: pd.DataFrame, path: Path,
                     nn_label: str, season: int, n: int = 4):
    """Top scorers per position: actual average (bar) vs model expectation (tick)."""
    apply_style()
    info = totals.set_index("player_name")[["start_price", "price", "avg_points",
                                           "games_played"]]
    players = _player_table(test, nn_label).join(info)
    rows = []
    for pos in POSITIONS:
        best = players[players["pos"] == pos].nlargest(n, "avg_points")
        rows += [(pos, name, r) for name, r in best.iterrows()]

    fig, ax = plt.subplots(figsize=(12, 11.5))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.9, bottom=0.06)
    ys, headers, yy, prev = [], [], 0.0, None
    for pos, _, _ in rows:
        if pos != prev:
            if prev is not None:
                yy -= 0.8
            headers.append((yy + 0.62, POSITION_NAMES[pos].upper()))
            prev = pos
        ys.append(yy)
        yy -= 1
    ys = np.array(ys)
    actual = np.array([r["avg_points"] for _, _, r in rows])
    pred = np.array([r["predicted"] for _, _, r in rows])
    ax.barh(ys, actual, height=0.62, color=BLUE)
    ax.scatter(pred, ys, marker="|", s=240, color=INK, linewidths=2.2, zorder=3)
    for y0, label in headers:
        ax.text(0, y0, label, fontsize=8.5, fontweight="semibold", color=MUTED,
                ha="left", va="bottom")
    for y0, (_, _, r) in zip(ys, rows):
        ax.text(max(r["avg_points"], r["predicted"]) + 1.5, y0,
                f"{r['avg_points']:.0f} avg · {int(r['games_played'])} games · "
                rf"\${r['start_price'] / 1000:.0f}K → \${r['price'] / 1000:.0f}K",
                fontsize=8.5, color=INK_2, va="center")
    ax.set_yticks(ys, [f"{short_name(name)} ({r['team']})" for _, name, r in rows],
                  fontsize=9)
    ax.set_xlim(0, actual.max() * 1.55)
    ax.set_xlabel(f"{season} average points per game")
    ax.grid(axis="y", visible=False)
    handles = [Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Line2D([], [], color=INK, marker="|", ls="", markersize=12, mew=2)]
    ax.legend(handles, ["Actual average", "Model's average prediction"],
              loc="lower right")
    _titles(fig, f"The best of {season}, by position",
            "Top scorers at each position, what the model expected of them, and how their "
            "price moved (start → end of season).", y=0.985)
    _save(fig, path)


def plot_human_comparison(simulated: dict, humans: dict, highlight: str, path: Path,
                          season: int):
    """Season totals of the simulated strategies alongside real coaches."""
    apply_style()
    rows = [(name, total, "human") for name, total in humans.items()]
    rows += [(name, total, "sim") for name, total in simulated.items()]
    rows.sort(key=lambda r: r[1])
    colour = {"human": DE_EMPHASIS, "sim": LIGHT_BLUE}

    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.subplots_adjust(left=0.3, right=0.95, top=0.8, bottom=0.11)
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.6,
            color=[BLUE if r[0] == highlight else colour[r[2]] for r in rows])
    for yy, (name, total, _) in zip(y, rows):
        strong = name == highlight
        ax.text(total + 300, yy, f"{total:,.0f}", va="center", fontsize=9.5,
                color=INK if strong else INK_2, fontweight="semibold" if strong else "normal")
    ax.set_yticks(y, [("System: " + r[0]) if r[2] == "sim" else r[0] for r in rows],
                  fontsize=9.5)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.1)
    ax.set_xlabel(f"{season} season total (points)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.grid(axis="y", visible=False)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (DE_EMPHASIS, BLUE)]
    ax.legend(handles, [f"Real {season} coaches", "Simulated by this system"],
              loc="lower right")

    total = simulated[highlight]
    avg = humans.get("Average coach")
    top10 = humans.get("Top 10% of coaches (avg)")
    _titles(fig, f"Where the system would have finished in {season}",
            f"{total:,.0f} points: {total - avg:+,.0f} on the average coach, but "
            f"{top10 - total:,.0f} short of the top-10% average.")
    _footnote(fig, "Real-coach figures from the SuperCoach app's season scoring trends. "
                   "Simulated teams follow the same squad, trade and captain rules.")
    _save(fig, path)


def plot_strategy_improvement(by_season: dict, original: str, improved: str,
                              test_season: int, path: Path):
    """Original vs improved trading strategy, per season (dev seasons + test)."""
    apply_style()
    seasons = sorted(by_season)
    x = np.arange(len(seasons))
    w = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    for off, (label, colour) in zip((-w / 2, w / 2), [(original, ORANGE), (improved, BLUE)]):
        vals = [by_season[y][label] for y in seasons]
        ax.bar(x + off, vals, width=w * 0.92, color=colour, label=label)
        for xx, v in zip(x + off, vals):
            ax.text(xx, v + 250, f"{v:,.0f}", ha="center", fontsize=9, color=INK_2)
    top = max(max(d.values()) for d in by_season.values())
    for xx, y in zip(x, seasons):
        gain = by_season[y][improved] - by_season[y][original]
        ax.text(xx, top * 1.07, f"{gain:+,.0f}", ha="center", fontsize=10.5,
                fontweight="semibold", color=INK)
    ax.set_xticks(x, [f"{y}\n{'test season' if y == test_season else 'development'}"
                      for y in seasons])
    ax.set_ylim(0, top * 1.14)
    ax.set_ylabel("Season points")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0, 1.1))
    wins = all(by_season[y][improved] > by_season[y][original] for y in seasons)
    _titles(fig, "The improved strategy wins in every season" if wins
            else "Improved vs. original strategy, season by season",
            "Every setting was tuned on 2024 and 2025 (models trained on earlier years); "
            f"{test_season} is the held-out test season and was never used for tuning.")
    _save(fig, path)

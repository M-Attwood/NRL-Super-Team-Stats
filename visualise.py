"""
visualise.py — every chart the project draws, in one consistent style.

Weekly pipeline:   plot_squad, plot_season_plan
Season review:     plot_model_benchmark, plot_error_by_round, plot_calibration,
                   plot_strategy_backtest, plot_team_of_season, plot_value_picks

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
        "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
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
    fig.text(0.02, y, title, fontsize=15, fontweight="semibold", color=INK,
             ha="left", va="top")
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
            f"The network cuts error {100 * (base - best) / base:.0f}% vs. the season-average "
            "heuristic, level with a tuned linear model.")
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
    _titles(fig, f"Picking a fresh squad every round of {season}",
            f"Same optimizer and $11.95M cap, three ways of valuing players. "
            f"The model's squad beat {beats}.")
    _footnote(fig, f"Pool each round = players who took the field, at that round's prices; the "
                   f"model only sees earlier games. Hindsight-perfect picks score {best:,.0f} "
                   "for reference. Captaincy ignored.")
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

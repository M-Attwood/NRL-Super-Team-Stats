"""
optimizer.py — PuLP linear program to select the optimal 26-player
NRL Supercoach squad.

Objective: maximise (0.75 × predicted_points + 0.20 × diversity_score
                     + 0.05 × versatility_score)
           minus a soft penalty for picking >3 players from one NRL club.

Constraints:
  - Total price  ≤ salary_cap (defaults to SALARY_CAP = $11,950,000;
                  callers may pass a higher mid-season value to reflect
                  earned price-rise allowance)
  - Squad size   = 26 players (25 position slots + 1 flex)
  - Position quotas: HOK=2, FRF=4, 2RF=6, HFB=2, 5/8=2, CTW=7, FLB=2
  - Multi-position players fill exactly one quota slot
  - Injury coverage: each starting position has ≥1 extra eligible player

Usage:
    python optimizer.py --predictions outputs/predictions_round_10.csv --round 10
"""

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pulp
import yaml

from paths import DATA_INPUTS, OUTPUTS_DIR

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SALARY_CAP = 11_950_000
SQUAD_SIZE = 26
HARD_MAX_PER_TEAM = 5      # Hard cap: never more than this many from one team
SOFT_MAX_PER_TEAM = 3      # Soft target: penalty applies above this threshold
PENALTY_PER_PLAYER = 0.15  # Objective penalty per player over the soft target

# Objective weights: (predicted points, team diversity, positional versatility).
# Diversity and versatility hedge season-long bye / injury risk.
SEASON_WEIGHTS = (0.75, 0.20, 0.05)
POINTS_ONLY = (1.0, 0.0, 0.0)   # single-round pick with known team lists
MIN_COVERAGE_DEPTH = 1     # Extra eligible players beyond quota, per starting position

INJURIES_PATH = DATA_INPUTS / "injuries.yaml"


@lru_cache(maxsize=1)
def _injury_list() -> tuple[tuple[str, int], ...]:
    """(player, available_from_round) pairs from data/inputs/injuries.yaml."""
    if not INJURIES_PATH.exists():
        return ()
    doc = yaml.safe_load(INJURIES_PATH.read_text(encoding="utf-8")) or {}
    return tuple((e["player"], int(e["available_from"]))
                 for e in doc.get("unavailable", []))


def unavailable_players(round_number: int) -> set[str]:
    """Players listed as injured / suspended for this round."""
    return {name for name, available_from in _injury_list()
            if round_number < available_from}


POSITION_QUOTAS = {
    "HOK": 2,
    "FRF": 4,
    "2RF": 6,
    "HFB": 2,
    "5/8": 2,
    "CTW": 7,
    "FLB": 2,
}
assert sum(POSITION_QUOTAS.values()) + 1 == SQUAD_SIZE, "Position quotas + 1 flex must equal squad size"

# Starting XIII, Bench 4, Reserves 8 — by position
STARTING_SLOTS = {"FLB": 1, "CTW": 4, "5/8": 1, "HFB": 1, "FRF": 2, "HOK": 1, "2RF": 3}
BENCH_SLOTS    = {"HOK": 1, "FRF": 2, "2RF": 1}
# Reserves = whatever remains after starters + bench

assert sum(STARTING_SLOTS.values()) == 13
assert sum(BENCH_SLOTS.values()) == 4

# Active (scoring) vs Reserve quotas — only 18 players earn points per round
ACTIVE_QUOTAS = {pos: STARTING_SLOTS.get(pos, 0) + BENCH_SLOTS.get(pos, 0)
                 for pos in POSITION_QUOTAS}
# HOK:2, FRF:4, 2RF:4, HFB:1, 5/8:1, CTW:4, FLB:1  (+1 FLEX = 18 scoring)
RESERVE_QUOTAS = {pos: POSITION_QUOTAS[pos] - ACTIVE_QUOTAS[pos]
                  for pos in POSITION_QUOTAS}
# HOK:0, FRF:0, 2RF:2, HFB:1, 5/8:1, CTW:3, FLB:1  (= 8 reserves)
RESERVE_WEIGHT = 0.10  # Reserves get 10% objective weight

assert sum(ACTIVE_QUOTAS.values()) + 1 == 18, "Active quotas + FLEX must equal 18"
assert sum(RESERVE_QUOTAS.values()) == 8, "Reserve quotas must equal 8"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]. Returns 0.5 if constant."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _compute_diversity_scores(df: pd.DataFrame) -> pd.Series:
    """
    Diversity score = 1 / (number of players from this team in the pool).
    Normalised to [0, 1] so it's on the same scale as predicted_points.
    """
    if "team" not in df.columns:
        return pd.Series(0.5, index=df.index)
    team_counts = df["team"].map(df["team"].value_counts())
    raw = 1.0 / team_counts.replace(0, 1)
    return _normalise(raw)


def _compute_versatility_scores(df: pd.DataFrame) -> pd.Series:
    """
    Versatility score based on number of eligible positions.
    Players with more eligible positions score higher.
    Normalised to [0, 1].
    """
    n_positions = df["_eligible"].map(len)
    raw = (n_positions - 1).clip(lower=0).astype(float)
    return _normalise(raw)


def _eligible_positions(positions_str: str) -> list[str]:
    """
    Parse pipe-separated positions string and return only those
    that exist in POSITION_QUOTAS.
    e.g. "2RF|FRF" → ["2RF", "FRF"]
    """
    if not isinstance(positions_str, str) or not positions_str.strip():
        return []
    parts = [p.strip() for p in positions_str.split("|")]
    return [p for p in parts if p in POSITION_QUOTAS]


# ── Core LP Solver ────────────────────────────────────────────────────────────

def select_team(df: pd.DataFrame, round_number: int = 1,
                salary_cap: int | None = None,
                apply_injuries: bool = True,
                weights: tuple[float, float, float] = SEASON_WEIGHTS,
                team_penalty: float = PENALTY_PER_PLAYER) -> dict:
    """
    Run the PuLP linear program and return the selected squad.

    df must contain: player_name, positions, team, price, predicted_points.
    apply_injuries=False skips the injuries.yaml exclusions (used by the
    season backtest, which compares strategies on equal footing).
    weights / team_penalty: objective weights for (points, diversity,
    versatility) and the per-player penalty above SOFT_MAX_PER_TEAM.

    Returns:
        {
            "starting_13": list of dicts,
            "bench_4":     list of dicts,
            "flex_1":      list of dicts,
            "reserves_8":  list of dicts,
            "total_price": int,
            "solver_status": str,
            "df":          selected players DataFrame,
        }
    """
    df = df.copy().reset_index(drop=True)

    if apply_injuries:
        out = unavailable_players(round_number)
        excluded = int(df["player_name"].isin(out).sum())
        df = df[~df["player_name"].isin(out)].reset_index(drop=True)
        if excluded:
            log.info("Round %d: excluded %d injured/suspended player(s)",
                     round_number, excluded)

    # Validate required columns
    required = {"player_name", "positions", "team", "predicted_points"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    # Use price_usd (original $) if available; fall back to price
    price_col = "price_usd" if "price_usd" in df.columns else "price"

    # Drop players with no valid position
    df["_eligible"] = df["positions"].apply(_eligible_positions)
    df = df[df["_eligible"].map(len) > 0].reset_index(drop=True)

    if len(df) < SQUAD_SIZE:
        raise ValueError(
            f"Only {len(df)} eligible players in pool — need at least {SQUAD_SIZE}"
        )

    players = df.index.tolist()

    # Pre-compute objective weights (versatility requires _eligible, computed above)
    norm_pred   = _normalise(df["predicted_points"].fillna(0))
    diversity   = _compute_diversity_scores(df)
    versatility = _compute_versatility_scores(df)
    w_points, w_diversity, w_versatility = weights
    objective_weight = (w_points * norm_pred + w_diversity * diversity
                        + w_versatility * versatility)

    # ── Decision variables ────────────────────────────────────────────────────
    prob = pulp.LpProblem("NRL_Supercoach_Optimizer", pulp.LpMaximize)

    # x[i] = 1 if player i is selected
    x = pulp.LpVariable.dicts("select", players, cat="Binary")

    # Two-tier position assignment: active (scoring 18) vs reserve (non-scoring 8)
    # active_assign[i][p] = 1 if player i fills an active (scoring) slot for position p
    # reserve_assign[i][p] = 1 if player i fills a reserve slot for position p
    active_assign = {}
    reserve_assign = {}
    for i in players:
        active_assign[i] = {}
        reserve_assign[i] = {}
        for p in df.at[i, "_eligible"]:
            active_assign[i][p] = pulp.LpVariable(f"act_{i}_{p}", cat="Binary")
            if RESERVE_QUOTAS.get(p, 0) > 0:
                reserve_assign[i][p] = pulp.LpVariable(f"res_{i}_{p}", cat="Binary")
        # FLEX: every player can fill the active flex slot (scoring)
        active_assign[i]["FLEX"] = pulp.LpVariable(f"act_{i}_FLEX", cat="Binary")

    # over_excess[t] = number of players from team t beyond the soft target (>= 0)
    teams = df["team"].dropna().unique().tolist()
    over_excess = pulp.LpVariable.dicts("over_excess", teams, lowBound=0, cat="Integer")

    # ── Objective function ────────────────────────────────────────────────────
    # Only active (scoring) players get full objective weight; reserves get RESERVE_WEIGHT
    prob += (
        pulp.lpSum(
            pulp.lpSum(active_assign[i][p] * float(objective_weight.iloc[i])
                       for p in active_assign[i])
            for i in players
        )
        + RESERVE_WEIGHT * pulp.lpSum(
            pulp.lpSum(reserve_assign[i][p] * float(objective_weight.iloc[i])
                       for p in reserve_assign[i])
            for i in players
        )
        - team_penalty * pulp.lpSum(over_excess[t] for t in teams)
    )

    # ── Constraints ───────────────────────────────────────────────────────────

    # 1. Squad size
    prob += pulp.lpSum(x[i] for i in players) == SQUAD_SIZE

    # 2. Salary cap
    cap = salary_cap if salary_cap is not None else SALARY_CAP
    prob += pulp.lpSum(
        x[i] * float(df.at[i, price_col]) for i in players
        if pd.notna(df.at[i, price_col])
    ) <= cap

    # 3. Each selected player assigned to exactly one slot (active or reserve)
    for i in players:
        all_slots = list(active_assign[i].values()) + list(reserve_assign[i].values())
        if all_slots:
            prob += pulp.lpSum(all_slots) == x[i]

    # 4. Active (scoring) quota constraints — 17 active position slots + 1 FLEX = 18
    for pos, quota in ACTIVE_QUOTAS.items():
        players_for_pos = [i for i in players if pos in active_assign[i]]
        prob += pulp.lpSum(
            active_assign[i][pos] for i in players_for_pos
        ) == quota

    # 5. Reserve quota constraints — 8 reserve slots
    for pos, quota in RESERVE_QUOTAS.items():
        if quota > 0:
            players_for_pos = [i for i in players if pos in reserve_assign[i]]
            prob += pulp.lpSum(
                reserve_assign[i][pos] for i in players_for_pos
            ) == quota

    # 6. Flex slot: exactly 1 active player assigned to FLEX
    prob += pulp.lpSum(active_assign[i]["FLEX"] for i in players) == 1

    # 7. Team diversity constraints
    for t in teams:
        team_players = [i for i in players if df.at[i, "team"] == t]
        if team_players:
            team_count = pulp.lpSum(x[i] for i in team_players)
            # Hard cap: never more than HARD_MAX_PER_TEAM from one team
            prob += team_count <= HARD_MAX_PER_TEAM
            # Soft cap: over_excess[t] tracks players above SOFT_MAX_PER_TEAM
            prob += over_excess[t] >= team_count - SOFT_MAX_PER_TEAM

    # 8. Injury coverage: for each starting position, ensure at least
    #    (quota + depth) selected players are ELIGIBLE for it, even if
    #    they are assigned to a different position slot.
    for pos in STARTING_SLOTS:
        quota = POSITION_QUOTAS[pos]
        eligible_for_pos = [i for i in players if pos in df.at[i, "_eligible"]]
        prob += (
            pulp.lpSum(x[i] for i in eligible_for_pos) >= quota + MIN_COVERAGE_DEPTH,
            f"coverage_depth_{pos}",
        )

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    status_str = pulp.LpStatus[prob.status]
    log.info("Solver status: %s", status_str)

    if prob.status != pulp.constants.LpStatusOptimal:
        log.error("Optimizer did not find an optimal solution (status=%s)", status_str)
        return {"solver_status": status_str, "starting_13": [], "bench_4": [], "reserves_8": []}

    # ── Extract selected players ──────────────────────────────────────────────
    selected_indices = [i for i in players if pulp.value(x[i]) > 0.5]

    # Determine assigned position and tier (active/reserve) for each selected player
    assigned_pos = {}
    assigned_tier = {}  # "active" or "reserve"
    for i in selected_indices:
        found = False
        for p, var in active_assign[i].items():
            if pulp.value(var) > 0.5:
                assigned_pos[i] = p
                assigned_tier[i] = "active"
                found = True
                break
        if not found:
            for p, var in reserve_assign[i].items():
                if pulp.value(var) > 0.5:
                    assigned_pos[i] = p
                    assigned_tier[i] = "reserve"
                    found = True
                    break
        if not found:
            # Fallback
            assigned_pos[i] = df.at[i, "_eligible"][0]
            assigned_tier[i] = "reserve"

    selected_df = df.loc[selected_indices].copy()
    selected_df["assigned_position"] = [assigned_pos[i] for i in selected_indices]
    selected_df["_tier"] = [assigned_tier[i] for i in selected_indices]
    selected_df = selected_df.sort_values("predicted_points", ascending=False)

    total_price = int(selected_df[price_col].fillna(0).sum())
    log.info("Squad selected: %d players, total price $%s",
             len(selected_df), f"{total_price:,}")

    # Teams count for info
    team_counts = selected_df["team"].value_counts()
    log.info("Teams represented: %s", dict(team_counts))

    # ── Position-aware squad split ─────────────────────────────────────────────
    starting_13, bench_4, flex_1, reserves_8 = _split_squad(selected_df)

    return {
        "starting_13":   starting_13,
        "bench_4":       bench_4,
        "flex_1":        flex_1,
        "reserves_8":    reserves_8,
        "total_price":   total_price,
        "solver_status": status_str,
        "df":            selected_df,
    }


def _split_squad(selected_df: pd.DataFrame) -> tuple[list, list, list, list]:
    """
    Split the 26-player selected squad into Starting 13, Bench 4, Flex 1, Reserves 8.

    Uses the LP's _tier assignment (active vs reserve) to separate scoring players
    from reserves. Within active players per position, highest predicted_points get
    starting spots; the rest are bench.
    """
    # Separate active (scoring) and reserve players using LP tier
    active_players = selected_df[selected_df["_tier"] == "active"].copy()
    reserve_players = selected_df[selected_df["_tier"] == "reserve"].copy()

    # Group active players by assigned_position
    active_by_pos = {}
    for _, row in active_players.iterrows():
        pos = row["assigned_position"]
        active_by_pos.setdefault(pos, []).append(row.to_dict())

    # Sort each active position group by predicted_points descending
    for pos in active_by_pos:
        active_by_pos[pos].sort(key=lambda r: r.get("predicted_points", 0), reverse=True)

    starting_13 = []
    bench_4 = []
    flex_1 = []
    reserves_8 = []

    # Extract flex player
    for player in active_by_pos.get("FLEX", []):
        flex_1.append({**player, "role": "Flex"})

    # Within active players per position: top N are Starting, rest are Bench
    for pos in POSITION_QUOTAS:
        group = active_by_pos.get(pos, [])
        start_slots = STARTING_SLOTS.get(pos, 0)
        for j, player in enumerate(group):
            if j < start_slots:
                starting_13.append({**player, "role": "Starting"})
            else:
                bench_4.append({**player, "role": "Bench"})

    # All reserve-tier players go to reserves
    for _, row in reserve_players.iterrows():
        reserves_8.append({**row.to_dict(), "role": "Reserve"})
    reserves_8.sort(key=lambda r: r.get("predicted_points", 0), reverse=True)

    return starting_13, bench_4, flex_1, reserves_8


# ── Display ───────────────────────────────────────────────────────────────────

def print_team(result: dict):
    """Pretty-print the selected squad to stdout."""
    if not result.get("starting_13"):
        print(f"\nNo team selected. Solver status: {result.get('solver_status')}")
        return

    def _row(p: dict) -> str:
        name  = p.get("player_name", "")[:22].ljust(22)
        pos   = p.get("assigned_position", "").ljust(5)
        team  = p.get("team", "").ljust(4)
        price = f"${int(p.get('price_usd', p.get('price', 0)) or 0):>9,}"
        pred  = f"{p.get('predicted_points', 0):>6.1f} pts"
        return f"  {name} {pos} {team} {price}  {pred}"

    print("\n" + "=" * 65)
    print("  NRL SUPERCOACH OPTIMAL SQUAD")
    print(f"  Total Price: ${result['total_price']:,}  |  Cap: ${SALARY_CAP:,}")
    print("=" * 65)

    print("\n  -- STARTING XIII " + "-" * 48)
    for p in result["starting_13"]:
        print(_row(p))

    print("\n  -- BENCH " + "-" * 56)
    for p in result["bench_4"]:
        print(_row(p))

    print("\n  -- FLEX " + "-" * 57)
    for p in result.get("flex_1", []):
        print(_row(p))

    print("\n  -- RESERVES " + "-" * 53)
    for p in result["reserves_8"]:
        print(_row(p))

    print("=" * 65 + "\n")


def export_team(result: dict, round_number: int,
                output_dir: Path = OUTPUTS_DIR) -> Path:
    """Export the selected squad to outputs/team_round_<N>.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"team_round_{round_number}.csv"

    rows = []
    for group, label in [("starting_13", "Starting"),
                          ("bench_4", "Bench"),
                          ("flex_1", "Flex"),
                          ("reserves_8", "Reserve")]:
        for p in result.get(group, []):
            rows.append({
                "role":               label,
                "player_name":        p.get("player_name"),
                "assigned_position":  p.get("assigned_position"),
                "positions":          p.get("positions"),
                "team":               p.get("team"),
                "price":              p.get("price_usd", p.get("price")),
                "avg_points":         p.get("avg_points"),
                "predicted_points":   p.get("predicted_points"),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    log.info("Team exported -> %s", out_path)
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Re-run the squad optimizer on saved predictions")
    parser.add_argument("--predictions", required=True,
                        help="CSV written by main.py (outputs/predictions_round_N.csv)")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--cap", type=int, default=SALARY_CAP)
    args = parser.parse_args()

    result = select_team(pd.read_csv(args.predictions), round_number=args.round,
                         salary_cap=args.cap)
    print_team(result)
    export_team(result, args.round)

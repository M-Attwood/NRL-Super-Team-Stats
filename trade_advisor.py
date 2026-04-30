"""
trade_advisor.py — Weekly trade advisor for NRL Supercoach.

Each round:
  1. Scrapes fresh 2026 stats (actual game performance)
  2. Builds the best possible 26-player squad from confirmed starters only
  3. Compares the ideal team to your current squad
  4. Recommends the best 2 trades to move your squad toward the ideal

Usage:
    python trade_advisor.py                   # scrape fresh data + advise
    python trade_advisor.py --no-scrape       # use existing data
"""

import argparse
import logging
import sys
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  USER INPUT — UPDATE THESE EACH ROUND
# ═══════════════════════════════════════════════════════════════════════════════

CURRENT_ROUND = 9

# ── YOUR CURRENT 26-PLAYER SQUAD ──
MY_SQUAD =[
    # Starting 13
    "Harry Grant",
    "Jackson Ford",
    "Arama Hau",
    "Siua Wong",
    "Nathan Cleary",
    "Cameron Munster",
    "Latrell Mitchell",
    "Herbie Farnworth",
    "Thomas Jenkins",
    "Tom Chester",
    "Dylan Edwards",
    "Sialetili Faeamani",
    "Jack Williams",
    # Bench / Reserves 13
    "Brandon Wakeham",
    "Dominic Young",
    "Francis Manuleleua",
    "Cooper Clarke",
    "Kalani Going",
    "John Fineanganofo",
    "Zane Harrison",
    "Moses Leo",
    "Noah Martin",
    "Hudson Young",
    "Reece Foley",
    "Setu Tu",
    "Motu Pasikala",
]

ROUND_STARTERS =[
    # Bulldogs
    "Connor Tracey",
    "Jonathan Sua",
    "Bronson Xerri",
    "Stephen Crichton",
    "Enari Tuala",
    "Matt Burton",
    "Lachlan Galvin",
    "Samuel Hughes",
    "Bailey Hayward",
    "Leo Thompson",
    "Sitili Tupouniua",
    "Jacob Preston",
    "Jaeman Salmon",
    # Cowboys
    "Scott Drinkwater",
    "Braidon Burns",
    "Jaxon Purdue",
    "Tom Chester",
    "Zac Laybutt",
    "Jake Clifford",
    "Tom Dearden",
    "Coen Hess",
    "Reed Mahoney",
    "Jason Taumalolo",
    "Heilum Luki",
    "Jeremiah Nanai",
    "Reuben Cotter",
    # Dolphins
    "Hamiso Tabuai-Fidow",
    "Jamayne Isaako",
    "Trai Fuller",
    "Herbie Farnworth",
    "Selwyn Cobbo",
    "Bradley Schneider",
    "Isaiya Katoa",
    "Tom Gilbert",
    "Max Plath",
    "Francis Molo",
    "Connelly Lemuelu",
    "Kulikefu Finefeuiaki",
    "Morgan Knowles",
    # Storm
    "Sualauvi Faalogo",
    "Will Warbrick",
    "Jack Howarth",
    "Nick Meaney",
    "Hugo Peel",
    "Cameron Munster",
    "Tyran Wishart",
    "Stefano Utoikamanu",
    "Harry Grant",
    "Josh King",
    "Shawn Blore",
    "Ativalu Lisati",
    "Alec MacDonald",
    # Titans
    "Keano Kini",
    "Sialetili Faeamani",
    "Jojo Fifita",
    "AJ Brimson",
    "Phillip Sami",
    "Lachlan Ilias",
    "Jayden Campbell",
    "Kurtis Morrin",
    "Sam Verrills",
    "Tino Fa'asuamaleaui",
    "Arama Hau",
    "Beau Fermor",
    "Chris Randall",
    # Raiders
    "Kaeo Weekes",
    "Savelio Tamale",
    "Sebastian Kris",
    "Matthew Timoko",
    "Jed Stuart",
    "Ethan Strange",
    "Ethan Sanders",
    "Corey Horsburgh",
    "Tom Starling",
    "Joseph Tapine",
    "Ata Mariota",
    "Simi Sasagi",
    "Jayden Brailey",
    # Eels
    "Joash Papalii",
    "Brian Kelly",
    "Will Penisini",
    "Sean Russell",
    "Josh Addo-Carr",
    "Ronald Volkman",
    "Mitchell Moses",
    "Luca Moretti",
    "Ryley Smith",
    "Junior Paulo",
    "Charlie Guymer",
    "Jack Williams",
    "Jack De Belin",
    # Warriors
    "Taine Tuaupiki",
    "Dallin Watene-Zelezniak",
    "Roger Tuivasa-Sheck",
    "Adam Pompey",
    "Alofiana Khan-Pereira",
    "Chanel Harris-Tavita",
    "Tanah Boyd",
    "James Fisher-Harris",
    "Wayde Egan",
    "Jackson Ford",
    "Leka Halasima",
    "Kurt Capewell",
    "Erin Clark",
    # Roosters
    "James Tedesco",
    "Daniel Tupou",
    "Hugo Savala",
    "Robert Toia",
    "Mark Nawaqanitawase",
    "Daly Cherry-Evans",
    "Sam Walker",
    "Naufahu Whyte",
    "Reece Robson",
    "Lindsay Collins",
    "Angus Crichton",
    "Siua Wong",
    "Victor Radley",
    # Broncos
    "Reece Walsh",
    "Josiah Karapani",
    "Kotoni Staggs",
    "Gehamat Shibasaki",
    "Deine Mariner",
    "Ezra Mam",
    "Adam Reynolds",
    "Ben Talty",
    "Cory Paix",
    "Jack Gosiewski",
    "Xavier Willison",
    "Jordan Riki",
    "Patrick Carrigan",
    # Knights
    "Kalyn Ponga",
    "Dominic Young",
    "Dane Gagai",
    "Bradman Best",
    "Greg Marzhew",
    "Fletcher Sharpe",
    "Dylan Brown",
    "Jacob Saifiti",
    "Phoenix Crossland",
    "Trey Mooney",
    "Dylan Lucas",
    "Jermaine McEwen",
    "Mat Croker",
    # Rabbitohs
    "Matthew Dufty",
    "Alex Johnston",
    "Latrell Mitchell",
    "Jack Wighton",
    "Campbell Graham",
    "Cody Walker",
    "Jamie Humphreys",
    "Tevita Tatola",
    "Bronson Garlick",
    "Sean Keppie",
    "Keaon Koloamatangi",
    "Tallis Duncan",
    "Cameron Murray",
    # Sharks
    "William Kennedy",
    "Mawene Hiroti",
    "Siosifa Talakai",
    "KL Iro",
    "Samuel Stonestreet",
    "Braydon Trindall",
    "Nicholas Hynes",
    "Addin Fonua-Blake",
    "Blayke Brailey",
    "Toby Rudolf",
    "Briton Nikora",
    "Teig Wilton",
    "Cameron McInnes",
    # Wests Tigers
    "Sunia Turuva",
    "Jeral Skelton",
    "Taylan May",
    "Starford To'a",
    "Luke Laulilii",
    "Jarome Luai",
    "Adam Doueihi",
    "Terrell May",
    "Tristan Hope",
    "Fonua Pole",
    "Samuela Fainu",
    "Tony Sukkar",
    "Alex Twal",
    # Panthers
    "Dylan Edwards",
    "Thomas Jenkins",
    "Paul Alamoti",
    "Casey McLean",
    "Brian To'o",
    "Blaize Talagi",
    "Nathan Cleary",
    "Moses Leota",
    "Freddy Lussick",
    "Lindsay Smith",
    "Isaiah Papali'i",
    "Luke Garner",
    "Isaah Yeo",
    # Sea Eagles
    "Tolutau Koula",
    "Jason Saab",
    "Clayton Faulalo",
    "Reuben Garrick",
    "Lehi Hopoate",
    "Luke Brooks",
    "Jamal Fogarty",
    "Taniela Paseka",
    "Brandon Wakeham",
    "Kobe Hetherington",
    "Haumole Olakau'atu",
    "Ben Trbojevic",
    "Jake Trbojevic",
    # Dragons on bye for Round 9
]


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

from datetime import datetime as _dt
HIST_YEARS = list(range(2022, _dt.now().year))


def load_player_pool(round_num: int, year: int = 2026,
                     no_scrape: bool = False) -> pd.DataFrame:
    """
    Run the full data pipeline: scrape -> clean -> engineer -> predict.
    Returns df_pred with predicted_points for all 2026 players.
    """
    from scraper import scrape_full, update_historical_data
    from model import clean_data, engineer_features, load_or_train_model, predict_next_round_scores

    from paths import (
        DATA_RAW, DATA_ROUNDS, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR,
    )
    for d in (DATA_RAW, DATA_ROUNDS, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Scrape or load 2026 data
    if no_scrape:
        from paths import DATA_RAW
        raw_files = sorted(DATA_RAW.glob("nrl_data_*.csv"), reverse=True)
        if not raw_files:
            log.error("No data files found. Run without --no-scrape first.")
            sys.exit(1)
        df_2026 = pd.read_csv(raw_files[0], low_memory=False)
        log.info("Loaded %d players from %s", len(df_2026), raw_files[0])
    else:
        log.info("Scraping fresh 2026 data ...")
        df_2026 = scrape_full(year=year, save=True)
        if df_2026.empty:
            log.error("Scraper returned no data.")
            sys.exit(1)
        log.info("Scraped %d players", len(df_2026))

    # Step 2: Load historical data
    from main import load_or_scrape_historical
    df_hist_raw = load_or_scrape_historical(df_2026, years=HIST_YEARS)

    # Step 3: Merge and engineer features
    if not df_hist_raw.empty:
        df_all = update_historical_data(
            pd.concat([df_2026, df_hist_raw], ignore_index=True)
        )
    else:
        df_all = update_historical_data(df_2026)

    df_clean = clean_data(df_all)
    df_feat, _ = engineer_features(df_clean, fit_scaler=True)

    # Step 4: Train/fine-tune model
    load_or_train_model(df_feat)

    # Step 5: Predict 2026 scores
    if "scrape_year" in df_feat.columns:
        df_2026_feat = df_feat[df_feat["scrape_year"] == 2026].copy()
    else:
        df_2026_feat = df_feat.copy()

    df_pred = predict_next_round_scores(df_2026_feat, df_historical=df_hist_raw)
    log.info("Predictions ready: %d players", len(df_pred))
    return df_pred


# ═══════════════════════════════════════════════════════════════════════════════
#  NAME MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_name(name: str) -> str:
    """Normalise a name for matching: lowercase, strip surrounding whitespace
    and apostrophe variants.

    Why: the scraper strips all apostrophes from player names (e.g. "Brian
    To'o" arrives as "Too, Brian"), so apostrophe variants in user input
    must collapse to the same form.

    Contract: this function MUST NOT collapse two distinct players to the
    same key. The variants below are pure punctuation; nothing in this set
    occurs in a way that would distinguish two players. If you ever extend
    this list, run a uniqueness check across the full pool first.
    """
    out = name.strip().lower()
    for ch in ("'", "’", "ʻ", "`"):
        out = out.replace(ch, "")
    return out


def _flip_name(name: str) -> str:
    """
    Flip between 'Firstname Lastname' and 'Lastname, Firstname' formats.
    Also handles 'F. Lastname' -> tries last name match.
    """
    name = name.strip()
    if "," in name:
        # "Lastname, Firstname" -> "Firstname Lastname"
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    else:
        # "Firstname Lastname" -> "Lastname, Firstname"
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            return f"{parts[1]}, {parts[0]}"
    return name


def _match_initial_lastname(name: str, pool_names: list[str]) -> str | None:
    """
    Match 'F. Lastname' format against pool names in 'Lastname, Firstname' format.
    E.g. 'H. Grant' matches 'Grant, Harry'.
    """
    name = name.strip()
    # Check if it matches "X. Lastname" pattern
    if len(name) >= 4 and name[1] == "." and name[2] == " ":
        initial = name[0].lower()
        lastname = name[3:].strip().lower()
        # Also handle multi-part lastnames like "Graham-Taufa"
        for pn in pool_names:
            pn_lower = pn.lower()
            if "," in pn_lower:
                pn_last, pn_first = pn_lower.split(",", 1)
                pn_last = pn_last.strip()
                pn_first = pn_first.strip()
                if pn_last == lastname and pn_first.startswith(initial):
                    return pn
        # Try partial last name match (e.g. surname hyphenated)
        for pn in pool_names:
            pn_lower = pn.lower()
            if "," in pn_lower:
                pn_last, pn_first = pn_lower.split(",", 1)
                pn_last = pn_last.strip()
                pn_first = pn_first.strip()
                if (lastname in pn_last or pn_last in lastname) and \
                   pn_first.startswith(initial):
                    return pn
    return None


def match_player(name: str, pool: pd.DataFrame,
                 threshold: float = 0.70) -> str | None:
    """
    Match a user-provided name against the player pool.
    Returns the matched player_name or None.

    Handles formats: 'Lastname, Firstname', 'Firstname Lastname', 'F. Lastname'.
    """
    pool_names = pool["player_name"].dropna().unique().tolist()

    # Exact match
    if name in pool_names:
        return name

    # Case-insensitive exact
    name_lower = _normalise_name(name)
    for pn in pool_names:
        if _normalise_name(pn) == name_lower:
            return pn

    # Try flipped format: "Firstname Lastname" <-> "Lastname, Firstname"
    flipped = _flip_name(name)
    if flipped in pool_names:
        return flipped
    flipped_lower = _normalise_name(flipped)
    for pn in pool_names:
        if _normalise_name(pn) == flipped_lower:
            return pn

    # Try "F. Lastname" -> "Lastname, Firstname" match
    initial_match = _match_initial_lastname(name, pool_names)
    if initial_match:
        return initial_match

    # Fuzzy match — require surname to match exactly before accepting
    # This prevents "Keenan Going" matching "Kalani Going" as a silent wrong match.
    def _surname(n: str) -> str:
        n = n.strip()
        if "," in n:
            return n.split(",")[0].strip().lower()
        parts = n.rsplit(" ", 1)
        return parts[-1].lower() if len(parts) > 1 else n.lower()

    input_surname = _surname(name)
    best_score = 0.0
    best_match = None
    for pn in pool_names:
        # Require surname match — avoids "Kalani Going" for "Keenan Going"
        if _surname(pn) != input_surname:
            continue
        pn_lower = _normalise_name(pn)
        s1 = SequenceMatcher(None, name_lower, pn_lower).ratio()
        s2 = SequenceMatcher(None, flipped_lower, pn_lower).ratio()
        score = max(s1, s2)
        if score > best_score:
            best_score = score
            best_match = pn

    if best_score >= threshold:
        return best_match

    return None


def resolve_names(names: list[str], pool: pd.DataFrame,
                  label: str = "players") -> list[str]:
    """
    Resolve a list of user-provided names against the pool.
    Warns clearly when a name cannot be matched. Returns matched pool names.
    """
    matched = []
    unmatched = []
    for name in names:
        m = match_player(name, pool)
        if m:
            matched.append(m)
        else:
            unmatched.append(name)

    if unmatched:
        print(f"\n  WARNING: {len(unmatched)} {label} could not be matched in the data pool:")
        for n in unmatched:
            print(f"    - {n}  (check spelling or update to the name used in the dataset)")
        print()

    return matched


# ═══════════════════════════════════════════════════════════════════════════════
#  SQUAD HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_MISSING_PRICE_WARNED: set[str] = set()


def _get_price(p: dict) -> float:
    """Extract player price (prefer price_usd over scaled price).

    Warns once per player when neither field is present — a missing price
    is data corruption (the salary cap math depends on it), not a normal
    case. Defaulting silently to 0 has caused phantom-cap-room bugs.
    """
    raw = p.get("price_usd")
    if raw in (None, "", 0):
        raw = p.get("price")
    if raw in (None, "", 0):
        name = p.get("player_name", "?")
        if name not in _MISSING_PRICE_WARNED:
            log.warning("No price for %s — treating as $0 (cap math may be wrong)", name)
            _MISSING_PRICE_WARNED.add(name)
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        log.warning("Unparseable price %r for %s — treating as $0",
                    raw, p.get("player_name", "?"))
        return 0.0


def _is_on_bye(player: dict, round_num: int) -> bool:
    """True if this player's team has a bye in round_num."""
    try:
        return int(player.get("bye_round", -1)) == round_num
    except (TypeError, ValueError):
        return False


def _player_dict(row) -> dict:
    """Convert a DataFrame row to a player dict."""
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    return d


def build_squad_dicts(squad_names: list[str],
                      pool: pd.DataFrame) -> list[dict]:
    """Build list of player dicts from matched names."""
    squad = []
    for name in squad_names:
        rows = pool[pool["player_name"] == name]
        if not rows.empty:
            squad.append(_player_dict(rows.iloc[0]))
    return squad


def flatten_team_result(result: dict, pool: pd.DataFrame) -> list[dict]:
    """Flatten select_team() result into a list of player dicts.

    Adds a "role" field tagging Starting/Bench/Flex/Reserve so downstream
    code (e.g. scoring-18 totals) can respect the LP's position
    assignments instead of guessing from predicted_points.
    """
    role_for_group = {
        "starting_13": "Starting",
        "bench_4": "Bench",
        "flex_1": "Flex",
        "reserves_8": "Reserve",
    }
    squad = []
    for group, role in role_for_group.items():
        for p in result.get(group, []):
            name = p["player_name"]
            rows = pool[pool["player_name"] == name]
            if not rows.empty:
                d = _player_dict(rows.iloc[0])
                d.update(p)  # overlay optimizer assignments
            else:
                d = dict(p)
            d["role"] = role
            squad.append(d)
    return squad


def _scoring_18_total(squad: list[dict],
                      confirmed_starters: set[str] | None = None) -> float:
    """Sum the predicted points of the 18 players who would actually score
    this round, respecting Supercoach position quotas (13 starting +
    4 bench + 1 flex). Optionally restricts the candidate pool to
    confirmed starters — players not in the set are skipped because they
    would score 0.

    The greedy: walk players highest-predicted first, drop them into the
    first eligible (position, role) slot that still has room. This is the
    same heuristic Supercoach itself uses for emergencies, so it's a
    realistic ceiling rather than an inflated top-18-by-points sum.
    """
    from optimizer import STARTING_SLOTS, BENCH_SLOTS, _eligible_positions

    quotas: dict[tuple[str, str], int] = {}
    for pos, n in STARTING_SLOTS.items():
        quotas[(pos, "Starting")] = n
    for pos, n in BENCH_SLOTS.items():
        quotas[(pos, "Bench")] = n
    flex_filled = False

    eligible = [
        p for p in squad
        if confirmed_starters is None
        or p.get("player_name") in confirmed_starters
    ]
    eligible.sort(key=lambda p: p.get("predicted_points", 0), reverse=True)

    total = 0.0
    used = set()
    for p in eligible:
        name = p.get("player_name")
        if name in used:
            continue
        positions = _eligible_positions(p.get("positions", ""))
        placed = False
        for pos in positions:
            for role in ("Starting", "Bench"):
                if quotas.get((pos, role), 0) > 0:
                    quotas[(pos, role)] -= 1
                    total += p.get("predicted_points", 0)
                    used.add(name)
                    placed = True
                    break
            if placed:
                break
        if not placed and not flex_filled:
            total += p.get("predicted_points", 0)
            used.add(name)
            flex_filled = True
    return total


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def filter_to_starters(df_pred: pd.DataFrame,
                       starters: set[str]) -> pd.DataFrame:
    """Keep only players in the confirmed starters set."""
    mask = df_pred["player_name"].isin(starters)
    df_starters = df_pred[mask].copy()
    log.info("Filtered to %d confirmed starters (from %d total)",
             len(df_starters), len(df_pred))
    return df_starters


def build_ideal_team(df_starters: pd.DataFrame, round_num: int,
                     salary_cap: int | None = None) -> dict:
    """Build the best possible team from confirmed starters only."""
    from optimizer import select_team
    log.info("Building ideal team from %d confirmed starters (cap $%s) ...",
             len(df_starters),
             f"{salary_cap:,}" if salary_cap else "default")
    result = select_team(df_starters, round_number=round_num,
                         salary_cap=salary_cap)
    if result.get("solver_status") != "Optimal":
        log.warning("Ideal team solver status: %s", result.get("solver_status"))
    return result


def compare_squads(ideal_squad: list[dict], current_squad: list[dict],
                   confirmed_starters: set[str]) -> dict:
    """
    Compare ideal team vs current squad.
    Returns dict with players_to_add, players_to_drop, overlap, non_starters.
    """
    ideal_names = {p["player_name"] for p in ideal_squad}
    current_names = {p["player_name"] for p in current_squad}

    overlap_names = ideal_names & current_names
    add_names = ideal_names - current_names
    drop_names = current_names - ideal_names

    # Build sorted lists
    players_to_add = sorted(
        [p for p in ideal_squad if p["player_name"] in add_names],
        key=lambda p: p.get("predicted_points", 0),
        reverse=True,
    )
    players_to_drop = sorted(
        [p for p in current_squad if p["player_name"] in drop_names],
        key=lambda p: p.get("predicted_points", 0),
    )

    # Non-starters in current squad
    non_starters = [
        p for p in current_squad
        if p["player_name"] not in confirmed_starters
    ]
    non_starters.sort(key=lambda p: p.get("predicted_points", 0))

    # Scoring totals (Supercoach scores the 18 active players: 13 starting
    # + 4 bench + 1 flex). The ideal team already has roles assigned by the
    # LP, so we trust those. The current squad is a flat list, so we pick
    # the highest-predicted player into each quota slot greedily — this
    # gives a position-respecting ceiling rather than the inflated
    # "top 18 regardless of role" sum that the previous code computed.
    ideal_pts = sum(
        p.get("predicted_points", 0) for p in ideal_squad
        if p.get("role", "").lower() in ("starting", "bench", "flex")
    )
    current_pts = _scoring_18_total(current_squad, confirmed_starters)

    return {
        "overlap": [p for p in current_squad if p["player_name"] in overlap_names],
        "players_to_add": players_to_add,
        "players_to_drop": players_to_drop,
        "non_starters": non_starters,
        "ideal_scoring_total": ideal_pts,
        "current_scoring_total": current_pts,
        "point_gap": ideal_pts - current_pts,
    }


def recommend_trades(current_squad: list[dict],
                     comparison: dict,
                     confirmed_starters: set[str],
                     pool: pd.DataFrame,
                     max_trades: int = 2,
                     round_num: int = 0,
                     salary_cap: int | None = None) -> list[dict]:
    """
    Find the best trades to move current squad toward ideal.

    Priority for OUT:
      1. Genuine non-starters (not playing AND not on bye) — score 0 this round
      2. Bye players and poor performers — ranked by predicted_points ascending
         (bye players use their real predicted_points, not 0, so they only
          surface if they are genuinely bad long-term options)

    Priority for IN:
      1. Highest value from ideal team not in current
      2. Must be a confirmed starter

    Constraints: salary cap, position quotas.
    """
    from optimizer import SALARY_CAP, _eligible_positions
    from planner import validate_position_quotas

    cap = salary_cap if salary_cap is not None else SALARY_CAP

    non_starters = comparison["non_starters"]
    to_drop = comparison["players_to_drop"]
    to_add = comparison["players_to_add"]

    # Build OUT candidates: all non-starters then lowest-value drops
    out_candidates = []
    seen_out = set()

    for p in non_starters:
        name = p["player_name"]
        if name not in seen_out:
            out_candidates.append(p)
            seen_out.add(name)

    for p in to_drop:
        name = p["player_name"]
        if name not in seen_out:
            out_candidates.append(p)
            seen_out.add(name)

    # Build IN candidates: ideal team additions first, then all starters
    in_candidates = []
    seen_in = set()
    current_names = {p["player_name"] for p in current_squad}

    for p in to_add:
        name = p["player_name"]
        if name not in seen_in and name not in current_names:
            in_candidates.append(p)
            seen_in.add(name)

    # Also consider high-value starters not in ideal (broader search)
    starter_rows = pool[
        (pool["player_name"].isin(confirmed_starters))
        & (~pool["player_name"].isin(current_names))
        & (~pool["player_name"].isin(seen_in))
    ].sort_values("predicted_points", ascending=False)
    for _, row in starter_rows.iterrows():
        in_candidates.append(_player_dict(row))

    # Evaluate all possible trades
    current_salary = sum(_get_price(p) for p in current_squad)
    trade_options = []

    for p_out in out_candidates:
        out_price = _get_price(p_out)
        out_pts = p_out.get("predicted_points", 0)
        out_name = p_out["player_name"]
        on_bye = _is_on_bye(p_out, round_num)

        if out_name not in confirmed_starters and not on_bye:
            # Genuinely not playing this round (injured/dropped/not named)
            out_effective = 0.0
            out_reason = "not in lineup"
        elif on_bye:
            # On bye — will return, so value them at their predicted points.
            # They only become a trade target if the incoming player is
            # genuinely better long-term.
            out_effective = out_pts
            out_reason = "on bye"
        else:
            out_effective = out_pts
            out_reason = "not in ideal team"

        out_positions = set(_eligible_positions(p_out.get("positions", "")))

        for p_in in in_candidates:
            in_price = _get_price(p_in)
            in_pts = p_in.get("predicted_points", 0)
            in_positions = set(_eligible_positions(p_in.get("positions", "")))

            # Salary cap check
            new_salary = current_salary - out_price + in_price
            if new_salary > cap:
                continue

            # Position compatibility: must share at least one position
            if not out_positions & in_positions:
                continue

            immediate_gain = in_pts - out_effective
            trade_options.append({
                "out": p_out,
                "in": p_in,
                "out_reason": out_reason,
                "immediate_gain": immediate_gain,
                "salary_delta": in_price - out_price,
                "new_salary": new_salary,
                "in_ideal": p_in["player_name"] in {
                    p["player_name"] for p in comparison["players_to_add"]
                },
            })

    # Sort: prioritize immediate gain (non-starters OUT = big gain)
    trade_options.sort(key=lambda t: t["immediate_gain"], reverse=True)

    # Greedily select top trades, validating quotas
    selected = []
    used_out = set()
    used_in = set()
    working_squad = [p.copy() for p in current_squad]

    for trade in trade_options:
        if len(selected) >= max_trades:
            break

        out_name = trade["out"]["player_name"]
        in_name = trade["in"]["player_name"]

        if out_name in used_out or in_name in used_in:
            continue

        # Build new squad and validate
        new_squad = [p for p in working_squad if p["player_name"] != out_name]
        new_squad.append(trade["in"])

        if validate_position_quotas(new_squad):
            selected.append(trade)
            used_out.add(out_name)
            used_in.add(in_name)
            working_squad = new_squad
            # Update salary for next iteration
            current_salary = trade["new_salary"]

    # Defensive: each individual trade was validated, but if multiple trades
    # interact (one creates an imbalance another fixes), the intermediate
    # state could be invalid. Re-validate the final composite squad.
    if selected and not validate_position_quotas(working_squad):
        log.error("Composite squad after trades fails position quotas — "
                  "discarding all trade recommendations to be safe.")
        return []

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def _print_team_group(label: str, players: list[dict]):
    """Print a group of players (e.g. Starting XIII) with position/team/price/pred."""
    print(f"\n    {label}:")
    for i, p in enumerate(players, 1):
        name = p.get("player_name", "?")
        pos = p.get("assigned_position", p.get("positions", "?").split("|")[0])
        team = p.get("team", "?")
        price = _get_price(p) / 1000
        pred = p.get("predicted_points", 0)
        print(f"      {i:2d}. {name:<28s} | {pos:<4s} | {team:<4s} | "
              f"${price:>6.0f}K | Pred: {pred:.1f}")


def print_report(current_squad: list[dict], ideal_squad: list[dict],
                 comparison: dict, trades: list[dict],
                 confirmed_starters: set[str], round_num: int,
                 ideal_result: dict = None,
                 salary_cap: int | None = None):
    """Print the full trade advisor report."""

    from optimizer import SALARY_CAP as DEFAULT_CAP
    cap = salary_cap if salary_cap is not None else DEFAULT_CAP

    current_salary = sum(_get_price(p) for p in current_squad)
    ideal_salary = sum(_get_price(p) for p in ideal_squad)
    n_starters = sum(1 for p in current_squad
                     if p["player_name"] in confirmed_starters)

    print("\n" + "=" * 70)
    print(f"  TRADE ADVISOR -- Round {round_num}")
    print("=" * 70)

    if cap > DEFAULT_CAP:
        print(f"\n  Effective cap: ${cap:,} "
              f"(default ${DEFAULT_CAP:,} + ${cap - DEFAULT_CAP:,} "
              f"price-rise allowance)")

    # Your squad summary
    print(f"\n  YOUR SQUAD ({len(current_squad)} players):")
    print(f"    Salary: ${current_salary:,.0f} / ${cap:,}")
    print(f"    Confirmed starters: {n_starters}/{len(current_squad)}")
    print(f"    Non-starters (will score 0): "
          f"{len(current_squad) - n_starters}/{len(current_squad)}")
    print(f"    Est. scoring total: {comparison['current_scoring_total']:.0f} pts")

    # Ideal team summary
    print(f"\n  IDEAL TEAM (from confirmed starters only):")
    print(f"    Salary: ${ideal_salary:,.0f} / ${cap:,}")
    print(f"    Est. scoring total: {comparison['ideal_scoring_total']:.0f} pts")

    # Full ideal team breakdown (proves all constraints are followed)
    if ideal_result:
        print(f"    Solver status: {ideal_result.get('solver_status', '?')}")
        if ideal_result.get("starting_13"):
            _print_team_group("Starting XIII", ideal_result["starting_13"])
        if ideal_result.get("bench_4"):
            _print_team_group("Bench (4)", ideal_result["bench_4"])
        if ideal_result.get("flex_1"):
            _print_team_group("Flex (1)", ideal_result["flex_1"])
        if ideal_result.get("reserves_8"):
            _print_team_group("Reserves (8)", ideal_result["reserves_8"])

        # Position quota summary
        from optimizer import POSITION_QUOTAS
        pos_counts = {}
        for group in ["starting_13", "bench_4", "flex_1", "reserves_8"]:
            for p in ideal_result.get(group, []):
                apos = p.get("assigned_position", "?")
                pos_counts[apos] = pos_counts.get(apos, 0) + 1
        quota_parts = []
        for pos in ["HOK", "FRF", "2RF", "HFB", "5/8", "CTW", "FLB"]:
            count = pos_counts.get(pos, 0)
            quota = POSITION_QUOTAS.get(pos, "?")
            quota_parts.append(f"{pos}:{count}/{quota}")
        print(f"\n    Position quotas: {' | '.join(quota_parts)}")

        # Team diversity
        team_counts = {}
        for group in ["starting_13", "bench_4", "flex_1", "reserves_8"]:
            for p in ideal_result.get(group, []):
                t = p.get("team", "?")
                team_counts[t] = team_counts.get(t, 0) + 1
        max_team = max(team_counts.values()) if team_counts else 0
        print(f"    Max players from one team: {max_team}/5")

    # Gap
    gap = comparison["point_gap"]
    print(f"\n  GAP: {gap:+.0f} pts (your team vs ideal)")

    # Squad comparison
    n_overlap = len(comparison["overlap"])
    n_add = len(comparison["players_to_add"])
    n_drop = len(comparison["players_to_drop"])
    print(f"\n  SQUAD COMPARISON:")
    print(f"    Players in both teams: {n_overlap}/{len(current_squad)}")
    print(f"    In ideal but not yours: {n_add}")
    print(f"    In yours but not ideal: {n_drop}")

    # Non-starters
    non_starters = comparison["non_starters"]
    if non_starters:
        print(f"\n  NON-STARTERS IN YOUR SQUAD:")
        for i, p in enumerate(non_starters, 1):
            name = p["player_name"]
            pos = p.get("positions", "?").split("|")[0]
            price = _get_price(p) / 1000
            pred = p.get("predicted_points", 0)
            team = p.get("team", "?")
            print(f"    {i:2d}. {name:<28s} | {pos:<4s} | {team:<4s} | "
                  f"${price:>6.0f}K | Pred: {pred:.1f}")

    # Show current squad starters too
    starters_in_squad = [
        p for p in current_squad
        if p["player_name"] in confirmed_starters
    ]
    starters_in_squad.sort(key=lambda p: p.get("predicted_points", 0), reverse=True)
    if starters_in_squad:
        print(f"\n  STARTERS IN YOUR SQUAD:")
        for i, p in enumerate(starters_in_squad, 1):
            name = p["player_name"]
            pos = p.get("positions", "?").split("|")[0]
            price = _get_price(p) / 1000
            pred = p.get("predicted_points", 0)
            team = p.get("team", "?")
            in_ideal = name in {ip["player_name"] for ip in ideal_squad}
            tag = " *" if in_ideal else ""
            print(f"    {i:2d}. {name:<28s} | {pos:<4s} | {team:<4s} | "
                  f"${price:>6.0f}K | Pred: {pred:.1f}{tag}")
        print("    (* = also in ideal team)")

    # Recommended trades
    if trades:
        print(f"\n  RECOMMENDED TRADES ({len(trades)} available):")
        for i, t in enumerate(trades, 1):
            p_out = t["out"]
            p_in = t["in"]
            out_name = p_out["player_name"]
            in_name = p_in["player_name"]
            out_pos = p_out.get("positions", "?").split("|")[0]
            in_pos = p_in.get("positions", "?").split("|")[0]
            out_team = p_out.get("team", "?")
            in_team = p_in.get("team", "?")
            out_price = _get_price(p_out) / 1000
            in_price = _get_price(p_in) / 1000
            try:
                out_avg = float(p_out.get("avg_points", 0) or 0)
            except (ValueError, TypeError):
                out_avg = 0.0
            try:
                in_avg = float(p_in.get("avg_points", 0) or 0)
            except (ValueError, TypeError):
                in_avg = 0.0
            out_pred = float(p_out.get("predicted_points", 0) or 0)
            in_pred = float(p_in.get("predicted_points", 0) or 0)

            cap_delta = t["salary_delta"]
            cap_remaining = cap - t["new_salary"]
            if cap_delta < 0:
                cap_str = (f"frees ${abs(cap_delta)/1000:.0f}K  "
                           f"(${cap_remaining/1_000_000:.2f}M remaining)")
            else:
                cap_str = (f"costs ${cap_delta/1000:.0f}K extra  "
                           f"(${cap_remaining/1_000_000:.2f}M remaining)")

            out_reason = t.get("out_reason", "")
            ideal_tag = "  [IN IDEAL TEAM]" if t["in_ideal"] else ""

            print(f"\n    Trade {i}:  [{out_reason}]")
            print(f"      OUT: {out_name:<28s} ({out_pos}, {out_team})"
                  f"  ${out_price:.0f}K  |  avg {out_avg:.1f}  pred {out_pred:.1f}")
            print(f"      IN:  {in_name:<28s} ({in_pos}, {in_team})"
                  f"  ${in_price:.0f}K  |  avg {in_avg:.1f}  pred {in_pred:.1f}"
                  f"{ideal_tag}")
            print(f"      Point gain: +{t['immediate_gain']:.1f}  |  Cap: {cap_str}")

        # Post-trade summary
        final_salary = current_salary
        for t in trades:
            final_salary += t["salary_delta"]
        cap_left = cap - final_salary

        new_overlap = n_overlap + sum(1 for t in trades if t["in_ideal"])
        print(f"\n  AFTER TRADES:")
        print(f"    Salary used: ${final_salary:,.0f} / ${cap:,}"
              f"  (${cap_left:,.0f} remaining)")
        print(f"    Overlap with ideal: {new_overlap}/{len(current_squad)}"
              f" (was {n_overlap})")
    else:
        print("\n  No beneficial trades found.")

    print("\n" + "=" * 70 + "\n")


def export_report(current_squad: list[dict], ideal_squad: list[dict],
                  comparison: dict, trades: list[dict],
                  confirmed_starters: set[str], round_num: int):
    """Export trade advice to CSV."""
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export comparison
    rows = []
    ideal_names = {p["player_name"] for p in ideal_squad}
    for p in current_squad:
        name = p["player_name"]
        rows.append({
            "player_name": name,
            "team": p.get("team", ""),
            "positions": p.get("positions", ""),
            "price": _get_price(p),
            "predicted_points": p.get("predicted_points", 0),
            "is_starter": name in confirmed_starters,
            "in_ideal_team": name in ideal_names,
            "status": "keep" if name in ideal_names else "consider_dropping",
        })

    df_comp = pd.DataFrame(rows)
    comp_path = output_dir / f"trade_advice_r{round_num}.csv"
    df_comp.to_csv(comp_path, index=False)
    log.info("Trade advice exported -> %s", comp_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRADE MAP CHART
# ═══════════════════════════════════════════════════════════════════════════════

def _sname(name: str) -> str:
    """Last name only: 'Grant, Harry' → 'Grant'."""
    if "," in name:
        return name.split(",")[0].strip()
    return name.rsplit(" ", 1)[-1]


def generate_trade_chart(df_pred: pd.DataFrame,
                         current_names: set,
                         ideal_names: set,
                         trades: list,
                         confirmed_starters: set,
                         round_num: int) -> None:
    """
    Save outputs/trade_map_r{N}.png — a two-panel figure:
      Top: Price vs Predicted scatter
           Green  = your squad (Ashes)
           Red    = ideal team (not in your squad)
           Gold   = in both (keepers)
           Grey   = everyone else
           Arrows = recommended trades (OUT → IN)
      Bottom: Your squad bar chart coloured by round status
              Green=starter | Orange=on bye | Red=not named
    """
    if df_pred is None or df_pred.empty:
        log.warning("No prediction data — skipping trade chart.")
        return
    if "price_usd" not in df_pred.columns:
        log.warning("No price_usd column — skipping trade chart.")
        return

    df = (df_pred
          .dropna(subset=["price_usd", "predicted_points"])
          .query("price_usd > 0")
          .reset_index(drop=True))

    prices = df["price_usd"] / 1000
    preds  = df["predicted_points"]
    pnames = df["player_name"]

    in_current  = pnames.isin(current_names)
    in_ideal    = pnames.isin(ideal_names)
    in_both     = in_current & in_ideal
    cur_only    = in_current & ~in_ideal
    ideal_only  = ~in_current & in_ideal
    neither     = ~in_current & ~in_ideal

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(18, 22),
        gridspec_kw={"height_ratios": [1.6, 1]},
    )
    fig.suptitle(
        f"NRL Supercoach — Round {round_num}  Trade Map",
        fontsize=15, fontweight="bold", y=0.995,
    )
    plt.subplots_adjust(hspace=0.28, top=0.97, bottom=0.04,
                        left=0.14, right=0.97)

    # ── Top panel: scatter ───────────────────────────────────────────────────
    ax = ax_top

    # Background players
    ax.scatter(prices[neither], preds[neither],
               color="#d5d8dc", alpha=0.25, s=14, zorder=1)

    # Gold: in both (keep)
    ax.scatter(prices[in_both], preds[in_both],
               color="#f39c12", alpha=0.95, s=100,
               edgecolors="#7d6608", linewidths=0.8, zorder=3,
               label="Keep (in both)")
    for _, row in df[in_both].iterrows():
        ax.annotate(
            _sname(row["player_name"]),
            (row["price_usd"] / 1000, row["predicted_points"]),
            fontsize=7, color="#7d6608", fontweight="bold",
            xytext=(5, 4), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"),
        )

    # Green: current squad not in ideal
    ax.scatter(prices[cur_only], preds[cur_only],
               color="#27ae60", alpha=0.95, s=100,
               edgecolors="#1a5c34", linewidths=0.8, zorder=3,
               label="Your squad")
    for _, row in df[cur_only].iterrows():
        nm = row["player_name"]
        on_bye = _is_on_bye(row.to_dict(), round_num)
        not_starting = nm not in confirmed_starters
        tag = " [bye]" if (not_starting and on_bye) else (" [out]" if not_starting else "")
        ax.annotate(
            _sname(nm) + tag,
            (row["price_usd"] / 1000, row["predicted_points"]),
            fontsize=7, color="#1a5c34", fontweight="bold",
            xytext=(5, 4), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"),
        )

    # Red: ideal team not in current squad
    ax.scatter(prices[ideal_only], preds[ideal_only],
               color="#e74c3c", alpha=0.95, s=100,
               edgecolors="#922b21", linewidths=0.8, zorder=3,
               label="Ideal team (not yours)")
    for _, row in df[ideal_only].iterrows():
        ax.annotate(
            _sname(row["player_name"]),
            (row["price_usd"] / 1000, row["predicted_points"]),
            fontsize=7, color="#922b21", fontweight="bold",
            xytext=(5, 4), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"),
        )

    # Trade arrows
    arrow_palette = ["#8e44ad", "#2980b9", "#d35400"]
    for idx, trade in enumerate(trades or []):
        out_nm = trade["out"]["player_name"]
        in_nm  = trade["in"]["player_name"]
        out_row = df[df["player_name"] == out_nm]
        in_row  = df[df["player_name"] == in_nm]
        if out_row.empty or in_row.empty:
            continue
        ox = out_row.iloc[0]["price_usd"] / 1000
        oy = out_row.iloc[0]["predicted_points"]
        ix = in_row.iloc[0]["price_usd"] / 1000
        iy = in_row.iloc[0]["predicted_points"]
        col = arrow_palette[idx % len(arrow_palette)]
        ax.annotate(
            "", xy=(ix, iy), xytext=(ox, oy),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                            connectionstyle="arc3,rad=0.25"),
            zorder=7,
        )
        ax.annotate(
            f"Trade {idx + 1}",
            ((ox + ix) / 2, (oy + iy) / 2),
            fontsize=7.5, color=col, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      alpha=0.85, ec=col, lw=0.9),
            zorder=8,
        )

    ax.set_xlabel("Price ($K)", fontsize=11)
    ax.set_ylabel("Predicted Points", fontsize=11)
    ax.set_title(
        f"Price vs Predicted Value  —  "
        f"Green = your squad  |  Red = ideal only  |  "
        f"Gold = keep  |  Arrows = suggested trades",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")

    # ── Bottom panel: squad status bars ─────────────────────────────────────
    ax2 = ax_bot

    STATUS_COLOR = {"Starter": "#27ae60", "Bye": "#f39c12", "Out": "#e74c3c"}
    STATUS_ORDER = {"Starter": 0, "Bye": 1, "Out": 2}

    squad_rows = []
    for nm in current_names:
        rows_df = df_pred[df_pred["player_name"] == nm]
        if rows_df.empty:
            continue
        r = rows_df.iloc[0].to_dict()
        is_starter = nm in confirmed_starters
        on_bye = _is_on_bye(r, round_num)
        if is_starter:
            status = "Starter"
        elif on_bye:
            status = "Bye"
        else:
            status = "Out"
        squad_rows.append({
            "name":   nm,
            "label":  _sname(nm),
            "pos":    str(r.get("positions", "?")).split("|")[0],
            "team":   r.get("team", "?"),
            "price_k": float(r.get("price_usd", r.get("price", 0)) or 0) / 1000,
            "avg":    float(r.get("avg_points", 0) or 0),
            "pred":   float(r.get("predicted_points", 0) or 0),
            "status": status,
            "in_ideal": nm in ideal_names,
        })

    squad_rows.sort(key=lambda r: (STATUS_ORDER[r["status"]], -r["pred"]))

    y_pos  = list(range(len(squad_rows)))
    colors = [STATUS_COLOR[r["status"]] for r in squad_rows]
    pvals  = [r["pred"] for r in squad_rows]

    ax2.barh(y_pos, pvals, color=colors, alpha=0.82,
             edgecolor="white", linewidth=0.4)

    y_labels = []
    for r in squad_rows:
        star = "★" if r["in_ideal"] else " "
        y_labels.append(
            f"{star} {r['label']:<14}  {r['pos']:<5}  {r['team']:<4}"
            f"  ${r['price_k']:.0f}K  avg {r['avg']:.0f}  [{r['status']}]"
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(y_labels, fontsize=7, fontfamily="monospace")

    # Dividers between status groups
    prev = None
    for i, r in enumerate(squad_rows):
        if prev and r["status"] != prev:
            ax2.axhline(y=i - 0.5, color="black", lw=0.9,
                        linestyle="--", alpha=0.45)
        prev = r["status"]

    ax2.set_xlabel("Predicted Points", fontsize=11)
    ax2.set_title(
        f"Your Squad — Round {round_num} Status"
        f"  (★ = also in ideal team  |  "
        f"Green=Starter  Orange=Bye  Red=Out)",
        fontsize=10,
    )
    from matplotlib.patches import Patch
    ax2.legend(
        handles=[Patch(fc=c, alpha=0.82, label=s)
                 for s, c in STATUS_COLOR.items()],
        fontsize=8, loc="lower right",
    )

    out_path = Path("outputs") / f"trade_map_r{round_num}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Trade map saved -> %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _load_round_inputs(round_num: int) -> tuple[list[str], list[str]]:
    """Load (squad, starters) for a round from data/inputs/round_{n}.yaml.

    Falls back to the hardcoded MY_SQUAD / ROUND_STARTERS lists at the top
    of this module if the YAML file isn't present (so the existing
    workflow keeps working). Once all weekly data lives in YAML you can
    delete the hardcoded lists.

    Always logs which path was used so the data source is traceable from
    the run log alone — no detective work to figure out "where did this
    squad come from?" if a wrong lineup ever ships.
    """
    import yaml
    from paths import DATA_INPUTS

    yaml_path = DATA_INPUTS / f"round_{round_num}.yaml"
    if yaml_path.exists():
        log.info("[inputs source] YAML  →  %s", yaml_path)
        doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        squad = doc.get("squad", [])
        starters = doc.get("starters", [])
        if not squad or not starters:
            log.error("%s is missing 'squad' or 'starters' keys", yaml_path)
            sys.exit(1)
        return squad, starters

    if MY_SQUAD and ROUND_STARTERS:
        log.warning(
            "[inputs source] FALLBACK  →  hardcoded MY_SQUAD/ROUND_STARTERS in "
            "trade_advisor.py (no %s found). Migrate this round to YAML when "
            "you next update.",
            yaml_path,
        )
        return list(MY_SQUAD), list(ROUND_STARTERS)

    log.error(
        "[inputs source] MISSING — no %s, and the hardcoded MY_SQUAD/ROUND_STARTERS "
        "lists are empty. Cannot run the advisor without round inputs.",
        yaml_path,
    )
    sys.exit(1)


def run_advisor(round_num: int = None, no_scrape: bool = False,
                max_trades: int = 2):
    """Run the full trade advisor pipeline."""
    if round_num is None:
        round_num = CURRENT_ROUND

    squad_input, starters_input = _load_round_inputs(round_num)
    if not squad_input:
        log.error("No squad found for round %d (check data/inputs/round_%d.yaml)",
                  round_num, round_num)
        sys.exit(1)
    if not starters_input:
        log.error("No confirmed starters for round %d", round_num)
        sys.exit(1)

    # Step 1: Load player pool with predictions
    log.info("=" * 55)
    log.info("  TRADE ADVISOR -- Round %d", round_num)
    log.info("=" * 55)

    df_pred = load_player_pool(round_num, no_scrape=no_scrape)

    # Step 2: Resolve names against pool
    log.info("Matching squad names ...")
    squad_names = resolve_names(squad_input, df_pred, label="squad players")
    starter_names = resolve_names(starters_input, df_pred,
                                  label="confirmed starters")
    confirmed_starters = set(starter_names)

    if len(squad_names) < 20:
        log.error("Only matched %d/26 squad players. Check names.", len(squad_names))
        sys.exit(1)

    log.info("Matched %d/%d squad, %d starters",
             len(squad_names), len(squad_input), len(confirmed_starters))

    # Step 3: Build player dicts for current squad
    current_squad = build_squad_dicts(squad_names, df_pred)

    # Step 3b: Compute effective salary cap.
    # Supercoach prices move on a 3-game rolling average, so mid-season a
    # squad's current value can drift above the starting cap. The cap that
    # actually constrains trades is max(starting_cap, current_team_value):
    # you can't spend money you don't have, but price rises you've earned
    # shouldn't retroactively push you offside.
    from optimizer import SALARY_CAP as DEFAULT_CAP
    current_team_value = sum(_get_price(p) for p in current_squad)
    effective_cap = max(DEFAULT_CAP, int(current_team_value))
    if effective_cap > DEFAULT_CAP:
        log.info("Team value $%s exceeds default cap $%s -- using "
                 "effective cap $%s (price-rise allowance)",
                 f"{int(current_team_value):,}", f"{DEFAULT_CAP:,}",
                 f"{effective_cap:,}")
    else:
        log.info("Team value $%s within default cap $%s",
                 f"{int(current_team_value):,}", f"{DEFAULT_CAP:,}")

    # Step 4: Filter pool to confirmed starters and build ideal team
    df_starters = filter_to_starters(df_pred, confirmed_starters)
    ideal_result = build_ideal_team(df_starters, round_num,
                                    salary_cap=effective_cap)
    ideal_squad = flatten_team_result(ideal_result, df_pred)

    # Step 5: Compare squads
    comparison = compare_squads(ideal_squad, current_squad, confirmed_starters)

    # Step 6: Find best trades
    log.info("Evaluating trades ...")
    trades = recommend_trades(
        current_squad=current_squad,
        comparison=comparison,
        confirmed_starters=confirmed_starters,
        pool=df_pred,
        max_trades=max_trades,
        round_num=round_num,
        salary_cap=effective_cap,
    )

    # Step 7: Output
    print_report(current_squad, ideal_squad, comparison, trades,
                 confirmed_starters, round_num, ideal_result=ideal_result,
                 salary_cap=effective_cap)
    export_report(current_squad, ideal_squad, comparison, trades,
                  confirmed_starters, round_num)

    # Step 8: Trade map chart
    generate_trade_chart(
        df_pred=df_pred,
        current_names=set(squad_names),
        ideal_names={p["player_name"] for p in ideal_squad},
        trades=trades,
        confirmed_starters=confirmed_starters,
        round_num=round_num,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NRL Supercoach Trade Advisor"
    )
    parser.add_argument("--no-scrape", action="store_true",
                        help="Use existing data instead of scraping fresh")
    parser.add_argument("--round", type=int, default=None,
                        help="Round number (overrides CURRENT_ROUND)")
    parser.add_argument("--max-trades", type=int, default=2,
                        help="Max trades to recommend (default: 2)")
    args = parser.parse_args()

    run_advisor(
        round_num=args.round,
        no_scrape=args.no_scrape,
        max_trades=args.max_trades,
    )

"""
scraper.py — NRL Supercoach stats scraper.

Data source: the jqGrid JSON endpoint behind nrlsupercoachstats.com/stats.php.
One request page returns up to 200 rows; each row is a single player-round
snapshot (that round's score, minutes and stat line, plus season-to-date
aggregates), and every player also has one "Totals" row for the season.

Usage:
    python scraper.py --year 2026        # scrape one season to data/raw/
"""

import logging
import random
import time
from datetime import date

import pandas as pd
import requests

from paths import DATA_RAW

log = logging.getLogger(__name__)

BASE_URL = "https://www.nrlsupercoachstats.com"
STATS_URL = f"{BASE_URL}/stats.php"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edg/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
]

# API field -> clean column name. For per-round rows `total_points` and
# `total_minutes` are that round's score and minutes; on the "Totals" row
# they are season totals.
API_COLUMN_MAP = {
    "id":               "player_id",
    "Name2":            "player_name",      # plain-text "LastName, FirstName"
    "Posn1":            "_posn1",           # combined into 'positions' below
    "Posn2":            "_posn2",
    "Team":             "team",
    "Price":            "price",
    "Jersey":           "jersey_number",
    "ByeRd":            "bye_round",
    "Score":            "total_points",
    "Mins":             "total_minutes",
    "AvgScore":         "avg_points",
    "BE":               "break_even",
    "CVTotal":          "coeff_variation",
    "CVRd":             "cv_round",
    "AvgMins":          "avg_minutes",
    "AvgMinsCalc":      "avg_minutes_calc",
    "PPM":              "points_per_minute",
    "BPPM":             "base_ppm",
    "BasePowerPPM":     "base_power_ppm",
    "BasePower":        "base_power_pts",
    "BasePowerAvg":     "avg_base_power",
    "SeasonPriceChange": "season_price_change",
    "RoundPriceChange": "round_price_change",
    "StartPrice":       "start_price",
    "EndPrice":         "end_price",
    "PlayedCalc":       "games_played",
    "Rd":               "current_round",
    "vs":               "vs_team",
    "Venue":            "venue",
    "weather":          "weather",
    "Surface":          "surface",
    "TwoRdAvg":         "avg_last2",
    "ThreeRdAvg":       "avg_last3",
    "FiveRdAvg":        "avg_last5",
    "ThreeRdMins":      "avg_mins_last3",
    "FiveRdMins":       "avg_mins_last5",
    "SeasonAvg":        "season_avg",
    "Avg1to10":         "avg_r1_9",
    "Avg11to18":        "avg_r10_18",
    "Avg19to26":        "avg_r19_27",
    "SixtySixty":       "pct_60plus",
    "AvgPC":            "avg_penalties",
    "AvgER":            "avg_errors",
    "AvgPCER":          "avg_pc_er",
    "H8percent":        "pct_h8",
    "TBPERCENT":        "pct_tackle_break",
    "MTPERCENT":        "pct_missed_tackle",
    "OLILPERCENT":      "pct_offload",
    "BasePercent":      "pct_base",
    "Base":             "pts_base",
    "Attack":           "pts_attack",
    "Playmaking":       "pts_create",
    "Power":            "pts_evade",
    "Negative":         "pts_negative",
    "BaseAvg":          "avg_base",
    "ScoringAvg":       "avg_scoring",
    "CreateAvg":        "avg_create",
    "EvadeAvg":         "avg_evade",
    "NegativeAvg":      "avg_negative",
    "StdDevTotal":      "std_dev_total",
    "StdDevRd":         "std_dev_round",
    "MagicNumber":      "magic_number",
    "TR":               "stat_tries",
    "TS":               "stat_try_assists",
    "LT":               "stat_last_touch",
    "GO":               "stat_goals",
    "MG":               "stat_missed_goals",
    "FG":               "stat_field_goals",
    "MF":               "stat_missed_fg",
    "TA":               "stat_tackles",
    "MT":               "stat_missed_tackles",
    "TB":               "stat_tackle_breaks",
    "FD":               "stat_forced_dropouts",
    "OL":               "stat_offloads",
    "IO":               "stat_ineff_offloads",
    "LB":               "stat_line_breaks",
    "LA":               "stat_lb_assists",
    "FT":               "stat_forty_twenty",
    "KB":               "stat_kick_regather",
    "H8":               "stat_hitups_h8",
    "HU":               "stat_hitups_hu",
    "HG":               "stat_held_goal",
    "IT":               "stat_intercept",
    "KD":               "stat_kicked_dead",
    "PC":               "stat_penalties",
    "ER":               "stat_errors",
    "SS":               "stat_sinbin",
}

# HTML markup / duplicate fields returned by the API that we don't keep.
DROP_COLS = {"Name", "Team2", "Photo", "Namedot", "Posn", "Year",
             "Time", "Played", "_posn1", "_posn2"}


def _session(year: int) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE_URL}/stats.php?year={year}",
    })
    return s


def _get_retry(session: requests.Session, url: str, params: dict,
               max_retries: int = 3) -> requests.Response:
    """GET with a rotating user-agent, polite jitter and exponential backoff."""
    delay = 1
    for attempt in range(max_retries):
        try:
            session.headers["User-Agent"] = random.choice(USER_AGENTS)
            r = session.get(url, params=params, timeout=20)
            r.raise_for_status()
            time.sleep(random.uniform(0.8, 1.5))
            return r
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, exc)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"All retries failed for {url}")


def scrape_season(year: int, rows_per_page: int = 200) -> pd.DataFrame:
    """Scrape every player-round row for one season via the JSON API."""
    log.info("Scraping %d season from the stats API ...", year)
    session = _session(year)
    all_rows, page = [], 1

    while True:
        params = {
            "year": year, "grid_id": "list1", "jqgrid_page": page,
            "rows": rows_per_page, "sidx": "Name", "sord": "asc",
            "_search": "false",
        }
        data = _get_retry(session, STATS_URL, params).json()
        rows = data.get("rows", [])
        if not rows:
            break
        all_rows.extend(rows)
        total_pages = int(data.get("total", 1))
        log.info("Page %d/%d: %d rows (%s records)", page, total_pages,
                 len(rows), data.get("records", "?"))
        if page >= total_pages:
            break
        page += 1

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={k: v for k, v in API_COLUMN_MAP.items() if k in df.columns})

    # Positions are stored pipe-separated ("2RF|FRF"). Never split on "/" —
    # "5/8" (five-eighth) is a position name, not a separator.
    posn1 = df.get("_posn1", pd.Series("", index=df.index)).fillna("").str.strip()
    posn2 = df.get("_posn2", pd.Series("", index=df.index)).fillna("").str.strip()
    df["positions"] = posn1.where(posn2 == "", posn1 + "|" + posn2).str.strip("|")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("Int64")
    df = df[df["player_name"].notna() & df["player_name"].str.strip().ne("")]
    df["scrape_date"] = str(date.today())
    df["scrape_year"] = year

    log.info("Season %d: %d rows, %d players", year, len(df), df["player_name"].nunique())
    return df.reset_index(drop=True)


def season_path(year: int):
    return DATA_RAW / f"nrl_data_{year}.csv"


def load_season(year: int, scrape_if_missing: bool = True,
                force_scrape: bool = False) -> pd.DataFrame:
    """Return one season's rows, from the CSV cache or a fresh scrape."""
    path = season_path(year)
    if path.exists() and not force_scrape:
        return pd.read_csv(path, low_memory=False)
    if not (scrape_if_missing or force_scrape):
        raise FileNotFoundError(f"{path} not found — run without --no-scrape")
    df = scrape_season(year)
    if df.empty:
        raise RuntimeError(f"Scraper returned no rows for {year}")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %d rows -> %s", len(df), path)
    return df


# ── Fixtures ─────────────────────────────────────────────────────────────────
# data/raw/fixtures_<year>.csv: round, home_team, away_team[, date]
# Team codes match planner.BYE_ROUNDS (BRO, BUL, CBR, ...).

def load_fixtures(year: int) -> pd.DataFrame:
    """Load the season draw, or an empty frame if no fixture file exists."""
    path = DATA_RAW / f"fixtures_{year}.csv"
    if not path.exists():
        log.warning("No fixture file at %s — next-opponent features fall back "
                    "to league average.", path)
        return pd.DataFrame(columns=["round", "home_team", "away_team"])
    df = pd.read_csv(path)
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    for col in ("home_team", "away_team"):
        df[col] = df[col].astype(str).str.upper().str.strip()
    return df.dropna(subset=["round"])


def get_next_opponent(team: str, current_round: int,
                      fixtures: pd.DataFrame) -> tuple[str | None, bool]:
    """Return (opponent_code, is_home) for `team` in `current_round`.

    (None, False) when the team has a bye or the round isn't in the fixture.
    """
    if fixtures is None or fixtures.empty:
        return None, False
    team = str(team).upper().strip()
    rnd = fixtures[fixtures["round"] == current_round]
    home = rnd[rnd["home_team"] == team]
    if not home.empty:
        return home.iloc[0]["away_team"], True
    away = rnd[rnd["away_team"] == team]
    if not away.empty:
        return away.iloc[0]["home_team"], False
    return None, False


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="NRL Supercoach stats scraper")
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    df = load_season(args.year, force_scrape=True)
    totals = df[df["current_round"].astype(str) == "Totals"]
    print(f"\n{args.year}: {len(df)} rows, {len(totals)} players")

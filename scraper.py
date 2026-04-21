"""
scraper.py — NRL Supercoach Stats scraper
Data source: jqGrid JSON API at https://www.nrlsupercoachstats.com/stats.php
"""

import re
import time
import random
import logging
from datetime import date
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

# API field → clean column name
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
    "SeasonPriceChange":"season_price_change",
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

# Columns to drop (HTML markup / duplicates we don't need)
DROP_COLS = {"Name", "Team2", "Photo", "Namedot", "Posn", "Year",
             "Time", "Played", "_posn1", "_posn2"}

DATA_RAW       = Path("data/raw")
DATA_ROUNDS    = Path("data/rounds")
DATA_PROCESSED = Path("data/processed")


def _ensure_dirs():
    for d in [DATA_RAW, DATA_ROUNDS, DATA_PROCESSED]:
        d.mkdir(parents=True, exist_ok=True)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE_URL}/stats.php?year=2026",
    })
    return s


def _get_retry(session: requests.Session, url: str, params: dict,
               max_retries: int = 3) -> requests.Response:
    delay = 1
    for attempt in range(max_retries):
        try:
            session.headers["User-Agent"] = random.choice(USER_AGENTS)
            r = session.get(url, params=params, timeout=20)
            r.raise_for_status()
            time.sleep(random.uniform(0.8, 1.5))
            return r
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, exc)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"All retries failed for {url}")


# ── Main Stats Table (JSON API) ───────────────────────────────────────────────

def scrape_main_stats(year: int = 2026, rows_per_page: int = 200) -> pd.DataFrame:
    """
    Scrape the full player stats table via the jqGrid JSON API.
    Returns a clean DataFrame with all available columns.
    """
    log.info("Scraping main stats JSON API for year %d ...", year)
    session = _session()
    all_rows = []
    page = 1

    while True:
        params = {
            "year":         year,
            "grid_id":      "list1",
            "jqgrid_page":  page,
            "rows":         rows_per_page,
            "sidx":         "Name",
            "sord":         "asc",
            "_search":      "false",
        }
        r = _get_retry(session, STATS_URL, params)

        try:
            data = r.json()
        except Exception:
            log.error("Response is not JSON on page %d. First 500 chars:\n%s",
                      page, r.text[:500])
            break

        rows = data.get("rows", [])
        total_pages = int(data.get("total", 1))
        total_records = data.get("records", "?")

        if not rows:
            break

        all_rows.extend(rows)
        log.info("Page %d/%d: %d rows (total records: %s)",
                 page, total_pages, len(rows), total_records)

        if page >= total_pages:
            break
        page += 1

    if not all_rows:
        log.error("No rows returned from API")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # ── Rename API fields to clean names ──────────────────────────────────────
    df = df.rename(columns={k: v for k, v in API_COLUMN_MAP.items() if k in df.columns})

    # ── Build positions column from Posn1 / Posn2 ────────────────────────────
    posn1 = df.get("_posn1", pd.Series("", index=df.index)).fillna("").str.strip()
    posn2 = df.get("_posn2", pd.Series("", index=df.index)).fillna("").str.strip()
    df["positions"] = posn1.where(posn2 == "", posn1 + "|" + posn2)
    df["positions"] = df["positions"].str.strip("|")

    # ── Drop unwanted columns ─────────────────────────────────────────────────
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # ── Convert price to int ──────────────────────────────────────────────────
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("Int64")

    # ── Drop rows with no player name ─────────────────────────────────────────
    if "player_name" in df.columns:
        df = df[df["player_name"].str.strip().ne("") & df["player_name"].notna()]

    df["scrape_date"] = str(date.today())
    df["scrape_year"] = year

    log.info("Main stats: %d players, %d columns", len(df), len(df.columns))
    return df.reset_index(drop=True)


# ── Per-round data (same API, filtering by round) ────────────────────────────

def scrape_round_data(year: int = 2026, round_num: int = None) -> pd.DataFrame:
    """
    Scrape per-round scores from the API.
    If round_num is None, scrapes every round available and pivots to wide format.
    Returns a DataFrame: player_name × round_N columns.
    """
    session = _session()
    session.headers["Referer"] = f"{BASE_URL}/stats.php?year={year}"

    # Determine available rounds
    rounds_to_fetch = [round_num] if round_num else list(range(1, 28))
    records = {}  # player_name → {round_N: score}

    for rd in rounds_to_fetch:
        page = 1
        while True:
            params = {
                "year":         year,
                "grid_id":      "list1",
                "jqgrid_page":  page,
                "rows":         200,
                "sidx":         "Name",
                "sord":         "asc",
                "_search":      "true",
                "searchField":  "Rd",
                "searchString": str(rd),
                "searchOper":   "eq",
            }
            try:
                r = _get_retry(session, STATS_URL, params)
                data = r.json()
            except Exception:
                break

            rows = data.get("rows", [])
            total_pages = int(data.get("total", 1))

            for row in rows:
                name = row.get("Name2", "").strip()
                score = row.get("Score", None)
                mins  = row.get("Mins", None)
                if name:
                    if name not in records:
                        records[name] = {}
                    records[name][f"round_{rd}"] = score
                    records[name][f"mins_{rd}"]  = mins

            if page >= total_pages or not rows:
                break
            page += 1

        log.info("Round %d: %d player records", rd, len(records))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "player_name"
    df = df.reset_index()
    df["year"] = year
    return df


# ── Individual Player Pages ───────────────────────────────────────────────────

def _player_page_url(player_name_lastfirst: str, year: int) -> str:
    """
    Build player page URL from 'LastName, FirstName' format (as returned by API Name2).
    e.g. "Addo-Carr, Josh" → .../index.php?player=Addo-Carr%2C+Josh&year=2026
    """
    encoded = player_name_lastfirst.replace(", ", "%2C+").replace(" ", "+")
    return f"{BASE_URL}/index.php?player={encoded}&year={year}"


def scrape_player_page(player_name_lastfirst: str, year: int = 2026,
                       session: requests.Session = None) -> dict:
    """
    Scrape an individual player profile page for round-by-round history,
    score distributions, and MACD data.
    """
    if session is None:
        session = _session()

    url = _player_page_url(player_name_lastfirst, year)
    try:
        r = _get_retry(session, url, params={})
        html = r.text
    except RuntimeError:
        log.warning("Failed to scrape player page: %s %d", player_name_lastfirst, year)
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"player_name": player_name_lastfirst, "year": year}

    # ── Extract round scores from chart data in <script> tags ────────────────
    scripts = [s.string or "" for s in soup.find_all("script")]
    for src in scripts:
        # Look for round score arrays: typically like  data: [45, 62, 38, ...]
        # or pointsByRound = [...]
        arrays = re.findall(
            r'(?:pointsByRound|scoreData|data)\s*[=:]\s*\[([^\]]+)\]', src
        )
        for arr_str in arrays:
            items = [x.strip() for x in arr_str.split(",")]
            if len(items) >= 3 and all(
                re.match(r'^-?\d+(\.\d+)?$|^null$', x) for x in items[:5]
            ):
                for i, val in enumerate(items, start=1):
                    key = f"round_{i}"
                    if key not in data and val.lower() != "null":
                        try:
                            data[key] = float(val)
                        except ValueError:
                            pass

        # Look for minutes arrays
        min_arrays = re.findall(
            r'(?:minutesByRound|minutesData|minsData)\s*[=:]\s*\[([^\]]+)\]', src
        )
        for arr_str in min_arrays:
            items = [x.strip() for x in arr_str.split(",")]
            for i, val in enumerate(items, start=1):
                key = f"mins_{i}"
                if key not in data and val.lower() != "null":
                    try:
                        data[key] = float(val)
                    except ValueError:
                        pass

    # ── Score distribution from tables ───────────────────────────────────────
    for tbl in soup.find_all("table"):
        text = tbl.get_text(" ", strip=True)
        if any(kw in text for kw in ["<20", "20-39", "40-59", "60+"]):
            cells = [td.get_text(strip=True) for td in tbl.find_all("td")]
            for j, cell in enumerate(cells):
                if "<20" in cell and j + 1 < len(cells):
                    data["count_u20"] = _safe_num(cells[j + 1])
                elif "20-39" in cell and j + 1 < len(cells):
                    data["count_20_39"] = _safe_num(cells[j + 1])
                elif "40-59" in cell and j + 1 < len(cells):
                    data["count_40_59"] = _safe_num(cells[j + 1])
                elif "60+" in cell and j + 1 < len(cells):
                    data["count_60plus"] = _safe_num(cells[j + 1])

    return data


def _safe_num(val: str):
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def scrape_all_player_pages(df_main: pd.DataFrame, year: int = 2026,
                            historical: bool = False) -> pd.DataFrame:
    """
    Scrape individual pages for all players.
    historical=True also scrapes seasons 2009 to (year-1).
    """
    players = df_main["player_name"].dropna().unique().tolist()
    years_to_scrape = list(range(2009, year + 1)) if historical else [year]

    session = _session()
    session.headers["X-Requested-With"] = ""  # player pages are regular HTML
    session.headers["Accept"] = "text/html,application/xhtml+xml,*/*"

    all_records = []
    total = len(players) * len(years_to_scrape)
    count = 0

    for yr in years_to_scrape:
        for player in players:
            count += 1
            if count % 20 == 0:
                log.info("[%d/%d] Scraping player pages ...", count, total)
            record = scrape_player_page(player, yr, session)
            if record:
                all_records.append(record)

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


# ── Supplementary Pages ───────────────────────────────────────────────────────

def _scrape_html_table(url: str, params: dict = None,
                       session: requests.Session = None) -> pd.DataFrame:
    """Generic helper: fetch a page, parse its first meaningful table."""
    if session is None:
        session = _session()
    try:
        r = _get_retry(session, url, params or {})
    except RuntimeError:
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "lxml")
    # Find the largest table (most rows) as the data table
    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame()

    best = max(tables, key=lambda t: len(t.find_all("tr")))
    rows = best.find_all("tr")
    if len(rows) < 2:
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    records = []
    for tr in rows[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            records.append(dict(zip(headers, cells)))
    return pd.DataFrame(records)


def scrape_minutes_grid(year: int = 2026, session=None) -> pd.DataFrame:
    log.info("Scraping minutes grid ...")
    df = _scrape_html_table(f"{BASE_URL}/minutes.php", {"year": year}, session)
    log.info("Minutes grid: %d rows", len(df))
    return df


def scrape_dual_positions(year: int = 2026, session=None) -> pd.DataFrame:
    log.info("Scraping dual position grid ...")
    df = _scrape_html_table(f"{BASE_URL}/dualposngrid.php", {"year": year}, session)
    log.info("Dual positions: %d rows", len(df))
    return df


def scrape_prices_and_bes(session=None) -> pd.DataFrame:
    log.info("Scraping prices & BEs ...")
    df = _scrape_html_table(f"{BASE_URL}/TeamPricesAndBEs.php", {}, session)
    log.info("Prices & BEs: %d rows", len(df))
    return df


def scrape_draft_rankings(year: int = 2026, session=None) -> pd.DataFrame:
    log.info("Scraping draft rankings ...")
    df = _scrape_html_table(f"{BASE_URL}/draft.php", {"year": year}, session)
    log.info("Draft rankings: %d rows", len(df))
    return df


def scrape_position_vs_team(year: int = 2026, session=None) -> pd.DataFrame:
    log.info("Scraping position vs team ...")
    df = _scrape_html_table(f"{BASE_URL}/posnvsteam.php", {"year": year}, session)
    log.info("Position vs team: %d rows", len(df))
    return df


# ── Historical Master ─────────────────────────────────────────────────────────

def update_historical_data(new_df: pd.DataFrame,
                           processed_path: str = "data/processed/master_historical.csv"
                           ) -> pd.DataFrame:
    path = Path(processed_path)
    if path.exists():
        existing = pd.read_csv(path, low_memory=False)
        log.info("Loaded existing historical data: %d rows", len(existing))
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    # Dedup by (player_name, scrape_year) — one row per player per season
    if "player_name" in combined.columns and "scrape_year" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["player_name", "scrape_year"], keep="last"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    log.info("Historical data saved: %d rows → %s", len(combined), path)
    return combined


# ── Historical Season Scraper ─────────────────────────────────────────────────

def scrape_historical_seasons(active_players: list,
                              years: list = None,
                              save: bool = True) -> pd.DataFrame:
    """
    Scrape full-season stats for prior years (default: 2024 and 2025).
    Only retains rows for players currently active in the 2026 roster.

    This gives the model real performance data to train on instead of
    relying on the crude price-based fallback.

    Args:
        active_players: list of player_name strings from the 2026 scrape.
        years: seasons to scrape (default [2024, 2025]).
        save: whether to write CSVs to data/raw/.

    Returns:
        DataFrame with all historical rows (filtered to 2026-active players).
    """
    if years is None:
        years = [2022, 2023, 2024, 2025]

    active_set = set(active_players)
    all_frames = []

    for yr in sorted(years):
        raw_path = DATA_RAW / f"nrl_data_{yr}.csv"
        if raw_path.exists():
            log.info("Loading cached historical data for %d from %s", yr, raw_path)
            df_yr = pd.read_csv(raw_path, low_memory=False)
        else:
            log.info("Scraping historical season %d ...", yr)
            df_yr = scrape_main_stats(year=yr)
            if df_yr.empty:
                log.warning("No data returned for year %d — skipping", yr)
                continue
            if save:
                df_yr.to_csv(raw_path, index=False)
                log.info("Saved %d rows → %s", len(df_yr), raw_path)

        # API returns per-round snapshots (~15-25 per player). Keep only the
        # season-total row per player. The season-total row has the same
        # games_played as the last per-round row, so use total_points
        # (cumulative > any single round) as tiebreaker.
        if "player_name" in df_yr.columns and "games_played" in df_yr.columns:
            df_yr["games_played"] = pd.to_numeric(df_yr["games_played"], errors="coerce").fillna(0)
            if "total_points" in df_yr.columns:
                df_yr["total_points"] = pd.to_numeric(df_yr["total_points"], errors="coerce").fillna(0)
            before_dedup = len(df_yr)
            sort_cols = ["games_played"] + (["total_points"] if "total_points" in df_yr.columns else [])
            df_yr = (df_yr.sort_values(sort_cols, ascending=False)
                          .drop_duplicates(subset=["player_name"], keep="first"))
            log.info("Year %d: %d → %d rows after per-round dedup", yr, before_dedup, len(df_yr))

        # Filter to 2026-active players only
        before = len(df_yr)
        if "player_name" in df_yr.columns:
            df_yr = df_yr[df_yr["player_name"].isin(active_set)].copy()
        log.info("Year %d: %d → %d (2026-active players)", yr, before, len(df_yr))

        all_frames.append(df_yr)

    if not all_frames:
        return pd.DataFrame()

    df_hist = pd.concat(all_frames, ignore_index=True)
    log.info("Historical seasons total: %d rows across years %s", len(df_hist), years)
    return df_hist


# ── Full Orchestrated Scrape ──────────────────────────────────────────────────

def scrape_full(year: int = 2026, historical: bool = False, save: bool = True) -> pd.DataFrame:
    """
    Run all scrapers and return the main stats DataFrame.
    historical=True triggers a full history scrape of every player page back to 2009.
    """
    _ensure_dirs()
    session = _session()

    # 1. Main stats table (JSON API)
    df_main = scrape_main_stats(year=year)
    if df_main.empty:
        log.error("Main stats scrape returned no data — aborting")
        return df_main

    # 2. Individual player pages for round-by-round data
    log.info("Scraping individual player pages ...")
    df_player_pages = scrape_all_player_pages(df_main, year=year, historical=historical)
    if not df_player_pages.empty:
        rounds_path = DATA_ROUNDS / f"round_scores_{year}.csv"
        if rounds_path.exists():
            existing_rounds = pd.read_csv(rounds_path, low_memory=False)
            df_player_pages = pd.concat([existing_rounds, df_player_pages], ignore_index=True)
            df_player_pages = df_player_pages.drop_duplicates(
                subset=["player_name", "year"], keep="last"
            )
        df_player_pages.to_csv(rounds_path, index=False)
        log.info("Round scores saved → %s", rounds_path)

    # 3. Supplementary pages
    df_minutes = scrape_minutes_grid(year=year, session=session)
    df_dual    = scrape_dual_positions(year=year, session=session)
    df_prices  = scrape_prices_and_bes(session=session)
    df_draft   = scrape_draft_rankings(year=year, session=session)
    df_pvt     = scrape_position_vs_team(year=year, session=session)

    # Merge draft rank into main
    if not df_draft.empty:
        name_col = next((c for c in df_draft.columns if "name" in c.lower()), None)
        rank_col = next((c for c in df_draft.columns if "rank" in c.lower()), None)
        if name_col and rank_col:
            df_main = df_main.merge(
                df_draft[[name_col, rank_col]].rename(
                    columns={name_col: "player_name", rank_col: "draft_rank"}
                ),
                on="player_name", how="left"
            )

    # Save supplementary
    if not df_dual.empty:
        df_dual.to_csv(DATA_RAW / f"dual_positions_{year}.csv", index=False)
    if not df_minutes.empty:
        df_minutes.to_csv(DATA_ROUNDS / f"minutes_grid_{year}.csv", index=False)
    if not df_prices.empty:
        df_prices.to_csv(DATA_RAW / f"prices_bes_{year}.csv", index=False)
    if not df_pvt.empty:
        df_pvt.to_csv(DATA_RAW / f"position_vs_team_{year}.csv", index=False)

    # 4. Save main snapshot
    if save:
        raw_path = DATA_RAW / f"nrl_data_{date.today()}.csv"
        df_main.to_csv(raw_path, index=False)
        log.info("Main snapshot saved → %s", raw_path)

    return df_main


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NRL Supercoach Stats Scraper")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    df = scrape_full(year=args.year, historical=args.historical, save=not args.no_save)
    if not df.empty:
        print(f"\nCollected {len(df)} players, {len(df.columns)} columns")
        show = ["player_name", "positions", "team", "price", "avg_points", "avg_last3"]
        show = [c for c in show if c in df.columns]
        print(df[show].head(15).to_string(index=False))

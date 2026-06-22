import io
import re
from datetime import datetime

import pandas as pd
import requests

# Official FIFA World Cup 2026 schedule & results (community-maintained mirror of
# official fixtures/results, public domain, updated ~daily). No API key required.
# Source: https://github.com/openfootball/worldcup.json
WORLDCUP_2026_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"

# Broad historical baseline (international results since 2002+) used to give Elo/form
# real context for teams before the tournament started, and for general training depth.
HISTORICAL_DATA_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)


def _slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _match_id(date_str, team1, team2):
    # Deterministic ID based on content, not list position — stable across runs
    # even if the upstream file's row order ever changes.
    return f"WC26_{date_str}_{_slugify(team1)}_{_slugify(team2)}"


def fetch_worldcup_2026_schedule():
    """
    Downloads the full 104-match World Cup 2026 schedule and parses it into a
    single tidy DataFrame with a `played` boolean column. Finished matches carry
    real scores; future matches have NaN scores.
    """
    print("[Scraper] Fetching official World Cup 2026 schedule (openfootball)...")
    response = requests.get(WORLDCUP_2026_URL, timeout=15)
    response.raise_for_status()
    matches = response.json()["matches"]

    rows = []
    for m in matches:
        played = "score" in m and "ft" in m.get("score", {})
        home_score, away_score = (m["score"]["ft"] if played else (None, None))
        rows.append(
            {
                "match_id": _match_id(m["date"], m["team1"], m["team2"]),
                "date": m["date"],
                "round": m.get("round"),
                "group": m.get("group"),
                "ground": m.get("ground"),
                "home_team": m["team1"],
                "away_team": m["team2"],
                "home_score": home_score,
                "away_score": away_score,
                "played": played,
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    print(f"[Scraper] Schedule loaded: {len(df)} total matches, {df['played'].sum()} played so far.")
    return df


def get_upcoming_matches():
    """
    Returns the next unplayed slate of World Cup 2026 fixtures (all matches sharing
    the nearest upcoming date — there are often several matches per matchday).
    Falls back to a tiny placeholder fixture (dated tomorrow) only if the schedule
    itself can't be reached at all, so the pipeline never hard-crashes offline.
    """
    try:
        schedule_df = fetch_worldcup_2026_schedule()
        upcoming = schedule_df[~schedule_df["played"]].sort_values("date")
        if upcoming.empty:
            print("[Scraper] No upcoming matches remain — tournament appears complete.")
            return pd.DataFrame()

        next_date = upcoming["date"].min()
        next_slate = upcoming[upcoming["date"] == next_date].drop(
            columns=["home_score", "away_score", "played"]
        ).reset_index(drop=True)
        print(f"[Scraper] Next matchday: {next_date.date()} — {len(next_slate)} fixture(s).")
        return next_slate
    except Exception as e:
        print(f"[Scraper] Could not fetch official schedule ({e}). Using offline placeholder fixture.")
        tomorrow = (datetime.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return pd.DataFrame(
            [{"match_id": "OFFLINE_PLACEHOLDER", "date": pd.to_datetime(tomorrow), "home_team": "Brazil", "away_team": "Argentina"}]
        )


def get_past_match_results(start_date="2002-01-01", end_date=None):
    """
    Combines two sources of past match results:
    1. A broad open historical international-results dataset — gives every team a
       real Elo/form baseline rather than starting everyone at a flat 1500.
    2. Finished World Cup 2026 matches from the official schedule — the matches
       this project actually cares about predicting and evaluating.

    Official World Cup 2026 rows take priority over historical-dataset rows on
    the same (date, home_team, away_team), in case both sources cover an overlap.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print("[Scraper] Gathering integrated past match results...")
    frames = []

    # --- 1. Broad historical baseline ---
    try:
        print("[Scraper] Downloading historical results from open dataset...")
        response = requests.get(HISTORICAL_DATA_URL, timeout=15)
        response.raise_for_status()
        hist_df = pd.read_csv(io.StringIO(response.text))
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df = hist_df[
            (hist_df["date"] >= start_date) & (hist_df["date"] <= end_date)
        ].dropna(subset=["home_team", "away_team", "home_score", "away_score"]).copy()
        hist_df["match_id"] = [
            _match_id(d.strftime("%Y-%m-%d"), h, a)
            for d, h, a in zip(hist_df["date"], hist_df["home_team"], hist_df["away_team"])
        ]
        hist_df["home_score"] = hist_df["home_score"].astype(int)
        hist_df["away_score"] = hist_df["away_score"].astype(int)
        frames.append(hist_df[["match_id", "date", "home_team", "away_team", "home_score", "away_score"]])
        print(f"[Scraper] Added {len(hist_df)} historical rows for training context.")
    except Exception as e:
        print(f"[Scraper] Historical data source failed: {e}.")

    # --- 2. Official, finished World Cup 2026 matches ---
    try:
        schedule_df = fetch_worldcup_2026_schedule()
        finished_df = schedule_df[
            schedule_df["played"] & (schedule_df["date"] >= start_date) & (schedule_df["date"] <= end_date)
        ][["match_id", "date", "home_team", "away_team", "home_score", "away_score"]].copy()
        finished_df["home_score"] = finished_df["home_score"].astype(int)
        finished_df["away_score"] = finished_df["away_score"].astype(int)
        frames.append(finished_df)
        print(f"[Scraper] Added {len(finished_df)} finished official World Cup 2026 matches.")
    except Exception as e:
        print(f"[Scraper] Official schedule fetch failed: {e}.")

    if not frames:
        print("[Scraper] WARNING: both data sources failed — returning an empty dataset.")
        return pd.DataFrame(columns=["match_id", "date", "home_team", "away_team", "home_score", "away_score"])

    df = pd.concat(frames, ignore_index=True)
    # Official World Cup 2026 rows (added second) win any (date, home, away) overlap
    # with the historical baseline, since they're the authoritative source for this project.
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    df = df.sort_values(by="date").reset_index(drop=True)
    print(f"[Scraper] Total dataset compiled: {len(df)} matches.")
    return df


if __name__ == "__main__":
    print(get_upcoming_matches())
    print(get_past_match_results().tail())

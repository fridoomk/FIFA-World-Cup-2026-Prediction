"""
Predict a single, specific match on demand — useful for friendlies, qualifiers,
or any fixture that isn't in the live competition's official upcoming-matches
list (which predict_matchday.py relies on).

Usage:
    python3 predict_single_match.py --home "Uzbekistan" --away "Colombia"
    python3 predict_single_match.py --home "Uzbekistan" --away "Colombia" --date 2026-06-20
"""

import argparse
from datetime import datetime

import pandas as pd

from src.scraper import get_past_match_results
from src.features import generate_features
from src.model import load_model
from src.visualize import plot_next_matchday_predictions

MODELS_PATH = "models"


def predict_single_match(home_team, away_team, match_date=None):
    if match_date is None:
        match_date = datetime.now().strftime("%Y-%m-%d")

    print(f"--- Predicting: {home_team} vs {away_team} ({match_date}) ---")

    # 1. Load historical results up to (but not including) the match date, so
    # Elo/form reflect only what was knowable beforehand — no leakage.
    past_results_df = get_past_match_results(start_date="2002-01-01", end_date=match_date)

    for team in (home_team, away_team):
        n_matches = ((past_results_df["home_team"] == team) | (past_results_df["away_team"] == team)).sum()
        if n_matches == 0:
            print(
                f"[Warning] '{team}' has no matches in the historical dataset — "
                "it will default to a neutral 1500 Elo / 0.5 form, making the "
                "prediction effectively uninformative for that team. Double-check spelling."
            )
        else:
            print(f"[Info] Found {n_matches} historical matches for '{team}'.")

    # 2. Build a single-row "upcoming match" frame and reuse the normal feature pipeline
    match_df = pd.DataFrame(
        [{"match_id": "manual_1", "date": pd.to_datetime(match_date), "home_team": home_team, "away_team": away_team}]
    )
    features_df = generate_features(match_df, past_results_df)
    row = features_df.iloc[0]

    print(
        f"[Info] Elo: {home_team}={row['home_elo']:.0f}  {away_team}={row['away_elo']:.0f}  "
        f"(difference: {row['elo_difference']:+.0f})"
    )
    print(
        f"[Info] Form (last 5, pts/game): {home_team}={row['home_form']:.2f}  {away_team}={row['away_form']:.2f}"
    )

    # 3. Load trained models
    outcome_model = load_model(f"{MODELS_PATH}/match_outcome_model.joblib")
    score_model = load_model(f"{MODELS_PATH}/scoreline_model.joblib")

    # 4. Outcome probabilities
    outcome_probs = outcome_model.predict_proba(features_df)[0]
    features_df["prob_away_win"] = outcome_probs[0]
    features_df["prob_draw"] = outcome_probs[1]
    features_df["prob_home_win"] = outcome_probs[2]

    # 5. Scoreline probabilities + expected goals
    scoreline_probs = score_model.predict_score_probabilities(features_df)
    home_xg, away_xg = score_model.predict_expected_goals(row)
    top_scorelines = sorted(scoreline_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]

    # --- Report ---
    print("\n=== Match Outcome ===")
    print(f"{home_team} win : {outcome_probs[2]:.1%}")
    print(f"Draw            : {outcome_probs[1]:.1%}")
    print(f"{away_team} win : {outcome_probs[0]:.1%}")

    print(f"\n=== Expected Goals ===\n{home_team}: {home_xg:.2f}   {away_team}: {away_xg:.2f}")

    print("\n=== Most Likely Scorelines ===")
    for score, prob in top_scorelines:
        h, a = score.split("-")
        print(f"{home_team} {h} - {a} {away_team}: {prob:.1%}")

    # 6. Plot (saved separately so it never overwrites the regular matchday plot)
    safe_name = f"single_match_{home_team}_vs_{away_team}".replace(" ", "_").lower()
    plot_next_matchday_predictions(features_df, output_filename=f"{safe_name}.png")

    return features_df, scoreline_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a single specific match.")
    parser.add_argument("--home", required=True, help="Home team name, e.g. 'Uzbekistan'")
    parser.add_argument("--away", required=True, help="Away team name, e.g. 'Colombia'")
    parser.add_argument("--date", default=None, help="Match date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    predict_single_match(args.home, args.away, args.date)

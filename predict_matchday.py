import os
from datetime import datetime

import pandas as pd

from src.scraper import get_upcoming_matches, get_past_match_results
from src.features import generate_features
from src.model import load_model, MatchOutcomeModel, ScorelineModel
from src.visualize import plot_next_matchday_predictions

DATA_RAW_PATH = "data/raw"
DATA_TOURNAMENT_LOGS_PATH = "data/tournament_logs"
MODELS_PATH = "models"

os.makedirs(DATA_RAW_PATH, exist_ok=True)
os.makedirs(DATA_TOURNAMENT_LOGS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


def predict_matchday():
    print("--- Running Predict Matchday Script ---")

    # 1. Scrape upcoming matches
    upcoming_matches_df = get_upcoming_matches()
    if upcoming_matches_df.empty:
        print("No upcoming matches to predict. Exiting.")
        return

    today_str = datetime.now().strftime("%Y%m%d")
    upcoming_matches_df.to_parquet(
        os.path.join(DATA_RAW_PATH, f"upcoming_matches_{today_str}.parquet"), index=False
    )
    print(f"[Predict] Saved upcoming matches to {DATA_RAW_PATH}/upcoming_matches_{today_str}.parquet")

    # 2. Load historical data for feature generation
    past_results_df = get_past_match_results(
        end_date=upcoming_matches_df["date"].min().strftime("%Y-%m-%d")
    )
    if past_results_df.empty:
        print("No past results available for feature generation. Exiting.")
        return

    # 3. Generate features for upcoming matches (uses Elo/form as of "now", which is
    # correct here since these matches haven't happened yet)
    features_df = generate_features(upcoming_matches_df, past_results_df)

    # 4. Load trained models (or initialize fresh ones if this is the first run)
    try:
        outcome_model = load_model(os.path.join(MODELS_PATH, "match_outcome_model.joblib"))
        score_model = load_model(os.path.join(MODELS_PATH, "scoreline_model.joblib"))
    except FileNotFoundError:
        print("Models not found. Run update_and_evaluate.py first to train models.")
        print("Initializing untrained models for this run only.")
        outcome_model = MatchOutcomeModel()
        score_model = ScorelineModel()

    # 5. Run inference
    print("[Predict] Generating match outcome probabilities...")
    if hasattr(outcome_model, "model") and hasattr(outcome_model.model, "classes_"):
        outcome_probabilities = outcome_model.predict_proba(features_df)
        features_df["prob_away_win"] = outcome_probabilities[:, 0]
        features_df["prob_draw"] = outcome_probabilities[:, 1]
        features_df["prob_home_win"] = outcome_probabilities[:, 2]
    else:
        # Untrained fallback model: uninformative uniform prior
        features_df["prob_away_win"] = 0.33
        features_df["prob_draw"] = 0.34
        features_df["prob_home_win"] = 0.33

    print("[Predict] Generating scoreline probabilities...")
    scoreline_predictions = []
    predicted_home_goals = []
    predicted_away_goals = []
    for _, row in features_df.iterrows():
        row_df = pd.DataFrame([row])
        scoreline_predictions.append(score_model.predict_score_probabilities(row_df))
        home_xg, away_xg = score_model.predict_expected_goals(row)
        predicted_home_goals.append(home_xg)
        predicted_away_goals.append(away_xg)

    features_df["scoreline_probabilities"] = scoreline_predictions
    features_df["predicted_home_goals"] = predicted_home_goals
    features_df["predicted_away_goals"] = predicted_away_goals

    # 6. Log predictions (predicted_*_goals are kept so update_and_evaluate.py can
    # compute a real RMSE later, instead of a placeholder)
    predictions_log_df = features_df[
        [
            "match_id", "date", "home_team", "away_team",
            "prob_home_win", "prob_draw", "prob_away_win",
            "predicted_home_goals", "predicted_away_goals",
            "scoreline_probabilities",
        ]
    ].copy()
    predictions_log_df["prediction_timestamp"] = datetime.now()

    log_filename = os.path.join(DATA_TOURNAMENT_LOGS_PATH, f"predictions_{today_str}.parquet")
    predictions_log_df.to_parquet(log_filename, index=False)
    print(f"[Predict] Saved matchday predictions to {log_filename}")

    # 7. Plot
    plot_next_matchday_predictions(predictions_log_df)
    print("--- Predict Matchday Script Finished ---")


if __name__ == "__main__":
    predict_matchday()

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.scraper import fetch_worldcup_2026_schedule, get_past_match_results
from src.features import build_training_dataset, EloRating
from src.model import MatchOutcomeModel, ScorelineModel, save_model
from src.visualize import (
    plot_model_accuracy_leaderboard,
    plot_model_evolution_elo,
    plot_model_evolution_elo_interactive,
)

DATA_RAW_PATH = "data/raw"
DATA_TOURNAMENT_LOGS_PATH = "data/tournament_logs"
MODELS_PATH = "models"
PLOTS_PATH = "plots"

EVALUATED_MATCHES_PATH = os.path.join(DATA_TOURNAMENT_LOGS_PATH, "evaluated_matches.parquet")
MODEL_PERFORMANCE_PATH = os.path.join(DATA_TOURNAMENT_LOGS_PATH, "model_performance.csv")

os.makedirs(DATA_RAW_PATH, exist_ok=True)
os.makedirs(DATA_TOURNAMENT_LOGS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)


def _load_all_prediction_logs():
    files = [
        os.path.join(DATA_TOURNAMENT_LOGS_PATH, f)
        for f in os.listdir(DATA_TOURNAMENT_LOGS_PATH)
        if f.startswith("predictions_") and f.endswith(".parquet")
    ]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(subset=["match_id"])


def _load_evaluated_match_ids():
    if os.path.exists(EVALUATED_MATCHES_PATH):
        return set(pd.read_parquet(EVALUATED_MATCHES_PATH)["match_id"])
    return set()


def update_and_evaluate():
    print("--- Running Update and Evaluate Script ---")

    # 1. Pull the current, complete official World Cup 2026 schedule and snapshot
    # the finished matches. This is overwritten every run (the source of truth is
    # always "what does the official schedule currently say"), rather than
    # accumulated day-by-day, since match days aren't strictly daily.
    schedule_df = fetch_worldcup_2026_schedule()
    finished_df = schedule_df[schedule_df["played"]][
        ["match_id", "date", "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    finished_df.to_parquet(os.path.join(DATA_RAW_PATH, "worldcup2026_results.parquet"), index=False)
    print(f"[Update] {len(finished_df)} World Cup 2026 matches finished so far.")

    # 2. Find matches that are BOTH finished AND were previously predicted AND
    # haven't already been scored in an earlier run — this is what makes it safe
    # to run this script daily all the way to the final without double-counting
    # or assuming exactly one matchday happened "yesterday".
    all_predictions_df = _load_all_prediction_logs()
    already_evaluated = _load_evaluated_match_ids()

    if all_predictions_df.empty:
        print("[Update] No prediction logs found yet (run predict_matchday.py first). Skipping evaluation.")
        eval_df = pd.DataFrame()
    else:
        eval_df = pd.merge(
            finished_df, all_predictions_df, on=["date", "home_team", "away_team"], suffixes=("_actual", "_pred")
        )
        eval_df = eval_df[~eval_df["match_id_actual"].isin(already_evaluated)]

    # 3. Evaluate model performance on whatever is newly available, grouped by
    # match date so the leaderboard plot reflects real tournament progression.
    if not eval_df.empty:
        eval_df["actual_outcome_label"] = eval_df.apply(
            lambda r: 2 if r["home_score"] > r["away_score"] else (0 if r["home_score"] < r["away_score"] else 1),
            axis=1,
        )

        rows = []
        for eval_date, group in eval_df.groupby("date"):
            y_true = group["actual_outcome_label"]
            y_pred_proba = np.clip(group[["prob_away_win", "prob_draw", "prob_home_win"]].values, 1e-15, 1 - 1e-15)
            current_log_loss = log_loss(y_true, y_pred_proba, labels=[0, 1, 2])

            if {"predicted_home_goals", "predicted_away_goals"}.issubset(group.columns):
                sq_errors = np.concatenate([
                    (group["predicted_home_goals"] - group["home_score"]).values ** 2,
                    (group["predicted_away_goals"] - group["away_score"]).values ** 2,
                ])
                current_rmse = float(np.sqrt(sq_errors.mean()))
            else:
                current_rmse = np.nan

            rows.append({
                "date": eval_date.strftime("%Y-%m-%d"),
                "log_loss": current_log_loss,
                "rmse": current_rmse,
                "n_matches": len(group),
            })
            print(f"[Update] {eval_date.date()}: {len(group)} match(es) evaluated — Log-Loss={current_log_loss:.4f}, RMSE={current_rmse:.4f}")

        new_rows_df = pd.DataFrame(rows)
        if os.path.exists(MODEL_PERFORMANCE_PATH):
            new_rows_df.to_csv(MODEL_PERFORMANCE_PATH, mode="a", header=False, index=False)
        else:
            new_rows_df.to_csv(MODEL_PERFORMANCE_PATH, index=False)

        # Remember these matches as evaluated so they're never double-counted.
        newly_evaluated_ids = eval_df["match_id_actual"].rename("match_id")
        updated_evaluated = pd.concat(
            [pd.DataFrame({"match_id": list(already_evaluated)}), newly_evaluated_ids.to_frame()],
            ignore_index=True,
        ).drop_duplicates()
        updated_evaluated.to_parquet(EVALUATED_MATCHES_PATH, index=False)

        plot_model_accuracy_leaderboard(pd.read_csv(MODEL_PERFORMANCE_PATH))
    else:
        print("[Update] No newly-finished matches to evaluate this run (already up to date, or no predictions logged yet).")

    # 4. Rebuild a leak-free training set and retrain both models on everything
    # available right now: the broad historical baseline + every official World
    # Cup 2026 result so far.
    combined_history_df = get_past_match_results(start_date="2002-01-01", end_date=datetime.now().strftime("%Y-%m-%d"))
    training_df, _ = build_training_dataset(combined_history_df)

    outcome_model = MatchOutcomeModel()
    outcome_model.train(training_df, training_df["outcome_label"])
    save_model(outcome_model, os.path.join(MODELS_PATH, "match_outcome_model.joblib"))

    score_model = ScorelineModel()
    score_model.train(training_df, training_df["home_score"], training_df["away_score"])
    save_model(score_model, os.path.join(MODELS_PATH, "scoreline_model.joblib"))

    # 5. Elo evolution history (for the leaderboard/evolution plots) — only a
    # (date, team, rating) point for the two teams that actually played that
    # match, to keep the chart and the interactive HTML file lightweight.
    elo_system_for_history = EloRating()
    elo_history_records = []
    for _, row in combined_history_df.sort_values(by="date").iterrows():
        home, away = row["home_team"], row["away_team"]
        elo_system_for_history.update_ratings(home, away, row["home_score"], row["away_score"])
        elo_history_records.append({"date": row["date"], "team": home, "elo_rating": elo_system_for_history.get_rating(home)})
        elo_history_records.append({"date": row["date"], "team": away, "elo_rating": elo_system_for_history.get_rating(away)})

    elo_ratings_history_df = pd.DataFrame(elo_history_records)
    elo_ratings_history_df.to_parquet(os.path.join(DATA_TOURNAMENT_LOGS_PATH, "elo_ratings_history.parquet"), index=False)

    TOP_N_TEAMS = 12
    latest_ratings = elo_ratings_history_df.sort_values("date").groupby("team").tail(1)
    top_teams = latest_ratings.sort_values("elo_rating", ascending=False).head(TOP_N_TEAMS)["team"]
    elo_top_teams_df = elo_ratings_history_df[elo_ratings_history_df["team"].isin(top_teams)]

    plot_model_evolution_elo(elo_top_teams_df)
    plot_model_evolution_elo_interactive(elo_top_teams_df)

    print("--- Update and Evaluate Script Finished ---")


if __name__ == "__main__":
    update_and_evaluate()

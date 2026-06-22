
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os

PLOTS_PATH = "plots"

def plot_model_accuracy_leaderboard(eval_metrics_df, output_filename="model_accuracy_leaderboard.png"):
    """
    Generates a plot showing model accuracy (e.g., Log-Loss) over time.
    """
    print("[Visualize] Generating model accuracy leaderboard plot...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=eval_metrics_df, x="date", y="log_loss", marker="o")
    plt.title("Model Log-Loss Evolution Over Time")
    plt.xlabel("Date")
    plt.ylabel("Log-Loss")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(PLOTS_PATH, output_filename)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualize] Saved model accuracy leaderboard plot to {output_path}")

def plot_next_matchday_predictions(predictions_df, output_filename="next_matchday_predictions.png"):
    """
    Generates bar plots for next matchday prediction probabilities.
    """
    print("[Visualize] Generating next matchday predictions plot...")
    if predictions_df.empty:
        print("[Visualize] No predictions to plot.")
        return

    fig, axes = plt.subplots(len(predictions_df), 1, figsize=(10, 4 * len(predictions_df)))
    if len(predictions_df) == 1:
        axes = [axes]

    for i, (index, row) in enumerate(predictions_df.iterrows()):
        labels = [f"{row['home_team']} Win", "Draw", f"{row['away_team']} Win"]
        probabilities = [row["prob_home_win"], row["prob_draw"], row["prob_away_win"]]

        sns.barplot(
            x=labels, y=probabilities, hue=labels, ax=axes[i],
            palette=["skyblue", "lightgray", "salmon"], legend=False,
        )
        axes[i].set_title(f"Match: {row['home_team']} vs {row['away_team']} ({row['date'].strftime('%Y-%m-%d')})")
        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel("Probability")

    plt.tight_layout()
    output_path = os.path.join(PLOTS_PATH, output_filename)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualize] Saved next matchday predictions plot to {output_path}")

def plot_model_evolution_elo(elo_ratings_history_df, output_filename="model_evolution_elo.png"):
    """
    Generates a plot showing the evolution of Elo ratings for key teams over time.
    """
    print("[Visualize] Generating model evolution (Elo) plot...")
    if elo_ratings_history_df.empty:
        print("[Visualize] No Elo history to plot.")
        return

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=elo_ratings_history_df, x="date", y="elo_rating", hue="team", marker="o")
    plt.title("Elo Rating Evolution of Top Teams")
    plt.xlabel("Date")
    plt.ylabel("Elo Rating")
    plt.grid(True)
    plt.legend(title="Team", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    output_path = os.path.join(PLOTS_PATH, output_filename)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualize] Saved model evolution (Elo) plot to {output_path}")

def plot_model_evolution_elo_interactive(elo_ratings_history_df, output_filename="model_evolution_elo_interactive.html"):
    """
    Generates an interactive Plotly plot showing the evolution of Elo ratings for key teams over time.
    """
    print("[Visualize] Generating interactive model evolution (Elo) plot...")
    if elo_ratings_history_df.empty:
        print("[Visualize] No Elo history to plot.")
        return

    fig = go.Figure()
    for team in elo_ratings_history_df["team"].unique():
        team_df = elo_ratings_history_df[elo_ratings_history_df["team"] == team]
        fig.add_trace(go.Scatter(x=team_df["date"], y=team_df["elo_rating"], mode="lines+markers", name=team))

    fig.update_layout(
        title="Interactive Elo Rating Evolution of Top Teams",
        xaxis_title="Date",
        yaxis_title="Elo Rating",
        hovermode="x unified"
    )
    output_path = os.path.join(PLOTS_PATH, output_filename)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"[Visualize] Saved interactive model evolution (Elo) plot to {output_path}")

if __name__ == "__main__":
    print("--- Testing Visualize Module ---")
    # Dummy data for testing
    eval_data = {
        "date": pd.to_datetime(["2026-06-10", "2026-06-11", "2026-06-12"]),
        "log_loss": [0.6, 0.55, 0.58],
        "rmse": [1.0, 0.95, 0.98]
    }
    eval_metrics_df = pd.DataFrame(eval_data)
    plot_model_accuracy_leaderboard(eval_metrics_df)

    predictions_data = [
        {"match_id": "M001", "date": pd.to_datetime("2026-06-17"), "home_team": "Brazil", "away_team": "Argentina", "prob_home_win": 0.4, "prob_draw": 0.3, "prob_away_win": 0.3},
        {"match_id": "M002", "date": pd.to_datetime("2026-06-17"), "home_team": "Germany", "away_team": "France", "prob_home_win": 0.35, "prob_draw": 0.25, "prob_away_win": 0.4},
    ]
    predictions_df = pd.DataFrame(predictions_data)
    plot_next_matchday_predictions(predictions_df)

    elo_history_data = [
        {"date": pd.to_datetime("2026-06-01"), "team": "Brazil", "elo_rating": 1800},
        {"date": pd.to_datetime("2026-06-01"), "team": "Argentina", "elo_rating": 1780},
        {"date": pd.to_datetime("2026-06-01"), "team": "Germany", "elo_rating": 1750},
        {"date": pd.to_datetime("2026-06-05"), "team": "Brazil", "elo_rating": 1810},
        {"date": pd.to_datetime("2026-06-05"), "team": "Argentina", "elo_rating": 1770},
        {"date": pd.to_datetime("2026-06-05"), "team": "Germany", "elo_rating": 1760},
        {"date": pd.to_datetime("2026-06-10"), "team": "Brazil", "elo_rating": 1805},
        {"date": pd.to_datetime("2026-06-10"), "team": "Argentina", "elo_rating": 1790},
        {"date": pd.to_datetime("2026-06-10"), "team": "Germany", "elo_rating": 1740},
    ]
    elo_ratings_history_df = pd.DataFrame(elo_history_data)
    plot_model_evolution_elo(elo_ratings_history_df)
    plot_model_evolution_elo_interactive(elo_ratings_history_df)
    print("Test plots generated successfully.")

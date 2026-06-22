
import pandas as pd
import numpy as np
from collections import defaultdict

class EloRating:
    def __init__(self, k=30, initial_rating=1500):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = defaultdict(lambda: self.initial_rating)

    def _expected_score(self, r1, r2):
        return 1 / (1 + 10**((r2 - r1) / 400))

    def update_ratings(self, home_team, away_team, home_score, away_score):
        r_home = self.ratings[home_team]
        r_away = self.ratings[away_team]

        e_home = self._expected_score(r_home, r_away)
        e_away = self._expected_score(r_away, r_home)

        if home_score > away_score:
            s_home = 1
            s_away = 0
        elif home_score < away_score:
            s_home = 0
            s_away = 1
        else:
            s_home = 0.5
            s_away = 0.5

        self.ratings[home_team] += self.k * (s_home - e_home)
        self.ratings[away_team] += self.k * (s_away - e_away)

    def get_rating(self, team):
        return self.ratings[team]

def calculate_elo_ratings(match_results_df):
    elo_system = EloRating()
    # Sort by date to ensure correct Elo progression
    match_results_df = match_results_df.sort_values(by='date')

    for index, row in match_results_df.iterrows():
        elo_system.update_ratings(row["home_team"], row["away_team"], row["home_score"], row["away_score"])
    
    return elo_system.ratings

def generate_rolling_form(match_results_df, window=5):
    # This is a simplified rolling form calculation (e.g., recent wins/losses)
    # A more sophisticated approach would consider goal difference, strength of opponent, etc.
    team_form = defaultdict(lambda: [])
    rolling_form_features = {}

    match_results_df = match_results_df.sort_values(by='date')

    for index, row in match_results_df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]

        # Update form for home team
        if home_score > away_score:
            team_form[home_team].append(1) # Win
        elif home_score < away_score:
            team_form[home_team].append(0) # Loss
        else:
            team_form[home_team].append(0.5) # Draw
        
        # Update form for away team
        if away_score > home_score:
            team_form[away_team].append(1) # Win
        elif away_score < home_score:
            team_form[away_team].append(0) # Loss
        else:
            team_form[away_team].append(0.5) # Draw

        # Keep only the last 'window' results
        team_form[home_team] = team_form[home_team][-window:]
        team_form[away_team] = team_form[away_team][-window:]

        # Calculate rolling form (average points per game in the window)
        rolling_form_features[home_team] = np.mean(team_form[home_team]) if team_form[home_team] else 0.5
        rolling_form_features[away_team] = np.mean(team_form[away_team]) if team_form[away_team] else 0.5

    return rolling_form_features

def build_training_dataset(past_results_df, form_window=5):
    """
    Builds a leak-free training set from historical results.

    For every past match, each team's Elo rating and rolling form are captured
    *exactly as they stood immediately before that match was played*, then the
    match result is applied to update both. This avoids the look-ahead bias of
    naively computing final/end-of-history Elo and form and reusing them as
    features for every earlier match in the same dataset.

    Returns a DataFrame with one row per historical match, ready to train
    MatchOutcomeModel / ScorelineModel on.
    """
    df = past_results_df.sort_values(by="date").reset_index(drop=True)
    elo_system = EloRating()
    team_form_history = defaultdict(list)

    rows = []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]

        # --- capture pre-match state (what the model would have known beforehand) ---
        home_elo = elo_system.get_rating(home)
        away_elo = elo_system.get_rating(away)
        home_form = np.mean(team_form_history[home][-form_window:]) if team_form_history[home] else 0.5
        away_form = np.mean(team_form_history[away][-form_window:]) if team_form_history[away] else 0.5

        rows.append(
            {
                "match_id": row.get("match_id"),
                "date": row["date"],
                "home_team": home,
                "away_team": away,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_difference": home_elo - away_elo,
                "home_form": home_form,
                "away_form": away_form,
                "form_difference": home_form - away_form,
                "home_score": row["home_score"],
                "away_score": row["away_score"],
            }
        )

        # --- now apply this match's actual result to update state for future rows ---
        elo_system.update_ratings(home, away, row["home_score"], row["away_score"])
        if row["home_score"] > row["away_score"]:
            team_form_history[home].append(1)
            team_form_history[away].append(0)
        elif row["home_score"] < row["away_score"]:
            team_form_history[home].append(0)
            team_form_history[away].append(1)
        else:
            team_form_history[home].append(0.5)
            team_form_history[away].append(0.5)

    training_df = pd.DataFrame(rows)
    training_df["outcome_label"] = training_df.apply(
        lambda r: 2 if r["home_score"] > r["away_score"] else (0 if r["home_score"] < r["away_score"] else 1),
        axis=1,
    )
    return training_df, dict(elo_system.ratings)


def generate_features(matches_df, past_results_df):
    print("[Features] Generating Elo ratings...")
    elo_ratings = calculate_elo_ratings(past_results_df)
    print("[Features] Generating rolling form...")
    rolling_form = generate_rolling_form(past_results_df)

    features_df = matches_df.copy()
    features_df["home_elo"] = features_df["home_team"].apply(lambda x: elo_ratings.get(x, 1500))
    features_df["away_elo"] = features_df["away_team"].apply(lambda x: elo_ratings.get(x, 1500))
    features_df["elo_difference"] = features_df["home_elo"] - features_df["away_elo"]

    features_df["home_form"] = features_df["home_team"].apply(lambda x: rolling_form.get(x, 0.5))
    features_df["away_form"] = features_df["away_team"].apply(lambda x: rolling_form.get(x, 0.5))
    features_df["form_difference"] = features_df["home_form"] - features_df["away_form"]

    print(f"[Features] Generated features for {len(features_df)} matches.")
    return features_df

if __name__ == '__main__':
    print("--- Testing Features Module ---")
    # Create dummy past results for testing
    past_results_data = [
        {'match_id': 'P001', 'date': '2026-06-01', 'home_team': 'Brazil', 'away_team': 'Germany', 'home_score': 2, 'away_score': 1},
        {'match_id': 'P002', 'date': '2026-06-02', 'home_team': 'Argentina', 'away_team': 'France', 'home_score': 0, 'away_score': 0},
        {'match_id': 'P003', 'date': '2026-06-03', 'home_team': 'Germany', 'away_team': 'Brazil', 'home_score': 1, 'away_score': 3},
        {'match_id': 'P004', 'date': '2026-06-04', 'home_team': 'France', 'away_team': 'Argentina', 'home_score': 2, 'away_score': 1},
        {'match_id': 'P005', 'date': '2026-06-05', 'home_team': 'Brazil', 'away_team': 'France', 'home_score': 1, 'away_score': 0},
    ]
    past_results_df = pd.DataFrame(past_results_data)
    past_results_df['date'] = pd.to_datetime(past_results_df['date'])

    # Create dummy upcoming matches for testing
    upcoming_matches_data = [
        {'match_id': 'U001', 'date': '2026-06-06', 'home_team': 'Brazil', 'away_team': 'Argentina'},
        {'match_id': 'U002', 'date': '2026-06-06', 'home_team': 'Germany', 'away_team': 'France'},
    ]
    upcoming_matches_df = pd.DataFrame(upcoming_matches_data)
    upcoming_matches_df['date'] = pd.to_datetime(upcoming_matches_df['date'])

    features = generate_features(upcoming_matches_df, past_results_df)
    print("Generated Features:")
    print(features.head())

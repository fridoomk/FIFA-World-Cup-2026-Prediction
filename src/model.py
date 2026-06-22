import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from scipy.stats import poisson


class MatchOutcomeModel:
    """Predicts Home/Draw/Away win probabilities. Labels: 0=Away, 1=Draw, 2=Home."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=25,
            random_state=42,
        )
        self.features = ["elo_difference", "form_difference"]

    def train(self, X, y):
        print("[Model] Training Match Outcome Model...")
        self.model.fit(X[self.features], y)
        print("[Model] Match Outcome Model trained.")

    def predict_proba(self, X):
        proba = self.model.predict_proba(X[self.features])
        # RandomForest only learns classes present in the training labels (e.g. if no
        # draws were ever seen, it won't have a "draw" column). Re-expand to a fixed
        # [away, draw, home] layout so downstream code can always rely on 3 columns.
        full = np.zeros((len(X), 3))
        for i, cls in enumerate(self.model.classes_):
            full[:, int(cls)] = proba[:, i]
        return full

    def evaluate(self, X, y_true):
        """Real Log-Loss against actual outcomes (y_true: 0=Away, 1=Draw, 2=Home)."""
        y_pred_proba = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)
        return log_loss(y_true, y_pred_proba, labels=[0, 1, 2])


class ScorelineModel:
    """
    Simplified, match-aware Poisson scoreline model.

    Rather than a single global average for every match, expected goals are
    adjusted per-match using the Elo difference between the two teams (a stronger
    team is expected to score more / concede less). This is still a simplification
    of a full Dixon-Coles model (no explicit attack/defense parameters per team,
    no low-score correlation term) but it is no longer identical for every fixture.
    """

    def __init__(self):
        self.base_home_goals = 1.5
        self.base_away_goals = 1.1
        # Goals shift per 100 Elo points of advantage (fit during training).
        self.elo_sensitivity = 0.08

    def train(self, X, home_goals, away_goals):
        print("[Model] Training Scoreline Model (Elo-adjusted Poisson)...")
        self.base_home_goals = float(home_goals.mean())
        self.base_away_goals = float(away_goals.mean())

        if "elo_difference" in X.columns and X["elo_difference"].std() > 1e-6:
            goal_diff = home_goals.values - away_goals.values
            elo_diff = X["elo_difference"].values
            # slope of goal_diff vs elo_diff/100, via simple least squares
            slope, _ = np.polyfit(elo_diff / 100.0, goal_diff, 1)
            self.elo_sensitivity = float(np.clip(slope, -0.5, 0.5))

        print(
            f"[Model] base_home_goals={self.base_home_goals:.2f}, "
            f"base_away_goals={self.base_away_goals:.2f}, "
            f"elo_sensitivity={self.elo_sensitivity:.3f}"
        )

    def _expected_goals(self, elo_difference):
        adjustment = self.elo_sensitivity * (elo_difference / 100.0)
        home_rate = max(0.15, self.base_home_goals + adjustment / 2)
        away_rate = max(0.15, self.base_away_goals - adjustment / 2)
        return home_rate, away_rate

    def predict_expected_goals(self, row):
        elo_difference = row.get("elo_difference", 0.0)
        return self._expected_goals(elo_difference)

    def predict_score_probabilities(self, X, max_goals=5):
        row = X.iloc[0] if isinstance(X, pd.DataFrame) else X
        home_rate, away_rate = self.predict_expected_goals(row)

        home_probs = [poisson.pmf(i, home_rate) for i in range(max_goals + 1)]
        away_probs = [poisson.pmf(i, away_rate) for i in range(max_goals + 1)]

        return {
            f"{h}-{a}": home_probs[h] * away_probs[a]
            for h in range(max_goals + 1)
            for a in range(max_goals + 1)
        }

    def evaluate(self, X, home_goals, away_goals):
        """Real RMSE between expected goals and actual goals, home+away combined."""
        preds = X.apply(lambda row: self._expected_goals(row.get("elo_difference", 0.0)), axis=1)
        pred_home = np.array([p[0] for p in preds])
        pred_away = np.array([p[1] for p in preds])

        sq_errors = np.concatenate(
            [
                (pred_home - home_goals.values) ** 2,
                (pred_away - away_goals.values) ** 2,
            ]
        )
        return float(np.sqrt(sq_errors.mean()))


def save_model(model, path):
    joblib.dump(model, path)
    print(f"[Model] Model saved to {path}")


def load_model(path):
    model = joblib.load(path)
    print(f"[Model] Model loaded from {path}")
    return model

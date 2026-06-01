import pandas as pd


def load_results(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def match_result(row):
    if row["home_score"] > row["away_score"]:
        return 2
    if row["home_score"] == row["away_score"]:
        return 1
    return 0


def create_training_data(results, lookback=10):
    team_history = {}
    rows = []

    for _, match in results.iterrows():
        home = match["home_team"]
        away = match["away_team"]

        home_hist = team_history.get(home, [])
        away_hist = team_history.get(away, [])

        home_recent = home_hist[-lookback:]
        away_recent = away_hist[-lookback:]

        home_points = sum(x["points"] for x in home_recent) / len(home_recent) if home_recent else 1.0
        away_points = sum(x["points"] for x in away_recent) / len(away_recent) if away_recent else 1.0

        home_gd = sum(x["goal_diff"] for x in home_recent) / len(home_recent) if home_recent else 0.0
        away_gd = sum(x["goal_diff"] for x in away_recent) / len(away_recent) if away_recent else 0.0

        rows.append({
            "neutral": int(bool(match.get("neutral", False))),
            "team_a_form_points": home_points,
            "team_b_form_points": away_points,
            "team_a_form_goal_diff": home_gd,
            "team_b_form_goal_diff": away_gd,
            "form_points_diff": home_points - away_points,
            "form_goal_diff_diff": home_gd - away_gd,
            "target": match_result(match),
        })

        home_match_gd = match["home_score"] - match["away_score"]
        away_match_gd = -home_match_gd

        home_match_points = 3 if home_match_gd > 0 else 1 if home_match_gd == 0 else 0
        away_match_points = 3 if away_match_gd > 0 else 1 if away_match_gd == 0 else 0

        team_history.setdefault(home, []).append({
            "points": home_match_points,
            "goal_diff": home_match_gd,
        })

        team_history.setdefault(away, []).append({
            "points": away_match_points,
            "goal_diff": away_match_gd,
        })

    return pd.DataFrame(rows)


def latest_team_features(results, lookback=10):
    team_history = {}

    for _, match in results.iterrows():
        home = match["home_team"]
        away = match["away_team"]

        home_gd = match["home_score"] - match["away_score"]
        away_gd = -home_gd

        home_points = 3 if home_gd > 0 else 1 if home_gd == 0 else 0
        away_points = 3 if away_gd > 0 else 1 if away_gd == 0 else 0

        team_history.setdefault(home, []).append({
            "points": home_points,
            "goal_diff": home_gd,
        })

        team_history.setdefault(away, []).append({
            "points": away_points,
            "goal_diff": away_gd,
        })

    features = {}

    for team, history in team_history.items():
        recent = history[-lookback:]

        features[team] = {
            "form_points": sum(x["points"] for x in recent) / len(recent) if recent else 1.0,
            "form_goal_diff": sum(x["goal_diff"] for x in recent) / len(recent) if recent else 0.0,
        }

    return features
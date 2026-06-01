import argparse
import random
from collections import Counter

import joblib
import pandas as pd

from config import RESULTS_PATH, TEAMS_PATH, MODEL_PATH, FEATURE_COLUMNS_PATH
from data_preparation import load_results, latest_team_features


def load_teams(path, top_n=32):
    teams = pd.read_csv(path)
    teams = teams.rename(columns={"rank": "fifa_rank"})

    required = {"team", "fifa_rank"}
    missing = required - set(teams.columns)
    if missing:
        raise ValueError(f"teams.csv missing columns: {missing}")

    teams["team"] = teams["team"].astype(str).str.strip()
    teams["fifa_rank"] = pd.to_numeric(teams["fifa_rank"], errors="coerce")

    return teams.dropna(subset=["team", "fifa_rank"]).sort_values("fifa_rank").head(top_n)


def get_form(team, form_table):
    return form_table.get(team, {"form_points": 1.0, "form_goal_diff": 0.0})


def build_features(team_a, team_b, form_table):
    a = get_form(team_a, form_table)
    b = get_form(team_b, form_table)

    return {
        "neutral": 1,
        "team_a_form_points": a["form_points"],
        "team_b_form_points": b["form_points"],
        "team_a_form_goal_diff": a["form_goal_diff"],
        "team_b_form_goal_diff": b["form_goal_diff"],
        "form_points_diff": a["form_points"] - b["form_points"],
        "form_goal_diff_diff": a["form_goal_diff"] - b["form_goal_diff"],
    }


def create_probability_cache(teams, model, feature_cols, form_table):
    print("Creating match probability cache...")

    team_names = teams["team"].tolist()
    rank_map = dict(zip(teams["team"], teams["fifa_rank"]))
    cache = {}

    for team_a in team_names:
        for team_b in team_names:
            if team_a == team_b:
                continue

            features = build_features(team_a, team_b, form_table)
            X = pd.DataFrame([features])[feature_cols]

            probs = model.predict_proba(X)[0]
            prob_map = dict(zip(model.classes_, probs))

            team_b_win = prob_map.get(0, 0.33)
            draw = prob_map.get(1, 0.33)
            team_a_win = prob_map.get(2, 0.33)

            rank_diff = rank_map[team_b] - rank_map[team_a]
            rank_bonus = max(min(rank_diff * 0.003, 0.15), -0.15)

            team_a_win += rank_bonus
            team_b_win -= rank_bonus

            team_a_win = max(team_a_win, 0.01)
            team_b_win = max(team_b_win, 0.01)
            draw = max(draw, 0.01)

            total = team_a_win + draw + team_b_win

            cache[(team_a, team_b)] = (
                team_a_win / total,
                draw / total,
                team_b_win / total,
            )

    return cache


def predict_match(team_a, team_b, probability_cache):
    team_a_win, draw, team_b_win = probability_cache[(team_a, team_b)]

    result = random.choices(
        ["A", "Draw", "B"],
        weights=[team_a_win, draw, team_b_win],
        k=1,
    )[0]

    if result == "A":
        return team_a
    elif result == "B":
        return team_b
    else:
        return random.choice([team_a, team_b])


def simulate_tournament(teams, probability_cache):
    team_list = teams["team"].tolist()
    random.shuffle(team_list)

    while len(team_list) > 1:
        next_round = []

        for i in range(0, len(team_list), 2):
            winner = predict_match(team_list[i], team_list[i + 1], probability_cache)
            next_round.append(winner)

        team_list = next_round

    return team_list[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=32)
    args = parser.parse_args()

    print("Loading model and data...")

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLUMNS_PATH)

    results = load_results(RESULTS_PATH)
    teams = load_teams(TEAMS_PATH, top_n=args.top_n)

    if len(teams) % 2 != 0:
        raise ValueError("Number of teams must be even.")

    form_table = latest_team_features(results)

    probability_cache = create_probability_cache(
        teams,
        model,
        feature_cols,
        form_table,
    )

    print(f"Running {args.sims} simulations with {len(teams)} teams...")

    winners = []

    for i in range(args.sims):
        if i % 10 == 0:
            print(f"Simulation {i}/{args.sims}")

        winner = simulate_tournament(teams, probability_cache)
        winners.append(winner)

    counts = Counter(winners)

    print("\nWorld Cup Winner Probabilities:\n")

    for team, count in counts.most_common(20):
        probability = count / args.sims * 100
        print(f"{team}: {probability:.2f}%")


if __name__ == "__main__":
    main()
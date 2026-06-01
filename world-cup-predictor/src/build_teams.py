from config import RESULTS_PATH, FIFA_RANKINGS_PATH, TEAMS_PATH
from data_preparation import build_all_teams_file


def main():
    teams = build_all_teams_file(RESULTS_PATH, FIFA_RANKINGS_PATH, TEAMS_PATH)
    print(f"Saved {len(teams)} teams to {TEAMS_PATH}")
    if teams["rank"].replace("", None).notna().any():
        print("FIFA rankings merged successfully.")
    else:
        print("No FIFA rankings found. Add data/raw/fifa_rankings.csv for stronger simulations.")


if __name__ == "__main__":
    main()

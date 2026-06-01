import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from data_preparation import load_results, create_training_data
from config import RESULTS_PATH, MODEL_DIR, MODEL_PATH, FEATURE_COLUMNS_PATH


def main():

    print("Training started...")

    MODEL_DIR.mkdir(exist_ok=True)
    results = load_results(RESULTS_PATH)
    data = create_training_data(results)

    feature_cols = [
        "neutral",
        "team_a_form_points",
        "team_b_form_points",
        "team_a_form_goal_diff",
        "team_b_form_goal_diff",
        "form_points_diff",
        "form_goal_diff_diff",
    ]

    X = data[feature_cols]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, preds), 3))
    print(classification_report(y_test, preds, target_names=["Team B win", "Draw", "Team A win"]))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_COLUMNS_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

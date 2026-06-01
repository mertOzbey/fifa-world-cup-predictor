from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"

RESULTS_PATH = DATA_RAW_DIR / "results.csv"
TEAMS_PATH = DATA_DIR / "teams.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "match_outcome_model.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"
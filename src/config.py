"""Central configuration: paths, constants and column groups.

Keeping these in one place means every script and the notebook agree on
which columns are used and where files are saved.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "Dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "Aggregate rating"

# Columns removed BEFORE modelling, with the reason for each.
DROPPED_COLUMNS = {
    "Rating color": "LEAKAGE - deterministic bin of the target",
    "Rating text": "LEAKAGE - deterministic bin of the target",
    "Restaurant ID": "identifier, carries no signal",
    "Restaurant Name": "identifier-like free text (7446 unique names)",
    "Address": "free text, ~8900 unique values",
    "Locality": "very high cardinality (1208); City + lat/long cover location",
    "Locality Verbose": "very high cardinality (1265); duplicates Locality",
    "Switch to order menu": "constant column (only 'No')",
}

# Raw columns the model needs at prediction time.
INPUT_FEATURES = [
    "Country Code", "City", "Longitude", "Latitude", "Cuisines",
    "Average Cost for two", "Currency", "Has Table booking",
    "Has Online delivery", "Is delivering now", "Price range", "Votes",
]

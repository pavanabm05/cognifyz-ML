"""Loading and light cleaning of the Cognifyz restaurant dataset."""
import pandas as pd
from config import DATA_PATH, TARGET, INPUT_FEATURES


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Read the official CSV (utf-8 with BOM) exactly as provided."""
    return pd.read_csv(path, encoding="utf-8-sig")


def prepare_data(df: pd.DataFrame, drop_unrated: bool = True):
    """Return (X, y) ready for the modelling pipeline.

    Steps
    -----
    * remove exact duplicate rows (none exist in this file, kept for safety)
    * optionally drop 'Not rated' restaurants (rating == 0): a 0 means "no
      rating exists yet" (all have <= 3 votes), not "terrible food"
    * fill the 9 missing Cuisines with 'Unknown'
    * keep only the raw input columns; leakage/ID columns are excluded
    """
    df = df.drop_duplicates().copy()
    if drop_unrated:
        df = df[df[TARGET] > 0]
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")
    return df[INPUT_FEATURES].reset_index(drop=True), df[TARGET].reset_index(drop=True)

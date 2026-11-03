"""Use the saved model to predict a restaurant's aggregate rating.

    python src/predict.py --demo           # predict 5 real restaurants, compare with actual
    python src/predict.py --interactive    # type in your own restaurant's details
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from config import MODEL_PATH, INPUT_FEATURES, RANDOM_STATE, TEST_SIZE
from data_loader import load_data, prepare_data


def load_model():
    if not MODEL_PATH.exists():
        sys.exit("Model not found. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


def predict_rating(model, record: dict) -> float:
    """Predict from a dict of raw restaurant fields (see INPUT_FEATURES)."""
    row = pd.DataFrame([record])[INPUT_FEATURES]
    return float(np.clip(model.predict(row)[0], 0, 5))   # ratings live in [0, 5]


def demo(model):
    X, y = prepare_data(load_data())
    # Re-create the SAME split used in training so we only show unseen restaurants
    _, X, _, y = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print("Sample restaurants from the held-out TEST set (never seen in training):")
    idx = X.sample(5, random_state=7).index
    print(f"{'City':<18}{'Cuisines':<32}{'Votes':>6}{'Actual':>8}{'Predicted':>11}")
    for i in idx:
        p = predict_rating(model, X.loc[i].to_dict())
        print(f"{X.loc[i, 'City'][:17]:<18}{X.loc[i, 'Cuisines'][:30]:<32}{X.loc[i, 'Votes']:>6}{y[i]:>8.1f}{p:>11.2f}")


def ask(prompt, cast=str, default=None, choices=None):
    while True:
        raw = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        raw = raw or (str(default) if default is not None else "")
        try:
            val = cast(raw)
            if choices and val not in choices:
                raise ValueError
            return val
        except ValueError:
            print(f"  Invalid. Expected {choices or cast.__name__}.")


def interactive(model):
    print("Enter restaurant details (press Enter to accept the [default]).")
    rec = {
        "Country Code": ask("Country Code (1 = India)", int, 1),
        "City": ask("City", str, "New Delhi"),
        "Longitude": ask("Longitude", float, 77.2),
        "Latitude": ask("Latitude", float, 28.6),
        "Cuisines": ask("Cuisines (comma separated)", str, "North Indian, Chinese"),
        "Average Cost for two": ask("Average cost for two (local currency)", int, 600),
        "Currency": ask("Currency exactly as in dataset", str, "Indian Rupees(Rs.)"),
        "Has Table booking": ask("Table booking (Yes/No)", str, "No", ["Yes", "No"]),
        "Has Online delivery": ask("Online delivery (Yes/No)", str, "No", ["Yes", "No"]),
        "Is delivering now": ask("Delivering now (Yes/No)", str, "No", ["Yes", "No"]),
        "Price range": ask("Price range (1-4)", int, 2, [1, 2, 3, 4]),
        "Votes": ask("Number of votes", int, 100),
    }
    print(f"\nPredicted aggregate rating: {predict_rating(model, rec):.2f} / 5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()
    m = load_model()
    if args.interactive:
        interactive(m)
    else:
        demo(m)

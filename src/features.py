"""Custom feature engineering, written as a scikit-learn transformer.

Anything that LEARNS from data (top cuisines, median cost per currency) is
learned inside fit(), so within cross-validation and train/test splitting it
only ever sees training rows -> no data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BINARY_COLS = ["Has Table booking", "Has Online delivery", "Is delivering now"]


def _split_cuisines(series: pd.Series) -> pd.Series:
    return series.fillna("Unknown").apply(lambda s: [c.strip() for c in s.split(",")])


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Turn the raw restaurant columns into model-ready numeric features.

    Engineered features and why
    ---------------------------
    votes_log        Votes is extremely right-skewed (0 - 10,934); log1p
                     compresses it so linear models and plots behave.
    cost_ratio_log   'Average Cost for two' is in 12 different currencies, so
                     raw values are not comparable. cost / (median cost of that
                     currency in the training data), logged, says "how
                     expensive relative to local norms".
    num_cuisines     Number of cuisines served (menu breadth).
    cuisine_<name>   0/1 flags for the `top_n_cuisines` most common cuisines
                     (Cuisines is multi-valued, so plain one-hot is wrong).
    Yes/No columns   mapped to 1/0.
    """

    def __init__(self, top_n_cuisines: int = 15):
        self.top_n_cuisines = top_n_cuisines

    def fit(self, X, y=None):
        cuisines = _split_cuisines(X["Cuisines"]).explode()
        self.top_cuisines_ = cuisines.value_counts().head(self.top_n_cuisines).index.tolist()
        cost = X["Average Cost for two"].where(X["Average Cost for two"] > 0)
        self.currency_median_ = cost.groupby(X["Currency"]).median().to_dict()
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        out["votes_log"] = np.log1p(X["Votes"])

        cost = X["Average Cost for two"].where(X["Average Cost for two"] > 0)
        median = X["Currency"].map(self.currency_median_)
        ratio = cost / median
        # missing/zero cost or unseen currency -> neutral value (ratio 1 -> log 0)
        out["cost_ratio_log"] = np.log(ratio).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out["price_range"] = X["Price range"]
        lists = _split_cuisines(X["Cuisines"])
        out["num_cuisines"] = lists.apply(len)
        for c in self.top_cuisines_:
            out[f"cuisine_{c}"] = lists.apply(lambda l, c=c: int(c in l))
        for col in BINARY_COLS:
            out[col] = (X[col] == "Yes").astype(int)

        out["Latitude"] = X["Latitude"]
        out["Longitude"] = X["Longitude"]
        out["Country Code"] = X["Country Code"].astype(str)
        out["City"] = X["City"].astype(str)
        return out

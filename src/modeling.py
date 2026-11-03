"""Pipeline construction, evaluation helpers and hyper-parameter tuning."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import RANDOM_STATE, CV_FOLDS
from features import FeatureEngineer

CATEGORICAL = ["Country Code", "City"]
NUMERIC = ["votes_log", "cost_ratio_log", "num_cuisines", "price_range",
           "Latitude", "Longitude"]


def build_pipeline(estimator) -> Pipeline:
    """FeatureEngineer -> encode/scale -> estimator, as ONE object.

    Because everything is inside the pipeline, cross-validation re-fits the
    encoders/scalers on each training fold only (no leakage), and the saved
    model accepts raw restaurant columns directly.
    """
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                               min_frequency=20, sparse_output=False), CATEGORICAL),
         ("num", StandardScaler(), NUMERIC)],
        remainder="passthrough",          # 0/1 flags pass through unchanged
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", FeatureEngineer()), ("prep", prep), ("model", estimator)])


def get_models() -> dict:
    """Baseline + the three required regressors (default-ish settings)."""
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def evaluate(model, X_train, y_train, X_test, y_test, cv=True) -> dict:
    """Train/test metrics (+ 5-fold CV RMSE on the TRAIN set only)."""
    row = {}
    for name, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        for k, v in regression_metrics(y, model.predict(X)).items():
            row[f"{name}_{k}"] = v
    if cv:
        scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS,
                                 scoring="neg_root_mean_squared_error")
        row["cv_RMSE"], row["cv_RMSE_std"] = -scores.mean(), scores.std()
    return row


def compare_models(X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """Fit every baseline model and tabulate train/test/CV metrics."""
    fitted, rows = {}, []
    for name, est in get_models().items():
        pipe = build_pipeline(est).fit(X_train, y_train)
        fitted[name] = pipe
        rows.append({"Model": name, **evaluate(pipe, X_train, y_train, X_test, y_test)})
    return pd.DataFrame(rows).set_index("Model"), fitted


def tune_decision_tree(X_train, y_train):
    grid = {"model__max_depth": [4, 6, 8, 10, 12, None],
            "model__min_samples_leaf": [1, 5, 10, 20, 40]}
    gs = GridSearchCV(build_pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE)),
                      grid, cv=CV_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=-1)
    return gs.fit(X_train, y_train)


def tune_random_forest(X_train, y_train, n_iter=12):
    dist = {"model__n_estimators": [200, 300],
            "model__max_depth": [10, 15, 20, None],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__max_features": [0.3, 0.5, 0.7, 1.0]}
    rs = RandomizedSearchCV(
        build_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
        dist, n_iter=n_iter, cv=CV_FOLDS, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1)
    return rs.fit(X_train, y_train)

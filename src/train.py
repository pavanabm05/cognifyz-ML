"""End-to-end training script:  python src/train.py   (run from project root)

Steps: load -> split -> compare baseline models -> tune -> pick best by
cross-validation -> ablations -> feature importance -> save model + outputs.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import visualize as viz
from config import (FIGURES_DIR, MODEL_PATH, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, TEST_SIZE)
from data_loader import load_data, prepare_data
from modeling import build_pipeline, compare_models, evaluate, regression_metrics, tune_decision_tree, tune_random_forest
from sklearn.ensemble import RandomForestRegressor


def grouped_impurity_importance(pipe) -> pd.Series:
    """Random Forest importances per engineered feature; one-hot columns of
    Country Code / City are summed back into a single feature each."""
    names = pipe["prep"].get_feature_names_out()
    imp = pd.Series(pipe["model"].feature_importances_, index=names)
    groups = {}
    for n, v in imp.items():
        key = "Country Code" if n.startswith("Country Code_") else "City" if n.startswith("City_") else n
        groups[key] = groups.get(key, 0) + v
    return pd.Series(groups).sort_values(ascending=False)


def main():
    t0 = time.time()
    for d in (MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    X, y = prepare_data(raw)                      # rated restaurants only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Rows: {len(X)} | train {len(X_train)} | test {len(X_test)}")

    # ---- 1. baseline comparison (required: LR, DT, RF) --------------------
    base_table, base_models = compare_models(X_train, y_train, X_test, y_test)
    base_table.round(4).to_csv(OUTPUTS_DIR / "model_comparison_untuned.csv")
    viz.plot_model_comparison(base_table, "Untuned models (test set)", "model_comparison_untuned.png")
    viz.plot_overfit_check(base_table, "overfitting_check_untuned.png")
    print(base_table.round(3).to_string())

    # ---- 2. tuning (CV on the training set only) --------------------------
    print("Tuning Decision Tree ...")
    dt_search = tune_decision_tree(X_train, y_train)
    print("Tuning Random Forest ...")
    rf_search = tune_random_forest(X_train, y_train)
    tuned = {"Linear Regression": base_models["Linear Regression"],
             "Decision Tree (tuned)": dt_search.best_estimator_,
             "Random Forest (tuned)": rf_search.best_estimator_}
    rows = [{"Model": n, **evaluate(m, X_train, y_train, X_test, y_test)} for n, m in tuned.items()]
    tuned_table = pd.DataFrame(rows).set_index("Model")
    tuned_table.round(4).to_csv(OUTPUTS_DIR / "model_comparison_tuned.csv")
    viz.plot_model_comparison(tuned_table, "Final candidates after tuning (test set)", "model_comparison_tuned.png")
    viz.plot_overfit_check(tuned_table, "overfitting_check_tuned.png")
    print(tuned_table.round(3).to_string())

    best_params = {"Decision Tree": dt_search.best_params_, "Random Forest": rf_search.best_params_}
    (OUTPUTS_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

    # ---- 3. select by CROSS-VALIDATION RMSE (not by test score) -----------
    best_name = tuned_table["cv_RMSE"].idxmin()
    best = tuned[best_name]
    pred = best.predict(X_test)
    print(f"\nBest model by CV RMSE: {best_name}")

    viz.plot_actual_vs_predicted(y_test, pred, f"{best_name}: actual vs predicted", "actual_vs_predicted.png")
    viz.plot_residuals(y_test, pred, "residuals.png")
    viz.plot_learning_curve(best, X_train, y_train, "learning_curve.png")

    # ---- 4. ablations: how robust / how dependent on Votes? ---------------
    rf_params = {k.replace("model__", ""): v for k, v in rf_search.best_params_.items()}
    def rf_score(Xa, ya, label, mutate=None):
        a_tr, a_te, b_tr, b_te = train_test_split(Xa, ya, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        if mutate:
            a_tr, a_te = mutate(a_tr), mutate(a_te)
        m = build_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **rf_params)).fit(a_tr, b_tr)
        return {"Experiment": label, "n_rows": len(Xa), **regression_metrics(b_te, m.predict(a_te))}
    Xall, yall = prepare_data(raw, drop_unrated=False)
    ablation = pd.DataFrame([
        rf_score(X, y, "Main: rated restaurants, all features"),
        rf_score(X, y, "Rated restaurants, WITHOUT Votes", lambda d: d.assign(Votes=0)),
        rf_score(Xall, yall, "Including 'Not rated' (rating 0) rows"),
    ]).set_index("Experiment")
    ablation.round(4).to_csv(OUTPUTS_DIR / "ablation_results.csv")
    print(ablation.round(3).to_string())

    # ---- 5. feature importance ---------------------------------------------
    perm = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
                                  scoring="neg_root_mean_squared_error", n_jobs=1)
    perm_s = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    perm_s.to_csv(OUTPUTS_DIR / "permutation_importance.csv", header=["RMSE_increase"])
    viz.plot_importance(perm_s, "Permutation importance (raw input features, test set)",
                        "Increase in RMSE when the feature is shuffled", name="permutation_importance.png")
    if hasattr(best["model"], "feature_importances_"):
        imp = grouped_impurity_importance(best)
        imp.to_csv(OUTPUTS_DIR / "impurity_importance.csv", header=["importance"])
        viz.plot_importance(imp, f"{best_name}: built-in feature importance (engineered features)",
                            "Mean decrease in impurity", name="feature_importance.png")

    # ---- 6. sample predictions ---------------------------------------------
    sample = X_test.sample(10, random_state=RANDOM_STATE)
    demo = sample[["City", "Cuisines", "Price range", "Votes"]].copy()
    demo["Actual"] = y_test.loc[sample.index].values
    demo["Predicted"] = np.clip(best.predict(sample), 0, 5).round(2)
    demo["Abs error"] = (demo["Actual"] - demo["Predicted"]).abs().round(2)
    demo.to_csv(OUTPUTS_DIR / "sample_predictions.csv", index=False)
    print(demo.to_string(index=False))

    # ---- 7. save ------------------------------------------------------------
    joblib.dump(best, MODEL_PATH)
    summary = {"best_model": best_name, "selection_rule": "lowest 5-fold CV RMSE on training set",
               "n_train": len(X_train), "n_test": len(X_test),
               "test_metrics": regression_metrics(y_test, pred),
               "cv_RMSE": float(tuned_table.loc[best_name, "cv_RMSE"]), "best_params": best_params}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved model -> {MODEL_PATH}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()

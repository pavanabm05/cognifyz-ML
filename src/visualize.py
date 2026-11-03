"""Reusable plotting functions (all return the figure and optionally save it)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve

from config import FIGURES_DIR, CV_FOLDS, RANDOM_STATE

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
ACCENT = "#2a6f97"


def _finish(fig, name):
    fig.tight_layout()
    if name:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    return fig


def plot_rating_distribution(y_all, y_rated, name=None):
    """Left: all ratings incl. the spike at 0. Right: rated restaurants only."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    sns.histplot(y_all, bins=40, color=ACCENT, ax=ax[0])
    ax[0].set(title="All restaurants (note the spike at 0 = 'Not rated')",
              xlabel="Aggregate rating", ylabel="Count")
    sns.histplot(y_rated, bins=30, kde=True, color=ACCENT, ax=ax[1])
    ax[1].axvline(y_rated.mean(), color="crimson", ls="--", label=f"mean = {y_rated.mean():.2f}")
    ax[1].set(title="Rated restaurants only (modelling target)", xlabel="Aggregate rating", ylabel="")
    ax[1].legend()
    return _finish(fig, name)


def plot_model_comparison(table, title="Model comparison (test set)", name=None):
    """Side-by-side bars for RMSE (lower better) and R2 (higher better)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    order = table.index.tolist()
    sns.barplot(x=table["test_RMSE"], y=order, ax=ax[0], color=ACCENT)
    ax[0].set(title="Test RMSE (lower is better)", xlabel="RMSE", ylabel="")
    sns.barplot(x=table["test_R2"], y=order, ax=ax[1], color="#52b788")
    ax[1].set(title="Test R² (higher is better)", xlabel="R²", ylabel="")
    for a, col in zip(ax, ["test_RMSE", "test_R2"]):
        for i, v in enumerate(table[col]):
            a.text(max(v, 0) + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    fig.suptitle(title, y=1.02, fontsize=13)
    return _finish(fig, name)


def plot_overfit_check(table, name=None):
    """Train vs test R2 per model: a large gap signals overfitting."""
    d = table[["train_R2", "test_R2"]].reset_index().melt("Model", var_name="Split", value_name="R2")
    d["Split"] = d["Split"].map({"train_R2": "Train", "test_R2": "Test"})
    fig, ax = plt.subplots(figsize=(10, 4.2))
    sns.barplot(data=d, x="Model", y="R2", hue="Split", ax=ax)
    ax.set(title="Overfitting check: train vs test R²", xlabel="")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _finish(fig, name)


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs predicted", name=None):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(y_true, y_pred, s=12, alpha=0.35, color=ACCENT)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
    ax.set(title=title, xlabel="Actual rating", ylabel="Predicted rating")
    ax.legend()
    return _finish(fig, name)


def plot_residuals(y_true, y_pred, name=None):
    res = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].scatter(y_pred, res, s=12, alpha=0.35, color=ACCENT)
    ax[0].axhline(0, color="red", ls="--")
    ax[0].set(title="Residuals vs predicted", xlabel="Predicted rating", ylabel="Residual (actual - predicted)")
    sns.histplot(res, bins=40, kde=True, color=ACCENT, ax=ax[1])
    ax[1].set(title=f"Residual distribution (mean {res.mean():.3f}, std {res.std():.3f})", xlabel="Residual")
    return _finish(fig, name)


def plot_importance(series, title, xlabel, top=15, name=None):
    s = series.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(s) + 1.5))
    ax.barh(s.index, s.values, color=ACCENT)
    ax.set(title=title, xlabel=xlabel)
    return _finish(fig, name)


def plot_learning_curve(estimator, X, y, name=None):
    """Train vs CV error as training size grows (diagnoses under/overfitting)."""
    sizes, tr, va = learning_curve(
        estimator, X, y, cv=3, scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5), random_state=RANDOM_STATE, n_jobs=1)
    tr, va = -tr, -va
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arr, lab, c in ((tr, "Training RMSE", "#e76f51"), (va, "Cross-val RMSE", ACCENT)):
        ax.plot(sizes, arr.mean(1), "o-", label=lab, color=c)
        ax.fill_between(sizes, arr.mean(1) - arr.std(1), arr.mean(1) + arr.std(1), alpha=0.15, color=c)
    ax.set(title="Learning curve (final model)", xlabel="Training samples", ylabel="RMSE")
    ax.legend()
    return _finish(fig, name)

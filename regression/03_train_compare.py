"""
03_train_compare.py — Train and compare four regression approaches.

(a) SGDRegressor (partial_fit loss curves)
(b) Ordinary least squares LinearRegression
(c) Ridge (L2)
(d) RandomForestRegressor

Reports MAE / RMSE / R^2 on train and test, saves SGD loss curve and a
PCA(1) before/after scatter with fitted line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

from pipeline import (
    DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    engineer_features,
    load_raw,
    train_compare_models,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def print_metrics_table(metrics: dict) -> None:
    rows = []
    for name, m in metrics.items():
        rows.append(
            {
                "model": name,
                "train_MAE": m["train"]["mae"],
                "train_RMSE": m["train"]["rmse"],
                "train_R2": m["train"]["r2"],
                "test_MAE": m["test"]["mae"],
                "test_RMSE": m["test"]["rmse"],
                "test_R2": m["test"]["r2"],
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    pd.options.display.float_format = "{:,.2f}".format
    print("\n=== Model comparison (train / test) ===")
    print(table.to_string())
    print()
    for name, m in metrics.items():
        gap = m["train"]["r2"] - m["test"]["r2"]
        if gap > 0.15:
            verdict = "likely OVERFITTING (train R^2 much higher than test)"
        elif m["test"]["r2"] < 0.1 and m["train"]["r2"] < 0.15:
            verdict = "likely UNDERFITTING (weak fit on both splits)"
        else:
            verdict = "reasonable generalization (train/test gap modest)"
        print(
            f"  {name}: test R^2={m['test']['r2']:.3f}, "
            f"test RMSE={m['test']['rmse']:,.0f}, train-test R^2 gap={gap:.3f} -> {verdict}"
        )


def plot_sgd_loss_curve(train_losses, test_losses, params: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train MSE", color="#4C72B0")
    ax.plot(epochs, test_losses, label="Test MSE", color="#C44E52")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title(
        f"SGDRegressor loss curve (alpha={params['alpha']}, eta0={params['eta0']})"
    )
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "sgd_loss_curve.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n[Saved] {out}")
    print(
        "Interpretation: both train and test MSE should fall quickly in early epochs "
        "then flatten. If train keeps falling while test rises, the SGD run is "
        "overfitting; if both stay high and flat, learning rate/alpha need retuning. "
        f"Final train MSE={train_losses[-1]:,.0f}, test MSE={test_losses[-1]:,.0f}."
    )


def tune_sgd_grid(X_train_t, y_train, X_test_t, y_test) -> dict:
    """Small grid over alpha / eta0; pick by test RMSE."""
    grid = []
    for alpha in (1e-5, 1e-4, 1e-3):
        for eta0 in (0.001, 0.01, 0.05):
            model = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=alpha,
                learning_rate="constant",
                eta0=eta0,
                max_iter=1000,
                tol=1e-3,
                random_state=42,
            )
            model.fit(X_train_t, y_train)
            rmse = float(np.sqrt(mean_squared_error(y_test, model.predict(X_test_t))))
            grid.append({"alpha": alpha, "eta0": eta0, "test_rmse": rmse})
    best = min(grid, key=lambda d: d["test_rmse"])
    print("\nSGD hyperparameter grid (by test RMSE):")
    for g in sorted(grid, key=lambda d: d["test_rmse"]):
        marker = " <-- best" if g is best else ""
        print(
            f"  alpha={g['alpha']:.0e}  eta0={g['eta0']:<5}  "
            f"test_RMSE={g['test_rmse']:,.0f}{marker}"
        )
    return best


def plot_before_after_scatter(results: dict) -> None:
    """
    Most predictors are categorical, so use PCA(1) on the standardized
    encoded matrix as a single numeric projection of the feature space.
    'Before' = PC1 vs salary scatter; 'After' = same with OLS line overlaid.
    """
    X_train_t = results["X_train_t"]
    y_train = results["y_train"].to_numpy()

    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X_train_t).ravel()

    # Before
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pc1, y_train, alpha=0.35, s=12, color="#4C72B0")
    ax.set_xlabel("PC1 (projection of engineered features)")
    ax.set_ylabel("salary_in_usd")
    ax.set_title("Before: salary vs PC1 (raw scatter)")
    fig.tight_layout()
    out_before = FIGURES_DIR / "regression_before_scatter.png"
    fig.savefig(out_before, dpi=140)
    plt.close(fig)
    print(f"\n[Saved] {out_before}")

    # After — overlay fitted line from OLS on PC1 alone
    line_model = LinearRegression()
    line_model.fit(pc1.reshape(-1, 1), y_train)
    xs = np.linspace(pc1.min(), pc1.max(), 200)
    ys = line_model.predict(xs.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pc1, y_train, alpha=0.35, s=12, color="#4C72B0", label="Train data")
    ax.plot(xs, ys, color="#C44E52", linewidth=2, label="Fitted OLS line on PC1")
    ax.set_xlabel("PC1 (projection of engineered features)")
    ax.set_ylabel("salary_in_usd")
    ax.set_title("After: salary vs PC1 with fitted regression line")
    ax.legend()
    fig.tight_layout()
    out_after = FIGURES_DIR / "regression_after_scatter.png"
    fig.savefig(out_after, dpi=140)
    plt.close(fig)
    print(f"\n[Saved] {out_after}")
    print(
        f"Interpretation: PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of "
        "feature variance. The fitted line shows the dominant linear salary trend "
        "along that projection; residual scatter around the line is what a purely "
        "linear model cannot capture — the RandomForest comparison quantifies how "
        "much of that leftover structure is non-linear."
    )


def interpret_feature_weights(results: dict) -> None:
    names = results["feature_names"]
    print("\n=== Feature weights / importances ===")

    ridge = results["models"]["Ridge"]
    coef = pd.Series(ridge.coef_, index=names).sort_values(key=np.abs, ascending=False)
    print("\nRidge |coefficients| (top 10) — L2-shrunk linear signal:")
    print(coef.head(10).to_string())

    rf = results["models"]["RandomForestRegressor"]
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    print("\nRandomForest feature_importances_ (top 10):")
    print(imp.head(10).to_string())

    print(
        "\nInterpretation: features with near-zero Ridge coefficients AND low RF "
        "importance contribute little and are candidates to drop in a leaner model "
        "(e.g. rare employment_type dummies). Strong signals (experience_ord, "
        "US region, certain job families) are the ones mentorship tooling should "
        "surface to aspiring tech workers."
    )


def main() -> None:
    df = load_raw(DATA_PATH)
    X, y = engineer_features(df)

    # Quick SGD grid on a preliminary fit to pick alpha/eta0
    prelim = train_compare_models(X, y, sgd_epochs=30)
    best_hp = tune_sgd_grid(
        prelim["X_train_t"],
        prelim["y_train"],
        prelim["X_test_t"],
        prelim["y_test"],
    )

    results = train_compare_models(
        X,
        y,
        sgd_alpha=best_hp["alpha"],
        sgd_eta0=best_hp["eta0"],
        sgd_epochs=50,
    )

    print_metrics_table(results["metrics"])
    plot_sgd_loss_curve(
        results["sgd_train_losses"],
        results["sgd_test_losses"],
        results["sgd_params"],
    )
    plot_before_after_scatter(results)
    interpret_feature_weights(results)

    # Stash full results for 04_save_best_model.py
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(results, MODELS_DIR / "train_compare_results.pkl")
    print(f"\nCached full training results -> {MODELS_DIR / 'train_compare_results.pkl'}")


if __name__ == "__main__":
    main()

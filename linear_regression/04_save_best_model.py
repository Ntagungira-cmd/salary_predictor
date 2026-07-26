"""
04_save_best_model.py — Pick the best of the four models by test R^2 / RMSE,
save best_model.pkl + preprocessor.pkl, and print a short justification.
"""

from __future__ import annotations

import joblib

from pipeline import (
    DATA_PATH,
    MODELS_DIR,
    engineer_features,
    load_raw,
    select_and_save_best,
    train_compare_models,
)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache = MODELS_DIR / "train_compare_results.pkl"

    if cache.exists():
        print(f"Loading cached results from {cache}")
        results = joblib.load(cache)
    else:
        print("No cache found; re-running training...")
        df = load_raw(DATA_PATH)
        X, y = engineer_features(df)
        results = train_compare_models(X, y)

    summary = select_and_save_best(results, models_dir=MODELS_DIR)
    best = summary["best_name"]
    tm = summary["test_metrics"]

    print("\n=== Best model selection ===")
    print("Test metrics for all models:")
    for name, m in results["metrics"].items():
        marker = "  <-- WINNER" if name == best else ""
        print(
            f"  {name:22s}  R^2={m['test']['r2']:.4f}  "
            f"RMSE={m['test']['rmse']:,.1f}  MAE={m['test']['mae']:,.1f}{marker}"
        )

    print(
        f"\nJustification: {best} won on held-out test R^2 "
        f"({tm['r2']:.4f}) with test RMSE={tm['rmse']:,.1f}. "
        "Among SGD / OLS / DecisionTree / RandomForest, the non-linear tree models "
        "typically capture interactions (e.g. experience × region × role family) that "
        "a purely linear assumption leaves on the table — so if RF leads on test R^2 "
        "without a huge train/test gap, it is the right production choice."
        if best == "RandomForestRegressor"
        else f"\nJustification: {best} won on held-out test R^2 "
        f"({tm['r2']:.4f}) with test RMSE={tm['rmse']:,.1f}. "
        "It generalized better than the alternatives on this feature set; we deploy "
        "it as best_model.pkl for the FastAPI /predict endpoint."
    )

    print(f"\nSaved model         -> {summary['model_path']}")
    print(f"Saved preprocessor  -> {summary['preprocessor_path']}")


if __name__ == "__main__":
    main()

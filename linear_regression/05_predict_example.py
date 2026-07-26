"""
05_predict_example.py — Standalone predict_salary() used by Part 2's API.

Loads preprocessor.pkl + best_model.pkl and returns a USD salary prediction
from the six official inputs.
"""

from __future__ import annotations

import joblib

from pipeline import MODELS_DIR, predict_salary


def main() -> None:
    meta_path = MODELS_DIR / "model_meta.pkl"
    if meta_path.exists():
        meta = joblib.load(meta_path)
        print(f"Using model: {meta['model_name']}")
        print(f"Test metrics at save time: {meta['test_metrics']}")
    else:
        print("Warning: model_meta.pkl not found; proceeding with pickles only.")

    examples = [
        dict(
            experience_level="EN",
            employment_type="FT",
            job_title="Data Analyst",
            remote_ratio=100,
            company_size="M",
            company_location="US",
        ),
        dict(
            experience_level="SE",
            employment_type="FT",
            job_title="Machine Learning Engineer",
            remote_ratio=0,
            company_size="L",
            company_location="US",
        ),
        dict(
            experience_level="MI",
            employment_type="FT",
            job_title="Data Scientist",
            remote_ratio=50,
            company_size="M",
            company_location="GB",
        ),
        dict(
            experience_level="EX",
            employment_type="FT",
            job_title="Director of Data Science",
            remote_ratio=0,
            company_size="L",
            company_location="US",
        ),
    ]

    print("\nExample predictions:")
    for ex in examples:
        pred = predict_salary(**ex)
        print(
            f"  {ex['experience_level']} | {ex['job_title']:30s} | "
            f"{ex['company_location']} | remote={ex['remote_ratio']:>3} "
            f"-> ${pred:,.0f}"
        )


if __name__ == "__main__":
    main()

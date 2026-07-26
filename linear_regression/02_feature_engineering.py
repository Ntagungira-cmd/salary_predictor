"""
02_feature_engineering.py — CLI wrapper around pipeline.engineer_features /
build_preprocessor.

Drops leakage columns, collapses job titles into role families, ordinal-encodes
experience, one-hots nominal categoricals, buckets company_location into
regions, fits StandardScaler on train only, and saves preprocessor.pkl.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline import (
    DATA_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    engineer_features,
    get_feature_names,
    load_raw,
    map_job_title_to_family,
)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw(DATA_PATH)
    print(f"Raw shape: {df.shape}")

    # --- Drops explained in pipeline.engineer_features docstring ---
    # salary / salary_currency leak the target; employee_residence is redundant
    # with company_location and is not one of the six official predictors.
    X, y = engineer_features(df)
    print(f"Engineered feature columns: {list(X.columns)}")
    print(f"Target: salary_in_usd  (n={len(y)})")

    print("\nJob-family distribution:")
    print(X["job_family"].value_counts().to_string())

    print("\nCompany-region distribution:")
    print(X["company_region"].value_counts().to_string())

    print(
        "\nEncoding notes:\n"
        "  - experience_ord: EN=0 < MI=1 < SE=2 < EX=3 (ordinal beats one-hot\n"
        "    because seniority is ordered; avoids 3 extra sparse columns).\n"
        "  - employment_type, company_size, job_family, company_region: one-hot\n"
        "    (nominal, no natural order).\n"
        "  - company_location was bucketed US/Europe/Asia/Other instead of 72\n"
        "    one-hot columns — rare countries would be sparse and overfit;\n"
        "    regional buckets keep a stable US premium signal.\n"
        "  - StandardScaler fitted on TRAIN only to avoid leakage.\n"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    names = get_feature_names(preprocessor)

    print(f"Transformed train shape: {X_train_t.shape}")
    print(f"Feature names ({len(names)}):")
    for n in names:
        print(f"  - {n}")

    out = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor, out)
    print(f"\nSaved fitted preprocessor -> {out}")

    # Quick sanity: family mapping examples
    examples = [
        "Data Science Manager",
        "Machine Learning Engineer",
        "Research Scientist",
        "Data Engineer",
        "BI Data Analyst",
        "Data Strategist",
    ]
    print("\nSample job_title -> family mappings:")
    for t in examples:
        print(f"  {t!r:40s} -> {map_job_title_to_family(t)}")


if __name__ == "__main__":
    main()

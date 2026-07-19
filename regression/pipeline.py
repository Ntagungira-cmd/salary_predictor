"""
Shared feature-engineering and training pipeline.

Used by the numbered Part-1 scripts (02–05) and by api/main.py /retrain.
Numbered filenames like `02_feature_engineering.py` are not valid import
targets, so the real logic lives here as importable functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ds_salaries.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

REQUIRED_RAW_COLUMNS = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "salary",
    "salary_currency",
    "salary_in_usd",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size",
]

EXPERIENCE_ORDINAL = {"EN": 0, "MI": 1, "SE": 2, "EX": 3}

ROLE_FAMILIES = [
    "Leadership",
    "ML/Research family",
    "Data Scientist family",
    "Data Engineer family",
    "Analyst family",
    "Other",
]

# ISO 3166-1 alpha-2 -> region bucket.
# Tradeoff: one-hotting 72 company_location codes would create a sparse,
# high-dimensional design matrix with many rare countries (n=1–5) that
# overfit noise. Bucket into ~4 regions so the model can learn a stable
# US/Europe/Asia salary premium without drowning in sparsity.
EUROPE = {
    "GB", "ES", "DE", "FR", "PT", "GR", "NL", "IE", "AT", "PL", "IT", "CH",
    "BE", "DK", "SE", "NO", "FI", "CZ", "RO", "HU", "UA", "LT", "LV", "EE",
    "HR", "SI", "SK", "BG", "LU", "MT", "CY", "IS", "AL", "RS", "BA", "MD",
}
ASIA = {
    "IN", "SG", "JP", "CN", "HK", "KR", "TW", "TH", "MY", "ID", "PH", "VN",
    "PK", "BD", "AE", "IL", "TR", "IQ", "IR", "SA", "QA", "KW", "AM", "AZ",
    "UZ", "KZ",
}


def map_job_title_to_family(title: str) -> str:
    """
    Collapse 93 raw job titles into ~6 role families via keyword rules.

    Order matters (first match wins):
    - Leadership checked first so e.g. "Data Science Manager" / "Lead Data
      Engineer" are not swallowed by Scientist/Engineer keywords.
    - ML/Research before Scientist so "Machine Learning Scientist" and
      "Research Scientist" land in the ML/Research family.
    - Remaining Scientist / Engineer / Analyst keywords cover the big three
      IC families; everything else becomes Other.
    """
    t = str(title)
    leadership_kw = ("Manager", "Lead", "Head", "Director", "Principal", "Staff")
    if any(k in t for k in leadership_kw):
        return "Leadership"

    ml_kw = (
        "Machine Learning", "ML", "AI", "Research", "Deep Learning",
        "NLP", "Computer Vision",
    )
    if any(k in t for k in ml_kw):
        return "ML/Research family"

    if "Scientist" in t:
        return "Data Scientist family"

    if any(k in t for k in ("Engineer", "Architect", "Developer", "ETL")):
        return "Data Engineer family"

    if "Analyst" in t:
        return "Analyst family"

    return "Other"


def map_location_to_region(code: str) -> str:
    """Bucket ISO country code into US / Europe / Asia / Other."""
    c = str(code).upper().strip()
    if c == "US":
        return "US"
    if c in EUROPE:
        return "Europe"
    if c in ASIA:
        return "Asia"
    return "Other"


def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or DATA_PATH
    df = pd.read_csv(path)
    missing = set(REQUIRED_RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transform raw salary rows into model-ready features + target.

    Drops:
    - salary, salary_currency: leak / duplicate the target salary_in_usd
    - employee_residence: highly correlated with company_location and not
      one of the six official predictors the API / Flutter app accept
    - job_title (raw): replaced by job_family
    - company_location (raw): replaced by company_region
    - experience_level (raw string): replaced by ordinal int

    Ordinal experience_level (EN < MI < SE < EX) beats one-hot here because
    seniority has a natural order; one-hot would treat levels as unrelated
    categories and waste three extra columns.
    """
    out = df.copy()

    out = out.drop(columns=["salary", "salary_currency", "employee_residence"])

    out["job_family"] = out["job_title"].map(map_job_title_to_family)
    out["company_region"] = out["company_location"].map(map_location_to_region)
    out["experience_ord"] = out["experience_level"].map(EXPERIENCE_ORDINAL)

    if out["experience_ord"].isna().any():
        bad = out.loc[out["experience_ord"].isna(), "experience_level"].unique()
        raise ValueError(f"Unknown experience_level values: {bad}")

    y = out["salary_in_usd"].astype(float)
    feature_cols = [
        "work_year",
        "experience_ord",
        "remote_ratio",
        "employment_type",
        "company_size",
        "job_family",
        "company_region",
    ]
    X = out[feature_cols].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer:
    - StandardScaler on numeric (work_year, experience_ord, remote_ratio)
    - OneHotEncoder on nominal categoricals (employment_type, company_size,
      job_family, company_region)
    """
    numeric = ["work_year", "experience_ord", "remote_ratio"]
    categorical = ["employment_type", "company_size", "job_family", "company_region"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    return list(preprocessor.get_feature_names_out())


def _metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_sgd_with_loss_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float = 0.0001,
    eta0: float = 0.01,
    n_epochs: int = 50,
) -> Tuple[SGDRegressor, List[float], List[float]]:
    """
    Fit SGDRegressor epoch-by-epoch via partial_fit, recording train and
    test MSE loss each epoch for the loss-curve plot.
    """
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=alpha,
        learning_rate="constant",
        eta0=eta0,
        max_iter=1,
        tol=None,
        random_state=RANDOM_STATE,
        warm_start=True,
    )
    train_losses: List[float] = []
    test_losses: List[float] = []
    # partial_fit needs classes=None for regressors; just call repeatedly
    for _ in range(n_epochs):
        model.partial_fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_losses.append(float(mean_squared_error(y_train, train_pred)))
        test_losses.append(float(mean_squared_error(y_test, test_pred)))
    return model, train_losses, test_losses


def train_compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sgd_alpha: float = 0.0001,
    sgd_eta0: float = 0.01,
    sgd_epochs: int = 50,
) -> Dict[str, Any]:
    """
    Split 80/20, fit preprocessor on train only, train four models, return
    metrics, fitted objects, and SGD loss curves.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    results: Dict[str, Any] = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_t": X_train_t,
        "X_test_t": X_test_t,
        "models": {},
        "metrics": {},
    }

    # --- (a) SGDRegressor with loss curve ---
    sgd, train_losses, test_losses = train_sgd_with_loss_curve(
        X_train_t,
        y_train.to_numpy(),
        X_test_t,
        y_test.to_numpy(),
        alpha=sgd_alpha,
        eta0=sgd_eta0,
        n_epochs=sgd_epochs,
    )
    results["models"]["SGDRegressor"] = sgd
    results["metrics"]["SGDRegressor"] = {
        "train": _metrics(y_train, sgd.predict(X_train_t)),
        "test": _metrics(y_test, sgd.predict(X_test_t)),
    }
    results["sgd_train_losses"] = train_losses
    results["sgd_test_losses"] = test_losses
    results["sgd_params"] = {"alpha": sgd_alpha, "eta0": sgd_eta0, "n_epochs": sgd_epochs}

    # --- (b) OLS LinearRegression ---
    ols = LinearRegression()
    ols.fit(X_train_t, y_train)
    results["models"]["LinearRegression"] = ols
    results["metrics"]["LinearRegression"] = {
        "train": _metrics(y_train, ols.predict(X_train_t)),
        "test": _metrics(y_test, ols.predict(X_test_t)),
    }

    # --- (c) Ridge ---
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_train_t, y_train)
    results["models"]["Ridge"] = ridge
    results["metrics"]["Ridge"] = {
        "train": _metrics(y_train, ridge.predict(X_train_t)),
        "test": _metrics(y_test, ridge.predict(X_test_t)),
    }

    # --- (d) RandomForestRegressor ---
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_t, y_train)
    results["models"]["RandomForestRegressor"] = rf
    results["metrics"]["RandomForestRegressor"] = {
        "train": _metrics(y_train, rf.predict(X_train_t)),
        "test": _metrics(y_test, rf.predict(X_test_t)),
    }

    return results


def select_best_model(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Pick the model with the highest test R^2 (tie-break: lowest test RMSE)."""
    best_name = None
    best_r2 = -np.inf
    best_rmse = np.inf
    for name, m in metrics.items():
        r2 = m["test"]["r2"]
        rmse = m["test"]["rmse"]
        if (r2 > best_r2) or (np.isclose(r2, best_r2) and rmse < best_rmse):
            best_name = name
            best_r2 = r2
            best_rmse = rmse
    return best_name  # type: ignore[return-value]


def select_and_save_best(
    results: Dict[str, Any],
    *,
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Persist preprocessor.pkl + best_model.pkl (and a small metadata dict).
    Returns summary including best model name and test metrics.
    """
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    best_name = select_best_model(results["metrics"])
    best_model = results["models"][best_name]
    preprocessor = results["preprocessor"]

    model_path = models_dir / "best_model.pkl"
    prep_path = models_dir / "preprocessor.pkl"
    meta_path = models_dir / "model_meta.pkl"

    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor, prep_path)
    meta = {
        "model_name": best_name,
        "test_metrics": results["metrics"][best_name]["test"],
        "feature_names": results["feature_names"],
        "all_metrics": results["metrics"],
    }
    joblib.dump(meta, meta_path)

    return {
        "best_name": best_name,
        "test_metrics": results["metrics"][best_name]["test"],
        "model_path": str(model_path),
        "preprocessor_path": str(prep_path),
        "meta": meta,
    }


def prepare_single_row(
    experience_level: str,
    employment_type: str,
    job_title: str,
    remote_ratio: int,
    company_size: str,
    company_location: str,
    work_year: int = 2023,
) -> pd.DataFrame:
    """Build a one-row engineered feature frame for prediction."""
    raw = pd.DataFrame(
        [
            {
                "work_year": work_year,
                "experience_level": experience_level,
                "employment_type": employment_type,
                "job_title": job_title,
                "salary": 0,
                "salary_currency": "USD",
                "salary_in_usd": 0.0,
                "employee_residence": company_location,
                "remote_ratio": remote_ratio,
                "company_location": company_location,
                "company_size": company_size,
            }
        ]
    )
    X, _ = engineer_features(raw)
    return X


def predict_salary(
    experience_level: str,
    employment_type: str,
    job_title: str,
    remote_ratio: int,
    company_size: str,
    company_location: str,
    *,
    models_dir: Optional[Path] = None,
) -> float:
    """Load preprocessor + best model and return a salary prediction in USD."""
    models_dir = models_dir or MODELS_DIR
    preprocessor = joblib.load(models_dir / "preprocessor.pkl")
    model = joblib.load(models_dir / "best_model.pkl")
    X = prepare_single_row(
        experience_level,
        employment_type,
        job_title,
        remote_ratio,
        company_size,
        company_location,
    )
    X_t = preprocessor.transform(X)
    pred = float(model.predict(X_t)[0])
    return pred


def run_full_training(
    data_path: Optional[Path] = None,
    *,
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """End-to-end: load -> engineer -> train/compare -> save best."""
    df = load_raw(data_path)
    X, y = engineer_features(df)
    results = train_compare_models(X, y)
    summary = select_and_save_best(results, models_dir=models_dir)
    summary["results"] = results
    return summary

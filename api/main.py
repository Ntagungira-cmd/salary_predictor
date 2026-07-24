"""
FastAPI backend for the tech salary predictor.

Wraps regression/pipeline.predict_salary and exposes /predict + /retrain.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Make regression/ importable so we reuse the same pipeline as Part 1.
REPO_ROOT = Path(__file__).resolve().parent.parent
REGRESSION_DIR = REPO_ROOT / "regression"
sys.path.insert(0, str(REGRESSION_DIR))

from pipeline import (  # noqa: E402
    DATA_PATH,
    MODELS_DIR,
    REQUIRED_RAW_COLUMNS,
    ROLE_FAMILIES,
    map_job_title_to_family,
    predict_salary,
    run_full_training,
)

app = FastAPI(
    title="Tech Salary Predictor API",
    description=(
        "Predicts salary_in_usd from experience, employment type, job title, "
        "remote ratio, company size, and company location — closing the opportunity "
        "gap for young people entering tech by making career-outcome signals visible."
    ),
    version="1.0.0",
)

# CORS: for a public demo API with no auth and no sensitive user data, wildcard
# origins are acceptable since the only risk is who can *call* a free public
# prediction endpoint, not data leakage. If this were handling anything sensitive
# (auth tokens, PII), allow_origins should be locked to the specific Flutter
# web/app origin(s) and allow_credentials should stay False since no cookies/auth
# are used.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)


class SalaryPredictionRequest(BaseModel):
    """
    Request body for POST /predict.

    Bounds are tied to the training data's observed ranges in ds_salaries.csv:
    - experience_level only takes EN/MI/SE/EX in the dataset
    - employment_type only takes FT/PT/CT/FL
    - company_size only takes S/M/L
    - remote_ratio is observed as {0, 50, 100} but the API accepts the full
      0–100 continuum so callers can express partial-remote arrangements
      that the discrete training labels approximate
    - company_location is an ISO 3166-1 alpha-2 code (length exactly 2);
      unknown codes are bucketed to region "Other" by the preprocessor
    - job_title is free text mapped through the same keyword family function
      used in training; unknown titles fall into "Other" rather than 422,
      matching how the model was trained on a long tail of rare titles
    """

    experience_level: Literal["EN", "MI", "SE", "EX"] = Field(
        ...,
        description="Seniority band observed in training: EN < MI < SE < EX.",
    )
    employment_type: Literal["FT", "PT", "CT", "FL"] = Field(
        ...,
        description="Employment type: Full-time / Part-time / Contract / Freelance.",
    )
    job_title: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description=(
            "Raw or family job title. Mapped to one of "
            f"{ROLE_FAMILIES}; unknowns become 'Other'."
        ),
    )
    remote_ratio: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Percent of work done remotely. Training data only has 0/50/100, "
            "but 0–100 is accepted so the API covers the full remote spectrum."
        ),
    )
    company_size: Literal["S", "M", "L"] = Field(
        ...,
        description="Company size band: Small / Medium / Large (as in training data).",
    )
    company_location: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description=(
            "ISO 3166-1 alpha-2 country code of the employer. Bucketed into "
            "US/Europe/Asia/Other by the preprocessor (72 raw codes would be sparse)."
        ),
    )

    @field_validator("company_location")
    @classmethod
    def upper_iso(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) != 2 or not v.isalpha():
            raise ValueError("company_location must be a 2-letter ISO country code")
        return v

    @field_validator("job_title")
    @classmethod
    def strip_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("job_title must not be empty")
        return v


class SalaryPredictionResponse(BaseModel):
    predicted_salary_usd: float
    model_used: str
    job_family_mapped: str


class RetrainResponse(BaseModel):
    status: str
    rows_appended: int
    total_rows: int
    model_used: str
    test_r2: float
    test_rmse: float
    test_mae: float
    previous_test_r2: Optional[float] = None
    previous_test_rmse: Optional[float] = None
    backup_dir: Optional[str] = None


def _load_model_name() -> str:
    meta_path = MODELS_DIR / "model_meta.pkl"
    if meta_path.exists():
        meta = joblib.load(meta_path)
        return str(meta.get("model_name", "unknown"))
    return type(joblib.load(MODELS_DIR / "best_model.pkl")).__name__


def _previous_test_metrics() -> tuple[Optional[float], Optional[float]]:
    meta_path = MODELS_DIR / "model_meta.pkl"
    if not meta_path.exists():
        return None, None
    meta = joblib.load(meta_path)
    tm = meta.get("test_metrics", {})
    return tm.get("r2"), tm.get("rmse")


@app.get("/")
@app.get("/health")
def health() -> dict:
    model_ok = (MODELS_DIR / "best_model.pkl").exists()
    prep_ok = (MODELS_DIR / "preprocessor.pkl").exists()
    return {
        "status": "ok" if model_ok and prep_ok else "degraded",
        "model_loaded": model_ok,
        "preprocessor_loaded": prep_ok,
        "model_used": _load_model_name() if model_ok else None,
    }


@app.post("/predict", response_model=SalaryPredictionResponse)
def predict(req: SalaryPredictionRequest) -> SalaryPredictionResponse:
    """Validate inputs, run preprocessor + best model, return USD salary."""
    try:
        family = map_job_title_to_family(req.job_title)
        pred = predict_salary(
            experience_level=req.experience_level,
            employment_type=req.employment_type,
            job_title=req.job_title,
            remote_ratio=req.remote_ratio,
            company_size=req.company_size,
            company_location=req.company_location,
        )
        return SalaryPredictionResponse(
            predicted_salary_usd=round(pred, 2),
            model_used=_load_model_name(),
            job_family_mapped=family,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — return clean 400, not a 500 trace
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)) -> RetrainResponse:
    """
    Accept a CSV with the same 11-column schema as training data, APPEND it to
    the existing training set (grows over time rather than replacing), re-run
    the Part 1 pipeline, and overwrite best_model.pkl + preprocessor.pkl.

    Safety:
    - Column schema is validated before touching the live model.
    - Previous model files are copied to a timestamped backup folder first so
      a worse retrain can be rolled back manually.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv")

    try:
        raw_bytes = await file.read()
        from io import BytesIO

        upload_df = pd.read_csv(BytesIO(raw_bytes))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"Could not parse CSV: {exc}"
        ) from exc

    missing = set(REQUIRED_RAW_COLUMNS) - set(upload_df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {sorted(missing)}. "
            f"Expected: {REQUIRED_RAW_COLUMNS}",
        )
    if upload_df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows")

    prev_r2, prev_rmse = _previous_test_metrics()

    # Backup live artifacts before overwrite
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = MODELS_DIR / f"backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ("best_model.pkl", "preprocessor.pkl", "model_meta.pkl"):
        src = MODELS_DIR / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)

    # Append (documented choice): grow the training set over time.
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH)
        combined = pd.concat([existing, upload_df[REQUIRED_RAW_COLUMNS]], ignore_index=True)
    else:
        combined = upload_df[REQUIRED_RAW_COLUMNS].copy()
    combined.to_csv(DATA_PATH, index=False)

    try:
        summary = run_full_training(DATA_PATH, models_dir=MODELS_DIR)
    except Exception as exc:  # noqa: BLE001
        # Restore backup on failure
        for name in ("best_model.pkl", "preprocessor.pkl", "model_meta.pkl"):
            bak = backup_dir / name
            if bak.exists():
                shutil.copy2(bak, MODELS_DIR / name)
        raise HTTPException(
            status_code=400,
            detail=f"Retrain failed; previous model restored from backup. Error: {exc}",
        ) from exc

    tm = summary["test_metrics"]
    return RetrainResponse(
        status="ok",
        rows_appended=int(len(upload_df)),
        total_rows=int(len(combined)),
        model_used=summary["best_name"],
        test_r2=float(tm["r2"]),
        test_rmse=float(tm["rmse"]),
        test_mae=float(tm["mae"]),
        previous_test_r2=prev_r2,
        previous_test_rmse=prev_rmse,
        backup_dir=str(backup_dir),
    )

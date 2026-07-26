# Tech Salary Predictor

Mission: close the opportunity gap for young people entering tech by showing how experience, role, remote work, and company choices map to real salary outcomes.
Problem: predict `salary_in_usd` from six career features so mentors can target advice with data, not guesswork.
Dataset: Data Science Job Salaries (`ds_salaries.csv`).
Best model: RandomForestRegressor (compared against SGD, LinearRegression, DecisionTree).

## Public API (Swagger)

**Swagger UI:** https://tech-salary-predictor-api.onrender.com/docs  

## YouTube demo

**Video:** https://youtu.be/SLsl_l7sR4s 

## Project layout

```
summative/
  pyproject.toml          # uv package management
  uv.lock
  linear_regression/      # multivariate.ipynb + pipeline + models
  API/prediction.py       # FastAPI /predict + /retrain
  FlutterApp/             # single-page mobile client
```

## Setup with uv

```bash
# from this folder (summative/)
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
uv sync
```

Train / notebook:

```bash
cd linear_regression
uv run python 01_eda.py
uv run python 03_train_compare.py
uv run python 04_save_best_model.py
uv run jupyter notebook multivariate.ipynb
```

## Run API locally

```bash
cd API
uv run uvicorn prediction:app --reload --host 0.0.0.0 --port 8000
# http://127.0.0.1:8000/docs
```

## Run the Flutter mobile app

```bash
cd FlutterApp
# set apiBaseUrl in lib/main.dart to your Render URL first
flutter pub get
flutter devices
flutter run -d <android_emulator_or_iphone>   # mobile — not Chrome
```
## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Status |
| POST | `/predict` | Salary prediction (6 typed fields) |
| POST | `/retrain` | Upload CSV → append data → retrain → backup old model |

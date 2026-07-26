# Tech Salary Predictor

Mission: close the opportunity gap for young people entering tech by showing how experience, role, remote work, and company choices map to real salary outcomes.
Problem: predict `salary_in_usd` from six career features so mentors can target advice with data, not guesswork.
Dataset: Data Science Job Salaries (`ds_salaries.csv`) — not a house-price use case.
Best model: RandomForestRegressor (compared against SGD, LinearRegression, DecisionTree).

## Public API (Swagger)

**Swagger UI:** https://YOUR-RENDER-SERVICE.onrender.com/docs  

Replace the placeholder after you deploy on Render (see below). Graders should use this public URL, not localhost.

## YouTube demo (≤ 7 minutes)

**Video:** https://www.youtube.com/watch?v=YOUR_VIDEO_ID  

Script for recording: [`VIDEO_DEMO_SCRIPT.md`](VIDEO_DEMO_SCRIPT.md)

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

## Deploy API on Render

1. Push this repo to GitHub.
2. Create a **Web Service** on [Render](https://render.com).
3. Settings:
   - **Root Directory:** `API` (if repo root is `summative`) — or `summative/API` if the GitHub repo is `linear_regression_model` with a `summative/` folder.
   - **Build Command:** `pip install -r requirements.txt && pip install scikit-learn pandas numpy joblib`
   - **Start Command:** `uvicorn prediction:app --host 0.0.0.0 --port $PORT`
4. Ensure `../linear_regression/models/best_model.pkl` and `preprocessor.pkl` are in the repo (they are). The API resolves models relative to the repo parent.
5. After deploy, copy `https://<service>.onrender.com/docs` into this README and into `FlutterApp/lib/main.dart` (`apiBaseUrl`).

A [`render.yaml`](render.yaml) Blueprint is included for one-click setup from the monorepo root.

## Run the Flutter mobile app

```bash
cd FlutterApp
# set apiBaseUrl in lib/main.dart to your Render URL first
flutter pub get
flutter devices
flutter run -d <android_emulator_or_iphone>   # mobile — not Chrome
```

On a physical phone, the device must reach the public Render URL (not `localhost`).

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Status |
| POST | `/predict` | Salary prediction (6 typed fields) |
| POST | `/retrain` | Upload CSV → append data → retrain → backup old model |

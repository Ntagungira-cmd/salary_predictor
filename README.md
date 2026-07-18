# Tech Salary Predictor

**Mission:** close the opportunity gap for young people entering tech careers by
giving aspiring innovators clear, data-driven visibility into how their choices
translate into real-world salary outcomes, so mentorship can be targeted
effectively.

This monorepo has three parts that share one trained model:

| Folder | Role |
|--------|------|
| [`regression/`](regression/) | EDA, feature engineering, 4-model comparison, export `best_model.pkl` |
| [`api/`](api/) | FastAPI service wrapping the model (`/predict`, `/retrain`) |
| [`flutter_app/`](flutter_app/) | Material UI that POSTs to the API |

Dataset: [Data Science Job Salaries](ds_salaries.csv) (`salary_in_usd` target).

```
ds_salaries.csv ──► regression/pipeline.py ──► models/*.pkl ──► api/main.py
                                                              ▲
                                                     flutter_app (POST /predict)
```

## Quick start

### 1. Python env + Part 1 (train)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r regression/requirements.txt

cd regression
python 01_eda.py
python 02_feature_engineering.py
python 03_train_compare.py
python 04_save_best_model.py
python 05_predict_example.py
```

Artifacts land in `regression/figures/` and `regression/models/`.

### 2. Part 2 (API)

```bash
pip install -r api/requirements.txt
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger: http://127.0.0.1:8000/docs — see [`api/README.md`](api/README.md) for Render deploy.

### 3. Part 3 (Flutter)

```bash
cd flutter_app
flutter pub get
flutter run -d chrome   # or any device
```

Set `apiBaseUrl` at the top of [`flutter_app/lib/main.dart`](flutter_app/lib/main.dart)
to your local or Render URL.

## Model inputs

`experience_level`, `employment_type`, `job_title` (→ role family),
`remote_ratio`, `company_size`, `company_location` → predicted `salary_in_usd`.

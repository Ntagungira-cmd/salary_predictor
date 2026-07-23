# Tech Salary Predictor API

FastAPI service that loads `regression/models/best_model.pkl` +
`preprocessor.pkl` and exposes salary predictions for the Flutter client.

## Local development

From the repo root (with the project venv activated):

```bash
# install API deps (once)
.venv/bin/pip install -r api/requirements.txt

# run with auto-reload
cd api
../.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open the interactive Swagger UI at:

- http://127.0.0.1:8000/docs

Health check: `GET http://127.0.0.1:8000/health`

Example predict:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "experience_level": "SE",
    "employment_type": "FT",
    "job_title": "Data Scientist",
    "remote_ratio": 50,
    "company_size": "M",
    "company_location": "US"
  }'
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` or `/health` | Status + which model is loaded |
| POST | `/predict` | Score one salary from the 6 features |
| POST | `/retrain` | Upload a CSV (same schema as training), **append** to the training set, re-run the Part 1 pipeline, overwrite model pickles (previous files backed up under `regression/models/backup_<timestamp>/`) |

## Deploy on Render

1. Create a new **Web Service** pointed at this repo.
2. **Root directory**: `api` (or leave blank and adjust paths).
3. **Build command**:
   ```
   pip install -r requirements.txt && pip install -r ../regression/requirements.txt
   ```
   (Also ensure `regression/models/best_model.pkl` and `preprocessor.pkl` are present in the deploy artifact, or run Part 1 in the build step.)
4. **Start command**:
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
5. After deploy, Swagger is at:
   ```
   https://<your-service>.onrender.com/docs
   ```
   Update `API_BASE_URL` in `flutter_app/lib/main.dart` to that origin.

## CORS note

`allow_origins=["*"]` is intentional for this public, no-auth, no-PII demo.
If the API ever handled auth tokens or personal data, lock origins to the
Flutter web/app host(s) and keep `allow_credentials=False`.

# Tech Salary Predictor API

Entry point: [`prediction.py`](prediction.py) (`app = FastAPI(...)`).

## Local

```bash
# from repo root (summative/)
uv sync
cd API
uv run uvicorn prediction:app --reload --host 0.0.0.0 --port 8000
```

Swagger: http://127.0.0.1:8000/docs

## Render

- **Build:** `pip install -r requirements.txt`
- **Start:** `uvicorn prediction:app --host 0.0.0.0 --port $PORT`
- **Public docs:** `https://tech-salary-predictor-api.onrender.com`

## CORS

`allow_origins=["*"]` with `allow_credentials=False`, `allow_methods=["GET","POST"]`.
Public demo with no auth/PII — wildcard origins are acceptable for who can *call* the free endpoint.
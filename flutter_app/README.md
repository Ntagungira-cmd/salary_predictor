# Flutter Tech Salary Predictor

Single-page Material app that collects the six model features and POSTs to the
FastAPI `/predict` endpoint.

## Run

```bash
# Start the API first (from repo root)
cd ../api && ../.venv/bin/uvicorn main:app --reload --port 8000

# Then in another terminal:
flutter pub get
flutter run -d chrome   # or -d macos / an emulator
```

Change the API target in one place:

```dart
// lib/main.dart
const String apiBaseUrl = "http://127.0.0.1:8000";
// const String apiBaseUrl = "https://your-service.onrender.com";
```

# Flutter Tech Salary Predictor

Single-page Material app with **six TextFields** (one per API feature) and a **Predict** button.

## Configure API URL

In [`lib/main.dart`](lib/main.dart):

```dart
const String apiBaseUrl = "https://tech-salary-predictor-api.onrender.com";
```

## Run on mobile (required for demo)

```bash
flutter pub get
flutter run -d emulator-5554   # Android emulator
# or
flutter run -d <your_iphone_id>
```

Do **not** use Chrome/web for the graded video demo.

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/main.dart';

void main() {
  testWidgets('Salary predictor loads form', (WidgetTester tester) async {
    await tester.pumpWidget(const SalaryPredictorApp());
    expect(find.text('Tech Salary Predictor'), findsOneWidget);
    expect(find.text('Predict'), findsOneWidget);
  });
}

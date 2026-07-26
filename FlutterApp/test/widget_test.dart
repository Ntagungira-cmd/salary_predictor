import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_app/main.dart';

void main() {
  testWidgets('Salary predictor shows six fields and Predict', (tester) async {
    await tester.pumpWidget(const SalaryPredictorApp());
    expect(find.text('Tech Salary Predictor'), findsOneWidget);
    expect(find.text('Predict'), findsOneWidget);
    expect(find.byType(TextField), findsNWidgets(6));
  });
}

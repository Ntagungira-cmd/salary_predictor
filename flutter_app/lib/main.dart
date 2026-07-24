import 'package:flutter/material.dart';

void main() {
  runApp(const SalaryPredictorApp());
}

class SalaryPredictorApp extends StatelessWidget {
  const SalaryPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tech Salary Predictor',
      home: Scaffold(
        appBar: AppBar(title: const Text('Tech Salary Predictor')),
        body: const Center(
          child: Text('Salary predictor UI coming soon.'),
        ),
      ),
    );
  }
}

import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

// One-line swap after Render deploy, e.g.:
// const String apiBaseUrl = "https://your-service.onrender.com";
const String apiBaseUrl = "https://YOUR-RENDER-SERVICE.onrender.com";

void main() {
  runApp(const SalaryPredictorApp());
}

class SalaryPredictorApp extends StatelessWidget {
  const SalaryPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tech Salary Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1B4F72)),
        useMaterial3: true,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 14),
        ),
      ),
      home: const SalaryPredictorPage(),
    );
  }
}

class SalaryPredictorPage extends StatefulWidget {
  const SalaryPredictorPage({super.key});

  @override
  State<SalaryPredictorPage> createState() => _SalaryPredictorPageState();
}

class _SalaryPredictorPageState extends State<SalaryPredictorPage> {
  final _formKey = GlobalKey<FormState>();

  // Six TextFields — one per model feature (rubric requirement).
  final _experienceCtrl = TextEditingController();
  final _employmentCtrl = TextEditingController();
  final _jobTitleCtrl = TextEditingController();
  final _remoteCtrl = TextEditingController(text: '50');
  final _companySizeCtrl = TextEditingController();
  final _locationCtrl = TextEditingController();

  bool _loading = false;
  String? _error;
  double? _predictedSalary;
  String? _modelUsed;
  String? _jobFamilyMapped;

  @override
  void dispose() {
    _experienceCtrl.dispose();
    _employmentCtrl.dispose();
    _jobTitleCtrl.dispose();
    _remoteCtrl.dispose();
    _companySizeCtrl.dispose();
    _locationCtrl.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    final experience = _experienceCtrl.text.trim().toUpperCase();
    final employment = _employmentCtrl.text.trim().toUpperCase();
    final jobTitle = _jobTitleCtrl.text.trim();
    final remoteText = _remoteCtrl.text.trim();
    final companySize = _companySizeCtrl.text.trim().toUpperCase();
    final location = _locationCtrl.text.trim().toUpperCase();

    if (experience.isEmpty ||
        employment.isEmpty ||
        jobTitle.isEmpty ||
        remoteText.isEmpty ||
        companySize.isEmpty ||
        location.isEmpty) {
      setState(() {
        _error = 'Please fill in all six fields before predicting.';
        _predictedSalary = null;
      });
      return;
    }

    final remote = int.tryParse(remoteText);
    if (remote == null || remote < 0 || remote > 100) {
      setState(() {
        _error = 'remote_ratio must be an integer between 0 and 100.';
        _predictedSalary = null;
      });
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
      _predictedSalary = null;
      _modelUsed = null;
      _jobFamilyMapped = null;
    });

    final uri = Uri.parse('$apiBaseUrl/predict');
    final body = jsonEncode({
      'experience_level': experience,
      'employment_type': employment,
      'job_title': jobTitle,
      'remote_ratio': remote,
      'company_size': companySize,
      'company_location': location,
    });

    try {
      final response = await http
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: body,
          )
          .timeout(const Duration(seconds: 45));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _predictedSalary = (data['predicted_salary_usd'] as num).toDouble();
          _modelUsed = data['model_used']?.toString();
          _jobFamilyMapped = data['job_family_mapped']?.toString();
          _error = null;
        });
      } else {
        setState(() {
          _error = _parseApiError(response.body, response.statusCode);
          _predictedSalary = null;
        });
      }
    } catch (e) {
      setState(() {
        _error =
            'Could not reach the API at $apiBaseUrl.\n$e';
        _predictedSalary = null;
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String _parseApiError(String body, int statusCode) {
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map && decoded['detail'] != null) {
        final detail = decoded['detail'];
        if (detail is String) return detail;
        if (detail is List) {
          return detail.map((item) {
            if (item is Map) {
              final loc = (item['loc'] as List?)?.join('.') ?? 'field';
              final msg = item['msg'] ?? 'invalid';
              return '$loc: $msg';
            }
            return item.toString();
          }).join('\n');
        }
        return detail.toString();
      }
    } catch (_) {}
    return 'Request failed ($statusCode): $body';
  }

  String _formatUsd(double value) {
    final digits = value.round().toString();
    final buf = StringBuffer();
    for (var i = 0; i < digits.length; i++) {
      final fromEnd = digits.length - i;
      buf.write(digits[i]);
      if (fromEnd > 1 && fromEnd % 3 == 1) buf.write(',');
    }
    return '\$${buf.toString()}';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Tech Salary Predictor'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'See how experience, role, remote work, and company choices '
                  'translate into real tech-career salary outcomes — so mentorship '
                  'can be targeted effectively.',
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
                const SizedBox(height: 8),
                Text(
                  'Hints: EN/MI/SE/EX · FT/PT/CT/FL · S/M/L · location = 2-letter ISO (e.g. US)',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.black54,
                      ),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _experienceCtrl,
                  decoration: const InputDecoration(
                    labelText: 'experience_level',
                    hintText: 'EN, MI, SE, or EX',
                  ),
                  textCapitalization: TextCapitalization.characters,
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _employmentCtrl,
                  decoration: const InputDecoration(
                    labelText: 'employment_type',
                    hintText: 'FT, PT, CT, or FL',
                  ),
                  textCapitalization: TextCapitalization.characters,
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _jobTitleCtrl,
                  decoration: const InputDecoration(
                    labelText: 'job_title',
                    hintText: 'e.g. Data Scientist',
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _remoteCtrl,
                  decoration: const InputDecoration(
                    labelText: 'remote_ratio',
                    hintText: '0 to 100',
                  ),
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _companySizeCtrl,
                  decoration: const InputDecoration(
                    labelText: 'company_size',
                    hintText: 'S, M, or L',
                  ),
                  textCapitalization: TextCapitalization.characters,
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _locationCtrl,
                  decoration: const InputDecoration(
                    labelText: 'company_location',
                    hintText: 'e.g. US',
                  ),
                  textCapitalization: TextCapitalization.characters,
                  maxLength: 2,
                ),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: _loading ? null : _predict,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                  ),
                  child: _loading
                      ? const SizedBox(
                          height: 22,
                          width: 22,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('Predict'),
                ),
                const SizedBox(height: 16),
                Card(
                  elevation: 2,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: _buildResult(),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildResult() {
    if (_loading) {
      return const Row(
        children: [
          CircularProgressIndicator(),
          SizedBox(width: 16),
          Text('Asking the model…'),
        ],
      );
    }
    if (_error != null) {
      return Text(
        _error!,
        style: const TextStyle(color: Colors.red, fontSize: 15),
      );
    }
    if (_predictedSalary != null) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Predicted salary',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          Text(
            _formatUsd(_predictedSalary!),
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.primary,
                ),
          ),
          if (_jobFamilyMapped != null) ...[
            const SizedBox(height: 8),
            Text('Mapped role family: $_jobFamilyMapped'),
          ],
          if (_modelUsed != null) ...[
            const SizedBox(height: 4),
            Text('Model: $_modelUsed'),
          ],
        ],
      );
    }
    return const Text(
      'Enter all six values and tap Predict to see an estimated USD salary.',
    );
  }
}

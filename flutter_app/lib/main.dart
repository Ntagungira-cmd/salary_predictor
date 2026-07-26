import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

final String apiBaseUrl = kIsWeb
    ? "http://localhost:8000"
    : (defaultTargetPlatform == TargetPlatform.android
        ? "http://10.0.2.2:8000"
        : "http://localhost:8000");

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
  static const List<String> experienceLevels = ['EN', 'MI', 'SE', 'EX'];
  static const List<String> employmentTypes = ['FT', 'PT', 'CT', 'FL'];
  static const List<String> jobFamilies = [
    'Data Analyst',
    'Data Scientist',
    'Data Engineer',
    'Machine Learning Engineer',
    'Research Scientist',
    'Data Science Manager',
    'Director of Data Science',
    'ML Engineer',
    'Analytics Engineer',
    'AI Scientist',
  ];
  static const List<String> companySizes = ['S', 'M', 'L'];
  static const List<String> companyLocations = [
    'US',
    'GB',
    'CA',
    'ES',
    'IN',
    'DE',
    'FR',
    'AU',
    'BR',
    'NL',
  ];

  String? _experienceLevel;
  String? _employmentType;
  String? _jobTitle;
  String? _companySize;
  String? _companyLocation;
  double _remoteRatio = 50;

  bool _loading = false;
  String? _error;
  double? _predictedSalary;
  String? _modelUsed;
  String? _jobFamilyMapped;

  Future<void> _predict() async {
    if (_experienceLevel == null ||
        _employmentType == null ||
        _jobTitle == null ||
        _companySize == null ||
        _companyLocation == null) {
      setState(() {
        _error = 'Please fill in every field before predicting.';
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
      'experience_level': _experienceLevel,
      'employment_type': _employmentType,
      'job_title': _jobTitle,
      'remote_ratio': _remoteRatio.round(),
      'company_size': _companySize,
      'company_location': _companyLocation,
    });

    try {
      final response = await http
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: body,
          )
          .timeout(const Duration(seconds: 30));

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
            'Could not reach the API at $apiBaseUrl. Is the server running?\n$e';
        _predictedSalary = null;
      });
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  String _parseApiError(String body, int statusCode) {
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map && decoded['detail'] != null) {
        final detail = decoded['detail'];
        if (detail is String) {
          return detail;
        }
        if (detail is List) {
          final parts = detail.map((item) {
            if (item is Map) {
              final loc = (item['loc'] as List?)?.join('.') ?? 'field';
              final msg = item['msg'] ?? 'invalid';
              return '$loc: $msg';
            }
            return item.toString();
          }).join('\n');
          return parts;
        }
        return detail.toString();
      }
    } catch (_) {
      // fall through
    }
    return 'Request failed ($statusCode): $body';
  }

  String _formatUsd(double value) {
    final digits = value.round().toString();
    final buf = StringBuffer();
    for (var i = 0; i < digits.length; i++) {
      final fromEnd = digits.length - i;
      buf.write(digits[i]);
      if (fromEnd > 1 && fromEnd % 3 == 1) {
        buf.write(',');
      }
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
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Get clear, data-driven visibility into how experience, role, '
                'remote work, and company choices translate into real tech-career '
                'salary outcomes — so mentorship can be targeted effectively.',
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 16),
              _buildDropdown<String>(
                label: 'Experience level',
                value: _experienceLevel,
                items: experienceLevels,
                itemLabel: (v) {
                  const labels = {
                    'EN': 'EN — Entry',
                    'MI': 'MI — Mid',
                    'SE': 'SE — Senior',
                    'EX': 'EX — Executive',
                  };
                  return labels[v] ?? v;
                },
                onChanged: (v) => setState(() => _experienceLevel = v),
              ),
              const SizedBox(height: 16),
              _buildDropdown<String>(
                label: 'Employment type',
                value: _employmentType,
                items: employmentTypes,
                itemLabel: (v) {
                  const labels = {
                    'FT': 'FT — Full-time',
                    'PT': 'PT — Part-time',
                    'CT': 'CT — Contract',
                    'FL': 'FL — Freelance',
                  };
                  return labels[v] ?? v;
                },
                onChanged: (v) => setState(() => _employmentType = v),
              ),
              const SizedBox(height: 16),
              _buildDropdown<String>(
                label: 'Job title / family',
                value: _jobTitle,
                items: jobFamilies,
                itemLabel: (v) => v,
                onChanged: (v) => setState(() => _jobTitle = v),
              ),
              const SizedBox(height: 16),
              Text(
                'Remote ratio: ${_remoteRatio.round()}%',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              Slider(
                value: _remoteRatio,
                min: 0,
                max: 100,
                divisions: 20,
                label: '${_remoteRatio.round()}%',
                onChanged: (v) => setState(() => _remoteRatio = v),
              ),
              const SizedBox(height: 16),
              _buildDropdown<String>(
                label: 'Company size',
                value: _companySize,
                items: companySizes,
                itemLabel: (v) {
                  const labels = {
                    'S': 'S — Small',
                    'M': 'M — Medium',
                    'L': 'L — Large',
                  };
                  return labels[v] ?? v;
                },
                onChanged: (v) => setState(() => _companySize = v),
              ),
              const SizedBox(height: 16),
              _buildDropdown<String>(
                label: 'Company location',
                value: _companyLocation,
                items: companyLocations,
                itemLabel: (v) => v,
                onChanged: (v) => setState(() => _companyLocation = v),
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
      'Fill in the fields above and tap Predict to see an estimated USD salary.',
    );
  }

  Widget _buildDropdown<T>({
    required String label,
    required T? value,
    required List<T> items,
    required String Function(T) itemLabel,
    required ValueChanged<T?> onChanged,
  }) {
    return DropdownButtonFormField<T>(
      key: ValueKey<String>('$label-$value'),
      initialValue: value,
      decoration: InputDecoration(
        labelText: label,
        border: const OutlineInputBorder(),
      ),
      items: items
          .map(
            (item) => DropdownMenuItem<T>(
              value: item,
              child: Text(itemLabel(item)),
            ),
          )
          .toList(),
      onChanged: onChanged,
    );
  }
}

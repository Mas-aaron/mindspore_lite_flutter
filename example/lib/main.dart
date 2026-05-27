import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mindspore_lite_flutter/mindspore_lite_flutter.dart';

void main() => runApp(const ExampleApp());

class ExampleApp extends StatelessWidget {
  const ExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MindSpore Lite Flutter Example',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool _modelLoaded = false;
  bool _loading = false;
  File? _selectedImage;
  List<double>? _predictions;
  int? _inferenceMs;
  String? _error;

  // Example labels — replace with your own
  final List<String> _labels = [
    'Healthy',
    'Northern Leaf Blight',
    'Common Rust',
    'Gray Leaf Spot',
  ];

  Future<void> _loadModel() async {
    setState(() { _loading = true; _error = null; });
    try {
      final ok = await MindsporeLiteFlutter.initModel(
        'assets/models/maize_model.ms',
      );
      setState(() => _modelLoaded = ok);
      if (!ok) setState(() => _error = 'Failed to load model');
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _pickAndPredict() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() {
      _selectedImage = File(picked.path);
      _loading = true;
      _error = null;
      _predictions = null;
    });

    try {
      final result = await MindsporeLiteFlutter.predictImage(picked.path);
      final raw = (result['predictions'] as List).cast<double>();
      setState(() {
        _predictions = _softmax(raw);
        _inferenceMs = result['inferenceMs'] as int?;
      });
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  List<double> _softmax(List<double> logits) {
    if (logits.isEmpty) return [];
    final maxVal = logits.reduce((a, b) => a > b ? a : b);
    final exps = logits.map((v) => _exp(v - maxVal)).toList();
    final sum = exps.reduce((a, b) => a + b);
    return exps.map((v) => v / sum).toList();
  }

  double _exp(double x) {
    // Simple exp approximation safe for Dart web
    return x > 88 ? 1e38 : x < -88 ? 0 : _dartExp(x);
  }

  double _dartExp(double x) => (x == 0) ? 1.0 : (x > 0 ? _posExp(x) : 1.0 / _posExp(-x));
  double _posExp(double x) {
    double result = 1.0, term = 1.0;
    for (int i = 1; i <= 20; i++) {
      term *= x / i;
      result += term;
    }
    return result;
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('MindSpore Lite Flutter'),
        backgroundColor: scheme.inversePrimary,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Status card
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Plugin Status',
                      style: Theme.of(context)
                          .textTheme
                          .titleMedium
                          ?.copyWith(fontWeight: FontWeight.w700)),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Icon(
                        _modelLoaded
                            ? Icons.check_circle
                            : Icons.radio_button_unchecked,
                        color: _modelLoaded ? Colors.green : Colors.grey,
                      ),
                      const SizedBox(width: 8),
                      Text(_modelLoaded ? 'Model loaded' : 'Model not loaded'),
                    ],
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      onPressed: _loading ? null : _loadModel,
                      icon: _loading
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(
                                  strokeWidth: 2, color: Colors.white),
                            )
                          : const Icon(Icons.memory),
                      label: Text(_modelLoaded ? 'Reload Model' : 'Load Model'),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Image picker
          if (_modelLoaded) ...[
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    if (_selectedImage != null)
                      ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: Image.file(_selectedImage!,
                            height: 200, fit: BoxFit.cover),
                      ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed: _loading ? null : _pickAndPredict,
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Pick Image & Predict'),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
          ],

          // Results
          if (_predictions != null) ...[
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Predictions',
                            style: Theme.of(context)
                                .textTheme
                                .titleMedium
                                ?.copyWith(fontWeight: FontWeight.w700)),
                        if (_inferenceMs != null)
                          Text('${_inferenceMs}ms',
                              style: TextStyle(
                                  color: scheme.primary,
                                  fontWeight: FontWeight.w600)),
                      ],
                    ),
                    const SizedBox(height: 12),
                    for (int i = 0; i < _predictions!.length; i++) ...[
                      Row(
                        children: [
                          SizedBox(
                            width: 140,
                            child: Text(
                              i < _labels.length ? _labels[i] : 'Class $i',
                              style: const TextStyle(fontSize: 13),
                            ),
                          ),
                          Expanded(
                            child: LinearProgressIndicator(
                              value: _predictions![i],
                              backgroundColor: Colors.grey.shade200,
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            '${(_predictions![i] * 100).toStringAsFixed(1)}%',
                            style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                                color: scheme.primary),
                          ),
                        ],
                      ),
                      const SizedBox(height: 6),
                    ],
                  ],
                ),
              ),
            ),
          ],

          // Error
          if (_error != null)
            Card(
              color: Colors.red.shade50,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Text(_error!,
                    style: TextStyle(color: Colors.red.shade800)),
              ),
            ),

          const SizedBox(height: 24),
          // Usage instructions
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Quick Start',
                      style: Theme.of(context)
                          .textTheme
                          .titleSmall
                          ?.copyWith(fontWeight: FontWeight.w700)),
                  const SizedBox(height: 8),
                  const Text(
                    '1. Add your .ms model to assets/models/\n'
                    '2. Call MindsporeLiteFlutter.initModel(path)\n'
                    '3. Call MindsporeLiteFlutter.predictImage(imagePath)\n'
                    '4. Apply softmax to get class probabilities\n'
                    '5. Call disposeModel() when done',
                    style: TextStyle(fontSize: 12, height: 1.6),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

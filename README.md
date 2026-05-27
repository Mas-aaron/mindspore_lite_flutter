# mindspore_lite_flutter

[![pub.dev](https://img.shields.io/pub/v/mindspore_lite_flutter.svg)](https://pub.dev/packages/mindspore_lite_flutter)
[![Platform](https://img.shields.io/badge/platform-android-green.svg)](https://pub.dev/packages/mindspore_lite_flutter)

A Flutter plugin for **on-device AI inference** using [Huawei MindSpore Lite](https://www.mindspore.cn/lite/en) 1.7.0.

Built for the [AgriChain](https://github.com/Mas-aaron/agri-chain) project — crop disease detection and yield prediction for smallholder farmers in Africa.

## Features

- 🧠 Run `.ms` MindSpore Lite models on-device (no internet required)
- 📷 Image classification from camera or file path
- ⚡ ARM64 optimised — runs on any Android device with ARM64 architecture
- 🌽 Pre-trained models for maize and coffee disease detection included in AgriChain

## Platform Support

| Android | iOS |
|---------|-----|
| ✅ ARM64 | ❌ Not yet |

> iOS support is planned. MindSpore Lite has an iOS runtime but integration is not yet complete.

## Installation

```yaml
dependencies:
  mindspore_lite_flutter:
    git:
      url: https://github.com/Mas-aaron/agri-chain.git
      path: mindspore_lite_flutter
```

Or once published to pub.dev:

```yaml
dependencies:
  mindspore_lite_flutter: ^1.0.0
```

## Usage

### 1. Add your `.ms` model to Flutter assets

```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/models/maize_model.ms
    - assets/models/maize_labels.txt
```

### 2. Initialize the model

```dart
import 'package:mindspore_lite_flutter/mindspore_lite_flutter.dart';

final success = await MindsporeLiteFlutter.initModel(
  'assets/models/maize_model.ms',
);
```

### 3. Run inference with labels (recommended)

```dart
final results = await MindsporeLiteFlutter.predictImageWithLabels(
  imagePath: imageFile.path,
  labels: ['Healthy', 'Northern Leaf Blight', 'Common Rust', 'Gray Leaf Spot'],
);

// results is sorted by confidence:
// {'Healthy': 0.92, 'Common Rust': 0.05, ...}
final topLabel = results.keys.first;
final confidence = results.values.first;
print('$topLabel: ${(confidence * 100).toStringAsFixed(1)}%');
```

### 4. Or run raw inference and apply softmax yourself

```dart
final result = await MindsporeLiteFlutter.predictImage(imageFile.path);

// result contains:
// {
//   'predictions': [0.92, 0.05, 0.03],  // raw logits per class
//   'inferenceMs': 45,                   // inference time in ms
// }
```

### 5. Dispose when done

```dart
await MindsporeLiteFlutter.disposeModel();
```

## API Reference

| Method | Description |
|--------|-------------|
| `initModel(String modelPath)` | Load a `.ms` model from Flutter assets |
| `predictImage(String imagePath)` | Run inference, returns raw logits + inference time |
| `predictImageWithLabels({imagePath, labels})` | Run inference, applies softmax, returns `Map<String, double>` sorted by confidence |
| `disposeModel()` | Release native resources |

## Model Format

Models must be in **MindSpore Lite `.ms` format**. To convert from `.mindir`:

```bash
# Using MindSpore Lite converter
converter_lite \
  --fmk=MINDIR \
  --modelFile=model.mindir \
  --outputFile=model.ms \
  --targetDevice=ARM64
```

## Training Your Own Models

See the [AgriChain training notebook](https://github.com/Mas-aaron/agri-chain/blob/main/mindspore_disease_training2.ipynb) for a complete example using MobileNetV2 on Huawei ModelArts.

## Requirements

- Android API 21+ (Android 5.0)
- ARM64-v8a architecture
- MindSpore Lite 1.7.0 AAR (bundled)

## License

MIT — see [LICENSE](LICENSE)

## Credits

Built by the AgriChain team. Uses [Huawei MindSpore Lite](https://www.mindspore.cn/lite/en) runtime.

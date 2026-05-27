## 1.0.4

* Fix: Update repository URL to dedicated repo github.com/Mas-aaron/mindspore_lite_flutter
* This enables pub.dev repository verification for full 160/160 score

## 1.0.3

* Fix: Repository URL — pub.dev verification now resolves correctly
* New: `predictImageWithLabels()` — convenience API that applies softmax and maps results to label names automatically
* Improvement: Softmax now built into the Dart API (no need to implement it yourself)

## 1.0.2

* Fix: Shorten description to meet pub.dev 60-180 character requirement
* Fix: Remove unnecessary `dart:typed_data` import (lint warning)
* Fix: Simplify repository URL for pub.dev verification

## 1.0.1

* Fix: Replace placeholder LICENSE with proper MIT license
* Fix: Remove generic TODO from LICENSE file

## 1.0.0

* Initial public release
* Android ARM64 support via MindSpore Lite 1.7.0 JNI
* `initModel` — load `.ms` model from Flutter assets
* `predictImage` — run inference on image file path
* `disposeModel` — release native resources
* Supports maize and coffee disease detection models
* Built for AgriChain crop AI platform

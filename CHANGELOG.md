# Changelog

## 0.1.0 (unreleased)

- Initial release
- nanoflann v1.6.3 compiled to WASM via Emscripten
- Unified sklearn-style API: `create()`, `fit()`, `predict()`, `score()`, `save()`, `dispose()`
- k-nearest neighbor classification (majority vote) and regression (mean)
- L2 (Euclidean) and L1 (Manhattan) distance metrics
- `predictProba()` for class proportion estimates
- `kneighbors()` for raw neighbor index/distance queries
- NF01 binary serialization (training data stored, tree rebuilt on load)
- Accepts both typed matrices and number[][] with configurable coercion
- `getParams()`/`setParams()` for AutoML integration
- `defaultSearchSpace()` for hyperparameter search
- `FinalizationRegistry` safety net for leak detection
- BSD-2-Clause license (same as upstream nanoflann)

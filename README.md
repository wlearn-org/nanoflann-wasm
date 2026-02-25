# @wlearn/nanoflann

nanoflann v1.6.3 compiled to WebAssembly. k-nearest neighbor classification and regression via KD-trees in browsers and Node.js.

Based on [nanoflann v1.6.3](https://github.com/jlblancoc/nanoflann) (BSD-2-Clause). Zero dependencies. ESM.

## Install

```bash
npm install @wlearn/nanoflann
```

## Quick start

```js
import { KNNModel } from '@wlearn/nanoflann'

const model = await KNNModel.create({
  k: 5,
  metric: 'l2',
  task: 'classification'
})

// Train -- accepts number[][] or { data: Float64Array, rows, cols }
model.fit(
  [[1, 2], [3, 4], [5, 6], [7, 8]],
  [0, 0, 1, 1]
)

// Predict
const preds = model.predict([[2, 3], [6, 7]])  // Float64Array

// Score
const accuracy = model.score([[2, 3], [6, 7]], [0, 1])

// Save / load
const buf = model.save()  // Uint8Array (WLRN bundle)
const model2 = await KNNModel.load(buf)

// Clean up -- required, WASM memory is not garbage collected
model.dispose()
model2.dispose()
```

## API

### `KNNModel.create(params?)`

Async factory. Loads WASM module, returns a ready-to-use model.

Parameters:
- `k` -- number of neighbors (default: `5`)
- `metric` -- distance metric: `'l2'` or `'l1'` (default: `'l2'`)
- `leafMaxSize` -- KD-tree leaf size (default: `10`)
- `task` -- `'classification'` or `'regression'` (default: `'classification'`)
- `coerce` -- input coercion: `'auto'` | `'warn'` | `'error'` (default: `'auto'`)

### `model.fit(X, y)`

Build KD-tree from training data. Returns `this`.
- `X` -- `number[][]` or `{ data: Float64Array, rows, cols }`
- `y` -- `number[]` or `Float64Array`

### `model.predict(X)`

Returns `Float64Array` of predicted labels (classification: majority vote) or values (regression: mean of k neighbors).

### `model.predictProba(X)`

Returns `Float64Array` of shape `nrow * nclass` (row-major class proportions among k neighbors). Classification only. Rows sum to 1. Classes sorted ascending.

### `model.score(X, y)`

Returns accuracy (classification) or R-squared (regression).

### `model.kneighbors(X, k?)`

Raw neighbor search. Returns `{ indices: Int32Array, distances: Float64Array, k }`. Indices and distances are flat arrays of length `nrow * k`.

### `model.save()` / `KNNModel.load(buffer)`

Save to / load from `Uint8Array` (WLRN bundle). The bundle stores training data (X and y as raw float64/int32 little-endian arrays). The KD-tree is rebuilt on load.

### `model.dispose()`

Free WASM memory. Required. Idempotent.

### `model.getParams()` / `model.setParams(p)`

Get/set hyperparameters. Enables AutoML grid search and cloning.

### `KNNModel.defaultSearchSpace()`

Returns default hyperparameter search space for AutoML.

## Distance metrics

- `l2` -- Euclidean distance (sqrt of sum of squared differences)
- `l1` -- Manhattan distance (sum of absolute differences)

## Edge cases

- When `k > n_samples`, k is clamped to n_samples.
- Classification ties are broken by smallest class label.

## Resource management

WASM heap memory is not garbage collected. Call `.dispose()` on every model when done. A `FinalizationRegistry` safety net warns if you forget, but do not rely on it.

## Build from source

Requires [Emscripten](https://emscripten.org/) (emsdk) activated.

```bash
git clone --recurse-submodules https://github.com/wlearn-org/nanoflann-wasm
cd nanoflann-wasm
npm run build
npm test
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

## License

BSD-2-Clause (same as upstream nanoflann)

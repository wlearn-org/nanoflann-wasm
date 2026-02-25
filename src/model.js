import { getWasm, loadNanoflann } from './wasm.js'
import {
  normalizeX, normalizeY,
  encodeBundle, decodeBundle,
  register,
  DisposedError, NotFittedError
} from '@wlearn/core'

// FinalizationRegistry safety net -- warns if dispose() was never called
const leakRegistry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ptr, freeFn }) => {
    if (ptr[0]) {
      console.warn('@wlearn/nanoflann: Model was not disposed -- calling free() automatically. This is a bug in your code.')
      freeFn(ptr[0])
    }
  })
  : null

// --- Metric constants ---

export const Metric = { l2: 0, l1: 1 }

function resolveMetric(m) {
  if (typeof m === 'number') return m
  if (typeof m === 'string') {
    const lower = m.toLowerCase()
    if (lower in Metric) return Metric[lower]
  }
  return Metric.l2
}

function metricName(m) {
  return m === 1 ? 'l1' : 'l2'
}

function getLastError() {
  const wasm = getWasm()
  return wasm.ccall('wl_nf_get_last_error', 'string', [], [])
}

// --- Internal sentinel for load path ---
const LOAD_SENTINEL = Symbol('load')

// --- KNNModel ---

export class KNNModel {
  #handle = null
  #freed = false
  #ptrRef = null
  #params = {}
  #fitted = false
  #nSamples = 0
  #nFeatures = 0
  #nClasses = 0
  #classes = null  // sorted Int32Array of unique class labels

  constructor(handle, params, extra) {
    if (handle === LOAD_SENTINEL) {
      // Internal: created by load() / _fromBundle()
      this.#handle = params
      this.#params = extra.params || {}
      this.#nSamples = extra.nSamples || 0
      this.#nFeatures = extra.nFeatures || 0
      this.#nClasses = extra.nClasses || 0
      this.#classes = extra.classes || null
      this.#fitted = true
    } else {
      // Normal construction (from create())
      this.#handle = null
      this.#params = handle || {}
    }

    this.#freed = false
    if (this.#handle) {
      this.#registerLeak()
    }
  }

  static async create(params = {}) {
    await loadNanoflann()
    return new KNNModel(params)
  }

  // --- Estimator interface ---

  fit(X, y) {
    this.#ensureFitted(false)
    const wasm = getWasm()

    // Dispose previous model if refitting
    if (this.#handle) {
      wasm._wl_nf_free(this.#handle)
      this.#handle = null
      if (this.#ptrRef) this.#ptrRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)
    const yData = yNorm instanceof Float64Array ? yNorm : new Float64Array(yNorm)

    if (yData.length !== rows) {
      throw new Error(`y length (${yData.length}) does not match X rows (${rows})`)
    }

    const task = this.#taskEnum()
    const metric = resolveMetric(this.#params.metric)
    const leafMaxSize = this.#params.leafMaxSize ?? 10

    // Extract class info for classification
    if (task === 0) {
      const classSet = new Set()
      for (let i = 0; i < yData.length; i++) classSet.add(yData[i] | 0)
      this.#classes = new Int32Array([...classSet].sort((a, b) => a - b))
      this.#nClasses = this.#classes.length
    } else {
      this.#classes = null
      this.#nClasses = 0
    }

    // Ownership transfer: allocate X on WASM heap, pass ownership to C++
    const xBytes = xData.length * 8
    const xPtr = wasm._malloc(xBytes)
    wasm.HEAPF64.set(xData, xPtr / 8)

    // y is always copied by C++ (converted to int32 or float64)
    const yBytes = yData.length * 8
    const yPtr = wasm._malloc(yBytes)
    wasm.HEAPF64.set(yData, yPtr / 8)

    const modelPtr = wasm._wl_nf_build(
      xPtr, rows, cols,
      yPtr,
      task, metric, leafMaxSize,
      1 // take_ownership of X
    )

    // X ownership transferred; free y (C++ copied it)
    wasm._free(yPtr)

    if (!modelPtr) {
      // X was NOT freed if build failed before taking ownership
      // but wl_nf_build takes ownership even on error path after the copy
      // Actually, on error after X_owned is set, free_index frees it.
      // On error before X_owned is set (validation), we need to free xPtr.
      // The C code takes ownership of X_ptr when take_ownership=1,
      // so it will free it in the error path via free_index.
      throw new Error(`Training failed: ${getLastError()}`)
    }

    this.#handle = modelPtr
    this.#fitted = true
    this.#nSamples = rows
    this.#nFeatures = cols

    this.#registerLeak()
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const k = this.#params.k ?? 5
    const kEff = Math.min(k, this.#nSamples)

    // Allocate input
    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    // Allocate output (k_eff neighbors per query)
    const idxPtr = wasm._malloc(rows * kEff * 4)  // int32
    const distPtr = wasm._malloc(rows * kEff * 8)  // float64

    const ret = wasm._wl_nf_knn_search(
      this.#handle, xPtr, rows, cols, k, idxPtr, distPtr
    )

    wasm._free(xPtr)

    if (ret < 0) {
      wasm._free(idxPtr)
      wasm._free(distPtr)
      throw new Error(`Predict failed: ${getLastError()}`)
    }

    const actualK = ret
    const result = this.#vote(wasm, rows, actualK, idxPtr)

    wasm._free(idxPtr)
    wasm._free(distPtr)
    return result
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#taskEnum() !== 0) {
      throw new Error('predictProba is only available for classification')
    }

    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const k = this.#params.k ?? 5
    const kEff = Math.min(k, this.#nSamples)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const idxPtr = wasm._malloc(rows * kEff * 4)
    const distPtr = wasm._malloc(rows * kEff * 8)

    const ret = wasm._wl_nf_knn_search(
      this.#handle, xPtr, rows, cols, k, idxPtr, distPtr
    )

    wasm._free(xPtr)

    if (ret < 0) {
      wasm._free(idxPtr)
      wasm._free(distPtr)
      throw new Error(`predictProba failed: ${getLastError()}`)
    }

    const actualK = ret
    const nClasses = this.#nClasses
    const result = new Float64Array(rows * nClasses)

    // Build class-to-column index
    const classCol = new Map()
    for (let c = 0; c < this.#classes.length; c++) {
      classCol.set(this.#classes[c], c)
    }

    for (let i = 0; i < rows; i++) {
      const base = i * actualK
      for (let j = 0; j < actualK; j++) {
        const neighborIdx = wasm.HEAP32[(idxPtr / 4) + base + j]
        if (neighborIdx < 0) continue
        const label = wasm._wl_nf_get_label(this.#handle, neighborIdx) | 0
        const col = classCol.get(label)
        if (col !== undefined) {
          result[i * nClasses + col] += 1.0
        }
      }
      // Normalize row to probabilities
      let rowSum = 0
      for (let c = 0; c < nClasses; c++) rowSum += result[i * nClasses + c]
      if (rowSum > 0) {
        for (let c = 0; c < nClasses; c++) result[i * nClasses + c] /= rowSum
      }
    }

    wasm._free(idxPtr)
    wasm._free(distPtr)
    return result
  }

  kneighbors(X, k) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const kParam = k ?? this.#params.k ?? 5
    const kEff = Math.min(kParam, this.#nSamples)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const idxPtr = wasm._malloc(rows * kEff * 4)
    const distPtr = wasm._malloc(rows * kEff * 8)

    const ret = wasm._wl_nf_knn_search(
      this.#handle, xPtr, rows, cols, kParam, idxPtr, distPtr
    )

    wasm._free(xPtr)

    if (ret < 0) {
      wasm._free(idxPtr)
      wasm._free(distPtr)
      throw new Error(`kneighbors failed: ${getLastError()}`)
    }

    const actualK = ret
    const total = rows * actualK
    const indices = new Int32Array(total)
    const distances = new Float64Array(total)

    for (let i = 0; i < total; i++) {
      indices[i] = wasm.HEAP32[(idxPtr / 4) + i]
      distances[i] = wasm.HEAPF64[(distPtr / 8) + i]
    }

    wasm._free(idxPtr)
    wasm._free(distPtr)
    return { indices, distances, k: actualK }
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = normalizeY(y)

    if (this.#taskEnum() === 1) {
      // R-squared (regression)
      let ssRes = 0, ssTot = 0, yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    } else {
      // Accuracy (classification)
      let correct = 0
      for (let i = 0; i < preds.length; i++) {
        if (preds[i] === yArr[i]) correct++
      }
      return correct / preds.length
    }
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()
    const rawBytes = this.#saveRaw()
    const task = this.#params.task || 'classification'
    const typeId = task === 'regression'
      ? 'wlearn.nanoflann.regressor@1'
      : 'wlearn.nanoflann.classifier@1'

    const metadata = {
      nSamples: this.#nSamples,
      nFeatures: this.#nFeatures,
    }
    if (task !== 'regression') {
      metadata.nClasses = this.#nClasses
      metadata.classes = Array.from(this.#classes)
    }

    return encodeBundle(
      { typeId, params: this.getParams(), metadata },
      [{ id: 'model', data: rawBytes }]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return KNNModel._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    await loadNanoflann()
    const wasm = getWasm()

    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)

    const params = manifest.params || {}
    const metric = resolveMetric(params.metric)
    const leafMaxSize = params.leafMaxSize ?? 10

    // Copy blob to WASM heap
    const bufPtr = wasm._malloc(raw.length)
    wasm.HEAPU8.set(raw, bufPtr)
    const modelPtr = wasm._wl_nf_load(bufPtr, raw.length, metric, leafMaxSize)
    wasm._free(bufPtr)

    if (!modelPtr) {
      throw new Error(`load failed: ${getLastError()}`)
    }

    const metadata = manifest.metadata || {}
    const nSamples = metadata.nSamples || wasm._wl_nf_get_n_samples(modelPtr)
    const nFeatures = metadata.nFeatures || wasm._wl_nf_get_n_features(modelPtr)
    const nClasses = metadata.nClasses || 0
    const classes = metadata.classes
      ? new Int32Array(metadata.classes)
      : null

    return new KNNModel(LOAD_SENTINEL, modelPtr, {
      params, nSamples, nFeatures, nClasses, classes
    })
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#handle) {
      const wasm = getWasm()
      wasm._wl_nf_free(this.#handle)
    }

    if (this.#ptrRef) this.#ptrRef[0] = null
    if (leakRegistry) leakRegistry.unregister(this)

    this.#handle = null
    this.#fitted = false
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  static defaultSearchSpace() {
    return {
      k: { type: 'int_uniform', low: 1, high: 30 },
      metric: { type: 'categorical', values: ['l2', 'l1'] },
      leafMaxSize: { type: 'int_uniform', low: 5, high: 50 }
    }
  }

  // --- Inspection ---

  get nSamples() {
    return this.#nSamples
  }

  get nFeatures() {
    return this.#nFeatures
  }

  get nClasses() {
    return this.#nClasses
  }

  get classes() {
    return this.#classes ? new Int32Array(this.#classes) : null
  }

  get isFitted() {
    return this.#fitted && !this.#freed
  }

  get capabilities() {
    const isClassifier = this.#taskEnum() === 0
    return {
      classifier: isClassifier,
      regressor: !isClassifier,
      predictProba: isClassifier,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false
    }
  }

  // --- Private helpers ---

  #taskEnum() {
    return (this.#params.task || 'classification') === 'regression' ? 1 : 0
  }

  #normalizeX(X) {
    return normalizeX(X, 'auto')
  }

  #ensureFitted(requireFit = true) {
    if (this.#freed) throw new DisposedError('KNNModel has been disposed.')
    if (requireFit && !this.#fitted) throw new NotFittedError('KNNModel is not fitted. Call fit() first.')
  }

  #registerLeak() {
    this.#ptrRef = [this.#handle]
    if (leakRegistry) {
      leakRegistry.register(this, {
        ptr: this.#ptrRef,
        freeFn: (h) => getWasm()._wl_nf_free(h)
      }, this)
    }
  }

  #saveRaw() {
    const wasm = getWasm()

    const outBufPtr = wasm._malloc(4)
    const outLenPtr = wasm._malloc(4)

    const ret = wasm._wl_nf_save(this.#handle, outBufPtr, outLenPtr)

    if (ret !== 0) {
      wasm._free(outBufPtr)
      wasm._free(outLenPtr)
      throw new Error(`save failed: ${getLastError()}`)
    }

    const bufPtr = wasm.getValue(outBufPtr, 'i32')
    const bufLen = wasm.getValue(outLenPtr, 'i32')

    const result = new Uint8Array(bufLen)
    result.set(wasm.HEAPU8.subarray(bufPtr, bufPtr + bufLen))

    wasm._wl_nf_free_buffer(bufPtr)
    wasm._free(outBufPtr)
    wasm._free(outLenPtr)

    return result
  }

  /**
   * Majority vote (classification) or mean (regression).
   * Tie-breaking: smallest class label wins.
   */
  #vote(wasm, nQueries, kEff, idxPtr) {
    const task = this.#taskEnum()

    if (task === 1) {
      // Regression: mean of neighbor values
      const result = new Float64Array(nQueries)
      for (let i = 0; i < nQueries; i++) {
        let sum = 0, count = 0
        const base = i * kEff
        for (let j = 0; j < kEff; j++) {
          const neighborIdx = wasm.HEAP32[(idxPtr / 4) + base + j]
          if (neighborIdx < 0) continue
          sum += wasm._wl_nf_get_label(this.#handle, neighborIdx)
          count++
        }
        result[i] = count > 0 ? sum / count : 0
      }
      return result
    }

    // Classification: majority vote with smallest-label tie-breaking
    const nClasses = this.#nClasses
    const counts = new Int32Array(nClasses)
    const classCol = new Map()
    for (let c = 0; c < this.#classes.length; c++) {
      classCol.set(this.#classes[c], c)
    }

    const result = new Float64Array(nQueries)

    for (let i = 0; i < nQueries; i++) {
      counts.fill(0)
      const base = i * kEff
      for (let j = 0; j < kEff; j++) {
        const neighborIdx = wasm.HEAP32[(idxPtr / 4) + base + j]
        if (neighborIdx < 0) continue
        const label = wasm._wl_nf_get_label(this.#handle, neighborIdx) | 0
        const col = classCol.get(label)
        if (col !== undefined) counts[col]++
      }
      // Find max count; tie-break by smallest class label (classes are sorted)
      let bestCol = 0, bestCount = counts[0]
      for (let c = 1; c < nClasses; c++) {
        if (counts[c] > bestCount) {
          bestCount = counts[c]
          bestCol = c
        }
      }
      result[i] = this.#classes[bestCol]
    }

    return result
  }
}

// --- Register loaders with @wlearn/core ---

register('wlearn.nanoflann.classifier@1', (m, t, b) => KNNModel._fromBundle(m, t, b))
register('wlearn.nanoflann.regressor@1', (m, t, b) => KNNModel._fromBundle(m, t, b))

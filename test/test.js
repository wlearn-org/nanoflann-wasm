import { fileURLToPath } from 'url'
import { dirname } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

// --- Deterministic LCG PRNG ---
function makeLCG(seed = 42) {
  let s = seed | 0
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff
    return s / 0x7fffffff
  }
}

function makeClassificationData(rng, nSamples, nFeatures, nClasses = 2) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const label = i % nClasses
    const row = []
    for (let j = 0; j < nFeatures; j++) {
      row.push(label * 2 + (rng() - 0.5) * 0.5)
    }
    X.push(row)
    y.push(label)
  }
  return { X, y }
}

function makeRegressionData(rng, nSamples, nFeatures) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let target = 0
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 4 - 2
      row.push(v)
      target += v * (j + 1)
    }
    X.push(row)
    y.push(target + (rng() - 0.5) * 0.1)
  }
  return { X, y }
}

// ============================================================
// WASM loading
// ============================================================
console.log('\n=== WASM Loading ===')

const { loadNanoflann } = await import('../src/wasm.js')
const wasm = await loadNanoflann()

await test('WASM module loads', async () => {
  assert(wasm, 'wasm module is null')
  assert(typeof wasm.ccall === 'function', 'ccall not available')
})

await test('get_last_error returns string', async () => {
  const err = wasm.ccall('wl_nf_get_last_error', 'string', [], [])
  assert(typeof err === 'string', `expected string, got ${typeof err}`)
})

// ============================================================
// KNNModel basics
// ============================================================
console.log('\n=== KNNModel ===')

const { KNNModel, Metric } = await import('../src/model.js')

await test('create() returns model', async () => {
  const model = await KNNModel.create({ k: 3 })
  assert(model, 'model is null')
  assert(!model.isFitted, 'should not be fitted yet')
  model.dispose()
})

await test('Metric constants', async () => {
  assert(Metric.l2 === 0, 'l2 should be 0')
  assert(Metric.l1 === 1, 'l1 should be 1')
})

// ============================================================
// Classification
// ============================================================
console.log('\n=== Classification ===')

await test('Binary classification k=5', async () => {
  const rng = makeLCG(100)
  const { X, y } = makeClassificationData(rng, 80, 2)
  const model = await KNNModel.create({ k: 5, task: 'classification', metric: 'l2' })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  assert(model.nSamples === 80, `nSamples: ${model.nSamples}`)
  assert(model.nFeatures === 2, `nFeatures: ${model.nFeatures}`)
  assert(model.nClasses === 2, `nClasses: ${model.nClasses}`)

  const preds = model.predict(X)
  assert(preds.length === 80, `preds length: ${preds.length}`)

  // Training accuracy should be high for well-separated data
  const accuracy = model.score(X, y)
  assert(accuracy > 0.8, `accuracy too low: ${accuracy}`)

  model.dispose()
})

await test('Multiclass classification k=3', async () => {
  const rng = makeLCG(200)
  const { X, y } = makeClassificationData(rng, 90, 3, 3)
  const model = await KNNModel.create({ k: 3, task: 'classification' })
  model.fit(X, y)

  assert(model.nClasses === 3, `nClasses: ${model.nClasses}`)
  const classes = model.classes
  assert(classes.length === 3, 'should have 3 classes')
  assert(classes[0] === 0 && classes[1] === 1 && classes[2] === 2, 'classes should be [0,1,2]')

  const preds = model.predict(X)
  assert(preds.length === 90, `preds length: ${preds.length}`)

  const accuracy = model.score(X, y)
  assert(accuracy > 0.7, `accuracy too low: ${accuracy}`)

  model.dispose()
})

await test('predictProba returns valid probabilities', async () => {
  const rng = makeLCG(300)
  const { X, y } = makeClassificationData(rng, 60, 2)
  const model = await KNNModel.create({ k: 5, task: 'classification' })
  model.fit(X, y)

  const proba = model.predictProba(X)
  assert(proba.length === 60 * 2, `proba length: ${proba.length}, expected ${60 * 2}`)

  // Each row should sum to 1
  for (let i = 0; i < 60; i++) {
    const rowSum = proba[i * 2] + proba[i * 2 + 1]
    assertClose(rowSum, 1.0, 1e-10, `row ${i} sum: ${rowSum}`)
  }

  // Probabilities should be in [0, 1]
  for (let i = 0; i < proba.length; i++) {
    assert(proba[i] >= 0 && proba[i] <= 1, `proba[${i}] = ${proba[i]} out of range`)
  }

  model.dispose()
})

await test('predictProba fails for regression', async () => {
  const rng = makeLCG(301)
  const { X, y } = makeRegressionData(rng, 30, 2)
  const model = await KNNModel.create({ k: 3, task: 'regression' })
  model.fit(X, y)

  let threw = false
  try {
    model.predictProba(X)
  } catch (e) {
    threw = true
    assert(e.message.includes('classification'), `wrong error: ${e.message}`)
  }
  assert(threw, 'should throw for regression')
  model.dispose()
})

// ============================================================
// Regression
// ============================================================
console.log('\n=== Regression ===')

await test('Regression k=5', async () => {
  const rng = makeLCG(400)
  const { X, y } = makeRegressionData(rng, 80, 2)
  const model = await KNNModel.create({ k: 5, task: 'regression', metric: 'l2' })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  assert(model.nClasses === 0, 'regressor should have 0 classes')

  const preds = model.predict(X)
  assert(preds.length === 80, `preds length: ${preds.length}`)

  // R-squared on training data should be positive
  const r2 = model.score(X, y)
  assert(r2 > 0, `R2 too low: ${r2}`)

  model.dispose()
})

// ============================================================
// kneighbors
// ============================================================
console.log('\n=== kneighbors ===')

await test('kneighbors returns indices and distances', async () => {
  const rng = makeLCG(500)
  const { X, y } = makeClassificationData(rng, 50, 2)
  const model = await KNNModel.create({ k: 3, task: 'classification' })
  model.fit(X, y)

  const { indices, distances, k } = model.kneighbors(X)
  assert(k === 3, `k: ${k}`)
  assert(indices.length === 50 * 3, `indices length: ${indices.length}`)
  assert(distances.length === 50 * 3, `distances length: ${distances.length}`)

  // Distances should be non-negative and sorted per query
  for (let i = 0; i < 50; i++) {
    for (let j = 0; j < 3; j++) {
      const idx = indices[i * 3 + j]
      const dist = distances[i * 3 + j]
      assert(idx >= 0 && idx < 50, `invalid index: ${idx}`)
      assert(dist >= 0, `negative distance: ${dist}`)
    }
    // Sorted: dist[0] <= dist[1] <= dist[2]
    for (let j = 1; j < 3; j++) {
      assert(
        distances[i * 3 + j] >= distances[i * 3 + j - 1] - 1e-12,
        `distances not sorted at row ${i}`
      )
    }
  }

  model.dispose()
})

await test('kneighbors with custom k override', async () => {
  const rng = makeLCG(501)
  const { X, y } = makeClassificationData(rng, 40, 2)
  const model = await KNNModel.create({ k: 5, task: 'classification' })
  model.fit(X, y)

  const { k } = model.kneighbors(X, 10)
  assert(k === 10, `expected k=10, got ${k}`)

  model.dispose()
})

// ============================================================
// Edge cases
// ============================================================
console.log('\n=== Edge cases ===')

await test('k > n_samples clamps to n_samples', async () => {
  const X = [[0, 0], [1, 1], [2, 2]]
  const y = [0, 1, 0]
  const model = await KNNModel.create({ k: 100, task: 'classification' })
  model.fit(X, y)

  const { k } = model.kneighbors(X)
  assert(k === 3, `expected k clamped to 3, got ${k}`)

  const preds = model.predict(X)
  assert(preds.length === 3, `preds length: ${preds.length}`)

  model.dispose()
})

await test('Deterministic tie-breaking: smallest class label wins', async () => {
  // 4 points: 2 class-0, 2 class-1. k=4 -> 2 votes each -> class 0 wins
  const X = [[0, 0], [1, 0], [0, 1], [1, 1]]
  const y = [0, 1, 0, 1]
  const model = await KNNModel.create({ k: 4, task: 'classification' })
  model.fit(X, y)

  const query = [[0.5, 0.5]]  // equidistant from all
  const preds = model.predict(query)
  assert(preds[0] === 0, `expected class 0 (tie-break), got ${preds[0]}`)

  model.dispose()
})

await test('Refitting disposes previous model', async () => {
  const rng = makeLCG(600)
  const { X, y } = makeClassificationData(rng, 30, 2)
  const model = await KNNModel.create({ k: 3, task: 'classification' })
  model.fit(X, y)
  assert(model.nSamples === 30, 'first fit')

  const rng2 = makeLCG(601)
  const d2 = makeClassificationData(rng2, 50, 2)
  model.fit(d2.X, d2.y)
  assert(model.nSamples === 50, 'second fit')

  model.dispose()
})

await test('setParams / getParams', async () => {
  const model = await KNNModel.create({ k: 5, metric: 'l2' })
  const params = model.getParams()
  assert(params.k === 5, 'k should be 5')
  assert(params.metric === 'l2', 'metric should be l2')

  model.setParams({ k: 10 })
  assert(model.getParams().k === 10, 'k should be updated to 10')

  model.dispose()
})

await test('capabilities reflect task', async () => {
  const clf = await KNNModel.create({ task: 'classification' })
  assert(clf.capabilities.classifier === true, 'classifier')
  assert(clf.capabilities.predictProba === true, 'predictProba')
  clf.dispose()

  const reg = await KNNModel.create({ task: 'regression' })
  assert(reg.capabilities.regressor === true, 'regressor')
  assert(reg.capabilities.predictProba === false, 'no predictProba for regression')
  reg.dispose()
})

// ============================================================
// L1 metric
// ============================================================
console.log('\n=== L1 metric ===')

await test('Classification with L1 metric', async () => {
  const rng = makeLCG(700)
  const { X, y } = makeClassificationData(rng, 60, 2)
  const model = await KNNModel.create({ k: 5, task: 'classification', metric: 'l1' })
  model.fit(X, y)

  const accuracy = model.score(X, y)
  assert(accuracy > 0.8, `L1 accuracy too low: ${accuracy}`)

  model.dispose()
})

await test('Regression with L1 metric', async () => {
  const rng = makeLCG(701)
  const { X, y } = makeRegressionData(rng, 60, 2)
  const model = await KNNModel.create({ k: 5, task: 'regression', metric: 'l1' })
  model.fit(X, y)

  const r2 = model.score(X, y)
  assert(r2 > 0, `L1 R2 too low: ${r2}`)

  model.dispose()
})

// ============================================================
// Persistence (save / load)
// ============================================================
console.log('\n=== Persistence ===')

await test('Classification save/load round-trip', async () => {
  const rng = makeLCG(800)
  const { X, y } = makeClassificationData(rng, 50, 2)
  const model = await KNNModel.create({ k: 5, task: 'classification', metric: 'l2' })
  model.fit(X, y)

  const preds1 = model.predict(X)
  const bundle = model.save()
  model.dispose()

  assert(bundle instanceof Uint8Array, 'save should return Uint8Array')
  assert(bundle.length > 0, 'bundle should not be empty')

  // Load and predict
  const model2 = await KNNModel.load(bundle)
  assert(model2.isFitted, 'loaded model should be fitted')
  assert(model2.nSamples === 50, `nSamples: ${model2.nSamples}`)
  assert(model2.nFeatures === 2, `nFeatures: ${model2.nFeatures}`)
  assert(model2.nClasses === 2, `nClasses: ${model2.nClasses}`)

  const preds2 = model2.predict(X)
  assert(preds2.length === preds1.length, 'prediction length mismatch')

  // Predictions should be identical (same data, same tree)
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i], `pred mismatch at ${i}: ${preds1[i]} vs ${preds2[i]}`)
  }

  model2.dispose()
})

await test('Regression save/load round-trip', async () => {
  const rng = makeLCG(801)
  const { X, y } = makeRegressionData(rng, 50, 2)
  const model = await KNNModel.create({ k: 5, task: 'regression', metric: 'l2' })
  model.fit(X, y)

  const preds1 = model.predict(X)
  const bundle = model.save()
  model.dispose()

  const model2 = await KNNModel.load(bundle)
  const preds2 = model2.predict(X)

  for (let i = 0; i < preds1.length; i++) {
    assertClose(preds1[i], preds2[i], 1e-10, `pred mismatch at ${i}`)
  }

  model2.dispose()
})

await test('Bundle blob has NF01 header', async () => {
  const { decodeBundle } = await import('@wlearn/core')

  const rng = makeLCG(802)
  const { X, y } = makeClassificationData(rng, 20, 2)
  const model = await KNNModel.create({ k: 3, task: 'classification' })
  model.fit(X, y)
  const bundle = model.save()
  model.dispose()

  const { manifest, toc, blobs } = decodeBundle(bundle)
  assert(manifest.typeId === 'wlearn.nanoflann.classifier@1', `typeId: ${manifest.typeId}`)
  assert(manifest.metadata.nSamples === 20, `nSamples: ${manifest.metadata.nSamples}`)
  assert(manifest.metadata.nFeatures === 2, `nFeatures: ${manifest.metadata.nFeatures}`)
  assert(manifest.metadata.nClasses === 2, `nClasses: ${manifest.metadata.nClasses}`)

  const entry = toc.find(e => e.id === 'model')
  assert(entry, 'should have model artifact')

  const blob = blobs.subarray(entry.offset, entry.offset + entry.length)
  // Check NF01 magic
  const magic = String.fromCharCode(blob[0], blob[1], blob[2], blob[3])
  assert(magic === 'NF01', `magic: ${magic}`)

  // Check dimensions in header
  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength)
  assert(dv.getUint32(4, true) === 20, 'nSamples in header')
  assert(dv.getUint32(8, true) === 2, 'nFeatures in header')
  assert(blob[12] === 0, 'task byte should be 0 (classification)')

  // Check total blob size: 16 + 20*2*8 + 20*4 = 16 + 320 + 80 = 416
  assert(blob.length === 416, `blob length: ${blob.length}, expected 416`)
})

await test('Regressor bundle blob size', async () => {
  const { decodeBundle } = await import('@wlearn/core')

  const rng = makeLCG(803)
  const { X, y } = makeRegressionData(rng, 20, 2)
  const model = await KNNModel.create({ k: 3, task: 'regression' })
  model.fit(X, y)
  const bundle = model.save()
  model.dispose()

  const { toc, blobs } = decodeBundle(bundle)
  const entry = toc.find(e => e.id === 'model')
  const blob = blobs.subarray(entry.offset, entry.offset + entry.length)

  assert(blob[12] === 1, 'task byte should be 1 (regression)')
  // 16 + 20*2*8 + 20*8 = 16 + 320 + 160 = 496
  assert(blob.length === 496, `blob length: ${blob.length}, expected 496`)
})

await test('Save/load with L1 metric preserves predictions', async () => {
  const rng = makeLCG(804)
  const { X, y } = makeClassificationData(rng, 40, 2)
  const model = await KNNModel.create({ k: 3, task: 'classification', metric: 'l1' })
  model.fit(X, y)

  const preds1 = model.predict(X)
  const bundle = model.save()
  model.dispose()

  const model2 = await KNNModel.load(bundle)
  const preds2 = model2.predict(X)

  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i], `L1 pred mismatch at ${i}`)
  }

  model2.dispose()
})

// ============================================================
// Disposal & error handling
// ============================================================
console.log('\n=== Disposal ===')

await test('dispose() prevents further use', async () => {
  const rng = makeLCG(900)
  const { X, y } = makeClassificationData(rng, 20, 2)
  const model = await KNNModel.create({ k: 3, task: 'classification' })
  model.fit(X, y)
  model.dispose()

  assert(!model.isFitted, 'should not be fitted after dispose')

  let threw = false
  try { model.predict(X) } catch (e) {
    threw = true
    assert(e.message.includes('disposed'), `wrong error: ${e.message}`)
  }
  assert(threw, 'should throw after dispose')
})

await test('double dispose is safe', async () => {
  const model = await KNNModel.create({ k: 3 })
  model.dispose()
  model.dispose() // should not throw
})

await test('predict before fit throws NotFittedError', async () => {
  const model = await KNNModel.create({ k: 3, task: 'classification' })
  let threw = false
  try { model.predict([[1, 2]]) } catch (e) {
    threw = true
    assert(e.message.includes('not fitted'), `wrong error: ${e.message}`)
  }
  assert(threw, 'should throw')
  model.dispose()
})

// ============================================================
// defaultSearchSpace
// ============================================================
console.log('\n=== Search space ===')

await test('defaultSearchSpace returns valid IR', async () => {
  const space = KNNModel.defaultSearchSpace()
  assert(space.k, 'should have k')
  assert(space.k.type === 'int_uniform', 'k type')
  assert(space.metric, 'should have metric')
  assert(space.metric.type === 'categorical', 'metric type')
  assert(space.leafMaxSize, 'should have leafMaxSize')
})

// ============================================================
// Summary
// ============================================================
console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed\n`)
if (failed > 0) process.exit(1)

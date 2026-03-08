const { loadNanoflann, getWasm } = require('./wasm.js')
const { KNNModel, Metric } = require('./model.js')

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await KNNModel.create(params)
  model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
async function predict(bundleBytes, X) {
  const model = await KNNModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}

module.exports = { loadNanoflann, getWasm, KNNModel, Metric, train, predict }

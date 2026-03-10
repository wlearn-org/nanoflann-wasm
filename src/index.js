const { loadNanoflann, getWasm } = require('./wasm.js')
const { KNNModel: KNNModelImpl, Metric } = require('./model.js')
const { createModelClass } = require('@wlearn/core')

const KNNModel = createModelClass(KNNModelImpl, KNNModelImpl, { name: 'KNNModel', load: loadNanoflann })

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await KNNModel.create(params)
  await model.fit(X, y)
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

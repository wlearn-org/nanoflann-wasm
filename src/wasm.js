// WASM loader -- loads the nanoflann WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadNanoflann(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    const createNanoflann = require('../wasm/nanoflann.cjs')
    wasmModule = await createNanoflann(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadNanoflann() first')
  return wasmModule
}

module.exports = { loadNanoflann, getWasm }

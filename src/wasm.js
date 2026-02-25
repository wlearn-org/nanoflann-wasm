// WASM loader -- loads the nanoflann WASM module (singleton, lazy init)

import { createRequire } from 'module'

let wasmModule = null
let loading = null

export async function loadNanoflann(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    // Emscripten output is CJS, use createRequire for ESM compatibility
    const require = createRequire(import.meta.url)
    const createNanoflann = require('../wasm/nanoflann.cjs')
    wasmModule = await createNanoflann(options)
    return wasmModule
  })()

  return loading
}

export function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadNanoflann() first')
  return wasmModule
}

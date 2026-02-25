#!/bin/bash
set -euo pipefail

# Verify that the built WASM module exports all expected symbols.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WASM_FILE="${PROJECT_DIR}/wasm/nanoflann.cjs"

if [ ! -f "$WASM_FILE" ]; then
  echo "ERROR: ${WASM_FILE} not found. Run build-wasm.sh first."
  exit 1
fi

EXPECTED_EXPORTS=(
  wl_nf_get_last_error
  wl_nf_build
  wl_nf_knn_search
  wl_nf_save
  wl_nf_load
  wl_nf_free
  wl_nf_free_buffer
  wl_nf_get_n_samples
  wl_nf_get_n_features
  wl_nf_get_task
  wl_nf_get_label
)

missing=0
for fn in "${EXPECTED_EXPORTS[@]}"; do
  if ! grep -q "_${fn}" "$WASM_FILE"; then
    echo "MISSING: _${fn}"
    missing=$((missing + 1))
  fi
done

if [ $missing -gt 0 ]; then
  echo "ERROR: ${missing} exports missing from ${WASM_FILE}"
  exit 1
fi

echo "All ${#EXPECTED_EXPORTS[@]} exports verified."

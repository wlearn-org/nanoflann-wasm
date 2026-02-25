#!/bin/bash
set -euo pipefail

# Build nanoflann KD-tree as WASM via Emscripten
# Prerequisites: emsdk activated (em++ in PATH)
#
# nanoflann is header-only C++; only csrc/wl_api.cpp needs compiling.
# No MEMFS needed (binary serialization via NF01 blob format).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="${PROJECT_DIR}/upstream/nanoflann"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v em++ &> /dev/null; then
  echo "ERROR: em++ not found. Activate emsdk first:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

if [ ! -f "$UPSTREAM_DIR/include/nanoflann.hpp" ]; then
  echo "ERROR: nanoflann upstream not found at ${UPSTREAM_DIR}"
  echo "  git submodule update --init"
  exit 1
fi

echo "=== Compiling WASM ==="
mkdir -p "$OUTPUT_DIR"

EXPORTED_FUNCTIONS='[
  "_wl_nf_get_last_error",
  "_wl_nf_build",
  "_wl_nf_knn_search",
  "_wl_nf_save",
  "_wl_nf_load",
  "_wl_nf_free",
  "_wl_nf_free_buffer",
  "_wl_nf_get_n_samples",
  "_wl_nf_get_n_features",
  "_wl_nf_get_task",
  "_wl_nf_get_label",
  "_malloc",
  "_free"
]'

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","HEAP32"]'

em++ \
  "${PROJECT_DIR}/csrc/wl_api.cpp" \
  -I "${UPSTREAM_DIR}/include" \
  -o "${OUTPUT_DIR}/nanoflann.cjs" \
  -std=c++11 \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s EXPORT_NAME=createNanoflann \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 \
  -s ENVIRONMENT='web,node' \
  -O2

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: nanoflann v1.6.3 (header-only KD-tree)
upstream_commit: $(cd "$UPSTREAM_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 -std=c++11 SINGLE_FILE=1
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/nanoflann.cjs"
cat "${OUTPUT_DIR}/BUILD_INFO"

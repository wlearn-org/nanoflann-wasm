/*
 * wl_api.cpp -- C++ adapter for nanoflann v1.6.3
 *
 * Bridges dense JS float64 arrays to nanoflann KD-tree.
 * Provides batch KNN search, custom binary serialization (NF01 format),
 * and ownership-transfer semantics for zero-extra-copy from JS.
 *
 * Compile with: em++ csrc/wl_api.cpp -I upstream/nanoflann/include ...
 */

#include "nanoflann.hpp"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <new>

/* ---------- blob header format (NF01) ---------- */

// 16 bytes fixed header inside the "model" blob:
//   [0..3]   magic "NF01" (4 bytes ASCII)
//   [4..7]   nSamples  (uint32 LE)
//   [8..11]  nFeatures (uint32 LE)
//   [12]     task: 0 = classification, 1 = regression
//   [13..15] reserved (zeros)
//
// Then:
//   [16 .. 16 + nSamples*nFeatures*8)  X data (float64 LE, row-major)
//   classification: [... + nSamples*4)  y data (int32 LE)
//   regression:     [... + nSamples*8)  y data (float64 LE)

static const char NF01_MAGIC[4] = {'N', 'F', '0', '1'};
static const int  NF01_HEADER   = 16;

/* ---------- error handling ---------- */

static char last_error[512] = "";

static void set_error(const char *msg) {
  strncpy(last_error, msg, sizeof(last_error) - 1);
  last_error[sizeof(last_error) - 1] = '\0';
}

/* ---------- dense matrix adaptor (C++ linkage) ---------- */

struct DenseAdaptor {
  const double *data;
  size_t rows;
  size_t cols;

  inline size_t kdtree_get_point_count() const { return rows; }

  inline double kdtree_get_pt(size_t idx, size_t dim) const {
    return data[idx * cols + dim];
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX &) const { return false; }
};

/* ---------- tree type aliases ---------- */

typedef nanoflann::KDTreeSingleIndexAdaptor<
  nanoflann::L2_Simple_Adaptor<double, DenseAdaptor>,
  DenseAdaptor, -1, size_t
> KDTreeL2;

typedef nanoflann::KDTreeSingleIndexAdaptor<
  nanoflann::L1_Adaptor<double, DenseAdaptor>,
  DenseAdaptor, -1, size_t
> KDTreeL1;

/* ---------- opaque index handle ---------- */

struct WlIndex {
  DenseAdaptor adaptor;
  void *tree;           /* KDTreeL2* or KDTreeL1* */
  int metric;           /* 0 = l2, 1 = l1 */
  int leaf_max_size;
  double *X_owned;      /* owned training features */
  double *y_f64;        /* owned labels (regression: float64) */
  int32_t *y_i32;       /* owned labels (classification: int32) */
  int task;             /* 0 = classification, 1 = regression */
  size_t n_samples;
  size_t n_features;
};

/* ---------- internal: build tree ---------- */

static int build_tree(WlIndex *idx) {
  nanoflann::KDTreeSingleIndexAdaptorParams params(
    static_cast<size_t>(idx->leaf_max_size)
  );
  idx->adaptor.data = idx->X_owned;
  idx->adaptor.rows = idx->n_samples;
  idx->adaptor.cols = idx->n_features;

  if (idx->metric == 0) {
    KDTreeL2 *t = new (std::nothrow) KDTreeL2(
      static_cast<int>(idx->n_features), idx->adaptor, params
    );
    if (!t) return -1;
    t->buildIndex();
    idx->tree = t;
  } else {
    KDTreeL1 *t = new (std::nothrow) KDTreeL1(
      static_cast<int>(idx->n_features), idx->adaptor, params
    );
    if (!t) return -1;
    t->buildIndex();
    idx->tree = t;
  }
  return 0;
}

/* ---------- internal: free index ---------- */

static void free_index(WlIndex *idx) {
  if (!idx) return;
  if (idx->tree) {
    if (idx->metric == 0) delete (KDTreeL2 *)idx->tree;
    else delete (KDTreeL1 *)idx->tree;
  }
  free(idx->X_owned);
  free(idx->y_i32);
  free(idx->y_f64);
  free(idx);
}

/* ========== extern "C" exports ========== */

extern "C" {

const char* wl_nf_get_last_error(void) {
  return last_error;
}

/*
 * Build KD-tree index.
 *
 * If take_ownership is true, X_ptr must have been allocated via
 * malloc/_malloc and this function takes ownership (caller must NOT free).
 * Otherwise, data is copied into newly allocated buffers.
 *
 * task: 0 = classification (y treated as int32-valued float64), 1 = regression
 * metric: 0 = l2, 1 = l1
 */
WlIndex* wl_nf_build(
  double *X_ptr, int nrow, int ncol,
  const double *y_ptr,
  int task, int metric, int leaf_max_size,
  int take_ownership
) {
  last_error[0] = '\0';

  if (!X_ptr || !y_ptr || nrow <= 0 || ncol <= 0) {
    set_error("Invalid input: X, y, nrow, ncol required");
    return NULL;
  }
  if (metric < 0 || metric > 1) {
    set_error("Invalid metric: 0=l2, 1=l1");
    return NULL;
  }
  if (leaf_max_size <= 0) leaf_max_size = 10;

  WlIndex *idx = (WlIndex *)calloc(1, sizeof(WlIndex));
  if (!idx) {
    set_error("Failed to allocate WlIndex");
    return NULL;
  }

  idx->metric = metric;
  idx->leaf_max_size = leaf_max_size;
  idx->task = task;
  idx->n_samples = (size_t)nrow;
  idx->n_features = (size_t)ncol;

  /* X: take ownership or copy */
  if (take_ownership) {
    idx->X_owned = X_ptr;
  } else {
    size_t xbytes = (size_t)nrow * (size_t)ncol * sizeof(double);
    idx->X_owned = (double *)malloc(xbytes);
    if (!idx->X_owned) {
      set_error("Failed to allocate X copy");
      free(idx);
      return NULL;
    }
    memcpy(idx->X_owned, X_ptr, xbytes);
  }

  /* y: always copy (convert to int32 for classification) */
  if (task == 0) {
    idx->y_i32 = (int32_t *)malloc((size_t)nrow * sizeof(int32_t));
    if (!idx->y_i32) {
      set_error("Failed to allocate y_i32");
      free(idx->X_owned);
      free(idx);
      return NULL;
    }
    for (int i = 0; i < nrow; i++) {
      idx->y_i32[i] = (int32_t)y_ptr[i];
    }
    idx->y_f64 = NULL;
  } else {
    idx->y_f64 = (double *)malloc((size_t)nrow * sizeof(double));
    if (!idx->y_f64) {
      set_error("Failed to allocate y_f64");
      free(idx->X_owned);
      free(idx);
      return NULL;
    }
    memcpy(idx->y_f64, y_ptr, (size_t)nrow * sizeof(double));
    idx->y_i32 = NULL;
  }

  /* Build KD-tree */
  if (build_tree(idx) != 0) {
    set_error("Failed to build KD-tree");
    free_index(idx);
    return NULL;
  }

  return idx;
}

/*
 * Batch KNN search.
 *
 * For each query row, find k nearest neighbors.
 * out_indices:   int32 array of size n_queries * k_eff
 * out_distances: float64 array of size n_queries * k_eff
 *
 * Distances: L2 returns actual Euclidean distance (sqrt of squared).
 *            L1 returns actual L1 distance.
 *
 * k is clamped to n_samples. Returns effective k used.
 * Returns -1 on error.
 */
int wl_nf_knn_search(
  const WlIndex *idx,
  const double *Q, int n_queries, int ncol, int k,
  int *out_indices, double *out_distances
) {
  if (!idx || !Q || !out_indices || !out_distances) {
    set_error("knn_search: null argument");
    return -1;
  }
  if (ncol != (int)idx->n_features) {
    set_error("knn_search: query ncol does not match index n_features");
    return -1;
  }

  /* Clamp k to n_samples */
  int k_eff = k;
  if (k_eff > (int)idx->n_samples) k_eff = (int)idx->n_samples;
  if (k_eff <= 0) {
    set_error("knn_search: k must be > 0");
    return -1;
  }

  /* Temp buffers for nanoflann output (size_t indices, double distances) */
  size_t *tmp_idx = (size_t *)malloc((size_t)k_eff * sizeof(size_t));
  double *tmp_dist = (double *)malloc((size_t)k_eff * sizeof(double));
  if (!tmp_idx || !tmp_dist) {
    free(tmp_idx);
    free(tmp_dist);
    set_error("knn_search: allocation failed");
    return -1;
  }

  int is_l2 = (idx->metric == 0);

  for (int i = 0; i < n_queries; i++) {
    const double *qrow = Q + (size_t)i * (size_t)ncol;
    size_t found;

    if (is_l2) {
      found = ((KDTreeL2 *)idx->tree)->knnSearch(
        qrow, (size_t)k_eff, tmp_idx, tmp_dist
      );
    } else {
      found = ((KDTreeL1 *)idx->tree)->knnSearch(
        qrow, (size_t)k_eff, tmp_idx, tmp_dist
      );
    }

    int *oi = out_indices + (size_t)i * (size_t)k_eff;
    double *od = out_distances + (size_t)i * (size_t)k_eff;

    for (int j = 0; j < k_eff; j++) {
      if ((size_t)j < found) {
        oi[j] = (int)tmp_idx[j];
        /* L2_Simple returns squared distance; take sqrt for actual L2 */
        od[j] = is_l2 ? sqrt(tmp_dist[j]) : tmp_dist[j];
      } else {
        oi[j] = -1;
        od[j] = -1.0;
      }
    }
  }

  free(tmp_idx);
  free(tmp_dist);
  return k_eff;
}

/* ---------- save (NF01 binary format) ---------- */

int wl_nf_save(const WlIndex *idx, char **out_buf, int *out_len) {
  if (!idx || !out_buf || !out_len) {
    set_error("save: null argument");
    return -1;
  }

  size_t x_bytes = idx->n_samples * idx->n_features * sizeof(double);
  size_t y_bytes = (idx->task == 0)
    ? idx->n_samples * sizeof(int32_t)
    : idx->n_samples * sizeof(double);
  size_t total = (size_t)NF01_HEADER + x_bytes + y_bytes;

  char *buf = (char *)malloc(total);
  if (!buf) {
    set_error("save: allocation failed");
    return -1;
  }

  /* Write header */
  memcpy(buf, NF01_MAGIC, 4);
  uint32_t ns = (uint32_t)idx->n_samples;
  uint32_t nf = (uint32_t)idx->n_features;
  uint8_t tk = (uint8_t)idx->task;
  memcpy(buf + 4, &ns, 4);
  memcpy(buf + 8, &nf, 4);
  buf[12] = (char)tk;
  buf[13] = 0; buf[14] = 0; buf[15] = 0;

  /* Write X */
  memcpy(buf + NF01_HEADER, idx->X_owned, x_bytes);

  /* Write y */
  char *y_dest = buf + NF01_HEADER + x_bytes;
  if (idx->task == 0) {
    memcpy(y_dest, idx->y_i32, y_bytes);
  } else {
    memcpy(y_dest, idx->y_f64, y_bytes);
  }

  *out_buf = buf;
  *out_len = (int)total;
  return 0;
}

/* ---------- load (NF01 binary format) ---------- */

WlIndex* wl_nf_load(const char *buf, int len, int metric, int leaf_max_size) {
  last_error[0] = '\0';

  if (!buf || len < NF01_HEADER) {
    set_error("load: invalid buffer");
    return NULL;
  }

  /* Validate magic */
  if (memcmp(buf, NF01_MAGIC, 4) != 0) {
    set_error("load: invalid magic (expected NF01)");
    return NULL;
  }

  uint32_t ns, nf;
  memcpy(&ns, buf + 4, 4);
  memcpy(&nf, buf + 8, 4);
  uint8_t tk = (uint8_t)buf[12];

  if (tk > 1) {
    set_error("load: invalid task byte");
    return NULL;
  }

  size_t x_bytes = (size_t)ns * (size_t)nf * sizeof(double);
  size_t y_bytes = (tk == 0)
    ? (size_t)ns * sizeof(int32_t)
    : (size_t)ns * sizeof(double);
  size_t expected = (size_t)NF01_HEADER + x_bytes + y_bytes;

  if ((size_t)len < expected) {
    set_error("load: buffer too short for declared dimensions");
    return NULL;
  }

  WlIndex *idx = (WlIndex *)calloc(1, sizeof(WlIndex));
  if (!idx) {
    set_error("load: allocation failed");
    return NULL;
  }

  idx->metric = metric;
  idx->leaf_max_size = (leaf_max_size > 0) ? leaf_max_size : 10;
  idx->task = (int)tk;
  idx->n_samples = (size_t)ns;
  idx->n_features = (size_t)nf;

  /* Copy X */
  idx->X_owned = (double *)malloc(x_bytes);
  if (!idx->X_owned) {
    set_error("load: X allocation failed");
    free(idx);
    return NULL;
  }
  memcpy(idx->X_owned, buf + NF01_HEADER, x_bytes);

  /* Copy y */
  const char *y_src = buf + NF01_HEADER + x_bytes;
  if (tk == 0) {
    idx->y_i32 = (int32_t *)malloc(y_bytes);
    if (!idx->y_i32) {
      set_error("load: y allocation failed");
      free(idx->X_owned);
      free(idx);
      return NULL;
    }
    memcpy(idx->y_i32, y_src, y_bytes);
    idx->y_f64 = NULL;
  } else {
    idx->y_f64 = (double *)malloc(y_bytes);
    if (!idx->y_f64) {
      set_error("load: y allocation failed");
      free(idx->X_owned);
      free(idx);
      return NULL;
    }
    memcpy(idx->y_f64, y_src, y_bytes);
    idx->y_i32 = NULL;
  }

  /* Rebuild tree */
  if (build_tree(idx) != 0) {
    set_error("load: failed to build KD-tree");
    free_index(idx);
    return NULL;
  }

  return idx;
}

/* ---------- free ---------- */

void wl_nf_free(WlIndex *idx) {
  free_index(idx);
}

void wl_nf_free_buffer(void *ptr) {
  free(ptr);
}

/* ---------- inspection ---------- */

int wl_nf_get_n_samples(const WlIndex *idx) {
  return idx ? (int)idx->n_samples : 0;
}

int wl_nf_get_n_features(const WlIndex *idx) {
  return idx ? (int)idx->n_features : 0;
}

int wl_nf_get_task(const WlIndex *idx) {
  return idx ? idx->task : -1;
}

/* Return label at index i. Classification: from y_i32; regression: from y_f64. */
double wl_nf_get_label(const WlIndex *idx, int i) {
  if (!idx || i < 0 || (size_t)i >= idx->n_samples) return 0.0;
  if (idx->task == 0 && idx->y_i32) return (double)idx->y_i32[i];
  if (idx->y_f64) return idx->y_f64[i];
  return 0.0;
}

} /* extern "C" */

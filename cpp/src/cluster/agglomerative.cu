/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/agglomerative.cuh"

#include <cuvs/cluster/agglomerative.hpp>

namespace cuvs::cluster::agglomerative::helpers {

#define CUVS_INST_AGGLOMERATIVE(IdxT, ValueT)                        \
  void build_dendrogram_host(raft::resources const& handle,          \
                             const IdxT* rows,                       \
                             const IdxT* cols,                       \
                             const ValueT* data,                     \
                             size_t nnz,                             \
                             IdxT* children,                         \
                             ValueT* out_delta,                      \
                             IdxT* out_size)                         \
  {                                                                  \
    detail::build_dendrogram_host<IdxT, ValueT>(                     \
      handle, rows, cols, data, nnz, children, out_delta, out_size); \
  }

CUVS_INST_AGGLOMERATIVE(int64_t, float);

#undef CUVS_INST_AGGLOMERATIVE

}  // namespace cuvs::cluster::agglomerative::helpers

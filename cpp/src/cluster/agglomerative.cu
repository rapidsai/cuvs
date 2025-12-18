/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/agglomerative.cuh"

#include <cuvs/cluster/agglomerative.hpp>

namespace cuvs::cluster::agglomerative::helpers {

#define CUVS_INST_AGGLOMERATIVE(IdxT, ValueT)                                           \
  void build_dendrogram(raft::resources const& handle,                                  \
                        raft::device_vector_view<const IdxT, IdxT> rows,                \
                        raft::device_vector_view<const IdxT, IdxT> cols,                \
                        raft::device_vector_view<const ValueT, IdxT> data,              \
                        raft::device_matrix_view<IdxT, IdxT, raft::row_major> children, \
                        raft::device_vector_view<ValueT, IdxT> out_delta,               \
                        raft::device_vector_view<IdxT, IdxT> out_size)                  \
  {                                                                                     \
    size_t nnz = rows.extent(0);                                                        \
    detail::build_dendrogram_host<IdxT, ValueT>(handle,                                 \
                                                rows.data_handle(),                     \
                                                cols.data_handle(),                     \
                                                data.data_handle(),                     \
                                                nnz,                                    \
                                                children.data_handle(),                 \
                                                out_delta.data_handle(),                \
                                                out_size.data_handle());                \
  }

CUVS_INST_AGGLOMERATIVE(int64_t, float);

#undef CUVS_INST_AGGLOMERATIVE

}  // namespace cuvs::cluster::agglomerative::helpers

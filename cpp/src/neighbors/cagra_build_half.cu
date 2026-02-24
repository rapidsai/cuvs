/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra.cuh"
#include <cuda_fp16.h>
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

void build_knn_graph(raft::resources const& handle,
                     raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
                     raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
                     cuvs::neighbors::cagra::graph_build_params::ivf_pq_params params)
{
  cuvs::neighbors::cagra::build_knn_graph<half, uint32_t>(handle, dataset, knn_graph, params);
}

cuvs::neighbors::cagra::index<half, uint32_t> build(
  raft::resources const& handle,
  const cuvs::neighbors::cagra::index_params& params,
  raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
{
  return cuvs::neighbors::cagra::build<half, uint32_t>(handle, params, dataset);
}

cuvs::neighbors::cagra::index<half, uint32_t> build(
  raft::resources const& handle,
  const cuvs::neighbors::cagra::index_params& params,
  raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
{
  return cuvs::neighbors::cagra::build<half, uint32_t>(handle, params, dataset);
}

template auto build(raft::resources const& res,
                    const cuvs::neighbors::cagra::index_params& params,
                    cuvs::neighbors::device_padded_dataset_view<half, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::index<half, uint32_t>;
template auto build(raft::resources const& res,
                    const cuvs::neighbors::cagra::index_params& params,
                    cuvs::neighbors::device_padded_dataset<half, int64_t>&& dataset)
  -> cuvs::neighbors::cagra::index<half, uint32_t>;

}  // namespace cuvs::neighbors::cagra

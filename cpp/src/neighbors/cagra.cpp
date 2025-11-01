/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

auto hnsw_to_cagra_params(raft::matrix_extent<int64_t> dataset,
                          int M,
                          int ef_construction,
                          cuvs::distance::DistanceType metric) -> cagra::index_params
{
  auto ivf_pq_params = cuvs::neighbors::graph_build_params::ivf_pq_params(dataset, metric);
  ivf_pq_params.search_params.n_probes =
    std::round(std::sqrt(ivf_pq_params.build_params.n_lists) / 20 + ef_construction / 16);

  cagra::index_params params;
  params.graph_build_params        = ivf_pq_params;
  params.graph_degree              = M * 2;
  params.intermediate_graph_degree = M * 3;

  return params;
}

}  // namespace cuvs::neighbors::cagra

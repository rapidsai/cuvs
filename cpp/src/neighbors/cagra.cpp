/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::cagra {

inline auto graph_params_heuristic(raft::matrix_extent<int64_t> dataset,
                                   int intermediate_graph_degree,
                                   int ef_construction,
                                   cuvs::distance::DistanceType metric)
  -> decltype(index_params::graph_build_params)
{
  if (dataset.extent(0) < int64_t(1e6)) {
    // Use NN descent for smaller datasets
    auto nn_descent_params =
      graph_build_params::nn_descent_params(intermediate_graph_degree, metric);
    nn_descent_params.max_iterations = 5 + ef_construction / 16;
    return nn_descent_params;
  } else {
    // Otherwise, use IVF-PQ
    auto ivf_pq_params = cuvs::neighbors::graph_build_params::ivf_pq_params(dataset, metric);
    ivf_pq_params.search_params.n_probes =
      std::round(2 + std::sqrt(ivf_pq_params.build_params.n_lists) / 20 + ef_construction / 16);
    return ivf_pq_params;
  }
}

cagra::index_params index_params::from_hnsw_hard_m(raft::matrix_extent<int64_t> dataset,
                                                   int M,
                                                   int ef_construction,
                                                   cuvs::distance::DistanceType metric)
{
  cagra::index_params params;
  params.graph_degree              = M * 2;
  params.intermediate_graph_degree = M * 3;
  params.graph_build_params =
    graph_params_heuristic(dataset, params.intermediate_graph_degree, ef_construction, metric);
  return params;
}

cagra::index_params index_params::from_hnsw_soft_m(raft::matrix_extent<int64_t> dataset,
                                                   int M,
                                                   int ef_construction,
                                                   cuvs::distance::DistanceType metric)
{
  cagra::index_params params;
  params.graph_degree              = 2 + M * 2 / 3;
  params.intermediate_graph_degree = M + M * ef_construction / 256;
  params.graph_build_params =
    graph_params_heuristic(dataset, params.intermediate_graph_degree, ef_construction, metric);
  return params;
}

}  // namespace cuvs::neighbors::cagra

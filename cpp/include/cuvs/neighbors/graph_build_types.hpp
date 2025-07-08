/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

namespace cuvs::neighbors {

/**
 * @defgroup neighbors_build_algo Graph build algorithm types
 * @{
 */

enum GRAPH_BUILD_ALGO { BRUTE_FORCE = 0, IVF_PQ = 1, NN_DESCENT = 1 };

namespace graph_build_params {

/** Specialized parameters utilizing IVF-PQ to build knn graph */
struct ivf_pq_params {
  cuvs::neighbors::ivf_pq::index_params build_params;
  cuvs::neighbors::ivf_pq::search_params search_params;
  float refinement_rate = 1.0;

  ivf_pq_params() = default;

  /**
   * Set default parameters based on shape of the input dataset.
   * Usage example:
   * @code{.cpp}
   *   using namespace cuvs::neighbors;
   *   raft::resources res;
   *   // create index_params for a [N. D] dataset
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
   *   auto pq_params =
   *     graph_build_params::ivf_pq_params(dataset.extents());
   *   // modify/update index_params as needed
   *   pq_params.kmeans_trainset_fraction = 0.1;
   * @endcode
   */
  ivf_pq_params(raft::matrix_extent<int64_t> dataset_extents,
                cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
  {
    build_params = cuvs::neighbors::ivf_pq::index_params::from_dataset(dataset_extents, metric);

    search_params                         = cuvs::neighbors::ivf_pq::search_params{};
    search_params.n_probes                = std::max<uint32_t>(10, build_params.n_lists * 0.01);
    search_params.lut_dtype               = CUDA_R_16F;
    search_params.internal_distance_dtype = CUDA_R_16F;
    search_params.coarse_search_dtype     = CUDA_R_16F;
    search_params.max_internal_batch_size = 128 * 1024;

    refinement_rate = 1;
  }
};

using nn_descent_params = cuvs::neighbors::nn_descent::index_params;

struct brute_force_params {
  cuvs::neighbors::brute_force::index_params build_params;
  cuvs::neighbors::brute_force::search_params search_params;
};

// **** Experimental ****
using iterative_search_params = cuvs::neighbors::search_params;
}  // namespace graph_build_params

/** @} */  // end group neighbors_build_algo
}  // namespace cuvs::neighbors

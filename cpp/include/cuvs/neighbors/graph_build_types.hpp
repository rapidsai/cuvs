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

enum GRAPH_BUILD_ALGO { BRUTE_FORCE = 0, IVF_PQ = 1, NN_DESCENT = 2, ACE = 3 };

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
    build_params    = ivf_pq::index_params::from_dataset(dataset_extents, metric);
    auto n_rows     = dataset_extents.extent(0);
    auto n_features = dataset_extents.extent(1);
    if (n_features <= 32) {
      build_params.pq_dim  = 16;
      build_params.pq_bits = 8;
    } else {
      build_params.pq_bits = 4;
      if (n_features <= 64) {
        build_params.pq_dim = 32;
      } else if (n_features <= 128) {
        build_params.pq_dim = 64;
      } else if (n_features <= 192) {
        build_params.pq_dim = 96;
      } else {
        build_params.pq_dim = raft::round_up_safe<uint32_t>(n_features / 2, 128);
      }
    }

    build_params.n_lists        = std::max<uint32_t>(1, n_rows / 2000);
    build_params.kmeans_n_iters = 10;

    const double kMinPointsPerCluster         = 32;
    const double min_kmeans_trainset_points   = kMinPointsPerCluster * build_params.n_lists;
    const double max_kmeans_trainset_fraction = 1.0;
    const double min_kmeans_trainset_fraction =
      std::min(max_kmeans_trainset_fraction, min_kmeans_trainset_points / n_rows);
    build_params.kmeans_trainset_fraction = std::clamp(
      1.0 / std::sqrt(n_rows * 1e-5), min_kmeans_trainset_fraction, max_kmeans_trainset_fraction);
    build_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;

    search_params                         = cuvs::neighbors::ivf_pq::search_params{};
    search_params.n_probes                = std::round(std::sqrt(build_params.n_lists) / 20 + 4);
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

/** Specialized parameters for ACE (Augmented Core Extraction) graph build */
struct ace_params {
  /**
   * Number of partitions for ACE (Augmented Core Extraction) partitioned build.
   *
   * The search graph for very large datasets can be larger than the device or host memory.
   * To build such large graphs, we divide the graph into smaller partitions.
   * When set to a value > 1, enables the ACE partitioned approach for very large graphs.
   * Set to 0 or 1 to disable ACE partitioning and use standard build.
   */
  size_t ace_npartitions = 1;
  /**
   * The index quality for the ACE build.
   *
   * Bigger values increase the index quality. At some point, increasing this will no longer
   * improve the quality.
   */
  size_t ace_ef_construction = 120;
  /**
   * Directory to store ACE build artifacts (e.g., KNN graph, optimized graph).
   * Used when `ace_npartitions` > 1 or `ace_use_disk` is true.
   */
  std::string ace_build_dir = "";
  /**
   * Whether to use disk-based storage for ACE build.
   * When true, enables disk-based operations for memory-efficient graph construction.
   */
  bool ace_use_disk = false;

  ace_params() = default;
};

// **** Experimental ****
using iterative_search_params = cuvs::neighbors::search_params;
}  // namespace graph_build_params

/** @} */  // end group neighbors_build_algo
}  // namespace cuvs::neighbors

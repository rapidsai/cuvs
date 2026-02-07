/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
   * When set to 0 (default), the number of partitions is automatically derived
   * based on available host and GPU memory to maximize partition size while
   * ensuring the build fits in memory.
   *
   * Small values might improve recall but potentially degrade performance and
   * increase memory usage. Partitions should not be too small to prevent issues
   * in KNN graph construction. The partition size is on average 2 * (n_rows / npartitions) * dim *
   * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the
   * partition sizes (up to 3x in our tests).
   *
   * If the specified number of partitions results in partitions that exceed
   * available memory, the value will be automatically increased to fit memory
   * constraints and a warning will be issued.
   */
  size_t npartitions = 0;
  /**
   * The index quality for the ACE build.
   *
   * Bigger values increase the index quality. At some point, increasing this will no longer improve
   * the quality.
   */
  size_t ef_construction = 120;
  /**
   * Directory to store ACE build artifacts (e.g., KNN graph, optimized graph).
   *
   * Used when `use_disk` is true or when the graph does not fit in host and GPU
   * memory. This should be the fastest disk in the system and hold enough space
   * for twice the dataset, final graph, and label mapping.
   */
  std::string build_dir = "/tmp/ace_build";
  /**
   * Whether to use disk-based storage for ACE build.
   *
   * When true, enables disk-based operations for memory-efficient graph construction.
   */
  bool use_disk = false;
  /**
   * Maximum host memory to use for ACE build in GiB.
   *
   * When set to 0 (default), uses available host memory.
   * When set to a positive value, limits host memory usage to the specified amount.
   * Useful for testing or when running alongside other memory-intensive processes.
   */
  double max_host_memory_gb = 0;
  /**
   * Maximum GPU memory to use for ACE build in GiB.
   *
   * When set to 0 (default), uses available GPU memory.
   * When set to a positive value, limits GPU memory usage to the specified amount.
   * Useful for testing or when running alongside other memory-intensive processes.
   */
  double max_gpu_memory_gb = 0;

  ace_params() = default;
};

// **** Experimental ****
using iterative_search_params = cuvs::neighbors::search_params;
}  // namespace graph_build_params

/** @} */  // end group neighbors_build_algo
}  // namespace cuvs::neighbors

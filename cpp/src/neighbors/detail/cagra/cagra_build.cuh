/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "../../../core/nvtx.hpp"
#include "../../vpq_dataset.cuh"
#include "graph_core.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/neighbors/refine.hpp>

// TODO: This shouldn't be calling spatial/knn APIs
#include "../ann_utils.cuh"

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <omp.h>
#include <type_traits>
#include <vector>

#include <sys/mman.h>

namespace cuvs::neighbors::cagra::detail {

// ACE: Get partition labels for partitioned approach
template <typename DataT, typename IdxT, typename Accessor>
void get_partition_labels(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> labels,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_histogram,
  double sampling_rate = 0.01)
{
  uint64_t dataset_size = dataset.extent(0);
  uint64_t dataset_dim  = dataset.extent(1);
  uint64_t labels_size  = labels.extent(0);
  uint64_t labels_dim   = labels.extent(1);
  RAFT_EXPECTS(dataset_size == labels_size, "Dataset size must match labels extent");
  uint64_t n_partitions = partition_histogram.extent(0);
  RAFT_EXPECTS(labels_dim == 2, "Labels must have 2 columns");
  RAFT_EXPECTS(partition_histogram.extent(1) == 2, "Partition histogram must have 2 columns");
  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  // Sampling vectors from dataset
  RAFT_LOG_INFO("ACE: Sampling vectors from dataset ...");
  uint64_t n_samples = dataset_size * sampling_rate;
  if (n_samples < 100 * n_partitions) { n_samples = 100 * n_partitions; }
  if (n_samples > dataset_size) { n_samples = dataset_size; }
  RAFT_LOG_INFO("ACE: n_samples: %lu", n_samples);

  auto sample_db = raft::make_host_matrix<float>(n_samples, dataset_dim);
#pragma omp parallel for
  for (uint64_t i = 0; i < n_samples; i++) {
    uint64_t j = i * dataset_size / n_samples;
    for (uint64_t k = 0; k < dataset_dim; k++) {
      sample_db(i, k) = static_cast<float>(dataset(j, k));
    }
  }
  auto sample_db_dev = raft::make_device_matrix<float, int64_t>(res, n_samples, dataset_dim);
  raft::update_device(
    sample_db_dev.data_handle(), sample_db.data_handle(), sample_db.size(), stream);

  // K-means: partitioning dataset vectors and compute centroid for each partition
  RAFT_LOG_INFO("ACE: Running k-means partitioning ...");
  cuvs::cluster::kmeans::params params;
  params.n_clusters  = n_partitions;
  params.max_iter    = 100;
  params.init        = cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus;
  float inertia      = 0.0;
  int n_iter         = 0;
  auto centroids_dev = raft::make_device_matrix<float, int64_t>(res, n_partitions, dataset_dim);
  cuvs::cluster::kmeans::fit(res,
                             params,
                             sample_db_dev.view(),
                             std::nullopt,
                             centroids_dev.view(),
                             raft::make_host_scalar_view(&inertia),
                             raft::make_host_scalar_view(&n_iter));
  RAFT_LOG_INFO("ACE: K-means inertia: %f, iterations: %d", inertia, n_iter);

  auto partition_relations = raft::make_host_matrix<IdxT, int64_t>(n_partitions, n_partitions);
  for (uint64_t c0 = 0; c0 < n_partitions; c0++) {
    for (uint64_t c1 = 0; c1 < n_partitions; c1++) {
      partition_relations(c0, c1) = 0;
    }
  }

  // Compute distances between dataset and centroid vectors
  RAFT_LOG_INFO("ACE: Computing distances between dataset and centroid vectors ...");
  const uint64_t chunk_size = 32 * 1024;
  auto _sub_dataset         = raft::make_host_matrix<float, int64_t>(chunk_size, dataset_dim);
  auto _sub_distances       = raft::make_host_matrix<float, int64_t>(chunk_size, n_partitions);
  auto _sub_dataset_dev   = raft::make_device_matrix<float, int64_t>(res, chunk_size, dataset_dim);
  auto _sub_distances_dev = raft::make_device_matrix<float, int64_t>(res, chunk_size, n_partitions);

  for (uint64_t i_base = 0; i_base < dataset_size; i_base += chunk_size) {
    const uint64_t sub_dataset_size = std::min(chunk_size, dataset_size - i_base);
    if (i_base % (dataset_size / 10) == 0) {
      RAFT_LOG_INFO("ACE: Processing chunk %lu / %lu (%.1f%%)",
                    i_base,
                    dataset_size,
                    static_cast<double>(100 * i_base) / dataset_size);
    }

    auto sub_dataset = raft::make_host_matrix_view<float, int64_t>(
      _sub_dataset.data_handle(), sub_dataset_size, dataset_dim);
#pragma omp parallel for
    for (uint64_t i_sub = 0; i_sub < sub_dataset_size; i_sub++) {
      uint64_t i = i_base + i_sub;
      for (uint64_t k = 0; k < dataset_dim; k++) {
        sub_dataset(i_sub, k) = static_cast<float>(dataset(i, k));
      }
    }
    auto sub_dataset_dev = raft::make_device_matrix_view<const float, int64_t>(
      _sub_dataset_dev.data_handle(), sub_dataset_size, dataset_dim);
    raft::update_device(
      _sub_dataset_dev.data_handle(), sub_dataset.data_handle(), sub_dataset.size(), stream);

    auto sub_distances = raft::make_host_matrix_view<float, int64_t>(
      _sub_distances.data_handle(), sub_dataset_size, n_partitions);
    auto sub_distances_dev = raft::make_device_matrix_view<float, int64_t>(
      _sub_distances_dev.data_handle(), sub_dataset_size, n_partitions);

    cuvs::distance::pairwise_distance(res,
                                      sub_dataset_dev,
                                      centroids_dev.view(),
                                      sub_distances_dev,
                                      cuvs::distance::DistanceType::L2Expanded);

    raft::update_host(
      sub_distances.data_handle(), sub_distances_dev.data_handle(), sub_distances.size(), stream);
    raft::resource::sync_stream(res, stream);

    // Find two closest partitions to each dataset vector
#pragma omp parallel for
    for (uint64_t i_sub = 0; i_sub < sub_dataset_size; i_sub++) {
      IdxT label_0 = 0;
      IdxT label_1 = 1;
      if (sub_distances(i_sub, 0) > sub_distances(i_sub, 1)) {
        label_0 = 1;
        label_1 = 0;
      }
      for (uint64_t c = 2; c < n_partitions; c++) {
        if (sub_distances(i_sub, c) < sub_distances(i_sub, label_0)) {
          label_1 = label_0;
          label_0 = c;
        } else if (sub_distances(i_sub, c) < sub_distances(i_sub, label_1)) {
          label_1 = c;
        }
      }
      uint64_t i   = i_base + i_sub;
      labels(i, 0) = label_0;
      labels(i, 1) = label_1;

#pragma omp atomic update
      partition_histogram(label_0, 0) += 1;
#pragma omp atomic update
      partition_histogram(label_1, 1) += 1;
#pragma omp atomic update
      partition_relations(label_0, label_1) += 1;
    }
  }

  RAFT_LOG_INFO("ACE: Partition labeling completed");
}

// ACE: Build partitioned CAGRA index for very large graphs. Dataset must be on host.
template <typename T, typename IdxT, typename Accessor>
index<T, IdxT> build_ace(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  size_t num_partitions = 0)
{
  common::nvtx::range<common::nvtx::domain::cuvs> function_scope(
    "cagra::build_ace<%s>(%zu, %zu, %zu)",
    Accessor::is_managed_type::value ? "managed"
    : Accessor::is_host_type::value  ? "host"
                                     : "device",
    params.intermediate_graph_degree,
    params.graph_degree,
    num_partitions);

  // Use num_partitions parameter if provided, otherwise use params.ace_npartitions
  uint64_t n_partitions = (num_partitions > 0) ? num_partitions : params.ace_npartitions;
  RAFT_EXPECTS(n_partitions > 0, "num_partitions must be greater than 0");

  RAFT_LOG_INFO("ACE: Starting partitioned CAGRA build with %zu partitions", n_partitions);

  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;
  uint64_t dataset_size      = dataset.extent(0);
  uint64_t dataset_dim       = dataset.extent(1);

  if (intermediate_degree >= dataset_size) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than dataset size, reducing it to %lu",
      dataset_size);
    intermediate_degree = dataset_size - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  // Get partition labels
  auto labels              = raft::make_host_matrix<IdxT, int64_t>(dataset_size, 2);
  auto partition_histogram = raft::make_host_matrix<IdxT, int64_t>(n_partitions, 2);
  for (uint64_t c = 0; c < n_partitions; c++) {
    partition_histogram(c, 0) = 0;
    partition_histogram(c, 1) = 0;
  }
  get_partition_labels<T, IdxT, Accessor>(res, dataset, labels.view(), partition_histogram.view());

  // ACE: Check for very small partitions and merge them with the next closest partition
  // A partition is considered too small if it has fewer vectors than the minimum required
  // for stable KNN graph construction (below n_lists)
  uint64_t min_partition_size =
    std::max<uint64_t>(intermediate_degree * 2, std::sqrt(dataset_size));
  RAFT_LOG_INFO("ACE: Checking for small partitions (min_size: %lu)", min_partition_size);

  // Find partitions that are too small and need to be merged
  std::vector<uint64_t> small_partitions;
  for (uint64_t c = 0; c < n_partitions; c++) {
    uint64_t partition_size = partition_histogram(c, 0) + partition_histogram(c, 1);
    if (partition_size < min_partition_size) {
      small_partitions.push_back(c);
      RAFT_LOG_INFO(
        "ACE: Partition %lu is too small (%lu vectors), will be merged", c, partition_size);
    }
  }

  if (!small_partitions.empty()) {
    RAFT_LOG_INFO("ACE: Found %zu small partitions, merging with closest partitions",
                  small_partitions.size());

    // For each small partition, find the closest non-small partition to merge with
    for (uint64_t small_partition : small_partitions) {
      uint64_t target_partition = small_partition;

      // Find the closest partition that is not small
      // We'll use a simple heuristic: merge with the next partition (modulo n_partitions)
      // that is not also small
      for (uint64_t offset = 1; offset < n_partitions; offset++) {
        uint64_t candidate = (small_partition + offset) % n_partitions;
        uint64_t candidate_size =
          partition_histogram(candidate, 0) + partition_histogram(candidate, 1);

        // Skip if this candidate is also small
        if (candidate_size < min_partition_size) continue;

        target_partition = candidate;
        break;
      }

      // If we couldn't find a non-small partition, merge with the next partition anyway
      if (target_partition == small_partition) {
        target_partition = (small_partition + 1) % n_partitions;
      }

      RAFT_LOG_INFO(
        "ACE: Merging small partition %lu into partition %lu", small_partition, target_partition);

      // Update all labels that point to the small partition
#pragma omp parallel for
      for (uint64_t i = 0; i < dataset_size; i++) {
        if (labels(i, 0) == small_partition) { labels(i, 0) = target_partition; }
        if (labels(i, 1) == small_partition) { labels(i, 1) = target_partition; }
      }

      // Update the partition histogram
      partition_histogram(target_partition, 0) += partition_histogram(small_partition, 0);
      partition_histogram(target_partition, 1) += partition_histogram(small_partition, 1);
      partition_histogram(small_partition, 0) = 0;
      partition_histogram(small_partition, 1) = 0;
    }
  }

  // Create vector lists for each partition
  RAFT_LOG_INFO("ACE: Creating vector lists by partition labels ...");
  auto vector_fwd_list_0 = raft::make_host_vector<IdxT, int64_t>(dataset_size);
  auto vector_fwd_list_1 = raft::make_host_vector<IdxT, int64_t>(dataset_size);
  auto vector_bwd_list_0 = raft::make_host_vector<IdxT, int64_t>(dataset_size);
  auto vector_bwd_list_1 = raft::make_host_vector<IdxT, int64_t>(dataset_size);
  auto idxptr_0          = raft::make_host_vector<IdxT, int64_t>(n_partitions + 1);
  auto idxptr_1          = raft::make_host_vector<IdxT, int64_t>(n_partitions + 1);

  idxptr_0(0) = 0;
  idxptr_1(0) = 0;
  for (uint64_t c = 1; c < n_partitions; c++) {
    idxptr_0(c) = idxptr_0(c - 1) + partition_histogram(c - 1, 0);
    idxptr_1(c) = idxptr_1(c - 1) + partition_histogram(c - 1, 1);
  }

#pragma omp parallel for
  for (uint64_t i = 0; i < dataset_size; i++) {
    uint64_t c_0 = labels(i, 0);
    uint64_t j_0;
#pragma omp atomic capture
    j_0 = idxptr_0(c_0)++;
    RAFT_EXPECTS(j_0 < dataset_size, "Vector ID must be smaller than dataset_size");
    vector_fwd_list_0(i)   = j_0;
    vector_bwd_list_0(j_0) = i;

    uint64_t c_1 = labels(i, 1);
    uint64_t j_1;
#pragma omp atomic capture
    j_1 = idxptr_1(c_1)++;
    RAFT_EXPECTS(j_1 < dataset_size, "Vector ID must be smaller than dataset_size");
    vector_fwd_list_1(i)   = j_1;
    vector_bwd_list_1(j_1) = i;
  }

  // Restore idxptr arrays
  for (uint64_t c = n_partitions + 1; c > 0; c--) {
    idxptr_0(c) = idxptr_0(c - 1);
    idxptr_1(c) = idxptr_1(c - 1);
  }
  idxptr_0(0) = 0;
  idxptr_1(0) = 0;

  auto search_graph_0 = raft::make_host_matrix<IdxT, int64_t>(dataset_size, graph_degree);

  // Process each partition
  for (uint64_t c = 0; c < n_partitions; c++) {
    // Skip partitions that have been merged (empty partitions)
    uint64_t partition_size = partition_histogram(c, 0) + partition_histogram(c, 1);
    if (partition_size == 0) {
      RAFT_LOG_INFO("ACE: Skipping empty partition %lu (merged into another partition)", c);
      continue;
    }

    RAFT_LOG_INFO("ACE: Processing partition %lu/%lu", c + 1, n_partitions);
    auto start = std::chrono::high_resolution_clock::now();

    // Extract vectors for this partition
    uint64_t sub_dataset_size_0 = idxptr_0(c + 1) - idxptr_0(c);
    uint64_t sub_dataset_size_1 = idxptr_1(c + 1) - idxptr_1(c);
    uint64_t sub_dataset_size   = sub_dataset_size_0 + sub_dataset_size_1;
    auto sub_dataset            = raft::make_host_matrix<T, int64_t>(sub_dataset_size, dataset_dim);

    RAFT_LOG_INFO("ACE: Sub-dataset size: %lu (%lu + %lu)",
                  sub_dataset_size,
                  sub_dataset_size_0,
                  sub_dataset_size_1);

    // Copy vectors belonging to this as closest partition
#pragma omp parallel for
    for (uint64_t j = 0; j < sub_dataset_size_0; j++) {
      uint64_t i = vector_bwd_list_0(j + idxptr_0(c));
      RAFT_EXPECTS(labels(i, 0) == c, "Vector does not belong to partition");
      for (uint64_t k = 0; k < dataset_dim; k++) {
        sub_dataset(j, k) = dataset(i, k);
      }
    }

    // Copy vectors belonging to this as 2nd closest partition
#pragma omp parallel for
    for (uint64_t j = 0; j < sub_dataset_size_1; j++) {
      uint64_t i = vector_bwd_list_1(j + idxptr_1(c));
      RAFT_EXPECTS(labels(i, 1) == c, "Vector does not belong to partition");
      for (uint64_t k = 0; k < dataset_dim; k++) {
        sub_dataset(j + sub_dataset_size_0, k) = dataset(i, k);
      }
    }

    // Create index for this partition
    cuvs::neighbors::cagra::index_params sub_index_params;
    sub_index_params.graph_degree              = graph_degree;
    sub_index_params.intermediate_graph_degree = intermediate_degree;
    sub_index_params.guarantee_connectivity    = params.guarantee_connectivity;
    sub_index_params.metric                    = params.metric;
    sub_index_params.graph_build_params        = params.graph_build_params;

    if (std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
          sub_index_params.graph_build_params)) {
      // If IVF-PQ is used, set nprobes and nlists based on the number of vectors in the partition
      // to ensure correct KNN graph construction. Here, we use the same parameters as in hnsw.cpp
      auto ivf_pq_params = cuvs::neighbors::graph_build_params::ivf_pq_params(
        raft::make_extents<int64_t>(sub_dataset_size, dataset_dim), params.metric);
      int ef_construction = 120;  // TODO: Might be a user-specified parameter
      ivf_pq_params.search_params.n_probes =
        std::round(std::sqrt(ivf_pq_params.build_params.n_lists) / 20 + ef_construction / 16);
      sub_index_params.graph_build_params = ivf_pq_params;
      RAFT_LOG_INFO(
        "ACE: IVF-PQ nlists: %u, pq_bits: %u, pq_dim: %u, nprobes: %u, refinement_rate: %.2f",
        ivf_pq_params.build_params.n_lists,
        ivf_pq_params.build_params.pq_bits,
        ivf_pq_params.build_params.pq_dim,
        ivf_pq_params.search_params.n_probes,
        ivf_pq_params.refinement_rate);
    } else if (std::holds_alternative<
                 cuvs::neighbors::cagra::graph_build_params::nn_descent_params>(
                 sub_index_params.graph_build_params)) {
      sub_index_params.graph_build_params =
        cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
          sub_index_params.intermediate_graph_degree, sub_index_params.metric);
      auto nn_descent_params =
        std::get<cuvs::neighbors::cagra::graph_build_params::nn_descent_params>(
          sub_index_params.graph_build_params);
      RAFT_LOG_INFO("ACE: NN-Descent max_iterations: %u, termination_threshold: %u",
                    nn_descent_params.max_iterations,
                    nn_descent_params.termination_threshold);
    } else if (std::holds_alternative<std::monostate>(sub_index_params.graph_build_params)) {
      // Set default build params if not specified
      if (cuvs::neighbors::nn_descent::has_enough_device_memory(
            res, raft::make_extents<int64_t>(sub_dataset_size, dataset_dim), sizeof(IdxT))) {
        sub_index_params.graph_build_params =
          cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
        auto nn_descent_params =
          std::get<cuvs::neighbors::cagra::graph_build_params::nn_descent_params>(
            sub_index_params.graph_build_params);
        RAFT_LOG_INFO(
          "ACE: NN-Descent with default params: max_iterations: %u, termination_threshold: %.2f",
          nn_descent_params.max_iterations,
          nn_descent_params.termination_threshold);
      } else {
        sub_index_params.graph_build_params = cagra::graph_build_params::ivf_pq_params(
          raft::make_extents<int64_t>(sub_dataset_size, dataset_dim), params.metric);
        auto ivf_pq_params = std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
          sub_index_params.graph_build_params);
        RAFT_LOG_INFO(
          "ACE: IVF-PQ with default params: nlists: %u, pq_bits: %u, pq_dim: %u, nprobes: %u, "
          "refinement_rate: %.2f",
          ivf_pq_params.build_params.n_lists,
          ivf_pq_params.build_params.pq_bits,
          ivf_pq_params.build_params.pq_dim,
          ivf_pq_params.search_params.n_probes,
          ivf_pq_params.refinement_rate);
      }
    }

    auto sub_index = cuvs::neighbors::cagra::build(
      res, sub_index_params, raft::make_const_mdspan(sub_dataset.view()));

    // Copy graph edges for core members of this partition
    auto sub_search_graph = raft::make_host_matrix<IdxT, int64_t>(sub_dataset_size_0, graph_degree);
    cudaStream_t stream   = raft::resource::get_cuda_stream(res);
    raft::update_host(sub_search_graph.data_handle(),
                      sub_index.graph().data_handle(),
                      sub_search_graph.size(),
                      stream);
    raft::resource::sync_stream(res, stream);

    // Adjust IDs in sub_search_graph and save to search_graph_0
#pragma omp parallel for
    for (uint64_t i_0 = 0; i_0 < sub_dataset_size_0; i_0++) {
      for (uint64_t k = 0; k < graph_degree; k++) {
        uint64_t j_0 = sub_search_graph(i_0, k);
        RAFT_EXPECTS(j_0 < sub_dataset_size, "Invalid neighbor ID");
        if (j_0 < sub_dataset_size_0) {
          search_graph_0(i_0 + idxptr_0(c), k) = j_0 + idxptr_0(c);
        } else {
          uint64_t j_1 = j_0 - sub_dataset_size_0;
          RAFT_EXPECTS(j_1 < sub_dataset_size_1, "Invalid secondary neighbor ID");
          uint64_t j                           = vector_bwd_list_1(j_1 + idxptr_1(c));
          search_graph_0(i_0 + idxptr_0(c), k) = vector_fwd_list_0(j);
        }
      }
    }

    auto end        = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    RAFT_LOG_INFO(
      "ACE: Partition %lu completed in %.3f sec", c + 1, static_cast<double>(elapsed_ms) / 1000);
  }

  // Convert IDs in search_graph_0 to create final search_graph
  auto search_graph = raft::make_host_matrix<IdxT, int64_t>(dataset_size, graph_degree);
  RAFT_LOG_INFO("ACE: Converting graph IDs to final format ...");
#pragma omp parallel for
  for (uint64_t i = 0; i < dataset_size; i++) {
    uint64_t i_0 = vector_fwd_list_0(i);
    for (uint64_t k = 0; k < graph_degree; k++) {
      uint64_t j_0       = search_graph_0(i_0, k);
      uint64_t j         = vector_bwd_list_0(j_0);
      search_graph(i, k) = j;
    }
  }

  RAFT_LOG_INFO("ACE: Creating final index ...");
  index<T, IdxT> idx(res, params.metric);
  idx.update_graph(res, raft::make_const_mdspan(search_graph.view()));

  if (params.attach_dataset_on_build) {
    try {
      idx.update_dataset(res, dataset);
    } catch (std::bad_alloc& e) {
      RAFT_LOG_WARN(
        "Insufficient GPU memory to attach dataset to ACE index. Only the graph will be stored.");
    } catch (raft::logic_error& e) {
      RAFT_LOG_WARN(
        "Insufficient GPU memory to attach dataset to ACE index. Only the graph will be stored.");
    }
  }

  RAFT_LOG_INFO("ACE: Partitioned CAGRA build completed");
  return idx;
}

template <typename IdxT>
void write_to_graph(raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
                    raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors_host_view,
                    size_t& num_self_included,
                    size_t batch_size,
                    size_t batch_offset)
{
  uint32_t node_degree = knn_graph.extent(1);
  size_t top_k         = neighbors_host_view.extent(1);
  // omit itself & write out
  for (std::size_t i = 0; i < batch_size; i++) {
    size_t vec_idx = i + batch_offset;
    for (std::size_t j = 0, num_added = 0; j < top_k && num_added < node_degree; j++) {
      const auto v = neighbors_host_view(i, j);
      if (static_cast<size_t>(v) == vec_idx) {
        num_self_included++;
        continue;
      }
      knn_graph(vec_idx, num_added) = v;
      num_added++;
    }
  }
}

template <typename DataT, typename IdxT, typename accessor>
void refine_host_and_write_graph(
  raft::resources const& res,
  raft::host_matrix<DataT, int64_t>& queries_host,
  raft::host_matrix<int64_t, int64_t>& neighbors_host,
  raft::host_matrix<int64_t, int64_t>& refined_neighbors_host,
  raft::host_matrix<float, int64_t>& refined_distances_host,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::distance::DistanceType metric,
  size_t& num_self_included,
  size_t batch_size,
  size_t batch_offset,
  int top_k,
  int gpu_top_k)
{
  bool do_refine = top_k != gpu_top_k;

  auto refined_neighbors_host_view = raft::make_host_matrix_view<int64_t, int64_t>(
    do_refine ? refined_neighbors_host.data_handle() : neighbors_host.data_handle(),
    batch_size,
    top_k);

  if (do_refine) {
    // needed for compilation as this routine will also be run for device data with !do_refine
    if constexpr (raft::is_host_mdspan_v<decltype(dataset)>) {
      auto queries_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
        queries_host.data_handle(), batch_size, dataset.extent(1));
      auto neighbors_host_view = raft::make_host_matrix_view<const int64_t, int64_t>(
        neighbors_host.data_handle(), batch_size, neighbors_host.extent(1));
      auto refined_distances_host_view = raft::make_host_matrix_view<float, int64_t>(
        refined_distances_host.data_handle(), batch_size, top_k);
      cuvs::neighbors::refine(res,
                              dataset,
                              queries_host_view,
                              neighbors_host_view,
                              refined_neighbors_host_view,
                              refined_distances_host_view,
                              metric);
    }
  }

  write_to_graph(
    knn_graph, refined_neighbors_host_view, num_self_included, batch_size, batch_offset);
}

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::neighbors::cagra::graph_build_params::ivf_pq_params pq)
{
  RAFT_EXPECTS(pq.build_params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 pq.build_params.metric == cuvs::distance::DistanceType::InnerProduct,
               "Currently only L2Expanded or InnerProduct metric are supported");

  uint32_t node_degree = knn_graph.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::build_graph(%zu, %zu, %u)",
    size_t(dataset.extent(0)),
    size_t(dataset.extent(1)),
    node_degree);

  // Make model name
  const std::string model_name = [&]() {
    char model_name[1024];
    sprintf(model_name,
            "%s-%lux%lu.cluster_%u.pq_%u.%ubit.itr_%u.metric_%u.pqcenter_%u",
            "IVF-PQ",
            static_cast<size_t>(dataset.extent(0)),
            static_cast<size_t>(dataset.extent(1)),
            pq.build_params.n_lists,
            pq.build_params.pq_dim,
            pq.build_params.pq_bits,
            pq.build_params.kmeans_n_iters,
            pq.build_params.metric,
            static_cast<uint32_t>(pq.build_params.codebook_kind));
    return std::string(model_name);
  }();

  RAFT_LOG_DEBUG("# Building IVF-PQ index %s", model_name.c_str());
  auto index = cuvs::neighbors::ivf_pq::build(res, pq.build_params, dataset);

  //
  // search top (k + 1) neighbors
  //

  const auto top_k       = node_degree + 1;
  uint32_t gpu_top_k     = node_degree * pq.refinement_rate;
  gpu_top_k              = std::min<IdxT>(std::max(gpu_top_k, top_k), dataset.extent(0));
  const auto num_queries = dataset.extent(0);

  // Use the same maximum batch size as the ivf_pq::search to avoid allocating more than needed.
  uint32_t max_queries = pq.search_params.max_internal_batch_size;

  // Heuristic: the build_knn_graph code should use only a fraction of the workspace memory; the
  // rest should be used by the ivf_pq::search. Here we say that the workspace size should be a good
  // multiple of what is required for the I/O batching below.
  constexpr size_t kMinWorkspaceRatio = 5;
  constexpr size_t kMinLargeBatchSize = 512;
  auto desired_workspace_size =
    max_queries * (sizeof(DataT) * dataset.extent(1)  // queries (dataset batch)
                   + sizeof(float) * gpu_top_k        // distances
                   + sizeof(int64_t) * gpu_top_k      // neighbors
                   + sizeof(float) * top_k            // refined_distances
                   + sizeof(int64_t) * top_k          // refined_neighbors
                  );
  auto free_space_ratio    = raft::resource::get_workspace_free_bytes(res) / desired_workspace_size;
  bool use_large_workspace = false;
  if (free_space_ratio < kMinWorkspaceRatio) {
    auto adjusted_max_queries =
      static_cast<uint32_t>(max_queries * free_space_ratio / kMinWorkspaceRatio);
    if (adjusted_max_queries >= kMinLargeBatchSize) {
      // adjust max_queries, so that the ratio free_space_ratio gets not larger than
      // kMinWorkspaceRatio.
      RAFT_LOG_INFO(
        "CAGRA graph build: reducing IVF-PQ search max_internal_batch_size from %u -> %u to fit "
        "the workspace",
        max_queries,
        adjusted_max_queries);
      max_queries                              = adjusted_max_queries;
      pq.search_params.max_internal_batch_size = adjusted_max_queries;
    } else {
      // adjusting max_queries to a very small value isn't practical, so we use the large workspace
      // instead.
      use_large_workspace = true;
      RAFT_LOG_WARN(
        "Using large workspace memory for IVF-PQ search during CAGRA graph build. Desired "
        "workspace size: %zu, free workspace size: %zu",
        desired_workspace_size * kMinWorkspaceRatio,
        raft::resource::get_workspace_free_bytes(res));
    }
  }

  // If the workspace is smaller than desired, put the I/O buffers into the large workspace.
  rmm::device_async_resource_ref workspace_mr =
    use_large_workspace ? raft::resource::get_large_workspace_resource(res)
                        : raft::resource::get_workspace_resource(res);

  RAFT_LOG_DEBUG(
    "IVF-PQ search node_degree: %d, top_k: %d,  gpu_top_k: %d,  max_batch_size:: %d, n_probes: %u",
    node_degree,
    top_k,
    gpu_top_k,
    max_queries,
    pq.search_params.n_probes);

  auto distances = raft::make_device_mdarray<float>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, gpu_top_k));
  auto neighbors = raft::make_device_mdarray<int64_t>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, gpu_top_k));
  auto refined_distances = raft::make_device_mdarray<float>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, top_k));
  auto refined_neighbors = raft::make_device_mdarray<int64_t>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, top_k));
  auto neighbors_host = raft::make_host_matrix<int64_t, int64_t>(max_queries, gpu_top_k);
  auto queries_host   = raft::make_host_matrix<DataT, int64_t>(max_queries, dataset.extent(1));
  auto refined_neighbors_host = raft::make_host_matrix<int64_t, int64_t>(max_queries, top_k);
  auto refined_distances_host = raft::make_host_matrix<float, int64_t>(max_queries, top_k);

  // TODO(tfeher): batched search with multiple GPUs
  std::size_t num_self_included = 0;
  bool first                    = true;
  const auto start_clock        = std::chrono::system_clock::now();

  cuvs::spatial::knn::detail::utils::batch_load_iterator<DataT> vec_batches(
    dataset.data_handle(),
    dataset.extent(0),
    dataset.extent(1),
    static_cast<int64_t>(max_queries),
    raft::resource::get_cuda_stream(res),
    workspace_mr);

  size_t next_report_offset = 0;
  size_t d_report_offset    = dataset.extent(0) / 100;  // Report progress in 1% steps.

  bool async_host_processing   = raft::is_host_mdspan_v<decltype(dataset)> || top_k == gpu_top_k;
  size_t previous_batch_size   = 0;
  size_t previous_batch_offset = 0;

  for (const auto& batch : vec_batches) {
    // Map int64_t to uint32_t because ivf_pq requires the latter.
    // TODO(tfeher): remove this mapping once ivf_pq accepts mdspan with int64_t index type
    auto queries_view = raft::make_device_matrix_view<const DataT, uint32_t>(
      batch.data(), batch.size(), batch.row_width());
    auto neighbors_view = raft::make_device_matrix_view<int64_t, uint32_t>(
      neighbors.data_handle(), batch.size(), neighbors.extent(1));
    auto distances_view = raft::make_device_matrix_view<float, uint32_t>(
      distances.data_handle(), batch.size(), distances.extent(1));

    cuvs::neighbors::ivf_pq::search(
      res, pq.search_params, index, queries_view, neighbors_view, distances_view);

    if (async_host_processing) {
      // process previous batch async on host
      // NOTE: the async path also covers disabled refinement (top_k == gpu_top_k)
      if (previous_batch_size > 0) {
        refine_host_and_write_graph(res,
                                    queries_host,
                                    neighbors_host,
                                    refined_neighbors_host,
                                    refined_distances_host,
                                    dataset,
                                    knn_graph,
                                    pq.build_params.metric,
                                    num_self_included,
                                    previous_batch_size,
                                    previous_batch_offset,
                                    top_k,
                                    gpu_top_k);
      }

      // copy next batch to host
      raft::copy(neighbors_host.data_handle(),
                 neighbors.data_handle(),
                 neighbors_view.size(),
                 raft::resource::get_cuda_stream(res));
      if (top_k != gpu_top_k) {
        // can be skipped for disabled refinement
        raft::copy(queries_host.data_handle(),
                   batch.data(),
                   queries_view.size(),
                   raft::resource::get_cuda_stream(res));
      }

      previous_batch_size   = batch.size();
      previous_batch_offset = batch.offset();

      // we need to ensure the copy operations are done prior using the host data
      raft::resource::sync_stream(res);

      // process last batch
      if (previous_batch_offset + previous_batch_size == (size_t)num_queries) {
        refine_host_and_write_graph(res,
                                    queries_host,
                                    neighbors_host,
                                    refined_neighbors_host,
                                    refined_distances_host,
                                    dataset,
                                    knn_graph,
                                    pq.build_params.metric,
                                    num_self_included,
                                    previous_batch_size,
                                    previous_batch_offset,
                                    top_k,
                                    gpu_top_k);
      }
    } else {
      auto neighbor_candidates_view = raft::make_device_matrix_view<const int64_t, uint64_t>(
        neighbors.data_handle(), batch.size(), gpu_top_k);
      auto refined_neighbors_view = raft::make_device_matrix_view<int64_t, int64_t>(
        refined_neighbors.data_handle(), batch.size(), top_k);
      auto refined_distances_view = raft::make_device_matrix_view<float, int64_t>(
        refined_distances.data_handle(), batch.size(), top_k);

      auto dataset_view = raft::make_device_matrix_view<const DataT, int64_t>(
        dataset.data_handle(), dataset.extent(0), dataset.extent(1));
      cuvs::neighbors::refine(res,
                              dataset_view,
                              queries_view,
                              neighbor_candidates_view,
                              refined_neighbors_view,
                              refined_distances_view,
                              pq.build_params.metric);
      raft::copy(refined_neighbors_host.data_handle(),
                 refined_neighbors_view.data_handle(),
                 refined_neighbors_view.size(),
                 raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);

      auto refined_neighbors_host_view = raft::make_host_matrix_view<int64_t, int64_t>(
        refined_neighbors_host.data_handle(), batch.size(), top_k);
      write_to_graph(
        knn_graph, refined_neighbors_host_view, num_self_included, batch.size(), batch.offset());
    }

    size_t num_queries_done = batch.offset() + batch.size();
    const auto end_clock    = std::chrono::system_clock::now();
    if (batch.offset() > next_report_offset) {
      next_report_offset += d_report_offset;
      const auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
        1e-6;
      const auto throughput = num_queries_done / time;

      RAFT_LOG_DEBUG(
        "# Search %12lu / %12lu (%3.2f %%), %e queries/sec, %.2f minutes ETA, self included = "
        "%3.2f %%    \r",
        num_queries_done,
        dataset.extent(0),
        num_queries_done / static_cast<double>(dataset.extent(0)) * 100,
        throughput,
        (num_queries - num_queries_done) / throughput / 60,
        static_cast<double>(num_self_included) / num_queries_done * 100.);
    }
    first = false;
  }

  if (!first) RAFT_LOG_DEBUG("# Finished building kNN graph");
  if (static_cast<double>(num_self_included) / dataset.extent(0) * 100. < 5) {
    RAFT_LOG_WARN(
      "Self-included ratio is low: %2.2f %%. This can lead to poor recall. "
      "Consider using a different configuration for the IVF-PQ index, "
      "increasing the refinement rate, or using higher-precision data type for "
      "LUT/Internal Distance.",
      static_cast<double>(num_self_included) / dataset.extent(0) * 100.);
  }
}

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::neighbors::nn_descent::index_params build_params)
{
  std::optional<raft::host_matrix_view<IdxT, int64_t, row_major>> graph_view = knn_graph;
  auto nn_descent_idx = cuvs::neighbors::nn_descent::build(res, build_params, dataset, graph_view);

  using internal_IdxT = typename std::make_unsigned<IdxT>::type;
  using g_accessor    = typename decltype(nn_descent_idx.graph())::accessor_type;
  using g_accessor_internal =
    raft::host_device_accessor<std::experimental::default_accessor<internal_IdxT>,
                               g_accessor::mem_type>;

  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(nn_descent_idx.graph().data_handle()),
      nn_descent_idx.graph().extent(0),
      nn_descent_idx.graph().extent(1));

  cuvs::neighbors::cagra::detail::graph::sort_knn_graph(
    res, build_params.metric, dataset, knn_graph_internal);
}

template <
  typename IdxT = uint32_t,
  typename g_accessor =
    raft::host_device_accessor<std::experimental::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> new_graph,
  const bool guarantee_connectivity = false)
{
  using internal_IdxT = typename std::make_unsigned<IdxT>::type;

  auto new_graph_internal = raft::make_host_matrix_view<internal_IdxT, int64_t>(
    reinterpret_cast<internal_IdxT*>(new_graph.data_handle()),
    new_graph.extent(0),
    new_graph.extent(1));

  using g_accessor_internal =
    raft::host_device_accessor<std::experimental::default_accessor<internal_IdxT>,
                               raft::memory_type::host>;
  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(knn_graph.data_handle()),
      knn_graph.extent(0),
      knn_graph.extent(1));

  cagra::detail::graph::optimize(
    res, knn_graph_internal, new_graph_internal, guarantee_connectivity);
}

// RAII wrapper for allocating memory with Transparent HugePage
struct mmap_owner {
  // Allocate a new memory (not backed by a file)
  mmap_owner(size_t size) : size_{size}
  {
    int flags = MAP_ANONYMOUS | MAP_PRIVATE;
    ptr_      = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr_ == MAP_FAILED) {
      ptr_ = nullptr;
      throw std::runtime_error("cuvs::mmap_owner error");
    }
    if (madvise(ptr_, size, MADV_HUGEPAGE) != 0) {
      munmap(ptr_, size);
      ptr_ = nullptr;
      throw std::runtime_error("cuvs::mmap_owner error");
    }
  }

  ~mmap_owner() noexcept
  {
    if (ptr_ != nullptr) { munmap(ptr_, size_); }
  }

  // No copies for owning struct
  mmap_owner(const mmap_owner& res)                      = delete;
  auto operator=(const mmap_owner& other) -> mmap_owner& = delete;
  // Moving is fine
  mmap_owner(mmap_owner&& other)
    : ptr_{std::exchange(other.ptr_, nullptr)}, size_{std::exchange(other.size_, 0)}
  {
  }
  auto operator=(mmap_owner&& other) -> mmap_owner&
  {
    std::swap(this->ptr_, other.ptr_);
    std::swap(this->size_, other.size_);
    return *this;
  }

  [[nodiscard]] auto data() const -> void* { return ptr_; }
  [[nodiscard]] auto size() const -> size_t { return size_; }

 private:
  void* ptr_;
  size_t size_;
};

template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
auto iterative_build_graph(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(0, 0);

  // Iteratively improve the accuracy of the graph by repeatedly running
  // CAGRA's search() and optimize(). As for the size of the graph, instead
  // of targeting all nodes from the beginning, the number of nodes is
  // initially small, and the number of nodes is doubled with each iteration.
  RAFT_LOG_INFO("Iteratively creating/improving graph index using CAGRA's search() and optimize()");

  // If dataset is a host matrix, change it to a device matrix. Also, if the
  // dimensionality of the dataset does not meet the alighnemt restriction,
  // add extra dimensions and change it to a strided matrix.
  std::unique_ptr<strided_dataset<T, int64_t>> dev_aligned_dataset;
  try {
    dev_aligned_dataset = make_aligned_dataset(res, dataset);
  } catch (raft::logic_error& e) {
    RAFT_LOG_ERROR("Iterative CAGRA graph build requires the dataset to fit GPU memory");
    throw e;
  }
  auto dev_aligned_dataset_view = dev_aligned_dataset.get()->view();

  // If the matrix stride and extent do no match, the extra dimensions are
  // also as extent since it cannot be used as query matrix.
  auto dev_dataset =
    raft::make_device_matrix_view<const T, int64_t>(dev_aligned_dataset_view.data_handle(),
                                                    dev_aligned_dataset_view.extent(0),
                                                    dev_aligned_dataset_view.stride(0));

  // Determine initial graph size.
  uint64_t final_graph_size   = (uint64_t)dataset.extent(0);
  uint64_t initial_graph_size = (final_graph_size + 1) / 2;
  while (initial_graph_size > graph_degree * 64) {
    initial_graph_size = (initial_graph_size + 1) / 2;
  }
  RAFT_LOG_DEBUG("# initial graph size = %lu", (uint64_t)initial_graph_size);

  // Allocate memory for search results.
  constexpr uint64_t max_chunk_size = 8192;
  auto topk                         = intermediate_degree;
  auto dev_neighbors = raft::make_device_matrix<IdxT, int64_t>(res, max_chunk_size, topk);
  auto dev_distances = raft::make_device_matrix<float, int64_t>(res, max_chunk_size, topk);

  // Determine graph degree and number of search results while increasing
  // graph size.
  auto small_graph_degree = std::max(graph_degree / 2, std::min(graph_degree, (uint64_t)24));
  RAFT_LOG_DEBUG("# small_graph_degree = %lu", (uint64_t)small_graph_degree);
  RAFT_LOG_DEBUG("# graph_degree = %lu", (uint64_t)graph_degree);
  RAFT_LOG_DEBUG("# topk = %lu", (uint64_t)topk);

  // Create an initial graph. The initial graph created here is not suitable for
  // searching, but connectivity is guaranteed.
  auto offset = raft::make_host_vector<IdxT, int64_t>(small_graph_degree);
  for (uint64_t j = 0; j < small_graph_degree; j++) {
    if (j == 0) {
      offset(j) = 1;
    } else {
      offset(j) = offset(j - 1) + 1;
    }
    IdxT ofst = pow((double)(initial_graph_size - 1) / 2, (double)(j + 1) / small_graph_degree);
    if (offset(j) < ofst) { offset(j) = ofst; }
    RAFT_LOG_DEBUG("# offset(%lu) = %lu", (uint64_t)j, (uint64_t)offset(j));
  }
  cagra_graph = raft::make_host_matrix<IdxT, int64_t>(initial_graph_size, small_graph_degree);
  for (uint64_t i = 0; i < initial_graph_size; i++) {
    for (uint64_t j = 0; j < small_graph_degree; j++) {
      cagra_graph(i, j) = (i + offset(j)) % initial_graph_size;
    }
  }

  // Allocate memory for neighbors list using Transparent HugePage
  constexpr size_t thp_size = 2 * 1024 * 1024;
  size_t byte_size          = sizeof(IdxT) * final_graph_size * topk;
  if (byte_size % thp_size) { byte_size += thp_size - (byte_size % thp_size); }
  mmap_owner neighbors_list(byte_size);
  IdxT* neighbors_ptr = (IdxT*)neighbors_list.data();
  memset(neighbors_ptr, 0, byte_size);

  bool flag_last       = false;
  auto curr_graph_size = initial_graph_size;
  while (true) {
    auto start           = std::chrono::high_resolution_clock::now();
    auto curr_query_size = std::min(2 * curr_graph_size, final_graph_size);

    auto next_graph_degree = small_graph_degree;
    if (curr_graph_size == final_graph_size) { next_graph_degree = graph_degree; }

    // The search count (topk) is set to the next graph degree + 1, because
    // pruning is not used except in the last iteration.
    // (*) The appropriate setting for itopk_size requires careful consideration.
    auto curr_topk       = next_graph_degree + 1;
    auto curr_itopk_size = next_graph_degree + 32;
    if (flag_last) {
      curr_topk       = topk;
      curr_itopk_size = curr_topk + 32;
    }

    RAFT_LOG_INFO(
      "# graph_size = %lu (%.3lf), graph_degree = %lu, query_size = %lu, itopk = %lu, topk = %lu",
      (uint64_t)cagra_graph.extent(0),
      (double)cagra_graph.extent(0) / final_graph_size,
      (uint64_t)cagra_graph.extent(1),
      (uint64_t)curr_query_size,
      (uint64_t)curr_itopk_size,
      (uint64_t)curr_topk);

    cuvs::neighbors::cagra::search_params search_params;
    search_params.algo        = cuvs::neighbors::cagra::search_algo::AUTO;
    search_params.max_queries = max_chunk_size;
    search_params.itopk_size  = curr_itopk_size;

    // Create an index (idx), a query view (dev_query_view), and a mdarray for
    // search results (neighbors).
    auto dev_dataset_view = raft::make_device_matrix_view<const T, int64_t>(
      dev_dataset.data_handle(), (int64_t)curr_graph_size, dev_dataset.extent(1));

    auto idx = index<T, IdxT>(
      res, params.metric, dev_dataset_view, raft::make_const_mdspan(cagra_graph.view()));

    auto dev_query_view = raft::make_device_matrix_view<const T, int64_t>(
      dev_dataset.data_handle(), (int64_t)curr_query_size, dev_dataset.extent(1));

    auto neighbors_view =
      raft::make_host_matrix_view<IdxT, int64_t>(neighbors_ptr, curr_query_size, curr_topk);

    // Search.
    // Since there are many queries, divide them into batches and search them.
    cuvs::spatial::knn::detail::utils::batch_load_iterator<T> query_batch(
      dev_query_view.data_handle(),
      curr_query_size,
      dev_query_view.extent(1),
      max_chunk_size,
      raft::resource::get_cuda_stream(res),
      raft::resource::get_workspace_resource(res));
    for (const auto& batch : query_batch) {
      auto batch_dev_query_view = raft::make_device_matrix_view<const T, int64_t>(
        batch.data(), batch.size(), dev_query_view.extent(1));
      auto batch_dev_neighbors_view = raft::make_device_matrix_view<IdxT, int64_t>(
        dev_neighbors.data_handle(), batch.size(), curr_topk);
      auto batch_dev_distances_view = raft::make_device_matrix_view<float, int64_t>(
        dev_distances.data_handle(), batch.size(), curr_topk);

      cuvs::neighbors::cagra::search(res,
                                     search_params,
                                     idx,
                                     batch_dev_query_view,
                                     batch_dev_neighbors_view,
                                     batch_dev_distances_view);

      auto batch_neighbors_view = raft::make_host_matrix_view<IdxT, int64_t>(
        neighbors_view.data_handle() + batch.offset() * curr_topk, batch.size(), curr_topk);
      raft::copy(batch_neighbors_view.data_handle(),
                 batch_dev_neighbors_view.data_handle(),
                 batch_neighbors_view.size(),
                 raft::resource::get_cuda_stream(res));
    }

    // Optimize graph
    auto next_graph_size = curr_query_size;
    cagra_graph          = raft::make_host_matrix<IdxT, int64_t>(0, 0);  // delete existing grahp
    cagra_graph = raft::make_host_matrix<IdxT, int64_t>(next_graph_size, next_graph_degree);
    optimize<IdxT>(
      res, neighbors_view, cagra_graph.view(), flag_last ? params.guarantee_connectivity : 0);

    auto end        = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    RAFT_LOG_INFO("# elapsed time: %.3lf sec", (double)elapsed_ms / 1000);

    if (flag_last) { break; }
    flag_last       = (curr_graph_size == final_graph_size);
    curr_graph_size = next_graph_size;
  }

  return cagra_graph;
}

template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;
  common::nvtx::range<common::nvtx::domain::cuvs> function_scope(
    "cagra::build<%s>(%zu, %zu)",
    Accessor::is_managed_type::value ? "managed"
    : Accessor::is_host_type::value  ? "host"
                                     : "device",
    intermediate_degree,
    graph_degree);
  if (intermediate_degree >= static_cast<size_t>(dataset.extent(0))) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than dataset size, reducing it to %lu",
      dataset.extent(0));
    intermediate_degree = dataset.extent(0) - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  // Set default value in case knn_build_params is not defined.
  auto knn_build_params = params.graph_build_params;
  if (std::holds_alternative<std::monostate>(params.graph_build_params)) {
    // Heuristic to decide default build algo and its params.
    if (cuvs::neighbors::nn_descent::has_enough_device_memory(
          res, dataset.extents(), sizeof(IdxT))) {
      RAFT_LOG_DEBUG("NN descent solver");
      knn_build_params =
        cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
    } else {
      RAFT_LOG_DEBUG("Selecting IVF-PQ solver");
      knn_build_params = cagra::graph_build_params::ivf_pq_params(dataset.extents(), params.metric);
    }
  }
  RAFT_EXPECTS(
    params.metric != BitwiseHamming ||
      std::holds_alternative<cagra::graph_build_params::iterative_search_params>(
        knn_build_params) ||
      std::holds_alternative<cagra::graph_build_params::nn_descent_params>(knn_build_params),
    "IVF_PQ for CAGRA graph build does not support BitwiseHamming as a metric. Please "
    "use nn-descent or the iterative CAGRA search build.");

  // Validate data type for BitwiseHamming metric
  RAFT_EXPECTS(params.metric != cuvs::distance::DistanceType::BitwiseHamming ||
                 (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>),
               "BitwiseHamming distance is only supported for int8_t and uint8_t data types. "
               "Current data type is not supported.");

  auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(0, 0);

  // Dispatch based on graph_build_params
  if (std::holds_alternative<cagra::graph_build_params::iterative_search_params>(
        knn_build_params)) {
    cagra_graph = iterative_build_graph<T, IdxT, Accessor>(res, params, dataset);
  } else {
    std::optional<raft::host_matrix<IdxT, int64_t>> knn_graph(
      raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), intermediate_degree));

    if (std::holds_alternative<cagra::graph_build_params::ivf_pq_params>(knn_build_params)) {
      auto ivf_pq_params =
        std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(knn_build_params);
      if (ivf_pq_params.build_params.metric != params.metric) {
        RAFT_LOG_WARN(
          "Metric (%lu) for IVF-PQ needs to match cagra metric (%lu), "
          "aligning IVF-PQ metric.",
          ivf_pq_params.build_params.metric,
          params.metric);
        ivf_pq_params.build_params.metric = params.metric;
      }
      build_knn_graph(res, dataset, knn_graph->view(), ivf_pq_params);
    } else {
      auto nn_descent_params =
        std::get<cagra::graph_build_params::nn_descent_params>(knn_build_params);

      if (nn_descent_params.metric != params.metric) {
        RAFT_LOG_WARN(
          "Metric (%lu) for nn-descent needs to match cagra metric (%lu), "
          "aligning nn-descent metric.",
          nn_descent_params.metric,
          params.metric);
        nn_descent_params.metric = params.metric;
      }
      if (nn_descent_params.graph_degree != intermediate_degree) {
        RAFT_LOG_WARN(
          "Graph degree (%lu) for nn-descent needs to match cagra intermediate graph degree (%lu), "
          "aligning "
          "nn-descent graph_degree.",
          nn_descent_params.graph_degree,
          intermediate_degree);
        nn_descent_params =
          cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
      }

      // Use nn-descent to build CAGRA knn graph
      nn_descent_params.return_distances = false;
      build_knn_graph<T, IdxT>(res, dataset, knn_graph->view(), nn_descent_params);
    }

    cagra_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), graph_degree);

    RAFT_LOG_TRACE("optimizing graph");
    optimize<IdxT>(res, knn_graph->view(), cagra_graph.view(), params.guarantee_connectivity);

    // free intermediate graph before trying to create the index
    knn_graph.reset();
  }

  RAFT_LOG_TRACE("Graph optimized, creating index");

  // Construct an index from dataset and optimized knn graph.
  if (params.compression.has_value()) {
    RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
                 "VPQ compression is only supported with L2Expanded distance mertric");
    index<T, IdxT> idx(res, params.metric);
    idx.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
    idx.update_dataset(
      res,
      // TODO: hardcoding codebook math to `half`, we can do runtime dispatching later
      cuvs::neighbors::vpq_build<decltype(dataset), half, int64_t>(
        res, *params.compression, dataset));

    return idx;
  }
  if (params.attach_dataset_on_build) {
    try {
      return index<T, IdxT>(
        res, params.metric, dataset, raft::make_const_mdspan(cagra_graph.view()));
    } catch (std::bad_alloc& e) {
      RAFT_LOG_WARN(
        "Insufficient GPU memory to construct CAGRA index with dataset on GPU. Only the graph will "
        "be added to the index");
      // We just add the graph. User is expected to update dataset separately (e.g allocating in
      // managed memory).
    } catch (raft::logic_error& e) {
      // The memory error can also manifest as logic_error.
      RAFT_LOG_WARN(
        "Insufficient GPU memory to construct CAGRA index with dataset on GPU. Only the graph will "
        "be added to the index");
    }
  }
  index<T, IdxT> idx(res, params.metric);
  idx.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
  return idx;
}
}  // namespace cuvs::neighbors::cagra::detail

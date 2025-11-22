/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 8/18/25.
//

#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

class BatchedQueryGatherer {
  // private:

 public:
  // Configuration
  int D;                       // Dimension of each query
  int inner_batch_size;        // Maximum queries per batch
  int max_clusters_per_batch;  // Maximum clusters per batch

  // Device buffers (pre-allocated)
  float* d_gathered_queries = nullptr;  // [inner_batch_size * D]
  int* d_cluster_offsets    = nullptr;  // [max_clusters_per_batch + 1]
  int* d_cluster_ids        = nullptr;  // [max_clusters_per_batch]
  int* d_gather_indices     = nullptr;  // [inner_batch_size]

  // Host pinned buffers for faster transfers
  int* h_gather_indices  = nullptr;  // Pinned memory
  int* h_cluster_offsets = nullptr;  // Pinned memory
  int* h_cluster_ids     = nullptr;  // Pinned memory

  // Batch state
  int current_batch_queries  = 0;
  int current_batch_clusters = 0;
  int start_cluster_idx      = 0;

  // CUDA stream for async operations
  cudaStream_t stream = nullptr;

  // Ctors / Dtor
  BatchedQueryGatherer(int dim,
                       int max_batch_size,
                       int max_clusters          = 1000,
                       cudaStream_t stream_input = nullptr);
  ~BatchedQueryGatherer();

  // Batch control
  void reset_batch();

  // Main processing (kept in header because it's a template)
  template <typename ProcessFunc>
  void process_all_clusters(
    const std::unordered_map<int, std::vector<int>>& cluster_to_queries_simple,
    const float* d_rotated_queries,
    int batch_size,
    ProcessFunc process_func)
  {
    // Sort clusters for consistent processing
    std::vector<std::pair<int, std::vector<int>>> sorted_clusters(cluster_to_queries_simple.begin(),
                                                                  cluster_to_queries_simple.end());
    std::sort(sorted_clusters.begin(), sorted_clusters.end());

    reset_batch();
    start_cluster_idx = 0;

    for (const auto& kv : sorted_clusters) {
      int cluster_id                        = kv.first;
      const std::vector<int>& query_indices = kv.second;

      // Check if adding this cluster would exceed limits
      bool would_overflow =
        (current_batch_queries + static_cast<int>(query_indices.size()) > inner_batch_size) ||
        (current_batch_clusters >= max_clusters_per_batch);

      if (would_overflow && current_batch_queries > 0) {
        // Execute current batch
        execute_batch(d_rotated_queries, batch_size);

        // Process the gathered batch
        process_func(d_gathered_queries,
                     d_cluster_offsets,
                     d_cluster_ids,
                     current_batch_queries,
                     current_batch_clusters,
                     start_cluster_idx);

        // Reset for next batch
        reset_batch();
        start_cluster_idx = cluster_id;
      }

      // Add cluster to current batch if it still fits
      if (current_batch_queries + static_cast<int>(query_indices.size()) <= inner_batch_size &&
          current_batch_clusters < max_clusters_per_batch) {
        add_cluster_to_batch(cluster_id, query_indices);
      }
    }

    // Process final batch
    if (current_batch_queries > 0) {
      execute_batch(d_rotated_queries, batch_size);
      process_func(d_gathered_queries,
                   d_cluster_offsets,
                   d_cluster_ids,
                   current_batch_queries,
                   current_batch_clusters,
                   start_cluster_idx);
    }
  }

  // Add a cluster’s queries into the current batch
  void add_cluster_to_batch(int cluster_id, const std::vector<int>& query_indices);

  // Execute the GPU-side gather for the current batch
  void execute_batch(const float* d_rotated_queries, int batch_size);

  // Getters for batch info
  float* get_gathered_queries() { return d_gathered_queries; }
  int* get_cluster_offsets() { return d_cluster_offsets; }
  int* get_cluster_ids() { return d_cluster_ids; }
  void set_cluster_start_idx(int k) { start_cluster_idx = k; }
  int get_current_batch_queries() const { return current_batch_queries; }
  int get_current_batch_clusters() const { return current_batch_clusters; }
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail

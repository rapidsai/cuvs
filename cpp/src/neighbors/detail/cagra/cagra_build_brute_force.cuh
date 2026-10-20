/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>

#include <algorithm>
#include <optional>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

// Exact kNN construction. Request one extra
// neighbor and remove self by ID, including when duplicate vectors tie with self.
template <typename T, typename IdxT, typename Layout, typename Accessor>
void build_knn_graph(raft::resources const& res,
                     raft::mdspan<const T, raft::matrix_extent<int64_t>, Layout, Accessor> dataset,
                     raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
                     graph_build_params::brute_force_params params)
{
  auto const partition_size            = static_cast<size_t>(dataset.extent(0));
  auto const intermediate_graph_degree = static_cast<size_t>(knn_graph.extent(1));
  RAFT_EXPECTS(knn_graph.extent(0) == dataset.extent(0),
               "CAGRA: kNN graph and dataset must have the same number of rows");
  RAFT_EXPECTS(intermediate_graph_degree > 0 && partition_size > intermediate_graph_degree,
               "CAGRA: kNN graph degree must be positive and smaller than the dataset size");
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
    // Borrow contiguous device data. Host and strided inputs need a contiguous device copy
    // for the brute-force index, which must stay alive across all query batches.
    std::optional<raft::device_matrix<T, int64_t>> device_dataset;
    auto dataset_view = [&]() {
      if constexpr (raft::is_device_mdspan_v<decltype(dataset)>) {
        if (dataset.stride(1) == 1 && dataset.stride(0) == dataset.extent(1)) {
          return raft::make_device_matrix_view<const T, int64_t>(
            dataset.data_handle(), dataset.extent(0), dataset.extent(1));
        }
      }
      device_dataset.emplace(
        raft::make_device_matrix<T, int64_t>(res, partition_size, dataset.extent(1)));
      raft::copy(res, device_dataset->view(), dataset);
      return raft::make_const_mdspan(device_dataset->view());
    }();
    auto brute_force_index =
      cuvs::neighbors::brute_force::build(res, params.build_params, dataset_view);

    // Size result buffers from the full remaining workspace after building the index.
    // Allocate them from the large workspace resource, as in the other graph builders.
    auto const free_bytes     = raft::resource::get_workspace_free_bytes(res);
    auto const top_k          = intermediate_graph_degree + 1;
    auto const bytes_per_row  = top_k * (sizeof(int64_t) + sizeof(float));
    auto const max_batch_size = std::min(partition_size, free_bytes / bytes_per_row);
    RAFT_EXPECTS(max_batch_size > 0,
                 "CAGRA: insufficient workspace for a brute-force kNN query batch");
    auto workspace_mr     = raft::resource::get_large_workspace_resource_ref(res);
    auto device_neighbors = raft::make_device_mdarray<int64_t>(
      res, workspace_mr, raft::make_extents<int64_t>(max_batch_size, top_k));
    auto device_distances = raft::make_device_mdarray<float>(
      res, workspace_mr, raft::make_extents<int64_t>(max_batch_size, top_k));
    auto host_neighbors = raft::make_host_matrix<int64_t, int64_t>(max_batch_size, top_k);

    for (size_t offset = 0; offset < partition_size; offset += max_batch_size) {
      auto const batch_size = std::min(max_batch_size, partition_size - offset);
      auto queries          = raft::make_device_matrix_view<const T, int64_t>(
        dataset_view.data_handle() + offset * dataset.extent(1), batch_size, dataset.extent(1));
      auto neighbors = raft::make_device_matrix_view<int64_t, int64_t>(
        device_neighbors.data_handle(), batch_size, top_k);
      auto distances = raft::make_device_matrix_view<float, int64_t>(
        device_distances.data_handle(), batch_size, top_k);
      auto neighbors_host = raft::make_host_matrix_view<int64_t, int64_t>(
        host_neighbors.data_handle(), batch_size, top_k);
      cuvs::neighbors::brute_force::search(
        res, params.search_params, brute_force_index, queries, neighbors, distances);
      raft::copy(res, neighbors_host, neighbors);
      raft::resource::sync_stream(res);
      for (size_t row = 0; row < batch_size; ++row) {
        size_t output_rank = 0;
        for (size_t rank = 0; rank < top_k; ++rank) {
          auto const neighbor = neighbors_host(row, rank);
          if (neighbor == static_cast<int64_t>(offset + row)) { continue; }
          RAFT_EXPECTS(neighbor >= 0 && static_cast<size_t>(neighbor) < partition_size,
                       "CAGRA: brute-force kNN returned an invalid neighbor");
          knn_graph(offset + row, output_rank++) = static_cast<IdxT>(neighbor);
          if (output_rank == intermediate_graph_degree) { break; }
        }
        RAFT_EXPECTS(output_rank == intermediate_graph_degree,
                     "CAGRA: brute-force kNN did not return enough non-self neighbors");
      }
    }
  } else {
    RAFT_FAIL("CAGRA: brute-force kNN build supports only float and half datasets");
  }
}

}  // namespace cuvs::neighbors::cagra::detail

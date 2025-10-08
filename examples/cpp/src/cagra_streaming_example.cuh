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

#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

#include <thrust/sequence.h>

namespace {

using HostBatch   = raft::host_matrix_view<const float, int64_t, raft::row_major>;
using DeviceBatch = raft::device_matrix_view<const float, int64_t, raft::row_major>;

struct StreamingState {
  StreamingState(raft::device_resources const& res,
                 int64_t dims,
                 int64_t rows_per_batch,
                 int64_t kmeans_rows,
                 cuvs::neighbors::ivf_pq::index_params ivf_params)
    : dims(dims),
      kmeans_rows(kmeans_rows),
      num_rows_staged(0),
      rows_per_batch(rows_per_batch),
      next_id(0),
      ivf_built(false),
      ivf_params(std::move(ivf_params)),
      ivf_index(res),
      kmeans_staging_buffer{raft::make_device_matrix<float, int64_t>(res, kmeans_rows, dims)},
      batch_device_buffer{raft::make_device_matrix<float, int64_t>(res, rows_per_batch, dims)},
      batch_id_buffer{raft::make_device_vector<int64_t, int64_t>(res, rows_per_batch)}
  {
  }

  bool has_ivf_index() const { return ivf_built; }

  void ingest_batch(raft::device_resources const& res, HostBatch batch)
  {
    batches.push_back(batch);
    auto stream = raft::resource::get_cuda_stream(res);

    auto num_rows_to_copy = std::min(batch.extent(0), kmeans_rows - num_rows_staged);
    if (num_rows_to_copy > 0) {
      raft::copy(kmeans_staging_buffer.data_handle() + num_rows_staged * dims,
                 batch.data_handle(),
                 num_rows_to_copy * dims,
                 stream);
      num_rows_staged += num_rows_to_copy;
    }

    if (num_rows_staged >= kmeans_rows) {
      build_initial_index(res);
      // Backfill all staged batches into the IVF-PQ index. This is the first call to extend,
      // hence we can pass nullopt for the IDs and let the function generate sequential IDs.
      auto staged_view = raft::make_device_matrix_view<const float, int64_t>(
        kmeans_staging_buffer.data_handle(), num_rows_staged, dims);
      cuvs::neighbors::ivf_pq::extend(res, staged_view, std::nullopt, &ivf_index);

      next_id += num_rows_staged;
      raft::resource::sync_stream(res, stream);
    }
  }

  void extend_ivf_index(raft::device_resources const& res, HostBatch batch)
  {
    if (!ivf_built) { throw std::runtime_error("IVF-PQ index not built yet"); }

    auto rows   = batch.extent(0);
    auto stream = raft::resource::get_cuda_stream(res);
    auto exec   = raft::resource::get_thrust_policy(res);

    batches.push_back(batch);

    raft::copy(batch_device_buffer.data_handle(), batch.data_handle(), rows * dims, stream);

    auto device_view = raft::make_device_matrix_view<const float, int64_t>(
      batch_device_buffer.data_handle(), rows, dims);

    // Generate sequential IDs for the new rows
    thrust::sequence(
      exec, batch_id_buffer.data_handle(), batch_id_buffer.data_handle() + rows, next_id);

    auto ids_view =
      raft::make_device_vector_view<const int64_t, int64_t>(batch_id_buffer.data_handle(), rows);

    cuvs::neighbors::ivf_pq::extend(res, device_view, ids_view, &ivf_index);
    next_id += rows;

    raft::resource::sync_stream(res, stream);
  }

  auto compute_intermediate_graph(raft::device_resources const& res,
                                  int64_t intermediate_degree,
                                  int64_t top_k) -> raft::host_matrix<uint32_t, int64_t>
  {
    auto total = std::accumulate(
      batches.begin(), batches.end(), int64_t{0}, [](int64_t acc, const auto& batch) {
        return acc + batch.extent(0);
      });

    auto h_knn       = raft::make_host_matrix<uint32_t, int64_t>(total, intermediate_degree);
    auto h_neighbors = raft::make_host_matrix<int64_t, int64_t>(rows_per_batch, top_k);

    auto d_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, rows_per_batch, top_k);
    auto d_distances = raft::make_device_matrix<float, int64_t>(res, rows_per_batch, top_k);

    auto stream        = raft::resource::get_cuda_stream(res);
    int64_t global_row = 0;

    cuvs::neighbors::ivf_pq::search_params search_params{
      .n_probes                = 64,
      .max_internal_batch_size = static_cast<uint32_t>(rows_per_batch),
    };

    for (auto const& batch : batches) {
      int64_t rows = batch.extent(0);
      raft::copy(batch_device_buffer.data_handle(), batch.data_handle(), rows * dims, stream);

      auto query_view = raft::make_device_matrix_view<const float, int64_t>(
        batch_device_buffer.data_handle(), rows, dims);
      auto neighbors_view =
        raft::make_device_matrix_view<int64_t, int64_t>(d_neighbors.data_handle(), rows, top_k);
      auto distances_view =
        raft::make_device_matrix_view<float, int64_t>(d_distances.data_handle(), rows, top_k);

      cuvs::neighbors::ivf_pq::search(
        res, search_params, ivf_index, query_view, neighbors_view, distances_view);

      raft::copy(h_neighbors.data_handle(), d_neighbors.data_handle(), rows * top_k, stream);
      raft::resource::sync_stream(res, stream);

      for (int64_t i = 0; i < rows; ++i) {
        // ivf_pq::search result can possibly include the query point itself,
        // so we skip it when building the k-NN graph. And because the PQ compression
        // does not guarantee exact distance calculation, the query point may not
        // be the closest one in the result. So we scan the whole result to find
        // and skip the query point if found.
        int64_t row_id = global_row + i;
        for (int64_t j = 0, num_added = 0; j < top_k && num_added < intermediate_degree; ++j) {
          auto neighbor = h_neighbors(i, j);
          if (neighbor == row_id) { continue; }
          h_knn(row_id, num_added++) = neighbor;
        }
      }
      global_row += rows;
    }

    return h_knn;
  }

 private:
  void build_initial_index(raft::device_resources const& res)
  {
    if (ivf_built) { return; }
    auto build_view = raft::make_device_matrix_view<const float, int64_t>(
      kmeans_staging_buffer.data_handle(), kmeans_rows, dims);
    cuvs::neighbors::ivf_pq::build(res, ivf_params, build_view, &ivf_index);
    ivf_built = true;
  }

  int64_t dims;
  int64_t kmeans_rows;
  int64_t num_rows_staged;
  int64_t rows_per_batch;
  int64_t next_id;
  bool ivf_built;

  cuvs::neighbors::ivf_pq::index_params ivf_params;
  cuvs::neighbors::ivf_pq::index<int64_t> ivf_index;

  raft::device_matrix<float, int64_t> kmeans_staging_buffer;
  raft::device_matrix<float, int64_t> batch_device_buffer;
  raft::device_vector<int64_t, int64_t> batch_id_buffer;

  std::vector<HostBatch> batches;
};

}  // namespace

/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>

#include <thrust/sequence.h>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/cagra_optimize.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

/*
Streaming CAGRA index build example
-----------------------------------

This example shows how to build a CAGRA graph index by streaming host batches:
1) Stage the first few batches to train an IVF-PQ index.
2) Incrementally extend the IVF-PQ index with every batch.
3) Run IVF-PQ search over the full dataset to form an intermediate k-NN graph.
4) Optimize that graph into the final fixed-degree CAGRA graph.

The dataset is kept on the host the whole time to mimic a scenario where the
full corpus cannot live on the device.
*/

namespace {

void make_host_dataset(raft::host_matrix_view<float, int64_t, raft::row_major> dataset)
{
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int64_t row = 0; row < dataset.extent(0); ++row) {
    for (int64_t col = 0; col < dataset.extent(1); ++col) {
      dataset(row, col) = dist(gen);
    }
  }
}

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

    auto stream = raft::resource::get_cuda_stream(res);
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

void streaming_cagra_build_example(
  raft::device_resources const& res,
  raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  const int64_t total_rows = dataset.extent(0);
  const int64_t dims       = dataset.extent(1);

  const int64_t batch_rows          = 2048;
  const int64_t num_kmeans_batches  = 4;
  const int64_t intermediate_degree = 32;
  const int64_t final_degree        = 16;
  const int64_t top_k               = intermediate_degree + 1;

  const int64_t kmeans_rows = std::min(total_rows, num_kmeans_batches * batch_rows);

  cuvs::neighbors::ivf_pq::index_params ivf_params;
  ivf_params.add_data_on_build        = false;
  ivf_params.kmeans_trainset_fraction = 1.0;
  ivf_params.n_lists                  = 256;
  ivf_params.pq_dim                   = 32;
  ivf_params.pq_bits                  = 8;
  ivf_params.metric                   = cuvs::distance::DistanceType::L2Expanded;

  StreamingState state(res, dims, batch_rows, kmeans_rows, ivf_params);

  std::cout << "Incrementally building the IVF-PQ index... ";

  for (int64_t offset = 0; offset < total_rows; offset += batch_rows) {
    auto rows  = std::min(batch_rows, total_rows - offset);
    auto batch = raft::make_host_matrix_view<const float, int64_t>(
      dataset.data_handle() + offset * dims, rows, dims);

    if (!state.has_ivf_index()) {
      // This will stage the batch for later IVF-PQ training. If
      // enough data has been staged, the initial IVF-PQ index will be built.
      state.ingest_batch(res, batch);
    } else {
      // The IVF-PQ index has been constructed, extend it with the new batch
      state.extend_ivf_index(res, batch);
    }
  }

  // Now we run IVF-PQ search over the full dataset to form an intermediate k-NN graph.
  std::cout << "Done.\n Constructing intermediate kNN graph... ";
  auto h_knn = state.compute_intermediate_graph(res, intermediate_degree, top_k);

  // Optimize that graph into the final fixed-degree CAGRA graph.
  std::cout << "Done.\n Constructing optimized CAGRA graph... ";
  auto h_graph = raft::make_host_matrix<uint32_t, int64_t>(total_rows, final_degree);
  cuvs::neighbors::cagra::helpers::optimize(res, h_knn.view(), h_graph.view());
  std::cout << "Done." << std::endl;

  cuvs::neighbors::cagra::index<float, uint32_t> cagra_index(res);
  cagra_index.update_graph(res, raft::make_const_mdspan(h_graph.view()));
  std::cout << "Final CAGRA graph size [" << cagra_index.graph().extent(0) << ", "
            << cagra_index.graph().extent(1) << "]" << std::endl;
}

int main()
{
  using namespace rmm::mr;

  raft::device_resources res;

  pool_memory_resource<device_memory_resource> pool_mr(get_current_device_resource(),
                                                       1024 * 1024 * 1024ull);
  set_current_device_resource(&pool_mr);

  const int64_t n_samples = 20000;
  const int64_t dims      = 64;

  auto dataset = raft::make_host_matrix<float, int64_t>(n_samples, dims);
  make_host_dataset(dataset.view());

  streaming_cagra_build_example(res, raft::make_const_mdspan(dataset.view()));
}

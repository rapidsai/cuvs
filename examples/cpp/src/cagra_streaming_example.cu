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

#include <algorithm>
#include <cstdint>
#include <iostream>

#include "cagra_streaming_example.cuh"
#include "common.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/cagra_optimize.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_resources.hpp>

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

  const int64_t n_samples  = 20000;
  const int64_t n_features = 64;

  auto dataset = generate_host_dataset(res, n_samples, n_features);

  streaming_cagra_build_example(res, raft::make_const_mdspan(dataset.view()));
}

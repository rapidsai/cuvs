/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"
#include "dlpack/dlpack.h"

void cagra_build_search_simple(
    raft::device_resources const &dev_resources,
    raft::device_matrix_view<const float, int64_t> dataset_mds,
    raft::device_matrix_view<const float, int64_t> queries_mds) {

  int64_t n_rows = dataset_mds.extent(0);
  int64_t n_cols = dataset_mds.extent(1);
  int64_t topk = 12;
  int64_t n_queries = queries_mds.extent(0);

  // Create a cuvsResources_t object
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  cuvsStreamSet(res, raft::resource::get_cuda_stream(dev_resources));

  // Use DLPack to represent `dataset` as a tensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data =
      const_cast<float *>(dataset_mds.data_handle());
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim = 2;
  dataset_tensor.dl_tensor.dtype.code = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits = 32;
  dataset_tensor.dl_tensor.dtype.lanes = 1;
  int64_t dataset_shape[2] = {n_rows, n_cols};
  dataset_tensor.dl_tensor.shape = dataset_shape;
  dataset_tensor.dl_tensor.strides = NULL;

  // Build the CAGRA index
  cuvsCagraIndexParams_t index_params;
  cuvsCagraIndexParamsCreate(&index_params);

  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  cuvsCagraBuild(res, index_params, &dataset_tensor, index);

  // Allocate memory for `neighbors` and `distances` output
  uint32_t *neighbors;
  float *distances;
  cudaMalloc(&neighbors, sizeof(uint32_t) * n_queries * topk);
  cudaMalloc(&distances, sizeof(float) * n_queries * topk);

  // Use DLPack to represent `queries`, `neighbors` and `distances` as tensors
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data =
      const_cast<float *>(queries_mds.data_handle());
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim = 2;
  queries_tensor.dl_tensor.dtype.code = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits = 32;
  queries_tensor.dl_tensor.dtype.lanes = 1;
  int64_t queries_shape[2] = {n_queries, n_cols};
  queries_tensor.dl_tensor.shape = queries_shape;
  queries_tensor.dl_tensor.strides = NULL;

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data = neighbors;
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim = 2;
  neighbors_tensor.dl_tensor.dtype.code = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits = 32;
  neighbors_tensor.dl_tensor.dtype.lanes = 1;
  int64_t neighbors_shape[2] = {n_queries, topk};
  neighbors_tensor.dl_tensor.shape = neighbors_shape;
  neighbors_tensor.dl_tensor.strides = NULL;

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data = distances;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim = 2;
  distances_tensor.dl_tensor.dtype.code = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits = 32;
  distances_tensor.dl_tensor.dtype.lanes = 1;
  int64_t distances_shape[2] = {n_queries, topk};
  distances_tensor.dl_tensor.shape = distances_shape;
  distances_tensor.dl_tensor.strides = NULL;

  // Search the CAGRA index
  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);

  cuvsCagraSearch(res, search_params, index, &queries_tensor, &neighbors_tensor,
                  &distances_tensor);

  // print results
  raft::resource::sync_stream(dev_resources);
  auto neighbors_mds = raft::make_device_matrix_view<uint32_t, int64_t>(
      neighbors, n_queries, topk);
  auto distances_mds =
      raft::make_device_matrix_view<float, int64_t>(distances, n_queries, topk);
  print_results(dev_resources, neighbors_mds, distances_mds);

  // Free or destroy all allocations
  cuvsCagraSearchParamsDestroy(search_params);

  cudaFree(neighbors);
  cudaFree(distances);

  cuvsCagraIndexDestroy(index);
  cuvsCagraIndexParamsDestroy(index_params);
  cuvsResourcesDestroy(res);
}

int main() {
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use
  // the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
      rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used
  // within RAFT algorithms). In that case only the internal arrays would use
  // the pool, any other allocation uses the default RMM memory resource. Here
  // is how to change the workspace memory resource to a pool with 2 GiB upper
  // limit. raft::resource::set_workspace_to_pool_resource(dev_resources, 2 *
  // 1024 * 1024 * 1024ull);

  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim = 90;
  int64_t n_queries = 10;
  auto dataset =
      raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries =
      raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  generate_dataset(dev_resources, dataset.view(), queries.view());
  raft::resource::sync_stream(dev_resources);

  // Simple build and search example.
  cagra_build_search_simple(dev_resources,
                            raft::make_const_mdspan(dataset.view()),
                            raft::make_const_mdspan(queries.view()));
}

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

#include <cuda.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include "neighbors/ann_utils.cuh"
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/tiered_index.h>

#include <gtest/gtest.h>

template <typename T> void generate_random_data(T *devPtr, size_t size) {
  raft::handle_t handle;
  raft::random::RngState r(1234ULL);
  raft::random::uniform(handle, r, devPtr, size, T(0.1), T(2.0));
}

TEST(TieredIndexC, BuildSearchBitsetFiltered) {
  int64_t n_rows = 1000;
  int64_t n_queries = 10;
  int64_t n_dim = 16;
  uint32_t n_neighbors = 10;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  // Create input data
  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);

  generate_random_data(index_data.data(), n_rows * n_dim);
  generate_random_data(query_data.data(), n_queries * n_dim);

  // Create resources
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // Create index params using CAGRA backend
  cuvsTieredIndexParams_t params;
  cuvsTieredIndexParamsCreate(&params);
  params->algo = CUVS_TIERED_INDEX_ALGO_CAGRA;
  params->metric = L2Expanded;
  params->min_ann_rows = 100;
  params->cagra_params = new cuvsCagraIndexParams;
  params->cagra_params->intermediate_graph_degree = 64;
  params->cagra_params->graph_degree = 32;

  // Create DLPack tensor for index data
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data = index_data.data();
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.device.device_id = 0;
  dataset_tensor.dl_tensor.ndim = 2;
  dataset_tensor.dl_tensor.dtype.code = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits = 32;
  dataset_tensor.dl_tensor.dtype.lanes = 1;
  int64_t dataset_shape[2] = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape = dataset_shape;
  dataset_tensor.dl_tensor.strides = nullptr;
  dataset_tensor.dl_tensor.byte_offset = 0;

  // Build index
  cuvsTieredIndex_t index;
  cuvsTieredIndexCreate(&index);
  cuvsError_t build_status =
      cuvsTieredIndexBuild(res, params, &dataset_tensor, index);
  ASSERT_EQ(build_status, CUVS_SUCCESS);

  // Create DLPack tensor for queries
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data = query_data.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.device.device_id = 0;
  queries_tensor.dl_tensor.ndim = 2;
  queries_tensor.dl_tensor.dtype.code = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits = 32;
  queries_tensor.dl_tensor.dtype.lanes = 1;
  int64_t queries_shape[2] = {n_queries, n_dim};
  queries_tensor.dl_tensor.shape = queries_shape;
  queries_tensor.dl_tensor.strides = nullptr;
  queries_tensor.dl_tensor.byte_offset = 0;

  // Create DLPack tensor for neighbors
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data = neighbors_data.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.device.device_id = 0;
  neighbors_tensor.dl_tensor.ndim = 2;
  neighbors_tensor.dl_tensor.dtype.code = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits = 64;
  neighbors_tensor.dl_tensor.dtype.lanes = 1;
  int64_t neighbors_shape[2] = {n_queries, n_neighbors};
  neighbors_tensor.dl_tensor.shape = neighbors_shape;
  neighbors_tensor.dl_tensor.strides = nullptr;
  neighbors_tensor.dl_tensor.byte_offset = 0;

  // Create DLPack tensor for distances
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data = distances_data.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.device.device_id = 0;
  distances_tensor.dl_tensor.ndim = 2;
  distances_tensor.dl_tensor.dtype.code = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits = 32;
  distances_tensor.dl_tensor.dtype.lanes = 1;
  int64_t distances_shape[2] = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape = distances_shape;
  distances_tensor.dl_tensor.strides = nullptr;
  distances_tensor.dl_tensor.byte_offset = 0;

  // Create bitset filter (removes even indices: 0, 2, 4, 6, ...)
  int64_t bitset_size = (n_rows + 31) / 32;
  rmm::device_uvector<uint32_t> removed_indices_bitset(bitset_size, stream);

  // Initialize to 0xAAAAAAAA (binary: 10101010...) to remove even indices
  thrust::fill(rmm::exec_policy(stream), removed_indices_bitset.begin(),
               removed_indices_bitset.end(), 0xAAAAAAAA);

  // Create DLPack tensor for filter
  DLManagedTensor filter_tensor;
  filter_tensor.dl_tensor.data = removed_indices_bitset.data();
  filter_tensor.dl_tensor.device.device_type = kDLCUDA;
  filter_tensor.dl_tensor.device.device_id = 0;
  filter_tensor.dl_tensor.ndim = 1;
  filter_tensor.dl_tensor.dtype.code = kDLUInt;
  filter_tensor.dl_tensor.dtype.bits = 32;
  filter_tensor.dl_tensor.dtype.lanes = 1;
  int64_t filter_shape[1] = {bitset_size};
  filter_tensor.dl_tensor.shape = filter_shape;
  filter_tensor.dl_tensor.strides = nullptr;
  filter_tensor.dl_tensor.byte_offset = 0;

  // Create filter struct
  cuvsFilter filter;
  filter.type = BITSET;
  filter.addr = (uintptr_t)&filter_tensor;

  // Perform search with filter
  cuvsError_t search_status =
      cuvsTieredIndexSearch(res, NULL, index, &queries_tensor,
                            &neighbors_tensor, &distances_tensor, filter);
  ASSERT_EQ(search_status, CUVS_SUCCESS);

  // Verify results - all neighbors should be odd indices
  std::vector<int64_t> neighbors_h(n_queries * n_neighbors);
  raft::copy(neighbors_h.data(), neighbors_data.data(), n_queries * n_neighbors,
             stream);
  raft::resource::sync_stream(handle);

  for (int i = 0; i < n_queries * n_neighbors; i++) {
    ASSERT_TRUE(neighbors_h[i] % 2 == 1)
        << "Found even index " << neighbors_h[i]
        << " but filter should remove all even indices";
  }

  // Cleanup
  delete params->cagra_params;
  cuvsTieredIndexParamsDestroy(params);
  cuvsTieredIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

TEST(TieredIndexC, BuildSearchBitmapFiltered) {
  int64_t n_rows = 1000;
  int64_t n_queries = 10;
  int64_t n_dim = 16;
  uint32_t n_neighbors = 10;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  // Create input data
  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);

  generate_random_data(index_data.data(), n_rows * n_dim);
  generate_random_data(query_data.data(), n_queries * n_dim);

  // Create resources
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // Create index params using CAGRA backend
  cuvsTieredIndexParams_t params;
  cuvsTieredIndexParamsCreate(&params);
  params->algo = CUVS_TIERED_INDEX_ALGO_CAGRA;
  params->metric = L2Expanded;
  params->min_ann_rows = 100;
  params->cagra_params = new cuvsCagraIndexParams;
  params->cagra_params->intermediate_graph_degree = 64;
  params->cagra_params->graph_degree = 32;

  // Create DLPack tensor for index data
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data = index_data.data();
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.device.device_id = 0;
  dataset_tensor.dl_tensor.ndim = 2;
  dataset_tensor.dl_tensor.dtype.code = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits = 32;
  dataset_tensor.dl_tensor.dtype.lanes = 1;
  int64_t dataset_shape[2] = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape = dataset_shape;
  dataset_tensor.dl_tensor.strides = nullptr;
  dataset_tensor.dl_tensor.byte_offset = 0;

  // Build index
  cuvsTieredIndex_t index;
  cuvsTieredIndexCreate(&index);
  cuvsError_t build_status =
      cuvsTieredIndexBuild(res, params, &dataset_tensor, index);
  ASSERT_EQ(build_status, CUVS_SUCCESS);

  // Create DLPack tensor for queries
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data = query_data.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.device.device_id = 0;
  queries_tensor.dl_tensor.ndim = 2;
  queries_tensor.dl_tensor.dtype.code = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits = 32;
  queries_tensor.dl_tensor.dtype.lanes = 1;
  int64_t queries_shape[2] = {n_queries, n_dim};
  queries_tensor.dl_tensor.shape = queries_shape;
  queries_tensor.dl_tensor.strides = nullptr;
  queries_tensor.dl_tensor.byte_offset = 0;

  // Create DLPack tensor for neighbors
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data = neighbors_data.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.device.device_id = 0;
  neighbors_tensor.dl_tensor.ndim = 2;
  neighbors_tensor.dl_tensor.dtype.code = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits = 64;
  neighbors_tensor.dl_tensor.dtype.lanes = 1;
  int64_t neighbors_shape[2] = {n_queries, n_neighbors};
  neighbors_tensor.dl_tensor.shape = neighbors_shape;
  neighbors_tensor.dl_tensor.strides = nullptr;
  neighbors_tensor.dl_tensor.byte_offset = 0;

  // Create DLPack tensor for distances
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data = distances_data.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.device.device_id = 0;
  distances_tensor.dl_tensor.ndim = 2;
  distances_tensor.dl_tensor.dtype.code = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits = 32;
  distances_tensor.dl_tensor.dtype.lanes = 1;
  int64_t distances_shape[2] = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape = distances_shape;
  distances_tensor.dl_tensor.strides = nullptr;
  distances_tensor.dl_tensor.byte_offset = 0;

  // Create bitmap filter (removes even indices for all queries)
  int64_t bitmap_size = n_queries * ((n_rows + 31) / 32);
  rmm::device_uvector<uint32_t> removed_indices_bitmap(bitmap_size, stream);

  // Initialize to 0xAAAAAAAA (binary: 10101010...) to remove even indices
  thrust::fill(rmm::exec_policy(stream), removed_indices_bitmap.begin(),
               removed_indices_bitmap.end(), 0xAAAAAAAA);

  // Create DLPack tensor for filter
  DLManagedTensor filter_tensor;
  filter_tensor.dl_tensor.data = removed_indices_bitmap.data();
  filter_tensor.dl_tensor.device.device_type = kDLCUDA;
  filter_tensor.dl_tensor.device.device_id = 0;
  filter_tensor.dl_tensor.ndim = 1;
  filter_tensor.dl_tensor.dtype.code = kDLUInt;
  filter_tensor.dl_tensor.dtype.bits = 32;
  filter_tensor.dl_tensor.dtype.lanes = 1;
  int64_t filter_shape[1] = {bitmap_size};
  filter_tensor.dl_tensor.shape = filter_shape;
  filter_tensor.dl_tensor.strides = nullptr;
  filter_tensor.dl_tensor.byte_offset = 0;

  // Create filter struct
  cuvsFilter filter;
  filter.type = BITMAP;
  filter.addr = (uintptr_t)&filter_tensor;

  // Perform search with filter
  cuvsError_t search_status =
      cuvsTieredIndexSearch(res, NULL, index, &queries_tensor,
                            &neighbors_tensor, &distances_tensor, filter);
  ASSERT_EQ(search_status, CUVS_SUCCESS);

  // Verify results - all neighbors should be odd indices
  std::vector<int64_t> neighbors_h(n_queries * n_neighbors);
  raft::copy(neighbors_h.data(), neighbors_data.data(), n_queries * n_neighbors,
             stream);
  raft::resource::sync_stream(handle);

  for (int i = 0; i < n_queries * n_neighbors; i++) {
    ASSERT_TRUE(neighbors_h[i] % 2 == 1)
        << "Found even index " << neighbors_h[i]
        << " but filter should remove all even indices";
  }

  // Cleanup
  delete params->cagra_params;
  cuvsTieredIndexParamsDestroy(params);
  cuvsTieredIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

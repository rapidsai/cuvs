/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include "neighbors/ann_utils.cuh"
#include <cuvs/neighbors/ivf_flat.h>

extern "C" void run_ivf_flat(int64_t n_rows,
                             int64_t n_queries,
                             int64_t n_dim,
                             uint32_t n_neighbors,
                             float* index_data,
                             float* query_data,
                             float* distances_data,
                             int64_t* neighbors_data,
                             cuvsDistanceType metric,
                             size_t n_probes,
                             size_t n_lists);

template <typename T>
void generate_random_data(T* devPtr, size_t size)
{
  raft::handle_t handle;
  raft::random::RngState r(1234ULL);
  raft::random::uniform(handle, r, devPtr, size, T(0.1), T(2.0));
};

template <typename T, typename IdxT>
void recall_eval(T* query_data,
                 T* index_data,
                 IdxT* neighbors,
                 T* distances,
                 size_t n_queries,
                 size_t n_rows,
                 size_t n_dim,
                 size_t n_neighbors,
                 cuvsDistanceType metric,
                 size_t n_probes,
                 size_t n_lists)
{
  raft::handle_t handle;
  auto distances_ref = raft::make_device_matrix<T, IdxT>(handle, n_queries, n_neighbors);
  auto neighbors_ref = raft::make_device_matrix<IdxT, IdxT>(handle, n_queries, n_neighbors);
  cuvs::neighbors::naive_knn<T, T, IdxT>(
    handle,
    distances_ref.data_handle(),
    neighbors_ref.data_handle(),
    query_data,
    index_data,
    n_queries,
    n_rows,
    n_dim,
    n_neighbors,
    static_cast<cuvs::distance::DistanceType>((uint16_t)metric));

  size_t size = n_queries * n_neighbors;
  std::vector<IdxT> neighbors_h(size);
  std::vector<T> distances_h(size);
  std::vector<IdxT> neighbors_ref_h(size);
  std::vector<T> distances_ref_h(size);

  auto stream = raft::resource::get_cuda_stream(handle);
  raft::copy(neighbors_h.data(), neighbors, size, stream);
  raft::copy(distances_h.data(), distances, size, stream);
  raft::copy(neighbors_ref_h.data(), neighbors_ref.data_handle(), size, stream);
  raft::copy(distances_ref_h.data(), distances_ref.data_handle(), size, stream);

  // verify output
  double min_recall = static_cast<double>(n_probes) / static_cast<double>(n_lists);
  ASSERT_TRUE(cuvs::neighbors::eval_neighbours(neighbors_ref_h,
                                               neighbors_h,
                                               distances_ref_h,
                                               distances_h,
                                               n_queries,
                                               n_neighbors,
                                               0.001,
                                               min_recall));
};

TEST(IvfFlatC, BuildSearch)
{
  int64_t n_rows       = 8096;
  int64_t n_queries    = 128;
  int64_t n_dim        = 32;
  uint32_t n_neighbors = 8;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvsDistanceType metric = L2Expanded;
  size_t n_probes         = 20;
  size_t n_lists          = 1024;

  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);

  generate_random_data(index_data.data(), n_rows * n_dim);
  generate_random_data(query_data.data(), n_queries * n_dim);

  run_ivf_flat(n_rows,
               n_queries,
               n_dim,
               n_neighbors,
               index_data.data(),
               query_data.data(),
               distances_data.data(),
               neighbors_data.data(),
               metric,
               n_probes,
               n_lists);

  recall_eval(query_data.data(),
              index_data.data(),
              neighbors_data.data(),
              distances_data.data(),
              n_queries,
              n_rows,
              n_dim,
              n_neighbors,
              metric,
              n_probes,
              n_lists);
}
TEST(IvfFlatC, BuildSearchBitsetFiltered)
{
  int64_t n_rows       = 1000;
  int64_t n_queries    = 10;
  int64_t n_dim        = 16;
  uint32_t n_neighbors = 10;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvsDistanceType metric = L2Expanded;
  size_t n_probes         = 10;
  size_t n_lists          = 20;

  // Generate data
  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  generate_random_data(index_data.data(), n_rows * n_dim);
  generate_random_data(query_data.data(), n_queries * n_dim);

  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data.data();
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // create index
  cuvsIvfFlatIndex_t index;
  cuvsIvfFlatIndexCreate(&index);

  // build index
  cuvsIvfFlatIndexParams_t build_params;
  cuvsIvfFlatIndexParamsCreate(&build_params);
  build_params->metric  = metric;
  build_params->n_lists = n_lists;
  cuvsIvfFlatBuild(res, build_params, &dataset_tensor, index);

  // create queries DLTensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = query_data.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, n_dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // create neighbors DLTensor
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_data.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {n_queries, n_neighbors};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // create distances DLTensor
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_data.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  // Create bitset filter - remove every other index
  auto bitset_size = (n_rows + 31) / 32;  // number of uint32_t needed
  rmm::device_uvector<uint32_t> filter_bitset(bitset_size, stream);
  std::vector<uint32_t> filter_bitset_h(bitset_size);
  for (size_t i = 0; i < bitset_size; ++i) {
    filter_bitset_h[i] = 0xAAAAAAAA;  // 10101010... pattern - removes even indices
  }
  raft::copy(filter_bitset.data(), filter_bitset_h.data(), bitset_size, stream);

  DLManagedTensor filter_tensor;
  filter_tensor.dl_tensor.data               = filter_bitset.data();
  filter_tensor.dl_tensor.device.device_type = kDLCUDA;
  filter_tensor.dl_tensor.ndim               = 1;
  filter_tensor.dl_tensor.dtype.code         = kDLUInt;
  filter_tensor.dl_tensor.dtype.bits         = 32;
  filter_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t filter_shape[1]                    = {bitset_size};
  filter_tensor.dl_tensor.shape              = filter_shape;
  filter_tensor.dl_tensor.strides            = NULL;

  cuvsFilter filter;
  filter.type = BITSET;
  filter.addr = (uintptr_t)&filter_tensor;

  // search index with filter
  cuvsIvfFlatSearchParams_t search_params;
  cuvsIvfFlatSearchParamsCreate(&search_params);
  search_params->n_probes = n_probes;
  cuvsIvfFlatSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter);

  // Verify all returned neighbors are odd indices (not filtered out)
  std::vector<int64_t> neighbors_h(n_queries * n_neighbors);
  raft::copy(neighbors_h.data(), neighbors_data.data(), n_queries * n_neighbors, stream);
  raft::resource::sync_stream(handle);

  for (size_t i = 0; i < n_queries * n_neighbors; ++i) {
    // All neighbors should be odd indices (since even indices are filtered)
    ASSERT_TRUE(neighbors_h[i] % 2 == 1 || neighbors_h[i] == -1)
      << "Neighbor at position " << i << " has value " << neighbors_h[i]
      << " which is an even index (should be filtered)";
  }

  // de-allocate index and res
  cuvsIvfFlatSearchParamsDestroy(search_params);
  cuvsIvfFlatIndexParamsDestroy(build_params);
  cuvsIvfFlatIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

TEST(IvfFlatC, BuildSearchBitmapFiltered)
{
  int64_t n_rows       = 1000;
  int64_t n_queries    = 10;
  int64_t n_dim        = 16;
  uint32_t n_neighbors = 10;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvsDistanceType metric = L2Expanded;
  size_t n_probes         = 10;
  size_t n_lists          = 20;

  // Generate data
  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  generate_random_data(index_data.data(), n_rows * n_dim);
  generate_random_data(query_data.data(), n_queries * n_dim);

  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data.data();
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // create index
  cuvsIvfFlatIndex_t index;
  cuvsIvfFlatIndexCreate(&index);

  // build index
  cuvsIvfFlatIndexParams_t build_params;
  cuvsIvfFlatIndexParamsCreate(&build_params);
  build_params->metric  = metric;
  build_params->n_lists = n_lists;
  cuvsIvfFlatBuild(res, build_params, &dataset_tensor, index);

  // create queries DLTensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = query_data.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, n_dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // create neighbors DLTensor
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_data.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {n_queries, n_neighbors};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // create distances DLTensor
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_data.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  // Create bitmap filter - per query filter
  // For each query, remove even indices
  auto bitmap_size = n_queries * ((n_rows + 31) / 32);  // n_queries x (bits for n_rows)
  rmm::device_uvector<uint32_t> filter_bitmap(bitmap_size, stream);
  std::vector<uint32_t> filter_bitmap_h(bitmap_size);
  for (size_t q = 0; q < n_queries; ++q) {
    for (size_t i = 0; i < (n_rows + 31) / 32; ++i) {
      filter_bitmap_h[q * ((n_rows + 31) / 32) + i] =
        0xAAAAAAAA;  // 10101010... pattern - removes even indices
    }
  }
  raft::copy(filter_bitmap.data(), filter_bitmap_h.data(), bitmap_size, stream);

  DLManagedTensor filter_tensor;
  filter_tensor.dl_tensor.data               = filter_bitmap.data();
  filter_tensor.dl_tensor.device.device_type = kDLCUDA;
  filter_tensor.dl_tensor.ndim               = 1;
  filter_tensor.dl_tensor.dtype.code         = kDLUInt;
  filter_tensor.dl_tensor.dtype.bits         = 32;
  filter_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t filter_shape[1]                    = {bitmap_size};
  filter_tensor.dl_tensor.shape              = filter_shape;
  filter_tensor.dl_tensor.strides            = NULL;

  cuvsFilter filter;
  filter.type = BITMAP;
  filter.addr = (uintptr_t)&filter_tensor;

  // search index with bitmap filter
  cuvsIvfFlatSearchParams_t search_params;
  cuvsIvfFlatSearchParamsCreate(&search_params);
  search_params->n_probes = n_probes;
  cuvsIvfFlatSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter);

  // Verify all returned neighbors are odd indices (not filtered out)
  std::vector<int64_t> neighbors_h(n_queries * n_neighbors);
  raft::copy(neighbors_h.data(), neighbors_data.data(), n_queries * n_neighbors, stream);
  raft::resource::sync_stream(handle);

  for (size_t i = 0; i < n_queries * n_neighbors; ++i) {
    // All neighbors should be odd indices (since even indices are filtered)
    ASSERT_TRUE(neighbors_h[i] % 2 == 1 || neighbors_h[i] == -1)
      << "Neighbor at position " << i << " has value " << neighbors_h[i]
      << " which is an even index (should be filtered)";
  }

  // de-allocate index and res
  cuvsIvfFlatSearchParamsDestroy(search_params);
  cuvsIvfFlatIndexParamsDestroy(build_params);
  cuvsIvfFlatIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

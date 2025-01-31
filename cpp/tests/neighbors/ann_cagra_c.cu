/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include <cstddef>
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.hpp>
#include <dlpack/dlpack.h>

#include <cstdint>
#include <cuvs/neighbors/cagra.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/math.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/argmin.cuh>
#include <raft/matrix/linewise_op.cuh>
#include <sys/types.h>

#include <raft/random/make_blobs.cuh>

float dataset[4][2] = {{0.74021935, 0.9209938},
                       {0.03902049, 0.9689629},
                       {0.92514056, 0.4463501},
                       {0.6673192, 0.10993068}};
float queries[4][2] = {{0.48216683, 0.0428398},
                       {0.5084142, 0.6545497},
                       {0.51260436, 0.2643005},
                       {0.05198065, 0.5789965}};

uint32_t filter[1] = {0b1001};  // index 1 and 2 are removed

uint32_t neighbors_exp[4] = {3, 0, 3, 1};
float distances_exp[4]    = {0.03878258, 0.12472608, 0.04776672, 0.15224178};

uint32_t neighbors_exp_filtered[4] = {3, 0, 3, 0};
float distances_exp_filtered[4]    = {0.03878258, 0.12472608, 0.04776672, 0.59063464};

TEST(CagraC, BuildSearch)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {4, 2};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  // create index
  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  // build index
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  cuvsCagraBuild(res, build_params, &dataset_tensor, index);

  // create queries DLTensor
  rmm::device_uvector<float> queries_d(4 * 2, stream);
  raft::copy(queries_d.data(), (float*)queries, 4 * 2, stream);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create neighbors DLTensor
  rmm::device_uvector<uint32_t> neighbors_d(4, stream);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  rmm::device_uvector<float> distances_d(4, stream);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)NULL;

  // search index
  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  cuvsCagraSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter);

  // verify output
  ASSERT_TRUE(
    cuvs::devArrMatchHost(neighbors_exp, neighbors_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(cuvs::devArrMatchHost(
    distances_exp, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  // de-allocate index and res
  cuvsCagraSearchParamsDestroy(search_params);
  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

TEST(CagraC, BuildExtendSearch)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  raft::resources handle;

  const int32_t dimensions = 16;
  // main_data_size needs to be >= 128 (see issue #486)
  const int32_t main_data_size       = 1024;
  const int32_t additional_data_size = 64;
  const int32_t num_queries          = 4;

  // create random data for datasets and queries
  rmm::device_uvector<float> random_data_d(
    (main_data_size + additional_data_size + num_queries) * dimensions, stream);
  rmm::device_uvector<int32_t> random_labels_d(
    (main_data_size + additional_data_size + num_queries) * dimensions, stream);
  raft::random::make_blobs(random_data_d.data(),
                           random_labels_d.data(),
                           main_data_size + additional_data_size + num_queries,
                           dimensions,
                           10,
                           stream);

  // create  dataset DLTensor
  rmm::device_uvector<float> main_d(main_data_size * dimensions, stream);
  rmm::device_uvector<int32_t> main_labels_d(main_data_size, stream);
  raft::copy(main_d.data(), random_data_d.data(), main_data_size * dimensions, stream);
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = main_d.data();
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {main_data_size, dimensions};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  // create additional dataset DLTensor
  rmm::device_uvector<float> additional_d(additional_data_size * dimensions, stream);
  raft::copy(additional_d.data(),
             random_data_d.data() + main_d.size(),
             additional_data_size * dimensions,
             stream);
  DLManagedTensor additional_dataset_tensor;
  additional_dataset_tensor.dl_tensor.data               = additional_d.data();
  additional_dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  additional_dataset_tensor.dl_tensor.ndim               = 2;
  additional_dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  additional_dataset_tensor.dl_tensor.dtype.bits         = 32;
  additional_dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t additional_dataset_shape[2]                    = {additional_data_size, dimensions};
  additional_dataset_tensor.dl_tensor.shape              = additional_dataset_shape;
  additional_dataset_tensor.dl_tensor.strides            = nullptr;

  // create tensor for that points to the extended dataset
  rmm::device_uvector<float> extend_return_d((additional_data_size + main_data_size) * dimensions,
                                             stream);
  DLManagedTensor additional_dataset_return_tensor;
  additional_dataset_return_tensor.dl_tensor.data               = extend_return_d.data();
  additional_dataset_return_tensor.dl_tensor.device.device_type = kDLCUDA;
  additional_dataset_return_tensor.dl_tensor.ndim               = 2;
  additional_dataset_return_tensor.dl_tensor.dtype.code         = kDLFloat;
  additional_dataset_return_tensor.dl_tensor.dtype.bits         = 32;
  additional_dataset_return_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t additional_return_dataset_shape[2] = {additional_data_size + main_data_size, dimensions};
  additional_dataset_return_tensor.dl_tensor.shape   = additional_return_dataset_shape;
  additional_dataset_return_tensor.dl_tensor.strides = nullptr;

  // create index
  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  // build index
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  cuvsCagraBuild(res, build_params, &dataset_tensor, index);

  cuvsStreamSync(res);

  // extend index
  cuvsCagraExtendParams_t extend_params;
  cuvsCagraExtendParamsCreate(&extend_params);
  cuvsCagraExtend(
    res, extend_params, &additional_dataset_tensor, index, &additional_dataset_return_tensor);

  // create queries DLTensor
  rmm::device_uvector<float> queries_d(num_queries * dimensions, stream);
  raft::copy(queries_d.data(),
             random_data_d.data() + (main_data_size + additional_data_size) * dimensions,
             num_queries * dimensions,
             stream);
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, dimensions};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create pairwise distance matrix for dataset and queries
  auto pairwise_distance_dataset_input =
    raft::make_device_matrix<float>(handle, main_data_size + additional_data_size, dimensions);

  raft::copy(pairwise_distance_dataset_input.data_handle(), main_d.data(), main_d.size(), stream);
  raft::copy(pairwise_distance_dataset_input.data_handle() + main_d.size(),
             additional_d.data(),
             additional_d.size(),
             stream);

  auto pairwise_distance_queries_input =
    raft::make_device_matrix<float>(handle, num_queries, dimensions);

  raft::copy(pairwise_distance_queries_input.data_handle(),
             (float*)queries_d.data(),
             num_queries * dimensions,
             stream);

  auto pairwise_distances =
    raft::make_device_matrix<float>(handle, num_queries, (main_data_size + additional_data_size));
  auto metric = cuvs::distance::DistanceType::L2Expanded;

  cuvs::distance::pairwise_distance(handle,
                                    pairwise_distance_queries_input.view(),
                                    pairwise_distance_dataset_input.view(),

                                    pairwise_distances.view(),
                                    metric);

  auto min_cols =
    raft::make_device_vector<uint32_t, uint32_t>(handle, pairwise_distances.extent(0));

  auto distances_const_view = raft::make_device_matrix_view<const float, uint32_t>(
    pairwise_distances.data_handle(), pairwise_distances.extent(0), pairwise_distances.extent(1));

  raft::matrix::argmin(handle, distances_const_view, min_cols.view());

  float min_cols_distances[num_queries];

  for (uint32_t i = 0; i < min_cols.extent(0); i++) {
    uint32_t mc           = min_cols(i);
    min_cols_distances[i] = pairwise_distances(i, mc);
  }

  // create neighbors DLTensor
  rmm::device_uvector<uint32_t> neighbors_d(4, stream);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {num_queries, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  rmm::device_uvector<float> distances_d(4, stream);

  distances_d.resize(4, stream);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {num_queries, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)NULL;

  // search index
  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  cuvsCagraSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter);

  // make sure that extend_return_d points to the extended dataset
  ASSERT_TRUE(cuvs::devArrMatch(
    main_d.data(), extend_return_d.data(), main_d.size(), cuvs::Compare<float>()));

  ASSERT_TRUE(cuvs::devArrMatch(additional_d.data(),
                                extend_return_d.data() + main_d.size(),
                                additional_d.size(),
                                cuvs::Compare<float>()));

  // check neighbors
  ASSERT_TRUE(
    cuvs::devArrMatch(min_cols.data_handle(), neighbors_d.data(), 4, cuvs::Compare<uint32_t>()));

  // check distances
  ASSERT_TRUE(cuvs::devArrMatchHost(
    min_cols_distances, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  // de-allocate index and res
  cuvsCagraSearchParamsDestroy(search_params);
  cuvsCagraExtendParamsDestroy(extend_params);
  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

TEST(CagraC, BuildSearchFiltered)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {4, 2};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  // create index
  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  // build index
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  cuvsCagraBuild(res, build_params, &dataset_tensor, index);

  // create queries DLTensor
  rmm::device_uvector<float> queries_d(4 * 2, stream);
  raft::copy(queries_d.data(), (float*)queries, 4 * 2, stream);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create neighbors DLTensor
  rmm::device_uvector<uint32_t> neighbors_d(4, stream);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  rmm::device_uvector<float> distances_d(4, stream);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  // create filter DLTensor
  rmm::device_uvector<uint32_t> filter_d(1, stream);
  raft::copy(filter_d.data(), filter, 1, stream);

  cuvsFilter filter;

  DLManagedTensor filter_tensor;
  filter_tensor.dl_tensor.data               = filter_d.data();
  filter_tensor.dl_tensor.device.device_type = kDLCUDA;
  filter_tensor.dl_tensor.ndim               = 1;
  filter_tensor.dl_tensor.dtype.code         = kDLUInt;
  filter_tensor.dl_tensor.dtype.bits         = 32;
  filter_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t filter_shape[1]                    = {1};
  filter_tensor.dl_tensor.shape              = filter_shape;
  filter_tensor.dl_tensor.strides            = nullptr;

  filter.type = BITSET;
  filter.addr = (uintptr_t)&filter_tensor;

  // search index
  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  cuvsCagraSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter);
  // verify output
  ASSERT_TRUE(cuvs::devArrMatchHost(
    neighbors_exp_filtered, neighbors_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(cuvs::devArrMatchHost(
    distances_exp_filtered, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  // de-allocate index and res
  cuvsCagraSearchParamsDestroy(search_params);
  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

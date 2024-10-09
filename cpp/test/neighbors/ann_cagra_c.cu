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
#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>

#include <cstdint>
#include <cuvs/neighbors/cagra.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
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

// uint32_t neighbors_exp[4] = {3, 0, 3, 1};
uint32_t neighbors_exp[4] = {0, 1, 2, 3};
// float distances_exp[4]    = {0.03878258, 0.12472608, 0.04776672, 0.15224178};
float distances_exp[4] = {0.0, 0.0, 0.0, 0.0};

TEST(CagraC, BuildSearch)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  //  rmm::device_uvector<float> additional_d(4 * 2, stream);
  // rmm::device_uvector<int32_t> additional_labels_d(4 * 2, stream);
  // raft::random::make_blobs(additional_d.data(), additional_labels_d.data(),16,2,5, stream);

  // // create dataset DLTensor
  // DLManagedTensor dataset_tensor;
  // dataset_tensor.dl_tensor.data               = dataset;
  // dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  // dataset_tensor.dl_tensor.ndim               = 2;
  // dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  // dataset_tensor.dl_tensor.dtype.bits         = 32;
  // dataset_tensor.dl_tensor.dtype.lanes        = 1;
  // int64_t dataset_shape[2]                    = {4, 2};
  // dataset_tensor.dl_tensor.shape              = dataset_shape;
  // dataset_tensor.dl_tensor.strides            = nullptr;

  // create  dataset DLTensor
  int main_num_rows = 1024;
  rmm::device_uvector<float> main_d(main_num_rows * 2, stream);
  rmm::device_uvector<int32_t> main_labels_d(main_num_rows, stream);
  raft::random::make_blobs(main_d.data(), main_labels_d.data(), main_num_rows, 2, 5, stream);
  // raft::copy(additional_d.data(), (float*)dataset, 4 * 2, stream);
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = main_d.data();
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {main_num_rows, 2};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  // create additional dataset DLTensor
  int additional_num_rows = 64;
  rmm::device_uvector<float> additional_d(additional_num_rows * 2, stream);
  rmm::device_uvector<int32_t> additional_labels_d(additional_num_rows, stream);
  raft::random::make_blobs(
    additional_d.data(), additional_labels_d.data(), additional_num_rows, 2, 5, stream);
  raft::copy(additional_d.data(), (float*)dataset, 4 * 2, stream);
  DLManagedTensor additional_dataset_tensor;
  additional_dataset_tensor.dl_tensor.data               = additional_d.data();
  additional_dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  additional_dataset_tensor.dl_tensor.ndim               = 2;
  additional_dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  additional_dataset_tensor.dl_tensor.dtype.bits         = 32;
  additional_dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t additional_dataset_shape[2]                    = {additional_num_rows, 2};
  additional_dataset_tensor.dl_tensor.shape              = additional_dataset_shape;
  additional_dataset_tensor.dl_tensor.strides            = nullptr;

  rmm::device_uvector<float> extend_return_d((additional_num_rows + main_num_rows) * 2, stream);
  DLManagedTensor additional_dataset_return_tensor;
  additional_dataset_return_tensor.dl_tensor.data               = extend_return_d.data();
  additional_dataset_return_tensor.dl_tensor.device.device_type = kDLCUDA;
  additional_dataset_return_tensor.dl_tensor.ndim               = 2;
  additional_dataset_return_tensor.dl_tensor.dtype.code         = kDLFloat;
  additional_dataset_return_tensor.dl_tensor.dtype.bits         = 32;
  additional_dataset_return_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t additional_return_dataset_shape[2]         = {additional_num_rows + main_num_rows, 2};
  additional_dataset_return_tensor.dl_tensor.shape   = additional_return_dataset_shape;
  additional_dataset_return_tensor.dl_tensor.strides = nullptr;

  // create index
  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  // build index
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  cuvsCagraBuild(res, build_params, &dataset_tensor, index);

  cuvsCagraExtendParams_t extend_params;
  cuvsCagraExtendParamsCreate(&extend_params);
  extend_params->max_chunk_size = 100;
  cuvsCagraExtend(
    res, extend_params, &additional_dataset_tensor, &additional_dataset_return_tensor, index);

  extend_return_d.resize(main_num_rows * 2, stream);

  // create queries DLTensor
  rmm::device_uvector<float> queries_d(4 * 2, stream);
  raft::copy(queries_d.data(), (float*)queries, 4 * 2, stream);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = main_d.data();
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

  distances_d.resize(4, stream);

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

  // search index
  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  cuvsCagraSearch(res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor);

  // verify output
  ASSERT_TRUE(
    cuvs::devArrMatchHost(neighbors_exp, neighbors_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(cuvs::devArrMatchHost(
    distances_exp, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  // de-allocate index and res
  cuvsCagraSearchParamsDestroy(search_params);
  cuvsCagraExtendParamsDestroy(extend_params);
  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

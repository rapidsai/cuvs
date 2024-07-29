/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/brute_force.h>
#include <stdint.h>

void run_brute_force(int64_t n_rows,
                     int64_t n_queries,
                     int64_t n_dim,
                     uint32_t n_neighbors,
                     float* index_data,
                     float* query_data,
                     uint32_t* prefilter_data,
                     float* distances_data,
                     int64_t* neighbors_data,
                     cuvsDistanceType metric)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data;
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // create index
  cuvsBruteForceIndex_t index;
  cuvsBruteForceIndexCreate(&index);

  // build index
  cuvsBruteForceBuild(res, &dataset_tensor, metric, 0.0f, index);

  // create queries DLTensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = (void*)query_data;
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, n_dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // create neighbors DLTensor
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = (void*)neighbors_data;
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {n_queries, n_neighbors};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // create distances DLTensor
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = (void*)distances_data;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  cuvsFilter prefilter;

  DLManagedTensor prefilter_tensor;
  if (prefilter_data == NULL) {
    prefilter.type = NO_FILTER;
    prefilter.addr = (uintptr_t)NULL;
  } else {
    prefilter_tensor.dl_tensor.data               = (void*)prefilter_data;
    prefilter_tensor.dl_tensor.device.device_type = kDLCUDA;
    prefilter_tensor.dl_tensor.ndim               = 1;
    prefilter_tensor.dl_tensor.dtype.code         = kDLUInt;
    prefilter_tensor.dl_tensor.dtype.bits         = 32;
    prefilter_tensor.dl_tensor.dtype.lanes        = 1;
    int64_t prefilter_shape[1]                    = {(n_queries * n_rows + 31) / 32};
    prefilter_tensor.dl_tensor.shape              = prefilter_shape;
    prefilter_tensor.dl_tensor.strides            = NULL;

    prefilter.type = BITMAP;
    prefilter.addr = (uintptr_t)&prefilter_tensor;
  }

  // search index
  cuvsBruteForceSearch(
    res, index, &queries_tensor, &neighbors_tensor, &distances_tensor, prefilter);

  // de-allocate index and res
  cuvsBruteForceIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

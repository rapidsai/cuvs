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

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

cuvsResources_t create_resource(int *returnValue) {
  cuvsResources_t cuvsResources;
  *returnValue = cuvsResourcesCreate(&cuvsResources);
  return cuvsResources;
}

DLManagedTensor prepare_tensor(void *data, int64_t shape[], DLDataTypeCode code, long dimensions) {
  DLManagedTensor tensor;

  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = kDLCUDA;
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.dtype.code = code;
  tensor.dl_tensor.dtype.bits = 32;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = NULL;

  return tensor;
}

cuvsCagraIndex_t build_cagra_index(float *dataset, long rows, long dimensions, cuvsResources_t cuvsResources, int *returnValue,
    cuvsCagraIndexParams_t index_params) {

  int64_t dataset_shape[2] = {rows, dimensions};
  DLManagedTensor dataset_tensor = prepare_tensor(dataset, dataset_shape, kDLFloat, dimensions);

  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  *returnValue = cuvsCagraBuild(cuvsResources, index_params, &dataset_tensor, index);
  return index;
}

void serialize_cagra_index(cuvsResources_t cuvsResources, cuvsCagraIndex_t index, int *returnValue, char* filename) {
  *returnValue = cuvsCagraSerialize(cuvsResources, filename, index, true);
}

void deserialize_cagra_index(cuvsResources_t cuvsResources, cuvsCagraIndex_t index, int *rv, char* filename) {
  *rv = cuvsCagraDeserialize(cuvsResources, filename, index);
}

void search_cagra_index(cuvsCagraIndex_t index, float *queries, int topk, long n_queries, int dimensions, 
    cuvsResources_t cuvsResources, int *neighbors_h, float *distances_h, int *returnValue, cuvsCagraSearchParams_t search_params) {

  uint32_t *neighbors;
  float *distances, *queries_d;
  cuvsRMMAlloc(cuvsResources, (void**) &queries_d, sizeof(float) * n_queries * dimensions);
  cuvsRMMAlloc(cuvsResources, (void**) &neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMAlloc(cuvsResources, (void**) &distances, sizeof(float) * n_queries * topk);

  cudaMemcpy(queries_d, queries, sizeof(float) * n_queries * dimensions, cudaMemcpyDefault);

  int64_t queries_shape[2] = {n_queries, dimensions};
  DLManagedTensor queries_tensor = prepare_tensor(queries_d, queries_shape, kDLFloat, dimensions);

  int64_t neighbors_shape[2] = {n_queries, topk};
  DLManagedTensor neighbors_tensor = prepare_tensor(neighbors, neighbors_shape, kDLUInt, dimensions);

  int64_t distances_shape[2] = {n_queries, topk};
  DLManagedTensor distances_tensor = prepare_tensor(distances, distances_shape, kDLFloat, dimensions);

  cuvsCagraSearchParamsCreate(&search_params);

  *returnValue = cuvsCagraSearch(cuvsResources, search_params, index, &queries_tensor, &neighbors_tensor,
                  &distances_tensor);

  cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);
}

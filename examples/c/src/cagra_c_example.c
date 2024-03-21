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

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>

#include <dlpack/dlpack.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

float dataset[4][2] = {{0.74021935, 0.9209938},
                       {0.03902049, 0.9689629},
                       {0.92514056, 0.4463501},
                       {0.6673192, 0.10993068}};
float queries[4][2] = {{0.48216683, 0.0428398},
                       {0.5084142, 0.6545497},
                       {0.51260436, 0.2643005},
                       {0.05198065, 0.5789965}};

void cagra_build_search_simple() {

  int64_t n_rows = 4;
  int64_t n_cols = 2;
  int64_t topk = 2;
  int64_t n_queries = 4;

  // Create a cuvsResources_t object
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // Use DLPack to represent `dataset` as a tensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
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

  // Allocate memory for `queries`, `neighbors` and `distances` output
  uint32_t *neighbors;
  float *distances, *queries_d;
  cuvsRMMAlloc(res, (void**) &queries_d, sizeof(float) * n_queries * n_cols);
  cuvsRMMAlloc(res, (void**) &neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMAlloc(res, (void**) &distances, sizeof(float) * n_queries * topk);

  // Use DLPack to represent `queries`, `neighbors` and `distances` as tensors
  cudaMemcpy(queries_d, queries, sizeof(float) * 4 * 2, cudaMemcpyDefault);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data = queries_d;
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
  uint32_t *neighbors_h =
      (uint32_t *)malloc(sizeof(uint32_t) * n_queries * topk);
  float *distances_h = (float *)malloc(sizeof(float) * n_queries * topk);
  cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * n_queries * topk,
             cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk,
             cudaMemcpyDefault);
  printf("Query 0 neighbor indices: =[%d, %d]\n", neighbors_h[0],
         neighbors_h[1]);
  printf("Query 0 neighbor distances: =[%f, %f]\n", distances_h[0],
         distances_h[1]);

  // Free or destroy all allocations
  free(neighbors_h);
  free(distances_h);

  cuvsCagraSearchParamsDestroy(search_params);

  cuvsRMMFree(res, distances, sizeof(float) * n_queries * topk);
  cuvsRMMFree(res, neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMFree(res, queries_d, sizeof(float) * n_queries * n_cols);

  cuvsCagraIndexDestroy(index);
  cuvsCagraIndexParamsDestroy(index_params);
  cuvsResourcesDestroy(res);
}

int main() {
  // Simple build and search example.
  cagra_build_search_simple();
}

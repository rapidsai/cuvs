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
#include <cuvs/preprocessing/quantize/scalar.h>

#include <dlpack/dlpack.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#define N_ROWS 1000
#define N_COLS 2
float dataset[N_ROWS][N_COLS];

void init_dataset() {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < N_ROWS; ++i) {
        for (int j = 0; j < N_COLS; ++j) {
            dataset[i][j] = ((float)rand() / RAND_MAX);
        }
    }
    dataset[0][0] = 0.74021935; dataset[0][1] = 0.9209938;
    dataset[1][0] = 0.03902049; dataset[1][1] = 0.9689629;
    dataset[2][0] = 0.92514056; dataset[2][1] = 0.4463501;
    dataset[3][0] = 0.6673192;  dataset[3][1] = 0.10993068;
}
float queries[4][2] = {{0.48216683, 0.0428398},
                       {0.5084142, 0.6545497},
                       {0.51260436, 0.2643005},
                       {0.05198065, 0.5789965}};

void cagra_build_search_simple() {
  init_dataset();
  int64_t n_rows = N_ROWS;
  int64_t n_cols = N_COLS;
  int64_t topk = 20;
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
  cuvsRMMAlloc(res, (void **)&queries_d, sizeof(float) * n_queries * n_cols);
  cuvsRMMAlloc(res, (void **)&neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMAlloc(res, (void **)&distances, sizeof(float) * n_queries * topk);

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

  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)NULL;

  cuvsCagraSearch(res, search_params, index, &queries_tensor, &neighbors_tensor,
                  &distances_tensor, filter);

  // print all topk results for each query
  uint32_t *neighbors_h = (uint32_t *)malloc(sizeof(uint32_t) * n_queries * topk);
  float *distances_h = (float *)malloc(sizeof(float) * n_queries * topk);
  cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);
  for (int q = 0; q < n_queries; ++q) {
    printf("Query %d neighbor indices: =[", q);
    for (int k = 0; k < topk; ++k) {
      printf("%d%s", neighbors_h[q * topk + k], (k < topk - 1) ? ", " : "]\n");
    }
    printf("Query %d neighbor distances: =[", q);
    for (int k = 0; k < topk; ++k) {
      printf("%f%s", distances_h[q * topk + k], (k < topk - 1) ? ", " : "]\n");
    }
  }

  // Free or destroy all allocations
  
  free(distances_h);

  cuvsCagraSearchParamsDestroy(search_params);

  cuvsRMMFree(res, distances, sizeof(float) * n_queries * topk);
  cuvsRMMFree(res, neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMFree(res, queries_d, sizeof(float) * n_queries * n_cols);

  // --- Quantization workflow ---
  cuvsScalarQuantizerParams_t quant_params;
  cuvsScalarQuantizerParamsCreate(&quant_params);
  quant_params->quantile = 0.99;

  cuvsScalarQuantizer_t quantizer;
  cuvsScalarQuantizerCreate(&quantizer);

  cuvsScalarQuantizerTrain(res, quant_params, &dataset_tensor, quantizer);

  int8_t *quantized_dataset, *quantized_queries;
  cuvsRMMAlloc(res, (void **)&quantized_dataset, sizeof(int8_t) * n_rows * n_cols);
  cuvsRMMAlloc(res, (void **)&quantized_queries, sizeof(int8_t) * n_queries * n_cols);

  DLManagedTensor quantized_dataset_tensor;
  quantized_dataset_tensor.dl_tensor.data = quantized_dataset;
  quantized_dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  quantized_dataset_tensor.dl_tensor.ndim = 2;
  quantized_dataset_tensor.dl_tensor.dtype.code = kDLInt;
  quantized_dataset_tensor.dl_tensor.dtype.bits = 8;
  quantized_dataset_tensor.dl_tensor.dtype.lanes = 1;
  int64_t quantized_dataset_shape[2] = {n_rows, n_cols};
  quantized_dataset_tensor.dl_tensor.shape = quantized_dataset_shape;
  quantized_dataset_tensor.dl_tensor.strides = NULL;

  DLManagedTensor quantized_queries_tensor;
  quantized_queries_tensor.dl_tensor.data = quantized_queries;
  quantized_queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  quantized_queries_tensor.dl_tensor.ndim = 2;
  quantized_queries_tensor.dl_tensor.dtype.code = kDLInt;
  quantized_queries_tensor.dl_tensor.dtype.bits = 8;
  quantized_queries_tensor.dl_tensor.dtype.lanes = 1;
  int64_t quantized_queries_shape[2] = {n_queries, n_cols};
  quantized_queries_tensor.dl_tensor.shape = quantized_queries_shape;
  quantized_queries_tensor.dl_tensor.strides = NULL;

  cuvsScalarQuantizerTransform(res, quantizer, &dataset_tensor, &quantized_dataset_tensor);
  cuvsScalarQuantizerTransform(res, quantizer, &queries_tensor, &quantized_queries_tensor);

  cuvsCagraIndex_t quant_index;
  cuvsCagraIndexCreate(&quant_index);
  cuvsCagraBuild(res, index_params, &quantized_dataset_tensor, quant_index);

  uint32_t *quant_neighbors;
  float *quant_distances;
  cuvsRMMAlloc(res, (void **)&quant_neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMAlloc(res, (void **)&quant_distances, sizeof(float) * n_queries * topk);

  DLManagedTensor quant_neighbors_tensor;
  quant_neighbors_tensor.dl_tensor.data = quant_neighbors;
  quant_neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  quant_neighbors_tensor.dl_tensor.ndim = 2;
  quant_neighbors_tensor.dl_tensor.dtype.code = kDLUInt;
  quant_neighbors_tensor.dl_tensor.dtype.bits = 32;
  quant_neighbors_tensor.dl_tensor.dtype.lanes = 1;
  int64_t quant_neighbors_shape[2] = {n_queries, topk};
  quant_neighbors_tensor.dl_tensor.shape = quant_neighbors_shape;
  quant_neighbors_tensor.dl_tensor.strides = NULL;

  DLManagedTensor quant_distances_tensor;
  quant_distances_tensor.dl_tensor.data = quant_distances;
  quant_distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  quant_distances_tensor.dl_tensor.ndim = 2;
  quant_distances_tensor.dl_tensor.dtype.code = kDLFloat;
  quant_distances_tensor.dl_tensor.dtype.bits = 32;
  quant_distances_tensor.dl_tensor.dtype.lanes = 1;
  int64_t quant_distances_shape[2] = {n_queries, topk};
  quant_distances_tensor.dl_tensor.shape = quant_distances_shape;
  quant_distances_tensor.dl_tensor.strides = NULL;

  cuvsCagraSearch(res, search_params, quant_index, &quantized_queries_tensor,
                  &quant_neighbors_tensor, &quant_distances_tensor, filter);

  uint32_t *quant_neighbors_h = (uint32_t *)malloc(sizeof(uint32_t) * n_queries * topk);
  float *quant_distances_h = (float *)malloc(sizeof(float) * n_queries * topk);
  cudaMemcpy(quant_neighbors_h, quant_neighbors, sizeof(uint32_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(quant_distances_h, quant_distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);
  for (int q = 0; q < n_queries; ++q) {
    printf("Quantized Query %d neighbor indices: =[", q);
    for (int k = 0; k < topk; ++k) {
      printf("%d%s", quant_neighbors_h[q * topk + k], (k < topk - 1) ? ", " : "]\n");
    }
    printf("Quantized Query %d neighbor distances: =[", q);
    for (int k = 0; k < topk; ++k) {
      printf("%f%s", quant_distances_h[q * topk + k], (k < topk - 1) ? ", " : "]\n");
    }

    // Compare overlap with float32 results
    int overlap = 0;
    printf("Overlap indices with float32 for Query %d: [", q);
    for (int k = 0; k < topk; ++k) {
      int idx_q = quant_neighbors_h[q * topk + k];
      int found = 0;
      for (int k2 = 0; k2 < topk; ++k2) {
        if (neighbors_h[q * topk + k2] == idx_q) {
          found = 1;
          break;
        }
      }
      if (found) {
        if (overlap > 0) printf(", ");
        printf("%d", idx_q);
        overlap++;
      }
    }
    printf("] (count: %d)\n", overlap);
  }

  free(quant_neighbors_h);
  free(quant_distances_h);
  free(neighbors_h);
  cuvsRMMFree(res, quant_distances, sizeof(float) * n_queries * topk);
  cuvsRMMFree(res, quant_neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMFree(res, quantized_dataset, sizeof(int8_t) * n_rows * n_cols);
  cuvsRMMFree(res, quantized_queries, sizeof(int8_t) * n_queries * n_cols);

  cuvsCagraIndexDestroy(quant_index);
  cuvsScalarQuantizerDestroy(quantizer);
  cuvsScalarQuantizerParamsDestroy(quant_params);

  cuvsCagraIndexParamsDestroy(index_params);
  cuvsResourcesDestroy(res);
}

int main() {
  // Simple build and search example.
  cagra_build_search_simple();
}

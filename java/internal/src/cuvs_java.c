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
#include <cuvs/neighbors/brute_force.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define try bool __HadError=false;
#define catch(x) ExitJmp:if(__HadError)
#define throw(x) {__HadError=true;goto ExitJmp;}

/**
 * Create an Initialized opaque C handle
 * 
 * @param return_value return value for cuvsResourcesCreate function call
 */
cuvsResources_t create_resources(int *return_value) {
  cuvsResources_t cuvs_resources;
  *return_value = cuvsResourcesCreate(&cuvs_resources);
  return cuvs_resources;
}

/**
 * Destroy and de-allocate opaque C handle
 * 
 * @param cuvs_resources an opaque C handle
 * @param return_value return value for cuvsResourcesDestroy function call
 */
void destroy_resources(cuvsResources_t cuvs_resources, int *return_value) {
  *return_value = cuvsResourcesDestroy(cuvs_resources);
}

/**
 * Helper function for creating DLManagedTensor instance
 * 
 * @param data the data pointer points to the allocated data
 * @param shape the shape of the tensor
 * @param code the type code of base types
 * @param bits the shape of the tensor
 * @param ndim the number of dimensions
 */
DLManagedTensor prepare_tensor(void *data, int64_t shape[], DLDataTypeCode code, int bits, int ndim) {
  DLManagedTensor tensor;

  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = kDLCUDA;
  tensor.dl_tensor.ndim = ndim;
  tensor.dl_tensor.dtype.code = code;
  tensor.dl_tensor.dtype.bits = bits;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = NULL;

  return tensor;
}

/**
 * Function for building CAGRA index
 * 
 * @param dataset index dataset
 * @param rows number of dataset rows
 * @param dimensions vector dimension of the dataset
 * @param cuvs_resources reference of the underlying opaque C handle
 * @param return_value return value for cuvsCagraBuild function call
 * @param index_params a reference to the index parameters
 * @param compression_params a reference to the compression parameters
 * @param n_writer_threads number of omp threads to use
 */
cuvsCagraIndex_t build_cagra_index(float *dataset, long rows, long dimensions, cuvsResources_t cuvs_resources, int *return_value,
    cuvsCagraIndexParams_t index_params, cuvsCagraCompressionParams_t compression_params, int n_writer_threads) {

  cudaStream_t stream;
  cuvsStreamGet(cuvs_resources, &stream);

  omp_set_num_threads(n_writer_threads);
  cuvsRMMPoolMemoryResourceEnable(95, 95, false);

  int64_t dataset_shape[2] = {rows, dimensions};
  DLManagedTensor dataset_tensor = prepare_tensor(dataset, dataset_shape, kDLFloat, 32, 2);

  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  index_params->compression = compression_params;
  cuvsStreamSync(cuvs_resources);
  *return_value = cuvsCagraBuild(cuvs_resources, index_params, &dataset_tensor, index);

  omp_set_num_threads(1);

  return index;
}

/**
 * A function to de-allocate CAGRA index
 * 
 * @param index cuvsCagraIndex_t to de-allocate
 * @param return_value return value for cuvsCagraIndexDestroy function call
 */
void destroy_cagra_index(cuvsCagraIndex_t index, int *return_value) {
  *return_value = cuvsCagraIndexDestroy(index);
}

/**
 * A function to serialize a CAGRA index
 * 
 * @param cuvs_resources reference of the underlying opaque C handle
 * @param index cuvsCagraIndex_t reference
 * @param return_value return value for cuvsCagraSerialize function call
 * @param filename the filename of the index file
 */
void serialize_cagra_index(cuvsResources_t cuvs_resources, cuvsCagraIndex_t index, int *return_value, char* filename) {
  *return_value = cuvsCagraSerialize(cuvs_resources, filename, index, true);
}

/**
 * A function to de-serialize a CAGRA index
 * 
 * @param cuvs_resources reference to the underlying opaque C handle
 * @param index cuvsCagraIndex_t reference
 * @param return_value return value for cuvsCagraDeserialize function call
 * @param filename the filename of the index file
 */
void deserialize_cagra_index(cuvsResources_t cuvs_resources, cuvsCagraIndex_t index, int *return_value, char* filename) {
  *return_value = cuvsCagraDeserialize(cuvs_resources, filename, index);
}

/**
 * A function to search a CAGRA index and return results
 * 
 * @param index reference to a CAGRA index to search on
 * @param queries query vectors
 * @param topk topK results to return
 * @param n_queries number of queries
 * @param dimensions vector dimension
 * @param cuvs_resources reference to the underlying opaque C handle
 * @param neighbors_h reference to the neighbor results on the host memory
 * @param distances_h reference to the distance results on the host memory
 * @param return_value return value for cuvsCagraSearch function call
 * @param search_params reference to cuvsCagraSearchParams_t holding the search parameters
 */
void search_cagra_index(cuvsCagraIndex_t index, float *queries, int topk, long n_queries, int dimensions, 
    cuvsResources_t cuvs_resources, int *neighbors_h, float *distances_h, int *return_value, cuvsCagraSearchParams_t search_params) {

  cudaStream_t stream;
  cuvsStreamGet(cuvs_resources, &stream);

  uint32_t *neighbors;
  float *distances, *queries_d;
  cuvsRMMAlloc(cuvs_resources, (void**) &queries_d, sizeof(float) * n_queries * dimensions);
  cuvsRMMAlloc(cuvs_resources, (void**) &neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMAlloc(cuvs_resources, (void**) &distances, sizeof(float) * n_queries * topk);

  cudaMemcpy(queries_d, queries, sizeof(float) * n_queries * dimensions, cudaMemcpyDefault);

  int64_t queries_shape[2] = {n_queries, dimensions};
  DLManagedTensor queries_tensor = prepare_tensor(queries_d, queries_shape, kDLFloat, 32, 2);

  int64_t neighbors_shape[2] = {n_queries, topk};
  DLManagedTensor neighbors_tensor = prepare_tensor(neighbors, neighbors_shape, kDLUInt, 32, 2);

  int64_t distances_shape[2] = {n_queries, topk};
  DLManagedTensor distances_tensor = prepare_tensor(distances, distances_shape, kDLFloat, 32, 2);

  cuvsStreamSync(cuvs_resources);
  *return_value = cuvsCagraSearch(cuvs_resources, search_params, index, &queries_tensor, &neighbors_tensor,
                  &distances_tensor);

  cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);

  cuvsRMMFree(cuvs_resources, distances, sizeof(float) * n_queries * topk);
  cuvsRMMFree(cuvs_resources, neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMFree(cuvs_resources, queries_d, sizeof(float) * n_queries * dimensions);
}

/**
 * De-allocate BRUTEFORCE index
 * 
 * @param index reference to BRUTEFORCE index
 * @param return_value return value for cuvsBruteForceIndexDestroy function call
 */
void destroy_brute_force_index(cuvsBruteForceIndex_t index, int *return_value) {
  *return_value = cuvsBruteForceIndexDestroy(index);
}

/**
 * A function to build BRUTEFORCE index
 * 
 * @param dataset the dataset to be indexed
 * @param rows the number of rows in the dataset
 * @param dimensions the vector dimension
 * @param cuvs_resources reference to the underlying opaque C handle
 * @param return_value return value for cuvsBruteForceBuild function call
 * @param n_writer_threads number of threads to use while indexing
 */
cuvsBruteForceIndex_t build_brute_force_index(float *dataset, long rows, long dimensions, cuvsResources_t cuvs_resources,
  int *return_value, int n_writer_threads) {

  omp_set_num_threads(n_writer_threads);
  cuvsRMMPoolMemoryResourceEnable(95, 95, false);

  cudaStream_t stream;
  cuvsStreamGet(cuvs_resources, &stream);

  float *dataset_d;
  cuvsRMMAlloc(cuvs_resources, (void**) &dataset_d, sizeof(float) * rows * dimensions);
  cudaMemcpy(dataset_d, dataset, sizeof(float) * rows * dimensions, cudaMemcpyDefault);

  int64_t dataset_shape[2] = {rows, dimensions};
  DLManagedTensor dataset_tensor = prepare_tensor(dataset_d, dataset_shape, kDLFloat, 32, 2);

  cuvsBruteForceIndex_t index;
  cuvsError_t index_create_status = cuvsBruteForceIndexCreate(&index);

  cuvsStreamSync(cuvs_resources);
  *return_value = cuvsBruteForceBuild(cuvs_resources, &dataset_tensor, L2Expanded, 0.f, index);

  cuvsRMMFree(cuvs_resources, dataset_d, sizeof(float) * rows * dimensions);
  omp_set_num_threads(1);

  return index;
}

/**
 * A function to search the BRUTEFORCE index
 * 
 * @param index reference to a BRUTEFORCE index to search on
 * @param queries reference to query vectors
 * @param topk the top k results to return
 * @param n_queries number of queries
 * @param dimensions vector dimension
 * @param cuvs_resources reference to the underlying opaque C handle
 * @param neighbors_h reference to the neighbor results on the host memory
 * @param distances_h reference to the distance results on the host memory
 * @param return_value return value for cuvsBruteForceSearch function call
 * @param prefilter_data cuvsFilter input prefilter that can be used to filter queries and neighbors based on the given bitmap
 * @param prefilter_data_length prefilter length input
 */
void search_brute_force_index(cuvsBruteForceIndex_t index, float *queries, int topk, long n_queries, int dimensions, 
    cuvsResources_t cuvs_resources, int64_t *neighbors_h, float *distances_h, int *return_value, long *prefilter_data,
    long prefilter_data_length) {

  cudaStream_t stream;
  cuvsStreamGet(cuvs_resources, &stream);

  int64_t *neighbors;
  float *distances, *queries_d;
  long *prefilter_data_d;
  cuvsRMMAlloc(cuvs_resources, (void**) &queries_d, sizeof(float) * n_queries * dimensions);
  cuvsRMMAlloc(cuvs_resources, (void**) &neighbors, sizeof(int64_t) * n_queries * topk);
  cuvsRMMAlloc(cuvs_resources, (void**) &distances, sizeof(float) * n_queries * topk);
  cuvsRMMAlloc(cuvs_resources, (void**) &prefilter_data_d, sizeof(long) * prefilter_data_length);

  cudaMemcpy(queries_d, queries, sizeof(float) * n_queries * dimensions, cudaMemcpyDefault);
  cudaMemcpy(prefilter_data_d, prefilter_data, sizeof(long) * prefilter_data_length, cudaMemcpyDefault);

  int64_t queries_shape[2] = {n_queries, dimensions};
  DLManagedTensor queries_tensor = prepare_tensor(queries_d, queries_shape, kDLFloat, 32, 2);

  int64_t neighbors_shape[2] = {n_queries, topk};
  DLManagedTensor neighbors_tensor = prepare_tensor(neighbors, neighbors_shape, kDLInt, 64, 2);

  int64_t distances_shape[2] = {n_queries, topk};
  DLManagedTensor distances_tensor = prepare_tensor(distances, distances_shape, kDLFloat, 32, 2);

  cuvsFilter prefilter;
  if (prefilter_data == NULL) {
    prefilter.type = NO_FILTER;
    prefilter.addr = (uintptr_t)NULL;
  } else {
    int64_t prefilter_shape[1] = {prefilter_data_length};
    DLManagedTensor prefilter_tensor = prepare_tensor(prefilter_data_d, prefilter_shape, kDLUInt, 32, 1);
    prefilter.type = BITMAP;
    prefilter.addr = (uintptr_t)&prefilter_tensor;
  }

  cuvsStreamSync(cuvs_resources);
  *return_value = cuvsBruteForceSearch(cuvs_resources, index, &queries_tensor, &neighbors_tensor, &distances_tensor, prefilter);

  cudaMemcpy(neighbors_h, neighbors, sizeof(int64_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);

  cuvsRMMFree(cuvs_resources, neighbors, sizeof(int64_t) * n_queries * topk);
  cuvsRMMFree(cuvs_resources, distances, sizeof(float) * n_queries * topk);
  cuvsRMMFree(cuvs_resources, queries_d, sizeof(float) * n_queries * dimensions);
}

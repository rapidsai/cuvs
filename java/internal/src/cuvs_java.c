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

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/hnsw.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define try bool __HadError=false;
#define catch(x) ExitJmp:if(__HadError)
#define throw(x) {__HadError=true;goto ExitJmp;}

/**
 * @brief Create an Initialized opaque C handle
 *
 * @param[out] return_value return value for cuvsResourcesCreate function call
 * @return cuvsResources_t
 */
cuvsResources_t create_resources(int *return_value) {
  cuvsResources_t cuvs_resources;
  *return_value = cuvsResourcesCreate(&cuvs_resources);
  return cuvs_resources;
}

/**
 * @brief Destroy and de-allocate opaque C handle
 *
 * @param[in] cuvs_resources an opaque C handle
 * @param[out] return_value return value for cuvsResourcesDestroy function call
 */
void destroy_resources(cuvsResources_t cuvs_resources, int *return_value) {
  *return_value = cuvsResourcesDestroy(cuvs_resources);
}

/**
 * @brief Helper function for creating DLManagedTensor instance
 *
 * @param[in] data the data pointer points to the allocated data
 * @param[in] shape the shape of the tensor
 * @param[in] code the type code of base types
 * @param[in] bits the shape of the tensor
 * @param[in] ndim the number of dimensions
 * @return DLManagedTensor
 */
DLManagedTensor prepare_tensor(void *data, int64_t shape[], DLDataTypeCode code, int bits, int ndim, DLDeviceType device_type) {
  DLManagedTensor tensor;

  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = device_type; //kDLCUDA;
  tensor.dl_tensor.ndim = ndim;
  tensor.dl_tensor.dtype.code = code;
  tensor.dl_tensor.dtype.bits = bits;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = NULL;

  return tensor;
}

/**
 * @brief Function for building CAGRA index
 *
 * @param[in] dataset index dataset
 * @param[in] rows number of dataset rows
 * @param[in] dimensions vector dimension of the dataset
 * @param[in] cuvs_resources reference of the underlying opaque C handle
 * @param[out] return_value return value for cuvsCagraBuild function call
 * @param[in] index_params a reference to the index parameters
 * @param[in] compression_params a reference to the compression parameters
 * @param[in] n_writer_threads number of omp threads to use
 * @return cuvsCagraIndex_t
 */
cuvsCagraIndex_t build_cagra_index(float *dataset, long rows, long dimensions, cuvsResources_t cuvs_resources, int *return_value,
    cuvsCagraIndexParams_t index_params, cuvsCagraCompressionParams_t compression_params, int n_writer_threads) {

  cudaStream_t stream;
  cuvsStreamGet(cuvs_resources, &stream);

  omp_set_num_threads(n_writer_threads);

  int64_t dataset_shape[2] = {rows, dimensions};
  DLManagedTensor dataset_tensor = prepare_tensor(dataset, dataset_shape, kDLFloat, 32, 2, kDLCUDA);

  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  if (index_params->build_algo == 1) { // when build algo is IVF_PQ
    uint32_t n_lists = index_params->graph_build_params->ivf_pq_build_params->n_lists;
    // As rows cannot be less than n_lists value so trim down.
    index_params->graph_build_params->ivf_pq_build_params->n_lists = rows < n_lists ? rows : n_lists;
  }

  index_params->compression = compression_params;
  cuvsStreamSync(cuvs_resources);
  *return_value = cuvsCagraBuild(cuvs_resources, index_params, &dataset_tensor, index);

  omp_set_num_threads(1);

  return index;
}

/**
 * @brief A function to de-allocate CAGRA index
 *
 * @param[in] index cuvsCagraIndex_t to de-allocate
 * @param[out] return_value return value for cuvsCagraIndexDestroy function call
 */
void destroy_cagra_index(cuvsCagraIndex_t index, int *return_value) {
  *return_value = cuvsCagraIndexDestroy(index);
}

/**
 * @brief A function to serialize a CAGRA index
 *
 * @param[in] cuvs_resources reference of the underlying opaque C handle
 * @param[in] index cuvsCagraIndex_t reference
 * @param[out] return_value return value for cuvsCagraSerialize function call
 * @param[in] filename the filename of the index file
 */
void serialize_cagra_index(cuvsResources_t cuvs_resources, cuvsCagraIndex_t index, int *return_value, char* filename) {
  *return_value = cuvsCagraSerialize(cuvs_resources, filename, index, true);
}

/**
 * @brief A function to de-serialize a CAGRA index
 *
 * @param[in] cuvs_resources reference to the underlying opaque C handle
 * @param[in] index cuvsCagraIndex_t reference
 * @param[out] return_value return value for cuvsCagraDeserialize function call
 * @param[in] filename the filename of the index file
 */
void deserialize_cagra_index(cuvsResources_t cuvs_resources, cuvsCagraIndex_t index, int *return_value, char* filename) {
  *return_value = cuvsCagraDeserialize(cuvs_resources, filename, index);
}

/**
 * @brief A function to search a CAGRA index and return results
 *
 * @param[in] index reference to a CAGRA index to search on
 * @param[in] queries query vectors
 * @param[in] topk topK results to return
 * @param[in] n_queries number of queries
 * @param[in] dimensions vector dimension
 * @param[in] cuvs_resources reference to the underlying opaque C handle
 * @param[out] neighbors_h reference to the neighbor results on the host memory
 * @param[out] distances_h reference to the distance results on the host memory
 * @param[out] return_value return value for cuvsCagraSearch function call
 * @param[in] search_params reference to cuvsCagraSearchParams_t holding the search parameters
 */
void search_cagra_index(cuvsCagraIndex_t index,
                        float *queries,
                        int topk,
                        long n_queries,
                        int dimensions,
                        cuvsResources_t cuvs_resources,
                        int *neighbors_h,
                        float *distances_h,
                        int *return_value,
                        cuvsCagraSearchParams_t search_params,
                        uint32_t *prefilter_data,
                        long prefilter_data_length) {
  cudaStream_t stream;
  cuvsStreamGet(cuvs_resources, &stream);

  uint32_t *neighbors;
  float *distances, *queries_d;

  cuvsRMMAlloc(cuvs_resources, (void **) &queries_d, sizeof(float) * n_queries * dimensions);
  cuvsRMMAlloc(cuvs_resources, (void **) &neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMAlloc(cuvs_resources, (void **) &distances, sizeof(float) * n_queries * topk);

  cudaMemcpy(queries_d, queries, sizeof(float) * n_queries * dimensions, cudaMemcpyDefault);

  int64_t queries_shape[2] = {n_queries, dimensions};
  DLManagedTensor queries_tensor = prepare_tensor(queries_d, queries_shape, kDLFloat, 32, 2, kDLCUDA);

  int64_t neighbors_shape[2] = {n_queries, topk};
  DLManagedTensor neighbors_tensor = prepare_tensor(neighbors, neighbors_shape, kDLUInt, 32, 2, kDLCUDA);

  int64_t distances_shape[2] = {n_queries, topk};
  DLManagedTensor distances_tensor = prepare_tensor(distances, distances_shape, kDLFloat, 32, 2, kDLCUDA);

  cuvsStreamSync(cuvs_resources);

  cuvsFilter filter;
  uint32_t *prefilter_d = NULL;
  int64_t prefilter_len = 0;
  DLManagedTensor *prefilter_tensor_ptr = NULL;

  if (prefilter_data == NULL || prefilter_data_length == 0) {
    filter.type = NO_FILTER;
    filter.addr = (uintptr_t) NULL;
  } else {
    int64_t prefilter_shape[1] = {(prefilter_data_length + 31) / 32};
    prefilter_len = prefilter_shape[0];

    cuvsRMMAlloc(cuvs_resources, (void **) &prefilter_d, sizeof(uint32_t) * prefilter_len);
    cudaMemcpy(prefilter_d, prefilter_data, sizeof(uint32_t) * prefilter_len, cudaMemcpyHostToDevice);

    prefilter_tensor_ptr = (DLManagedTensor *) malloc(sizeof(DLManagedTensor));
    *prefilter_tensor_ptr = prepare_tensor(prefilter_d, prefilter_shape, kDLUInt, 32, 1, kDLCUDA);

    filter.type = BITSET;
    filter.addr = (uintptr_t) prefilter_tensor_ptr;
  }

  *return_value = cuvsCagraSearch(cuvs_resources,
                                  search_params,
                                  index,
                                  &queries_tensor,
                                  &neighbors_tensor,
                                  &distances_tensor,
                                  filter);

  cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);

  cuvsRMMFree(cuvs_resources, distances, sizeof(float) * n_queries * topk);
  cuvsRMMFree(cuvs_resources, neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMFree(cuvs_resources, queries_d, sizeof(float) * n_queries * dimensions);
  if (prefilter_d != NULL) {
    cuvsRMMFree(cuvs_resources, prefilter_d, sizeof(uint32_t) * prefilter_len);
  }
  if (prefilter_tensor_ptr != NULL) {
    free(prefilter_tensor_ptr);
  }
}


/**
 * @brief De-allocate BRUTEFORCE index
 *
 * @param[in] index reference to BRUTEFORCE index
 * @param[out] return_value return value for cuvsBruteForceIndexDestroy function call
 */
void destroy_brute_force_index(cuvsBruteForceIndex_t index, int *return_value) {
  *return_value = cuvsBruteForceIndexDestroy(index);
}

/**
 * @brief A function to build BRUTEFORCE index
 *
 * @param[in] dataset the dataset to be indexed
 * @param[in] rows the number of rows in the dataset
 * @param[in] dimensions the vector dimension
 * @param[in] cuvs_resources reference to the underlying opaque C handle
 * @param[out] return_value return value for cuvsBruteForceBuild function call
 * @param[in] n_writer_threads number of threads to use while indexing
 * @return cuvsBruteForceIndex_t
 */
cuvsBruteForceIndex_t build_brute_force_index(float *dataset, long rows, long dimensions, cuvsResources_t cuvs_resources,
  int *return_value, int n_writer_threads) {

  omp_set_num_threads(n_writer_threads);

  cudaStream_t stream;
  cuvsStreamGet(cuvs_resources, &stream);

  float *dataset_d;
  cuvsRMMAlloc(cuvs_resources, (void**) &dataset_d, sizeof(float) * rows * dimensions);
  cudaMemcpy(dataset_d, dataset, sizeof(float) * rows * dimensions, cudaMemcpyDefault);

  int64_t dataset_shape[2] = {rows, dimensions};
  DLManagedTensor dataset_tensor = prepare_tensor(dataset_d, dataset_shape, kDLFloat, 32, 2, kDLCUDA);

  cuvsBruteForceIndex_t index;
  cuvsError_t index_create_status = cuvsBruteForceIndexCreate(&index);

  cuvsStreamSync(cuvs_resources);
  *return_value = cuvsBruteForceBuild(cuvs_resources, &dataset_tensor, L2Expanded, 0.0f, index);

  omp_set_num_threads(1);

  return index;
}

/**
 * @brief A function to search the BRUTEFORCE index
 *
 * @param[in] index reference to a BRUTEFORCE index to search on
 * @param[in] queries reference to query vectors
 * @param[in] topk the top k results to return
 * @param[in] n_queries number of queries
 * @param[in] dimensions vector dimension
 * @param[in] cuvs_resources reference to the underlying opaque C handle
 * @param[out] neighbors_h reference to the neighbor results on the host memory
 * @param[out] distances_h reference to the distance results on the host memory
 * @param[out] return_value return value for cuvsBruteForceSearch function call
 * @param[in] prefilter_data cuvsFilter input prefilter that can be used to filter queries and neighbors based on the given bitmap
 * @param[in] prefilter_data_length prefilter length input
 * @param[in] n_rows number of rows in the dataset
 */
void search_brute_force_index(cuvsBruteForceIndex_t index, float *queries, int topk, long n_queries, int dimensions,
    cuvsResources_t cuvs_resources, int64_t *neighbors_h, float *distances_h, int *return_value, uint32_t *prefilter_data,
    long prefilter_data_length) {

  cudaStream_t stream;
  cuvsStreamGet(cuvs_resources, &stream);

  int64_t *neighbors;
  float *distances, *queries_d;
  uint32_t *prefilter_d = NULL;
  int64_t prefilter_len = 0;

  cuvsRMMAlloc(cuvs_resources, (void**) &queries_d, sizeof(float) * n_queries * dimensions);
  cuvsRMMAlloc(cuvs_resources, (void**) &neighbors, sizeof(int64_t) * n_queries * topk);
  cuvsRMMAlloc(cuvs_resources, (void**) &distances, sizeof(float) * n_queries * topk);

  cudaMemcpy(queries_d, queries, sizeof(float) * n_queries * dimensions, cudaMemcpyDefault);

  int64_t queries_shape[2] = {n_queries, dimensions};
  DLManagedTensor queries_tensor = prepare_tensor(queries_d, queries_shape, kDLFloat, 32, 2, kDLCUDA);

  int64_t neighbors_shape[2] = {n_queries, topk};
  DLManagedTensor neighbors_tensor = prepare_tensor(neighbors, neighbors_shape, kDLInt, 64, 2, kDLCUDA);

  int64_t distances_shape[2] = {n_queries, topk};
  DLManagedTensor distances_tensor = prepare_tensor(distances, distances_shape, kDLFloat, 32, 2, kDLCUDA);

  cuvsFilter prefilter;
  DLManagedTensor prefilter_tensor;
  if (prefilter_data == NULL) {
    prefilter.type = NO_FILTER;
    prefilter.addr = (uintptr_t)NULL;
  } else {
    // Parse the filters data
    int num_integers = (prefilter_data_length+63)/64 * 2;
    int extraPaddingByteExists = prefilter_data_length % 64 > 32? 0: 1;
    int64_t prefilter_shape[1] = {(prefilter_data_length + 31) / 32};

	prefilter_len = prefilter_shape[0];
    cuvsRMMAlloc(cuvs_resources, (void**) &prefilter_d, sizeof(uint32_t) * prefilter_len);
	cudaMemcpy(prefilter_d, prefilter_data, sizeof(uint32_t) * prefilter_len, cudaMemcpyHostToDevice);

    prefilter_tensor = prepare_tensor(prefilter_d, prefilter_shape, kDLUInt, 32, 1, kDLCUDA);
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
  if(prefilter_d != NULL) {
    cuvsRMMFree(cuvs_resources, prefilter_d, sizeof(uint32_t) * prefilter_len);
  }
}

/**
 * @brief A function to serialize a BRUTEFORCE index
 *
 * @param[in] cuvs_resources reference of the underlying opaque C handle
 * @param[in] index cuvsBruteForceIndex_t reference
 * @param[out] return_value return value for cuvsBruteForceSerialize function call
 * @param[in] filename the filename of the index file
 */
void serialize_brute_force_index(cuvsResources_t cuvs_resources, cuvsBruteForceIndex_t index, int *return_value, char* filename) {
  *return_value = cuvsBruteForceSerialize(cuvs_resources, filename, index);
}

/**
 * @brief A function to de-serialize a BRUTEFORCE index
 *
 * @param[in] cuvs_resources reference to the underlying opaque C handle
 * @param[in] index cuvsBruteForceIndex_t reference
 * @param[out] return_value return value for cuvsBruteForceDeserialize function call
 * @param[in] filename the filename of the index file
 */
void deserialize_brute_force_index(cuvsResources_t cuvs_resources, cuvsBruteForceIndex_t index, int *return_value, char* filename) {
  *return_value = cuvsBruteForceDeserialize(cuvs_resources, filename, index);
}

/**
 * @brief A function to create and serialize an HNSW index from CAGRA index
 *
 * @param[in] cuvs_resources reference to the underlying opaque C handle
 * @param[in] file_path the path to the file of the created HNSW index
 * @param[in] index cuvsCagraIndex_t reference to the existing CAGRA index
 * @param[out] return_value return value for cuvsCagraSerializeToHnswlib function call
 */
void serialize_cagra_index_to_hnsw(cuvsResources_t cuvs_resources, char *file_path, cuvsCagraIndex_t index, int *return_value) {
  *return_value = cuvsCagraSerializeToHnswlib(cuvs_resources, file_path, index);
}

/**
 * @brief A function to deserialize the persisted HNSW index
 *
 * @param[in] cuvs_resources reference to the underlying opaque C handle
 * @param[in] file_path the path to the persisted HNSW index file
 * @param[in] hnsw_params reference to the HNSW index params
 * @param[out] return_value return value for cuvsHnswDeserialize function call
 * @param[in] vector_dimension the dimension of the vectors in the HNSW index
 * @returns cuvsHnswIndex_t reference to the created HNSW index
 */
cuvsHnswIndex_t deserialize_hnsw_index(cuvsResources_t cuvs_resources, char *file_path,
  cuvsHnswIndexParams_t hnsw_params, int *return_value, int vector_dimension) {
  cuvsHnswIndex_t hnsw_index;
  cuvsError_t rv = cuvsHnswIndexCreate(&hnsw_index);
  hnsw_index->dtype.bits = 32;
  hnsw_index->dtype.code = kDLFloat;
  hnsw_index->dtype.lanes = 1;
  *return_value = cuvsHnswDeserialize(cuvs_resources, hnsw_params, file_path, vector_dimension, L2Expanded, hnsw_index);
  return hnsw_index;
}

/**
 * @brief A Function to search in the HNSW index
 *
 * @param[in] cuvs_resources reference to the underlying opaque C handle
 * @param[in] hnsw_index the HNSW index reference
 * @param[in] search_params reference to the HNSW search parameters
 * @param[out] return_value return value for cuvsHnswSearch function call
 * @param[out] neighbors_h result container on host holding the neighbor ids
 * @param[out] distances_h result container on host holding the distances
 * @param[in] queries reference to the queries
 * @param[in] topk the top k results to return
 * @param[in] query_dimension the dimension of the query vectors
 * @param[in] n_queries the number of queries passed to the function
 */
void search_hnsw_index(cuvsResources_t cuvs_resources, cuvsHnswIndex_t hnsw_index, cuvsHnswSearchParams_t search_params,
  int *return_value, uint64_t *neighbors_h, float *distances_h, float *queries, int topk, int query_dimension, int n_queries) {

  int64_t queries_shape[2] = {n_queries, query_dimension};
  DLManagedTensor queries_tensor = prepare_tensor(queries, queries_shape, kDLFloat, 32, 2, kDLCPU);

  int64_t neighbors_shape[2] = {n_queries, topk};
  DLManagedTensor neighbors_tensor = prepare_tensor(neighbors_h, neighbors_shape, kDLUInt, 64, 2, kDLCPU);

  int64_t distances_shape[2] = {n_queries, topk};
  DLManagedTensor distances_tensor = prepare_tensor(distances_h, distances_shape, kDLFloat, 32, 2, kDLCPU);

  *return_value = cuvsHnswSearch(
    cuvs_resources, search_params, hnsw_index, &queries_tensor, &neighbors_tensor, &distances_tensor);
}

/**
 * @brief A function to destroy the HNSW index
 *
 * @param[in] hnsw_index the HNSW index reference
 * @param[out] return_value return value for cuvsHnswIndexDestroy function call
 */
void destroy_hnsw_index(cuvsHnswIndex_t hnsw_index, int *return_value) {
  *return_value = cuvsHnswIndexDestroy(hnsw_index);
}

/**
 * @brief struct for containing gpu information
 */
typedef struct gpuInfo {
  int gpu_id;
  char name[256];
  long free_memory;
  long total_memory;
  float compute_capability;
} gpuInfo;

/**
 * @brief A function to get GPU details
 *
 * @param[out] return_value return value for cudaMemGetInfo function call
 * @param[out] num_gpus the number of devices found
 * @param[out] gpu_info_arr reference to the array of gpuInfo objects
 */
void get_gpu_info(int *return_value, int *num_gpus, gpuInfo *gpu_info_arr) {
  cudaGetDeviceCount(num_gpus);
  // Limiting the num_gpus to 1024. For more details please see comments in Util.availableGPUs()
  *num_gpus = (*num_gpus > 1024) ? 1024 : *num_gpus;
  struct gpuInfo gpuInfos[*num_gpus];
  size_t free, total;
  // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp
  struct cudaDeviceProp deviceProp;
  for (int i = 0; i < *num_gpus; i++) {
    cudaSetDevice(i);
    cudaGetDeviceProperties(&deviceProp, i);
    char buffer[10];
    sprintf(buffer, "%d.%d", deviceProp.major, deviceProp.minor);
    *return_value = cudaMemGetInfo(&free, &total);
    gpuInfos[i].gpu_id = i;
    strcpy(gpuInfos[i].name, deviceProp.name);
    gpuInfos[i].free_memory = free;
    gpuInfos[i].total_memory = total;
    gpuInfos[i].compute_capability = atof(buffer);
    *(gpu_info_arr + i) = gpuInfos[i];
  }
}

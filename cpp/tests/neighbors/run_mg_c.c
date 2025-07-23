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
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/mg_cagra.h>
#include <cuvs/neighbors/mg_ivf_flat.h>
#include <cuvs/neighbors/mg_ivf_pq.h>
#include <stdio.h>

// Forward declarations for multi-GPU resource functions
extern cuvsError_t cuvsMgResourcesCreate(cuvsResources_t* res);
extern cuvsError_t cuvsMgResourcesDestroy(cuvsResources_t res);

typedef enum { MG_ALGO_IVF_FLAT, MG_ALGO_IVF_PQ, MG_ALGO_CAGRA } mg_algo_t;

typedef enum { MG_MODE_REPLICATED, MG_MODE_SHARDED, MG_MODE_LOCAL_THEN_DISTRIBUTED } mg_mode_t;

typedef struct {
  int64_t num_queries;
  int64_t num_db_vecs;
  int64_t dim;
  int64_t k;
  mg_mode_t mode;
  mg_algo_t algo;
  int64_t nprobe;
  int64_t nlist;
  cuvsDistanceType metric;
} mg_test_params;

// Test IVF-Flat multi-GPU functionality
int run_mg_ivf_flat_test(mg_test_params params,
                         float* index_data,
                         float* query_data,
                         float* distances_data,
                         int64_t* neighbors_data,
                         float* ref_distances_data,
                         int64_t* ref_neighbors_data)
{
  printf("Running MG IVF-Flat test (mode=%d, queries=%ld, db_vecs=%ld, dim=%ld, k=%ld)\n",
         params.mode,
         params.num_queries,
         params.num_db_vecs,
         params.dim,
         params.k);

  // Create multi-GPU resources
  cuvsResources_t res;
  cuvsMgResourcesCreate(&res);

  // Create dataset tensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {params.num_db_vecs, params.dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // Create index
  cuvsMgIvfFlatIndex_t index;
  cuvsMgIvfFlatIndexCreate(&index);

  // Build index
  cuvsMgIvfFlatIndexParams_t build_params;
  cuvsMgIvfFlatIndexParamsCreate(&build_params);
  build_params->base_params->metric                   = params.metric;
  build_params->base_params->n_lists                  = params.nlist;
  build_params->base_params->add_data_on_build        = false;
  build_params->base_params->kmeans_trainset_fraction = 1.0;
  build_params->base_params->metric_arg               = 0;

  if (params.mode == MG_MODE_REPLICATED) {
    build_params->mode = CUVS_MG_REPLICATED;
  } else {
    build_params->mode = CUVS_MG_SHARDED;
  }

  cuvsError_t build_result = cuvsMgIvfFlatBuild(res, build_params, &dataset_tensor, index);
  if (build_result != CUVS_SUCCESS) {
    const char* error_msg = cuvsGetLastErrorText();
    printf("MG IVF-Flat build failed: %s\n", error_msg ? error_msg : "Unknown error");
    goto cleanup;
  }

  // Extend index with data
  cuvsError_t extend_result = cuvsMgIvfFlatExtend(res, index, &dataset_tensor, NULL);
  if (extend_result != CUVS_SUCCESS) {
    printf("MG IVF-Flat extend failed\n");
    goto cleanup;
  }

  // Create queries tensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = query_data;
  queries_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {params.num_queries, params.dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // Create neighbors tensor
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_data;
  neighbors_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {params.num_queries, params.k};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // Create distances tensor
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_data;
  distances_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {params.num_queries, params.k};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  // Search index
  cuvsMgIvfFlatSearchParams_t search_params;
  cuvsMgIvfFlatSearchParamsCreate(&search_params);
  search_params->base_params->n_probes = params.nprobe;
  search_params->search_mode           = CUVS_MG_LOAD_BALANCER;
  search_params->merge_mode            = CUVS_MG_TREE_MERGE;
  search_params->n_rows_per_batch      = 3000;

  cuvsError_t search_result = cuvsMgIvfFlatSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor);
  if (search_result != CUVS_SUCCESS) {
    printf("MG IVF-Flat search failed\n");
    goto cleanup;
  }

  printf("MG IVF-Flat test completed successfully\n");

cleanup:
  cuvsMgIvfFlatSearchParamsDestroy(search_params);
  cuvsMgIvfFlatIndexParamsDestroy(build_params);
  cuvsMgIvfFlatIndexDestroy(index);
  cuvsMgResourcesDestroy(res);

  return (build_result == CUVS_SUCCESS && extend_result == CUVS_SUCCESS &&
          search_result == CUVS_SUCCESS)
           ? 0
           : 1;
}

// Test IVF-PQ multi-GPU functionality
int run_mg_ivf_pq_test(mg_test_params params,
                       float* index_data,
                       float* query_data,
                       float* distances_data,
                       int64_t* neighbors_data,
                       float* ref_distances_data,
                       int64_t* ref_neighbors_data)
{
  printf("Running MG IVF-PQ test (mode=%d, queries=%ld, db_vecs=%ld, dim=%ld, k=%ld)\n",
         params.mode,
         params.num_queries,
         params.num_db_vecs,
         params.dim,
         params.k);

  // Create multi-GPU resources
  cuvsResources_t res;
  cuvsMgResourcesCreate(&res);

  // Create dataset tensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {params.num_db_vecs, params.dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // Create index
  cuvsMgIvfPqIndex_t index;
  cuvsMgIvfPqIndexCreate(&index);

  // Build index
  cuvsMgIvfPqIndexParams_t build_params;
  cuvsMgIvfPqIndexParamsCreate(&build_params);
  build_params->base_params->metric                   = params.metric;
  build_params->base_params->n_lists                  = params.nlist;
  build_params->base_params->add_data_on_build        = false;
  build_params->base_params->kmeans_trainset_fraction = 1.0;
  build_params->base_params->metric_arg               = 0;

  if (params.mode == MG_MODE_REPLICATED) {
    build_params->mode = CUVS_MG_REPLICATED;
  } else {
    build_params->mode = CUVS_MG_SHARDED;
  }

  cuvsError_t build_result = cuvsMgIvfPqBuild(res, build_params, &dataset_tensor, index);
  if (build_result != CUVS_SUCCESS) {
    const char* error_msg = cuvsGetLastErrorText();
    printf("MG IVF-PQ build failed: %s\n", error_msg ? error_msg : "Unknown error");
    goto cleanup;
  }

  // Extend index with data
  cuvsError_t extend_result = cuvsMgIvfPqExtend(res, index, &dataset_tensor, NULL);
  if (extend_result != CUVS_SUCCESS) {
    printf("MG IVF-PQ extend failed\n");
    goto cleanup;
  }

  // Create queries tensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = query_data;
  queries_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {params.num_queries, params.dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // Create neighbors tensor
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_data;
  neighbors_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {params.num_queries, params.k};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // Create distances tensor
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_data;
  distances_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {params.num_queries, params.k};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  // Search index
  cuvsMgIvfPqSearchParams_t search_params;
  cuvsMgIvfPqSearchParamsCreate(&search_params);
  search_params->base_params->n_probes = params.nprobe;
  search_params->search_mode           = CUVS_MG_LOAD_BALANCER;
  search_params->merge_mode            = CUVS_MG_TREE_MERGE;
  search_params->n_rows_per_batch      = 3000;

  cuvsError_t search_result = cuvsMgIvfPqSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor);
  if (search_result != CUVS_SUCCESS) {
    printf("MG IVF-PQ search failed\n");
    goto cleanup;
  }

  printf("MG IVF-PQ test completed successfully\n");

cleanup:
  cuvsMgIvfPqSearchParamsDestroy(search_params);
  cuvsMgIvfPqIndexParamsDestroy(build_params);
  cuvsMgIvfPqIndexDestroy(index);
  cuvsMgResourcesDestroy(res);

  return (build_result == CUVS_SUCCESS && extend_result == CUVS_SUCCESS &&
          search_result == CUVS_SUCCESS)
           ? 0
           : 1;
}

// Test CAGRA multi-GPU functionality
int run_mg_cagra_test(mg_test_params params,
                      float* index_data,
                      float* query_data,
                      float* distances_data,
                      int64_t* neighbors_data,
                      float* ref_distances_data,
                      int64_t* ref_neighbors_data)
{
  printf("Running MG CAGRA test (mode=%d, queries=%ld, db_vecs=%ld, dim=%ld, k=%ld)\n",
         params.mode,
         params.num_queries,
         params.num_db_vecs,
         params.dim,
         params.k);

  // Create multi-GPU resources
  cuvsResources_t res;
  cuvsMgResourcesCreate(&res);

  // Create dataset tensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {params.num_db_vecs, params.dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // Create index
  cuvsMgCagraIndex_t index;
  cuvsMgCagraIndexCreate(&index);

  // Build index
  cuvsMgCagraIndexParams_t build_params;
  cuvsMgCagraIndexParamsCreate(&build_params);

  if (params.mode == MG_MODE_REPLICATED) {
    build_params->mode = CUVS_MG_REPLICATED;
  } else {
    build_params->mode = CUVS_MG_SHARDED;
  }

  cuvsError_t build_result = cuvsMgCagraBuild(res, build_params, &dataset_tensor, index);
  if (build_result != CUVS_SUCCESS) {
    const char* error_msg = cuvsGetLastErrorText();
    printf("MG CAGRA build failed: %s\n", error_msg ? error_msg : "Unknown error");
    goto cleanup;
  }

  // Create queries tensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = query_data;
  queries_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {params.num_queries, params.dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // Create neighbors tensor
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_data;
  neighbors_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {params.num_queries, params.k};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // Create distances tensor
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_data;
  distances_tensor.dl_tensor.device.device_type = kDLCPU;  // Multi-GPU requires host memory
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {params.num_queries, params.k};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  // Search index
  cuvsMgCagraSearchParams_t search_params;
  cuvsMgCagraSearchParamsCreate(&search_params);
  search_params->search_mode      = CUVS_MG_LOAD_BALANCER;
  search_params->merge_mode       = CUVS_MG_TREE_MERGE;
  search_params->n_rows_per_batch = 3000;

  cuvsError_t search_result = cuvsMgCagraSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor);
  if (search_result != CUVS_SUCCESS) {
    printf("MG CAGRA search failed\n");
    goto cleanup;
  }

  printf("MG CAGRA test completed successfully\n");

cleanup:
  cuvsMgCagraSearchParamsDestroy(search_params);
  cuvsMgCagraIndexParamsDestroy(build_params);
  cuvsMgCagraIndexDestroy(index);
  cuvsMgResourcesDestroy(res);

  return (build_result == CUVS_SUCCESS && search_result == CUVS_SUCCESS) ? 0 : 1;
}

// Generate reference results using brute force
int generate_reference_results(mg_test_params params,
                               float* index_data,
                               float* query_data,
                               float* ref_distances_data,
                               int64_t* ref_neighbors_data)
{
  printf("Generating reference results using brute force\n");

  // Create resources
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // Create dataset tensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data;
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;  // Brute force can use device memory
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {params.num_db_vecs, params.dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // Create queries tensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = query_data;
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;  // Brute force can use device memory
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {params.num_queries, params.dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // Create neighbors tensor
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = ref_neighbors_data;
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {params.num_queries, params.k};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // Create distances tensor
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = ref_distances_data;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {params.num_queries, params.k};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  // Build brute force index
  cuvsBruteForceIndex_t index;
  cuvsBruteForceIndexCreate(&index);

  cuvsError_t build_result = cuvsBruteForceBuild(res, &dataset_tensor, params.metric, 0.0f, index);
  if (build_result != CUVS_SUCCESS) {
    printf("Brute force build failed\n");
    goto cleanup;
  }

  // Search with brute force
  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)NULL;

  cuvsError_t search_result =
    cuvsBruteForceSearch(res, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter);
  if (search_result != CUVS_SUCCESS) {
    printf("Brute force search failed\n");
    goto cleanup;
  }

  printf("Reference results generated successfully\n");

cleanup:
  cuvsBruteForceIndexDestroy(index);
  cuvsResourcesDestroy(res);

  return (build_result == CUVS_SUCCESS && search_result == CUVS_SUCCESS) ? 0 : 1;
}

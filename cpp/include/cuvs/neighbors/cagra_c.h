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

#pragma once

#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

/**
 * @defgroup cagra_c C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enum to denote which ANN algorithm is used to build CAGRA graph
 *
 */
enum cagraGraphBuildAlgo {
  /* Use IVF-PQ to build all-neighbors knn graph */
  IVF_PQ,
  /* Experimental, use NN-Descent to build all-neighbors knn graph */
  NN_DESCENT
};

/**
 * @brief Supplemental parameters to build CAGRA Index
 *
 */
struct cagraIndexParams {
  /** Degree of input graph for pruning. */
  size_t intermediate_graph_degree = 128;
  /** Degree of output graph. */
  size_t graph_degree = 64;
  /** ANN algorithm to build knn graph. */
  cagraGraphBuildAlgo build_algo = IVF_PQ;
  /** Number of Iterations to run if building with NN_DESCENT */
  size_t nn_descent_niter = 20;
};

/**
 * @brief Enum to denote algorithm used to search CAGRA Index
 *
 */
enum cagraSearchAlgo {
  /** For large batch sizes. */
  SINGLE_CTA,
  /** For small batch sizes. */
  MULTI_CTA,
  MULTI_KERNEL,
  AUTO
};

/**
 * @brief Enum to denote Hash Mode used while searching CAGRA index
 *
 */
enum cagraHashMode { HASH, SMALL, AUTO_HASH };

/**
 * @brief Supplemental parameters to search CAGRA index
 *
 */
typedef struct {
  /** Maximum number of queries to search at the same time (batch size). Auto select when 0.*/
  size_t max_queries = 0;

  /** Number of intermediate search results retained during the search.
   *
   *  This is the main knob to adjust trade off between accuracy and search speed.
   *  Higher values improve the search accuracy.
   */
  size_t itopk_size = 64;

  /** Upper limit of search iterations. Auto select when 0.*/
  size_t max_iterations = 0;

  // In the following we list additional search parameters for fine tuning.
  // Reasonable default values are automatically chosen.

  /** Which search implementation to use. */
  cagraSearchAlgo algo = AUTO;

  /** Number of threads used to calculate a single distance. 4, 8, 16, or 32. */
  size_t team_size = 0;

  /** Number of graph nodes to select as the starting point for the search in each iteration. aka
   * search width?*/
  size_t search_width = 1;
  /** Lower limit of search iterations. */
  size_t min_iterations = 0;

  /** Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. */
  size_t thread_block_size = 0;
  /** Hashmap type. Auto selection when AUTO. */
  cagraHashMode hashmap_mode = AUTO_HASH;
  /** Lower limit of hashmap bit length. More than 8. */
  size_t hashmap_min_bitlen = 0;
  /** Upper limit of hashmap fill rate. More than 0.1, less than 0.9.*/
  float hashmap_max_fill_rate = 0.5;

  /** Number of iterations of initial random seed node selection. 1 or more. */
  uint32_t num_random_samplings = 1;
  /** Bit mask used for initial random seed node selection. */
  uint64_t rand_xor_mask = 0x128394;
} cagraSearchParams;

/**
 * @brief Struct to hold address of cuvs::neighbors::cagra::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;

} cagraIndex;

typedef cagraIndex* cagraIndex_t;

/**
 * @brief Allocate CAGRA index
 *
 * @param[in] index cagraIndex_t to allocate
 * @return cagraError_t
 */
cuvsError_t cagraIndexCreate(cagraIndex_t* index);

/**
 * @brief De-allocate CAGRA index
 *
 * @param[in] index cagraIndex_t to de-allocate
 */
cuvsError_t cagraIndexDestroy(cagraIndex_t index);

/**
 * @brief Build a CAGRA index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        3. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra_c.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create CAGRA index
 * cagraIndex_t index;
 * cuvsError_t index_create_status = cagraIndexCreate(&index);
 *
 * // Build the CAGRA Index
 * cuvsError_t build_status = cagraBuild(res, params, &dataset, index);
 *
 * // de-allocate `index` and `res`
 * cuvsError_t index_destroy_status = cagraIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cagraIndexParams used to build CAGRA index
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[out] index cagraIndex_t Newly built CAGRA index
 * @return cuvsError_t
 */
cuvsError_t cagraBuild(cuvsResources_t res,
                       cagraIndexParams params,
                       DLManagedTensor* dataset,
                       cagraIndex_t index);

/**
 * @brief Build a CAGRA index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the CAGRA Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 * queries.dl_tensor.dtype.code` Types for input are:
 *        1. `queries`: kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra_c.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 * DLManagedTensor queries;
 * DLManagedTensor neighbors;
 *
 * // Search the `index` built using `cagraBuild`
 * cagraSearchParams params;
 * cuvsError_t search_status = cagraSearch(res, params, index, queries, neighbors, distances);
 *
 * // de-allocate `index` and `res`
 * cuvsError_t index_destroy_status = cagraIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cagraSearchParams used to search CAGRA index
 * @param[in] index cagraIndex which has been returned by `cagraBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 */
cuvsError_t cagraSearch(cuvsResources_t res,
                        cagraSearchParams params,
                        cagraIndex_t index,
                        DLManagedTensor* queries,
                        DLManagedTensor* neighbors,
                        DLManagedTensor* distances);

#ifdef __cplusplus
}
#endif

/**
 * @}
 */

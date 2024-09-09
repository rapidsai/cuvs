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
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup cagra_c_index_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Enum to denote which ANN algorithm is used to build CAGRA graph
 *
 */
enum cuvsCagraGraphBuildAlgo {
  /* Select build algorithm automatically */
  AUTO_SELECT,
  /* Use IVF-PQ to build all-neighbors knn graph */
  IVF_PQ,
  /* Experimental, use NN-Descent to build all-neighbors knn graph */
  NN_DESCENT
};

/** Parameters for VPQ compression. */
struct cuvsCagraCompressionParams {
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   */
  uint32_t vq_n_centers;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters;
  /**
   * The fraction of data to use during iterative kmeans building (VQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double vq_kmeans_trainset_fraction;
  /**
   * The fraction of data to use during iterative kmeans building (PQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double pq_kmeans_trainset_fraction;
};

typedef struct cuvsCagraCompressionParams* cuvsCagraCompressionParams_t;

/**
 * @brief Supplemental parameters to build CAGRA Index
 *
 */
struct cuvsCagraIndexParams {
  /** Degree of input graph for pruning. */
  size_t intermediate_graph_degree;
  /** Degree of output graph. */
  size_t graph_degree;
  /** ANN algorithm to build knn graph. */
  enum cuvsCagraGraphBuildAlgo build_algo;
  /** Number of Iterations to run if building with NN_DESCENT */
  size_t nn_descent_niter;
  /**
   * Optional: specify compression parameters if compression is desired.
   *
   * NOTE: this is experimental new API, consider it unsafe.
   */
  cuvsCagraCompressionParams_t compression;
};

typedef struct cuvsCagraIndexParams* cuvsCagraIndexParams_t;

/**
 * @brief Allocate CAGRA Index params, and populate with default values
 *
 * @param[in] params cuvsCagraIndexParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params);

/**
 * @brief De-allocate CAGRA Index params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params);

/**
 * @brief Allocate CAGRA Compression params, and populate with default values
 *
 * @param[in] params cuvsCagraCompressionParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsCagraCompressionParamsCreate(cuvsCagraCompressionParams_t* params);

/**
 * @brief De-allocate CAGRA Compression params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsCagraCompressionParamsDestroy(cuvsCagraCompressionParams_t params);

/**
 * @}
 */

/**
 * @defgroup cagra_c_search_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Enum to denote algorithm used to search CAGRA Index
 *
 */
enum cuvsCagraSearchAlgo {
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
enum cuvsCagraHashMode { HASH, SMALL, AUTO_HASH };

/**
 * @brief Supplemental parameters to search CAGRA index
 *
 */
struct cuvsCagraSearchParams {
  /** Maximum number of queries to search at the same time (batch size). Auto select when 0.*/
  size_t max_queries;

  /** Number of intermediate search results retained during the search.
   *
   *  This is the main knob to adjust trade off between accuracy and search speed.
   *  Higher values improve the search accuracy.
   */
  size_t itopk_size;

  /** Upper limit of search iterations. Auto select when 0.*/
  size_t max_iterations;

  // In the following we list additional search parameters for fine tuning.
  // Reasonable default values are automatically chosen.

  /** Which search implementation to use. */
  enum cuvsCagraSearchAlgo algo;

  /** Number of threads used to calculate a single distance. 4, 8, 16, or 32. */
  size_t team_size;

  /** Number of graph nodes to select as the starting point for the search in each iteration. aka
   * search width?*/
  size_t search_width;
  /** Lower limit of search iterations. */
  size_t min_iterations;

  /** Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. */
  size_t thread_block_size;
  /** Hashmap type. Auto selection when AUTO. */
  enum cuvsCagraHashMode hashmap_mode;
  /** Lower limit of hashmap bit length. More than 8. */
  size_t hashmap_min_bitlen;
  /** Upper limit of hashmap fill rate. More than 0.1, less than 0.9.*/
  float hashmap_max_fill_rate;

  /** Number of iterations of initial random seed node selection. 1 or more. */
  uint32_t num_random_samplings;
  /** Bit mask used for initial random seed node selection. */
  uint64_t rand_xor_mask;
};

typedef struct cuvsCagraSearchParams* cuvsCagraSearchParams_t;

/**
 * @brief Allocate CAGRA search params, and populate with default values
 *
 * @param[in] params cuvsCagraSearchParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params);

/**
 * @brief De-allocate CAGRA search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup cagra_c_index C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Struct to hold address of cuvs::neighbors::cagra::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;

} cuvsCagraIndex;

typedef cuvsCagraIndex* cuvsCagraIndex_t;

/**
 * @brief Allocate CAGRA index
 *
 * @param[in] index cuvsCagraIndex_t to allocate
 * @return cagraError_t
 */
cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* index);

/**
 * @brief De-allocate CAGRA index
 *
 * @param[in] index cuvsCagraIndex_t to de-allocate
 */
cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup cagra_c_index_build C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

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
 * #include <cuvs/neighbors/cagra.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create default index params
 * cuvsCagraIndexParams_t params;
 * cuvsError_t params_create_status = cuvsCagraIndexParamsCreate(&params);
 *
 * // Create CAGRA index
 * cuvsCagraIndex_t index;
 * cuvsError_t index_create_status = cuvsCagraIndexCreate(&index);
 *
 * // Build the CAGRA Index
 * cuvsError_t build_status = cuvsCagraBuild(res, params, &dataset, index);
 *
 * // de-allocate `params`, `index` and `res`
 * cuvsError_t params_destroy_status = cuvsCagraIndexParamsDestroy(params);
 * cuvsError_t index_destroy_status = cuvsCagraIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsCagraIndexParams_t used to build CAGRA index
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[out] index cuvsCagraIndex_t Newly built CAGRA index
 * @return cuvsError_t
 */
cuvsError_t cuvsCagraBuild(cuvsResources_t res,
                           cuvsCagraIndexParams_t params,
                           DLManagedTensor* dataset,
                           cuvsCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup cagra_c_index_search C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */
/**
 * @brief Search a CAGRA index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the CAGRA Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 * queries.dl_tensor.dtype.code` Types for input are:
 *        1. `queries`:
 *`         a. kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *          b. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *          c. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
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
 * // Create default search params
 * cuvsCagraSearchParams_t params;
 * cuvsError_t params_create_status = cuvsCagraSearchParamsCreate(&params);
 *
 * // Search the `index` built using `cuvsCagraBuild`
 * cuvsError_t search_status = cuvsCagraSearch(res, params, index, &queries, &neighbors,
 * &distances);
 *
 * // de-allocate `params` and `res`
 * cuvsError_t params_destroy_status = cuvsCagraSearchParamsDestroy(params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsCagraSearchParams_t used to search CAGRA index
 * @param[in] index cuvsCagraIndex which has been returned by `cuvsCagraBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 */
cuvsError_t cuvsCagraSearch(cuvsResources_t res,
                            cuvsCagraSearchParams_t params,
                            cuvsCagraIndex_t index,
                            DLManagedTensor* queries,
                            DLManagedTensor* neighbors,
                            DLManagedTensor* distances);

/**
 * @}
 */

/**
 * @defgroup cagra_c_serialize CAGRA C-API serialize functions
 * @{
 */
/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.c}
 * #include <cuvs/neighbors/cagra.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsCagraBuild`
 * cuvsCagraSerialize(res, "/path/to/index", index, true);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 *
 */
cuvsError_t cuvsCagraSerialize(cuvsResources_t res,
                               const char* filename,
                               cuvsCagraIndex_t index,
                               bool include_dataset);

/**
 * Save the CAGRA index to file in hnswlib format.
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsCagraBuild`
 * cuvsCagraSerializeHnswlib(res, "/path/to/index", index);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 *
 */
cuvsError_t cuvsCagraSerializeToHnswlib(cuvsResources_t res,
                                        const char* filename,
                                        cuvsCagraIndex_t index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index CAGRA index loaded disk
 */
cuvsError_t cuvsCagraDeserialize(cuvsResources_t res, const char* filename, cuvsCagraIndex_t index);
/**
 * @}
 */
#ifdef __cplusplus
}
#endif

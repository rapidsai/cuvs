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

#include "cagra.h"

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup hnsw_c_index_params C API for HNSW index params
 * @{
 */

/**
 * @brief Hierarchy for HNSW index when converting from CAGRA index
 *
 * NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only index.
 */
enum cuvsHnswHierarchy {
  /* Flat hierarchy, search is base-layer only */
  NONE,
  /* Full hierarchy is built using the CPU */
  CPU,
  /* Full hierarchy is built using the GPU */
  GPU
};

struct cuvsHnswIndexParams {
  /* hierarchy of the hnsw index */
  enum cuvsHnswHierarchy hierarchy;
  /** Size of the candidate list during hierarchy construction when hierarchy is `CPU`*/
  int ef_construction;
  /** Number of host threads to use to construct hierarchy when hierarchy is `CPU` or `GPU`.
      When the value is 0, the number of threads is automatically determined to the
      maximum number of threads available.
      NOTE: When hierarchy is `GPU`, while the majority of the work is done on the GPU,
      initialization of the HNSW index itself and some other work
      is parallelized with the help of CPU threads.
  */
  int num_threads;
};

typedef struct cuvsHnswIndexParams* cuvsHnswIndexParams_t;

/**
 * @brief Allocate HNSW Index params, and populate with default values
 *
 * @param[in] params cuvsHnswIndexParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsHnswIndexParamsCreate(cuvsHnswIndexParams_t* params);

/**
 * @brief De-allocate HNSW Index params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsHnswIndexParamsDestroy(cuvsHnswIndexParams_t params);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_index C API for hnswlib wrapper index
 * @{
 */

/**
 * @brief Struct to hold address of cuvs::neighbors::Hnsw::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;

} cuvsHnswIndex;

typedef cuvsHnswIndex* cuvsHnswIndex_t;

/**
 * @brief Allocate HNSW index
 *
 * @param[in] index cuvsHnswIndex_t to allocate
 * @return HnswError_t
 */
cuvsError_t cuvsHnswIndexCreate(cuvsHnswIndex_t* index);

/**
 * @brief De-allocate HNSW index
 *
 * @param[in] index cuvsHnswIndex_t to de-allocate
 */
cuvsError_t cuvsHnswIndexDestroy(cuvsHnswIndex_t index);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_extend_params Parameters for extending HNSW index
 * @{
 */

struct cuvsHnswExtendParams {
  /** Number of CPU threads used to extend additional vectors */
  int num_threads;
};

typedef struct cuvsHnswExtendParams* cuvsHnswExtendParams_t;

/**
 * @brief Allocate HNSW extend params, and populate with default values
 *
 * @param[in] params cuvsHnswExtendParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsHnswExtendParamsCreate(cuvsHnswExtendParams_t* params);

/**
 * @brief De-allocate HNSW extend params
 *
 * @param[in] params cuvsHnswExtendParams_t to de-allocate
 * @return cuvsError_t
 */

cuvsError_t cuvsHnswExtendParamsDestroy(cuvsHnswExtendParams_t params);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_index_load Load CAGRA index as hnswlib index
 * @{
 */

/**
 * @brief Convert a CAGRA Index to an HNSW index.
 * NOTE: When hierarchy is:
 *       1. `NONE`: This method uses the filesystem to write the CAGRA index in
 * `/tmp/<random_number>.bin` before reading it as an hnswlib index, then deleting the temporary
 * file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS,
 * as the format is not compatible with the original hnswlib.
 *       2. `CPU`: The returned index is mutable and can be extended with additional vectors. The
 * serialized index is also compatible with the original hnswlib library.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsHnswIndexParams_t used to load Hnsw index
 * @param[in] cagra_index cuvsCagraIndex_t to convert to HNSW index
 * @param[out] hnsw_index cuvsHnswIndex_t to return the HNSW index
 *
 * @return cuvsError_t
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 * #include <cuvs/neighbors/hnsw.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create a CAGRA index with `cuvsCagraBuild`
 *
 * // Convert the CAGRA index to an HNSW index
 * cuvsHnswIndex_t hnsw_index;
 * cuvsHnswIndexCreate(&hnsw_index);
 * cuvsHnswIndexParams_t hnsw_params;
 * cuvsHnswIndexParamsCreate(&hnsw_params);
 * cuvsHnswFromCagra(res, hnsw_params, cagra_index, hnsw_index);
 *
 * // de-allocate `hnsw_params`, `hnsw_index` and `res`
 * cuvsError_t hnsw_params_destroy_status = cuvsHnswIndexParamsDestroy(hnsw_params);
 * cuvsError_t hnsw_index_destroy_status = cuvsHnswIndexDestroy(hnsw_index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 */
cuvsError_t cuvsHnswFromCagra(cuvsResources_t res,
                              cuvsHnswIndexParams_t params,
                              cuvsCagraIndex_t cagra_index,
                              cuvsHnswIndex_t hnsw_index);

cuvsError_t cuvsHnswFromCagraWithDataset(cuvsResources_t res,
                                         cuvsHnswIndexParams_t params,
                                         cuvsCagraIndex_t cagra_index,
                                         cuvsHnswIndex_t hnsw_index,
                                         DLManagedTensor* dataset_tensor);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_index_extend Extend HNSW index with additional vectors
 * @{
 */

/**
 * @brief Add new vectors to an HNSW index
 * NOTE: The HNSW index can only be extended when the hierarchy is `CPU`
 *       when converting from a CAGRA index.

 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsHnswExtendParams_t used to extend Hnsw index
 * @param[in] additional_dataset DLManagedTensor* additional dataset to extend the index
 * @param[inout] index cuvsHnswIndex_t to extend
  *
  * @return cuvsError_t
  *
  * @code{.c}
  * #include <cuvs/core/c_api.h>
  * #include <cuvs/neighbors/cagra.h>
  * #include <cuvs/neighbors/hnsw.h>
  *
  * // Create cuvsResources_t
  * cuvsResources_t res;
  * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
  *
  * // create an index with `cuvsCagraBuild`
  *
  * // Convert the CAGRA index to an HNSW index
  * cuvsHnswIndex_t hnsw_index;
  * cuvsHnswIndexCreate(&hnsw_index);
  * cuvsHnswIndexParams_t hnsw_params;
  * cuvsHnswIndexParamsCreate(&hnsw_params);
  * cuvsHnswFromCagra(res, hnsw_params, cagra_index, hnsw_index);
  *
  * // Extend the HNSW index with additional vectors
  * DLManagedTensor additional_dataset;
  * cuvsHnswExtendParams_t extend_params;
  * cuvsHnswExtendParamsCreate(&extend_params);
  * cuvsHnswExtend(res, extend_params, additional_dataset, hnsw_index);
  *
  * // de-allocate `hnsw_params`, `hnsw_index`, `extend_params` and `res`
  * cuvsError_t hnsw_params_destroy_status = cuvsHnswIndexParamsDestroy(hnsw_params);
  * cuvsError_t hnsw_index_destroy_status = cuvsHnswIndexDestroy(hnsw_index);
  * cuvsError_t extend_params_destroy_status = cuvsHnswExtendParamsDestroy(extend_params);
  * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
  * @endcode
  */

cuvsError_t cuvsHnswExtend(cuvsResources_t res,
                           cuvsHnswExtendParams_t params,
                           DLManagedTensor* additional_dataset,
                           cuvsHnswIndex_t index);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_search_params C API for hnswlib wrapper search params
 * @{
 */

struct cuvsHnswSearchParams {
  int32_t ef;
  int32_t num_threads;
};

typedef struct cuvsHnswSearchParams* cuvsHnswSearchParams_t;

/**
 * @brief Allocate HNSW search params, and populate with default values
 *
 * @param[in] params cuvsHnswSearchParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsHnswSearchParamsCreate(cuvsHnswSearchParams_t* params);

/**
 * @brief De-allocate HNSW search params
 *
 * @param[in] params cuvsHnswSearchParams_t to de-allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsHnswSearchParamsDestroy(cuvsHnswSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_index_search C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */
/**
 * @brief Search a HNSW index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCPU`, `kDLCUDAHost`, or `kDLCUDAManaged`.
 *        It is also important to note that the HNSW Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 *        queries.dl_tensor.dtype.code`
 *        Supported types for input are:
 *        1. `queries`:
 *          a. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *          b. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *          c. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 64`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 * NOTE: When hierarchy is `NONE`, the HNSW index can only be searched by the hnswlib wrapper in
 * cuVS, as the format is not compatible with the original hnswlib.
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/hnsw.h>
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
 * cuvsHnswSearchParams_t params;
 * cuvsError_t params_create_status = cuvsHnswSearchParamsCreate(&params);
 *
 * // Search the `index` built using `cuvsHnswFromCagra`
 * cuvsError_t search_status = cuvsHnswSearch(res, params, index, &queries, &neighbors,
 * &distances);
 *
 * // de-allocate `params` and `res`
 * cuvsError_t params_destroy_status = cuvsHnswSearchParamsDestroy(params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsHnswSearchParams_t used to search Hnsw index
 * @param[in] index cuvsHnswIndex which has been returned by `cuvsHnswFromCagra`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 */
cuvsError_t cuvsHnswSearch(cuvsResources_t res,
                           cuvsHnswSearchParams_t params,
                           cuvsHnswIndex_t index,
                           DLManagedTensor* queries,
                           DLManagedTensor* neighbors,
                           DLManagedTensor* distances);

/**
 * @}
 */

/**
 * @defgroup hnsw_c_serialize HNSW C-API serialize functions
 * @{
 */

/**
 * @brief Serialize a CAGRA index to a file as an hnswlib index
 * NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by
 * the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original
 * hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the
 * original hnswlib library.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the name of the file to save the index
 * @param[in] index cuvsHnswIndex_t to serialize
 * @return cuvsError_t
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 * #include <cuvs/neighbors/hnsw.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsCagraBuild`
 *
 * // Convert the CAGRA index to an HNSW index
 * cuvsHnswIndex_t hnsw_index;
 * cuvsHnswIndexCreate(&hnsw_index);
 * cuvsHnswIndexParams_t hnsw_params;
 * cuvsHnswIndexParamsCreate(&hnsw_params);
 * cuvsHnswFromCagra(res, hnsw_params, cagra_index, hnsw_index);
 *
 * // Serialize the HNSW index
 * cuvsHnswSerialize(res, "/path/to/index", hnsw_index);
 *
 * // de-allocate `hnsw_params`, `hnsw_index` and `res`
 * cuvsError_t hnsw_params_destroy_status = cuvsHnswIndexParamsDestroy(hnsw_params);
 * cuvsError_t hnsw_index_destroy_status = cuvsHnswIndexDestroy(hnsw_index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 */
cuvsError_t cuvsHnswSerialize(cuvsResources_t res, const char* filename, cuvsHnswIndex_t index);

/**
 * Load hnswlib index from file which was serialized from a HNSW index.
 * NOTE: When hierarchy is `NONE`, the loaded hnswlib index is immutable, and only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original
 * hnswlib. Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 * #include <cuvs/neighbors/hnsw.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsCagraBuild`
 * cuvsCagraSerializeHnswlib(res, "/path/to/index", index);
 *
 * // Load the serialized CAGRA index from file as an hnswlib index
 * // The index should have the same dtype as the one used to build CAGRA the index
 * cuvsHnswIndex_t hnsw_index;
 * cuvsHnswIndexCreate(&hnsw_index);
 * cuvsHnsWIndexParams_t hnsw_params;
 * cuvsHnswIndexParamsCreate(&hnsw_params);
 * hnsw_params->hierarchy = NONE;
 * hnsw_index->dtype = index->dtype;
 * cuvsHnswDeserialize(res, hnsw_params, "/path/to/index", dim, metric hnsw_index);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsHnswIndexParams_t used to load Hnsw index
 * @param[in] filename the name of the file that stores the index
 * @param[in] dim the dimension of the vectors in the index
 * @param[in] metric the distance metric used to build the index
 * @param[out] index HNSW index loaded disk
 */
cuvsError_t cuvsHnswDeserialize(cuvsResources_t res,
                                cuvsHnswIndexParams_t params,
                                const char* filename,
                                int dim,
                                cuvsDistanceType metric,
                                cuvsHnswIndex_t index);
/**
 * @}
 */

#ifdef __cplusplus
}
#endif

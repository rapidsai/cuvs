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
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup hnsw_c_search_params C API for hnswlib wrapper search params
 * @{
 */

struct cuvsHnswSearchParams {
  int32_t ef;
  int32_t numThreads;
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
 *        1. `queries`: `kDLDataType.code == kDLFloat` or `kDLDataType.code == kDLInt` and
 * `kDLDataType.bits = 32`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 64`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 * NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS,
 *       as the format is not compatible with the original hnswlib.
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
 * // Search the `index` built using `cuvsHnswBuild`
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
 * @param[in] index cuvsHnswIndex which has been returned by `cuvsHnswBuild`
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
 * Load hnswlib index from file which was serialized from a HNSW index.
 * NOTE: The loaded hnswlib index is immutable, and only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 * Experimental, both the API and the serialization format are subject to change.
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
 * hnsw_index->dtype = index->dtype;
 * cuvsCagraDeserialize(res, "/path/to/index", hnsw_index);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] dim the dimension of the vectors in the index
 * @param[in] metric the distance metric used to build the index
 * @param[out] index HNSW index loaded disk
 */
cuvsError_t cuvsHnswDeserialize(cuvsResources_t res,
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

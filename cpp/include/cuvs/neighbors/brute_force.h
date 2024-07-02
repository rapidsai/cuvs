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
#include <cuvs/neighbors/common.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup bruteforce_c_index Bruteforce index
 * @{
 */
/**
 * @brief Struct to hold address of cuvs::neighbors::brute_force::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsBruteForceIndex;

typedef cuvsBruteForceIndex* cuvsBruteForceIndex_t;

/**
 * @brief Allocate BRUTEFORCE index
 *
 * @param[in] index cuvsBruteForceIndex_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsBruteForceIndexCreate(cuvsBruteForceIndex_t* index);

/**
 * @brief De-allocate BRUTEFORCE index
 *
 * @param[in] index cuvsBruteForceIndex_t to de-allocate
 */
cuvsError_t cuvsBruteForceIndexDestroy(cuvsBruteForceIndex_t index);
/**
 * @}
 */

/**
 * @defgroup bruteforce_c_index_build Bruteforce index build
 * @{
 */
/**
 * @brief Build a BRUTEFORCE index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        3. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/brute_force.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create BRUTEFORCE index
 * cuvsBruteForceIndex_t index;
 * cuvsError_t index_create_status = cuvsBruteForceIndexCreate(&index);
 *
 * // Build the BRUTEFORCE Index
 * cuvsError_t build_status = cuvsBruteForceBuild(res, &dataset_tensor, L2Expanded, 0.f, index);
 *
 * // de-allocate `index` and `res`
 * cuvsError_t index_destroy_status = cuvsBruteForceIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[in] metric metric
 * @param[in] metric_arg metric_arg
 * @param[out] index cuvsBruteForceIndex_t Newly built BRUTEFORCE index
 * @return cuvsError_t
 */
cuvsError_t cuvsBruteForceBuild(cuvsResources_t res,
                                DLManagedTensor* dataset,
                                cuvsDistanceType metric,
                                float metric_arg,
                                cuvsBruteForceIndex_t index);
/**
 * @}
 */

/**
 * @defgroup bruteforce_c_index_search Bruteforce index search
 * @{
 */
/**
 * @brief Search a BRUTEFORCE index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the BRUTEFORCE index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 *        queries.dl_tensor.dtype.code` Types for input are:
 *        1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/brute_force.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 * DLManagedTensor queries;
 * DLManagedTensor neighbors;
 * DLManagedTensor bitmap;
 *
 * cuvsFilter prefilter{(uintptr_t)&bitmap, BITMAP};
 *
 * // Search the `index` built using `cuvsBruteForceBuild`
 * cuvsError_t search_status = cuvsBruteForceSearch(res, index, &queries, &neighbors, &distances,
 * prefilter);
 *
 * // de-allocate `res`
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index cuvsBruteForceIndex which has been returned by `cuvsBruteForceBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 * @param[in] prefilter cuvsFilter input prefilter that can be used
              to filter queries and neighbors based on the given bitmap.
 */
cuvsError_t cuvsBruteForceSearch(cuvsResources_t res,
                                 cuvsBruteForceIndex_t index,
                                 DLManagedTensor* queries,
                                 DLManagedTensor* neighbors,
                                 DLManagedTensor* distances,
                                 cuvsFilter prefilter);
/**
 * @}
 */

#ifdef __cplusplus
}
#endif

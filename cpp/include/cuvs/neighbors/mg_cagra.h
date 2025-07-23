/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/mg_common.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup mg_cagra_c_index_params Multi-GPU CAGRA index build parameters
 * @{
 */

/**
 * @brief Multi-GPU parameters to build CAGRA Index
 *
 * This structure extends the base CAGRA index parameters with multi-GPU specific settings.
 */
struct cuvsMgCagraIndexParams {
  /** Base CAGRA index parameters */
  cuvsCagraIndexParams_t base_params;
  /** Distribution mode for multi-GPU setup */
  cuvsMgDistributionMode mode;
};

typedef struct cuvsMgCagraIndexParams* cuvsMgCagraIndexParams_t;

/**
 * @brief Allocate Multi-GPU CAGRA Index params, and populate with default values
 *
 * @param[in] index_params cuvsMgCagraIndexParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraIndexParamsCreate(cuvsMgCagraIndexParams_t* index_params);

/**
 * @brief De-allocate Multi-GPU CAGRA Index params
 *
 * @param[in] index_params
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraIndexParamsDestroy(cuvsMgCagraIndexParams_t index_params);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_search_params Multi-GPU CAGRA index search parameters
 * @{
 */

/**
 * @brief Multi-GPU parameters to search CAGRA index
 *
 * This structure extends the base CAGRA search parameters with multi-GPU specific settings.
 */
struct cuvsMgCagraSearchParams {
  /** Base CAGRA search parameters */
  cuvsCagraSearchParams_t base_params;
  /** Replicated search mode */
  cuvsMgReplicatedSearchMode search_mode;
  /** Sharded merge mode */
  cuvsMgShardedMergeMode merge_mode;
  /** Number of rows per batch */
  int64_t n_rows_per_batch;
};

typedef struct cuvsMgCagraSearchParams* cuvsMgCagraSearchParams_t;

/**
 * @brief Allocate Multi-GPU CAGRA search params, and populate with default values
 *
 * @param[in] params cuvsMgCagraSearchParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraSearchParamsCreate(cuvsMgCagraSearchParams_t* params);

/**
 * @brief De-allocate Multi-GPU CAGRA search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraSearchParamsDestroy(cuvsMgCagraSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index Multi-GPU CAGRA index
 * @{
 */

/**
 * @brief Struct to hold address of cuvs::neighbors::mg_index<cagra::index> and its active trained
 * dtype
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsMgCagraIndex;

typedef cuvsMgCagraIndex* cuvsMgCagraIndex_t;

/**
 * @brief Allocate Multi-GPU CAGRA index
 *
 * @param[in] index cuvsMgCagraIndex_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraIndexCreate(cuvsMgCagraIndex_t* index);

/**
 * @brief De-allocate Multi-GPU CAGRA index
 *
 * @param[in] index cuvsMgCagraIndex_t to de-allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraIndexDestroy(cuvsMgCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_build Multi-GPU CAGRA index build
 * @{
 */

/**
 * @brief Build a Multi-GPU CAGRA index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params Multi-GPU CAGRA index parameters
 * @param[in] dataset_tensor DLManagedTensor* training dataset
 * @param[out] index Multi-GPU CAGRA index
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraBuild(cuvsResources_t res,
                             cuvsMgCagraIndexParams_t params,
                             DLManagedTensor* dataset_tensor,
                             cuvsMgCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_search Multi-GPU CAGRA index search
 * @{
 */

/**
 * @brief Search a Multi-GPU CAGRA index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params Multi-GPU CAGRA search parameters
 * @param[in] index Multi-GPU CAGRA index
 * @param[in] queries_tensor DLManagedTensor* queries dataset
 * @param[out] neighbors_tensor DLManagedTensor* output neighbors
 * @param[out] distances_tensor DLManagedTensor* output distances
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraSearch(cuvsResources_t res,
                              cuvsMgCagraSearchParams_t params,
                              cuvsMgCagraIndex_t index,
                              DLManagedTensor* queries_tensor,
                              DLManagedTensor* neighbors_tensor,
                              DLManagedTensor* distances_tensor);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_extend Multi-GPU CAGRA index extend
 * @{
 */

/**
 * @brief Extend a Multi-GPU CAGRA index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in,out] index Multi-GPU CAGRA index to extend
 * @param[in] new_vectors_tensor DLManagedTensor* new vectors to add
 * @param[in] new_indices_tensor DLManagedTensor* new indices (optional, can be NULL)
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraExtend(cuvsResources_t res,
                              cuvsMgCagraIndex_t index,
                              DLManagedTensor* new_vectors_tensor,
                              DLManagedTensor* new_indices_tensor);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_serialize Multi-GPU CAGRA index serialize
 * @{
 */

/**
 * @brief Serialize a Multi-GPU CAGRA index to file
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index Multi-GPU CAGRA index to serialize
 * @param[in] filename Path to the output file
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraSerialize(cuvsResources_t res,
                                 cuvsMgCagraIndex_t index,
                                 const char* filename);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_deserialize Multi-GPU CAGRA index deserialize
 * @{
 */

/**
 * @brief Deserialize a Multi-GPU CAGRA index from file
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename Path to the input file
 * @param[out] index Multi-GPU CAGRA index
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraDeserialize(cuvsResources_t res,
                                   const char* filename,
                                   cuvsMgCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_cagra_c_index_distribute Multi-GPU CAGRA index distribute
 * @{
 */

/**
 * @brief Distribute a local CAGRA index to create a Multi-GPU index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename Path to the local index file
 * @param[out] index Multi-GPU CAGRA index
 * @return cuvsError_t
 */
cuvsError_t cuvsMgCagraDistribute(cuvsResources_t res,
                                  const char* filename,
                                  cuvsMgCagraIndex_t index);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

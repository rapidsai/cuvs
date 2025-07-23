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
#include <cuvs/neighbors/ivf_flat.h>
#include <cuvs/neighbors/mg_common.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup mg_ivf_flat_c_index_params Multi-GPU IVF-Flat index build parameters
 * @{
 */

/**
 * @brief Multi-GPU parameters to build IVF-Flat Index
 *
 * This structure extends the base IVF-Flat index parameters with multi-GPU specific settings.
 */
struct cuvsMgIvfFlatIndexParams {
  /** Base IVF-Flat index parameters */
  cuvsIvfFlatIndexParams_t base_params;
  /** Distribution mode for multi-GPU setup */
  cuvsMgDistributionMode mode;
};

typedef struct cuvsMgIvfFlatIndexParams* cuvsMgIvfFlatIndexParams_t;

/**
 * @brief Allocate Multi-GPU IVF-Flat Index params, and populate with default values
 *
 * @param[in] index_params cuvsMgIvfFlatIndexParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatIndexParamsCreate(cuvsMgIvfFlatIndexParams_t* index_params);

/**
 * @brief De-allocate Multi-GPU IVF-Flat Index params
 *
 * @param[in] index_params
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatIndexParamsDestroy(cuvsMgIvfFlatIndexParams_t index_params);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_flat_c_search_params Multi-GPU IVF-Flat index search parameters
 * @{
 */

/**
 * @brief Multi-GPU parameters to search IVF-Flat index
 *
 * This structure extends the base IVF-Flat search parameters with multi-GPU specific settings.
 */
struct cuvsMgIvfFlatSearchParams {
  /** Base IVF-Flat search parameters */
  cuvsIvfFlatSearchParams_t base_params;
  /** Replicated search mode */
  cuvsMgReplicatedSearchMode search_mode;
  /** Sharded merge mode */
  cuvsMgShardedMergeMode merge_mode;
  /** Number of rows per batch */
  int64_t n_rows_per_batch;
};

typedef struct cuvsMgIvfFlatSearchParams* cuvsMgIvfFlatSearchParams_t;

/**
 * @brief Allocate Multi-GPU IVF-Flat search params, and populate with default values
 *
 * @param[in] params cuvsMgIvfFlatSearchParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatSearchParamsCreate(cuvsMgIvfFlatSearchParams_t* params);

/**
 * @brief De-allocate Multi-GPU IVF-Flat search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatSearchParamsDestroy(cuvsMgIvfFlatSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_flat_c_index Multi-GPU IVF-Flat index
 * @{
 */

/**
 * @brief Struct to hold address of cuvs::neighbors::mg_index<ivf_flat::index> and its active
 * trained dtype
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsMgIvfFlatIndex;

typedef cuvsMgIvfFlatIndex* cuvsMgIvfFlatIndex_t;

/**
 * @brief Allocate Multi-GPU IVF-Flat index
 *
 * @param[in] index cuvsMgIvfFlatIndex_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatIndexCreate(cuvsMgIvfFlatIndex_t* index);

/**
 * @brief De-allocate Multi-GPU IVF-Flat index
 *
 * @param[in] index cuvsMgIvfFlatIndex_t to de-allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatIndexDestroy(cuvsMgIvfFlatIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_flat_c_index_build Multi-GPU IVF-Flat index build
 * @{
 */

/**
 * @brief Build a Multi-GPU IVF-Flat index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params Multi-GPU IVF-Flat index parameters
 * @param[in] dataset_tensor DLManagedTensor* training dataset
 * @param[out] index Multi-GPU IVF-Flat index
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatBuild(cuvsResources_t res,
                               cuvsMgIvfFlatIndexParams_t params,
                               DLManagedTensor* dataset_tensor,
                               cuvsMgIvfFlatIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_flat_c_index_search Multi-GPU IVF-Flat index search
 * @{
 */

/**
 * @brief Search a Multi-GPU IVF-Flat index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params Multi-GPU IVF-Flat search parameters
 * @param[in] index Multi-GPU IVF-Flat index
 * @param[in] queries_tensor DLManagedTensor* queries dataset
 * @param[out] neighbors_tensor DLManagedTensor* output neighbors
 * @param[out] distances_tensor DLManagedTensor* output distances
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatSearch(cuvsResources_t res,
                                cuvsMgIvfFlatSearchParams_t params,
                                cuvsMgIvfFlatIndex_t index,
                                DLManagedTensor* queries_tensor,
                                DLManagedTensor* neighbors_tensor,
                                DLManagedTensor* distances_tensor);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_flat_c_index_extend Multi-GPU IVF-Flat index extend
 * @{
 */

/**
 * @brief Extend a Multi-GPU IVF-Flat index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in,out] index Multi-GPU IVF-Flat index to extend
 * @param[in] new_vectors_tensor DLManagedTensor* new vectors to add
 * @param[in] new_indices_tensor DLManagedTensor* new indices (optional, can be NULL)
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatExtend(cuvsResources_t res,
                                cuvsMgIvfFlatIndex_t index,
                                DLManagedTensor* new_vectors_tensor,
                                DLManagedTensor* new_indices_tensor);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_flat_c_index_serialize Multi-GPU IVF-Flat index serialize
 * @{
 */

/**
 * @brief Serialize a Multi-GPU IVF-Flat index to file
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index Multi-GPU IVF-Flat index to serialize
 * @param[in] filename Path to the output file
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatSerialize(cuvsResources_t res,
                                   cuvsMgIvfFlatIndex_t index,
                                   const char* filename);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_flat_c_index_deserialize Multi-GPU IVF-Flat index deserialize
 * @{
 */

/**
 * @brief Deserialize a Multi-GPU IVF-Flat index from file
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename Path to the input file
 * @param[out] index Multi-GPU IVF-Flat index
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatDeserialize(cuvsResources_t res,
                                     const char* filename,
                                     cuvsMgIvfFlatIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_flat_c_index_distribute Multi-GPU IVF-Flat index distribute
 * @{
 */

/**
 * @brief Distribute a local IVF-Flat index to create a Multi-GPU index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename Path to the local index file
 * @param[out] index Multi-GPU IVF-Flat index
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfFlatDistribute(cuvsResources_t res,
                                    const char* filename,
                                    cuvsMgIvfFlatIndex_t index);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

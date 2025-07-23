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
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/mg_common.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup mg_ivf_pq_c_index_params Multi-GPU IVF-PQ index build parameters
 * @{
 */

/**
 * @brief Multi-GPU parameters to build IVF-PQ Index
 *
 * This structure extends the base IVF-PQ index parameters with multi-GPU specific settings.
 */
struct cuvsMgIvfPqIndexParams {
  /** Base IVF-PQ index parameters */
  cuvsIvfPqIndexParams_t base_params;
  /** Distribution mode for multi-GPU setup */
  cuvsMgDistributionMode mode;
};

typedef struct cuvsMgIvfPqIndexParams* cuvsMgIvfPqIndexParams_t;

/**
 * @brief Allocate Multi-GPU IVF-PQ Index params, and populate with default values
 *
 * @param[in] index_params cuvsMgIvfPqIndexParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqIndexParamsCreate(cuvsMgIvfPqIndexParams_t* index_params);

/**
 * @brief De-allocate Multi-GPU IVF-PQ Index params
 *
 * @param[in] index_params
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqIndexParamsDestroy(cuvsMgIvfPqIndexParams_t index_params);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_pq_c_search_params Multi-GPU IVF-PQ index search parameters
 * @{
 */

/**
 * @brief Multi-GPU parameters to search IVF-PQ index
 *
 * This structure extends the base IVF-PQ search parameters with multi-GPU specific settings.
 */
struct cuvsMgIvfPqSearchParams {
  /** Base IVF-PQ search parameters */
  cuvsIvfPqSearchParams_t base_params;
  /** Replicated search mode */
  cuvsMgReplicatedSearchMode search_mode;
  /** Sharded merge mode */
  cuvsMgShardedMergeMode merge_mode;
  /** Number of rows per batch */
  int64_t n_rows_per_batch;
};

typedef struct cuvsMgIvfPqSearchParams* cuvsMgIvfPqSearchParams_t;

/**
 * @brief Allocate Multi-GPU IVF-PQ search params, and populate with default values
 *
 * @param[in] params cuvsMgIvfPqSearchParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqSearchParamsCreate(cuvsMgIvfPqSearchParams_t* params);

/**
 * @brief De-allocate Multi-GPU IVF-PQ search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqSearchParamsDestroy(cuvsMgIvfPqSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_pq_c_index Multi-GPU IVF-PQ index
 * @{
 */

/**
 * @brief Struct to hold address of cuvs::neighbors::mg_index<ivf_pq::index> and its active trained
 * dtype
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsMgIvfPqIndex;

typedef cuvsMgIvfPqIndex* cuvsMgIvfPqIndex_t;

/**
 * @brief Allocate Multi-GPU IVF-PQ index
 *
 * @param[in] index cuvsMgIvfPqIndex_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqIndexCreate(cuvsMgIvfPqIndex_t* index);

/**
 * @brief De-allocate Multi-GPU IVF-PQ index
 *
 * @param[in] index cuvsMgIvfPqIndex_t to de-allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqIndexDestroy(cuvsMgIvfPqIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_pq_c_index_build Multi-GPU IVF-PQ index build
 * @{
 */

/**
 * @brief Build a Multi-GPU IVF-PQ index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params Multi-GPU IVF-PQ index parameters
 * @param[in] dataset_tensor DLManagedTensor* training dataset
 * @param[out] index Multi-GPU IVF-PQ index
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqBuild(cuvsResources_t res,
                             cuvsMgIvfPqIndexParams_t params,
                             DLManagedTensor* dataset_tensor,
                             cuvsMgIvfPqIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_pq_c_index_search Multi-GPU IVF-PQ index search
 * @{
 */

/**
 * @brief Search a Multi-GPU IVF-PQ index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params Multi-GPU IVF-PQ search parameters
 * @param[in] index Multi-GPU IVF-PQ index
 * @param[in] queries_tensor DLManagedTensor* queries dataset
 * @param[out] neighbors_tensor DLManagedTensor* output neighbors
 * @param[out] distances_tensor DLManagedTensor* output distances
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqSearch(cuvsResources_t res,
                              cuvsMgIvfPqSearchParams_t params,
                              cuvsMgIvfPqIndex_t index,
                              DLManagedTensor* queries_tensor,
                              DLManagedTensor* neighbors_tensor,
                              DLManagedTensor* distances_tensor);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_pq_c_index_extend Multi-GPU IVF-PQ index extend
 * @{
 */

/**
 * @brief Extend a Multi-GPU IVF-PQ index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in,out] index Multi-GPU IVF-PQ index to extend
 * @param[in] new_vectors_tensor DLManagedTensor* new vectors to add
 * @param[in] new_indices_tensor DLManagedTensor* new indices (optional, can be NULL)
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqExtend(cuvsResources_t res,
                              cuvsMgIvfPqIndex_t index,
                              DLManagedTensor* new_vectors_tensor,
                              DLManagedTensor* new_indices_tensor);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_pq_c_index_serialize Multi-GPU IVF-PQ index serialize
 * @{
 */

/**
 * @brief Serialize a Multi-GPU IVF-PQ index to file
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index Multi-GPU IVF-PQ index to serialize
 * @param[in] filename Path to the output file
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqSerialize(cuvsResources_t res,
                                 cuvsMgIvfPqIndex_t index,
                                 const char* filename);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_pq_c_index_deserialize Multi-GPU IVF-PQ index deserialize
 * @{
 */

/**
 * @brief Deserialize a Multi-GPU IVF-PQ index from file
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename Path to the input file
 * @param[out] index Multi-GPU IVF-PQ index
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqDeserialize(cuvsResources_t res,
                                   const char* filename,
                                   cuvsMgIvfPqIndex_t index);

/**
 * @}
 */

/**
 * @defgroup mg_ivf_pq_c_index_distribute Multi-GPU IVF-PQ index distribute
 * @{
 */

/**
 * @brief Distribute a local IVF-PQ index to create a Multi-GPU index
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename Path to the local index file
 * @param[out] index Multi-GPU IVF-PQ index
 * @return cuvsError_t
 */
cuvsError_t cuvsMgIvfPqDistribute(cuvsResources_t res,
                                  const char* filename,
                                  cuvsMgIvfPqIndex_t index);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

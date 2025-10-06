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

#pragma once

#include <cuvs/core/c_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup mg_c_common_types Multi-GPU common types and enums
 * @{
 */

/**
 * @brief Distribution mode for multi-GPU indexes
 */
typedef enum {
  /** Index is replicated on each device, favors throughput */
  CUVS_NEIGHBORS_MG_REPLICATED,
  /** Index is split on several devices, favors scaling */
  CUVS_NEIGHBORS_MG_SHARDED
} cuvsMultiGpuDistributionMode;

/**
 * @brief Search mode when using a replicated index
 */
typedef enum {
  /** Search queries are split to maintain equal load on GPUs */
  CUVS_NEIGHBORS_MG_LOAD_BALANCER,
  /** Each search query is processed by a single GPU in a round-robin fashion */
  CUVS_NEIGHBORS_MG_ROUND_ROBIN
} cuvsMultiGpuReplicatedSearchMode;

/**
 * @brief Merge mode when using a sharded index
 */
typedef enum {
  /** Search batches are merged on the root rank */
  CUVS_NEIGHBORS_MG_MERGE_ON_ROOT_RANK,
  /** Search batches are merged in a tree reduction fashion */
  CUVS_NEIGHBORS_MG_TREE_MERGE
} cuvsMultiGpuShardedMergeMode;

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

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

#include <cuvs/distance/distance.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup filters Filters APIs
 * @brief APIs related to filter functionality.
 * @{
 */

/**
 * @brief Enum to denote filter type.
 */
enum cuvsFilterType {
  /* No filter */
  NO_FILTER,
  /* Filter an index with a bitset */
  BITSET,
  /* Filter an index with a bitmap */
  BITMAP
};

/**
 * @brief Struct to hold address of cuvs::neighbors::prefilter and its type
 *
 */
typedef struct {
  uintptr_t addr;
  enum cuvsFilterType type;
} cuvsFilter;

/**
 * @}
 */

/**
 * @defgroup index_merge Index Merge
 * @brief Common definitions related to index merging.
 * @{
 */

/**
 * @brief Strategy for merging indices.
 */
typedef enum {
  MERGE_STRATEGY_PHYSICAL = 0,  ///< Merge indices physically
  MERGE_STRATEGY_LOGICAL  = 1   ///< Merge indices logically
} cuvsMergeStrategy;

/**
 * @}
 */
#ifdef __cplusplus
}
#endif

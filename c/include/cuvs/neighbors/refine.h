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
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @defgroup ann_refine_c Approximate Nearest Neighbors Refinement C-API
 * @{
 */
/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] dataset device matrix that stores the dataset [n_rows, dims]
 * @param[in] queries device matrix of the queries [n_queris, dims]
 * @param[in] candidates indices of candidate vectors [n_queries, n_candidates], where
 *   n_candidates >= k
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 * @param[out] indices device matrix that stores the refined indices [n_queries, k]
 * @param[out] distances device matrix that stores the refined distances [n_queries, k]
 */
cuvsError_t cuvsRefine(cuvsResources_t res,
                       DLManagedTensor* dataset,
                       DLManagedTensor* queries,
                       DLManagedTensor* candidates,
                       cuvsDistanceType metric,
                       DLManagedTensor* indices,
                       DLManagedTensor* distances);
/**
 * @}
 */

#ifdef __cplusplus
}
#endif

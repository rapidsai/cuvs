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
#include <dlpack/dlpack.h>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute pairwise distances for two matrices
 *
 *
 * Usage example:
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/distance/pairwise_distance.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor x;
 * DLManagedTensor y;
 * DLManagedTensor dist;
 *
 * cuvsPairwiseDistance(res, &x, &y, &dist, L2SqrtUnexpanded, 2.0);
 * @endcode
 *
 * @param[in] res cuvs resources object for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
cuvsError_t cuvsPairwiseDistance(cuvsResources_t res,
                                 DLManagedTensor* x,
                                 DLManagedTensor* y,
                                 DLManagedTensor* dist,
                                 cuvsDistanceType metric,
                                 float metric_arg);
#ifdef __cplusplus
}
#endif

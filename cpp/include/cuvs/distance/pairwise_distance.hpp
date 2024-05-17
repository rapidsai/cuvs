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

#include <cstdint>
#include <cuvs/distance/distance_types.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::distance {

/**
 * @defgroup pairwise_distance Pairwise Distances API
 * @{
 */

/**
 * @brief Compute pairwise distances for two matrices
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <cuvs/distance/pairwise_distance.hpp>
 *
 * raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<float>(handle, n_samples, n_features);
 *
 * // ... fill input with data ...
 *
 * auto output = raft::make_device_matrix<float>(handle, n_samples, n_samples);
 *
 * auto metric = cuvs::distance::DistanceType::L2SqrtExpanded;
 * cuvs::distance::pairwise_distance(handle,
 *                                   raft::make_const(input.view()),
 *                                   raft::make_const(input.view()),
 *                                   output.view(),
 *                                   metric);
 * @endcode
 *
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous> dist,
  cuvs::distance::DistanceType metric,
  float metric_arg = 2.0f);

/**
 * @brief Compute pairwise distances for two matrices
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <cuvs/distance/pairwise_distance.hpp>
 *
 * raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<double>(handle, n_samples, n_features);
 *
 * // ... fill input with data ...
 *
 * auto output = raft::make_device_matrix<double>(handle, n_samples, n_samples);
 *
 * auto metric = cuvs::distance::DistanceType::L2SqrtExpanded;
 * cuvs::distance::pairwise_distance(handle,
 *                                   raft::make_const(input.view()),
 *                                   raft::make_const(input.view()),
 *                                   output.view(),
 *                                   metric);
 * @endcode
 *
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const y,
  raft::device_matrix_view<double, std::int64_t, raft::layout_c_contiguous> dist,
  cuvs::distance::DistanceType metric,
  double metric_arg = 2.0f);

/**
 * @brief Compute pairwise distances for two matrices
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <cuvs/distance/pairwise_distance.hpp>
 *
 * raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<float>(handle, n_samples, n_features);
 *
 * // ... fill input with data ...
 *
 * auto output = raft::make_device_matrix<float>(handle, n_samples, n_samples);
 *
 * auto metric = cuvs::distance::DistanceType::L2SqrtExpanded;
 * cuvs::distance::pairwise_distance(handle,
 *                                   raft::make_const(input.view()),
 *                                   raft::make_const(input.view()),
 *                                   output.view(),
 *                                   metric);
 * @endcode
 *
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const x,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous> dist,
  cuvs::distance::DistanceType metric,
  float metric_arg = 2.0f);
/**
 * @brief Compute pairwise distances for two matrices
 *
 * Note: Only contiguous row- or column-major layouts supported currently.
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_mdarray.hpp>
 * #include <cuvs/distance/pairwise_distance.hpp>
 *
 * raft::resources handle;
 * int n_samples = 5000;
 * int n_features = 50;
 *
 * auto input = raft::make_device_matrix<double>(handle, n_samples, n_features);
 *
 * // ... fill input with data ...
 *
 * auto output = raft::make_device_matrix<double>(handle, n_samples, n_samples);
 *
 * auto metric = cuvs::distance::DistanceType::L2SqrtExpanded;
 * cuvs::distance::pairwise_distance(handle,
 *                                   raft::make_const(input.view()),
 *                                   raft::make_const(input.view()),
 *                                   output.view(),
 *                                   metric);
 * @endcode
 *
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const x,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const y,
  raft::device_matrix_view<double, std::int64_t, raft::layout_f_contiguous> dist,
  cuvs::distance::DistanceType metric,
  double metric_arg = 2.0f);

/** @} */  // end group pairwise_distance_runtime

}  // namespace cuvs::distance

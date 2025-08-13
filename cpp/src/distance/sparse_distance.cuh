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

#include "detail/sparse/bin_distance.cuh"
#include "detail/sparse/common.hpp"
#include "detail/sparse/ip_distance.cuh"
#include "detail/sparse/l2_distance.cuh"
#include "detail/sparse/lp_distance.cuh"

#include <cuvs/distance/distance.hpp>

#include <raft/core/device_csr_matrix.hpp>

#include <unordered_set>

namespace cuvs {
namespace distance {
/**
 * Compute pairwise distances between A and B, using the provided
 * input configuration and distance function.
 *
 * @tparam value_idx index type
 * @tparam value_t value type
 * @param[out] out dense output array (size A.nrows * B.nrows)
 * @param[in] input_config input argument configuration
 * @param[in] metric distance metric to use
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
template <typename value_idx = int, typename value_t = float>
void pairwiseDistance(value_t* out,
                      detail::sparse::distances_config_t<value_idx, value_t> input_config,
                      cuvs::distance::DistanceType metric,
                      float metric_arg)
{
  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded:
      detail::sparse::l2_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::L2SqrtExpanded:
      detail::sparse::l2_sqrt_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::InnerProduct:
      detail::sparse::ip_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::L2Unexpanded:
      detail::sparse::l2_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::L2SqrtUnexpanded:
      detail::sparse::l2_sqrt_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::L1:
      detail::sparse::l1_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::LpUnexpanded:
      detail::sparse::lp_unexpanded_distances_t<value_idx, value_t>(input_config, metric_arg)
        .compute(out);
      break;
    case cuvs::distance::DistanceType::Linf:
      detail::sparse::linf_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::Canberra:
      detail::sparse::canberra_unexpanded_distances_t<value_idx, value_t>(input_config)
        .compute(out);
      break;
    case cuvs::distance::DistanceType::JaccardExpanded:
      detail::sparse::jaccard_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::CosineExpanded:
      detail::sparse::cosine_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::HellingerExpanded:
      detail::sparse::hellinger_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::DiceExpanded:
      detail::sparse::dice_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::CorrelationExpanded:
      detail::sparse::correlation_expanded_distances_t<value_idx, value_t>(input_config)
        .compute(out);
      break;
    case cuvs::distance::DistanceType::RusselRaoExpanded:
      detail::sparse::russelrao_expanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::HammingUnexpanded:
      detail::sparse::hamming_unexpanded_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case cuvs::distance::DistanceType::JensenShannon:
      detail::sparse::jensen_shannon_unexpanded_distances_t<value_idx, value_t>(input_config)
        .compute(out);
      break;
    case cuvs::distance::DistanceType::KLDivergence:
      detail::sparse::kl_divergence_unexpanded_distances_t<value_idx, value_t>(input_config)
        .compute(out);
      break;

    default: THROW("Unsupported distance: %d", metric);
  }
}
};  // namespace distance
};  // namespace cuvs

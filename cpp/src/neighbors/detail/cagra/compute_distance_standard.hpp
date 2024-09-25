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

#include "compute_distance.hpp"

#include <cuvs/distance/distance.hpp>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
struct standard_descriptor_spec : public instance_spec<DataT, IndexT, DistanceT> {
  using base_type = instance_spec<DataT, IndexT, DistanceT>;
  using typename base_type::data_type;
  using typename base_type::distance_type;
  using typename base_type::host_type;
  using typename base_type::index_type;

  template <typename DatasetT>
  constexpr static inline bool accepts_dataset()
  {
    return is_strided_dataset_v<DatasetT>;
  }

  template <typename DatasetT>
  static auto init(const cagra::search_params& params,
                   const DatasetT& dataset,
                   cuvs::distance::DistanceType metric,
                   rmm::cuda_stream_view stream) -> host_type
  {
    return init_(params,
                 dataset.view().data_handle(),
                 IndexT(dataset.n_rows()),
                 dataset.dim(),
                 dataset.stride(),
                 stream);
  }

  template <typename DatasetT>
  static auto priority(const cagra::search_params& params,
                       const DatasetT& dataset,
                       cuvs::distance::DistanceType metric) -> double
  {
    // If explicit team_size is specified and doesn't match the instance, discard it
    if (params.team_size != 0 && TeamSize != params.team_size) { return -1.0; }
    if (Metric != metric) { return -1.0; }
    // Otherwise, favor the closest dataset dimensionality.
    return 1.0 / (0.1 + std::abs(double(dataset.dim()) - double(DatasetBlockDim)));
  }

 private:
  static dataset_descriptor_host<DataT, IndexT, DistanceT> init_(const cagra::search_params& params,
                                                                 const DataT* ptr,
                                                                 IndexT size,
                                                                 uint32_t dim,
                                                                 uint32_t ld,
                                                                 rmm::cuda_stream_view stream);
};

}  // namespace cuvs::neighbors::cagra::detail

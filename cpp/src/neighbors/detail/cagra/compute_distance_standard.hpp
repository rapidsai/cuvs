/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
          typename index_t,
          typename distance_t>
struct standard_descriptor_spec : public instance_spec<DataT, index_t, distance_t> {
  using base_type = instance_spec<DataT, index_t, distance_t>;
  using typename base_type::data_type;
  using typename base_type::distance_type;
  using typename base_type::host_type;
  using typename base_type::index_type;

  template <typename DatasetT>
  constexpr static inline auto accepts_dataset() -> bool
  {
    return is_strided_dataset_v<DatasetT>;
  }

  template <typename DatasetT>
  static auto init(const cagra::search_params& params,
                   const DatasetT& dataset,
                   cuvs::distance::DistanceType metric,
                   const distance_t* dataset_norms = nullptr) -> host_type
  {
    return init_(params,
                 dataset.view().data_handle(),
                 index_t(dataset.n_rows()),
                 dataset.dim(),
                 dataset.stride(),
                 dataset_norms);
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
  static auto init_(const cagra::search_params& params,
                    const DataT* ptr,
                    index_t size,
                    uint32_t dim,
                    uint32_t ld,
                    const distance_t* dataset_norms = nullptr)
    -> dataset_descriptor_host<DataT, index_t, distance_t>;
};

}  // namespace cuvs::neighbors::cagra::detail

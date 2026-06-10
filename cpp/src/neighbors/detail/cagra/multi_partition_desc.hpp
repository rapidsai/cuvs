/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <neighbors/detail/cagra/compute_distance.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

template <typename DataT, typename IndexT, typename DistanceT>
struct alignas(16) multi_partition_desc_t {
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc;
  const IndexT* graph;
  uint32_t graph_degree;
  uint32_t _pad;
};

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

template <typename DataT, typename IndexT, typename DistanceT>
struct alignas(16) multi_partition_desc_t {
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc;
  const IndexT* graph;
  uint32_t graph_degree;
  uint32_t _pad;
};

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search

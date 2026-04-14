/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::detail {

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_none(uint32_t query_id, IndexT node_id, void* filter_data);

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_bitset(uint32_t query_id, IndexT node_id, void* filter_data);

}  // namespace cuvs::neighbors::detail

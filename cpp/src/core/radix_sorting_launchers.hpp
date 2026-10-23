/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <cuda/std/utility>
#include <cuda_runtime_api.h>
#include <iterator>

namespace cuvs::detail {
template <typename KeyIn, typename KeyOut, typename ValueIn, typename ValueOut, typename NumItems>
cudaError_t cub_sort_pairs(void* storage,
                           size_t& bytes,
                           KeyIn keys_in,
                           KeyOut keys_out,
                           ValueIn values_in,
                           ValueOut values_out,
                           NumItems count,
                           int begin_bit,
                           int end_bit,
                           cudaStream_t stream);
template <typename InputIt, typename OutputIt, typename NumItems>
cudaError_t cub_sort_keys_descending(void* storage,
                                     size_t& bytes,
                                     InputIt input,
                                     OutputIt output,
                                     NumItems count,
                                     int begin_bit,
                                     int end_bit,
                                     cudaStream_t stream);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename Policy, typename Iterator>
void thrust_sort(Policy const& policy, Iterator first, Iterator last);

template <typename Policy, typename Iterator, typename Compare>
void thrust_sort(Policy const& policy, Iterator first, Iterator last, Compare compare);

template <typename Policy, typename Key, typename Value>
void thrust_sort_by_key(Policy const& policy, Key first, Key last, Value values);

template <typename Policy, typename Key, typename Value, typename Compare>
void thrust_sort_by_key(Policy const& policy, Key first, Key last, Value values, Compare compare);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename Key, typename Value>
void thrust_sort_by_key(Key first, Key last, Value values);
}

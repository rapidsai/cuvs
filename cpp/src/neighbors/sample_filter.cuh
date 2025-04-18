/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/common.hpp>
#include <raft/core/bitmap.cuh>
#include <raft/core/bitset.cuh>
#include <raft/core/detail/macros.hpp>
#include <raft/sparse/convert/csr.cuh>

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::filtering {

/* A filter that filters nothing. This is the default behavior. */
inline _RAFT_HOST_DEVICE bool none_sample_filter::operator()(
  // query index
  const uint32_t query_ix,
  // the current inverted list index
  const uint32_t cluster_ix,
  // the index of the current sample inside the current inverted list
  const uint32_t sample_ix) const
{
  return true;
}

/* A filter that filters nothing. This is the default behavior. */
inline _RAFT_HOST_DEVICE bool none_sample_filter::operator()(
  // query index
  const uint32_t query_ix,
  // the index of the current sample
  const uint32_t sample_ix) const
{
  return true;
}

template <typename filter_t, typename = void>
struct takes_three_args : std::false_type {};
template <typename filter_t>
struct takes_three_args<
  filter_t,
  std::void_t<decltype(std::declval<filter_t>()(uint32_t{}, uint32_t{}, uint32_t{}))>>
  : std::true_type {};

/**
 * @brief Filter used to convert the cluster index and sample index
 * of an IVF search into a sample index. This can be used as an
 * intermediate filter.
 *
 * @tparam index_t Indexing type
 * @tparam filter_t
 */
template <typename index_t, typename filter_t>
ivf_to_sample_filter<index_t, filter_t>::ivf_to_sample_filter(const index_t* const* inds_ptrs,
                                                              const filter_t next_filter)
  : inds_ptrs_{inds_ptrs}, next_filter_{next_filter}
{
}

/** If the original filter takes three arguments, then don't modify the arguments.
 * If the original filter takes two arguments, then we are using `inds_ptr_` to obtain the sample
 * index.
 */
template <typename index_t, typename filter_t>
inline _RAFT_HOST_DEVICE bool ivf_to_sample_filter<index_t, filter_t>::operator()(
  // query index
  const uint32_t query_ix,
  // the current inverted list index
  const uint32_t cluster_ix,
  // the index of the current sample inside the current inverted list
  const uint32_t sample_ix) const
{
  if constexpr (takes_three_args<filter_t>::value) {
    return next_filter_(query_ix, cluster_ix, sample_ix);
  } else {
    return next_filter_(query_ix, inds_ptrs_[cluster_ix][sample_ix]);
  }
}

template <typename bitset_t, typename index_t>
bitset_filter<bitset_t, index_t>::bitset_filter(
  const cuvs::core::bitset_view<bitset_t, index_t> bitset_for_filtering)
  : bitset_view_{bitset_for_filtering}
{
}

template <typename bitset_t, typename index_t>
inline _RAFT_HOST_DEVICE bool bitset_filter<bitset_t, index_t>::operator()(
  // query index
  const uint32_t query_ix,
  // the index of the current sample
  const uint32_t sample_ix) const
{
  return bitset_view_.test(sample_ix);
}

template <typename bitset_t, typename index_t>
template <typename csr_matrix_t>
void bitset_filter<bitset_t, index_t>::to_csr(raft::resources const& handle, csr_matrix_t& csr)
{
  raft::sparse::convert::bitset_to_csr(handle, bitset_view_, csr);
}

template <typename bitmap_t, typename index_t>
bitmap_filter<bitmap_t, index_t>::bitmap_filter(
  const cuvs::core::bitmap_view<bitmap_t, index_t> bitmap_for_filtering)
  : bitmap_view_{bitmap_for_filtering}
{
}

template <typename bitmap_t, typename index_t>
inline _RAFT_HOST_DEVICE bool bitmap_filter<bitmap_t, index_t>::operator()(
  // query index
  const uint32_t query_ix,
  // the index of the current sample
  const uint32_t sample_ix) const
{
  return bitmap_view_.test(query_ix, sample_ix);
}

template <typename bitmap_t, typename index_t>
template <typename csr_matrix_t>
void bitmap_filter<bitmap_t, index_t>::to_csr(raft::resources const& handle, csr_matrix_t& csr)
{
  raft::sparse::convert::bitmap_to_csr(handle, bitmap_view_, csr);
}

}  // namespace cuvs::neighbors::filtering

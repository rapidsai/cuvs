/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
_RAFT_HOST_DEVICE ivf_to_sample_filter<index_t, filter_t>::ivf_to_sample_filter(
  const index_t* const* inds_ptrs, const filter_t next_filter)
  : inds_ptrs_{inds_ptrs}, next_filter_{next_filter}
{
}

template <typename index_t, typename filter_t>
_RAFT_HOST_DEVICE ivf_to_sample_filter<index_t, filter_t>::~ivf_to_sample_filter() = default;

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
_RAFT_HOST_DEVICE bitset_filter<bitset_t, index_t>::bitset_filter(
  const cuvs::core::bitset_view<bitset_t, index_t> bitset_for_filtering)
  : bitset_view_{bitset_for_filtering}
{
}

template <typename bitset_t, typename index_t>
_RAFT_HOST_DEVICE bitset_filter<bitset_t, index_t>::~bitset_filter() = default;

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

struct ivf_to_sample_filter_dev {
  FilterType tag;
  const void* obj_ptr;

  __device__ bool operator()(uint32_t q, uint32_t c, uint32_t s) const
  {
    switch (tag) {
      case FilterType::None:
        return reinterpret_cast<
                 const filtering::ivf_to_sample_filter<int64_t, filtering::none_sample_filter>*>(
                 obj_ptr)
          ->operator()(q, c, s);
      case FilterType::Bitset:
        return reinterpret_cast<const filtering::ivf_to_sample_filter<
          int64_t,
          filtering::bitset_filter<uint32_t, int64_t>>*>(obj_ptr)
          ->operator()(q, c, s);
      default: return true;
    }
  }
};

namespace {
template <typename dev_filter_t>
__global__ void destruct_dev_filter(dev_filter_t* p_filter)
{
  p_filter->~dev_filter_t();
}

template <typename InnerFilterT, typename... Args>
__global__ void init_inner_filter(InnerFilterT* p_inner_filter, Args... args)
{
  new (p_inner_filter) InnerFilterT(args...);
}

template <typename OuterFilterT, typename InnerFilterT, typename... Args>
__global__ void init_outer_filter(OuterFilterT* p_outer_filter,
                                  const int64_t* const* inds_ptrs,
                                  InnerFilterT* p_inner_filter)
{
  new (p_outer_filter) OuterFilterT(inds_ptrs, *p_inner_filter);
}
}  // namespace

/* Device side filter wrapper for ivf_to_sample_filter. ivf_to_sample_filter_dev is used to avoid
 * dynamic dispatching on device. */
template <typename InnerFilterT, typename... InnerArgs>
struct ivf_to_sample_dev_wrapper {
  using OuterFilterT = filtering::ivf_to_sample_filter<int64_t, InnerFilterT>;
  OuterFilterT* p_outer_filter;
  InnerFilterT* p_inner_filter;
  filtering::ivf_to_sample_filter_dev h_placeholder;

  ivf_to_sample_dev_wrapper(FilterType tag,
                            const int64_t* const* inds_ptrs,
                            InnerArgs... inner_args)
  {
    cudaMalloc(&p_inner_filter, sizeof(InnerFilterT));
    init_inner_filter<<<1, 1>>>(p_inner_filter, inner_args...);
    cudaDeviceSynchronize();

    cudaMalloc(&p_outer_filter, sizeof(OuterFilterT));
    init_outer_filter<<<1, 1>>>(p_outer_filter, inds_ptrs, p_inner_filter);
    cudaDeviceSynchronize();

    h_placeholder = {tag, p_outer_filter};
  }

  ~ivf_to_sample_dev_wrapper()
  {
    destruct_dev_filter<<<1, 1>>>(p_outer_filter);
    cudaDeviceSynchronize();
    cudaFree(p_outer_filter);

    destruct_dev_filter<<<1, 1>>>(p_inner_filter);
    cudaDeviceSynchronize();
    cudaFree(p_inner_filter);
  }

  filtering::ivf_to_sample_filter_dev get_dev_filter() const { return h_placeholder; }
};

using ivf_to_sample_dev_none = filtering::ivf_to_sample_dev_wrapper<filtering::none_sample_filter>;
using ivf_to_sample_dev_bitset =
  filtering::ivf_to_sample_dev_wrapper<filtering::bitset_filter<uint32_t, int64_t>,
                                       cuvs::core::bitset_view<uint32_t, int64_t>>;

}  // namespace cuvs::neighbors::filtering

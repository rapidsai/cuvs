/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../core/cub_runtime_launchers.hpp"
#include <cuvs/core/cuda_fp16.hpp>

#include "ivf_pq_fp16_overflow.cuh"

namespace cuvs::neighbors::ivf_pq::helpers {

template <typename DataT, typename Accessor>
bool detect_fp16_overflow(
  raft::resources const& handle,
  const cuvs::neighbors::ivf_pq::index<int64_t>& index,
  cuvs::neighbors::ivf_pq::search_params search_params,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  uint32_t k)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("fp16_ovfl_detect");
  const int64_t n_rows = dataset.extent(0);
  if (n_rows == 0) { return false; }

  auto stream       = raft::resource::get_cuda_stream(handle);
  const int64_t dim = dataset.extent(1);

  constexpr int64_t kMaxSampleQueries = 128;
  const int64_t n_sample              = std::min<int64_t>(n_rows, kMaxSampleQueries);
  const uint32_t top_k                = std::min<uint32_t>(static_cast<uint32_t>(n_rows), k);

  auto mr = raft::resource::get_workspace_resource_ref(handle);
  auto queries =
    raft::make_device_mdarray<DataT>(handle, mr, raft::make_extents<int64_t>(n_sample, dim));
  raft::copy(queries.data_handle(), dataset.data_handle(), n_sample * dim, stream);

  auto neighbors =
    raft::make_device_mdarray<int64_t>(handle, mr, raft::make_extents<int64_t>(n_sample, top_k));
  auto distances =
    raft::make_device_mdarray<float>(handle, mr, raft::make_extents<int64_t>(n_sample, top_k));

  cuvs::neighbors::ivf_pq::search(handle,
                                  search_params,
                                  index,
                                  raft::make_const_mdspan(queries.view()),
                                  neighbors.view(),
                                  distances.view());

  const int64_t count       = n_sample * static_cast<int64_t>(top_k);
  const bool any_non_finite = cuvs::detail::has_non_finite(handle, distances.data_handle(), count);
  raft::resource::sync_stream(handle);
  return any_non_finite;
}

template bool cuvs::neighbors::ivf_pq::helpers::detect_fp16_overflow<
  __half,
  raft::host_device_accessor<::cuda::std::default_accessor<__half const>, (raft::memory_type)0>>(
  raft::resources const&,
  cuvs::neighbors::ivf_pq::index<long> const&,
  cuvs::neighbors::ivf_pq::search_params,
  ::cuda::std::mdspan<
    __half const,
    ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
    ::cuda::std::layout_right,
    raft::host_device_accessor<::cuda::std::default_accessor<__half const>, (raft::memory_type)0>>,
  unsigned int);
template bool cuvs::neighbors::ivf_pq::helpers::detect_fp16_overflow<
  __half,
  raft::host_device_accessor<::cuda::std::default_accessor<__half const>, (raft::memory_type)2>>(
  raft::resources const&,
  cuvs::neighbors::ivf_pq::index<long> const&,
  cuvs::neighbors::ivf_pq::search_params,
  ::cuda::std::mdspan<
    __half const,
    ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
    ::cuda::std::layout_right,
    raft::host_device_accessor<::cuda::std::default_accessor<__half const>, (raft::memory_type)2>>,
  unsigned int);
template bool cuvs::neighbors::ivf_pq::helpers::detect_fp16_overflow<
  float,
  raft::host_device_accessor<::cuda::std::default_accessor<float const>, (raft::memory_type)0>>(
  raft::resources const&,
  cuvs::neighbors::ivf_pq::index<long> const&,
  cuvs::neighbors::ivf_pq::search_params,
  ::cuda::std::mdspan<
    float const,
    ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
    ::cuda::std::layout_right,
    raft::host_device_accessor<::cuda::std::default_accessor<float const>, (raft::memory_type)0>>,
  unsigned int);
template bool cuvs::neighbors::ivf_pq::helpers::detect_fp16_overflow<
  float,
  raft::host_device_accessor<::cuda::std::default_accessor<float const>, (raft::memory_type)2>>(
  raft::resources const&,
  cuvs::neighbors::ivf_pq::index<long> const&,
  cuvs::neighbors::ivf_pq::search_params,
  ::cuda::std::mdspan<
    float const,
    ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
    ::cuda::std::layout_right,
    raft::host_device_accessor<::cuda::std::default_accessor<float const>, (raft::memory_type)2>>,
  unsigned int);
template bool cuvs::neighbors::ivf_pq::helpers::detect_fp16_overflow<
  signed char,
  raft::host_device_accessor<::cuda::std::default_accessor<signed char const>,
                             (raft::memory_type)0>>(
  raft::resources const&,
  cuvs::neighbors::ivf_pq::index<long> const&,
  cuvs::neighbors::ivf_pq::search_params,
  ::cuda::std::mdspan<signed char const,
                      ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
                      ::cuda::std::layout_right,
                      raft::host_device_accessor<::cuda::std::default_accessor<signed char const>,
                                                 (raft::memory_type)0>>,
  unsigned int);
template bool cuvs::neighbors::ivf_pq::helpers::detect_fp16_overflow<
  signed char,
  raft::host_device_accessor<::cuda::std::default_accessor<signed char const>,
                             (raft::memory_type)2>>(
  raft::resources const&,
  cuvs::neighbors::ivf_pq::index<long> const&,
  cuvs::neighbors::ivf_pq::search_params,
  ::cuda::std::mdspan<signed char const,
                      ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
                      ::cuda::std::layout_right,
                      raft::host_device_accessor<::cuda::std::default_accessor<signed char const>,
                                                 (raft::memory_type)2>>,
  unsigned int);
template bool cuvs::neighbors::ivf_pq::helpers::detect_fp16_overflow<
  unsigned char,
  raft::host_device_accessor<::cuda::std::default_accessor<unsigned char const>,
                             (raft::memory_type)0>>(
  raft::resources const&,
  cuvs::neighbors::ivf_pq::index<long> const&,
  cuvs::neighbors::ivf_pq::search_params,
  ::cuda::std::mdspan<unsigned char const,
                      ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
                      ::cuda::std::layout_right,
                      raft::host_device_accessor<::cuda::std::default_accessor<unsigned char const>,
                                                 (raft::memory_type)0>>,
  unsigned int);
template bool cuvs::neighbors::ivf_pq::helpers::detect_fp16_overflow<
  unsigned char,
  raft::host_device_accessor<::cuda::std::default_accessor<unsigned char const>,
                             (raft::memory_type)2>>(
  raft::resources const&,
  cuvs::neighbors::ivf_pq::index<long> const&,
  cuvs::neighbors::ivf_pq::search_params,
  ::cuda::std::mdspan<unsigned char const,
                      ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
                      ::cuda::std::layout_right,
                      raft::host_device_accessor<::cuda::std::default_accessor<unsigned char const>,
                                                 (raft::memory_type)2>>,
  unsigned int);

}  // namespace cuvs::neighbors::ivf_pq::helpers

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "map_launchers.hpp"

#include <cuvs/core/cuda_fp16.hpp>
#include <raft/linalg/multiply.cuh>

extern template void
raft::linalg::detail::map<false, float, long, raft::plug_const_op<float, raft::mul_op>, float>(
  cuda::stream_ref, float*, long, raft::plug_const_op<float, raft::mul_op>, float const*);

namespace cuvs::detail {

template <typename InType, typename OutType, typename ScalarIdxType, typename, typename>
void multiply_scalar(
  raft::resources const& handle,
  InType in,
  OutType out,
  raft::host_scalar_view<const typename InType::value_type, ScalarIdxType> scalar)

{
  return raft::linalg::multiply_scalar(handle, in, out, scalar);
}

}  // namespace cuvs::detail

template void cuvs::detail::multiply_scalar(
  raft::resources const&,
  raft::device_vector_view<float const, long, raft::row_major>,
  raft::device_vector_view<float, long, raft::row_major>,
  cuda::std::mdspan<
    cuda::std::mdspan<float const,
                      cuda::std::extents<long, cuda::std::dynamic_extent>,
                      cuda::std::layout_right,
                      raft::host_device_accessor<cuda::std::default_accessor<float const>,
                                                 (raft::memory_type)2>>::value_type const,
    cuda::std::extents<unsigned int, 1ul>,
    cuda::std::layout_right,
    raft::host_device_accessor<
      cuda::std::default_accessor<
        cuda::std::mdspan<float const,
                          cuda::std::extents<long, cuda::std::dynamic_extent>,
                          cuda::std::layout_right,
                          raft::host_device_accessor<cuda::std::default_accessor<float const>,
                                                     (raft::memory_type)2>>::value_type const>,
      (raft::memory_type)0>>);

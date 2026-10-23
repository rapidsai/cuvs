/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_transform_launchers.hpp"

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_0_TransformKernel_pointer
cuvs::detail::transform_launch::transform_0::TransformKernel()
{
  return &cub::detail::transform::transform_kernel<
    cub::detail::transform::policy_selector_from_types<false, true, cuda::std::tuple<>, double*>,
    long,
    cuda::always_true,
    cub::detail::__return_constant<double>,
    double*>;
}

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_1_TransformKernel_pointer
cuvs::detail::transform_launch::transform_1::TransformKernel()
{
  return &cub::detail::transform::transform_kernel<
    cub::detail::transform::policy_selector_from_types<false, true, cuda::std::tuple<>, float*>,
    long,
    cuda::always_true,
    cub::detail::__return_constant<float>,
    float*>;
}

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_2_TransformKernel_pointer
cuvs::detail::transform_launch::transform_2::TransformKernel()
{
  return &cub::detail::transform::transform_kernel<
    cub::detail::transform::policy_selector_from_types<false, true, cuda::std::tuple<>, int*>,
    long,
    cuda::always_true,
    cub::detail::__return_constant<int>,
    int*>;
}

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_3_TransformKernel_pointer
cuvs::detail::transform_launch::transform_3::TransformKernel()
{
  return &cub::detail::transform::transform_kernel<
    cub::detail::transform::
      policy_selector_from_types<false, true, cuda::std::tuple<>, unsigned int*>,
    long,
    cuda::always_true,
    cub::detail::__return_constant<unsigned int>,
    unsigned int*>;
}

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_4_TransformKernel_pointer
cuvs::detail::transform_launch::transform_4::TransformKernel()
{
  return &cub::detail::transform::transform_kernel<
    cub::detail::transform::policy_selector_from_types<
      false,
      true,
      cuda::std::tuple<cuda::counting_iterator<long, __int128>>,
      int*>,
    long,
    cuda::always_true,
    cuda::__callable_permitting_copied_arguments<
      thrust::system::detail::generic::detail::compute_sequence_value<int>>,
    int*,
    cuda::counting_iterator<long, __int128>>;
}

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_5_TransformKernel_pointer
cuvs::detail::transform_launch::transform_5::TransformKernel()
{
  return &cub::detail::transform::transform_kernel<
    cub::detail::transform::policy_selector_from_types<
      true,
      true,
      cuda::std::tuple<thrust::permutation_iterator<float const*, long*>>,
      float*>,
    long,
    cuda::always_true,
    cuda::std::identity,
    float*,
    thrust::permutation_iterator<float const*, long*>>;
}

template <>
__host__ __device__ cuvs::detail::transform_launch::transform_6_TransformKernel_pointer
cuvs::detail::transform_launch::transform_6::TransformKernel()
{
  return &cub::detail::transform::transform_kernel<
    cub::detail::transform::policy_selector_from_types<
      true,
      true,
      cuda::std::tuple<thrust::permutation_iterator<long*, long*>>,
      long*>,
    long,
    cuda::always_true,
    cuda::std::identity,
    long*,
    thrust::permutation_iterator<long*, long*>>;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_device_flagged_if_launchers.hpp"

template <>
CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceSelect::FlaggedIf<
  cuda::counting_iterator<int, long>,
  cub::NullType*,
  int*,
  long*,
  cuvs::distance::detail::sparse::hash_strategy<int, double, 1024>::fits_in_hash_table,
  cudaStream_t>(
  void* storage,
  size_t& bytes,
  cuda::counting_iterator<int, long> input,
  cub::NullType* flags,
  int* output,
  long* selected,
  cuda::std::int64_t items,
  cuvs::distance::detail::sparse::hash_strategy<int, double, 1024>::fits_in_hash_table predicate,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::FlaggedIf");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<cuda::counting_iterator<int, long>,
                                                    cub::NullType*,
                                                    int*,
                                                    cuda::std::int64_t,
                                                    cub::SelectImpl::Select>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::Select>(temporary,
                                                                        temporary_bytes,
                                                                        input,
                                                                        flags,
                                                                        output,
                                                                        selected,
                                                                        predicate,
                                                                        cub::NullType{},
                                                                        items,
                                                                        stream,
                                                                        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceSelect::FlaggedIf<
  cuda::counting_iterator<int, long>,
  cub::NullType*,
  int*,
  long*,
  cuvs::distance::detail::sparse::hash_strategy<int, float, 1024>::fits_in_hash_table,
  cudaStream_t>(
  void* storage,
  size_t& bytes,
  cuda::counting_iterator<int, long> input,
  cub::NullType* flags,
  int* output,
  long* selected,
  cuda::std::int64_t items,
  cuvs::distance::detail::sparse::hash_strategy<int, float, 1024>::fits_in_hash_table predicate,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::FlaggedIf");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<cuda::counting_iterator<int, long>,
                                                    cub::NullType*,
                                                    int*,
                                                    cuda::std::int64_t,
                                                    cub::SelectImpl::Select>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::Select>(temporary,
                                                                        temporary_bytes,
                                                                        input,
                                                                        flags,
                                                                        output,
                                                                        selected,
                                                                        predicate,
                                                                        cub::NullType{},
                                                                        items,
                                                                        stream,
                                                                        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::FlaggedIf<thrust::counting_iterator<long>,
                             cub::NullType*,
                             long*,
                             long*,
                             cuvs::detail::bitset_row_filter<__half, long, unsigned int, float>,
                             cudaStream_t>(
  void* storage,
  size_t& bytes,
  thrust::counting_iterator<long> input,
  cub::NullType* flags,
  long* output,
  long* selected,
  cuda::std::int64_t items,
  cuvs::detail::bitset_row_filter<__half, long, unsigned int, float> predicate,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::FlaggedIf");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<thrust::counting_iterator<long>,
                                                    cub::NullType*,
                                                    long*,
                                                    cuda::std::int64_t,
                                                    cub::SelectImpl::Select>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::Select>(temporary,
                                                                        temporary_bytes,
                                                                        input,
                                                                        flags,
                                                                        output,
                                                                        selected,
                                                                        predicate,
                                                                        cub::NullType{},
                                                                        items,
                                                                        stream,
                                                                        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::FlaggedIf<thrust::counting_iterator<long>,
                             cub::NullType*,
                             long*,
                             long*,
                             cuvs::detail::bitset_row_filter<float, long, unsigned int, float>,
                             cudaStream_t>(
  void* storage,
  size_t& bytes,
  thrust::counting_iterator<long> input,
  cub::NullType* flags,
  long* output,
  long* selected,
  cuda::std::int64_t items,
  cuvs::detail::bitset_row_filter<float, long, unsigned int, float> predicate,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::FlaggedIf");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<thrust::counting_iterator<long>,
                                                    cub::NullType*,
                                                    long*,
                                                    cuda::std::int64_t,
                                                    cub::SelectImpl::Select>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::Select>(temporary,
                                                                        temporary_bytes,
                                                                        input,
                                                                        flags,
                                                                        output,
                                                                        selected,
                                                                        predicate,
                                                                        cub::NullType{},
                                                                        items,
                                                                        stream,
                                                                        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::FlaggedIf<thrust::zip_iterator<cuda::std::tuple<int*, int*, float*>>,
                             cub::NullType*,
                             thrust::zip_iterator<cuda::std::tuple<int*, int*, float*>>,
                             long*,
                             raft::sparse::solver::new_edges_functor<int, float>,
                             cudaStream_t>(
  void* storage,
  size_t& bytes,
  thrust::zip_iterator<cuda::std::tuple<int*, int*, float*>> input,
  cub::NullType* flags,
  thrust::zip_iterator<cuda::std::tuple<int*, int*, float*>> output,
  long* selected,
  cuda::std::int64_t items,
  raft::sparse::solver::new_edges_functor<int, float> predicate,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::FlaggedIf");
  return cub::detail::dispatch_with_env_and_tuning<cub::detail::select::policy_selector_from_types<
    thrust::zip_iterator<cuda::std::tuple<int*, int*, float*>>,
    cub::NullType*,
    thrust::zip_iterator<cuda::std::tuple<int*, int*, float*>>,
    cuda::std::int64_t,
    cub::SelectImpl::Select>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::Select>(temporary,
                                                                        temporary_bytes,
                                                                        input,
                                                                        flags,
                                                                        output,
                                                                        selected,
                                                                        predicate,
                                                                        cub::NullType{},
                                                                        items,
                                                                        stream,
                                                                        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::FlaggedIf<thrust::zip_iterator<cuda::std::tuple<long*, long*, float*>>,
                             cub::NullType*,
                             thrust::zip_iterator<cuda::std::tuple<long*, long*, float*>>,
                             long*,
                             raft::sparse::solver::new_edges_functor<long, float>,
                             cudaStream_t>(
  void* storage,
  size_t& bytes,
  thrust::zip_iterator<cuda::std::tuple<long*, long*, float*>> input,
  cub::NullType* flags,
  thrust::zip_iterator<cuda::std::tuple<long*, long*, float*>> output,
  long* selected,
  cuda::std::int64_t items,
  raft::sparse::solver::new_edges_functor<long, float> predicate,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::FlaggedIf");
  return cub::detail::dispatch_with_env_and_tuning<cub::detail::select::policy_selector_from_types<
    thrust::zip_iterator<cuda::std::tuple<long*, long*, float*>>,
    cub::NullType*,
    thrust::zip_iterator<cuda::std::tuple<long*, long*, float*>>,
    cuda::std::int64_t,
    cub::SelectImpl::Select>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::Select>(temporary,
                                                                        temporary_bytes,
                                                                        input,
                                                                        flags,
                                                                        output,
                                                                        selected,
                                                                        predicate,
                                                                        cub::NullType{},
                                                                        items,
                                                                        stream,
                                                                        selector);
    });
}

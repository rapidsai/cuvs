/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_device_select_if_launchers.hpp"

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::If<cub::ArgIndexInputIterator<double*, long, double>,
                      raft::KeyValuePair<long, double>*,
                      int*,
                      cuvs::cluster::kmeans::detail::SamplingOp<double, int>,
                      cudaStream_t,
                      0>(void* storage,
                         size_t& bytes,
                         cub::ArgIndexInputIterator<double*, long, double> input,
                         raft::KeyValuePair<long, double>* output,
                         int* selected,
                         cuda::std::int64_t items,
                         cuvs::cluster::kmeans::detail::SamplingOp<double, int> predicate,
                         const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::If");
  return cub::detail::dispatch_with_env_and_tuning<cub::detail::select::policy_selector_from_types<
    cub::ArgIndexInputIterator<double*, long, double>,
    cub::NullType*,
    raft::KeyValuePair<long, double>*,
    cuda::std::int64_t,
    cub::SelectImpl::Select>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::Select>(
        temporary,
        temporary_bytes,
        input,
        static_cast<cub::NullType*>(nullptr),
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
cub::DeviceSelect::If<cub::ArgIndexInputIterator<double*, long, double>,
                      raft::KeyValuePair<long, double>*,
                      long*,
                      cuvs::cluster::kmeans::detail::SamplingOp<double, long>,
                      cudaStream_t,
                      0>(void* storage,
                         size_t& bytes,
                         cub::ArgIndexInputIterator<double*, long, double> input,
                         raft::KeyValuePair<long, double>* output,
                         long* selected,
                         cuda::std::int64_t items,
                         cuvs::cluster::kmeans::detail::SamplingOp<double, long> predicate,
                         const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::If");
  return cub::detail::dispatch_with_env_and_tuning<cub::detail::select::policy_selector_from_types<
    cub::ArgIndexInputIterator<double*, long, double>,
    cub::NullType*,
    raft::KeyValuePair<long, double>*,
    cuda::std::int64_t,
    cub::SelectImpl::Select>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::Select>(
        temporary,
        temporary_bytes,
        input,
        static_cast<cub::NullType*>(nullptr),
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
cub::DeviceSelect::If<cub::ArgIndexInputIterator<float*, long, float>,
                      raft::KeyValuePair<long, float>*,
                      int*,
                      cuvs::cluster::kmeans::detail::SamplingOp<float, int>,
                      cudaStream_t,
                      0>(void* storage,
                         size_t& bytes,
                         cub::ArgIndexInputIterator<float*, long, float> input,
                         raft::KeyValuePair<long, float>* output,
                         int* selected,
                         cuda::std::int64_t items,
                         cuvs::cluster::kmeans::detail::SamplingOp<float, int> predicate,
                         const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::If");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<cub::ArgIndexInputIterator<float*, long, float>,
                                                    cub::NullType*,
                                                    raft::KeyValuePair<long, float>*,
                                                    cuda::std::int64_t,
                                                    cub::SelectImpl::Select>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::Select>(
        temporary,
        temporary_bytes,
        input,
        static_cast<cub::NullType*>(nullptr),
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
cub::DeviceSelect::If<cub::ArgIndexInputIterator<float*, long, float>,
                      raft::KeyValuePair<long, float>*,
                      long*,
                      cuvs::cluster::kmeans::detail::SamplingOp<float, long>,
                      cudaStream_t,
                      0>(void* storage,
                         size_t& bytes,
                         cub::ArgIndexInputIterator<float*, long, float> input,
                         raft::KeyValuePair<long, float>* output,
                         long* selected,
                         cuda::std::int64_t items,
                         cuvs::cluster::kmeans::detail::SamplingOp<float, long> predicate,
                         const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::If");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<cub::ArgIndexInputIterator<float*, long, float>,
                                                    cub::NullType*,
                                                    raft::KeyValuePair<long, float>*,
                                                    cuda::std::int64_t,
                                                    cub::SelectImpl::Select>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::Select>(
        temporary,
        temporary_bytes,
        input,
        static_cast<cub::NullType*>(nullptr),
        output,
        selected,
        predicate,
        cub::NullType{},
        items,
        stream,
        selector);
    });
}

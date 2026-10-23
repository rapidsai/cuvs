/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_device_unique_launchers.hpp"

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<float*, long*, cuda::std::equal_to<float>, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  float* input,
  long* selected,
  cuda::std::int64_t items,
  cuda::std::equal_to<float> equality,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<float*,
                                                    cub::NullType*,
                                                    float*,
                                                    cuda::std::int64_t,
                                                    cub::SelectImpl::SelectPotentiallyInPlace>>(
    storage,
    bytes,
    environment,
    [&](auto selector, void* temporary, size_t& temporary_bytes, auto stream) {
      return cuvs::detail::cub_select_dispatch<cub::SelectImpl::SelectPotentiallyInPlace>(
        temporary,
        temporary_bytes,
        input,
        static_cast<cub::NullType*>(nullptr),
        input,
        selected,
        cub::NullType{},
        equality,
        items,
        stream,
        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<int*, int*, int*, cudaStream_t, 0>(void* storage,
                                                             size_t& bytes,
                                                             int* input,
                                                             int* output,
                                                             int* selected,
                                                             cuda::std::int64_t items,
                                                             const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<int*,
                                                    cub::NullType*,
                                                    int*,
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
        cub::NullType{},
        cuda::std::equal_to<>{},
        items,
        stream,
        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<int*, int*, int*, cuda::std::equal_to<void>, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  int* input,
  int* output,
  int* selected,
  cuda::std::int64_t items,
  cuda::std::equal_to<void> equality,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<int*,
                                                    cub::NullType*,
                                                    int*,
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
        cub::NullType{},
        equality,
        items,
        stream,
        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<long*, long*, int*, cudaStream_t, 0>(void* storage,
                                                               size_t& bytes,
                                                               long* input,
                                                               long* output,
                                                               int* selected,
                                                               cuda::std::int64_t items,
                                                               const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<long*,
                                                    cub::NullType*,
                                                    long*,
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
        cub::NullType{},
        cuda::std::equal_to<>{},
        items,
        stream,
        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<long*, long*, int*, cuda::std::equal_to<void>, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  long* input,
  long* output,
  int* selected,
  cuda::std::int64_t items,
  cuda::std::equal_to<void> equality,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<long*,
                                                    cub::NullType*,
                                                    long*,
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
        cub::NullType{},
        equality,
        items,
        stream,
        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<unsigned int const*, unsigned int*, long*, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  unsigned int const* input,
  unsigned int* output,
  long* selected,
  cuda::std::int64_t items,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<unsigned int const*,
                                                    cub::NullType*,
                                                    unsigned int*,
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
        cub::NullType{},
        cuda::std::equal_to<>{},
        items,
        stream,
        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceSelect::
  Unique<unsigned int const*, unsigned int*, long*, cuda::std::equal_to<void>, cudaStream_t, 0>(
    void* storage,
    size_t& bytes,
    unsigned int const* input,
    unsigned int* output,
    long* selected,
    cuda::std::int64_t items,
    cuda::std::equal_to<void> equality,
    const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<unsigned int const*,
                                                    cub::NullType*,
                                                    unsigned int*,
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
        cub::NullType{},
        equality,
        items,
        stream,
        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<unsigned int*, unsigned int*, int*, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  unsigned int* input,
  unsigned int* output,
  int* selected,
  cuda::std::int64_t items,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<unsigned int*,
                                                    cub::NullType*,
                                                    unsigned int*,
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
        cub::NullType{},
        cuda::std::equal_to<>{},
        items,
        stream,
        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceSelect::
  Unique<unsigned int*, unsigned int*, int*, cuda::std::equal_to<void>, cudaStream_t, 0>(
    void* storage,
    size_t& bytes,
    unsigned int* input,
    unsigned int* output,
    int* selected,
    cuda::std::int64_t items,
    cuda::std::equal_to<void> equality,
    const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<unsigned int*,
                                                    cub::NullType*,
                                                    unsigned int*,
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
        cub::NullType{},
        equality,
        items,
        stream,
        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::Unique<unsigned int*, unsigned int*, long*, cudaStream_t, 0>(
  void* storage,
  size_t& bytes,
  unsigned int* input,
  unsigned int* output,
  long* selected,
  cuda::std::int64_t items,
  const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<unsigned int*,
                                                    cub::NullType*,
                                                    unsigned int*,
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
        cub::NullType{},
        cuda::std::equal_to<>{},
        items,
        stream,
        selector);
    });
}

template <>
CUB_RUNTIME_FUNCTION cudaError_t cub::DeviceSelect::
  Unique<unsigned int*, unsigned int*, long*, cuda::std::equal_to<void>, cudaStream_t, 0>(
    void* storage,
    size_t& bytes,
    unsigned int* input,
    unsigned int* output,
    long* selected,
    cuda::std::int64_t items,
    cuda::std::equal_to<void> equality,
    const cudaStream_t& environment)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(storage, "cub::DeviceSelect::Unique");
  return cub::detail::dispatch_with_env_and_tuning<
    cub::detail::select::policy_selector_from_types<unsigned int*,
                                                    cub::NullType*,
                                                    unsigned int*,
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
        cub::NullType{},
        equality,
        items,
        stream,
        selector);
    });
}

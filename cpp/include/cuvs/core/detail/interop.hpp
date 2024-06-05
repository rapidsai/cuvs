/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuda_fp16.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>

#include <dlpack/dlpack.h>
#include <rmm/device_buffer.hpp>
#include <sys/types.h>

namespace cuvs::core::detail {

template <typename AccessorType>
DLDevice accessor_type_to_DLDevice()
{
  if constexpr (AccessorType::is_host_accessible and AccessorType::is_device_accessible) {
    return DLDevice{kDLCUDAManaged};
  } else if constexpr (AccessorType::is_device_accessible) {
    return DLDevice{kDLCUDA};
  } else if constexpr (AccessorType::is_host_accessible) {
    return DLDevice{kDLCPU};
  }
}

template <typename T>
DLDataType data_type_to_DLDataType()
{
  uint8_t const bits{sizeof(T) * 8};
  uint16_t const lanes{1};
  // std::is_floating_point returns false for the half type - handle
  // that here
  if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, half>) {
    return DLDataType{kDLFloat, bits, lanes};
  } else if constexpr (std::is_signed_v<T>) {
    return DLDataType{kDLInt, bits, lanes};
  } else {
    return DLDataType{kDLUInt, bits, lanes};
  }
}

inline bool is_dlpack_device_compatible(DLTensor tensor)
{
  return tensor.device.device_type == kDLCUDAManaged || tensor.device.device_type == kDLCUDAHost ||
         tensor.device.device_type == kDLCUDA;
}

inline bool is_dlpack_host_compatible(DLTensor tensor)
{
  return tensor.device.device_type == kDLCUDAManaged || tensor.device.device_type == kDLCUDAHost ||
         tensor.device.device_type == kDLCPU;
}

template <typename MdspanType, typename = raft::is_mdspan_t<MdspanType>>
inline MdspanType from_dlpack(DLManagedTensor* managed_tensor)
{
  auto tensor = managed_tensor->dl_tensor;

  auto to_data_type = data_type_to_DLDataType<typename MdspanType::value_type>();
  RAFT_EXPECTS(to_data_type.code == tensor.dtype.code,
               "code mismatch between return mdspan (%i) and DLTensor (%i)",
               to_data_type.code,
               tensor.dtype.code);
  RAFT_EXPECTS(to_data_type.bits == tensor.dtype.bits,
               "bits mismatch between return mdspan (%i) and DLTensor (%i)",
               to_data_type.bits,
               tensor.dtype.bits);
  RAFT_EXPECTS(to_data_type.lanes == tensor.dtype.lanes,
               "lanes mismatch between return mdspan and DLTensor");
  RAFT_EXPECTS(tensor.dtype.lanes == 1, "More than 1 DLTensor lanes not supported");
  RAFT_EXPECTS(tensor.strides == nullptr, "Strided memory layout for DLTensor not supported");

  auto to_device = accessor_type_to_DLDevice<typename MdspanType::accessor_type>();
  if (to_device.device_type == kDLCUDA) {
    RAFT_EXPECTS(is_dlpack_device_compatible(tensor),
                 "device_type mismatch between return mdspan and DLTensor");
  } else if (to_device.device_type == kDLCPU) {
    RAFT_EXPECTS(is_dlpack_host_compatible(tensor),
                 "device_type mismatch between return mdspan and DLTensor");
  }

  RAFT_EXPECTS(MdspanType::extents_type::rank() == tensor.ndim,
               "ndim mismatch between return mdspan and DLTensor");

  // auto exts = typename MdspanType::extents_type{tensor.shape};
  std::array<int64_t, MdspanType::extents_type::rank()> shape{};
  for (int64_t i = 0; i < tensor.ndim; ++i) {
    shape[i] = tensor.shape[i];
  }
  auto exts = typename MdspanType::extents_type{shape};

  return MdspanType{reinterpret_cast<typename MdspanType::data_handle_type>(tensor.data), exts};
}

}  // namespace cuvs::core::detail

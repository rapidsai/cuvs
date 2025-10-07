/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/version_config.h>

#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <cstdint>
#include <memory>
#include <thread>

extern "C" cuvsError_t cuvsResourcesCreate(cuvsResources_t* res)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = new raft::resources{};
    *res         = reinterpret_cast<uintptr_t>(res_ptr);
  });
}

extern "C" cuvsError_t cuvsResourcesDestroy(cuvsResources_t res)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    delete res_ptr;
  });
}

extern "C" cuvsError_t cuvsMultiGpuResourcesCreate(cuvsResources_t* res)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = new raft::device_resources_snmg{};
    *res         = reinterpret_cast<uintptr_t>(res_ptr);
  });
}

extern "C" cuvsError_t cuvsMultiGpuResourcesCreateWithDeviceIds(cuvsResources_t* res,
                                                                DLManagedTensor* device_ids)
{
  return cuvs::core::translate_exceptions([=] {
    // Basic validation
    if (device_ids == nullptr || device_ids->dl_tensor.data == nullptr) {
      throw std::invalid_argument("device_ids cannot be null");
    }

    // Check data type is int32
    if (device_ids->dl_tensor.dtype.code != kDLInt || device_ids->dl_tensor.dtype.bits != 32) {
      throw std::invalid_argument("device_ids must be int32");
    }

    // Check data is on host
    if (device_ids->dl_tensor.device.device_type != kDLCPU) {
      throw std::invalid_argument("device_ids must be on host memory");
    }

    // Cast void* to int* to perform pointer arithmetic
    int* data_ptr = static_cast<int*>(device_ids->dl_tensor.data);
    std::vector<int> ids(data_ptr, data_ptr + device_ids->dl_tensor.shape[0]);

    auto res_ptr = new raft::device_resources_snmg{ids};
    *res         = reinterpret_cast<uintptr_t>(res_ptr);
  });
}

extern "C" cuvsError_t cuvsMultiGpuResourcesDestroy(cuvsResources_t res)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::device_resources_snmg*>(res);
    delete res_ptr;
  });
}

extern "C" cuvsError_t cuvsStreamSet(cuvsResources_t res, cudaStream_t stream)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    raft::resource::set_cuda_stream(*res_ptr, static_cast<rmm::cuda_stream_view>(stream));
  });
}

extern "C" cuvsError_t cuvsStreamGet(cuvsResources_t res, cudaStream_t* stream)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    *stream      = raft::resource::get_cuda_stream(*res_ptr);
  });
}

extern "C" cuvsError_t cuvsStreamSync(cuvsResources_t res)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    raft::resource::sync_stream(*res_ptr);
  });
}

extern "C" cuvsError_t cuvsDeviceIdGet(cuvsResources_t res, int* device_id)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    *device_id   = raft::resource::get_device_id(*res_ptr);
  });
}

extern "C" cuvsError_t cuvsRMMAlloc(cuvsResources_t res, void** ptr, size_t bytes)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    auto mr      = rmm::mr::get_current_device_resource();
    *ptr         = mr->allocate(bytes, raft::resource::get_cuda_stream(*res_ptr));
  });
}

extern "C" cuvsError_t cuvsRMMFree(cuvsResources_t res, void* ptr, size_t bytes)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    auto mr      = rmm::mr::get_current_device_resource();
    mr->deallocate(ptr, bytes, raft::resource::get_cuda_stream(*res_ptr));
  });
}

thread_local std::shared_ptr<
  rmm::mr::owning_wrapper<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>,
                          rmm::mr::device_memory_resource>>
  pool_mr;

extern "C" cuvsError_t cuvsRMMPoolMemoryResourceEnable(int initial_pool_size_percent,
                                                       int max_pool_size_percent,
                                                       bool managed)
{
  return cuvs::core::translate_exceptions([=] {
    // Upstream memory resource needs to be a cuda_memory_resource
    auto cuda_mr         = rmm::mr::get_current_device_resource();
    auto* cuda_mr_casted = dynamic_cast<rmm::mr::cuda_memory_resource*>(cuda_mr);
    if (cuda_mr_casted == nullptr) {
      throw std::runtime_error("Current memory resource is not a cuda_memory_resource");
    }

    auto initial_size = rmm::percent_of_free_device_memory(initial_pool_size_percent);
    auto max_size     = rmm::percent_of_free_device_memory(max_pool_size_percent);

    auto mr = std::shared_ptr<rmm::mr::device_memory_resource>();
    if (managed) {
      mr = std::static_pointer_cast<rmm::mr::device_memory_resource>(
        std::make_shared<rmm::mr::managed_memory_resource>());
    } else {
      mr = std::static_pointer_cast<rmm::mr::device_memory_resource>(
        std::make_shared<rmm::mr::cuda_memory_resource>());
    }

    pool_mr =
      rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(mr, initial_size, max_size);

    rmm::mr::set_current_device_resource(pool_mr.get());
  });
}

extern "C" cuvsError_t cuvsRMMMemoryResourceReset()
{
  return cuvs::core::translate_exceptions([=] {
    rmm::mr::set_current_device_resource(nullptr);
    pool_mr.reset();
  });
}

thread_local std::unique_ptr<rmm::mr::pinned_memory_resource> pinned_mr;

extern "C" cuvsError_t cuvsRMMHostAlloc(void** ptr, size_t bytes)
{
  return cuvs::core::translate_exceptions([=] {
    if (pinned_mr == nullptr) { pinned_mr = std::make_unique<rmm::mr::pinned_memory_resource>(); }
    *ptr = pinned_mr->allocate(bytes);
  });
}

extern "C" cuvsError_t cuvsRMMHostFree(void* ptr, size_t bytes)
{
  return cuvs::core::translate_exceptions([=] { pinned_mr->deallocate(ptr, bytes); });
}

thread_local std::string last_error_text = "";

extern "C" const char* cuvsGetLastErrorText()
{
  return last_error_text.empty() ? NULL : last_error_text.c_str();
}

extern "C" void cuvsSetLastErrorText(const char* error) { last_error_text = error ? error : ""; }

extern "C" cuvsError_t cuvsVersionGet(uint16_t* major, uint16_t* minor, uint16_t* patch)
{
  *major = CUVS_VERSION_MAJOR;
  *minor = CUVS_VERSION_MINOR;
  *patch = CUVS_VERSION_PATCH;
  return CUVS_SUCCESS;
}

namespace {
template <typename T>
void _copy_matrix(cuvsResources_t res, DLManagedTensor* src_managed, DLManagedTensor* dst_managed)
{
  DLTensor& src = src_managed->dl_tensor;
  DLTensor& dst = dst_managed->dl_tensor;

  int64_t src_row_stride = src.strides == nullptr ? src.shape[1] : src.strides[0];
  int64_t dst_row_stride = dst.strides == nullptr ? dst.shape[1] : dst.strides[0];
  auto res_ptr           = reinterpret_cast<raft::resources*>(res);

  raft::copy_matrix<T>(static_cast<T*>(dst.data),
                       dst_row_stride,
                       static_cast<const T*>(src.data),
                       src_row_stride,
                       src.shape[1],
                       src.shape[0],
                       raft::resource::get_cuda_stream(*res_ptr));
}
}  // namespace

extern "C" cuvsError_t cuvsMatrixCopy(cuvsResources_t res,
                                      DLManagedTensor* src_managed,
                                      DLManagedTensor* dst_managed)
{
  return cuvs::core::translate_exceptions([=] {
    DLTensor& src = src_managed->dl_tensor;
    DLTensor& dst = dst_managed->dl_tensor;

    RAFT_EXPECTS(src.ndim == 2, "src should be a 2 dimensional tensor");
    RAFT_EXPECTS(dst.ndim == 2, "dst should be a 2 dimensional tensor");

    for (int64_t i = 0; i < src.ndim; ++i) {
      RAFT_EXPECTS(src.shape[i] == dst.shape[i], "shape mismatch between src and dst tensors");
    }
    RAFT_EXPECTS(src.dtype.code == dst.dtype.code, "dtype mismatch between src and dst tensors");

    // at some point we could probably copy from a float32 to a float16 here, but for the
    // moment this isn't supported
    RAFT_EXPECTS(src.dtype.bits == dst.dtype.bits,
                 "dtype bits width mismatch between src and dst tensors");

    if (src.dtype.code == kDLFloat && src.dtype.bits == 32) {
      _copy_matrix<float>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLFloat && src.dtype.bits == 16) {
      _copy_matrix<half>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLFloat && src.dtype.bits == 64) {
      _copy_matrix<double>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLInt && src.dtype.bits == 8) {
      _copy_matrix<int8_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLInt && src.dtype.bits == 16) {
      _copy_matrix<int16_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLInt && src.dtype.bits == 32) {
      _copy_matrix<int32_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLInt && src.dtype.bits == 64) {
      _copy_matrix<int64_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLUInt && src.dtype.bits == 8) {
      _copy_matrix<uint8_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLUInt && src.dtype.bits == 16) {
      _copy_matrix<uint16_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLUInt && src.dtype.bits == 32) {
      _copy_matrix<uint32_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLUInt && src.dtype.bits == 64) {
      _copy_matrix<uint64_t>(res, src_managed, dst_managed);
    } else {
      RAFT_FAIL("Unsupported dtype: %d and bits: %d", src.dtype.code, src.dtype.bits);
    }
  });
}

extern "C" void cuvsMatrixDestroy(DLManagedTensor* tensor)
{
  if (tensor->dl_tensor.shape != nullptr) {
    delete[] tensor->dl_tensor.shape;
    tensor->dl_tensor.shape = nullptr;
  }
  if (tensor->dl_tensor.strides != nullptr) {
    delete[] tensor->dl_tensor.strides;
    tensor->dl_tensor.strides = nullptr;
  }
}

extern "C" cuvsError_t cuvsMatrixSliceRows(cuvsResources_t res,
                                           DLManagedTensor* src_managed,
                                           int64_t start,
                                           int64_t end,
                                           DLManagedTensor* dst_managed)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(end >= start, "end index must be greater than start index");

    DLTensor& src = src_managed->dl_tensor;
    DLTensor& dst = dst_managed->dl_tensor;
    RAFT_EXPECTS(src.ndim == 2, "src should be a 2 dimensional tensor");
    RAFT_EXPECTS(src.shape != nullptr, "shape should be initialized in the src tensor");

    dst.dtype    = src.dtype;
    dst.device   = src.device;
    dst.ndim     = 2;
    dst.shape    = new int64_t[2];
    dst.shape[0] = end - start;
    dst.shape[1] = src.shape[1];

    int64_t row_strides = dst.shape[1];
    if (src.strides) {
      dst.strides = new int64_t[2];
      row_strides = dst.strides[0] = src.strides[0];
      dst.strides[1]               = src.strides[1];
    }

    dst.data = static_cast<char*>(src.data) + start * row_strides * (dst.dtype.bits / 8);
    dst_managed->deleter = cuvsMatrixDestroy;
  });
}

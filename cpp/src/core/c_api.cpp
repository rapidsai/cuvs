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

#include <cstdint>
#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <memory>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
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

thread_local std::string last_error_text = "";

extern "C" const char* cuvsGetLastErrorText()
{
  return last_error_text.empty() ? NULL : last_error_text.c_str();
}

extern "C" void cuvsSetLastErrorText(const char* error) { last_error_text = error ? error : ""; }

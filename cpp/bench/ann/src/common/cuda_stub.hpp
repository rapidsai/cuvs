/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

/*
The content of this header is governed by two preprocessor definitions:

  - BUILD_CPU_ONLY - whether none of the CUDA functions are used.
  - ANN_BENCH_LINK_CUDART - dynamically link against this string if defined.

___________________________________________________________________________________
|BUILD_CPU_ONLY | ANN_BENCH_LINK_CUDART |         cudart      | cuda_runtime_api.h |
|               |                       |  found    |  needed |      included      |
|---------------|-----------------------|-----------|---------|--------------------|
|   ON          |    <not defined>      |  false    |  false  |       NO           |
|   ON          |   "cudart.so.xx.xx"   |  false    |  false  |       NO           |
|  OFF          |     <not defined>     |   true    |   true  |      YES           |
|  OFF          |   "cudart.so.xx.xx"   | <runtime> |   true  |      YES           |
------------------------------------------------------------------------------------
*/

#pragma once

#ifndef BUILD_CPU_ONLY
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#ifdef ANN_BENCH_LINK_CUDART
#include <dlfcn.h>

#include <cstring>
#endif
#else
#include <cstdint>

typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef uint16_t half;
#endif

namespace cuvs::bench {

struct cuda_lib_handle {
  void* handle{nullptr};
  explicit cuda_lib_handle()
  {
#ifdef ANN_BENCH_LINK_CUDART
    constexpr int kFlags = RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND | RTLD_NODELETE;
    // The full name of the linked cudart library 'cudart.so.MAJOR.MINOR.PATCH'
    char libname[] = ANN_BENCH_LINK_CUDART;  // NOLINT
    handle         = dlopen(ANN_BENCH_LINK_CUDART, kFlags);
    if (handle != nullptr) { return; }
    // try strip the PATCH
    auto p = strrchr(libname, '.');
    p[0]   = 0;
    handle = dlopen(libname, kFlags);
    if (handle != nullptr) { return; }
    // try set the MINOR version to 0
    p      = strrchr(libname, '.');
    p[1]   = '0';
    p[2]   = 0;
    handle = dlopen(libname, kFlags);
    if (handle != nullptr) { return; }
    // try strip the MINOR
    p[0]   = 0;
    handle = dlopen(libname, kFlags);
    if (handle != nullptr) { return; }
    // try strip the MAJOR
    p      = strrchr(libname, '.');
    p[0]   = 0;
    handle = dlopen(libname, kFlags);
#endif
  }
  ~cuda_lib_handle() noexcept
  {
#ifdef ANN_BENCH_LINK_CUDART
    if (handle != nullptr) { dlclose(handle); }
#endif
  }

  template <typename Symbol>
  auto sym(const char* name) -> Symbol
  {
#ifdef ANN_BENCH_LINK_CUDART
    return reinterpret_cast<Symbol>(dlsym(handle, name));
#else
    return nullptr;
#endif
  }

  /** Whether this is NOT a cpu-only package. */
  [[nodiscard]] constexpr inline auto needed() const -> bool
  {
#if defined(BUILD_CPU_ONLY)
    return false;
#else
    return true;
#endif
  }

  /** CUDA found, either at compile time or at runtime. */
  [[nodiscard]] inline auto found() const -> bool
  {
#if defined(BUILD_CPU_ONLY)
    return false;
#elif defined(ANN_BENCH_LINK_CUDART)
    return handle != nullptr;
#else
    return true;
#endif
  }
};

static inline cuda_lib_handle cudart{};

#ifdef ANN_BENCH_LINK_CUDART
namespace stub {

[[gnu::weak, gnu::noinline]] auto cuda_memcpy(void* dst,
                                              const void* src,
                                              size_t count,
                                              enum cudaMemcpyKind kind) -> cudaError_t
{
  return cudaSuccess;
}

[[gnu::weak, gnu::noinline]] auto cuda_malloc(void** ptr, size_t size) -> cudaError_t
{
  *ptr = nullptr;
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_memset(void* devPtr, int value, size_t count) -> cudaError_t
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_free(void* devPtr) -> cudaError_t { return cudaSuccess; }
[[gnu::weak, gnu::noinline]] auto cuda_stream_create(cudaStream_t* pStream) -> cudaError_t
{
  *pStream = nullptr;
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_stream_create_with_flags(cudaStream_t* pStream,
                                                                unsigned int flags) -> cudaError_t
{
  *pStream = nullptr;
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_stream_destroy(cudaStream_t pStream) -> cudaError_t
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_device_synchronize() -> cudaError_t { return cudaSuccess; }

[[gnu::weak, gnu::noinline]] auto cuda_stream_synchronize(cudaStream_t pStream) -> cudaError_t
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_event_create(cudaEvent_t* event) -> cudaError_t
{
  *event = nullptr;
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_event_record(cudaEvent_t event, cudaStream_t stream)
  -> cudaError_t
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_event_synchronize(cudaEvent_t event) -> cudaError_t
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_event_elapsed_time(float* ms,
                                                          cudaEvent_t start,
                                                          cudaEvent_t end) -> cudaError_t
{
  *ms = 0;
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_event_destroy(cudaEvent_t event) -> cudaError_t
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] auto cuda_get_device(int* device) -> cudaError_t
{
  *device = 0;
  return cudaSuccess;
};
[[gnu::weak, gnu::noinline]] auto cuda_driver_get_version(int* driver) -> cudaError_t
{
  *driver = 0;
  return cudaSuccess;
};
[[gnu::weak, gnu::noinline]] auto cuda_runtime_get_version(int* runtime) -> cudaError_t
{
  *runtime = 0;
  return cudaSuccess;
};
[[gnu::weak, gnu::noinline]] cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop,
                                                                 int device)
{
  *prop = cudaDeviceProp{};
  return cudaSuccess;
}

}  // namespace stub

#define RAFT_DECLARE_CUDART(fun)           \
  static inline decltype(&stub::fun) fun = \
    cudart.found() ? cudart.sym<decltype(&stub::fun)>(#fun) : &stub::fun

RAFT_DECLARE_CUDART(cuda_memcpy);
RAFT_DECLARE_CUDART(cuda_malloc);
RAFT_DECLARE_CUDART(cuda_memset);
RAFT_DECLARE_CUDART(cuda_free);
RAFT_DECLARE_CUDART(cuda_stream_create);
RAFT_DECLARE_CUDART(cuda_stream_create_with_flags);
RAFT_DECLARE_CUDART(cuda_stream_destroy);
RAFT_DECLARE_CUDART(cuda_device_synchronize);
RAFT_DECLARE_CUDART(cuda_stream_synchronize);
RAFT_DECLARE_CUDART(cuda_event_create);
RAFT_DECLARE_CUDART(cuda_event_record);
RAFT_DECLARE_CUDART(cuda_event_synchronize);
RAFT_DECLARE_CUDART(cuda_event_elapsed_time);
RAFT_DECLARE_CUDART(cuda_event_destroy);
RAFT_DECLARE_CUDART(cuda_get_device);
RAFT_DECLARE_CUDART(cuda_driver_get_version);
RAFT_DECLARE_CUDART(cuda_runtime_get_version);
RAFT_DECLARE_CUDART(cudaGetDeviceProperties);

#undef RAFT_DECLARE_CUDART
#endif

};  // namespace cuvs::bench

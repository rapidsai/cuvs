/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <stdint.h>
#include <sys/mman.h>

#include <cstdlib>
#include <cstring>

// jamxia edit
// #include "utils/tools.hpp"
#include "tools.hpp"

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace memory {
template <size_t alignment, class T, bool HUGE_PAGE = false>
inline T* align_mm(size_t nbytes)
{
  size_t size = rd_up_to_multiple_of(nbytes, alignment);
  void* p     = std::aligned_alloc(alignment, size);
  if (HUGE_PAGE) { madvise(p, nbytes, MADV_HUGEPAGE); }
  std::memset(p, 0, size);
  return static_cast<T*>(p);
}

template <typename T, size_t alignment = 64>
struct align_allocator {
  T* ptr            = nullptr;
  size_t alignment_ = alignment;
  using value_type  = T;
  T* allocate(size_t n)
  {
    size_t nbytes = rd_up_to_multiple_of(n * sizeof(T), alignment_);
    return ptr    = (T*)std::aligned_alloc(alignment_, nbytes);
  }
  void deallocate(T* p, size_t)
  {
    std::free(p);
    p = nullptr;
  }
  template <typename U>
  struct rebind {
    typedef align_allocator<U> other;
  };
  bool operator!=(const align_allocator& rhs) { return alignment_ != rhs.alignment_; }
  bool operator==(const align_allocator& rhs) { return alignment_ == rhs.alignment_; }
};
}  // namespace memory

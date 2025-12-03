/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#include <sys/mman.h>

#include <cstdlib>
#include <cstring>

// jamxia edit
// #include "utils/tools.hpp"
#include <cuvs/neighbors/ivf_rabitq/utils/tools.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

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
}  // namespace cuvs::neighbors::ivf_rabitq::detail

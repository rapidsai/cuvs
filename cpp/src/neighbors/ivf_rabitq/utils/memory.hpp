/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#include <sys/mman.h>

#include <cstdlib>
#include <cstring>

#include "tools.hpp"

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

}  // namespace memory
}  // namespace cuvs::neighbors::ivf_rabitq::detail

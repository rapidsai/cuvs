/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/FragmentEntry.hpp>

namespace cuvs::neighbors::cagra::detail {

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename QueryTag,
          typename CodebookTag>
struct SetupWorkspaceFragmentEntry final
  : StaticFatbinFragmentEntry<
      SetupWorkspaceFragmentEntry<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag>> {
  static const uint8_t* const data;
  static const size_t length;
};

}  // namespace cuvs::neighbors::cagra::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "FragmentEntry.hpp"

namespace cuvs::detail::jit_lto {

template <typename IdxTag, typename FilterTag>
struct FilterFragmentEntry final
  : StaticFatbinFragmentEntry<FilterFragmentEntry<IdxTag, FilterTag>> {
  static const uint8_t* const data;
  static const size_t length;
};

}  // namespace cuvs::detail::jit_lto

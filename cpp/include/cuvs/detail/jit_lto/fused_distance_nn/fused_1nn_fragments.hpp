/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/common_fragments.hpp>

namespace cuvs::distance::detail {

template <typename DataTag, typename ArchTag>
struct fragment_tag_fused_1nn_cubin {
  static constexpr int cc_major = ArchTag::cc_major;
  static constexpr int cc_minor = ArchTag::cc_minor;
};

template <typename DataTag>
struct fragment_tag_fused_1nn_tileir {};

}  // namespace cuvs::distance::detail

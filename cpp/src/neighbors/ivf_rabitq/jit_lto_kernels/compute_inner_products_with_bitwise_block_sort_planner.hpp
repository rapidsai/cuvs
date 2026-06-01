/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentEntry.hpp>
#include <cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

struct ComputeInnerProductsWithBitwiseBlockSortPlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  ComputeInnerProductsWithBitwiseBlockSortPlanner()
    : AlgorithmPlanner("compute_inner_products_with_bitwise_block_sort", launcher_jit_cache)
  {
  }

  template <int NumBits, bool WithEx>
  void add_entrypoint()
  {
    this->add_static_fragment<
      fragment_tag_compute_inner_products_with_bitwise_block_sort<NumBits, WithEx>>();
  }

  void add_extract_code_device_function()
  {
    this->add_static_fragment<fragment_tag_extract_code>();
  }
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail

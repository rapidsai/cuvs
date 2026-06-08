/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentEntry.hpp>
#include <cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

struct ComputeInnerProductsWithBitwisePlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  ComputeInnerProductsWithBitwisePlanner()
    : AlgorithmPlanner("compute_inner_products_with_bitwise", launcher_jit_cache)
  {
  }

  template <bool WithEx>
  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_compute_inner_products_with_bitwise<WithEx>>();
  }

  template <int EX_BITS>
  void add_extract_code_device_function()
  {
    this->add_static_fragment<fragment_tag_extract_code<EX_BITS>>();
  }

  template <int EX_BITS>
  void add_compute_ip2_from_long_codes_warp_device_function()
  {
    this->add_static_fragment<fragment_tag_compute_ip2_from_long_codes_warp<EX_BITS>>();
  }
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail

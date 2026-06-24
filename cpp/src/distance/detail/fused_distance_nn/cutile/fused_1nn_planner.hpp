/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/cutile_arch_tags.hpp>
#include <cuvs/detail/jit_lto/fused_distance_nn/fused_1nn_fragments.hpp>

namespace cuvs::distance::detail {

/** Must match KERNEL_SYMBOLS in fused_1nn_kernel.py (export uses with_symbol). */
template <typename DataTag>
inline const char* fused_1nn_kernel_entrypoint()
{
  if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_h>) {
    return "fused_1nn_half";
  } else if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_f>) {
    return "fused_1nn_float";
  } else {
    static_assert(sizeof(DataTag) == 0, "unsupported fused 1-NN cuTile data type");
    return "";
  }
}

template <typename DataTag>
struct Fused1nnTilePlanner : TileAlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  Fused1nnTilePlanner()
    : TileAlgorithmPlanner(fused_1nn_kernel_entrypoint<DataTag>(), launcher_jit_cache)
  {
  }

  /** Registers embedded cubin modules (one per SM); see register_cubin.cpp object files. */
  void add_entrypoint()
  {
    using cuvs::detail::jit_lto::cutile_arch_12_0;
    using cuvs::detail::jit_lto::cutile_arch_8_0;
    using cuvs::detail::jit_lto::cutile_arch_8_6;
    using cuvs::detail::jit_lto::cutile_arch_9_0;

    this->add_static_fragment<fragment_tag_fused_1nn_cubin<DataTag, cutile_arch_8_0>>();
    this->add_static_fragment<fragment_tag_fused_1nn_cubin<DataTag, cutile_arch_8_6>>();
    this->add_static_fragment<fragment_tag_fused_1nn_cubin<DataTag, cutile_arch_9_0>>();
    this->add_static_fragment<fragment_tag_fused_1nn_cubin<DataTag, cutile_arch_12_0>>();
  }

  void add_tileir_fallback()
  {
    this->add_static_tileir_fragment<fragment_tag_fused_1nn_tileir<DataTag>>();
  }
};

}  // namespace cuvs::distance::detail

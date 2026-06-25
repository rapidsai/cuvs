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

#include "fused_1nn_cutile_tiles.hpp"

namespace cuvs::distance::detail {

/** Must match kernel_symbol() in fused_1nn_kernel.py (export uses with_symbol). */
template <typename DataTag, typename MetricTag>
inline const char* fused_1nn_kernel_entrypoint()
{
  if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_f>) {
    if constexpr (std::is_same_v<MetricTag, metric_tag_ip>) {
      return "fused_1nn_f_ip";
    } else if constexpr (std::is_same_v<MetricTag, metric_tag_l2>) {
      return "fused_1nn_f_l2";
    } else if constexpr (std::is_same_v<MetricTag, metric_tag_cos>) {
      return "fused_1nn_f_cos";
    }
  } else if constexpr (std::is_same_v<DataTag, cuvs::neighbors::detail::tag_h>) {
    if constexpr (std::is_same_v<MetricTag, metric_tag_ip>) {
      return "fused_1nn_h_ip";
    } else if constexpr (std::is_same_v<MetricTag, metric_tag_l2>) {
      return "fused_1nn_h_l2";
    } else if constexpr (std::is_same_v<MetricTag, metric_tag_cos>) {
      return "fused_1nn_h_cos";
    }
  }
  static_assert(sizeof(DataTag) == 0, "unsupported fused 1-NN cuTile data/metric combination");
  return "";
}

template <typename DataT, cuvs::distance::DistanceType Metric>
struct Fused1nnTilePlanner : TileAlgorithmPlanner {
  using DataTag   = fused_1nn_data_tag_t<DataT>;
  using MetricTag = fused_1nn_metric_tag_t<Metric>;

  inline static LauncherJitCache launcher_jit_cache{};

  Fused1nnTilePlanner()
    : TileAlgorithmPlanner(fused_1nn_kernel_entrypoint<DataTag, MetricTag>(), launcher_jit_cache)
  {
  }

  /** Registers embedded cubin modules (one per SM); see register_cutile_fragment.cpp object files.
   */
  void add_entrypoint()
  {
    using cuvs::detail::jit_lto::cutile_arch_12_0;
    using cuvs::detail::jit_lto::cutile_arch_8_0;
    using cuvs::detail::jit_lto::cutile_arch_8_6;
    using cuvs::detail::jit_lto::cutile_arch_9_0;

    this->add_static_fragment<
      fragment_tag_fused_1nn_cubin<DataTag, MetricTag, fused_1nn_matrix_tile, cutile_arch_8_0>>();
    this->add_static_fragment<
      fragment_tag_fused_1nn_cubin<DataTag, MetricTag, fused_1nn_matrix_tile, cutile_arch_8_6>>();
    this->add_static_fragment<
      fragment_tag_fused_1nn_cubin<DataTag, MetricTag, fused_1nn_matrix_tile, cutile_arch_9_0>>();
    this->add_static_fragment<
      fragment_tag_fused_1nn_cubin<DataTag, MetricTag, fused_1nn_matrix_tile, cutile_arch_12_0>>();
  }

  void add_tileir_fallback()
  {
    this->add_static_tileir_fragment<
      fragment_tag_fused_1nn_tileir<DataTag, MetricTag, fused_1nn_matrix_tile>>();
  }
};

}  // namespace cuvs::distance::detail

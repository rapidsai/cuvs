/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>

#include "registration_tags.hpp"

#include <string>
#include <utility>

namespace cuvs::distance::detail {

inline constexpr char kPairwiseMatrixJitEntrypoint[] = "pairwise_matrix_kernel";

struct PairwiseMatrixPlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  PairwiseMatrixPlanner() : AlgorithmPlanner(kPairwiseMatrixJitEntrypoint, launcher_jit_cache) {}

  explicit PairwiseMatrixPlanner(std::string entrypoint)
    : AlgorithmPlanner(std::move(entrypoint), launcher_jit_cache)
  {
  }

  template <typename DistanceTag,
            typename DataTag,
            typename AccTag,
            typename OutTag,
            typename IndexTag,
            typename FinOpTag,
            typename LayoutTag,
            int Veclen>
  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_pairwise_matrix<DistanceTag,
                                                           DataTag,
                                                           AccTag,
                                                           OutTag,
                                                           IndexTag,
                                                           FinOpTag,
                                                           LayoutTag,
                                                           Veclen>>();
  }

  template <typename DistanceTag, typename DataTag, typename AccTag, typename IndexTag>
  void add_compute_distance_function()
  {
    this->add_static_fragment<
      fragment_tag_compute_distance<DistanceTag, DataTag, AccTag, IndexTag>>();
  }

  template <typename DistanceTag,
            typename DataTag,
            typename AccTag,
            typename IndexTag,
            typename LayoutTag,
            int Veclen>
  void add_compute_distance_epilog_function()
  {
    this->add_static_fragment<fragment_tag_compute_distance_epilog<DistanceTag,
                                                                   DataTag,
                                                                   AccTag,
                                                                   IndexTag,
                                                                   LayoutTag,
                                                                   Veclen>>();
  }
};

}  // namespace cuvs::distance::detail

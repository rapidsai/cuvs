/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

struct ComputeSimilarityPlanner : AlgorithmPlanner {
  ComputeSimilarityPlanner() : AlgorithmPlanner("compute_similarity") {}

  template <typename OutTag, typename LutTag>
  void add_entrypoint()
  {
    this->add_fragment<ComputeSimilarityFragmentEntry<OutTag, LutTag>>();
  }

  template <typename LutTag, bool EnableSMemLut, uint32_t PqBits>
  void add_prepare_lut_function()
  {
    this->add_fragment<PrepareLutFragmentEntry<LutTag, EnableSMemLut, PqBits>>();
  }

  template <typename OutTag, bool kManageLocalTopK>
  void add_store_calculated_distances_function()
  {
    this->add_fragment<StoreCalculatedDistancesFragmentEntry<OutTag, kManageLocalTopK>>();
  }

  template <bool PrecompBaseDiff>
  void add_precompute_base_diff_function()
  {
    this->add_fragment<PrecomputeBaseDiffFragmentEntry<PrecompBaseDiff>>();
  }

  template <typename LutTag, bool PrecompBaseDiff, uint32_t PqBits>
  void add_create_lut_function()
  {
    this->add_fragment<CreateLutFragmentEntry<LutTag, PrecompBaseDiff, PqBits>>();
  }

  template <typename OutTag, typename LutTag, int Capacity, uint32_t PqBits>
  void add_compute_distances_function()
  {
    this->add_fragment<ComputeDistancesFragmentEntry<OutTag, LutTag, Capacity, PqBits>>();
  }
};

}  // namespace cuvs::neighbors::ivf_pq::detail

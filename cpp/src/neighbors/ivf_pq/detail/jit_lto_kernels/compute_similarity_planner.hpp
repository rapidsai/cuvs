/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentEntry.hpp>
#include <cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

struct ComputeSimilarityPlanner : AlgorithmPlanner {
  ComputeSimilarityPlanner() : AlgorithmPlanner("compute_similarity") {}

  template <typename OutTag, typename LutTag>
  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_compute_similarity<OutTag, LutTag>>();
  }

  template <typename LutTag, bool EnableSMemLut, uint32_t PqBits>
  void add_prepare_lut_function()
  {
    this->add_static_fragment<fragment_tag_prepare_lut<LutTag, EnableSMemLut, PqBits>>();
  }

  template <typename OutTag, bool kManageLocalTopK>
  void add_store_calculated_distances_function()
  {
    this->add_static_fragment<fragment_tag_store_calculated_distances<OutTag, kManageLocalTopK>>();
  }

  template <bool PrecompBaseDiff>
  void add_precompute_base_diff_function()
  {
    this->add_static_fragment<fragment_tag_precompute_base_diff<PrecompBaseDiff>>();
  }

  template <typename LutTag, bool PrecompBaseDiff, uint32_t PqBits>
  void add_create_lut_function()
  {
    this->add_static_fragment<fragment_tag_create_lut<LutTag, PrecompBaseDiff, PqBits>>();
  }

  template <typename OutTag, typename LutTag, int Capacity>
  void add_compute_distances_function()
  {
    this->add_static_fragment<fragment_tag_compute_distances<OutTag, LutTag, Capacity>>();
  }

  template <typename FilterTag>
  void add_sample_filter_function()
  {
    this->add_static_fragment<fragment_tag_sample_filter<FilterTag>>();
  }

  template <uint32_t PqBits>
  void add_get_line_width_function()
  {
    this->add_static_fragment<fragment_tag_get_line_width<PqBits>>();
  }

  template <typename OutTag, typename LutTag, uint32_t PqBits>
  void add_compute_score_function()
  {
    this->add_static_fragment<fragment_tag_compute_score<OutTag, LutTag, PqBits>>();
  }
};

}  // namespace cuvs::neighbors::ivf_pq::detail

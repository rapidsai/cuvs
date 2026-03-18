/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/detail/jit_lto/fragments.hpp>
#include <cuvs/detail/jit_lto/ivf_flat/interleaved_scan_fragments.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <iostream>
#include <string>

namespace cuvs::neighbors::ivf_flat::detail {

struct InterleavedScanPlanner : AlgorithmPlanner {
  InterleavedScanPlanner() : AlgorithmPlanner("interleaved_scan") {}

  template <typename DataTag,
            typename AccTag,
            typename IdxTag,
            int Capacity,
            int Veclen,
            bool Ascending,
            bool ComputeNorm>
  void add_entrypoint()
  {
    this->add_fragment<InterleavedScanFragmentEntry<DataTag,
                                                    AccTag,
                                                    IdxTag,
                                                    Capacity,
                                                    Veclen,
                                                    Ascending,
                                                    ComputeNorm>>();
  }

  template <int Veclen, typename DataTag, typename AccTag, typename MetricTag>
  void add_metric_device_function()
  {
    this->add_fragment<MetricFragmentEntry<Veclen, DataTag, AccTag, MetricTag>>();
  }

  template <typename IvfSampleFilterTag>
  void add_filter_device_function()
  {
    this->add_fragment<cuvs::detail::jit_lto::FilterFragmentEntry<tag_idx_l, IvfSampleFilterTag>>();
  }

  template <typename PostLambdaTag>
  void add_post_lambda_device_function()
  {
    this->add_fragment<PostLambdaFragmentEntry<PostLambdaTag>>();
  }
};

}  // namespace cuvs::neighbors::ivf_flat::detail

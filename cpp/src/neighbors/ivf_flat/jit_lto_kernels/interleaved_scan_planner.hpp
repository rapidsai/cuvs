/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentEntry.hpp>
#include <cuvs/detail/jit_lto/ivf_flat/interleaved_scan_fragments.hpp>
#include <cuvs/detail/jit_lto/ivf_flat/interleaved_scan_tags.hpp>
#include <iostream>
#include <memory>
#include <string>

namespace cuvs::neighbors::ivf_flat::detail {

struct InterleavedScanPlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  InterleavedScanPlanner() : AlgorithmPlanner("interleaved_scan", launcher_jit_cache) {}

  template <typename DataTag,
            typename AccTag,
            typename IdxTag,
            int Capacity,
            int Veclen,
            bool Ascending,
            bool ComputeNorm>
  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_interleaved_scan<DataTag,
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
    this->add_static_fragment<fragment_tag_metric<Veclen, DataTag, AccTag, MetricTag>>();
  }

  void add_metric_udf_fragment(std::unique_ptr<UDFFatbinFragment> fragment)
  {
    this->add_fragment(std::move(fragment));
  }

  template <typename IvfSampleFilterTag>
  void add_filter_device_function()
  {
    this->add_static_fragment<fragment_tag_filter<IvfSampleFilterTag>>();
  }

  template <typename PostLambdaTag>
  void add_post_lambda_device_function()
  {
    this->add_static_fragment<fragment_tag_post_lambda<PostLambdaTag>>();
  }
};

}  // namespace cuvs::neighbors::ivf_flat::detail

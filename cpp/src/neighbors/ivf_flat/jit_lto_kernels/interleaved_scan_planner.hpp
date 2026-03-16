/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/detail/jit_lto/ivf_flat/interleaved_scan_tags.hpp>
#include <iostream>
#include <string>

template <typename DataTypeTag, typename AccTypeTag, typename IdxTypeTag>
struct InterleavedScanPlanner : AlgorithmPlanner {
  InterleavedScanPlanner(int Capacity, int Veclen, bool Ascending, bool ComputeNorm)
    : AlgorithmPlanner("interleaved_scan_capacity_" + std::to_string(Capacity) + "_veclen_" +
                         std::to_string(Veclen) + "_" + (Ascending ? "ascending" : "descending") +
                         "_" + (ComputeNorm ? "compute_norm" : "no_compute_norm") + "_data_" +
                         cuvs::neighbors::ivf_flat::detail::tag_abbrev<DataTypeTag>::value +
                         "_acc_" +
                         cuvs::neighbors::ivf_flat::detail::tag_abbrev<AccTypeTag>::value +
                         "_idx_" + cuvs::neighbors::ivf_flat::detail::tag_abbrev<IdxTypeTag>::value,
                       "interleaved_scan")
  {
  }

  template <typename... FuncTags>
  void add_metric_device_function(std::string metric_name, int Veclen)
  {
    auto key    = metric_name + "_veclen_" + std::to_string(Veclen);
    auto params = make_fragment_key<FuncTags...>();
    this->device_functions.push_back(key + "_" + params);
  }

  void add_filter_device_function(std::string filter_name)
  {
    auto key = filter_name;
    this->device_functions.push_back(key);
  }

  void add_post_lambda_device_function(std::string post_lambda_name)
  {
    auto key = post_lambda_name;
    this->device_functions.push_back(key);
  }
};

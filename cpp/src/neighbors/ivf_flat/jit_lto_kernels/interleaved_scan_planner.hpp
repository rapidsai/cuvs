/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <iostream>
#include <string>

inline std::string bool_to_string(bool b) { return b ? "true" : "false"; }

template <typename... Args>
struct InterleavedScanPlanner : AlgorithmPlanner {
  InterleavedScanPlanner(int Capacity, int Veclen, bool Ascending, bool ComputeNorm)
    : AlgorithmPlanner("interleaved_scan_kernel_" + std::to_string(Capacity) + "_" +
                         std::to_string(Veclen) + "_" + bool_to_string(Ascending) + "_" +
                         bool_to_string(ComputeNorm),
                       make_fragment_key<Args...>())
  {
  }

  template <typename... FuncTags>
  void add_metric_device_function(std::string metric_name, int Veclen)
  {
    auto key    = metric_name + "_" + std::to_string(Veclen);
    auto params = make_fragment_key<FuncTags...>();
    this->device_functions.push_back(key + "_" + params);
  }

  void add_filter_device_function(std::string filter_name)
  {
    auto key = "sample_filter_" + filter_name;
    this->device_functions.push_back(key);
  }

  void add_post_lambda_device_function(std::string post_lambda_name)
  {
    auto key = post_lambda_name;
    this->device_functions.push_back(key);
  }
};

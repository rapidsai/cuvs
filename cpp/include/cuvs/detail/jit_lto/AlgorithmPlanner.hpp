/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "AlgorithmLauncher.hpp"

struct FragmentEntry;

struct AlgorithmPlanner {
  AlgorithmPlanner(std::string entrypoint) : entrypoint(std::move(entrypoint)) {}

  std::shared_ptr<AlgorithmLauncher> get_launcher();

  std::string entrypoint;
  std::vector<const FragmentEntry*> fragments;

  void add_fragment(const FragmentEntry& fragment);

  template <typename FragmentT>
  void add_fragment()
  {
    add_fragment(FragmentT{});
  }

 private:
  std::string get_fragments_key() const;
  std::shared_ptr<AlgorithmLauncher> build();
};

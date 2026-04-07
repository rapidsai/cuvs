/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "AlgorithmLauncher.hpp"
#include "FragmentEntry.hpp"

struct AlgorithmPlanner {
  AlgorithmPlanner(std::string entrypoint) : entrypoint(std::move(entrypoint)) {}

  std::shared_ptr<AlgorithmLauncher> get_launcher();

  std::string entrypoint;
  std::vector<std::unique_ptr<FragmentEntry>> fragments;

  template <typename T, typename = std::enable_if_t<std::is_convertible_v<T*, FragmentEntry*>>>
  void add_fragment(std::unique_ptr<T> fragment)
  {
    fragments.push_back(std::unique_ptr<FragmentEntry>(std::move(fragment)));
  }

  template <typename FragmentTag>
  void add_static_fragment()
  {
    add_fragment(std::make_unique<StaticFatbinFragmentEntry<FragmentTag>>());
  }

 private:
  std::string get_fragments_key() const;
  std::shared_ptr<AlgorithmLauncher> build();
};

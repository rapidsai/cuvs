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
  AlgorithmPlanner(std::string&& fragment_key, std::string&& entrypoint)
    : fragment_key(std::move(fragment_key)), entrypoint(std::move(entrypoint))
  {
  }

  std::shared_ptr<AlgorithmLauncher> get_launcher();

  std::string fragment_key;
  std::string entrypoint;
  std::vector<std::string> device_functions;
  std::vector<FragmentEntry*> fragments;

 private:
  void add_entrypoint();
  void add_device_functions();
  std::string get_device_functions_key() const;
  std::shared_ptr<AlgorithmLauncher> build();
};

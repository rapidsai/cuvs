/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>
#include <vector>

#include "AlgorithmLauncher.h"

struct FragmentEntry;

struct AlgorithmPlanner {
  AlgorithmPlanner(std::string const& n, std::string const& p) : entrypoint(n + "_" + p) {}

  AlgorithmLauncher get_launcher();

  std::string entrypoint;
  std::vector<std::string> device_functions;
  std::vector<FragmentEntry*> fragments;

 private:
  void add_entrypoint();
  void add_device_functions();
  std::string get_device_functions_key();
  AlgorithmLauncher build();
};

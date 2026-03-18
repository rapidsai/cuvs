/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "FragmentEntry.hpp"

struct NVRTCLTOFragmentCompiler {
  NVRTCLTOFragmentCompiler();

  const NVRTCFatbinFragmentEntry& compile(std::string const& code);

  std::vector<std::string> standard_compile_opts;
  std::unordered_map<std::string, NVRTCFatbinFragmentEntry> compiled_fragments;
};

NVRTCLTOFragmentCompiler& nvrtc_compiler();

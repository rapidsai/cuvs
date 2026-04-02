/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/FragmentEntry.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct NVRTCLTOFragmentCompiler {
  NVRTCLTOFragmentCompiler();

  std::vector<std::string> standard_compile_opts;
  std::unordered_map<std::string, std::vector<uint8_t>> cache;

  std::unique_ptr<UDFFatbinFragment> compile(std::string const& key, std::string const& code);
};

NVRTCLTOFragmentCompiler& nvrtc_compiler();

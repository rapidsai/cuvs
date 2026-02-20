/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

struct NVRTCLTOFragmentCompiler {
  NVRTCLTOFragmentCompiler();

  void compile(std::string const& key, std::string const& code) const;

  std::vector<std::string> standard_compile_opts;
};

NVRTCLTOFragmentCompiler& nvrtc_compiler();

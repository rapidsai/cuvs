/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct NRTCLTOFragmentCompiler {
  NRTCLTOFragmentCompiler();

  void compile(std::string const& key, std::string const& code) const;

  std::vector<std::string> standard_compile_opts;
};

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuvs/detail/jit_lto/nvjitlink_helper.hpp>

struct FragmentEntry {
  FragmentEntry(std::string const& key);

  bool operator==(const FragmentEntry& rhs) const { return compute_key == rhs.compute_key; }

  virtual bool add_to(nvJitLinkHandle& handle) const = 0;

  std::string compute_key{};
};

struct FatbinFragmentEntry final : FragmentEntry {
  FatbinFragmentEntry(std::string const& key, unsigned char const* view, std::size_t size);

  virtual bool add_to(nvJitLinkHandle& handle) const;

  std::size_t data_size          = 0;
  unsigned char const* data_view = nullptr;
};

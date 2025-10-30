/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <nvJitLink.h>

struct FragmentEntry {
  FragmentEntry(std::string const& params);

  bool operator==(const FragmentEntry& rhs) const { return compute_key == rhs.compute_key; }

  virtual bool add_to(nvJitLinkHandle& handle) const = 0;

  std::string compute_key{};
};

struct FatbinFragmentEntry final : FragmentEntry {
  FatbinFragmentEntry(std::string const& params, unsigned char const* view, std::size_t size);

  virtual bool add_to(nvJitLinkHandle& handle) const;

  std::size_t data_size          = 0;
  unsigned char const* data_view = nullptr;
};
